"""[Stage 1] Fetch RSS/Atom feeds.

Conditional GET (ETag/Last-Modified) keeps us polite on a 20-minute cadence —
most polls return 304 and cost the publisher nothing. The cache lives only for
the process lifetime on CI, so this mainly benefits local iteration; the
identifying User-Agent is what matters for publisher relations.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import feedparser
import httpx

log = logging.getLogger(__name__)

USER_AGENT = (
    "WhatItMeansBot/0.1 (+https://github.com/Ayushsub/WhatItMeans; financial news explainer; "
    "respects robots.txt; contact via repository)"
)

_MAX_PARALLEL = 6


@dataclass
class FeedResult:
    source: dict[str, Any]
    entries: list[dict] = field(default_factory=list)
    status: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.entries)


# Module-level conditional-GET cache: source_id -> (etag, last_modified)
_cache: dict[str, tuple[str | None, str | None]] = {}


def fetch_one(source: dict[str, Any], client: httpx.Client) -> FeedResult:
    headers = {"User-Agent": USER_AGENT}
    etag, last_mod = _cache.get(source["id"], (None, None))
    if etag:
        headers["If-None-Match"] = etag
    if last_mod:
        headers["If-Modified-Since"] = last_mod

    # One retry on transport faults. At 37 feeds a single DNS or TLS blip is
    # routine — observed live: a resolver hiccup failed 32 of 37 sources at
    # once, and every one of them resolved fine seconds later. Without a retry
    # that blip costs an entire run's coverage. HTTP status errors are NOT
    # retried here: a 403 is an answer, not a fault.
    r = None
    for attempt in (1, 2):
        try:
            r = client.get(
                source["url"],
                headers=headers,
                timeout=source.get("timeout_seconds", 20),
                follow_redirects=True,
            )
            break
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == 1:
                log.debug("fetch retry   %-22s %s", source["id"], type(e).__name__)
                time.sleep(1.5)
                continue
            log.warning("fetch failed  %-22s %s: %s", source["id"], type(e).__name__, e)
            return FeedResult(source=source, error=f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - one bad feed must not kill the run
            log.warning("fetch failed  %-22s %s: %s", source["id"], type(e).__name__, e)
            return FeedResult(source=source, error=f"{type(e).__name__}: {e}")

    if r.status_code == 304:
        log.info("unchanged     %-22s 304", source["id"])
        return FeedResult(source=source, status=304)

    if r.status_code != 200:
        # 403 means the publisher is refusing automated access. We record and
        # move on — never retry with a disguised User-Agent (see sources.yml).
        log.warning("bad status    %-22s %s", source["id"], r.status_code)
        return FeedResult(source=source, status=r.status_code, error=f"HTTP {r.status_code}")

    if r.headers.get("etag") or r.headers.get("last-modified"):
        _cache[source["id"]] = (r.headers.get("etag"), r.headers.get("last-modified"))

    parsed = feedparser.parse(r.content)
    if parsed.bozo and not parsed.entries:
        log.warning("unparseable   %-22s %s", source["id"], parsed.get("bozo_exception"))
        return FeedResult(source=source, status=200, error="unparseable feed")

    log.info("fetched       %-22s %d entries", source["id"], len(parsed.entries))
    return FeedResult(source=source, entries=list(parsed.entries), status=200)


def fetch_all(sources: list[dict[str, Any]]) -> list[FeedResult]:
    """Fetch every feed concurrently. Failures are returned, not raised —
    a dead feed degrades coverage, it does not stop the run."""
    with httpx.Client() as client:
        with cf.ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as ex:
            results = list(ex.map(lambda s: fetch_one(s, client), sources))

    ok = sum(1 for r in results if r.ok)
    log.info("fetch complete: %d/%d sources returned entries", ok, len(results))
    return results
