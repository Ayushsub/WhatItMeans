"""[Stage 2] Normalize raw feed entries into Article objects.

Three things here are load-bearing and were driven by what real feeds actually
emit (see plan Part 10, feed probing):

1. DATE PARSING. feedparser returns published_parsed=None for RBI, whose dates
   look like 'Tue, 11 Aug 2026 10:30:00' with no timezone. Naive timestamps are
   localized to the source's default_tz — assuming UTC would backdate Indian
   items by 5.5h and age them out of the 24h window early.

2. SNIPPET CAPPING. Enforced here, not by convention, because it is the
   copyright control (plan Part 6). Feeds hand us full paragraphs; we keep ~40
   words.

3. HARD EXCLUDES. Broker recommendation headlines are dropped at the door
   rather than scrubbed downstream.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from models import Article, SourceTier

from . import config
from .fetch import FeedResult

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "source")

# Date formats seen in the wild that email.utils cannot handle.
_FALLBACK_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d %b %Y %H:%M:%S",
    "%a, %d %b %Y %H:%M:%S",
)


def _clean_text(raw: str | None) -> str:
    """Strip HTML and unescape entities.

    Unescaped twice on purpose: several Indian feeds double-encode, which
    surfaces as literal '&#39;' or the mangled 'day#39;s' in rendered output.
    """
    if not raw:
        return ""
    text = html.unescape(html.unescape(raw))
    text = _TAG_RE.sub(" ", text)
    text = text.replace("#39;", "'").replace("&nbsp;", " ")
    return _WS_RE.sub(" ", text).strip()


def _canonical_url(url: str) -> str:
    """Drop tracking params so the same story from one source hashes to one id."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.query:
        kept = [
            kv
            for kv in parts.query.split("&")
            if kv and not any(kv.lower().startswith(p) for p in _TRACKING_PARAMS)
        ]
        parts = parts._replace(query="&".join(kept))
    return urlunsplit(parts._replace(fragment=""))


def _parse_date(entry: dict, default_tz: str) -> datetime | None:
    """Best-effort publication time. Returns None if genuinely unknowable."""
    # 1. feedparser's own struct_time (already UTC when it succeeds)
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime(*st[:6], tzinfo=timezone.utc)

    # 2. Raw strings — RFC-822 via email.utils, then explicit formats
    for key in ("published", "updated", "created", "date", "dc_date"):
        raw = entry.get(key)
        if not raw or not isinstance(raw, str):
            continue
        dt = None
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            for fmt in _FALLBACK_DATE_FORMATS:
                try:
                    dt = datetime.strptime(raw.strip(), fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            continue
        if dt.tzinfo is None:
            # The RBI case: localize to the publisher's timezone, not UTC.
            dt = dt.replace(tzinfo=ZoneInfo(default_tz))
        return dt.astimezone(timezone.utc)

    return None


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:. ") + "…"


def _compile_hard_excludes() -> list[re.Pattern]:
    pats = config.watchlist().get("hard_exclude_patterns", [])
    return [re.compile(p, re.IGNORECASE) for p in pats]


def _cap_per_publisher(articles: list[Article]) -> tuple[list[Article], int]:
    """Trim each publisher to MAX_ARTICLES_PER_PUBLISHER, newest first.

    Applied as a post-pass over the whole corpus rather than inside the feed
    loop on purpose: capping during the loop would spend a newsroom's entire
    budget on whichever of its feeds happens to be listed first in sources.yml,
    so ET Markets would starve ET Banking for no reason anyone chose.
    """
    limit = config.MAX_ARTICLES_PER_PUBLISHER
    by_pub: dict[str, list[Article]] = {}
    for art in articles:
        by_pub.setdefault(art.masthead, []).append(art)

    kept: list[Article] = []
    trimmed = 0
    for pub, items in by_pub.items():
        if len(items) <= limit:
            kept.extend(items)
            continue
        items.sort(key=lambda a: a.published_at, reverse=True)
        kept.extend(items[:limit])
        trimmed += len(items) - limit
        log.info("publisher cap: %s %d -> %d articles", pub, len(items), limit)

    return kept, trimmed


def normalize_all(results: list[FeedResult]) -> list[Article]:
    """Flatten fetch results into deduplicated, in-window Articles."""
    hard_excludes = _compile_hard_excludes()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.RETENTION_HOURS)
    now = datetime.now(timezone.utc)

    articles: list[Article] = []
    seen_ids: set[str] = set()
    counts = {"excluded": 0, "stale": 0, "undated": 0, "dupe": 0, "bad": 0, "capped": 0}

    for result in results:
        if not result.ok:
            continue
        src = result.source
        max_words = src.get("snippet_max_words", 40)
        default_tz = src.get("default_tz", "Asia/Kolkata")
        publisher = src.get("publisher") or src["id"]

        # Cap per feed. Some desks return 200 entries (Indian Express, News18);
        # left uncapped, two such feeds supply a third of the entire corpus and
        # skew clustering toward whatever those desks happen to cover. Feeds are
        # newest-first, so a head slice keeps the most recent items.
        entries = result.entries[: src.get("max_entries", 40)]
        if len(entries) < len(result.entries):
            counts["capped"] += len(result.entries) - len(entries)

        for entry in entries:
            title = _clean_text(entry.get("title"))
            link = entry.get("link") or ""
            if not title or not link:
                counts["bad"] += 1
                continue

            if any(p.search(title) for p in hard_excludes):
                counts["excluded"] += 1
                continue

            published = _parse_date(entry, default_tz)
            if published is None:
                # Undated items are treated as "now". Feeds only list recent
                # items, so this is safe and keeps high-value undated sources
                # usable rather than silently dropping them.
                published = now
                counts["undated"] += 1
            elif published < cutoff:
                counts["stale"] += 1
                continue

            # Clocks drift and some feeds post-date; clamp so a bad timestamp
            # cannot pin an item to the top of the feed forever.
            if published > now + timedelta(minutes=30):
                published = now

            canon = _canonical_url(link)
            aid = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
            if aid in seen_ids:
                counts["dupe"] += 1
                continue
            seen_ids.add(aid)

            summary = _clean_text(entry.get("summary") or entry.get("description"))
            # Feeds often repeat the title as the summary; that adds no signal.
            if summary.lower().startswith(title.lower()[:40]):
                summary = summary[len(title) :].strip(" -–—:")

            try:
                articles.append(
                    Article(
                        id=aid,
                        title=title,
                        snippet=_cap_words(summary, max_words),
                        url=canon,
                        source_id=src["id"],
                        source_name=src["name"],
                        source_tier=SourceTier(src["tier"]),
                        publisher=publisher,
                        published_at=published,
                        fetched_at=now,
                    )
                )
            except Exception as e:  # noqa: BLE001 - bad URL etc, skip the item
                counts["bad"] += 1
                log.debug("normalize skip %s: %s", link[:60], e)

    articles, trimmed = _cap_per_publisher(articles)
    counts["capped"] += trimmed

    log.info(
        "normalized %d articles (dropped: %d broker-calls, %d stale, %d dupes, "
        "%d malformed, %d over per-feed cap; %d undated->now)",
        len(articles),
        counts["excluded"],
        counts["stale"],
        counts["dupe"],
        counts["bad"],
        counts["capped"],
        counts["undated"],
    )
    return articles
