"""[Stage 5] Watchlist tagging. Fully deterministic — no LLM (plan Part 4).

Alias matching is word-boundary anchored so 'ITC' does not match 'switch' and
'L&T' does not match 'salt'. Aliases exist because feeds are inconsistent about
entity names.
"""

from __future__ import annotations

import functools
import logging
import re

from models import Cluster, Tags

from . import config

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _compiled() -> dict[str, list[tuple[str, re.Pattern, float, str | None]]]:
    """Pre-compile every alias into (label, pattern, weight, ticker) per group.

    Cached because this runs against every cluster on every run.
    """
    wl = config.watchlist()
    out: dict[str, list[tuple[str, re.Pattern, float, str | None]]] = {}

    for group in ("macro_themes", "institutions", "sectors", "companies", "indices"):
        items = []
        for item in wl.get(group, []):
            weight = float(item.get("weight", 4))
            ticker = item.get("ticker")
            label = item["label"]
            for alias in item.get("aliases", []):
                # \b fails on aliases ending in punctuation (L&T, S&P 500), so
                # only apply boundaries where the edge chars are word chars.
                esc = re.escape(alias)
                left = r"\b" if alias[0].isalnum() else ""
                right = r"\b" if alias[-1].isalnum() else ""
                items.append(
                    (label, re.compile(f"{left}{esc}{right}", re.IGNORECASE), weight, ticker)
                )
        out[group] = items
    return out


def tag_cluster(cluster: Cluster) -> Cluster:
    """Attach watchlist tags. Matches against every article in the cluster,
    since a story's defining entity may only appear in the follow-up piece."""
    haystack = " \n ".join(f"{a.title} {a.snippet}" for a in cluster.articles)
    compiled = _compiled()

    entities: list[str] = []
    tickers: list[str] = []
    sectors: list[str] = []
    themes: list[str] = []

    for group, bucket in (
        ("macro_themes", themes),
        ("institutions", entities),
        ("sectors", sectors),
        ("companies", entities),
        ("indices", entities),
    ):
        for label, pattern, _weight, ticker in compiled[group]:
            if label in bucket:
                continue
            if pattern.search(haystack):
                bucket.append(label)
                if ticker and ticker not in tickers:
                    tickers.append(ticker)

    cluster.tags = Tags(
        entities=entities, tickers=tickers, sectors=sectors, macro_themes=themes
    )
    return cluster


def tag_all(clusters: list[Cluster]) -> list[Cluster]:
    tagged = [tag_cluster(c) for c in clusters]
    hit = sum(1 for c in tagged if not c.tags.is_empty())
    log.info("tagged %d clusters (%d matched the watchlist)", len(tagged), hit)
    return tagged
