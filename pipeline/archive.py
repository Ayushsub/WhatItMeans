"""Carry forward still-fresh stories from the previously published site.

WHY THIS EXISTS. Each run analyzes only a handful of new clusters. Without
carry-forward the rendered site would contain ONLY those, so the homepage would
show ~6 stories and be wiped every 20 minutes — not a feed. Worse, a run where
every provider returns 429 would publish an empty site.

RETENTION IS NOT WEAKENED. The only source read here is the currently published
site, whose contents are by definition already inside the retention window.
Items past RETENTION_HOURS are dropped and never republished, so a story still
disappears within 24h of publication. Nothing is stored anywhere new.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models import PublishedItem

from . import config

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PAGES_BRANCH = "gh-pages"
_CLUSTER_DIR = "api/v1/clusters"


def _from_local(site_dir: Path) -> list[dict]:
    """Read the previous build from disk. Used on local re-runs."""
    d = site_dir / _CLUSTER_DIR
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("*.json"):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _from_git() -> list[dict]:
    """Read the previous build out of the published branch.

    CI starts from a clean checkout with no ./site, so git is the only place
    the last build exists.
    """
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", check=False,
            )
            return r.stdout if r.returncode == 0 else None
        except (FileNotFoundError, OSError):
            return None

    ref = None
    for candidate in (f"origin/{PAGES_BRANCH}", PAGES_BRANCH):
        git("fetch", "-q", "origin", PAGES_BRANCH)
        if git("rev-parse", "--verify", "-q", candidate):
            ref = candidate
            break
    if ref is None:
        return []

    listing = git("ls-tree", "--name-only", f"{ref}:{_CLUSTER_DIR}")
    if not listing:
        return []

    out = []
    for name in listing.split():
        if not name.endswith(".json"):
            continue
        blob = git("show", f"{ref}:{_CLUSTER_DIR}/{name}")
        if not blob:
            continue
        try:
            out.append(json.loads(blob))
        except json.JSONDecodeError:
            continue
    return out


def load_previous(site_dir: Path) -> list[PublishedItem]:
    """Previously published items that are still inside the retention window."""
    raw = _from_local(site_dir) or _from_git()
    if not raw:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.RETENTION_HOURS)
    kept: list[PublishedItem] = []
    expired = 0

    for row in raw:
        try:
            item = PublishedItem.model_validate(row)
        except Exception:  # noqa: BLE001 - schema drift between versions
            continue
        published = item.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            expired += 1
            continue
        kept.append(item)

    log.info(
        "carried forward %d previous stories (%d aged out past %dh)",
        len(kept), expired, config.RETENTION_HOURS,
    )
    return kept


def _source_urls(item: PublishedItem) -> set[str]:
    return {s["url"] for s in item.sources if s.get("url")}


def published_signature(items: list[PublishedItem]) -> tuple[set[str], set[str]]:
    """(cluster_ids, source_urls) already covered by the live site.

    Cluster ids alone are NOT a reliable identity across runs. A cluster's id
    derives from its earliest article, so anything that changes which article
    is earliest — a new article joining, or the per-publisher cap trimming an
    older one — mints a new id for the same story. Observed live: one gold
    story published twice, 40 minutes apart, under two ids.

    Source URLs are stable, so overlap on them is the dependable signal.
    """
    ids: set[str] = set()
    urls: set[str] = set()
    for item in items:
        ids.add(item.cluster_id)
        urls |= _source_urls(item)
    return ids, urls


def merge(new: list[PublishedItem], previous: list[PublishedItem]) -> list[PublishedItem]:
    """Combine new and carried-forward items, newest analysis winning.

    Deduplicated on cluster_id AND on shared source articles. The second check
    is what keeps the same story from appearing twice under different ids when
    the cluster's lead article changed between runs.
    """
    by_id: dict[str, PublishedItem] = {i.cluster_id: i for i in previous}
    for item in new:
        by_id[item.cluster_id] = item

    # Drop older items that share source articles with a newer one.
    ordered = sorted(by_id.values(), key=lambda i: i.updated_at, reverse=True)
    kept: list[PublishedItem] = []
    claimed: set[str] = set()
    dropped = 0
    for item in ordered:
        urls = _source_urls(item)
        if urls and urls & claimed:
            dropped += 1
            continue
        claimed |= urls
        kept.append(item)

    if dropped:
        log.info("merge: dropped %d stale re-runs of stories already carried", dropped)

    return sorted(kept, key=lambda i: i.published_at, reverse=True)
