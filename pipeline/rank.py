"""[Stage 6] Rank clusters and select which get an LLM call.

This stage decides the entire inference budget (plan Part 4). Deterministic
scoring runs first and is always authoritative about what is ELIGIBLE; the
optional LLM triage (PROMPT 2) only reorders the shortlist.

Keeping selection deterministic-first means a triage outage degrades ranking
quality, it never stops the site updating.
"""

from __future__ import annotations

import functools
import logging
import math
import re
from datetime import datetime, timezone

from models import Cluster, SourceTier

from . import config

log = logging.getLogger(__name__)

# Editorial weight per source tier — corroboration by a wire is worth more
# than corroboration by an aggregator that re-reports it.
_TIER_WEIGHT = {
    SourceTier.WIRE: 3.0,
    SourceTier.MAJOR: 2.0,
    SourceTier.SECTORAL: 1.5,
    SourceTier.AGGREGATOR: 0.8,
}

_RECENCY_HALFLIFE_HOURS = 6.0


@functools.lru_cache(maxsize=1)
def _noise_patterns() -> list[re.Pattern]:
    pats = config.watchlist().get("noise_patterns", [])
    return [re.compile(re.escape(p), re.IGNORECASE) for p in pats]


@functools.lru_cache(maxsize=1)
def _weight_lookup() -> dict[str, float]:
    """label -> weight, flattened across every watchlist group."""
    wl = config.watchlist()
    out: dict[str, float] = {}
    for group in ("macro_themes", "institutions", "sectors", "indices"):
        for item in wl.get(group, []):
            out[item["label"]] = float(item.get("weight", 4))
    return out


def score_cluster(
    cluster: Cluster,
    seen_ids: set[str] | None = None,
    seen_urls: set[str] | None = None,
) -> Cluster:
    """Deterministic importance score. Breakdown is retained for debugging —
    when a bad story gets analyzed, the breakdown tells you which knob to turn.
    """
    weights = _weight_lookup()
    b: dict[str, float] = {}

    # 1. Corroboration: distinct PUBLISHERS, weighted by tier. Sub-linear
    #    because the 5th outlet covering a story adds less than the 2nd.
    #
    #    Keyed on masthead, not feed. This is the largest single term in the
    #    score, so counting ET Markets + ET Stocks + ET Banking as three
    #    independent confirmations would let one publisher's editorial focus
    #    decide which stories get an LLM call.
    tier_sum = sum(
        _TIER_WEIGHT.get(a.source_tier, 1.0)
        for a in {a.masthead: a for a in cluster.articles}.values()
    )
    b["corroboration"] = 4.0 * math.log1p(tier_sum)

    # 2. Macro themes — longest transmission chains, the core of the product.
    b["macro"] = sum(weights.get(t, 5.0) for t in cluster.tags.macro_themes)

    # 3. Institutions/entities — trigger events.
    b["entities"] = sum(weights.get(e, 3.0) for e in cluster.tags.entities) * 0.6

    # 4. Sectors — breadth of impact.
    b["sectors"] = sum(weights.get(s, 4.0) for s in cluster.tags.sectors) * 0.4

    # 5. Recency, exponential decay on the newest article in the cluster.
    age_h = max(
        0.0,
        (datetime.now(timezone.utc) - cluster.newest_at).total_seconds() / 3600.0,
    )
    b["recency"] = 12.0 * math.exp(-age_h / _RECENCY_HALFLIFE_HOURS)

    # 6. Novelty — a story already on the site should not consume a second call.
    #
    # Checked on SOURCE URLS as well as cluster id, because the id is not a
    # stable identity across runs: it derives from the cluster's earliest
    # article, so a new article joining (or the per-publisher cap trimming an
    # old one) mints a fresh id for the same story. Observed live as the same
    # gold story analyzed twice, 40 minutes apart.
    #
    # A penalty, not a filter, deliberately: a genuinely developing story can
    # still outweigh -25 on corroboration and recency and earn a re-analysis.
    already_published = bool(seen_ids and cluster.id in seen_ids)
    if not already_published and seen_urls:
        already_published = any(str(a.url) in seen_urls for a in cluster.articles)
    b["novelty"] = -25.0 if already_published else 5.0

    # 7. Noise penalty.
    text = " ".join(a.title for a in cluster.articles)
    hits = sum(1 for p in _noise_patterns() if p.search(text))
    b["noise"] = -6.0 * hits

    # 8. Untagged stories are usually not market-relevant at all.
    if cluster.tags.is_empty():
        b["untagged"] = -8.0

    cluster.score_breakdown = {k: round(v, 2) for k, v in b.items()}
    cluster.score = round(sum(b.values()), 2)
    return cluster


def rank(
    clusters: list[Cluster],
    seen_ids: set[str] | None = None,
    seen_urls: set[str] | None = None,
) -> list[Cluster]:
    """Score and sort. Does not select — selection is a separate decision."""
    scored = [score_cluster(c, seen_ids, seen_urls) for c in clusters]
    scored.sort(key=lambda c: c.score, reverse=True)
    log.info(
        "ranked %d clusters (top score %.1f, median %.1f)",
        len(scored),
        scored[0].score if scored else 0.0,
        scored[len(scored) // 2].score if scored else 0.0,
    )
    return scored


def select_for_analysis(
    ranked: list[Cluster], limit: int | None = None
) -> tuple[list[Cluster], list[Cluster]]:
    """Split into (analyze_now, triage_candidates).

    Returns the shortlist for optional LLM triage as well, so the caller can
    decide whether to spend one triage call to reorder it.
    """
    limit = limit or config.MAX_ANALYZED_PER_RUN
    candidates = [c for c in ranked if c.score > 0][: config.TRIAGE_CANDIDATE_COUNT]
    return candidates[:limit], candidates


def apply_triage(candidates: list[Cluster], triage: dict[str, dict]) -> list[Cluster]:
    """Reorder candidates using PROMPT 2 output.

    Deterministic score and triage score are averaged rather than the LLM
    overriding outright — the model is good at judging explainer-worthiness and
    bad at judging corroboration, which the deterministic score already knows.
    """
    for c in candidates:
        entry = triage.get(c.id)
        if not entry:
            continue
        c.triage_score = entry.get("score")
        c.triage_reason = entry.get("reason")
        if entry.get("explainer_worthy") is False:
            c.score -= 20.0
        if c.triage_score is not None:
            # Deterministic scores run roughly 0-80; triage is 0-100. Scale
            # before blending so neither dominates by units alone.
            c.score = round(0.5 * c.score + 0.5 * (c.triage_score * 0.8), 2)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
