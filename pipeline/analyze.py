"""[Stages 6b-8] Triage, analysis, and schema validation.

One analysis call per selected cluster produces the headline, hook, gist AND
the full breakdown together (plan Part 4, step 7). Splitting these across calls
would cost 3-4x the inference and let the tiers drift in tone.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from llm import load_prompt
from llm.providers import CHEAP, QUALITY, available
from llm.router import AllProvidersFailed, complete_json
from models import Analysis, Cluster

from . import config
from .ledger import Ledger
from .rank import apply_triage

log = logging.getLogger(__name__)


def triage(candidates: list[Cluster]) -> dict[str, dict]:
    """One batched call ranking the shortlist (PROMPT 2).

    Optional by design: any failure returns {} and the deterministic ranking
    stands. Triage improves selection, it is never required for the site to
    update.
    """
    if not candidates or not available(CHEAP):
        return {}

    payload = [
        {
            "cluster_id": c.id,
            "headline": c.lead.title,
            "source_count": c.source_count,
            "sources": sorted({a.source_name for a in c.articles}),
            "tags": c.tags.macro_themes + c.tags.entities + c.tags.sectors,
            "age_minutes": round(c.lead.age_minutes),
        }
        for c in candidates
    ]

    try:
        out = complete_json(
            CHEAP,
            load_prompt("triage_system.txt"),
            json.dumps(payload, ensure_ascii=False),
            max_tokens=2048,
            temperature=0.1,
            label="triage",
        )
    except Exception as e:  # noqa: BLE001
        # Triage is strictly optional (plan Part 4, step 6): deterministic
        # ranking already selected a valid set. NOTHING here may abort the run
        # — a narrow except let an AttributeError kill a whole run once.
        log.warning("triage unavailable (%s); using deterministic ranking only", e)
        return {}

    # Models return either {"ranked": [...]} or a bare [...] despite the schema.
    if isinstance(out, dict):
        ranked = out.get("ranked") or []
    elif isinstance(out, list):
        ranked = out
    else:
        log.warning("triage returned %s; ignoring", type(out).__name__)
        return {}

    if not isinstance(ranked, list):
        log.warning("triage 'ranked' was %s; ignoring", type(ranked).__name__)
        return {}

    return {
        r["cluster_id"]: r
        for r in ranked
        if isinstance(r, dict) and isinstance(r.get("cluster_id"), str)
    }


def _build_user_prompt(cluster: Cluster, ledger: Ledger) -> str:
    articles = [
        {
            "source": a.source_name,
            "headline": a.title,
            "snippet": a.snippet,
            "published_at": a.published_at.isoformat(),
            "url": str(a.url),
        }
        for a in cluster.articles
    ]
    facts = ledger.related(cluster)
    return json.dumps(
        {
            "cluster_id": cluster.id,
            "articles": articles,
            "watchlist_tags": cluster.tags.model_dump(),
            "prior_related_events": facts,
        },
        ensure_ascii=False,
        indent=2,
    )


def analyze_one(cluster: Cluster, ledger: Ledger) -> Analysis | None:
    """PROMPT 1 with one schema-failure retry (plan Part 4, step 8)."""
    system = load_prompt("analysis_system.txt")
    user = _build_user_prompt(cluster, ledger)

    for attempt in (1, 2):
        try:
            raw = complete_json(
                QUALITY,
                system,
                user,
                max_tokens=4096,
                temperature=0.3 if attempt == 1 else 0.1,
                label=f"analyze:{cluster.id}",
            )
        except AllProvidersFailed as e:
            log.error("analysis failed for %s: %s", cluster.id, e)
            return None
        except ValueError as e:
            log.warning("unparseable JSON for %s (attempt %d): %s", cluster.id, attempt, e)
            continue

        try:
            return Analysis.model_validate(raw)
        except ValidationError as e:
            # Feed the actual error back so the retry can correct itself —
            # cheaper and more reliable than a generic "try again".
            log.warning(
                "schema mismatch for %s (attempt %d): %s",
                cluster.id, attempt, str(e)[:200],
            )
            if attempt == 1:
                user += (
                    "\n\nYour previous response failed schema validation with:\n"
                    f"{str(e)[:800]}\n"
                    "Return corrected JSON matching the schema exactly."
                )

    log.error("dropping %s: could not produce valid analysis", cluster.id)
    return None


def analyze_selected(
    selected: list[Cluster], ledger: Ledger, candidates: list[Cluster] | None = None
) -> list[tuple[Cluster, Analysis]]:
    """Run optional triage, then analyze the final selection."""
    if candidates:
        scores = triage(candidates)
        if scores:
            reordered = apply_triage(candidates, scores)
            selected = reordered[: config.MAX_ANALYZED_PER_RUN]
            log.info("triage reordered selection; analyzing %d clusters", len(selected))

    out: list[tuple[Cluster, Analysis]] = []
    for cluster in selected:
        analysis = analyze_one(cluster, ledger)
        if analysis is not None:
            out.append((cluster, analysis))

    log.info("analyzed %d/%d clusters successfully", len(out), len(selected))
    return out
