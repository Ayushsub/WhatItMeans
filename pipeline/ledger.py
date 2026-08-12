"""[Stage 10] Derived-facts ledger.

This is the answer to "continuity without retention" (plan Part 2, RAG verdict
and Part 3). It is NOT a database of news and NOT a vector store — it is a
few-KB append-only list of derived facts, looked up by dictionary key.

RETENTION CONTRACT: rows contain no headline, no snippet, no article text, no
URL. A row is too thin to reconstruct a news article from, which is precisely
what lets it outlive the 24h window without breaking the retention rule.

  ALLOWED:  {"date": "2026-08-11", "event_type": "policy_rate_change",
             "entity": "RBI", "value": "-25bps"}
  FORBIDDEN: anything resembling reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from models import Analysis, Cluster, LedgerFact

log = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "ledger.json"

# Facts older than this are pruned. Long enough for "third cut this year",
# short enough that the file stays a few KB.
LEDGER_RETENTION_DAYS = 400

# Derived from watchlist macro themes -> a stable event_type vocabulary.
_EVENT_TYPES = {
    "Interest Rates": "policy_rate",
    "Inflation": "inflation_print",
    "Currency": "currency_move",
    "Crude & Energy": "energy_price",
    "Fiscal & Budget": "fiscal_action",
    "Trade & Tariffs": "trade_action",
    "Institutional Flows": "fund_flows",
    "Credit & Liquidity": "credit_conditions",
    "Growth": "growth_print",
}

# Numbers worth remembering. Deliberately narrow — we store magnitudes, not prose.
#
# A bare "%" is NOT safe to capture: on the first live run this attached "15%"
# (a US tariff figure) to a policy_rate fact for the US Fed. That fact is fed
# back into future analysis prompts as prior context, so a mis-scoped number
# is worse than no number — it invites the model to state a wrong rate.
# A percentage is therefore only recorded when it sits near a word that matches
# the event type.
_VALUE_PATTERNS = (
    (r"(-?\d+(?:\.\d+)?)\s*bps\b", "{}bps"),  # unambiguous, always safe
    (r"₹\s?([\d,]+(?:\.\d+)?)\s*(lakh crore|crore|lakh)", "₹{} {}"),
)

# WHICH event types a given kind of magnitude may attach to.
#
# The unit itself carries the constraint: a policy rate is measured in bps or
# percent and can never be a crore amount. Without this the value was extracted
# once per cluster and stamped onto EVERY theme it carried, which wrote the row
#   {"event_type": "policy_rate", "entity": "US Fed", "value": "₹10,463 crore"}
# from a story about customs-duty collections. These rows are fed back into
# future analysis prompts as prior context, so a wrong number here does not
# just sit in a file — it invites the model to state a wrong rate.
_BPS_EVENTS = {"policy_rate", "credit_conditions"}
_CURRENCY_EVENTS = {"fiscal_action", "trade_action", "fund_flows", "credit_conditions"}

# Events driven by a named institution; everything else is better identified by
# its theme. Stops "energy_price / RBI" from being written just because RBI
# happened to be tagged on a crude story.
_INSTITUTION_EVENTS = {"policy_rate", "credit_conditions", "fiscal_action"}

_THEME_FOR_EVENT = {v: k for k, v in _EVENT_TYPES.items()}

# Percentages are only trusted within this many characters of a matching term.
_PCT_CONTEXT = {
    "policy_rate": ("repo", "policy rate", "interest rate", "rate to", "rate at"),
    "inflation_print": ("inflation", "cpi", "wpi"),
    "growth_print": ("gdp", "growth", "gva"),
    "credit_conditions": ("yield", "credit growth", "deposit growth"),
}
_PCT_WINDOW = 60


def _value_for(text: str, event_type: str) -> str | None:
    """The one magnitude worth remembering for THIS event type, or None.

    Scoped per event type rather than per cluster: the same sentence can carry
    a rate in bps and a collection figure in crore, and only one of them
    belongs on any given fact.
    """
    import re

    if event_type in _BPS_EVENTS:
        m = re.search(_VALUE_PATTERNS[0][0], text)
        if m:
            return _VALUE_PATTERNS[0][1].format(*m.groups())
    if event_type in _CURRENCY_EVENTS:
        m = re.search(_VALUE_PATTERNS[1][0], text)
        if m:
            return _VALUE_PATTERNS[1][1].format(*m.groups())
    if event_type in _PCT_CONTEXT:
        return _scoped_percentage(text, event_type)
    return None


def _scoped_percentage(text: str, event_type: str) -> str | None:
    """Return a percentage only if it sits near a term matching event_type.

    Prevents a tariff or market-move percentage being recorded as, say, a
    policy rate — a wrong number here propagates into future prompts.
    """
    import re

    terms = _PCT_CONTEXT.get(event_type, ())
    low = text.lower()
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
        start = max(0, m.start() - _PCT_WINDOW)
        window = low[start : m.end() + _PCT_WINDOW]
        if any(term in window for term in terms):
            return f"{m.group(1)}%"
    return None


class Ledger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or LEDGER_PATH
        self.facts: list[LedgerFact] = self._load()

    def _load(self) -> list[LedgerFact]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("ledger unreadable (%s); starting empty", e)
            return []
        out = []
        for row in raw:
            try:
                out.append(LedgerFact.model_validate(row))
            except Exception:  # noqa: BLE001 - drop malformed rows silently
                continue
        return out

    def save(self) -> None:
        cutoff = (date.today() - timedelta(days=LEDGER_RETENTION_DAYS)).isoformat()
        self.facts = [f for f in self.facts if f.date >= cutoff]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([f.model_dump() for f in self.facts], indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("ledger saved: %d facts (%s)", len(self.facts), self.path.name)

    # -- read ------------------------------------------------------------

    def related(self, cluster: Cluster, limit: int = 6) -> list[dict]:
        """Prior facts sharing an entity or event_type with this cluster.

        A plain filter over a short list. This is the whole "retrieval" system,
        and it is why no vector database is needed.
        """
        entities = {e.lower() for e in cluster.tags.entities}
        types = {_EVENT_TYPES[t] for t in cluster.tags.macro_themes if t in _EVENT_TYPES}
        if not entities and not types:
            return []

        hits = [
            f
            for f in self.facts
            if f.entity.lower() in entities or f.event_type in types
        ]
        hits.sort(key=lambda f: f.date, reverse=True)
        return [
            {"date": f.date, "event": f.event_type, "entity": f.entity, "value": f.value}
            for f in hits[:limit]
        ]

    # -- write -----------------------------------------------------------

    def record(self, approved: list[tuple[Cluster, Analysis]]) -> None:
        """Extract and append derived facts for published clusters."""
        import re

        today = datetime.now(timezone.utc).date().isoformat()
        existing = {(f.date, f.event_type, f.entity) for f in self.facts}
        added = 0

        for cluster, analysis in approved:
            types = [
                _EVENT_TYPES[t] for t in cluster.tags.macro_themes if t in _EVENT_TYPES
            ]
            if not types:
                continue
            # Only what_happened is used — the factual section.
            text = analysis.what_happened

            for event_type in types[:2]:
                # Identify the fact by its institution only when the event is
                # institution-driven; otherwise the theme is the honest label.
                if event_type in _INSTITUTION_EVENTS and cluster.tags.entities:
                    entity = cluster.tags.entities[0]
                else:
                    entity = _THEME_FOR_EVENT.get(event_type, "market")

                fact_value = _value_for(text, event_type)
                key = (today, event_type, entity)
                if key in existing:
                    continue
                existing.add(key)
                self.facts.append(
                    LedgerFact(
                        date=today,
                        event_type=event_type,
                        entity=entity,
                        value=fact_value,
                        cluster_id=cluster.id,
                    )
                )
                added += 1

        if added:
            log.info("ledger: recorded %d new derived facts", added)
        self.save()
