"""Derived-facts ledger tests.

Two things are checked here, and only the first is obvious:

1. RETENTION. A ledger row must stay too thin to reconstruct an article from.
   This is what lets it outlive the 24h window (plan Part 3).

2. FACTUAL SCOPING. Ledger rows are fed back into future analysis prompts as
   prior context, so a wrong number does not merely sit in a file — it invites
   the model to state a wrong rate to readers. Two such rows shipped:

     {"event_type": "policy_rate",     "entity": "US Fed", "value": "15%"}
     {"event_type": "policy_rate",     "entity": "US Fed", "value": "₹10,463 crore"}

   The first was a tariff percentage, the second a customs-duty collection.
   Neither is a policy rate. The unit itself carries the constraint.

Run: python tests/test_ledger.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import LedgerFact  # noqa: E402
from pipeline.ledger import _THEME_FOR_EVENT, _value_for  # noqa: E402


def main() -> int:
    failures: list[str] = []
    checked = 0

    # --- value scoping: the unit must match the event type ------------------
    duty = "The government collected ₹10,463 crore in customs duty on gold imports."
    tariff = "The US imposed a 15% tariff on imports from India."

    cases: list[tuple[str, str, str | None]] = [
        # (text, event_type, expected value)
        ("The RBI cut the repo rate by 25 bps to 5.25%.", "policy_rate", "25bps"),
        ("Retail inflation held at 4.4% in July.", "inflation_print", "4.4%"),
        (duty, "trade_action", "₹10,463 crore"),
        (duty, "fiscal_action", "₹10,463 crore"),
        # REGRESSIONS — both of these shipped to the live ledger.
        (duty, "policy_rate", None),
        (duty, "inflation_print", None),
        (tariff, "policy_rate", None),
        # A market move is not a growth print.
        ("The Sensex fell 1.2% on the day.", "growth_print", None),
        # Crude prices are not measured in crore.
        ("Brent rose to $88 while duty collections hit ₹10,463 crore.", "energy_price", None),
    ]

    for text, event_type, expected in cases:
        checked += 1
        got = _value_for(text, event_type)
        if got != expected:
            failures.append(
                f"_value_for({event_type!r}) -> {got!r}, want {expected!r} "
                f"| text: {text[:60]!r}"
            )

    # --- every event type must have a human-readable theme fallback ---------
    for event_type in ("policy_rate", "energy_price", "trade_action", "fund_flows"):
        checked += 1
        if event_type not in _THEME_FOR_EVENT:
            failures.append(f"no theme label for event_type {event_type!r}")

    # --- retention: a row must not be able to carry article text ------------
    checked += 1
    allowed = set(LedgerFact.model_fields)
    forbidden = {"title", "headline", "snippet", "url", "summary", "text", "body"}
    leaked = allowed & forbidden
    if leaked:
        failures.append(f"LedgerFact exposes article-text fields: {leaked}")

    checked += 1
    if allowed != {"date", "event_type", "entity", "value", "cluster_id"}:
        failures.append(f"LedgerFact schema changed unexpectedly: {sorted(allowed)}")

    # --- the live ledger on disk must satisfy the same rules ----------------
    ledger_path = ROOT / "data" / "ledger.json"
    if ledger_path.exists():
        import json

        rows = json.loads(ledger_path.read_text(encoding="utf-8"))
        rate_events = {
            "policy_rate", "inflation_print", "growth_print",
            "currency_move", "energy_price",
        }
        for row in rows:
            checked += 1
            value = row.get("value") or ""
            if "₹" in value and row.get("event_type") in rate_events:
                failures.append(
                    f"live ledger holds a currency amount on a rate event: {row}"
                )
            if set(row) - allowed:
                failures.append(f"live ledger row has unexpected fields: {row}")

    print(f"ran {checked} ledger checks")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all ledger checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
