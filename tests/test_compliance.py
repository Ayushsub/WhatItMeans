"""Compliance gate tests (plan Part 10).

The deterministic layer must reject 100% of the blocking fixtures WITHOUT any
LLM involvement. If this file ever fails, the site must not deploy.

Run: python -m tests.test_compliance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    Analysis,
    AffectedGroup,
    Case,
    JargonTerm,
    SectorImpact,
    TransmissionMechanism,
    WatchVariable,
)
from pipeline.comply import check_attribution, repair_attribution, scan  # noqa: E402

FIXTURES = json.loads(
    (ROOT / "tests" / "fixtures" / "violations.json").read_text(encoding="utf-8")
)


def _analysis(**overrides) -> Analysis:
    """A clean, compliant baseline analysis. Tests inject one bad field."""
    base = dict(
        headline="Rupee slips to 95.28 as crude climbs for a fourth session",
        hook="Why a weaker rupee shows up in your fuel bill months later",
        gist=[
            "Rupee fell 11 paise to 95.28 against the dollar",
            "Costlier crude widens the import bill",
            "Watch the next inflation print",
        ],
        what_happened=(
            "The rupee closed 11 paise lower at 95.28 against the US dollar on "
            "11 August 2026, its fourth straight decline, as Brent traded near $87."
        ),
        who_is_affected=[
            AffectedGroup(group="Importers", effect="Input costs rise roughly in line with the currency move."),
            AffectedGroup(group="IT exporters", effect="Dollar revenue converts into more rupees, a modest tailwind."),
        ],
        sectors=[
            SectorImpact(sector="Oil & Gas", direction="headwind", why="Crude is imported and priced in dollars."),
            SectorImpact(sector="IT & Tech", direction="tailwind", why="Export revenue is dollar denominated."),
        ],
        transmission_mechanism=TransmissionMechanism(
            chain=[
                "Crude rises and the rupee weakens",
                "The landed cost of imported oil increases",
                "State fuel retailers absorb part of the gap as under-recovery",
                "Subsidy burden and fuel inflation feed into the next CPI print",
            ],
            explanation=(
                "A weaker rupee raises the landed cost of crude within days. Fuel "
                "retailers absorb the gap for weeks before it reaches pump prices. "
                "The inflation effect typically appears with a one to two month lag."
            ),
            time_horizon="months",
        ),
        market_reaction="The Sensex closed 312 points lower, according to the source reporting.",
        bull_case=Case(
            argument="Bulls argue that if crude retreats below $85, the pressure reverses quickly.",
            depends_on=["Crude falling below $85", "Stable portfolio flows"],
        ),
        bear_case=Case(
            argument="Bears contend that persistent dollar strength keeps the import bill elevated.",
            depends_on=["Dollar index staying firm"],
        ),
        variables_to_watch=[
            WatchVariable(variable="CPI print", why_it_matters="Confirms whether fuel costs passed through", when="12 September 2026"),
            WatchVariable(variable="Brent crude", why_it_matters="The upstream driver of the whole chain", when="ongoing"),
        ],
        confidence="medium",
        confidence_reason="Four sources agree on the move but differ on the cause.",
        jargon=[JargonTerm(term="Under-recovery", plain_meaning="The gap between what a fuel costs and what it is sold for.")],
    )
    base.update(overrides)
    return Analysis(**base)


def _set_path(analysis: Analysis, path: str, text: str) -> Analysis:
    """Inject text at a dotted/indexed path like 'bull_case.argument'."""
    data = analysis.model_dump()
    parts = path.replace("]", "").split(".")
    cur = data
    for part in parts[:-1]:
        if "[" in part:
            key, idx = part.split("[")
            cur = cur[key][int(idx)]
        else:
            cur = cur[part]
    last = parts[-1]
    if "[" in last:
        key, idx = last.split("[")
        cur[key][int(idx)] = text
    else:
        cur[last] = text
    return Analysis.model_validate(data)


def main() -> int:
    failures: list[str] = []
    checked = 0

    # --- blocking and minor fixtures must be caught -----------------------
    for case in FIXTURES["cases"]:
        checked += 1
        analysis = _set_path(_analysis(), case["field"], case["text"])
        blocking, minor = scan(analysis)

        if case["expect"] == "blocking":
            if not blocking:
                failures.append(
                    f"NOT BLOCKED: {case['name']!r} -> {case['text'][:70]!r}"
                )
        elif case["expect"] == "minor":
            if not minor and not blocking:
                failures.append(
                    f"NOT FLAGGED: {case['name']!r} -> {case['text'][:70]!r}"
                )

    # --- clean text must NOT trip the gate (false-positive check) ---------
    for clean in FIXTURES["clean"]:
        checked += 1
        analysis = _set_path(_analysis(), "what_happened", clean["text"])
        blocking, _ = scan(analysis)
        if blocking:
            failures.append(
                f"FALSE POSITIVE: {clean['name']!r} blocked by "
                f"{blocking[0]['rule']!r} on {blocking[0]['quote']!r}"
            )

    # --- attribution layer ------------------------------------------------
    checked += 1
    unattributed = _analysis(
        bull_case=Case(argument="The sector recovers as input costs normalise.", depends_on=[])
    )
    if not check_attribution(unattributed):
        failures.append("attribution check missed an unattributed bull case")

    checked += 1
    if check_attribution(_analysis()):
        failures.append("attribution check false-positived on a clean analysis")

    # --- attribution repair: fix framing instead of dropping the story ----
    checked += 1
    repaired, fields = repair_attribution(unattributed)
    if not fields:
        failures.append("repair_attribution did not repair an unattributed case")
    elif check_attribution(repaired):
        failures.append("repaired analysis still fails the attribution check")
    elif not repaired.bull_case.argument.startswith("Bulls argue that "):
        failures.append(f"unexpected repair result: {repaired.bull_case.argument!r}")

    checked += 1
    # Repair must be a no-op on already-attributed text.
    clean = _analysis()
    unchanged, fields = repair_attribution(clean)
    if fields or unchanged.bull_case.argument != clean.bull_case.argument:
        failures.append("repair_attribution modified an already-attributed case")

    checked += 1
    # Repair must not introduce a blocking phrase.
    risky = _analysis(
        bear_case=Case(argument="Margins will compress over the next two quarters.", depends_on=[])
    )
    repaired_risky, _ = repair_attribution(risky)
    if scan(repaired_risky)[0]:
        failures.append("repair introduced a blocking violation")

    # --- acronym casing restoration ---------------------------------------
    from pipeline.comply import fix_casing

    # KNOWN MISS, accepted deliberately: "India relies on us for two-thirds of
    # its LPG" leaves "us" lowercase, because "for" is not a US-context noun and
    # "relies on us" is a grammatical pronoun phrase. Recall was traded for
    # precision here on purpose — a stray lowercase "us" reads as a typo, while
    # uppercasing a pronoun reads as gibberish and shipped to the live site once.
    for got, want in [
        ("gold rises as us inflation data looms", "Gold rises as US inflation data looms"),
        ("the us dollar strengthened overnight", "The US dollar strengthened overnight"),
        ("rbi holds the repo rate steady", "RBI holds the repo rate steady"),
        ("Rupee slips to 95.28 as crude climbs", "Rupee slips to 95.28 as crude climbs"),
        ("Gold prices rise rs 6,600/10g in 3 days", "Gold prices rise Rs 6,600/10g in 3 days"),
        ("indian it firms face margin pressure", "Indian IT firms face margin pressure"),
        # REGRESSION: blanket uppercasing published "IT establishes AI computing
        # hardware as a new asset class" from a correctly-written pronoun.
        ("It establishes a new asset class", "It establishes a new asset class"),
        ("The deal matters because it changes how banks fund themselves",
         "The deal matters because it changes how banks fund themselves"),
        ("Tariffs give us reason to expect slower growth",
         "Tariffs give us reason to expect slower growth"),
    ]:
        checked += 1
        if fix_casing(got) != want:
            failures.append(f"fix_casing({got!r}) -> {fix_casing(got)!r}, want {want!r}")

    checked += 1
    # Must not mangle words that merely contain an acronym as a substring.
    if fix_casing("The business unit uses gas") != "The business unit uses gas":
        failures.append(f"fix_casing over-matched: {fix_casing('The business unit uses gas')!r}")

    # --- redundant gist labels ---------------------------------------------
    from pipeline.comply import strip_gist_label

    for got, want in [
        ("What to watch: the next CPI print", "the next CPI print"),
        ("Why it matters — importers pay more", "importers pay more"),
        ("The rupee fell 11 paise", "The rupee fell 11 paise"),
        # Must not empty a bullet that is nothing but the label.
        ("What to watch", "What to watch"),
    ]:
        checked += 1
        if strip_gist_label(got) != want:
            failures.append(f"strip_gist_label({got!r}) -> {strip_gist_label(got)!r}, want {want!r}")

    print(f"ran {checked} compliance checks")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all compliance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
