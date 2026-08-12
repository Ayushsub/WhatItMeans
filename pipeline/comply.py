"""[Stage 9] Compliance gate (plan Part 6).

Three layers, in order of authority:

  Layer 1  deterministic regex   — hard fail, no LLM, cannot be argued with
  Layer 2  attribution check     — bull/bear must be someone else's view
  Layer 3  LLM reviewer          — catches implied advice regex cannot see

Layer 1 runs FIRST and is the real guarantee. The LLM is the safety net, not
the gate: a model that hallucinates "pass" must not be able to release content
that the regex would have blocked.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any

from llm import load_prompt
from llm.providers import CHEAP, available
from llm.router import AllProvidersFailed, complete_json
from models import Analysis, Cluster, ComplianceVerdict

from . import config

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _patterns() -> tuple[list[tuple[re.Pattern, str]], list[tuple[re.Pattern, str]]]:
    cfg = config.compliance()
    blocking = [
        (re.compile(p["pattern"], re.IGNORECASE), p["rule"])
        for p in cfg.get("blocking_patterns", [])
    ]
    minor = [
        (re.compile(p["pattern"], re.IGNORECASE), p["rule"])
        for p in cfg.get("minor_patterns", [])
    ]
    return blocking, minor


def _walk_strings(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """Flatten every string in the analysis to (json_path, text)."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, str):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, f"{path}[{i}]"))
    return out


def _context(text: str, m: re.Match, window: int = 70) -> str:
    """Surrounding text for a match.

    Logging the bare matched phrase is not enough to tell a real violation from
    a false positive. "must buy" is advice when a reader is the subject and
    plain mechanism when an index fund is; only the surrounding sentence says
    which. Every false positive found so far cost a legitimate story, so make
    them diagnosable from the log alone.
    """
    start = max(0, m.start() - window)
    end = min(len(text), m.end() + window)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def scan(analysis: Analysis) -> tuple[list[dict], list[dict]]:
    """Layer 1. Returns (blocking_hits, minor_hits). Pure, no network."""
    blocking_pats, minor_pats = _patterns()
    data = analysis.model_dump()

    blocking_hits: list[dict] = []
    minor_hits: list[dict] = []

    for path, text in _walk_strings(data):
        for pattern, rule in blocking_pats:
            m = pattern.search(text)
            if m:
                blocking_hits.append(
                    {
                        "field": path,
                        "rule": rule,
                        "quote": m.group(0),
                        "context": _context(text, m),
                    }
                )
        for pattern, rule in minor_pats:
            m = pattern.search(text)
            if m:
                minor_hits.append(
                    {
                        "field": path,
                        "rule": rule,
                        "quote": m.group(0),
                        "context": _context(text, m),
                    }
                )

    return blocking_hits, minor_hits


_ATTRIBUTION_PREFIX = {
    "bull_case": "Bulls argue that ",
    "bear_case": "Bears argue that ",
}

# Acronyms models lowercase when told to avoid Title Case. Restored
# deterministically because prompting alone does not do this reliably —
# "India relies on us for two-thirds of its lpg" shipped on a live run.
#
# "US" and "IT" are DELIBERATELY ABSENT — they are handled contextually below.
# Blanket-uppercasing them corrupts ordinary English: a live run published the
# gist bullet "IT establishes AI computing hardware as a new asset class",
# where the model had correctly written the pronoun "It". Same hazard with
# "us". Every other entry here is not an English word, so it is unambiguous.
_ACRONYMS = (
    "UK", "EU", "UAE", "RBI", "SEBI", "GST", "LPG", "CNG", "AI",
    "CPI", "WPI", "GDP", "GVA", "IIP", "PMI", "FII", "FPI", "DII", "IPO",
    "NBFC", "NPA", "MPC", "LAF", "OPEC", "FOMC", "USD", "INR", "EV", "NSE",
    "BSE", "SIP", "ETF", "TRAI", "IRDAI", "NCLT", "PSU", "MSME", "FMCG",
    "USFDA", "API", "ARPU", "EPS", "EBITDA", "QIP", "FDI", "NRI",
)
_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(sorted(_ACRONYMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Word-vs-acronym disambiguation, by what the token sits next to. Tuned for
# precision, not recall: leaving a genuine "US" lowercased reads as a typo,
# but uppercasing a pronoun reads as gibberish, so only near-certain contexts
# are converted.
_US_NOUNS = (
    r"dollars?|Fed(?:eral)?|Treasur(?:y|ies)|tariffs?|inflation|econom(?:y|ic)|"
    r"GDP|CPI|jobs|payrolls|president|Congress|Senate|markets?|crude|shale|"
    r"trade|imports?|exports?|administration|government|consumers?|recession|"
    r"rates?|bonds?|equities|stocks?|data|sanctions|yields?|banks?|firms?"
)
_IT_NOUNS = (
    r"services|sector|firms?|compan(?:y|ies)|majors?|industry|stocks?|exports?|"
    r"spending|hiring|budgets?|outsourcing|bellwethers?|pack"
)
_CONTEXTUAL = (
    (re.compile(rf"\bus\b(?=\s+(?:{_US_NOUNS})\b)", re.IGNORECASE), "US"),
    # The pronoun never follows "the", so "the us" is always the country.
    (re.compile(r"(?<=\bthe\s)us\b"), "US"),
    (re.compile(rf"\bit\b(?=\s+(?:{_IT_NOUNS})\b)", re.IGNORECASE), "IT"),
    (re.compile(r"(?<=\bIndian\s)it\b"), "IT"),
)


# The three gist bullets are already labelled by the template, so a model that
# restates the label wastes words from a 15-word budget and reads as a stutter
# next to the heading: "What to watch — What to watch: the next CPI print".
_GIST_LABEL_RE = re.compile(
    r"^\s*(what happened|why it matters|what to watch|watch(?: for)?)\s*[:\-–—]\s*",
    re.IGNORECASE,
)


def strip_gist_label(text: str) -> str:
    return _GIST_LABEL_RE.sub("", text, count=1).strip() or text


def fix_casing(text: str) -> str:
    """Restore acronym capitalisation and the leading capital.

    A safety net, not the primary mechanism: the analysis prompt already asks
    for normal sentence capitalisation. This exists because prompting alone did
    not do it reliably — "India relies on us for two-thirds of its lpg" shipped
    on a live run.
    """
    if not text:
        return text
    fixed = _ACRONYM_RE.sub(lambda m: m.group(1).upper(), text)
    for pattern, replacement in _CONTEXTUAL:
        fixed = pattern.sub(replacement, fixed)
    # "Rs" is a currency abbreviation, not an acronym — title case, not upper.
    fixed = re.sub(r"\brs\b(?=\s*[\d₹])", "Rs", fixed, flags=re.IGNORECASE)
    if fixed[0].islower():
        fixed = fixed[0].upper() + fixed[1:]
    return fixed


def repair_attribution(analysis: Analysis) -> tuple[Analysis, list[str]]:
    """Prepend an attribution to any case that lacks one.

    Dropping an otherwise-sound story because the model wrote "Margins will
    compress" instead of "Bears argue that margins will compress" throws away
    good analysis over framing. The repair is deterministic and cannot make the
    text less compliant: it converts an assertion into an attributed argument,
    which is exactly the required posture.

    Blocking remains the fallback if a repaired case still fails the check.
    """
    problems = check_attribution(analysis)
    if not problems:
        return analysis, []

    data = analysis.model_dump()
    repaired: list[str] = []
    for problem in problems:
        name = problem["field"].split(".")[0]
        prefix = _ATTRIBUTION_PREFIX.get(name)
        if not prefix:
            continue
        text = data[name]["argument"].strip()
        if not text:
            continue
        # Lowercase the first character so the joined sentence reads naturally,
        # but leave acronyms and proper nouns (RBI, US, Nifty) alone.
        if len(text) > 1 and not text[:4].isupper() and text[1].islower():
            text = text[0].lower() + text[1:]
        data[name]["argument"] = prefix + text
        repaired.append(name)

    if not repaired:
        return analysis, []
    return Analysis.model_validate(data), repaired


def check_attribution(analysis: Analysis) -> list[dict]:
    """Layer 2. Bull/bear arguments must be attributed to someone else.

    This is what keeps the site's most valuable feature — the two-sided case —
    on the explanation side of the SEBI line.
    """
    markers = [m.lower() for m in config.compliance().get("attribution_markers", [])]
    problems: list[dict] = []
    for name, case in (("bull_case", analysis.bull_case), ("bear_case", analysis.bear_case)):
        text = case.argument.lower()
        if not any(m in text for m in markers):
            problems.append(
                {
                    "field": f"{name}.argument",
                    "rule": "unattributed case",
                    "quote": case.argument[:120],
                }
            )
    return problems


def llm_review(analysis: Analysis, cluster: Cluster) -> ComplianceVerdict | None:
    """Layer 3 (PROMPT 3). Returns None if unavailable."""
    if not available(CHEAP):
        return None

    payload = {
        "analysis": analysis.model_dump(),
        "source_articles": [
            {"source": a.source_name, "headline": a.title, "snippet": a.snippet}
            for a in cluster.articles
        ],
    }
    try:
        raw = complete_json(
            CHEAP,
            load_prompt("compliance_system.txt"),
            json.dumps(payload, ensure_ascii=False),
            max_tokens=2048,
            temperature=0.0,
            label=f"comply:{cluster.id}",
        )
        return ComplianceVerdict.model_validate(raw)
    except (AllProvidersFailed, ValueError, Exception) as e:  # noqa: BLE001
        log.warning("compliance LLM review unavailable for %s: %s", cluster.id, e)
        return None


def _apply_rewrites(analysis: Analysis, rewrites: dict[str, str]) -> Analysis:
    """Apply the reviewer's field rewrites. Only dotted top-level and one-level
    nested paths are supported; anything deeper is rejected rather than guessed
    at, since a wrong write here would silently corrupt published content."""
    data = analysis.model_dump()
    applied = 0
    for path, value in rewrites.items():
        if not isinstance(value, str):
            continue
        parts = path.split(".")
        if len(parts) == 1 and isinstance(data.get(parts[0]), str):
            data[parts[0]] = value
            applied += 1
        elif (
            len(parts) == 2
            and isinstance(data.get(parts[0]), dict)
            and isinstance(data[parts[0]].get(parts[1]), str)
        ):
            data[parts[0]][parts[1]] = value
            applied += 1
        else:
            log.debug("skipping unsupported rewrite path %s", path)
    if applied:
        log.info("applied %d compliance rewrites", applied)
    return Analysis.model_validate(data)


def enforce(
    analyzed: list[tuple[Cluster, Analysis]]
) -> list[tuple[Cluster, Analysis]]:
    """Run all three layers. Returns only clusters cleared for publication."""
    approved: list[tuple[Cluster, Analysis]] = []

    for cluster, analysis in analyzed:
        # --- Layer 1: deterministic, authoritative -----------------------
        blocking, minor = scan(analysis)
        if blocking:
            for h in blocking[:3]:
                log.warning(
                    "BLOCKED %s by regex: %s @%s matched '%s' in: %s",
                    cluster.id, h["rule"], h["field"], h["quote"],
                    h.get("context", ""),
                )
            continue

        # --- Layer 3: LLM reviewer (may rewrite minor issues) ------------
        verdict = llm_review(analysis, cluster)
        if verdict is not None:
            if verdict.has_blocking:
                log.warning(
                    "BLOCKED %s by reviewer: %s",
                    cluster.id,
                    "; ".join(v.quote[:60] for v in verdict.violations if v.severity == "blocking"),
                )
                continue
            if verdict.rewritten_fields:
                try:
                    analysis = _apply_rewrites(analysis, verdict.rewritten_fields)
                except Exception as e:  # noqa: BLE001
                    log.warning("rewrite failed for %s (%s); keeping original", cluster.id, e)

        # --- Re-scan after rewrites: a rewrite must not introduce a new
        #     violation, and this is cheap insurance against that.
        blocking, minor = scan(analysis)
        if blocking:
            log.warning("BLOCKED %s: rewrite introduced a violation", cluster.id)
            continue

        # --- Layer 2: attribution (repair first, block only if that fails) --
        analysis, repaired = repair_attribution(analysis)
        if repaired:
            log.info("%s: added attribution to %s", cluster.id, ", ".join(repaired))

        attribution = check_attribution(analysis)
        if attribution:
            log.warning(
                "BLOCKED %s: unattributed case survived repair (%s)",
                cluster.id,
                ", ".join(p["field"] for p in attribution),
            )
            continue

        # A repair must not have introduced a banned phrase.
        blocking, _ = scan(analysis)
        if blocking:
            log.warning("BLOCKED %s: attribution repair introduced a violation", cluster.id)
            continue

        if minor:
            log.info(
                "%s published with %d minor flags: %s",
                cluster.id, len(minor), ", ".join(h["rule"] for h in minor[:3]),
            )

        # Cosmetic, last: restore acronym casing in reader-facing text.
        data = analysis.model_dump()
        for field in ("headline", "hook", "what_happened", "market_reaction"):
            data[field] = fix_casing(data[field])
        data["gist"] = [fix_casing(strip_gist_label(g)) for g in data["gist"]]
        for case in ("bull_case", "bear_case"):
            data[case]["argument"] = fix_casing(data[case]["argument"])
        analysis = Analysis.model_validate(data)

        approved.append((cluster, analysis))

    log.info("compliance: %d/%d clusters approved", len(approved), len(analyzed))
    return approved
