"""Pydantic schemas — the contract between every pipeline stage.

Nothing here imports from pipeline/ or llm/. Every stage takes one of these
and returns another, which is what keeps the pipeline a straight line of pure
functions (see plan Part 2: LangGraph no-fit).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


# --------------------------------------------------------------------------
# Stage 1-2: fetch -> normalize
# --------------------------------------------------------------------------


class SourceTier(str, Enum):
    """Editorial weight of a feed. Drives the ranking score in stage 6."""

    WIRE = "wire"  # PTI, Reuters — first to report, high trust
    MAJOR = "major"  # ET, Mint, BS — full newsrooms
    SECTORAL = "sectoral"  # trade press, niche but authoritative
    AGGREGATOR = "aggregator"  # lowest weight, often re-reports others


class Article(BaseModel):
    """One normalized news item. Never holds full article text — see plan Part 6."""

    id: str  # stable hash of canonical url
    title: str
    snippet: str  # hard-capped at source.snippet_max_words
    url: HttpUrl
    source_id: str  # the FEED (e.g. "et_markets")
    source_name: str
    source_tier: SourceTier
    published_at: datetime
    fetched_at: datetime

    # The MASTHEAD (e.g. "economic_times"). Distinct from source_id because one
    # publisher runs several feeds — ET alone has Markets, Stocks, Banking and
    # Tech. Corroboration must be counted per publisher: three ET desks carrying
    # the same story is one outlet, not three independent confirmations.
    # Defaults to source_id, which is exactly right for single-feed publishers.
    publisher: str = ""

    @property
    def masthead(self) -> str:
        return self.publisher or self.source_id

    @field_validator("published_at", "fetched_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        """Naive datetimes from sloppy RSS feeds are assumed UTC.

        Without this the 24h retention window comparison raises on mixed
        aware/naive values, which is a real failure mode with Indian feeds.
        """
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def age_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.published_at).total_seconds() / 60


# --------------------------------------------------------------------------
# Stage 4-6: cluster -> tag -> rank
# --------------------------------------------------------------------------


class Tags(BaseModel):
    """Deterministic watchlist matches. No LLM involved (plan Part 4, step 5)."""

    entities: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    macro_themes: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.entities or self.tickers or self.sectors or self.macro_themes)


class Cluster(BaseModel):
    """A group of articles telling the same story."""

    id: str  # stable across runs: hash of the earliest article id
    articles: list[Article]
    tags: Tags = Field(default_factory=Tags)
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    triage_score: int | None = None  # from PROMPT 2, if triage ran
    triage_reason: str | None = None

    @property
    def lead(self) -> Article:
        """Earliest-published article — the one we treat as canonical."""
        return min(self.articles, key=lambda a: a.published_at)

    @property
    def source_count(self) -> int:
        """Distinct PUBLISHERS. Corroboration signal for ranking and confidence.

        Counted per masthead, not per feed. This number is shown to readers as
        "reported by N sources" and feeds the confidence label, so counting ET
        Markets + ET Stocks as two would be a factual overstatement on the page.
        """
        return len({a.masthead for a in self.articles})

    @property
    def newest_at(self) -> datetime:
        return max(a.published_at for a in self.articles)


# --------------------------------------------------------------------------
# Stage 7: analyze  (PROMPT 1 output contract)
# --------------------------------------------------------------------------

Direction = Literal["tailwind", "headwind", "mixed", "neutral"]
Horizon = Literal["days", "weeks", "months", "quarters"]
Confidence = Literal["high", "medium", "low"]


class AffectedGroup(BaseModel):
    group: str
    effect: str


class SectorImpact(BaseModel):
    sector: str
    direction: Direction
    why: str


class TransmissionMechanism(BaseModel):
    """The core of the product — the causal chain from event to price."""

    chain: list[str] = Field(min_length=2, max_length=6)
    explanation: str
    time_horizon: Horizon


class Case(BaseModel):
    """A bull or bear case. Always attributed, never the site's view."""

    argument: str
    depends_on: list[str] = Field(default_factory=list)


class WatchVariable(BaseModel):
    variable: str
    why_it_matters: str
    when: str


class JargonTerm(BaseModel):
    term: str
    plain_meaning: str


class Analysis(BaseModel):
    """Strict contract for PROMPT 1. A malformed LLM response fails here, once,
    then gets one retry (plan Part 4, step 8) before the cluster is dropped."""

    headline: str = Field(max_length=140)
    hook: str
    gist: list[str] = Field(min_length=3, max_length=3)
    what_happened: str
    who_is_affected: list[AffectedGroup]
    sectors: list[SectorImpact]
    transmission_mechanism: TransmissionMechanism
    market_reaction: str
    bull_case: Case
    bear_case: Case
    variables_to_watch: list[WatchVariable]
    confidence: Confidence
    confidence_reason: str
    jargon: list[JargonTerm] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 9: comply  (PROMPT 3 output contract)
# --------------------------------------------------------------------------


class Violation(BaseModel):
    field: str
    rule: int
    quote: str
    severity: Literal["blocking", "minor"]


class ComplianceVerdict(BaseModel):
    passed: bool = Field(alias="pass")
    violations: list[Violation] = Field(default_factory=list)
    rewritten_fields: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    model_config = {"populate_by_name": True}

    @property
    def has_blocking(self) -> bool:
        return any(v.severity == "blocking" for v in self.violations)


# --------------------------------------------------------------------------
# Stage 10: ledger — derived facts only, never article text (plan Part 3)
# --------------------------------------------------------------------------


class LedgerFact(BaseModel):
    """One derived fact. Deliberately too thin to reconstruct a news article
    from, which is what lets it outlive the 24h retention window."""

    date: str  # ISO date, no time — coarse on purpose
    event_type: str  # e.g. "policy_rate_change"
    entity: str  # e.g. "RBI"
    value: str | None = None  # e.g. "-25bps"
    cluster_id: str  # for dedupe only; the cluster itself is long gone


# --------------------------------------------------------------------------
# Stage 11: render — the published artifact
# --------------------------------------------------------------------------


class PublishedItem(BaseModel):
    """What actually reaches the site and the /api/v1 JSON.

    `tier` is unused today and always "free" — it exists so a future paywall is
    a render-time filter rather than a schema migration (plan Part 7).
    """

    cluster_id: str
    analysis: Analysis
    tags: Tags
    sources: list[dict[str, str]]  # {name, url, published_at} — attribution
    published_at: datetime
    updated_at: datetime
    confidence: Confidence
    tier: Literal["free", "premium"] = "free"
