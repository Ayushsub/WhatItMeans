"""Free-tier provider registry.

Every provider here exposes an OpenAI-compatible /chat/completions endpoint,
including Gemini, so one HTTP client covers all of them. Adding a provider is
one entry in this list (plan Part 2, LLM routing).

Two tiers, chosen by TASK not by a difficulty classifier — cost-based routing
solves a problem we do not have, since every option is free:

  QUALITY  the analysis call (PROMPT 1). Reasoning quality matters most.
  CHEAP    triage (PROMPT 2) and compliance review (PROMPT 3). High volume,
           mechanical judgment.

Daily capacity, verified Aug 2026 (plan Part 1):
  Groq                14,400 req/day, 30 RPM
  Gemini 2.5 Flash    250-1,500 req/day depending on model
  Gemini Flash-Lite   ~1,000 req/day
The pipeline needs ~60-95 calls/day, so any ONE of these covers it. The list
exists for resilience, not capacity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str
    # Gemini's OpenAI-compat layer rejects some params the others accept.
    supports_json_mode: bool = True

    @property
    def api_key(self) -> str | None:
        key = os.environ.get(self.key_env, "").strip()
        return key or None

    @property
    def configured(self) -> bool:
        return self.api_key is not None


_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
_GROQ = "https://api.groq.com/openai/v1"

# MODEL AVAILABILITY IS A REAL FAILURE MODE, and it has now bitten twice.
#
#   1. gemini-2.5-flash returned 404 "no longer available to new users" — the
#      model was retired out from under a valid key.
#   2. gemini-3.6-flash was listed here after that, then a key rotation showed
#      the replacement key does not expose it at all: GET /models tops out at
#      gemini-3.5-flash. Model access varies BY KEY, not just by date.
#
# The second one is the nastier failure. The router treats an unrecognised
# status as "advance to the next provider", so a wrong model name never
# crashes — it silently wastes the primary provider on every call and quietly
# downgrades quality. Nothing in the logs says "you are running on the backup".
#
# Hence: pin the newest model verified against THIS key, with the rolling
# "-latest" alias immediately behind it (survives the next retirement), then
# fail over to an entirely different provider family. Alias drift is safe here:
# compliance is enforced by deterministic regex and output shape by pydantic,
# so a changed model cannot leak advice or malformed JSON.
#
# To re-verify after any key change:
#   curl -H "Authorization: Bearer $GEMINI_API_KEY" \
#        https://generativelanguage.googleapis.com/v1beta/openai/models
#   curl -H "Authorization: Bearer $GROQ_API_KEY" \
#        https://api.groq.com/openai/v1/models

# Ordered by preference. The router walks this list top to bottom.
QUALITY: list[Provider] = [
    Provider("gemini-3.5-flash", _GEMINI, "gemini-3.5-flash", "GEMINI_API_KEY"),
    Provider("gemini-flash-latest", _GEMINI, "gemini-flash-latest", "GEMINI_API_KEY"),
    # A different architecture, not just a different host: if Gemini is
    # rate-limited account-wide, falling back to another Gemini alias achieves
    # nothing. gpt-oss-120b sits ahead of llama-70b because the analysis call
    # is structured reasoning into strict JSON, which it handles better.
    Provider("groq-gpt-oss-120b", _GROQ, "openai/gpt-oss-120b", "GROQ_API_KEY"),
    Provider("groq-llama-70b", _GROQ, "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    Provider(
        "openrouter-free",
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct:free",
        "OPENROUTER_API_KEY",
    ),
]

# Gemini Flash-Lite leads this tier on EVIDENCE, not preference. Groq's free
# tier caps tokens-per-minute (6k-30k), and the compliance call carries the
# whole analysis plus every source article — roughly 3k tokens each. Six of
# those back to back 429'd Groq on every single call in a live run, burning a
# wasted round-trip before failing over. Flash-Lite answered the same calls in
# 1.0-1.6s. Groq stays as failover, where its 14,400 req/day is the useful
# property; triage (one small payload per run) is well within its limits.
CHEAP: list[Provider] = [
    Provider("gemini-flash-lite-latest", _GEMINI, "gemini-flash-lite-latest", "GEMINI_API_KEY"),
    Provider("groq-llama-8b", _GROQ, "llama-3.1-8b-instant", "GROQ_API_KEY"),
    Provider("groq-llama-70b", _GROQ, "llama-3.3-70b-versatile", "GROQ_API_KEY"),
]


def available(tier: list[Provider]) -> list[Provider]:
    return [p for p in tier if p.configured]


def describe_configuration() -> str:
    """Human-readable key status. Printed at startup so a missing key is
    obvious in CI logs rather than surfacing as a confusing run-time failure."""
    seen: dict[str, bool] = {}
    for p in QUALITY + CHEAP:
        seen[p.key_env] = p.configured
    parts = [f"{env}={'set' if ok else 'MISSING'}" for env, ok in sorted(seen.items())]
    return " ".join(parts)
