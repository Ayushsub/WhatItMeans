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
_CEREBRAS = "https://api.cerebras.ai/v1"
_MISTRAL = "https://api.mistral.ai/v1"
_OPENROUTER = "https://openrouter.ai/api/v1"

# WHAT ACTUALLY RUNS OUT IS TOKENS, NOT REQUESTS (verified Aug 2026).
#
# Groq's headline "14,400 requests/day" is an org-wide ceiling that the
# per-model caps make unreachable for this workload:
#
#   llama-3.3-70b-versatile   1K req/day    100K tokens/day
#   openai/gpt-oss-120b       1K req/day    200K tokens/day
#   llama-3.1-8b-instant                    500K tokens/day
#
# An analysis call carries the cluster plus its articles — roughly 4-5k tokens
# in, 1.5k out. So 100K TPD is about twenty analysis calls, not a thousand
# requests. That is the wall being hit, and no amount of retry logic fixes it.
#
# Hence the additions below. Cerebras and Mistral are chosen for token headroom,
# not speed; the router already handles failover, so breadth of quota is the
# only thing worth buying with a new entry here.

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
    # Cerebras: ~1M tokens/day per model, no card. The largest single block of
    # free headroom available, which is exactly what this pipeline is short of.
    #
    # CAVEAT — the free tier caps context at 8K. An analysis payload runs 4-6k,
    # so it fits, but a cluster with many long articles could overflow and 400.
    # The router treats that as "advance to the next provider", so it degrades
    # rather than breaking. Cerebras deliberately sits BELOW Groq for that
    # reason: it is depth, not first choice.
    Provider("cerebras-llama4-scout", _CEREBRAS,
             "llama-4-scout-17b-16e-instruct", "CEREBRAS_API_KEY"),
    Provider("cerebras-qwen3-32b", _CEREBRAS, "qwen-3-32b", "CEREBRAS_API_KEY"),
    # Mistral's free "Experiment" tier is the most generous quota here, but it
    # REQUIRES opting into training on your data. Everything sent is derived
    # from public news and gets published anyway, so the cost is low — but it
    # is a real choice, so this only activates if you set the key.
    Provider("mistral-medium", _MISTRAL, "mistral-medium-latest", "MISTRAL_API_KEY"),
    Provider("mistral-small", _MISTRAL, "mistral-small-latest", "MISTRAL_API_KEY"),
    # OpenRouter is LAST ON PURPOSE and is close to a token gesture: free models
    # are capped at 50 requests/day unless you have bought $10 of credits (then
    # 1,000/day). At ~600 calls/day this covers under a tenth of the load.
    #
    # Its :free roster also churns hard — llama-3.3-70b:free and
    # deepseek-chat-v3:free both went paid and started returning 404 "unavailable
    # for free". Do not trust any name here for long; run
    #     python -m llm.providers --openrouter
    # to list what is actually free right now, and update this line.
    Provider("openrouter-gpt-oss-120b", _OPENROUTER,
             "openai/gpt-oss-120b:free", "OPENROUTER_API_KEY"),
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
    # 500K tokens/day — five times the 70b's cap, and this tier's work is
    # mechanical judgement where the smaller model is sufficient.
    Provider("groq-llama-8b", _GROQ, "llama-3.1-8b-instant", "GROQ_API_KEY"),
    Provider("cerebras-qwen3-32b", _CEREBRAS, "qwen-3-32b", "CEREBRAS_API_KEY"),
    Provider("mistral-small", _MISTRAL, "mistral-small-latest", "MISTRAL_API_KEY"),
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


def probe() -> int:
    """Send one tiny request to every configured provider and report.

    This exists because a wrong model name NEVER crashes — the router treats an
    unrecognised status as "advance to the next provider", so a retired or
    key-invisible model silently costs you a whole provider and nothing in the
    logs says so. Run this after any key change or when quota errors start.

        python -m llm.providers
    """
    import os
    from pathlib import Path

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    import httpx

    seen: dict[str, Provider] = {}
    for p in QUALITY + CHEAP:
        seen.setdefault(f"{p.name}|{p.model}", p)

    print(f"\n{describe_configuration()}\n")
    print(f"{'provider':26} {'model':40} status")
    print("-" * 88)

    ok = 0
    with httpx.Client(timeout=30.0) as client:
        for p in seen.values():
            if not p.configured:
                print(f"{p.name:26} {p.model:40} - no key")
                continue
            try:
                r = client.post(
                    f"{p.base_url}/chat/completions",
                    json={
                        "model": p.model,
                        "messages": [{"role": "user", "content": "reply with: ok"}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    },
                    headers={"Authorization": f"Bearer {p.api_key}",
                             "Content-Type": "application/json"},
                )
                if r.status_code == 200:
                    ok += 1
                    print(f"{p.name:26} {p.model:40} OK")
                else:
                    detail = r.text[:90].replace("\n", " ")
                    print(f"{p.name:26} {p.model:40} HTTP {r.status_code}  {detail}")
            except Exception as e:  # noqa: BLE001
                print(f"{p.name:26} {p.model:40} {type(e).__name__}: {e}")

    print(f"\n{ok}/{len(seen)} usable.")
    if ok == 0:
        print("Nothing is reachable — check keys before debugging anything else.")
    return 0 if ok else 1


def list_openrouter_free() -> int:
    """Print the models OpenRouter is currently serving at zero cost.

    The :free roster changes constantly — models get promoted to paid and then
    404 with "unavailable for free", which the router swallows as just another
    dead provider. Ask the API instead of trusting a hardcoded name.
    """
    import os
    from pathlib import Path

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    import httpx

    try:
        r = httpx.get(f"{_OPENROUTER}/models", timeout=30.0)
        r.raise_for_status()
        models = r.json().get("data", [])
    except Exception as e:  # noqa: BLE001
        print(f"could not reach OpenRouter: {e}")
        return 1

    free = []
    for m in models:
        p = m.get("pricing") or {}
        try:
            if float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0:
                free.append(m)
        except (TypeError, ValueError):
            continue

    free.sort(key=lambda m: -(m.get("context_length") or 0))
    print(f"\n{len(free)} zero-cost model(s) on OpenRouter right now:\n")
    print(f"{'id':58} {'context':>9}")
    print("-" * 70)
    for m in free:
        print(f"{m.get('id', '?'):58} {m.get('context_length') or 0:>9,}")
    print("\nFree models are capped at 50 requests/day (1,000 if you have ever")
    print("bought $10 of credits). Paste a chosen id into QUALITY in this file.")
    return 0


if __name__ == "__main__":
    import sys

    if "--openrouter" in sys.argv:
        raise SystemExit(list_openrouter_free())
    raise SystemExit(probe())
