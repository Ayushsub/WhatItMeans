"""Failover LLM router.

This is deliberately NOT a framework (plan Part 2). It is an ordered provider
list plus "on 429/5xx, try the next one" — which is exactly the behaviour that
LangChain's provider abstractions would have obscured.

Policy:
  - React to 429s, do not pre-count quota. Counters would need to persist
    across ephemeral CI runners for near-zero benefit.
  - A 401/403 is a configuration error, not a capacity problem: log loudly and
    advance, because a typo'd key should not silently halve throughput.
  - Retry the SAME provider once on a transient network error, then advance.
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from .providers import Provider, available, describe_configuration

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}


class AllProvidersFailed(RuntimeError):
    """Every provider in the tier refused or errored."""


# Providers that returned 429 in THIS process. A daily token quota does not
# refill mid-run, so re-asking an exhausted provider on every subsequent call
# buys nothing and costs a full round-trip each time. Observed live: with both
# Gemini entries quota-exhausted, every analysis call paid two dead requests
# before reaching a working provider — roughly 40 wasted round-trips per run.
#
# Deliberately in-process only, and deliberately 429-only:
#   - a fresh CI run starts clean, so a quota that reset overnight is picked up
#     again without any persistence or clock logic
#   - 5xx means transient capacity, which CAN recover within a run, so those
#     still advance without being remembered
#
# This is still "react to 429s, do not pre-count quota" — it just declines to
# make the same failing request twenty times.
_EXHAUSTED: set[str] = set()


def reset_exhausted() -> None:
    """Clear the 429 memo. For tests, and for long-lived callers."""
    _EXHAUSTED.clear()


def _repair_json(text: str) -> str:
    """Escape double quotes that appear INSIDE a JSON string value.

    The single most common way a model breaks its own JSON: it writes an
    ordinary English quotation and does not escape it —

        "explanation": "The so-called "risk-free" rate is the anchor."

    which the parser reports as `Expecting ',' delimiter`. Observed repeatedly
    on live runs, and it cost a whole cluster once because both the first
    attempt and its retry produced it.

    The rule for telling a closing quote from an inner one: a real closing
    quote is followed by structural punctuation (`,` `:` `}` `]`) or end of
    input. Anything else is prose. Also drops trailing commas, which are the
    second most common break.
    """
    out: list[str] = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
                continue
            # Decide whether this closes the string.
            rest = text[i + 1 :]
            stripped = rest.lstrip()
            if stripped == "" or stripped[0] in ",:}]":
                in_string = False
                out.append(ch)
            else:
                out.append('\\"')  # inner quote: escape it
            continue
        out.append(ch)

    repaired = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", repaired)


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Models wrap JSON in ```json fences, prepend "Here is the analysis:", or
    emit trailing prose despite instructions. Rather than fight that with
    prompt engineering alone, recover the object here.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    span = text[start : end + 1]
    try:
        return json.loads(span)
    except json.JSONDecodeError as first_error:
        # Last resort: repair, and say so, because silently accepting mangled
        # output is worse than a loud recovery.
        try:
            obj = json.loads(_repair_json(span))
        except json.JSONDecodeError:
            raise first_error from None
        log.warning("recovered malformed JSON by escaping inner quotes")
        return obj


def _call_provider(
    provider: Provider,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    client: httpx.Client,
) -> str:
    payload: dict = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode and provider.supports_json_mode:
        payload["response_format"] = {"type": "json_object"}

    r = client.post(
        f"{provider.base_url}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        },
        timeout=_TIMEOUT,
    )

    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            f"HTTP {r.status_code}: {r.text[:300]}", request=r.request, response=r
        )

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as e:
        raise ValueError(f"unexpected response shape from {provider.name}: {data}") from e


def complete(
    tier: list[Provider],
    system: str,
    user: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    json_mode: bool = True,
    label: str = "",
) -> str:
    """Try each configured provider in order. Returns raw response text."""
    providers = available(tier)
    if not providers:
        raise AllProvidersFailed(
            "no API keys configured. " + describe_configuration()
        )

    errors: list[str] = []
    live = [p for p in providers if p.name not in _EXHAUSTED]
    if not live:
        raise AllProvidersFailed(
            f"all providers already quota-exhausted this run for {label}: "
            + ", ".join(sorted(_EXHAUSTED))
        )

    with httpx.Client() as client:
        for provider in live:
            for attempt in (1, 2):
                try:
                    t0 = time.monotonic()
                    out = _call_provider(
                        provider,
                        system,
                        user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_mode=json_mode,
                        client=client,
                    )
                    log.info(
                        "llm ok  %-18s %-20s %.1fs %d chars",
                        label, provider.name, time.monotonic() - t0, len(out),
                    )
                    return out

                except httpx.HTTPStatusError as e:
                    code = e.response.status_code
                    if code in (401, 403):
                        # Configuration error — never retry, and make it loud.
                        log.error(
                            "llm AUTH FAIL %s (%s): check %s",
                            provider.name, code, provider.key_env,
                        )
                        errors.append(f"{provider.name}: auth {code}")
                        break
                    if code == 429:
                        # Quota, not congestion. Stop asking for this run.
                        _EXHAUSTED.add(provider.name)
                        log.warning(
                            "llm %s 429 -> next provider (skipping it for the "
                            "rest of this run)", provider.name,
                        )
                        errors.append(f"{provider.name}: 429")
                        break
                    if code in _RETRY_STATUSES:
                        log.warning(
                            "llm %s %s -> next provider", provider.name, code
                        )
                        errors.append(f"{provider.name}: {code}")
                        break  # capacity issue: advance, do not retry same one
                    log.warning("llm %s HTTP %s", provider.name, code)
                    errors.append(f"{provider.name}: {code}")
                    break

                except (httpx.TimeoutException, httpx.TransportError) as e:
                    # Transient network fault: one retry on the same provider.
                    if attempt == 1:
                        log.warning(
                            "llm %s network error (%s), retrying once",
                            provider.name, type(e).__name__,
                        )
                        time.sleep(2)
                        continue
                    errors.append(f"{provider.name}: {type(e).__name__}")
                    break

                except Exception as e:  # noqa: BLE001
                    log.warning("llm %s failed: %s", provider.name, e)
                    errors.append(f"{provider.name}: {type(e).__name__}")
                    break

    raise AllProvidersFailed(f"all providers failed for {label}: {'; '.join(errors)}")


def complete_json(
    tier: list[Provider],
    system: str,
    user: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "",
) -> dict:
    """complete() plus tolerant JSON extraction."""
    raw = complete(
        tier,
        system,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
        label=label,
    )
    return _extract_json(raw)
