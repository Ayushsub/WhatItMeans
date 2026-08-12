"""Router failover tests (plan Part 10).

Verifies the behaviour the whole $0 design leans on: a dead or rate-limited
provider must advance to the next one, and a total outage must fail cleanly
rather than crash the run.

Uses a stub transport — no network, no API keys, no spend.
Run: python tests/test_router.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import router  # noqa: E402
from llm.providers import Provider  # noqa: E402

P1 = Provider(name="p1", base_url="https://p1.test/v1", model="m1", key_env="TEST_K1")
P2 = Provider(name="p2", base_url="https://p2.test/v1", model="m2", key_env="TEST_K2")
P3 = Provider(name="p3", base_url="https://p3.test/v1", model="m3", key_env="TEST_K3")


def _ok_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# Captured ONCE at import. router.httpx is the same module object as httpx, so
# patching router.httpx.Client mutates the global — re-reading it inside
# install() would capture an already-patched factory and nest infinitely.
_REAL_CLIENT = httpx.Client


def install(handler) -> None:
    """Patch httpx.Client used inside router.complete with a stub transport."""

    def factory(*args, **kwargs):
        return _REAL_CLIENT(transport=httpx.MockTransport(handler))

    router.httpx.Client = factory


def restore() -> None:
    router.httpx.Client = _REAL_CLIENT


def main() -> int:
    failures: list[str] = []
    import os

    os.environ.update({"TEST_K1": "k1", "TEST_K2": "k2", "TEST_K3": "k3"})

    # --- 1. first provider 429s -> falls through to the second -----------
    calls: list[str] = []

    def h_429(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "p1.test":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_ok_body('{"ok": true}'))

    install(h_429)
    try:
        out = router.complete_json([P1, P2], "sys", "user", label="t1")
        if out != {"ok": True}:
            failures.append(f"429 failover returned {out!r}")
        if calls != ["p1.test", "p2.test"]:
            failures.append(f"429 failover call order was {calls}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"429 failover raised {type(e).__name__}: {e}")

    # --- 2. auth failure advances and does NOT retry the same provider ---
    calls.clear()

    def h_401(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "p1.test":
            return httpx.Response(401, text="bad key")
        return httpx.Response(200, json=_ok_body('{"ok": 1}'))

    install(h_401)
    try:
        router.complete_json([P1, P2], "sys", "user", label="t2")
        if calls.count("p1.test") != 1:
            failures.append(f"401 was retried {calls.count('p1.test')} times, expected 1")
    except Exception as e:  # noqa: BLE001
        failures.append(f"401 failover raised {type(e).__name__}: {e}")

    # --- 3. every provider down -> AllProvidersFailed, not a crash -------
    install(lambda r: httpx.Response(503, text="down"))
    try:
        router.complete_json([P1, P2, P3], "sys", "user", label="t3")
        failures.append("total outage did not raise AllProvidersFailed")
    except router.AllProvidersFailed:
        pass
    except Exception as e:  # noqa: BLE001
        failures.append(f"total outage raised {type(e).__name__}, expected AllProvidersFailed")

    # --- 4. no keys configured -> clear error ----------------------------
    for k in ("TEST_K1", "TEST_K2", "TEST_K3"):
        os.environ.pop(k, None)
    try:
        router.complete_json([P1, P2], "sys", "user", label="t4")
        failures.append("missing keys did not raise")
    except router.AllProvidersFailed as e:
        if "no API keys" not in str(e):
            failures.append(f"unhelpful missing-key error: {e}")
    os.environ.update({"TEST_K1": "k1", "TEST_K2": "k2"})

    # --- 5. tolerant JSON extraction -------------------------------------
    for label, body, expect in [
        ("fenced", '```json\n{"a": 1}\n```', {"a": 1}),
        ("bare fence", '```\n{"a": 2}\n```', {"a": 2}),
        ("prose prefix", 'Here is the analysis:\n{"a": 3}', {"a": 3}),
        ("trailing prose", '{"a": 4}\nHope that helps!', {"a": 4}),
        ("clean", '{"a": 5}', {"a": 5}),
        # --- malformed output the model actually produces -------------------
        # An unescaped English quotation inside a string is the single most
        # common way the model breaks its own JSON. It surfaced live as
        # "Expecting ',' delimiter" and cost a whole cluster once, because the
        # first attempt AND its retry both did it.
        (
            "unescaped inner quotes",
            '{"a": "The so-called "risk-free" rate is the anchor."}',
            {"a": 'The so-called "risk-free" rate is the anchor.'},
        ),
        ("trailing comma", '{"a": 6, "b": [1, 2,],}', {"a": 6, "b": [1, 2]}),
        # Must NOT corrupt text that merely contains structural punctuation,
        # nor double-escape quotes the model escaped correctly.
        ("punctuation in string", '{"a": "rose 4%, then fell: sharply"}',
         {"a": "rose 4%, then fell: sharply"}),
        ("already escaped", '{"a": ["step \\"one\\"", "step two"]}',
         {"a": ['step "one"', "step two"]}),
    ]:
        install(lambda r, b=body: httpx.Response(200, json=_ok_body(b)))
        try:
            got = router.complete_json([P1], "sys", "user", label=f"t5-{label}")
            if got != expect:
                failures.append(f"JSON extraction {label}: got {got!r}, want {expect!r}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"JSON extraction {label} raised {type(e).__name__}: {e}")

    # Genuinely unrecoverable output must still raise, not return junk.
    install(lambda r: httpx.Response(200, json=_ok_body("no json here at all")))
    try:
        router.complete_json([P1], "sys", "user", label="t5-garbage")
        failures.append("unparseable response did not raise")
    except Exception:  # noqa: BLE001 - any raise is correct here
        pass

    restore()

    print("ran 14 router checks")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all router checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
