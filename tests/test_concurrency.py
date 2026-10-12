"""The concurrency item from the adversarial tier in DESIGN.md 13: many calls, one key.

One plugin instance serves every concurrent user in the process, and `ToolResultCache` takes no
lock. That is deliberate and documented, but "deliberate" is a claim, and the claim is that
every mutation is one dict operation and correctness never depends on two of them being atomic
with respect to each other. This file tries to break that.

The interleaving is real, not simulated by ordering. A tool call is an await point: the plugin
serves, the tool runs, the plugin stores, and other invocations run in between. `call_tool`
below reproduces that shape, so `asyncio.gather` genuinely interleaves the halves rather than
running each call to completion in turn.

What is being defended here is not throughput. A thundering herd on a cold key is acceptable and
expected: several callers miss, several run the tool, several store the same answer. What is not
acceptable is a caller receiving another caller's payload, an entry ending up half written, or
the statistics disagreeing with what happened.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from google.adk.tools.base_tool import BaseTool
from test_plugin import FakeContext, make_plugin

from adk_smart_router.config import CACHE_HIT_KEY

TOOL = BaseTool(name="get_rate", description="")
OTHER = BaseTool(name="place_order", description="")  # never declared cacheable
ARGS: dict[str, Any] = {"currency": "GBP"}


async def call_tool(
    router: Any,
    user: str,
    result: dict[str, Any],
    *,
    tool: BaseTool = TOOL,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One full tool call: serve, yield to the loop while the tool "runs", then store.

    Returns whatever the caller would see, which is the served payload on a hit and the tool's
    own result on a miss.
    """
    ctx = FakeContext(user_id=user)
    served = await router.before_tool_callback(tool=tool, tool_args=args or ARGS, tool_context=ctx)
    await asyncio.sleep(0)  # the tool executing, and every other invocation getting a turn
    if served is not None:
        # ADK still runs after_tool on a served result, and that path must not re-store.
        await router.after_tool_callback(
            tool=tool, tool_args=args or ARGS, tool_context=ctx, result=served
        )
        return served
    await router.after_tool_callback(
        tool=tool, tool_args=args or ARGS, tool_context=ctx, result=result
    )
    return result


# --- The cold-key herd ----------------------------------------------------------


async def test_a_cold_key_herd_converges_on_one_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sixteen callers miss at once. Sixteen tools run. One entry exists afterwards.

    Every caller gets the real tool result, nobody gets a partial one, and the seventeenth call
    is a clean hit. Suppressing the herd would need a per-key in-flight lock, which buys a few
    duplicate tool calls on a cold key and costs a lock held across an arbitrary tool execution.
    """
    router = make_plugin(monkeypatch)

    results = await asyncio.gather(
        *(call_tool(router, "u1", {"rate": 1.27, "caller": n}) for n in range(16))
    )

    assert [r["rate"] for r in results] == [1.27] * 16
    assert sorted(r["caller"] for r in results) == list(range(16))
    assert not any(CACHE_HIT_KEY in r for r in results), "nothing should hit on a cold key"

    stats = router.cache.stats
    assert stats["misses"] == 16
    assert stats["hits"] == 0
    assert stats["stores"] == 16, "each caller stored, last write wins"

    served = await call_tool(router, "u1", {"rate": 9.99})
    assert served[CACHE_HIT_KEY]["cache_hit"] is True
    assert served["rate"] == 1.27, "the entry is one caller's payload, not a merge of sixteen"
    assert served["caller"] in range(16)


async def test_a_warm_key_herd_serves_everyone_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sixteen concurrent hits, each mutating what it got. The entry survives all of them."""
    router = make_plugin(monkeypatch)
    await call_tool(router, "u1", {"rate": 1.27, "nested": {"fees": [1, 2]}})

    async def hit_and_wreck(n: int) -> dict[str, Any]:
        got = await call_tool(router, "u1", {"unused": True})
        got["rate"] = n
        got["nested"]["fees"].append(n)
        return got

    await asyncio.gather(*(hit_and_wreck(n) for n in range(16)))

    clean = await call_tool(router, "u1", {"unused": True})
    assert clean["rate"] == 1.27
    assert clean["nested"]["fees"] == [1, 2]
    assert router.cache.stats["hits"] == 17


# --- Identity under concurrency --------------------------------------------------


async def test_sixteen_users_calling_the_same_tool_with_the_same_args_never_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arguments are identical for everyone. Only `user_id` separates the keys.

    This is the ambient-identity tool from DESIGN.md 8.1, run concurrently. If the key were
    built without the caller, this is the test that would return sixteen copies of one answer.
    """
    router = make_plugin(monkeypatch)
    users = [f"u{n}" for n in range(16)]

    await asyncio.gather(*(call_tool(router, u, {"rate": n}) for n, u in enumerate(users)))
    served = await asyncio.gather(*(call_tool(router, u, {"rate": -1}) for u in users))

    assert [s["rate"] for s in served] == list(range(16))
    assert all(s[CACHE_HIT_KEY]["cache_hit"] for s in served)


async def test_a_purge_racing_a_serve_leaves_no_partial_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write tool purges its user mid-flight. Every reader sees an entry or nothing, never half.

    `place_order` is not on the allowlist, so after it runs the whole user's cache is dropped
    (read-your-own-writes, DESIGN.md 8.4). Interleaving that with eight concurrent reads is the
    closest this cache comes to a torn read.
    """
    router = make_plugin(monkeypatch)
    await call_tool(router, "u1", {"rate": 1.27, "nested": {"fees": [1, 2]}})

    async def write() -> None:
        await call_tool(router, "u1", {"ok": True}, tool=OTHER, args={"qty": 1})

    reads = [call_tool(router, "u1", {"rate": 0.0}) for _ in range(8)]
    results = await asyncio.gather(*reads[:4], write(), *reads[4:])

    for got in results:
        if got is None or "rate" not in got:
            continue
        if CACHE_HIT_KEY in got:
            assert got == {
                "rate": 1.27,
                "nested": {"fees": [1, 2]},
                CACHE_HIT_KEY: got[CACHE_HIT_KEY],
            }
        else:
            assert got["rate"] == 0.0, "a miss returns the tool's own result, untouched"
    assert router.cache.stats["purges"] >= 1


async def test_blocking_a_tool_mid_flight_stops_every_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared-cacheable tool caught writing to `actions` is blocked while readers are live.

    From the moment the block lands, no reader may be served, including ones that were already
    in flight when it happened. `get` re-checks the blocklist on every read for this reason.
    """
    router = make_plugin(monkeypatch)
    await call_tool(router, "u1", {"rate": 1.27})

    order: list[str] = []

    async def read(n: int) -> None:
        await asyncio.sleep(0)  # queue behind whatever gather scheduled ahead of this
        # The serve is recorded at the instant it happens, not after the tool would have run,
        # or the ordering being tested would be lost to the await inside `call_tool`.
        served = await router.before_tool_callback(
            tool=TOOL, tool_args=ARGS, tool_context=FakeContext(user_id="u1")
        )
        order.append("served" if served is not None else "ran")

    async def block() -> None:
        await asyncio.sleep(0)
        router.cache.block(TOOL.name)
        order.append("blocked")

    # asyncio resumes the ready queue in order, so the reads scheduled after `block` are the
    # ones that have to notice it.
    await asyncio.gather(*(read(n) for n in range(3)), block(), *(read(n) for n in range(3, 8)))

    assert router.cache.is_blocked(TOOL.name)
    assert router.cache.stats["blocked"] == 1
    assert "blocked" in order
    before, after = order[: order.index("blocked")], order[order.index("blocked") + 1 :]
    assert "served" in before, "nothing was served before the block, so this proves nothing"
    assert set(after) == {"ran"}, "a reader was served after the tool was blocked"
    assert await call_tool(router, "u1", {"rate": 0.0}) == {"rate": 0.0}, "still serving"


# --- The statistics have to add up ------------------------------------------------


async def test_the_ledger_balances_after_a_mixed_concurrent_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hits plus misses equals reads, on every path, with sixteen invocations interleaved.

    A counter incremented on one branch and forgotten on another is invisible until someone
    trusts the hit rate. This is the arithmetic that catches it.
    """
    router = make_plugin(monkeypatch)
    users = ["u1", "u2", "u3", "u4"]

    reads = 0
    for _round in range(4):
        await asyncio.gather(*(call_tool(router, u, {"rate": 1.0}) for u in users))
        reads += len(users)

    stats = router.cache.stats
    assert stats["hits"] + stats["misses"] == reads
    assert stats["misses"] == len(users), "one cold miss each, then hits"
    assert stats["hits"] == reads - len(users)
    assert stats["stores"] == len(users), "a hit must never re-store"


# --- The routing half -------------------------------------------------------------


async def test_concurrent_first_turns_of_one_session_agree_on_the_kept_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four turns racing the first decision must not produce four different tool menus.

    There is no lock around the decision, and `before_model_callback` has real await points in
    it: the artifact load runs on a thread and the embedding is a network call. So turns do
    overlap here. Whichever one finishes first writes the decision, and the rest find it and
    reuse it rather than deciding again.

    The invariant is that every turn ends up with the same menu, because a divergence would be a
    mid-session change of the declared set, which is the one thing session stability exists to
    prevent. No turn reports a cache break either: they all agree, so there was nothing to break.
    """
    from test_plugin import declared_names, make_request, records

    router = make_plugin(monkeypatch)
    contexts = [FakeContext() for _ in range(4)]
    requests = [make_request() for _ in contexts]

    await asyncio.gather(
        *(
            router.before_model_callback(callback_context=ctx, llm_request=req)
            for ctx, req in zip(contexts, requests, strict=True)
        )
    )

    menus = {tuple(declared_names(req)) for req in requests}
    assert menus == {("tool_00",)}

    sources = [records(ctx)[0]["decision_source"] for ctx in contexts]
    assert set(sources) <= {"fresh", "reused"}
    assert "fresh" in sources, "nobody actually decided, so this proves nothing"
    assert not any(records(ctx)[0]["cache_break"] for ctx in contexts)

    # And the session has settled: a fifth turn reuses rather than deciding again.
    fifth_ctx, fifth_req = FakeContext(), make_request()
    await router.before_model_callback(callback_context=fifth_ctx, llm_request=fifth_req)
    assert records(fifth_ctx)[0]["decision_source"] == "reused"
    assert declared_names(fifth_req) == ["tool_00"]


async def test_a_burst_of_first_turns_loads_the_artifact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-flight: concurrent cold starts share one load instead of racing.

    `ModelStore.load` was measured at 2.2 s on the reference artifact, and every first turn
    that arrived while it ran used to start its own. The duplicates are pure waste, and there
    are more of them the busier the process is, which is exactly backwards.

    The store below actually blocks, in a worker thread, so the race is the real one rather
    than an ordering trick: without the lock all eight callers pass the `not in self._routers`
    check before the first load returns.
    """
    import threading

    from test_plugin import FakeContext, make_model, make_plugin, make_request

    class SlowStore:
        def __init__(self) -> None:
            self.calls = 0
            self.model = make_model()
            self._lock = threading.Lock()

        def load(self, agent_name: str) -> Any:
            with self._lock:
                self.calls += 1
            threading.Event().wait(0.05)  # the download and the booster parse
            return self.model

    router = make_plugin(monkeypatch)
    store = SlowStore()
    router._store = store  # type: ignore[assignment]
    router._routers.clear()

    contexts = [FakeContext() for _ in range(8)]
    requests = [make_request() for _ in contexts]
    await asyncio.gather(
        *(
            router.before_model_callback(callback_context=ctx, llm_request=req)
            for ctx, req in zip(contexts, requests, strict=True)
        )
    )

    assert store.calls == 1, f"{store.calls} concurrent loads of the same artifact"
    # The lock is kept rather than dropped once the agent resolves, because a load can now be
    # retried after `model_retry_after_s` and the retry needs the same single-flight the first
    # attempt got. One `asyncio.Lock` per agent for the life of the process is bounded by the
    # agent count, which is the same bound `_routers` itself lives under.
    assert list(router._loading) == ["root"]
