"""Regressions for the four cache-safety defects found in the build audit.

Each test here exists because a specific concrete failure was demonstrated against an earlier
version of this package, not because the behaviour seemed worth asserting in the abstract. The
scenario is written out in each test so the next person to change the code knows what breaks.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event_actions import EventActions
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from test_plugin import (
    FakeAgent,
    FakeContext,
    FakeInvocationContext,
    make_plugin,
    rate_tool,
    store_one,
)

from adk_smart_router.cache_hooks import HIT_MARKER_ATTR
from adk_smart_router.config import CACHE_HIT_KEY, RouterConfig
from adk_smart_router.startup_checks import StartupChecks
from adk_smart_router.tool_cache import actions_are_default

# --- The blocker: an agent address that outlived the agent ---------------------


def test_a_dead_agent_releases_its_address_from_the_scan_record() -> None:
    """An app that rebuilds its agent tree per request must not inherit an old scan result.

    The scan record used to be a `set[int]` of `id()` values, which hold no reference. Request
    one scans a non-gating `assistant` and the tree is freed. Request two builds
    `medical_records` with an authorization `before_tool_callback`; CPython hands it the
    freed address, the scan short-circuits, and the DESIGN.md 8.3 control never runs on it.
    """
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    agent = FakeAgent("assistant")
    address = id(agent)
    checks.run(FakeContext(invocation_context=FakeInvocationContext(agent, [])))  # type: ignore[arg-type]

    assert checks._seen(agent)
    del agent
    gc.collect()
    # The entry is gone, so no later object can inherit the "already scanned" verdict.
    assert address not in checks._scanned


def test_a_gating_agent_is_caught_after_an_earlier_agent_was_scanned() -> None:
    """The functional half of the same defect: the second tree is still scanned."""
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    first = FakeAgent("assistant")
    checks.run(FakeContext(invocation_context=FakeInvocationContext(first, [])))  # type: ignore[arg-type]
    del first
    gc.collect()

    gating = FakeAgent("medical_records", before_tool_callback=lambda **_: None)
    checks.run(FakeContext(invocation_context=FakeInvocationContext(gating, [])))  # type: ignore[arg-type]
    assert checks.gates_own_tools("medical_records")


# --- The reserved envelope key -------------------------------------------------


async def test_a_tool_returning_the_reserved_key_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that echoes third-party JSON must not be able to write the router's own label.

    Every training and TTL query filters on `_smart_router.cache_hit`, so a payload carrying
    that key could exclude its own real executions from the corpus, or forge a hit label into
    it. Neither is detectable from the warehouse, because the warehouse is what was corrupted.
    """
    router = make_plugin(monkeypatch)
    await store_one(router, FakeContext(), result={CACHE_HIT_KEY: "n/a", "rate": 1.1})
    assert router.cache.stats["stores"] == 0


async def test_the_envelope_wins_over_a_payload_key_of_the_same_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt and braces: even a payload written directly into the cache cannot shadow the label."""
    router = make_plugin(monkeypatch)
    ctx = FakeContext()
    router.cache.put(
        router._hooks._key_for(rate_tool(), {"pair": "usd_eur"}, ctx),  # type: ignore[arg-type]
        {CACHE_HIT_KEY: {"cache_hit": False}, "rate": 1.1},
        user_scope="support|u1",
        tool_name="get_rate",
        ttl_s=300,
        ttl_source="declared",
    )
    served = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=FakeContext(),  # type: ignore[arg-type]
    )
    assert served is not None
    assert served[CACHE_HIT_KEY]["cache_hit"] is True


# --- Side channels on EventActions ---------------------------------------------


def _truthy_for(name: str) -> Any:
    """A plausible non-empty value for one EventActions field."""
    if name in ("state_delta", "artifact_delta", "requested_auth_configs", "agent_state"):
        return {"k": "v"}
    if name == "requested_tool_confirmations":
        return {"k": 1}
    if name == "render_ui_widgets":
        return ["widget"]
    if name in ("transfer_to_agent", "rewind_before_invocation_id", "route"):
        return "other"
    if name in ("compaction", "set_model_response"):
        return "written"
    return True  # escalate, skip_summarization, end_of_agent


# The synthetic field values below are the right shape for the check and the wrong shape for
# pydantic's serializer, which is loud about it and irrelevant here.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_actions_are_default_notices_every_event_actions_field() -> None:
    """A hand-written field list fails open on whatever the next ADK release adds.

    The original list covered 7 of ADK 2.8.0's 14 fields, so a tool could set
    `render_ui_widgets`, `route`, `end_of_agent`, `agent_state`,
    `rewind_before_invocation_id` or `set_model_response`, be declared cacheable, and have that
    side effect silently dropped on every hit. This test fails on the next ADK upgrade that
    adds a field the check cannot see, which is the point of it.
    """
    assert actions_are_default(EventActions())
    for name in EventActions.model_fields:
        actions = EventActions()
        setattr(actions, name, _truthy_for(name))
        assert not actions_are_default(actions), f"a write to {name} was not noticed"


# --- The hit marker ------------------------------------------------------------


async def test_a_cache_hit_writes_no_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker lives on the call, not in the session, and this is the visible half of why.

    `State.__setitem__` writes to `actions.state_delta`, and ADK strips `temp:` keys only
    inside `append_event`, which runs after every plugin's `on_event_callback`. Anything the
    router put in state would therefore be visible to a co-registered logging plugin. The
    router puts nothing there, on this path or any other.
    """
    router = make_plugin(monkeypatch)
    await store_one(router, FakeContext())

    ctx = FakeContext()
    served = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=ctx,  # type: ignore[arg-type]
    )
    assert served is not None
    assert getattr(ctx, HIT_MARKER_ATTR) is True
    assert ctx.state == {}


async def test_a_writer_still_purges_when_another_tool_in_the_turn_hit_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker used to live in session state, which every call in a turn shares.

    ADK runs a turn's function calls concurrently (`asyncio.gather` in
    `flows/llm_flows/functions.py`) and gives each its own ToolContext, but `State.__setitem__`
    writes through to `invocation_context.session.state`, one dict for the whole invocation.
    So a turn that hit the cache on `get_rate` and called the writer `place_order` had the
    writer read the reader's marker, return before `purge_user`, and leave the stale rate
    behind. Read-your-own-writes failed silently, which is the worst way for it to fail.

    The shared `state` dict below is the whole point of the test: it is what ADK does.
    """
    router = make_plugin(monkeypatch)
    await store_one(router, FakeContext())

    reader, writer = FakeContext(), FakeContext()
    writer.state = reader.state  # one session, one state dict, two concurrent calls

    served = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=reader,  # type: ignore[arg-type]
    )
    assert served is not None

    await router.after_tool_callback(
        tool=BaseTool(name="place_order", description=""),
        tool_args={"sku": "a1"},
        tool_context=writer,  # type: ignore[arg-type]
        result={"ok": True},
    )

    again = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=FakeContext(),  # type: ignore[arg-type]
    )
    assert again is None, "the writer's purge was skipped and a stale entry survived it"


async def test_adk_shares_session_state_across_a_turns_tool_contexts_and_not_attributes() -> None:
    """Both halves of why the marker is an attribute, checked against ADK rather than assumed.

    ADK builds one ToolContext per function call, so an attribute on it is scoped to the call.
    Session state is not: two contexts from one invocation write through to the same dict, and
    that is what let one call's marker be read by another.

    The last assertion is ADK's vote on the mechanism. A release that gives `Context`
    `__slots__` or makes it a pydantic model would turn every cache hit into an AttributeError
    swallowed by the fail-open wrapper, which reads as the cache quietly never storing.
    """
    service = InMemorySessionService()
    session = await service.create_session(app_name="support", user_id="u1")
    invocation = InvocationContext(
        session_service=service, invocation_id="i1", agent=LlmAgent(name="root"), session=session
    )
    reader, writer = ToolContext(invocation), ToolContext(invocation)

    reader.state["temp:marker"] = True
    assert writer.state.get("temp:marker") is True, "a marker in state is not scoped to one call"

    setattr(reader, HIT_MARKER_ATTR, True)
    assert getattr(writer, HIT_MARKER_ATTR, False) is False
    assert getattr(reader, HIT_MARKER_ATTR) is True


# --- Read-your-own-writes, against the clock -------------------------------------


async def test_a_result_computed_before_a_concurrent_write_is_not_stored_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The purge is the invalidator, so a store must not be able to step over it.

    Same turn, two concurrent calls: `get_rate` misses the cache and starts running,
    `place_order` writes and purges, then `get_rate` returns with a rate it read before the
    write landed. Nothing is in the cache for the purge to drop, so the purge looks like a
    no-op and the store lands after it. The entry is then stale for a whole TTL, and every
    later turn is served the pre-write answer, which is the exact failure DESIGN.md 8.4 says
    a TTL cannot fix.

    `purge_epoch` dates the write rather than the entries, which is why it moves on a purge
    that dropped nothing.
    """
    router = make_plugin(monkeypatch)
    reader, writer = FakeContext(), FakeContext()

    served = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=reader,  # type: ignore[arg-type]
    )
    assert served is None, "nothing is stored yet, so this call has to run for real"

    await router.after_tool_callback(
        tool=BaseTool(name="place_order", description=""),
        tool_args={"sku": "a1"},
        tool_context=writer,  # type: ignore[arg-type]
        result={"ok": True},
    )
    await store_one(router, reader, {"rate": 1.1})

    assert router.cache.stats["stores"] == 0, "a pre-write result outlived the write"


async def test_one_tenants_writer_does_not_drop_another_tenants_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The purge scope joins two caller-controlled strings, so it has to be unambiguous.

    `"acme|eu" + "bob"` and `"acme" + "eu|bob"` are different principals that a plain `|` join
    renders identically. This cannot serve one of them the other's data, because identity is
    inside the cache key and not merely inside the scope. It can make one of them a writer for
    the other, which is a cross-tenant availability bug rather than a leak, and it is one
    length prefix away from not existing.
    """
    router = make_plugin(monkeypatch)
    first = FakeContext(app_name="acme|eu", user_id="bob")
    second = FakeContext(app_name="acme", user_id="eu|bob")

    await store_one(router, first)
    await router.after_tool_callback(
        tool=BaseTool(name="place_order", description=""),
        tool_args={"sku": "a1"},
        tool_context=second,  # type: ignore[arg-type]
        result={"ok": True},
    )

    served = await router.before_tool_callback(
        tool=rate_tool(),
        tool_args={"pair": "usd_eur"},
        tool_context=first,  # type: ignore[arg-type]
    )
    assert served is not None, "another principal's writer purged this one's cache"


# --- An anonymous caller --------------------------------------------------------


async def test_an_empty_user_id_disables_caching_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No identity means no cache, and it must not mean a stack trace on every tool call.

    `cache_key` refuses an empty principal by raising, which is right for a function whose job
    is identity scoping. Left to the plugin's fail-open wrapper it would log a full traceback
    and bump the failure counter once per tool call, which reads as an outage.
    """
    router = make_plugin(monkeypatch)
    anonymous = FakeContext(user_id="")

    await store_one(router, anonymous)
    served = await router.before_tool_callback(
        tool=BaseTool(name="get_rate", description=""),
        tool_args={"pair": "usd_eur"},
        tool_context=anonymous,  # type: ignore[arg-type]
    )

    assert served is None
    assert router.cache.stats["stores"] == 0
    assert router.failopen_counts == {}
