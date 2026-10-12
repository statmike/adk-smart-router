"""The fake-backends tier from DESIGN.md 13: the whole loop, through a real ADK Runner.

Every other test in this suite calls the plugin's callbacks directly. That proves the callbacks
are right and proves nothing about whether ADK calls them, in what order, or with what. This
file runs the reference agent's real 25 tools under a real `Runner`, with a real
`PluginManager`, and stubs exactly two things: the model, and the router's own backends.

The model is a `ScriptedLlm`. It answers from a list instead of a network, and it keeps every
`LlmRequest` it was handed, which is how these tests see what the router did to the request. The
router's model store, embedder and predictor are the same fakes the rest of the suite uses, so
no artifact is downloaded and no vector is embedded.

Nothing here touches the network, GCS, BigQuery or credentials.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import reference_agent.agent as reference
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types
from reference_agent.tools import ALL_TOOLS, reset_fixtures
from test_plugin import make_plugin

from adk_smart_router.config import CACHE_HIT_KEY
from adk_smart_router.predictor import Prediction
from adk_smart_router.provenance import PROVENANCE_KEY

APP = "support"
USER = "u-1"

TOOL_NAMES = tuple(fn.__name__ for fn in ALL_TOOLS)

# Descending, and reaching `keep_probability_mass` (0.90) only on the fifth, so the pruner keeps
# all five rather than stopping early. A flat vector would collapse to one tool, which is
# correct behaviour and a poor fixture: it would leave the interesting assertions untested.
PREDICTIONS = (
    ("get_order_status", 0.60),
    ("track_shipment", 0.12),
    ("list_orders", 0.09),
    ("get_return_policy", 0.06),
    ("get_invoice", 0.05),
)
KEPT = tuple(name for name, _ in PREDICTIONS)


class ScriptedLlm(BaseLlm):
    """A model that answers from a list and remembers every request it was given.

    `seen` is the point of the class. Each entry is the tool menu that actually reached the
    model on one turn, which is the only honest way to check that pruning happened where it was
    supposed to and nowhere else.
    """

    script: list[Any] = []
    seen: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.seen.append(llm_request)
        step = self.script.pop(0) if self.script else "All set."
        if isinstance(step, tuple):
            name, args = step
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text=step)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def declared(request: LlmRequest) -> list[str]:
    return sorted(
        declaration.name
        for tool in request.config.tools or []
        for declaration in tool.function_declarations or []
    )


def prompts(request: LlmRequest) -> list[str]:
    return [
        part.text
        for content in request.contents
        for part in content.parts or []
        if part.text is not None
    ]


def build(monkeypatch: pytest.MonkeyPatch, script: list[Any], **overrides: Any) -> Any:
    """The reference agent, wired to a scripted model, with the router installed on the App."""
    reset_fixtures()
    from test_plugin import FakeStore, make_model

    preds = tuple(Prediction(name, probability) for name, probability in PREDICTIONS)
    router = make_plugin(monkeypatch, preds=preds, **overrides)
    router._store = FakeStore(make_model(classes=TOOL_NAMES))  # noqa: SLF001
    router._routers.clear()  # noqa: SLF001

    llm = ScriptedLlm(model="gemini-2.5-flash")
    llm.script = list(script)
    llm.seen = []

    agent = LlmAgent(
        name=reference.AGENT_NAME,
        model=llm,
        instruction=reference.INSTRUCTION,
        tools=ALL_TOOLS,
    )
    return router, llm, App(name=APP, root_agent=agent, plugins=[router])


async def turn(runner: Runner, session_id: str, text: str) -> list[Any]:
    events = []
    async for event in runner.run_async(
        user_id=USER,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        events.append(event)
    return events


async def in_memory_runner(app: App) -> Runner:
    service = InMemorySessionService()
    await service.create_session(app_name=APP, user_id=USER, session_id="s-1")
    return Runner(app=app, session_service=service)


# --- The prune, where ADK can see it ----------------------------------------------


async def test_the_model_sees_a_pruned_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hundred tools go in and five reach the model, with the prompt left alone."""
    router, llm, app = build(monkeypatch, ["Your order is out for delivery."])
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order")

    assert len(llm.seen) == 1
    assert declared(llm.seen[0]) == sorted(KEPT)
    assert len(TOOL_NAMES) == 100, "the reference agent lost tools; the prune proves less"
    assert prompts(llm.seen[0]) == ["where is my order"]


async def test_a_pruned_tool_still_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason `apply_prune` leaves `tools_dict` alone, proven through ADK's own dispatch.

    The model is scripted to call `get_price`, which the router pruned out of the declaration
    list. ADK looks the tool up in `tools_dict`, which the router never touched, so the call
    runs and returns a real answer. Had the router filtered `tools_dict` too, this would raise
    inside `_get_tool` and BQAA would log a TOOL_ERROR against `get_price`, making the router's
    own mistake look like a tool failure in the telemetry used to detect over-pruning
    (DESIGN.md 7.4).
    """
    router, llm, app = build(
        monkeypatch, [("get_price", {"sku": "SKU-LMP-03", "quantity": 1}), "It is £24.99."]
    )
    runner = await in_memory_runner(app)

    events = await turn(runner, "s-1", "how much is SKU-LMP-03")

    assert "get_price" not in declared(llm.seen[0]), "the tool under test was not pruned"
    responses = [
        part.function_response
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_response is not None
    ]
    assert [r.name for r in responses] == ["get_price"]
    assert "error" not in (responses[0].response or {}), responses[0].response


# --- Session stability, across real turns ------------------------------------------


async def test_the_second_turn_reuses_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """One embedding for the session, the same menu on both turns, and one shared prefix.

    This is the economic claim end to end. Turn two must not re-decide, because re-deciding is
    what breaks the model's prefix cache, and turn one must not have anything in it that turn
    two lacks, for the same reason (DESIGN.md 18.18).
    """
    router, llm, app = build(monkeypatch, ["First.", "Second."])
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order")
    await turn(runner, "s-1", "and my other one")

    assert len(llm.seen) == 2
    assert declared(llm.seen[0]) == declared(llm.seen[1]) == sorted(KEPT)
    assert router.embedder_stats["misses"] == 1, "the session embedded more than once"
    assert prompts(llm.seen[0]) == ["where is my order"]
    assert prompts(llm.seen[1])[0] == "where is my order"


async def test_two_sessions_decide_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decision is keyed by session, so a second conversation embeds for itself."""
    router, llm, app = build(monkeypatch, ["First.", "Second."])
    service = InMemorySessionService()
    for session_id in ("s-1", "s-2"):
        await service.create_session(app_name=APP, user_id=USER, session_id=session_id)
    runner = Runner(app=app, session_service=service)

    await turn(runner, "s-1", "where is my order")
    await turn(runner, "s-2", "where is my order")

    assert router.embedder_stats["misses"] == 2
    assert declared(llm.seen[0]) == declared(llm.seen[1])


def routing_records(events: list[Any]) -> list[dict[str, Any]]:
    """Every routing record carried on an event. Cache serves are the ones with a tool name."""
    return [
        record
        for event in events
        for record in (event.custom_metadata or {}).get(PROVENANCE_KEY, [])
        if "tool_name" not in record
    ]


async def test_provenance_lands_on_the_event_not_just_the_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record per turn, stamped with the artifact that shaped it, carried on the event.

    On the event specifically. `emit` can only reach the invocation's metadata bag, ADK copies
    that bag onto nothing, and BQAA reads `event.custom_metadata`, so a record that stops at the
    invocation is a record that never reaches `agent_events`. `on_event_callback` is the hop
    that closes that gap, and asserting against `event.custom_metadata` is the only way to see
    whether it ran.
    """
    router, llm, app = build(monkeypatch, ["First.", "Second."])
    runner = await in_memory_runner(app)

    first = await turn(runner, "s-1", "where is my order")
    second = await turn(runner, "s-1", "and my other one")

    one, two = routing_records(first), routing_records(second)
    assert [r["decision_source"] for r in one] == ["fresh"]
    assert [r["decision_source"] for r in two] == ["reused"]
    assert one[0]["decision_id"] == two[0]["decision_id"]
    assert one[0]["kept_tools"] == sorted(KEPT)
    assert one[0]["model_version"] == "20260830T000000Z"
    assert not one[0]["cache_break"]


async def test_a_tool_round_trips_records_all_land_on_the_one_event_bqaa_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both records ride the final response event, in order, exactly once each.

    A tool round trip means two model calls, so two routing records: the decision and then its
    reuse. ADK yields three events for it and persists all three, but BQAA writes a row only
    for the last, so a record dropped on the function-call event is a record deleted. This test
    used to assert the opposite — two records on two separate events — and that shape cost 903
    of 2,793 decisions in run ab-v2, `fresh` among them, leaving `reused` as the first record
    of the session that made the decision (DESIGN.md 19.23).

    Still exactly once each. `carry_onto` moves rather than copies, and a bag that were copied
    instead of drained would inflate every rate computed from `agent_events`.
    """
    router, llm, app = build(
        monkeypatch, [("get_order_status", {"order_id": "A1001"}), "Out for delivery."]
    )
    runner = await in_memory_runner(app)

    events = await turn(runner, "s-1", "where is my order")

    assert len(llm.seen) == 2, "no tool round trip, so there is nothing to hold records past"
    carrying = [e for e in events if (e.custom_metadata or {}).get(PROVENANCE_KEY)]
    assert len(carrying) == 1, "records landed somewhere other than the single logged event"
    assert carrying[0].is_final_response()
    assert not carrying[0].get_function_calls()

    records = routing_records(events)
    assert [r["decision_source"] for r in records] == ["fresh", "reused"]
    assert len({id(r) for r in records}) == 2


async def test_a_cache_serve_is_recorded_on_the_turn_that_served_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serve record shares the turn's decision id, and carries no argument value.

    DESIGN.md 11.3 forbids the args hash here because a hash of low-cardinality arguments is
    reversible. This is that rule checked against what actually reaches an event.

    Reaches a *logged* event, specifically. A serve is emitted from `before_tool_callback`, so
    the very next event is always the function response, which BQAA never turns into a row. No
    cache serve had ever reached `agent_events` before `carry_onto` learned to hold on
    (DESIGN.md 19.23).
    """
    call = ("get_return_policy", {"category": "electronics"})
    router, llm, app = build(
        monkeypatch,
        [call, "Thirty days.", call, "Still thirty days."],
        cacheable_tools={"get_return_policy": 3600},
    )

    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")
        await service.create_session(app_name=APP, user_id=USER, session_id="s-1")
        runner = Runner(app=app, session_service=service)

        first = await turn(runner, "s-1", "what is the returns policy")
        second = await turn(runner, "s-1", "remind me of the returns policy")

    carried = [
        (event, record)
        for event in second
        for record in (event.custom_metadata or {}).get(PROVENANCE_KEY, [])
        if "tool_name" in record
    ]
    serves = [record for _, record in carried]
    assert [s["tool_name"] for s in serves] == ["get_return_policy"]
    assert [e.get_function_responses() for e, _ in carried] == [[]], "landed on a dropped event"
    assert serves[0]["decision_id"] == routing_records(first)[0]["decision_id"]
    assert serves[0]["ttl_used"] == 3600
    assert not [key for key in serves[0] if "arg" in key or "hash" in key]


# --- The cache, through the same Runner ---------------------------------------------


async def test_the_second_identical_call_is_served_from_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared-cacheable tool runs once and is replayed the second time, with the envelope.

    `DatabaseSessionService` rather than `InMemorySessionService`, because the cache refuses to
    serve over the latter on purpose: `AgentTool` mints a fresh in-memory session per call, so
    session-scoped entries could never hit and would only accrete dead keys (DESIGN.md 7.6).
    """
    call = ("get_return_policy", {"category": "electronics"})
    router, llm, app = build(
        monkeypatch,
        [call, "Thirty days.", call, "Still thirty days."],
        cacheable_tools={"get_return_policy": 3600},
    )

    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")
        await service.create_session(app_name=APP, user_id=USER, session_id="s-1")
        runner = Runner(app=app, session_service=service)

        await turn(runner, "s-1", "what is the returns policy")
        events = await turn(runner, "s-1", "remind me of the returns policy")

    payloads = [
        part.function_response.response
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_response is not None
    ]
    assert len(payloads) == 1
    assert payloads[0][CACHE_HIT_KEY]["cache_hit"] is True
    assert router.cache.stats["hits"] == 1
    assert router.cache.stats["stores"] == 1, "a hit must not re-store"


async def test_a_write_tool_drops_the_callers_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-your-own-writes, end to end: an undeclared tool purges the user, the next read misses.

    `update_email` is not on the allowlist, so the router assumes it had side effects and drops
    everything cached for that principal. A TTL bounds staleness against the world and does
    nothing against the agent's own writes (DESIGN.md 8.4).
    """
    read = ("get_return_policy", {"category": "electronics"})
    write = ("update_email", {"new_email": "a@example.com"})
    router, llm, app = build(
        monkeypatch,
        [read, "Thirty days.", write, "Updated.", read, "Thirty days."],
        cacheable_tools={"get_return_policy": 3600},
    )

    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")
        await service.create_session(app_name=APP, user_id=USER, session_id="s-1")
        runner = Runner(app=app, session_service=service)

        await turn(runner, "s-1", "what is the returns policy")
        await turn(runner, "s-1", "change my email to a@example.com")
        events = await turn(runner, "s-1", "remind me of the returns policy")

    payloads = [
        part.function_response.response
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_response is not None
    ]
    assert len(payloads) == 1
    assert CACHE_HIT_KEY not in payloads[0], "the write did not invalidate the entry"
    assert router.cache.stats["purges"] >= 1
    assert router.cache.stats["hits"] == 0
