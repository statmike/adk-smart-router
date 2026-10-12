"""Tools that are not local functions: MCP servers and sub-agents (ROADMAP E10).

Every measurement in this project was taken on an agent whose tools are Python functions in
the same process. That is not what people deploy. Two other origins are common enough that a
router which quietly mishandles either is not shippable: an MCP server, which supplies tools
over a subprocess or a socket, and a sub-agent wrapped in `AgentTool`, which looks like one
tool and is a whole other model behind it.

The claim under test is that origin does not matter, and the reason to believe it is
structural: `apply_prune` filters `llm_request.config.tools[].function_declarations`, and by
the time the router sees a request ADK has already resolved every toolset into exactly that
list. Structural arguments about somebody else's framework are hypotheses. This file runs a
real MCP server as a subprocess, a real sub-agent under a real `AgentTool`, and a real ADK
`Runner`, and checks the things that could actually be wrong: that the prune reaches every
origin and spares every origin, that a pruned MCP tool still executes, that a parent's
decision cannot empty a sub-agent's tool list, and that a server whose tool names the model
has never seen is passed through rather than silently dropped.

What is stubbed is the same pair `test_end_to_end.py` stubs and nothing else: the model, and
the router's own backends. The MCP server is real, the protocol traffic is real, and the tool
list the model is shown is whatever ADK actually built.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
from test_end_to_end import ScriptedLlm, declared, in_memory_runner, turn
from test_plugin import FakeStore, make_model, make_plugin

from adk_smart_router.predictor import Prediction
from adk_smart_router.reachability import survey

SERVER = Path(__file__).parent / "mcp_stub_server.py"

# The three the MCP server publishes, and the one the sub-agent is wrapped as.
MCP_TOOLS = ("get_air_quality", "get_alerts", "get_forecast")
SUB_AGENT = "historian"

# Six local functions, so the surface is a realistic mix rather than a demonstration. Their
# bodies do not matter; their names and signatures are what reach the model.
LOCAL_TOOLS = (
    "get_order_status",
    "get_invoice",
    "list_orders",
    "cancel_order",
    "get_price",
    "escalate_to_human",
)

# The sub-agent's own tools, which the parent has never heard of and the router has no class
# for. They exist so the sub-agent's model call is a real one with something to lose.
SUB_TOOLS = ("search_archive", "fetch_receipt")


def local_tool(name: str) -> Any:
    """A plain Python function ADK will wrap in `FunctionTool`, named at runtime."""

    def tool(order_id: str) -> str:
        return f"{name}: {order_id}"

    tool.__name__ = name
    tool.__doc__ = f"The {name.replace('_', ' ')} tool."
    return tool


def mcp_toolset(prefix: str | None = None) -> McpToolset:
    """The stub server, spoken to over stdio exactly as a deployed one would be."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=sys.executable, args=[str(SERVER)])
        ),
        tool_name_prefix=prefix,
    )


def build(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    kept: tuple[str, ...],
    prefix: str | None = None,
    via: str = "tool",
    **cfg_overrides: Any,
) -> Any:
    """An agent with all three origins, wired to a scripted model and the real plugin.

    `kept` is what the predictor will name. The probabilities descend and sum to 0.92, so the
    pruner's `keep_probability_mass` of 0.90 is reached on the last of them and not before:
    a flat or front-loaded vector would collapse to one tool, which is correct behaviour and
    a useless fixture for a test about which tools survive.

    `via` picks which of ADK's two delegation styles the sub-agent arrives through, because
    they are different tools with the same name in conversation. `"tool"` wraps it in
    `AgentTool`, which declares it like any other function and runs it in a session of its
    own. `"sub_agent"` hands it to `sub_agents`, and then ADK declares no `historian` at all:
    it declares `transfer_to_agent`, and the parent hands the session over rather than calling
    into a new one. The router has to be right about both, and only one of them is a name it
    can be trained on.

    `min_tools_to_route` drops to 5 because the shipped 20 would decline this agent outright
    and every assertion below would pass vacuously. Ten tools is small for a real deployment
    and large for a test that pays a subprocess and a protocol handshake per case. The
    threshold is about whether pruning earns its keep, which is measured elsewhere and has
    nothing to do with where a tool came from.
    """
    tilt = [len(kept) - n + 0.5 for n in range(len(kept))]
    preds = tuple(
        Prediction(name, 0.92 * weight / sum(tilt)) for name, weight in zip(kept, tilt, strict=True)
    )
    classes = (*LOCAL_TOOLS, *MCP_TOOLS, SUB_AGENT)

    router = make_plugin(monkeypatch, preds=preds, min_tools_to_route=5, **cfg_overrides)
    router._store = FakeStore(make_model(classes=classes))  # noqa: SLF001
    router._routers.clear()  # noqa: SLF001

    llm = ScriptedLlm(model="gemini-2.5-flash")
    llm.script = list(script)
    llm.seen = []

    sub_llm = ScriptedLlm(model="gemini-2.5-flash")
    sub_llm.script = ["Nothing in the archive."]
    sub_llm.seen = []

    historian = LlmAgent(
        name=SUB_AGENT,
        model=sub_llm,
        instruction="Answer questions about past orders.",
        description="Looks up historical orders older than one year.",
        tools=[local_tool(n) for n in SUB_TOOLS],
    )
    tools: list[Any] = [*[local_tool(n) for n in LOCAL_TOOLS], mcp_toolset(prefix)]
    if via == "tool":
        tools.insert(-1, AgentTool(agent=historian))
    agent = LlmAgent(
        name="support",
        model=llm,
        instruction="You are a support agent.",
        tools=tools,
        sub_agents=[historian] if via == "sub_agent" else [],
    )
    return router, llm, sub_llm, App(name="support", root_agent=agent, plugins=[router])


# --- What ADK hands the router ----------------------------------------------------


async def test_all_three_origins_reach_the_model_as_one_flat_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The premise. If this fails, nothing below it means anything.

    Routing is disabled here so the assertion is about ADK's resolution and not about the
    prune. Ten tools go in from three different places and the model is shown ten function
    declarations with no marking of where any of them came from, which is precisely why the
    router can treat them alike.
    """
    _, llm, _, app = build(monkeypatch, ["All set."], kept=LOCAL_TOOLS[:2], mode="off")
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "hello")

    assert declared(llm.seen[0]) == sorted([*LOCAL_TOOLS, *MCP_TOOLS, SUB_AGENT])


async def test_an_mcp_tool_and_a_sub_agent_can_both_be_pruned_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prune does not care where a declaration came from.

    The prediction names two local tools and one MCP tool. Everything else goes, including the
    sub-agent and the other two MCP tools, which are the declarations most likely to survive a
    filter that only understood `FunctionTool`.
    """
    kept = ("get_order_status", "get_forecast", "get_invoice")
    _, llm, _, app = build(monkeypatch, ["All set."], kept=kept)
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order and what is the weather")

    assert declared(llm.seen[0]) == sorted(kept)


async def test_a_sub_agent_survives_when_the_prediction_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction, because a filter that dropped every `AgentTool` unconditionally
    would pass the test above and fail here."""
    kept = (SUB_AGENT, "get_order_status")
    _, llm, _, app = build(monkeypatch, ["All set."], kept=kept)
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "what did I order in 2019")

    assert declared(llm.seen[0]) == sorted(kept)


# --- What happens when the router is wrong ----------------------------------------


async def test_a_pruned_mcp_tool_still_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The §7.4 invariant, on the origin where it is least obvious that it holds.

    `apply_prune` edits `config.tools` and leaves `tools_dict` alone, so a model that calls a
    pruned tool anyway still reaches it. For a local function that is easy to believe. For an
    MCP tool it means ADK must still have a live session to the server for a tool it was told
    not to mention, and the call has to go over the wire and come back. It does: the response
    below is the stub server's own string, not an error.
    """
    _, llm, _, app = build(
        monkeypatch,
        [("get_forecast", {"city": "Leeds"}), "It will be sunny."],
        kept=("get_order_status", "get_invoice"),
    )
    runner = await in_memory_runner(app)

    events = await turn(runner, "s-1", "where is my order")

    assert "get_forecast" not in declared(llm.seen[0]), "the tool under test was not pruned"
    responses = [
        part.function_response
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_response
    ]
    assert [r.name for r in responses] == ["get_forecast"]
    assert "sunny" in str(responses[0].response)


async def test_never_prune_holds_on_a_tool_that_came_over_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator's safety list is written by name, and a name is all the router has. This
    checks that the guarantee does not quietly depend on the tool being local."""
    _, llm, _, app = build(
        monkeypatch, ["All set."], kept=("get_order_status",), never_prune=frozenset({"get_alerts"})
    )
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order")

    assert declared(llm.seen[0]) == ["get_alerts", "get_order_status"]


async def test_a_renamed_mcp_server_is_passed_through_rather_than_pruned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`McpToolset` can prefix its tool names, and then the live names stop matching the model.

    An operator who sets `tool_name_prefix="weather"` gets `weather_get_forecast` on the wire
    while the artifact was trained on `get_forecast`. Nothing errors and nothing warns. This is
    the ordinary way an artifact goes stale, too: a server is added, a tool is renamed, and the
    published model has classes for neither.

    The rule that saves it is in `decide`. Any live tool the model has no class for is unioned
    into the kept set, because silence from a classifier that was never taught a name is not
    evidence against that tool. So all three renamed tools ship, the prune still removes what
    the model does know about, and the agent keeps working at a smaller saving until someone
    republishes. The failure mode this rules out is the loud one: predicting three names that
    match nothing live and declaring a list with those three names missing from it.
    """
    kept = ("get_order_status", "get_invoice")
    _, llm, _, app = build(monkeypatch, ["All set."], kept=kept, prefix="weather")
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order and what is the weather")

    renamed = [f"weather_{name}" for name in MCP_TOOLS]
    assert declared(llm.seen[0]) == sorted([*kept, *renamed])


async def test_prunable_by_default_reaches_a_renamed_mcp_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of that rule, which is the operator's way out of it.

    Passing unclassed tools through is safe and it is not free: every renamed tool rides along
    in every session until the next publish. `prunable_by_default` is how an operator says they
    know something the model does not, and it is matched on the live name, so it works on a
    prefix the artifact has never seen.
    """
    kept = ("get_order_status", "get_invoice")
    _, llm, _, app = build(
        monkeypatch,
        ["All set."],
        kept=kept,
        prefix="weather",
        prunable_by_default=frozenset(f"weather_{name}" for name in MCP_TOOLS),
    )
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order")

    assert declared(llm.seen[0]) == sorted(kept)


# --- What the install-time tools make of them -------------------------------------


async def test_reachability_reads_an_mcp_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """`qualify` tells an adopter which tools a turn-one router can ever predict, by reading
    required arguments off each declaration. MCP writes its schema into
    `parameters_json_schema` where a local function writes a typed `Schema`, so this is the
    branch of `reachability._schema` that only a real MCP tool exercises.

    All three MCP tools take one required string that a first message can plausibly supply, so
    all three should come back gated on something rather than unreadable.
    """
    _, llm, _, app = build(monkeypatch, ["All set."], kept=LOCAL_TOOLS[:1], mode="off")
    runner = await in_memory_runner(app)
    await turn(runner, "s-1", "hello")

    declarations = [
        declaration
        for tool in llm.seen[0].config.tools or []
        for declaration in tool.function_declarations or []
    ]
    found = survey(declarations)

    by_name = {tool.name: tool for tool in found.tools}
    for name in MCP_TOOLS:
        assert by_name[name].gates, f"{name} came back with no readable required argument"


# --- The sub-agent's own model call -----------------------------------------------


async def test_the_parents_decision_does_not_prune_the_sub_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this whole file was written to look for.

    `AgentTool` runs a second agent with a second tool list through the same plugin. If the
    router applied the parent's kept set to that request, every one of the sub-agent's tools
    would be filtered out, `config.tools` would come back empty, and the sub-agent would be
    silently lobotomised on every call. Nothing would raise. The parent would get a worse
    answer and the telemetry would show a healthy prune.

    It does not happen, and two independent things stop it. `rollout.scope_key` puts the agent
    name in the key, because a plugin installs on the Runner and sees every agent in the tree
    (DESIGN.md 7.6, D9). And `AgentTool` mints its own session, which is the same fact the tool
    cache leans on. Either one alone would be enough; the decision store below should hold two
    entries after one turn, one per agent.

    The sub-agent's own two tools are under `min_tools_to_route`, so the router looks at its
    request, declines it on its own merits, and leaves the list alone. That is the outcome, and
    it is a different outcome from the parent's set being applied and filtering everything out.
    """
    kept = ("get_order_status", SUB_AGENT)
    router, llm, sub_llm, app = build(
        monkeypatch,
        [(SUB_AGENT, {"request": "orders from 2019"}), "Nothing turned up."],
        kept=kept,
    )
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "what did I order in 2019")

    assert declared(llm.seen[0]) == sorted(kept), "the parent was not pruned"
    assert sub_llm.seen, "the sub-agent never ran, so this test proved nothing"
    assert declared(sub_llm.seen[0]) == sorted(SUB_TOOLS)
    assert len(router._decisions) == 2, "the two agents shared one routing decision"  # noqa: SLF001


async def test_a_transfer_survives_a_prune_that_kept_neither_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK's other delegation style, where the sub-agent is not a declaration at all.

    An agent with `sub_agents` gets one extra tool it did not write: `transfer_to_agent`, the
    control plane the whole tree moves on. It is never a training label, because every SQL
    query in `sql/` filters it out, so the classifier can never name it and a prune driven by
    predictions alone would delete it on the first session. Then the parent could not hand off,
    the transfer would stop appearing in traces, and the next model would be even surer. That
    is rule 1 in DESIGN.md 7.2, and `FRAMEWORK_TOOLS` is the constant that enforces it.

    `test_pruner.py` proves the rule on the function. This proves ADK actually puts the tool in
    the list where the rule can reach it, which is the part no unit test can see.
    """
    kept = ("get_order_status", "get_invoice")
    _, llm, _, app = build(monkeypatch, ["All set."], kept=kept, via="sub_agent")
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "where is my order")

    assert declared(llm.seen[0]) == sorted([*kept, "transfer_to_agent"])


async def test_an_agent_transferred_to_mid_session_is_routed_on_its_own_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same scope question as above, with one of the two defences removed.

    `AgentTool` mints a fresh session, so a router keyed on the session alone would still have
    got the case above right. A transfer does not: control moves to `historian` inside the same
    session, with the same app and the same user, and the only thing left in the key that
    differs is the agent name (DESIGN.md 7.6, D9). If that were missing, the parent's kept set
    would be found and reused for an agent whose tools have nothing in common with it, and
    `historian` would be handed an empty list.

    It is not missing. `historian` is looked at on its own, declined for having two tools, and
    left alone, and the store holds a separate decision for it. Its list is its own two tools
    plus a `transfer_to_agent` of its own, which ADK gives it so it can hand the session back.
    """
    kept = ("get_order_status", "get_invoice")
    router, llm, sub_llm, app = build(
        monkeypatch,
        [("transfer_to_agent", {"agent_name": SUB_AGENT})],
        kept=kept,
        via="sub_agent",
    )
    runner = await in_memory_runner(app)

    await turn(runner, "s-1", "what did I order in 2019")

    assert sub_llm.seen, "the transfer never happened, so this test proved nothing"
    assert declared(sub_llm.seen[0]) == sorted([*SUB_TOOLS, "transfer_to_agent"])
    assert len(router._decisions) == 2, "the two agents shared one routing decision"  # noqa: SLF001
