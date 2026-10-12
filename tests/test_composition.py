"""The composition tier from DESIGN.md 13: what a routed request looks like as a whole.

Every other test drives one operation. This file drives a whole request against ADK's own cache
machinery and asserts the three properties DESIGN.md 2.2 rests on. All three were argued for in
the design and none of them was covered anywhere until this file existed.

1. `system_instruction` comes out byte-identical. It sits at the very front of the cached
   prefix, so anything that writes to it breaks the cache on every turn it tries to help.
2. `contents` comes out untouched. The router edits the declared tool list and nothing else.
   This is the property that used to be violated: a steering directive was inserted on the
   deciding turn only, which left that turn's `contents` disagreeing with every later turn's
   from the third element on, and cost the session its own cached prefix for the rest of its
   life (DESIGN.md 18.18).
3. A session that reuses its decision produces a byte-identical cache fingerprint. This is the
   whole economic claim: the router prunes once, and turn two onward serve from cache.

The cache manager is real. Faking it would let the test agree with an assumption rather than
with ADK.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from google.adk.models.gemini_context_cache_manager import GeminiContextCacheManager
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from test_plugin import TOOLS, FakeContext, declared_names, make_plugin

from adk_smart_router.predictor import Prediction

SYSTEM_INSTRUCTION = (
    "You are a support assistant for a fictional retailer. Confirm the customer's "
    "identity before disclosing order, payment or shipment detail."
)


class _FakeApiClient:
    vertexai = True
    project = "test-project"
    location = "us-central1"
    _http_options = None


class _FakeGenAiClient:
    """Enough of a client for `_cache_scope`, which is all the fingerprint reads."""

    vertexai = True
    _api_client = _FakeApiClient()


def cache_manager() -> GeminiContextCacheManager:
    # ADK marks the manager experimental and warns on construction. Depending on it is the point
    # of this file, so the warning is noted here rather than printed on every test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return GeminiContextCacheManager(_FakeGenAiClient())  # type: ignore[arg-type]


def contents_to_cache(request: LlmRequest) -> int:
    return cache_manager()._find_count_of_contents_to_cache(request.contents)


def fingerprint(request: LlmRequest) -> str:
    manager = cache_manager()
    return manager._generate_cache_fingerprint(request, contents_to_cache(request))


def conversation(prompt: str) -> list[types.Content]:
    """Two settled turns and a fresh user message, which is the ordinary mid-session shape."""
    return [
        types.Content(role="user", parts=[types.Part(text="hello")]),
        types.Content(role="model", parts=[types.Part(text="Hi, how can I help?")]),
        types.Content(role="user", parts=[types.Part(text=prompt)]),
    ]


def make_request(prompt: str = "where is my invoice", tools: list[str] | None = None) -> LlmRequest:
    names = tools if tools is not None else TOOLS
    return LlmRequest(
        model="gemini-2.5-flash",
        contents=conversation(prompt),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[
                types.Tool(function_declarations=[types.FunctionDeclaration(name=n) for n in names])
            ],
        ),
        tools_dict={name: BaseTool(name=name, description="") for name in names},
    )


def snapshot(request: LlmRequest) -> list[str]:
    """`contents` as comparable text, so a change of any kind shows up as a diff."""
    return [content.model_dump_json(exclude_none=True) for content in request.contents]


# --- Property 1: the system instruction is never touched -----------------------


async def test_pruning_leaves_the_system_instruction_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`system_instruction` is the front of the cached prefix. Writing to it breaks every turn."""
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    request = make_request()
    before = request.config.system_instruction

    await router.before_model_callback(callback_context=FakeContext(), llm_request=request)  # type: ignore[arg-type]

    assert declared_names(request) == ["tool_00"], "the prune did not run"
    assert request.config.system_instruction == before
    assert request.config.system_instruction is before


# --- Property 2: the conversation is never touched -----------------------------


async def test_a_deciding_turn_leaves_the_contents_exactly_as_they_arrived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The turn the router decides on is the turn that used to gain an extra content.

    That content was inserted ahead of the live user message, so the deciding turn and every
    turn after it disagreed from that position on and no later turn could reuse the session's
    own prefix (DESIGN.md 18.18). Nothing is added now, on any path.
    """
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    request = make_request()
    before, cacheable = snapshot(request), contents_to_cache(request)

    await router.before_model_callback(callback_context=FakeContext(), llm_request=request)  # type: ignore[arg-type]

    assert declared_names(request) == ["tool_00"], "the prune did not run, so this proves nothing"
    assert snapshot(request) == before
    assert contents_to_cache(request) == cacheable == 2


async def test_a_tool_call_continuation_turn_is_left_alone_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trailing user batch is a function response here, which is the awkward shape."""
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    request = make_request()
    request.contents = [
        types.Content(role="user", parts=[types.Part(text="where is my invoice")]),
        types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name="tool_00", args={}))],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(name="tool_00", response={"ok": True})
                )
            ],
        ),
    ]
    before, cacheable = snapshot(request), contents_to_cache(request)

    await router.before_model_callback(callback_context=FakeContext(), llm_request=request)  # type: ignore[arg-type]

    assert snapshot(request) == before
    assert contents_to_cache(request) == cacheable == 2


async def test_the_deciding_turn_and_the_next_one_agree_on_their_shared_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression that DESIGN.md 18.18 is about, stated as an assertion.

    Turn two caches against turn one of the same session, so the two have to agree element by
    element for as far as turn one goes. They only do if the router adds nothing.
    """
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    ctx = FakeContext()

    first = make_request()
    await router.before_model_callback(callback_context=ctx, llm_request=first)  # type: ignore[arg-type]

    second = make_request()
    second.contents = [
        *first.contents,
        types.Content(role="model", parts=[types.Part(text="It shipped on Tuesday.")]),
        types.Content(role="user", parts=[types.Part(text="and what about my refund")]),
    ]
    await router.before_model_callback(callback_context=ctx, llm_request=second)  # type: ignore[arg-type]

    shared = len(first.contents)
    assert snapshot(second)[:shared] == snapshot(first)


# --- Property 3: a reused decision keeps the fingerprint -----------------------


async def test_a_reused_decision_reproduces_the_cache_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The economic claim in one assertion: turn two hashes to what turn one hashed.

    The fingerprint covers the model, the system instruction, the tools and the cached contents.
    Holding it steady across turns is what lets ADK create a cache at all: it needs two
    consecutive matching fingerprints, so a set that flips membership never caches once
    (DESIGN.md 2.2, 18.1).
    """
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    ctx = FakeContext()

    first = make_request()
    await router.before_model_callback(callback_context=ctx, llm_request=first)  # type: ignore[arg-type]

    second = make_request(prompt="and what about my refund")
    await router.before_model_callback(callback_context=ctx, llm_request=second)  # type: ignore[arg-type]

    assert declared_names(first) == declared_names(second) == ["tool_00"]
    assert fingerprint(first) == fingerprint(second)


async def test_an_unrouted_session_and_a_routed_one_do_not_share_a_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control.

    If pruning did not move the fingerprint, none of the assertions above mean anything.
    """
    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    pruned = make_request()
    await router.before_model_callback(callback_context=FakeContext(), llm_request=pruned)  # type: ignore[arg-type]

    untouched = make_request()
    assert fingerprint(pruned) != fingerprint(untouched)


async def test_ranking_is_free_and_only_membership_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK sorts declarations by name before hashing, so the router may rank however it likes.

    This is why `pruner.decide` returns a `frozenset` and not an ordered sequence: the order
    carries no billing consequence, and pretending otherwise would invite a stability rule that
    costs something and buys nothing.
    """
    forward = make_request(tools=["tool_00", "tool_01", "tool_02"])
    reversed_ = make_request(tools=["tool_02", "tool_01", "tool_00"])
    assert fingerprint(forward) == fingerprint(reversed_)

    smaller = make_request(tools=["tool_00", "tool_01"])
    assert fingerprint(forward) != fingerprint(smaller)


# --- The guard that protects all three ------------------------------------------


async def test_a_live_request_is_returned_exactly_as_it_arrived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live and bidi share `config` by reference, so a prune there is permanent (DESIGN.md 7.1).

    ADK preprocesses a live request once, before the socket opens. There is no later turn to
    correct on, which is why the router declines rather than pruning conservatively.
    """
    from google.adk.agents.run_config import StreamingMode

    router = make_plugin(monkeypatch, preds=(Prediction("tool_00", 0.95),))
    request = make_request()
    # A shallow copy is what ADK hands the model: `config` is the same object, so a prune
    # applied to the copy is visible through the original.
    shallow: Any = request.model_copy()
    before_fingerprint = fingerprint(request)

    await router.before_model_callback(
        callback_context=FakeContext(streaming_mode=StreamingMode.BIDI),  # type: ignore[arg-type]
        llm_request=shallow,
    )

    assert declared_names(request) == TOOLS
    assert fingerprint(request) == before_fingerprint
