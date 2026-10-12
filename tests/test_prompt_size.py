"""Tests for measuring a request instead of guessing at it.

Two things are being proven. First that `count_chars` walks the real `LlmRequest` shape
and puts every character on the right side of the prune. Second that `TokenScale` moves
toward what the model billed without letting one bad response move it somewhere it
cannot come back from.

The estimate is characters divided by a ratio, which is only ever approximately right.
So the tests assert bands and directions rather than exact token counts, except where an
exact answer is the point: the split between the prunable and unprunable sides, and the
refusal to learn from a response that cannot be true.
"""

from __future__ import annotations

import pytest
from google.genai import types

from adk_smart_router.prompt_size import (
    PROSE_CHARS_PER_TOKEN,
    SCHEMA_CHARS_PER_TOKEN,
    Chars,
    TokenScale,
    count_chars,
    counting_vertex_declarations,
)


def request(
    *,
    instruction: str | None = None,
    prompts: list[str] | None = None,
    tools: dict[str, str] | None = None,
) -> types.GenerateContentConfig:
    """A stand-in for LlmRequest. Only `.config` and `.contents` are ever read."""

    class Req:
        pass

    declarations = [
        types.FunctionDeclaration(name=name, description=description)
        for name, description in (tools or {}).items()
    ]
    req = Req()
    req.config = types.GenerateContentConfig(  # type: ignore[attr-defined]
        system_instruction=instruction,
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
    )
    req.contents = [  # type: ignore[attr-defined]
        types.Content(role="user", parts=[types.Part(text=text)]) for text in (prompts or [])
    ]
    return req  # type: ignore[return-value]


# --- count_chars --------------------------------------------------------------


def test_an_empty_request_is_zero_everywhere():
    chars = count_chars(request(), None)
    assert (chars.base, chars.kept, chars.dropped) == (0, 0, 0)


def test_the_system_instruction_and_history_are_the_unprunable_base():
    chars = count_chars(request(instruction="x" * 100, prompts=["y" * 40, "z" * 60]), None)
    assert chars.base == 200
    assert chars.dropped == 0


def test_kept_none_means_nothing_would_be_dropped():
    chars = count_chars(request(tools={"a": "d" * 100, "b": "d" * 100}), None)
    assert chars.dropped == 0
    assert chars.kept > 200  # the JSON envelope costs more than the descriptions alone


def test_declarations_split_by_whether_the_prune_keeps_them():
    chars = count_chars(request(tools={"a": "d" * 100, "b": "d" * 100}), frozenset({"a"}))
    assert chars.dropped > 100
    # Two identically shaped declarations, one each side of the line.
    assert abs(chars.kept - chars.dropped) < 5


def test_full_and_pruned_differ_by_exactly_what_is_dropped():
    chars = Chars(base=1000, kept=200, dropped=800)
    assert chars.full == 2000
    assert chars.pruned == 1200


def test_an_unnamed_declaration_always_survives():
    """Nothing can prune what it cannot name, so counting it as dropped would overstate
    the saving and let the guard approve a prune that never happens."""
    req = request(tools={"a": "d" * 50})
    req.config.tools[0].function_declarations.append(types.FunctionDeclaration())
    chars = count_chars(req, frozenset())
    assert chars.kept > 0


def test_a_function_call_bills_as_the_json_it_serialises_to():
    req = request()
    req.contents = [
        types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name="lookup", args={"id": "42"}))],
        )
    ]
    chars = count_chars(req, None)
    assert chars.base > len("lookup")


def test_inline_data_is_a_flat_rate_not_its_base64_length():
    """A megabyte of PNG does not bill as a megabyte of text, and counting it that way
    would swamp every other term in the estimate."""
    req = request()
    req.contents = [
        types.Content(
            role="user",
            parts=[types.Part(inline_data=types.Blob(mime_type="image/png", data=b"p" * 100_000))],
        )
    ]
    assert count_chars(req, None).base < 5_000


def test_a_string_system_instruction_and_a_content_one_agree():
    text = "you are a support agent" * 10
    as_string = count_chars(request(instruction=text), None).base
    req = request()
    req.config.system_instruction = types.Content(role="user", parts=[types.Part(text=text)])
    assert count_chars(req, None).base == as_string


# --- TokenScale ---------------------------------------------------------------


def test_one_blended_ratio_cannot_fit_both_terms():
    """Why the model has two terms rather than one, in arithmetic.

    A single ratio has to be fitted at some declaration count. Fit it where the request
    is mostly schema and it is badly wrong where the request is mostly prose, because
    the two tokenize at rates that differ by a factor of two. The prune is exactly the
    operation that moves a request from one end of that range to the other, so a blended
    ratio is at its worst on the only question the router asks.
    """
    base = 4_240  # 1000 tokens of conversation
    schema = 8_330  # 1000 tokens of declarations

    two_term = TokenScale().full_tokens(Chars(base=base, kept=schema, dropped=0))
    assert abs(two_term - 2_000) <= 2

    # The blend that fits the request above is 6.29 chars per token. Applied to the same
    # request with its declarations pruned away, it understates by a third.
    blended = (base + schema) / two_term
    assert abs(base / blended - 1_000) > 300


def test_before_any_response_the_ratios_are_the_shipped_defaults():
    scale = TokenScale()
    assert scale.samples == 0
    assert scale.prose_ratio == PROSE_CHARS_PER_TOKEN


def test_a_response_moves_the_prose_ratio_toward_what_was_billed():
    scale = TokenScale()
    # 8000 prose chars, no schema, billed 1000 tokens: 8.0 chars per token, denser prose
    # than the default assumes.
    scale.observe(8_000, 0, 1_000)
    assert scale.samples == 1
    assert abs(scale.prose_ratio - 8.0) < 0.01


def test_the_schema_term_is_subtracted_before_prose_is_attributed():
    scale = TokenScale()
    schema_chars = 8_330  # exactly 1000 tokens at the fixed schema rate
    scale.observe(4_240, schema_chars, 2_000)
    assert abs(scale.prose_ratio - 4.24) < 0.05  # the remaining 1000 tokens are the prose


def test_calibration_accumulates_rather_than_chasing_the_last_turn():
    scale = TokenScale()
    for _ in range(20):
        scale.observe(4_240, 0, 1_000)  # 4.24 chars per token, twenty times
    scale.observe(4_240, 0, 100)  # one wild outlier: 42.4 chars per token
    assert scale.samples == 20  # rejected outright, being outside the plausible band
    assert abs(scale.prose_ratio - 4.24) < 0.01


def test_an_implausible_ratio_is_rejected_not_learned_from():
    scale = TokenScale()
    scale.observe(100, 0, 1)  # 100 chars per token: no tokenizer does that
    scale.observe(1, 0, 100)  # 0.01 chars per token: nor that
    assert scale.samples == 0
    assert scale.prose_ratio == PROSE_CHARS_PER_TOKEN


def test_a_response_with_no_usage_numbers_is_ignored():
    scale = TokenScale()
    scale.observe(4_240, 0, 0)
    scale.observe(0, 0, 1_000)
    assert scale.samples == 0


def test_a_bill_smaller_than_the_schema_estimate_teaches_nothing():
    """If the declarations alone should have cost more than the whole request did, the
    leftover prose term is negative and attributing it would corrupt the ratio."""
    scale = TokenScale()
    scale.observe(4_240, 83_300, 100)
    assert scale.samples == 0


def test_a_calibrated_scale_changes_the_answer_it_gives():
    chars = Chars(base=10_000, kept=0, dropped=0)
    scale = TokenScale()
    before = scale.full_tokens(chars)
    for _ in range(10):
        scale.observe(8_000, 0, 1_000)  # this agent's prose is twice as dense
    assert scale.full_tokens(chars) < before / 1.5


def test_dropped_declarations_only_count_toward_the_full_size():
    scale = TokenScale()
    chars = Chars(base=1_000, kept=1_000, dropped=8_330)
    assert scale.full_tokens(chars) - scale.pruned_tokens(chars) == int(
        8_330 / SCHEMA_CHARS_PER_TOKEN
    )


# --- The joint fit --------------------------------------------------------------
#
# The numbers below are the reference agent's, measured against Vertex `count_tokens`:
# 88,431 characters of declarations bill 14,327 tokens, which is 6.17 characters per
# token and not the 8.33 the module shipped with (DEV_NOTES.md, 2026-09-04). The tests
# generate billing from those two ratios and check the fit gets them back.

TRUE_PROSE = 4.4
TRUE_SCHEMA = 6.17
FULL_LIST_CHARS = 88_431
PRUNED_LIST_CHARS = 38_000


def billed(base_chars: int, schema_chars: int) -> int:
    """What a model billing at TRUE_PROSE and TRUE_SCHEMA would charge."""
    return round(base_chars / TRUE_PROSE + schema_chars / TRUE_SCHEMA)


def feed(scale: TokenScale, samples: list[tuple[int, int]]) -> None:
    for base_chars, schema_chars in samples:
        scale.observe(base_chars, schema_chars, billed(base_chars, schema_chars))


def mixed_traffic(n: int = 40) -> list[tuple[int, int]]:
    """Alternating full and pruned requests, the shape a routing agent produces."""
    return [
        (8_400 + (i % 5) * 400, FULL_LIST_CHARS if i % 2 else PRUNED_LIST_CHARS) for i in range(n)
    ]


def test_a_routing_agent_separates_the_two_ratios():
    scale = TokenScale()
    feed(scale, mixed_traffic())
    assert scale.fitted
    assert abs(scale.schema_ratio - TRUE_SCHEMA) < 0.05
    assert abs(scale.prose_ratio - TRUE_PROSE) < 0.05


def test_the_fit_waits_for_enough_responses():
    scale = TokenScale()
    feed(scale, mixed_traffic(20))
    assert not scale.fitted
    assert scale.schema_ratio == SCHEMA_CHARS_PER_TOKEN


def test_one_unchanging_tool_list_cannot_separate_them():
    """Schema constant and history barely moving is the case the fit has to refuse."""
    scale = TokenScale()
    feed(scale, [(8_400 + (i % 9) * 200, FULL_LIST_CHARS) for i in range(40)])
    assert not scale.fitted
    assert scale.schema_ratio == SCHEMA_CHARS_PER_TOKEN


def test_a_long_conversation_separates_them_without_ever_pruning():
    """A base term spanning a factor of five pins the constant schema term by itself."""
    scale = TokenScale()
    feed(scale, [(8_400 + i * 1_200, FULL_LIST_CHARS) for i in range(40)])
    assert scale.fitted
    assert abs(scale.schema_ratio - TRUE_SCHEMA) < 0.05


def test_the_fit_corrects_the_saving_the_fixed_rate_understated():
    """The defect this exists for: what a prune is worth, at 8.33 against the truth."""
    chars = Chars(base=8_400, kept=PRUNED_LIST_CHARS, dropped=FULL_LIST_CHARS - PRUNED_LIST_CHARS)
    truth = round(chars.dropped / TRUE_SCHEMA)

    uncalibrated = TokenScale()
    understated = uncalibrated.full_tokens(chars) - uncalibrated.pruned_tokens(chars)
    assert understated < truth * 0.80  # 8.33 against 6.17 is a quarter of the saving lost

    scale = TokenScale()
    feed(scale, mixed_traffic())
    assert abs((scale.full_tokens(chars) - scale.pruned_tokens(chars)) - truth) <= 1


def test_an_impossible_fit_falls_back_rather_than_shipping_itself():
    """Billing that no pair of positive ratios explains must not become the answer."""
    scale = TokenScale()
    for i in range(40):
        # More declarations billed as strictly fewer tokens: the schema rate solves negative.
        schema_chars = FULL_LIST_CHARS if i % 2 else PRUNED_LIST_CHARS
        scale.observe(8_400, schema_chars, 20_000 - schema_chars // 10)
    assert not scale.fitted
    assert scale.schema_ratio == SCHEMA_CHARS_PER_TOKEN


def test_the_fallback_keeps_learning_while_the_fit_is_unavailable():
    """Traffic that cannot separate the terms must still calibrate prose, as it always did."""
    scale = TokenScale()
    for _ in range(40):
        scale.observe(8_000, 0, 1_000)
    assert not scale.fitted  # no schema characters at all: nothing to separate
    assert abs(scale.prose_ratio - 8.0) < 0.01


# --- Which backend is being counted -----------------------------------------------


@pytest.mark.parametrize(
    "value,vertex",
    [("true", True), ("TRUE", True), ("1", True), (" true ", True), ("false", False), ("", False)],
)
def test_the_backend_is_read_from_the_shell_and_not_from_the_installation(
    monkeypatch, value, vertex
):
    """Truthy the way ADK reads it, whitespace and case included, and nothing else.

    A wrong answer here is a caveat printed when it should not have been, or worse, missing
    when it should have been there. `qualify` and `predict_saving.py` both hang a warning on
    it, so both would mislead in the direction that costs the reader money.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)
    assert counting_vertex_declarations() is vertex


def test_either_of_the_two_names_counts_because_adk_is_mid_rename(monkeypatch):
    """ADK deprecated `GOOGLE_GENAI_USE_VERTEXAI` for `GOOGLE_GENAI_USE_ENTERPRISE`.

    Both are live, and an environment that sets only the new one is on Vertex just as much
    as one that sets only the old.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    assert counting_vertex_declarations()

    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE")
    assert not counting_vertex_declarations()
