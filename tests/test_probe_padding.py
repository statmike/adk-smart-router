"""The padding probe, checked without spending anything.

The run is cheap, about seven cents, so the reason to test it is not the money. It is that
the probe's whole claim rests on `padded` and `declared` being the same size and differing in
one thing only, and on the verdict firing for the reason it names. Both are decidable before
the first call, and neither is visible in the output if it is wrong.
"""

from __future__ import annotations

import probe_padding as probe
import pytest
from probe_padding import (
    COLD,
    FLOOR_TOKENS,
    GAPS,
    KEPT_DECLARATIONS,
    Call,
    build_arms,
    extra_declarations,
    head,
    hit_rate,
    padded_head_tokens,
    share,
    tools,
    verdict,
)


def arms() -> dict[str, probe.Arm]:
    return {arm.name: arm for arm in build_arms()}


# --- The plan ---


def test_padded_and_declared_are_the_same_size_within_a_percent() -> None:
    """If they are not, the probe is measuring size and calling it composition."""
    built = arms()
    padded, declared = built["padded"].prompt_tokens, built["declared"].prompt_tokens
    assert abs(padded - declared) / padded < 0.01, (padded, declared)


def test_padded_and_declared_differ_in_exactly_one_thing() -> None:
    built = arms()
    assert built["padded"].declarations == KEPT_DECLARATIONS
    assert built["declared"].declarations > KEPT_DECLARATIONS
    assert built["declared"].head_tokens == built["under"].head_tokens
    assert built["padded"].head_tokens > built["under"].head_tokens


def test_the_control_is_under_the_floor_and_the_other_two_are_over_it() -> None:
    built = arms()
    assert built["under"].prefix_tokens < FLOOR_TOKENS
    assert built["padded"].prefix_tokens > FLOOR_TOKENS
    assert built["declared"].prefix_tokens > FLOOR_TOKENS
    # And not by a handful of tokens, or drift in the sizing constants moves an arm across.
    assert FLOOR_TOKENS - built["under"].prefix_tokens > 200


def test_every_arm_has_a_private_head_and_a_private_namespace() -> None:
    """Two arms sharing either one would serve each other's cache and measure nothing."""
    built = build_arms()
    assert len({arm.namespace for arm in built}) == len(built)
    heads = {head(f"run {arm.name}", arm.head_tokens) for arm in built}
    assert len(heads) == len(built)
    for text in heads:
        assert text.startswith("PROBE ARM run ")


def test_the_declaration_count_is_what_the_request_actually_carries() -> None:
    built = arms()["declared"]
    declared = tools(built.namespace, built.declarations)[0].function_declarations
    assert declared is not None
    assert len(declared) == built.declarations
    assert extra_declarations() > 0 and padded_head_tokens() > 0


# --- The verdict ---


def _probe_calls(arm: str, hits: int, total: int, prompt: int = 4_000) -> list[Call]:
    return [
        Call(
            block="P",
            arm=arm,
            prefix_tokens=prompt,
            role="probe",
            index=i + 1,
            planned_gap_s=GAPS[0],
            actual_gap_s=GAPS[0],
            prompt_tokens=prompt,
            cached_tokens=prompt - 100 if i < hits else 0,
            opens_gap=True,
        )
        for i in range(total)
    ]


def _run(under: int, padded: int, declared: int, total: int = 20) -> list[Call]:
    return [
        *_probe_calls("under", under, total, prompt=1_700),
        *_probe_calls("padded", padded, total),
        *_probe_calls("declared", declared, total),
    ]


def test_the_measured_run_reads_as_padding_working() -> None:
    """The numbers this actually returned on 2026-09-05, so the reading cannot drift."""
    assert verdict(_run(under=4, padded=22, declared=21, total=40)).startswith("PADDING WORKS")


def test_a_cold_padded_arm_is_reported_as_the_remedy_failing() -> None:
    assert verdict(_run(under=0, padded=0, declared=18)).startswith("PADDING DOES NOT WORK")


def test_a_half_cold_padded_arm_is_reported_as_second_best_rather_than_as_working() -> None:
    assert verdict(_run(under=0, padded=4, declared=18)).startswith("PADDING IS SECOND BEST")


def test_a_warm_control_beats_every_other_reading() -> None:
    """An under-floor arm that caches invalidates the floor, so it is checked first."""
    assert verdict(_run(under=18, padded=18, declared=18)).startswith("THE FLOOR IS NOT WHERE")


def test_a_few_stray_hits_on_the_control_do_not_invalidate_the_floor() -> None:
    """Implicit caching is best-effort, and the real run put 4 of 40 under-floor probes warm."""
    assert hit_rate(_run(under=4, padded=22, declared=21, total=40), "under") < COLD
    assert verdict(_run(under=4, padded=22, declared=21, total=40)).startswith("PADDING WORKS")


def test_an_arm_with_no_usable_probe_is_inconclusive_rather_than_a_finding() -> None:
    assert verdict(_probe_calls("under", 0, 20)).startswith("INCONCLUSIVE")


def test_the_share_and_the_hit_rate_measure_different_things() -> None:
    """A cache that serves nearly the whole prefix when it serves at all: the observed shape."""
    calls = _probe_calls("padded", hits=10, total=20)
    assert hit_rate(calls, "padded") == pytest.approx(0.5)
    assert share(calls, "padded") == pytest.approx(0.5 * (3_900 / 4_000))


def test_the_plan_prices_itself_and_calls_nothing_without_confirm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["probe_padding.py", "--project", "test-project"])
    assert probe.main() == 2
    printed = capsys.readouterr().out
    assert "Nothing was called" in printed
    assert f"{sum(arm.calls for arm in build_arms()):,} fresh single-turn calls" in printed
    for arm in build_arms():
        assert f"{arm.prompt_tokens:>6,}" in printed
