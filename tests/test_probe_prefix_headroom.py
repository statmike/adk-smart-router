"""The headroom sweep, checked without spending anything.

The run is cheap, so the reason to test it is not the money. It is that the sweep's whole claim
rests on the arms landing where their labels say they land, and on the verdict refusing to name
a threshold that a single warm arm produced by accident. Both are decidable before the first
call, and neither is visible in the output if it is wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import probe_prefix_headroom as probe
import pytest
from probe_prefix_headroom import (
    CALIBRATION_COUNTS,
    COLD,
    FLOOR_TOKENS,
    FRAMING_TOKENS,
    GAP_S,
    HEADROOMS,
    PROBES,
    Arm,
    Call,
    build_arms,
    calls_per_arm,
    declarations_for,
    fit,
    head,
    hit_rate,
    is_live,
    namespace,
    smallest_live,
    tools,
    verdict,
)

# --- The plan ---


def test_every_arm_lands_where_its_label_says_it_does() -> None:
    """A step of 128 tokens is worth nothing if an arm can miss its target by that much."""
    for arm in build_arms():
        assert abs(arm.prefix_tokens - (FLOOR_TOKENS + arm.headroom)) <= 32, arm


def test_the_sweep_starts_under_the_floor_and_climbs_from_there() -> None:
    arms = build_arms()
    assert arms[0].headroom < 0
    assert arms[0].prefix_tokens < FLOOR_TOKENS
    assert [arm.headroom for arm in arms] == sorted(HEADROOMS)
    assert [arm.prefix_tokens for arm in arms] == sorted(arm.prefix_tokens for arm in arms)


def test_a_prompt_is_its_prefix_plus_the_part_no_two_sessions_share() -> None:
    for arm in build_arms():
        assert arm.prompt_tokens == arm.prefix_tokens + FRAMING_TOKENS


def test_every_arm_has_a_private_head_and_a_private_namespace() -> None:
    """Two arms sharing either one would serve each other's cache and measure nothing."""
    arms = build_arms()
    assert len({arm.namespace for arm in arms}) == len(arms)
    heads = {head(f"run {arm.name}") for arm in arms}
    assert len(heads) == len(arms)
    for text in heads:
        assert text.startswith("PROBE ARM run ")


def test_every_namespace_is_the_same_width_as_every_other() -> None:
    """A declaration is billed by the byte, so a longer namespace is a bigger arm.

    The first run of this probe had namespaces from three characters to six, calibration
    included, and landed two arms 150 tokens either side of their targets.
    """
    widths = {len(arm.namespace) for arm in build_arms()}
    widths |= {len(namespace(count)) for count in CALIBRATION_COUNTS}
    assert len(widths) == 1, widths


def test_the_declaration_count_is_what_the_request_actually_carries() -> None:
    arm = build_arms()[-1]
    declared = tools(arm.namespace, arm.declarations)[0].function_declarations
    assert declared is not None
    assert len(declared) == arm.declarations


# --- Calibration ---


def test_the_fit_recovers_a_line_it_was_given() -> None:
    base, per = fit((10, 110), (10 * 45 + 330, 110 * 45 + 330))
    assert per == pytest.approx(45.0)
    assert base == pytest.approx(330.0)


def test_the_fit_refuses_two_readings_of_the_same_size() -> None:
    with pytest.raises(ValueError):
        fit((10, 10), (500, 500))


def test_a_measured_declaration_moves_every_arm() -> None:
    """The point of calibrating: a declaration a tenth bigger is a different sweep."""
    default = build_arms()
    measured = build_arms(base=330.0, per=49.5)
    assert [arm.declarations for arm in measured] != [arm.declarations for arm in default]
    for arm in measured:
        assert abs(arm.prefix_tokens - (FLOOR_TOKENS + arm.headroom)) <= 32, arm


def test_an_arm_never_asks_for_fewer_than_one_declaration() -> None:
    """A base that already clears the target would otherwise ask for zero, or for less."""
    assert declarations_for(1_000, base=4_000.0, per=45.0) == 1


# --- The verdict ---


def _probes(arm: Arm, hits: int, total: int = 20) -> list[Call]:
    return [
        Call(
            block="H",
            arm=arm.name,
            prefix_tokens=arm.prefix_tokens,
            role="probe",
            index=i + 1,
            planned_gap_s=GAP_S,
            actual_gap_s=GAP_S,
            prompt_tokens=arm.prompt_tokens,
            cached_tokens=arm.prefix_tokens if i < hits else 0,
            opens_gap=True,
        )
        for i in range(total)
    ]


def _run(hits: dict[int, int], total: int = 20) -> tuple[list[Call], list[Arm]]:
    arms = build_arms()
    calls = [call for arm in arms for call in _probes(arm, hits[arm.headroom], total)]
    return calls, arms


def _dead() -> dict[int, int]:
    return dict.fromkeys(HEADROOMS, 0)


def _control(calls: Sequence[Call]) -> float:
    return hit_rate(calls, "-512") or 0.0


def test_a_clean_transition_names_the_headroom_it_found() -> None:
    hits = _dead() | {512: 16, 1_024: 17, 2_048: 18, 4_096: 18}
    assert verdict(*_run(hits)).startswith("HEADROOM IS 512 TOKENS")


def test_the_measured_run_reads_as_a_dead_band_under_a_live_one() -> None:
    """What the sweep returned on 2026-09-06, over 100 probes an arm, so it cannot drift."""
    hits = {-512: 1, 0: 2, 256: 1, 512: 28, 2_048: 26}
    arms = build_arms(headrooms=tuple(hits))
    calls = [call for arm in arms for call in _probes(arm, hits[arm.headroom], total=100)]
    said = verdict(calls, arms)
    assert said.startswith("HEADROOM IS 512 TOKENS")
    assert "necessary and not sufficient" in said


def test_the_follow_up_run_confirms_the_band_stays_live_above_the_edge() -> None:
    """2026-09-06's fourth run, which re-read the two arms the eight-arm sweep dipped on."""
    hits = {-512: 2, 512: 13, 1_024: 20, 4_096: 26}
    arms = build_arms(headrooms=tuple(hits))
    calls = [call for arm in arms for call in _probes(arm, hits[arm.headroom], total=100)]
    said = verdict(calls, arms)
    assert said.startswith("HEADROOM IS 512 TOKENS")
    # This sweep skipped the floor and +256, so it cannot bracket the edge from below.
    assert "the edge may be lower" in said


def test_a_live_arm_at_the_floor_leaves_qualify_alone() -> None:
    hits = {headroom: 16 for headroom in HEADROOMS} | {-512: 1}
    assert verdict(*_run(hits)).startswith("NO HEADROOM IS NEEDED")


def test_a_reliable_band_is_reported_as_sufficient_rather_than_necessary() -> None:
    hits = _dead() | {512: 18, 1_024: 18, 2_048: 18, 4_096: 18}
    assert "opened warm on at least" in verdict(*_run(hits))


def test_a_live_control_beats_every_other_reading() -> None:
    """An under-floor arm that caches invalidates the floor, so it is checked first."""
    hits = {headroom: 16 for headroom in HEADROOMS}
    assert verdict(*_run(hits)).startswith("THE FLOOR IS NOT WHERE")


def test_a_few_stray_hits_on_the_control_do_not_invalidate_the_floor() -> None:
    """Implicit caching is best-effort, and probe_padding put 4 of 40 under-floor probes warm."""
    hits = _dead() | {-512: 2, 1_024: 17, 2_048: 18, 4_096: 18}
    calls, arms = _run(hits)
    assert hit_rate(calls, "-512") < COLD
    assert verdict(calls, arms).startswith("HEADROOM IS 1,024 TOKENS")


def test_an_arm_that_only_matches_a_noisy_control_is_not_live() -> None:
    """Without the multiple, a control caching 8% would make a 10% arm a threshold."""
    hits = _dead() | {-512: 2, 512: 3, 1_024: 3, 2_048: 3, 4_096: 3}
    calls, arms = _run(hits)
    assert not is_live(calls, arms[-1], _control(calls))
    assert verdict(calls, arms).startswith("NO CLEAN THRESHOLD")


def test_one_live_arm_under_dead_ones_is_not_a_threshold() -> None:
    """The mistake this probe exists to avoid: calibrating a margin off a single reading."""
    hits = _dead() | {256: 18, 4_096: 18}
    calls, arms = _run(hits)
    assert smallest_live(calls, arms, _control(calls)) is None
    assert verdict(calls, arms).startswith("NO CLEAN THRESHOLD")


def test_a_live_top_arm_on_its_own_is_not_a_threshold_either() -> None:
    """Live from here up needs an up, and the largest arm in the sweep has none."""
    hits = _dead() | {4_096: 18}
    calls, arms = _run(hits)
    assert smallest_live(calls, arms, _control(calls)) is None
    assert "extend the sweep past it" in verdict(calls, arms)


def test_a_run_where_nothing_caches_reads_as_a_broken_run() -> None:
    hits = _dead() | {4_096: 1}
    assert verdict(*_run(hits)).startswith("NOTHING OVER THE FLOOR CACHES")


def test_an_arm_with_no_usable_probe_is_inconclusive_rather_than_a_finding() -> None:
    calls, arms = _run(_dead() | {1_024: 18, 2_048: 18, 4_096: 18})
    kept = [call for call in calls if call.arm != "+512"]
    assert verdict(kept, arms).startswith("INCONCLUSIVE")


def test_a_missing_control_is_inconclusive_rather_than_a_clean_sweep() -> None:
    calls, arms = _run({headroom: 18 for headroom in HEADROOMS})
    kept = [call for call in calls if call.arm != "-512"]
    assert verdict(kept, arms).startswith("INCONCLUSIVE")


def test_an_errored_probe_is_not_counted_either_way() -> None:
    arm = build_arms()[-1]
    calls = [*_probes(arm, hits=0, total=4)]
    calls = [Call(**{**vars(call), "error": "APIError 429"}) for call in calls]
    assert hit_rate(calls, arm.name) is None


def test_the_plan_prices_itself_and_calls_nothing_without_confirm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["probe_prefix_headroom.py", "--project", "test-project"])
    assert probe.main() == 2
    printed = capsys.readouterr().out
    assert "Nothing was called" in printed
    total = len(build_arms()) * calls_per_arm(PROBES)
    assert f"{total:,} fresh single-turn calls" in printed
    for arm in build_arms():
        assert f"{arm.prefix_tokens:>7,}  {arm.prompt_tokens:>7,}" in printed
