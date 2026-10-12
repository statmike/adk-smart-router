"""Tests for report_ab.py. Fully offline: no BigQuery client is ever built.

Everything here exercises the arithmetic and the verdict, which is the part that
can be wrong without anyone noticing. The SQL statements are not tested here; they
are checked by running them, and the runs are recorded in DESIGN.md 18.12.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from report_ab import (
    G1_TARGET,
    Arm,
    Completeness,
    PrunableTool,
    completeness_line,
    completeness_note,
    completeness_summary,
    infer_planned,
    parse_tools,
    prunable_note,
    relative_change,
    report,
    routing_stopped_early,
)


def timeline(*, run_minutes: float, routed_minutes: float, watch: int = 0) -> dict[str, object]:
    start = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    return {
        "run_start": start,
        "run_end": start + timedelta(minutes=run_minutes),
        "last_routed": start + timedelta(minutes=routed_minutes),
        "watch_decisions": watch,
    }


def arm(name: str, *, turns: int = 100, prompt: int = 6000, cached: int = 0) -> Arm:
    """One arm. Token counts are totals, so a per-turn figure is prompt/turns."""
    return Arm(
        name=name,
        sessions=max(turns // 3, 1),
        turns=turns,
        prompt_tokens=prompt * turns,
        cached_tokens=cached * turns,
        completion_tokens=180 * turns,
        total_ms=1500 * turns,
    )


def verdict(arms: Sequence[Arm]) -> bool:
    return report(arms, [])


# --- The billed-equivalent unit ---------------------------------------------------


def test_with_no_cache_billed_equals_prompt_tokens():
    assert arm("routed", turns=10, prompt=1000).billed_per_turn == pytest.approx(1000)


def test_cached_tokens_are_discounted_not_free():
    """Half the prompt cached is 0.5 + 0.5*0.1 = 0.55 of the uncached bill."""
    both = arm("routed", turns=10, prompt=1000, cached=500)
    assert both.billed_per_turn == pytest.approx(550)
    assert both.cached_share == pytest.approx(0.5)


def test_cached_never_exceeding_prompt_is_enforced_not_assumed():
    """A warehouse row with cached > prompt must not produce a negative bill."""
    weird = Arm(
        "routed", 1, 1, prompt_tokens=100, cached_tokens=400, completion_tokens=0, total_ms=0
    )
    assert weird.billed_per_turn == pytest.approx(40.0)


def test_an_empty_arm_reports_zero_rather_than_dividing_by_zero():
    empty = Arm("routed", 0, 0, 0, 0, 0, 0)
    assert empty.billed_per_turn == 0.0
    assert empty.prompt_per_turn == 0.0
    assert empty.cached_share == 0.0
    assert empty.completion_per_turn == 0.0
    assert empty.ms_per_turn == 0.0


# --- G1 ---------------------------------------------------------------------------


def test_relative_change_is_negative_when_routing_is_cheaper():
    change = relative_change(arm("routed", prompt=3000), arm("holdout", prompt=6000))
    assert change == pytest.approx(-0.5)


def test_relative_change_is_none_without_a_baseline():
    assert relative_change(arm("routed"), Arm("holdout", 0, 0, 0, 0, 0, 0)) is None


def test_turn_counts_are_normalised_before_comparing():
    """Unequal arm sizes must not move G1. Only the per-turn rate may."""
    change = relative_change(
        arm("routed", turns=17, prompt=3900), arm("holdout", turns=413, prompt=6000)
    )
    assert change == pytest.approx(-0.35)


# --- The verdict ------------------------------------------------------------------


def test_a_10_percent_drop_passes_exactly_on_the_line():
    assert relative_change(
        arm("routed", prompt=5400), arm("holdout", prompt=6000)
    ) == pytest.approx(G1_TARGET)
    assert verdict([arm("holdout", prompt=6000), arm("routed", prompt=5400)]) is True


def test_a_9_percent_drop_is_a_real_saving_and_still_fails():
    """The gate is a line, not a sentiment. 9% is a saving and it is not the gate."""
    assert verdict([arm("holdout", prompt=6000), arm("routed", prompt=5460)]) is False


def test_a_missing_arm_is_inconclusive_not_a_pass():
    assert verdict([arm("routed", prompt=100)]) is False
    assert verdict([arm("holdout", prompt=6000)]) is False


def test_too_few_turns_is_inconclusive_even_when_the_number_looks_good():
    thin = [arm("holdout", turns=12, prompt=6000), arm("routed", turns=12, prompt=1000)]
    assert verdict(thin) is False


def test_a_zero_token_baseline_is_inconclusive():
    hollow = Arm("holdout", 40, 100, 0, 0, 0, 0)
    assert verdict([hollow, arm("routed", prompt=3900)]) is False


def test_sessions_in_neither_arm_do_not_enter_g1():
    """The router records its decisions not to route. Those turns are not a control."""
    arms = [
        arm("holdout", prompt=6000),
        arm("routed", prompt=3900),
        arm("neither", turns=500, prompt=99_000),
    ]
    assert verdict(arms) is True


def test_the_source_table_is_printed_without_an_average(capsys: pytest.CaptureFixture[str]):
    """AVG over an empty group comes back NULL. It must render, not crash."""
    sources = [
        {"source": "reused", "decisions": 120, "sessions": 60, "avg_kept": 10.0},
        {"source": "holdout", "decisions": 110, "sessions": 55, "avg_kept": None},
    ]
    report([arm("holdout", prompt=6000), arm("routed", prompt=3900)], sources)
    out = capsys.readouterr().out
    assert "reused" in out
    assert "10.0" in out
    assert "PASS" in out


# --- Did the router route for the whole run? ----------------------------------
#
# A one-way cache_watch demotion turns the treatment arm into a second control arm and
# says nothing in the arms table. DESIGN.md 18.21 is that run: routing stopped eleven
# minutes into fifty-five, and it took a per-minute query after the fact to notice.


def test_a_run_that_routed_throughout_says_nothing():
    assert routing_stopped_early(timeline(run_minutes=55, routed_minutes=55)) is None


def test_a_few_unrouted_sessions_at_the_tail_are_not_worth_a_warning():
    assert routing_stopped_early(timeline(run_minutes=55, routed_minutes=51)) is None


def test_a_demotion_partway_through_is_called_out():
    warning = routing_stopped_early(timeline(run_minutes=55, routed_minutes=11, watch=511))
    assert warning is not None
    assert "11 minutes into a 55-minute run" in warning
    assert "cache_watch demoted the agent" in warning
    assert "20%" in warning


def test_routing_that_stopped_for_some_other_reason_says_so():
    """No `watch` decisions means something else stopped it, and the reader should not be
    sent looking at the wrong guard."""
    warning = routing_stopped_early(timeline(run_minutes=60, routed_minutes=6, watch=0))
    assert warning is not None
    assert "not a cache_watch demotion" in warning


def test_a_run_with_no_routed_decision_at_all_is_left_to_the_arms_table():
    """`last_routed` is NULL when nothing routed. That is already an inconclusive verdict
    and a timing warning on top of it would just be noise."""
    empty = timeline(run_minutes=55, routed_minutes=0)
    empty["last_routed"] = None
    assert routing_stopped_early(empty) is None
    assert routing_stopped_early({}) is None


def test_the_warning_is_printed_above_the_arms(capsys: pytest.CaptureFixture[str]):
    report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        timeline(run_minutes=55, routed_minutes=11, watch=511),
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert out.index("WARNING") < out.index("arms")


# --- A control arm borrowed from another run ---------------------------------------


def test_a_borrowed_control_arm_is_named_in_the_report(capsys: pytest.CaptureFixture[str]):
    """A holdout row that came from a different run must not read like this run's."""
    report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        None,
        baseline_prefix="ab-42b-",
    )
    out = capsys.readouterr().out
    assert "SEQUENTIAL" in out
    assert "ab-42b-" in out
    assert out.index("SEQUENTIAL") < out.index("arms")


def test_a_same_run_holdout_says_nothing_extra(capsys: pytest.CaptureFixture[str]):
    report([arm("holdout", prompt=6000), arm("routed", prompt=3900)], [])
    assert "SEQUENTIAL" not in capsys.readouterr().out


# --- How much of the run actually landed ----------------------------------------
#
# A ratio says nothing about the denominator behind it. DESIGN.md 18.23 lost 40 of 600
# planned sessions to quota and the report of that run read exactly like a whole one.
# The numbers below are that run and the two after it, taken from BigQuery.


def completeness(
    *, planned: int = 600, started: int = 600, failed: int = 0, quota: int = 0, breaker: int = 0
) -> Completeness:
    return Completeness(
        planned=planned, started=started, failed=failed, quota_failed=quota, breaker=breaker
    )


def test_a_run_that_landed_whole_is_not_warned_about():
    """ab-v2-100tools-: 600 planned, 600 started, none failed."""
    assert completeness_note(completeness()) is None


def test_a_handful_of_lost_sessions_is_not_worth_a_warning():
    """ab-20260910a-: 4 of 300 lost to quota is 1.3%, inside the 5% threshold."""
    assert completeness_note(completeness(planned=300, started=300, failed=4, quota=4)) is None


def test_a_run_that_lost_seven_percent_warns_and_names_quota():
    warning = completeness_note(completeness(planned=300, started=300, failed=21, quota=21))
    assert warning is not None
    assert "93%" in warning
    assert "21 of 300" in warning
    assert "7% of the plan went to quota" in warning
    assert "RESOURCE_EXHAUSTED" in warning


def test_a_loss_that_was_not_quota_sends_the_reader_elsewhere():
    """Naming quota for a loss that was something else would waste the next hour."""
    warning = completeness_note(completeness(planned=600, started=600, failed=90, quota=0))
    assert warning is not None
    assert "None of it was quota" in warning


def test_the_warning_explains_why_a_partial_session_biases_the_arm_it_was_in():
    """A session that died partway paid for its uncached turn one and none of the cheap
    cached turns after it, so its arm reads dearer per turn than it is."""
    warning = completeness_note(completeness(planned=600, started=600, failed=90, quota=90))
    assert warning is not None
    assert "uncached first turn" in warning
    assert "dearer per turn" in warning


def test_sessions_that_never_started_count_against_the_plan():
    """ab-20260907-: 600 planned, 592 started, 27 of those errored."""
    lost = completeness(planned=600, started=592, failed=27, quota=27)
    assert lost.never_started == 8
    assert lost.landed == 565
    assert lost.lost == 35
    assert lost.landed_share == pytest.approx(565 / 600)


def test_breaker_sessions_are_counted_apart_from_failures():
    """A breaker session ran end to end. It is whole, it is just in neither arm, so it
    must not be subtracted from what landed."""
    both = completeness(planned=600, started=600, failed=0, breaker=26)
    assert both.landed == 600
    assert both.landed_share == pytest.approx(1.0)
    assert completeness_note(both) is None
    summary = completeness_summary(both)
    assert summary is not None
    assert "26 more tripped the embedding breaker" in summary
    assert "off the routed side" in summary


def test_a_run_with_no_breaker_sessions_says_nothing_about_the_breaker():
    assert completeness_summary(completeness()) == "on 100% of 600 planned sessions"


def test_an_empty_prefix_has_nothing_to_summarise():
    empty = completeness(planned=0, started=0)
    assert completeness_summary(empty) is None
    assert completeness_note(empty) is None
    assert completeness_line(empty) == "no sessions under this prefix."


# --- What the plan was ------------------------------------------------------------


def test_the_plan_is_read_off_the_highest_session_index():
    """run_ab.py numbers sessions from zero, so the top index is the plan minus one."""
    assert infer_planned(top_index=599, started=592, override=None) == 600


def test_an_explicit_planned_count_beats_the_inferred_one():
    """A run whose tail never started leaves no row to infer from, so --planned wins."""
    assert infer_planned(top_index=411, started=400, override=600) == 600


def test_an_inferred_plan_below_what_started_never_books_a_negative_loss():
    """A malformed or renumbered session id must not report better than complete."""
    plan = infer_planned(top_index=3, started=100, override=None)
    assert plan == 100
    assert completeness(planned=plan, started=100).lost == 0
    assert completeness(planned=plan, started=100).landed_share == pytest.approx(1.0)


def test_an_explicit_planned_count_below_what_started_is_also_clamped():
    assert infer_planned(top_index=299, started=300, override=10) == 300


def test_a_prefix_with_no_rows_at_all_plans_nothing():
    assert infer_planned(top_index=None, started=0, override=None) == 0


# --- Completeness in the report ---------------------------------------------------


def test_the_completeness_block_is_printed_above_the_arms(capsys: pytest.CaptureFixture[str]):
    report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        None,
        None,
        completeness(planned=600, started=600),
    )
    out = capsys.readouterr().out
    assert "completeness" in out
    assert out.index("completeness") < out.index("\narms")


def test_the_verdict_line_carries_the_denominator_with_it(capsys: pytest.CaptureFixture[str]):
    """The G1 line is what gets quoted on its own, so the sample size travels with it."""
    passed = report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        None,
        None,
        completeness(planned=600, started=600, breaker=26),
    )
    out = capsys.readouterr().out
    assert passed is True
    g1 = out.index("G1  billed-equivalent")
    assert "on 100% of 600 planned sessions" in out[g1:]
    assert out.index("on 100% of 600 planned sessions") < out.index("target")


def test_an_incomplete_run_still_gets_its_verdict(capsys: pytest.CaptureFixture[str]):
    """Warn and state, do not gate. A run that lost a fifth of its sessions is worth a
    warning and is still worth reading; only a thin arm is INCONCLUSIVE."""
    passed = report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        None,
        None,
        completeness(planned=600, started=520, failed=40, quota=40),
    )
    out = capsys.readouterr().out
    assert passed is True
    assert "WARNING" in out
    assert "PASS" in out
    assert "INCONCLUSIVE" not in out


def test_a_thin_arm_is_still_inconclusive_however_complete_the_run_was():
    thin = [arm("holdout", turns=12, prompt=6000), arm("routed", turns=12, prompt=1000)]
    assert report(thin, [], None, None, completeness(planned=8, started=8)) is False


def test_a_report_without_completeness_prints_no_block(capsys: pytest.CaptureFixture[str]):
    """The three older callers pass nothing, and nothing is what they should see."""
    report([arm("holdout", prompt=6000), arm("routed", prompt=3900)], [])
    assert "completeness" not in capsys.readouterr().out


def test_the_control_period_of_a_sequential_ab_reports_its_own_completeness(
    capsys: pytest.CaptureFixture[str],
):
    """DESIGN.md 18.24 ran two periods. Either one can be the truncated one."""
    report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        None,
        "ab-42b-",
        completeness(planned=300, started=300),
        completeness(planned=300, started=240, failed=30, quota=30),
    )
    out = capsys.readouterr().out
    assert "control   210 of 300 planned sessions landed (70%)" in out
    assert "control run ab-42b-" in out


def test_the_completeness_line_reads_as_a_sentence():
    line = completeness_line(completeness(planned=600, started=592, failed=27, quota=27))
    assert line == (
        "565 of 600 planned sessions landed (94%); 8 never started, 27 errored "
        "(27 of those on quota)."
    )


# --- The declared-prunable detector ------------------------------------------------
#
# A tool named in `config.prunable_by_default` is removed from every routed prompt, so it
# costs nothing and G1 cannot see it. The only thing that can go wrong is a session that
# wanted it, and the only place that shows is a per-arm call count. The numbers below are
# the ab-v2 run, where all 32 classless tools were still declared to both arms: 3 holdout
# calls and 8 routed ones across 574 sessions, which is the whole live signal there is.


def called(name: str, *, holdout: int = 0, routed: int = 0) -> PrunableTool:
    return PrunableTool(
        name=name,
        holdout_calls=holdout,
        routed_calls=routed,
        holdout_sessions=holdout,
        routed_sessions=routed,
    )


def test_a_tool_only_the_holdout_called_is_named_as_a_miss_the_prune_caused():
    note = prunable_note([called("start_return", holdout=2)], ["start_return", "book_callback"])
    assert note is not None
    assert "start_return (2 sessions)" in note
    assert "1 declared-prunable tools were called by the holdout arm" in note


def test_the_single_session_case_reads_as_one_session_not_one_sessions():
    """It is the commonest case in this block: the measured signal is one or two calls."""
    note = prunable_note([called("start_return", holdout=1)], ["start_return"])
    assert note is not None
    assert "start_return (1 session)" in note


def test_a_tool_both_arms_called_is_not_read_as_a_miss():
    """Both arms calling it means the name is not yet in the deployed config, so the run
    measures demand rather than harm."""
    note = prunable_note([called("get_price", holdout=2, routed=2)], ["get_price"])
    assert note is not None
    assert "none of the 1 declared-prunable tools" in note


def test_a_silent_run_is_reported_as_no_evidence_rather_than_as_no_harm():
    note = prunable_note([], ["book_callback", "submit_review"])
    assert note is not None
    assert "absence of evidence" in note


def test_declaring_nothing_prunable_leaves_the_detector_silent():
    assert prunable_note([], []) is None


def test_the_detector_block_is_skipped_entirely_when_no_tool_was_declared(
    capsys: pytest.CaptureFixture[str],
):
    report([arm("holdout", prompt=6000), arm("routed", prompt=3900)], [])
    assert "declared prunable" not in capsys.readouterr().out


def test_a_declared_tool_no_arm_called_still_prints_its_row_of_the_block(
    capsys: pytest.CaptureFixture[str],
):
    """Silence is the expected reading and has to be visible, or a reader cannot tell the
    detector ran from the detector finding nothing."""
    report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        prunable=["book_callback"],
        prunable_calls=[],
    )
    out = capsys.readouterr().out
    assert "declared prunable, and called anyway (1 tools declared)" in out
    assert "no arm called any of them." in out


def test_the_detector_never_changes_the_g1_verdict(capsys: pytest.CaptureFixture[str]):
    """Underpowered by construction, so it reports and never gates (see `report`)."""
    passed = report(
        [arm("holdout", prompt=6000), arm("routed", prompt=3900)],
        [],
        prunable=["start_return"],
        prunable_calls=[called("start_return", holdout=3)],
    )
    assert passed
    assert "start_return" in capsys.readouterr().out


def test_a_tool_list_is_split_and_stripped_rather_than_passed_through_whole():
    assert parse_tools(" book_callback , submit_review ,, ") == ["book_callback", "submit_review"]


def test_an_unset_tool_list_is_empty_and_not_a_list_holding_one_empty_name():
    assert parse_tools("") == []
