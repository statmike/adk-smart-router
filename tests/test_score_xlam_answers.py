"""The answer-quality scorer's arithmetic, checked without the warehouse.

This is the only measurement in the project that can say whether a token saving was bought
with worse answers, so what has to be right is the accounting that reveals that: the three
per-arm rates, the exclusion of sessions with nothing to recall, and the significance test
that stops a few hundred sessions of noise being read as harm.

One test carries the real run. If `xlam-ab1` ever stops reproducing p = 0.003, the script and
the number written into DESIGN.md 19.66 have diverged.
"""

from __future__ import annotations

import json

import pytest
from score_xlam_answers import (
    Score,
    Session,
    missed_tools,
    read_journal,
    score,
    welch,
)


def a_session(sid: str = "s1", wanted: tuple[str, ...] = ("a",), called: tuple[str, ...] = ("a",)):
    return Session(session_id=sid, wanted=frozenset(wanted), called=frozenset(called))


# --- One session at a time ----------------------------------------------------------------


def test_calling_everything_wanted_is_a_hit_even_with_extras():
    """Extra calls are the router's business, not the answer key's. A session that made the
    call it needed answered the question, whatever else it did on the way."""
    s = a_session(wanted=("a", "b"), called=("a", "b", "c"))
    assert s.hit
    assert s.recall == 1.0


def test_half_the_wanted_calls_is_half_the_recall_and_not_a_hit():
    s = a_session(wanted=("a", "b"), called=("a",))
    assert not s.hit
    assert s.recall == pytest.approx(0.5)


def test_calling_the_wrong_tool_is_not_silence():
    """The two failures a prune causes look different and are counted separately. Calling
    something else means the tool was there and the model chose it; silence means the model
    had nothing to choose."""
    s = a_session(wanted=("a",), called=("z",))
    assert s.recall == 0.0
    assert not s.silent
    assert a_session(wanted=("a",), called=()).silent


# --- One arm at a time --------------------------------------------------------------------


def test_the_three_rates_are_shares_of_the_arm():
    sessions = [
        a_session("s1", ("a",), ("a",)),
        a_session("s2", ("a", "b"), ("a",)),
        a_session("s3", ("a",), ()),
        a_session("s4", ("a",), ("a",)),
    ]
    scored = score("routed", sessions)
    assert scored.sessions == 4
    assert scored.hit_rate == pytest.approx(0.5)
    assert scored.silent_rate == pytest.approx(0.25)
    assert scored.mean_recall == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)


def test_an_empty_arm_divides_by_nothing_rather_than_raising():
    scored = score("routed", [])
    assert scored.hit_rate == 0.0
    assert scored.silent_rate == 0.0


def test_the_commonest_miss_is_reported_first():
    sessions = [
        a_session("s1", ("a", "b"), ()),
        a_session("s2", ("a",), ()),
        a_session("s3", ("a",), ("a",)),
    ]
    assert missed_tools(sessions, 8) == [("a", 2), ("b", 1)]


# --- The journal --------------------------------------------------------------------------


def test_a_query_the_key_answers_with_no_call_is_not_scored(tmp_path):
    """The same exclusion `publish_router.py` makes when it gates on menu recall. A session
    with nothing to recall would score both arms a free one and shrink every difference."""
    path = tmp_path / "run.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"session_id": "s1", "wanted": ["a"], "called": ["a"]},
                {"session_id": "s2", "wanted": [], "called": []},
                {"session_id": "s3", "wanted": ["b"], "called": []},
            )
        )
        + "\n"
    )

    read = read_journal(path)

    assert [s.session_id for s in read] == ["s1", "s3"]


def test_a_journal_with_a_blank_line_still_reads(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"session_id": "s1", "wanted": ["a"], "called": []}) + "\n\n")
    assert len(read_journal(path)) == 1


def test_a_session_that_never_recorded_its_calls_is_read_as_silent(tmp_path):
    """A killed run leaves rows with no `called` key. Reading that as "called everything"
    would hide exactly the failure this script exists to find."""
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"session_id": "s1", "wanted": ["a"]}) + "\n")
    assert read_journal(path)[0].silent


# --- Is the difference real? ---------------------------------------------------------------


def test_the_measured_run_says_the_saving_was_not_free():
    """`xlam-ab1`, 2026-09-06. The routed arm called every wanted tool on 75.6% of sessions
    against the holdout arm's 87.2%, and this is the test that says that is a real gap and
    not the noise of 560 sessions. DESIGN.md 19.66."""
    # The measured means and spreads rather than 560 rows of corpus, because the arithmetic
    # on top of them is what is under test here.
    routed = Score("routed", 295, hits=223, silences=31, mean_recall=0.8102, sd_recall=0.3600)
    holdout = Score("holdout", 265, hits=231, silences=18, mean_recall=0.8912, sd_recall=0.2900)

    t, p = welch(routed, holdout)

    assert t == pytest.approx(-2.93, abs=0.02)
    assert p < 0.01


def test_two_arms_that_answer_the_same_are_not_a_finding():
    routed = score(
        "routed", [a_session(f"r{i}", ("a",), ("a",) if i % 4 else ()) for i in range(200)]
    )
    holdout = score(
        "holdout", [a_session(f"h{i}", ("a",), ("a",) if i % 4 else ()) for i in range(200)]
    )
    _, p = welch(routed, holdout)
    assert p > 0.05


def test_two_arms_with_no_spread_are_not_a_significance_test():
    routed = score("routed", [a_session(f"r{i}", ("a",), ("a",)) for i in range(50)])
    holdout = score("holdout", [a_session(f"h{i}", ("a",), ("a",)) for i in range(50)])
    assert welch(routed, holdout) is None


def test_a_single_session_arm_is_not_a_significance_test_either():
    routed = score("routed", [a_session("r0", ("a",), ())])
    holdout = score("holdout", [a_session(f"h{i}", ("a",), ("a",)) for i in range(50)])
    assert welch(routed, holdout) is None


def test_the_no_menus_run_says_the_quality_cost_was_the_menus():
    """`xlam-ab2`, 2026-09-07. Same booster and the same 600 queries as `xlam-ab1`, pruning per
    session instead of serving pooled menus. The gap the run above found is gone, and this is the
    test that says so: nine points became eight tenths of a point, and p went from 0.003 to 0.71.
    The saving went with it, from -13.4% of the bill to -2.0%, which the ledger measures rather
    than this. DESIGN.md 19.67."""
    routed = Score("routed", 228, hits=202, silences=9, mean_recall=0.9145, sd_recall=0.2568)
    holdout = Score("holdout", 308, hits=278, silences=13, mean_recall=0.9226, sd_recall=0.2513)

    t, p = welch(routed, holdout)

    assert t == pytest.approx(-0.37, abs=0.02)
    assert p > 0.05
