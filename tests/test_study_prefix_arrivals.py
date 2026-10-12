"""The arrival-rate study's arithmetic, checked without touching the warehouse.

The script makes one claim that costs money if it is wrong: that the routed arm would bill
about a third less than the control if its prefixes saw the control's traffic. That number
is a repricing of turn one and nothing else, so what has to be right is the bucketing that
finds the comparable calls, the repricing that holds everything else fixed, and the refusal
to report either when the data cannot support it.

One test carries the real run, for the same reason `test_report_ledger.py` does.
"""

from __future__ import annotations

import pytest
from study_prefix_arrivals import (
    FIRST,
    Bucket,
    Totals,
    TurnOne,
    bucket_of,
    buckets_for,
    busiest_bucket,
    median,
    report,
    reprice,
)


def a_call(
    arm: str = "routed",
    *,
    gap_s: float | None = 10.0,
    prompt_tokens: int = 8_000,
    cached_tokens: int = 6_000,
) -> TurnOne:
    return TurnOne(arm=arm, gap_s=gap_s, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)


# --- Which bucket a call lands in -------------------------------------------------


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (None, FIRST),
        (0.0, "<30s"),
        (29.9, "<30s"),
        (30.0, "<2m"),
        (119.0, "<2m"),
        (300.0, "<15m"),
        (3_600.0, "15m+"),
    ],
)
def test_a_gap_lands_in_the_bucket_its_edges_say(gap, expected):
    assert bucket_of(gap) == expected


def test_a_prefix_never_sent_before_is_not_a_long_idle_gap():
    """Cold because it has never existed is a different claim from cold because it aged."""
    assert bucket_of(None) == FIRST
    assert bucket_of(None) != "15m+"


def test_buckets_come_back_in_gap_order_with_the_first_send_leading():
    rows = [
        a_call(gap_s=1_000.0),
        a_call(gap_s=None),
        a_call(gap_s=5.0),
        a_call(gap_s=200.0),
    ]
    assert [bucket.name for bucket in buckets_for(rows, "routed")] == [
        FIRST,
        "<30s",
        "<5m",
        "15m+",
    ]


def test_an_empty_bucket_is_left_out_rather_than_printed_as_a_zero():
    assert [b.name for b in buckets_for([a_call(gap_s=5.0)], "routed")] == ["<30s"]


def test_a_bucket_reports_how_often_apart_from_how_much():
    """Warm rate and depth are two different deficits and the fix for each is different."""
    rows = [
        a_call(gap_s=5.0, cached_tokens=0),
        a_call(gap_s=6.0, cached_tokens=7_000),
        a_call(gap_s=7.0, cached_tokens=7_000),
        a_call(gap_s=8.0, cached_tokens=7_000),
    ]
    bucket = buckets_for(rows, "routed")[0]
    assert bucket.warm_rate == pytest.approx(0.75)
    assert bucket.mean_cached == pytest.approx(5_250.0)
    assert bucket.depth_when_warm == pytest.approx(7_000.0)


def test_calls_from_the_other_arm_do_not_leak_into_a_bucket():
    rows = [a_call("routed", gap_s=5.0), a_call("holdout", gap_s=5.0)]
    assert buckets_for(rows, "routed")[0].calls == 1


def test_the_busiest_bucket_is_the_one_the_counterfactual_reads_from():
    buckets = [
        Bucket(name="<30s", calls=3, warm_calls=1, prompt_tokens=1, cached_tokens=1),
        Bucket(name="<2m", calls=9, warm_calls=1, prompt_tokens=1, cached_tokens=1),
    ]
    assert busiest_bucket(buckets).name == "<2m"
    assert busiest_bucket([]) is None


# --- Repricing turn one -----------------------------------------------------------


def a_total(
    arm: str = "routed",
    *,
    sessions: int = 100,
    prompt_tokens: int = 1_000_000,
    cached_tokens: int = 400_000,
) -> Totals:
    return Totals(
        arm=arm, sessions=sessions, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens
    )


def test_repricing_at_what_already_happened_changes_nothing():
    rows = [a_call(gap_s=5.0, cached_tokens=6_000) for _ in range(10)]
    totals = a_total()
    assert reprice(totals, rows, 6_000) == pytest.approx(totals.billed_per_session)


def test_a_warmer_turn_one_bills_less_at_nine_tenths_of_the_tokens_it_gains():
    """The saving is the token moving from uncached to cached, so 0.9 of it, not all."""
    rows = [a_call(gap_s=5.0, cached_tokens=6_000) for _ in range(10)]
    totals = a_total(sessions=10)
    before = totals.billed_per_session
    after = reprice(totals, rows, 7_000)
    assert before - after == pytest.approx(0.9 * 1_000)


def test_a_target_larger_than_the_prompt_is_capped_at_the_prompt():
    rows = [a_call(gap_s=5.0, prompt_tokens=8_000, cached_tokens=0)]
    totals = a_total(sessions=1)
    capped = reprice(totals, rows, 50_000)
    exact = reprice(totals, rows, 8_000)
    assert capped == pytest.approx(exact)


def test_only_the_named_arm_is_repriced():
    rows = [a_call("routed", cached_tokens=6_000), a_call("holdout", cached_tokens=0)]
    totals = a_total("routed", sessions=1)
    moved = totals.billed_per_session - reprice(totals, rows, 7_000)
    assert moved == pytest.approx(0.9 * 1_000)


def test_an_arm_with_no_opening_calls_is_left_at_what_it_billed():
    totals = a_total("routed")
    assert reprice(totals, [a_call("holdout")], 9_000) == pytest.approx(totals.billed_per_session)


def test_the_median_arrivals_line_survives_an_even_count_and_an_empty_run():
    assert median([1, 3, 5, 9]) == pytest.approx(4.0)
    assert median([]) == pytest.approx(0.0)


# --- What it prints ---------------------------------------------------------------


def _split(total: int, parts: int) -> list[int]:
    """`total` over `parts` rows, exactly, so the fixture's sums are the run's sums."""
    if parts <= 0:
        return []
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _measured_rows() -> list[TurnOne]:
    """`ab-v2-100tools`, 2026-09-06. Per-bucket totals as the warehouse returned them.

    Warm and cold calls are kept apart rather than averaged together, because the whole
    point of the second scenario is that the routed arm's deficit is a warm *rate* and not
    a depth, and a fixture of bucket means would hide exactly that.
    """
    shape = [
        # arm, gap, calls, warm calls, prompt total, cached total
        ("holdout", None, 1, 0, 16_222, 0),
        ("holdout", 10.0, 284, 259, 4_609_505, 4_108_673),
        ("holdout", 60.0, 4, 3, 64_942, 47_877),
        ("routed", None, 24, 6, 192_951, 26_349),
        ("routed", 10.0, 67, 50, 537_117, 369_511),
        ("routed", 60.0, 57, 43, 456_536, 284_840),
        ("routed", 200.0, 73, 47, 585_353, 318_377),
        ("routed", 600.0, 54, 34, 433_703, 210_757),
        ("routed", 2_000.0, 10, 8, 80_927, 43_895),
    ]
    rows = []
    for arm, gap, calls, warm, prompt_total, cached_total in shape:
        prompts = _split(prompt_total, calls)
        cached = _split(cached_total, warm) + [0] * (calls - warm)
        rows.extend(
            TurnOne(arm=arm, gap_s=gap, prompt_tokens=p, cached_tokens=c)
            for p, c in zip(prompts, cached, strict=True)
        )
    return rows


def _measured_totals() -> list[Totals]:
    return [
        Totals(arm="routed", sessions=285, prompt_tokens=11_162_239, cached_tokens=9_769_636),
        Totals(arm="holdout", sessions=289, prompt_tokens=22_352_172, cached_tokens=21_376_791),
    ]


def _measured_arrivals() -> dict[str, list[int]]:
    return {"holdout": [289], "routed": [30, 29, 28, 20, 17, 17, 14, 12, 12, 11, 10, 10]}


def test_the_report_reproduces_the_measured_counterfactual(capsys):
    assert report(_measured_rows(), _measured_totals(), _measured_arrivals()) == 0
    printed = capsys.readouterr().out
    # What the run billed, and the two scenarios that bracket production shape.
    assert "-22.8%" in printed
    assert "-32.1%" in printed
    assert "-42.4%" in printed
    assert "an upper bound, not a measurement" in printed


def test_the_asymmetry_is_stated_before_any_counterfactual(capsys):
    report(_measured_rows(), _measured_totals(), _measured_arrivals())
    printed = capsys.readouterr().out
    assert "1 control prefix over 289 sessions" in printed
    assert printed.index("control prefix") < printed.index("arrival rate")


def test_a_one_armed_run_is_refused_rather_than_reported(capsys):
    assert report(_measured_rows(), [_measured_totals()[0]], {}) == 1
    assert "at least one is missing" in capsys.readouterr().out


def test_a_run_with_no_routed_calls_in_the_control_s_bucket_says_so(capsys):
    rows = [a_call("holdout", gap_s=5.0), a_call("routed", gap_s=5_000.0)]
    assert report(rows, _measured_totals(), _measured_arrivals()) == 0
    printed = capsys.readouterr().out
    assert "no routed traffic in the control's own bucket" in printed
    assert "upper bound" not in printed
