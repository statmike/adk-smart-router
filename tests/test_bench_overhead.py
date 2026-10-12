"""Offline tests for scripts/bench_overhead.py. No GCS, no artifact, no timing.

What is worth testing here is the arithmetic between the stopwatch and the verdict,
because that is where a benchmark quietly lies: a p95 taken the wrong way, or a
per-turn number compared against a per-session bar.
"""

from __future__ import annotations

import pytest
from bench_overhead import G3_BAR_MS, Timing, declared, per_session_p95, report


def timing(name: str, samples: list[float]) -> Timing:
    return Timing(name, samples)


# --- Percentiles --------------------------------------------------------------


def test_p95_is_a_sample_that_really_happened():
    """Nearest-rank, not interpolated. An interpolated p95 reports a run nobody had."""
    measured = timing("decide", [float(n) for n in range(1, 101)])

    assert measured.p95 == 95.0
    assert measured.worst == 100.0


def test_p95_of_a_short_run_is_the_worst_of_it():
    assert timing("decide", [1.0, 9.0]).p95 == 9.0


def test_p95_of_a_single_sample_is_that_sample():
    assert timing("decide", [4.0]).p95 == 4.0


def test_p50_and_mean_come_apart_on_a_skewed_run():
    """The decide path is skewed by GC pauses, so reporting only the mean would hide it."""
    measured = timing("decide", [1.0, 1.0, 1.0, 1.0, 96.0])

    assert measured.p50 == 1.0
    assert measured.mean == pytest.approx(20.0)


# --- The per-session total ----------------------------------------------------


def test_a_session_pays_for_one_decide_and_the_rest_in_reuse():
    decide = timing("decide", [10.0])
    reuse = timing("reuse", [1.0])

    # 5 turns: one decide, four reuses, plus the one embedding call.
    assert per_session_p95(decide, reuse, 5.0, 200.0) == pytest.approx(214.0)


def test_a_one_turn_session_pays_no_reuse_at_all():
    decide = timing("decide", [10.0])
    reuse = timing("reuse", [1.0])

    assert per_session_p95(decide, reuse, 1.0, 0.0) == pytest.approx(10.0)


def test_a_fractional_turn_count_never_charges_negative_reuse():
    """The mean turns per session can dip below one; the total must not follow it down."""
    decide = timing("decide", [10.0])
    reuse = timing("reuse", [1.0])

    assert per_session_p95(decide, reuse, 0.4, 0.0) == pytest.approx(10.0)


def test_the_embedding_term_is_charged_once_per_session_not_once_per_turn():
    """It is paid on the decide turn only. Charging it per turn is the obvious error."""
    decide = timing("decide", [0.0])
    reuse = timing("reuse", [0.0])

    assert per_session_p95(decide, reuse, 10.0, 214.0) == pytest.approx(214.0)


# --- The verdict --------------------------------------------------------------


def test_the_shipped_numbers_pass_the_gate(capsys):
    """9.95 ms in process plus a 214 ms embedding call, against a 250 ms bar."""
    report(timing("decide", [3.6]), timing("reuse", [1.2]), 4.67, 214.0)

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "250 ms bar" in out


def test_a_slow_plugin_fails_the_gate_and_says_so(capsys):
    """The bar is per session, so a 60 ms decide path fails even at a fast endpoint."""
    report(timing("decide", [60.0]), timing("reuse", [10.0]), 4.67, 214.0)

    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "over the bar" in out


def test_the_report_attributes_the_total_to_the_term_that_dominates(capsys):
    """An operator who reads 224 ms should know which 214 of it is not the plugin's."""
    report(timing("decide", [3.6]), timing("reuse", [1.2]), 4.67, 214.0)

    assert "96% of it" in capsys.readouterr().out


def test_a_zero_embedding_term_reports_the_plugin_alone(capsys):
    """`--embed-p95 0` is how you see what the code costs without the network."""
    report(timing("decide", [3.6]), timing("reuse", [1.2]), 4.67, 0.0)

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "0% of it" in out


def test_the_bar_is_the_one_design_states():
    assert G3_BAR_MS == 250.0


# --- Reading the request ------------------------------------------------------


def test_declared_counts_every_declaration_across_every_tool_block():
    """ADK may split declarations over several Tool entries; counting one would undercount."""
    from google.genai import types

    request = types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(name="a"),
                    types.FunctionDeclaration(name="b"),
                ]
            ),
            types.Tool(function_declarations=[types.FunctionDeclaration(name="c")]),
        ]
    )

    assert declared(_Request(request)) == 3


def test_declared_is_zero_when_the_request_carries_no_tools():
    from google.genai import types

    assert declared(_Request(types.GenerateContentConfig())) == 0


class _Request:
    """Only `.config` is read, and building a real LlmRequest here would prove nothing."""

    def __init__(self, config: object) -> None:
        self.config = config
