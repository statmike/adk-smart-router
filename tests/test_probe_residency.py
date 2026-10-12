"""The residency probe, checked without spending anything.

`probe_residency.py` costs about $2.25 to run, so everything that can be decided before the
first call is decided here: that the plan is the experiment the finding asked for, that the
arms are isolated from each other in the two ways that make it a residency probe at all,
that the price it quotes is the price of the plan it printed, and that each verdict fires on
the evidence it claims to fire on.
"""

from __future__ import annotations

import probe_residency as probe
import pytest
from probe_residency import (
    GAPS_A,
    LOAD_INTERVAL_S,
    LOAD_SENDS,
    MAX_DECLARATIONS,
    PROBES_PER_GAP,
    REQUEST_OVERHEAD_TOKENS,
    SIZES_A,
    SIZES_B,
    TOKENS_PER_DECLARATION,
    Call,
    arm_share,
    build_waves,
    built_tokens,
    clusters,
    declaration,
    estimate,
    first_after_gap,
    head,
    head_body,
    main,
    shape,
    tools,
    user_text,
    verdict_head,
    verdict_overall,
    verdict_rate,
    verdict_size,
    verdict_volume,
)


def all_arms():
    return [arm for wave in build_waves() for arm in wave.arms]


def call(
    arm: str,
    share: float,
    *,
    block: str = "A",
    size: int = 8_192,
    gap: float = 60.0,
    opens_gap: bool = True,
    index: int = 1,
    error: str = "",
) -> Call:
    prompt = size
    return Call(
        block=block,
        arm=arm,
        prefix_tokens=size,
        role="probe",
        index=index,
        planned_gap_s=gap,
        actual_gap_s=gap,
        prompt_tokens=prompt,
        cached_tokens=round(prompt * share),
        opens_gap=opens_gap,
        error=error,
    )


def arm_calls(arm: str, share: float, *, block: str = "A", size: int = 8_192, gaps=(60.0,)):
    """Both probes after each of `gaps`, the way a completed arm reports."""
    out = []
    index = 0
    for gap in gaps:
        for within in range(PROBES_PER_GAP):
            index += 1
            out.append(
                call(
                    arm,
                    share,
                    block=block,
                    size=size,
                    gap=gap,
                    opens_gap=within == 0,
                    index=index,
                )
            )
    return out


def block_a_calls(shares: dict[int, float]):
    return [
        c for size in SIZES_A for c in arm_calls(f"A-{size}", shares[size], size=size, gaps=GAPS_A)
    ]


def block_b_calls(by_arm: dict[str, float]):
    return [
        c
        for size in SIZES_B
        for suffix, share in by_arm.items()
        for c in arm_calls(f"B-{size}-{suffix}", share, block="B", size=size)
    ]


# --- The plan is the experiment the finding asked for --------------------------------


def test_block_a_varies_size_and_holds_the_schedule_identical():
    """Size is the independent variable, so nothing else in block A may differ."""
    arms = [arm for arm in all_arms() if arm.block == "A"]
    assert [arm.prefix_tokens for arm in arms] == list(SIZES_A)
    assert {arm.sends for arm in arms} == {LOAD_SENDS}
    assert {arm.interval_s for arm in arms} == {LOAD_INTERVAL_S}
    assert {arm.probe_gaps for arm in arms} == {GAPS_A}


def test_block_a_probes_five_seconds_a_minute_and_ten_minutes():
    assert GAPS_A == (5.0, 60.0, 600.0)


def test_block_b_varies_sends_and_cadence_at_both_sizes():
    """Volume and rate are the controls, and a control at one size says nothing about another."""
    arms = {arm.name: arm for arm in all_arms() if arm.block == "B"}
    for size in SIZES_B:
        assert [arms[f"B-{size}-x{n}"].sends for n in (4, 24, 96)] == [4, 24, 96]
        assert {arms[f"B-{size}-x{n}"].interval_s for n in (4, 24, 96)} == {LOAD_INTERVAL_S}
        slow = arms[f"B-{size}-slow"]
        assert slow.sends == LOAD_SENDS and slow.interval_s == 10 * LOAD_INTERVAL_S


def test_the_rate_arm_sends_as_often_as_the_volume_arm_it_is_compared_against():
    """Otherwise a colder slow arm could be fewer sends rather than a slower cadence."""
    arms = {arm.name: arm for arm in all_arms() if arm.block == "B"}
    for size in SIZES_B:
        assert arms[f"B-{size}-slow"].sends == arms[f"B-{size}-x{LOAD_SENDS}"].sends


def test_block_c_puts_all_three_arms_on_one_byte_identical_head():
    arms = [arm for arm in all_arms() if arm.block == "C"]
    assert len({arm.head_marker for arm in arms}) == 1
    assert {head(arm.head_marker) for arm in arms} == {head(arms[0].head_marker)}


def test_the_noise_arm_is_bigger_than_the_arm_it_is_meant_to_crowd_out():
    arms = {arm.name: arm for arm in all_arms() if arm.block == "C"}
    assert arms["C2-noise"].prefix_tokens > arms["C2-contended"].prefix_tokens
    assert arms["C2-noise"].probe_gaps == ()  # it measures nothing, it only occupies the head


def test_the_noise_arm_covers_the_whole_window_its_victim_is_idle_for():
    """A noise arm that finishes early would leave C2 to cool down in peace."""
    arms = {arm.name: arm for arm in all_arms() if arm.block == "C"}
    assert arms["C2-noise"].seconds >= arms["C2-contended"].seconds


# --- Isolation: the two rules that make it a residency probe --------------------------


def test_every_arm_outside_block_c_owns_a_private_head():
    """A shared head is block C's variable, so nowhere else may have one by accident."""
    markers = [arm.head_marker for arm in all_arms() if arm.block != "C"]
    assert len(markers) == len(set(markers))


def test_two_heads_with_different_markers_differ_at_the_first_byte():
    """Implicit caching matches a literal prefix, so nothing after byte zero can be shared."""
    one, other = head("A-2048"), head("A-4096")
    assert one[0:1] == other[0:1]  # both start "P"
    assert one.split("\n")[0] != other.split("\n")[0]
    assert one[: len("PROBE ARM A-")] == other[: len("PROBE ARM A-")]
    assert one[len("PROBE ARM A-") :] != other[len("PROBE ARM A-") :]


def test_the_head_body_is_identical_no_matter_what_the_marker_is():
    """So two arms that do share a marker share the head exactly, not approximately."""
    body = head_body()
    assert head("one").endswith(body)
    assert head("a much longer marker than that one").endswith(body)


def test_the_head_is_about_the_size_the_reference_agent_s_head_is():
    """Measured at 1,613 tokens for 8,169 characters, which is 5.06 characters to the token."""
    assert 1_400 <= len(head_body()) / 5.06 <= 1_800


def test_every_target_size_is_built_to_within_one_declaration_of_itself():
    """Block A compares sizes, so an arm that misses its target is comparing the wrong thing."""
    for target in (*SIZES_A, *SIZES_B):
        assert abs(built_tokens(*shape(target)) - target) <= TOKENS_PER_DECLARATION


def test_a_target_at_or_below_the_request_overhead_still_declares_something():
    """Zero declarations is not a tool list, and the API would not accept it."""
    count, pad_chars = shape(REQUEST_OVERHEAD_TOKENS - 500)
    assert count >= 1
    assert pad_chars == 0


def test_the_largest_arm_pads_its_descriptions_because_vertex_stops_at_512():
    """Past the ceiling the only way left to buy tokens is to make each declaration longer."""
    count, pad_chars = shape(32_768)
    assert count == MAX_DECLARATIONS
    assert pad_chars > 0


def test_an_arm_that_fits_under_the_ceiling_buys_its_tokens_in_whole_tools():
    """Padding is a workaround, not the mechanism; a real menu grows by having more tools."""
    for target in (2_048, 4_096, 8_192, 16_384):
        count, pad_chars = shape(target)
        assert count <= MAX_DECLARATIONS
        assert pad_chars == 0


def test_every_declaration_name_is_the_same_length_so_the_cost_stays_linear():
    """A variable-width index made per-declaration cost drift from 45 tokens up to 60."""
    names = {declaration("a0", i).name for i in range(MAX_DECLARATIONS)}
    assert len(names) == MAX_DECLARATIONS  # and none collide
    assert len({len(name) for name in names}) == 1


def test_a_built_tool_list_declares_exactly_what_shape_asked_for():
    count, pad_chars = shape(8_192)
    declarations = [d for tool in tools("a2", count, pad_chars) for d in tool.function_declarations]
    assert len(declarations) == count


def test_no_two_arms_share_a_declaration_namespace():
    """Two arms with the same tool names would share a cached prefix and measure each other."""
    namespaces = [arm.namespace for arm in all_arms()]
    assert len(namespaces) == len(set(namespaces))


def test_no_two_calls_send_the_same_user_message():
    """A literal repeat could be answered from a response cache, which measures nothing."""
    texts = [user_text("run1", "A-2048", i) for i in range(1, 40)]
    assert len(set(texts)) == len(texts)


def test_a_second_run_shares_no_message_with_the_first():
    assert user_text("run1", "A-2048", 1) != user_text("run2", "A-2048", 1)


# --- Waves ----------------------------------------------------------------------------


def test_a_and_b_run_together_and_c_runs_after_them():
    waves = build_waves()
    assert [wave.name for wave in waves] == ["A+B", "C1", "C2"]
    assert {arm.block for arm in waves[0].arms} == {"A", "B"}


def test_c1_runs_with_nothing_else_in_flight_because_that_is_the_control():
    waves = {wave.name: wave for wave in build_waves()}
    assert [arm.name for arm in waves["C1"].arms] == ["C1-alone"]
    assert {arm.name for arm in waves["C2"].arms} == {"C2-contended", "C2-noise"}


def test_asking_for_one_block_plans_only_that_block():
    assert {arm.block for wave in build_waves(["A"]) for arm in wave.arms} == {"A"}
    assert build_waves([]) == []


# --- Cost -----------------------------------------------------------------------------


def test_the_quoted_price_is_the_price_of_the_plan_it_printed():
    waves = build_waves()
    tokens, usd, minutes, lines = estimate(waves, "gemini-2.5-flash")
    assert tokens == sum(arm.calls * arm.prefix_tokens for wave in waves for arm in wave.arms)
    assert len(lines) == sum(len(wave.arms) for wave in waves)
    assert usd == tokens / 1e6 * 0.30
    assert 0 < minutes < 60


def test_the_whole_probe_stays_inside_its_three_dollar_budget():
    _, usd, _, _ = estimate(build_waves(), "gemini-2.5-flash")
    assert usd < 3.00


def test_a_pricier_model_costs_more_for_the_same_plan():
    cheap = estimate(build_waves(), "gemini-2.5-flash")[1]
    dear = estimate(build_waves(), "gemini-3-flash")[1]
    assert dear > cheap


def test_running_with_no_flags_prices_the_plan_and_sends_nothing(monkeypatch, capsys):
    """The gate. Anything that reaches the API without --confirm is a bug that costs money."""

    def refuse(*args, **kwargs):
        raise AssertionError("the dry run built a client")

    monkeypatch.setattr(probe.genai, "Client", refuse)
    monkeypatch.setattr("sys.argv", ["probe_residency.py", "--project", "test-project"])
    assert main() == 2
    printed = capsys.readouterr().out
    assert "Re-run with --confirm" in printed
    assert "input tokens, worst case $" in printed


# --- Reading the answer ----------------------------------------------------------------


def test_only_the_first_probe_after_a_gap_counts_as_a_turn_one():
    calls = arm_calls("A-8192", 0.5, gaps=(5.0, 60.0))
    assert len(first_after_gap(calls, "A-8192")) == 2
    assert len(first_after_gap(calls, "A-8192", 60.0)) == 1


def test_a_failed_call_is_not_evidence_of_a_cold_cache():
    """It reports zero cached tokens because it never happened, not because nothing cached."""
    calls = [call("A-8192", 0.9), call("A-8192", 0.0, index=3, error="ClientError 429")]
    assert arm_share(calls, "A-8192") == pytest.approx(0.9, abs=1e-3)


def test_an_arm_that_never_reported_has_no_share_rather_than_a_zero():
    assert arm_share([], "A-8192") is None


def test_cached_token_counts_cluster_so_a_head_only_match_is_visible():
    """The 1,586-token pile in the A/B is the shared head serving with nothing behind it."""
    calls = [
        call("x", 0.0),
        call("x", 0.0),
        *[Call("A", "x", 8_192, "probe", i, 60.0, 60.0, 8_192, 1_590 + i, True) for i in range(4)],
        *[Call("A", "x", 8_192, "probe", i, 60.0, 60.0, 8_192, 7_170 + i, True) for i in range(3)],
    ]
    assert clusters(calls) == [(0, 0, 2), (1_590, 1_593, 4), (7_170, 7_172, 3)]


def test_clustering_says_nothing_when_nothing_was_measured():
    assert clusters([]) == []


# --- Verdicts ---------------------------------------------------------------------------


def test_size_governs_when_the_small_arm_is_cold_and_the_big_ones_are_hot():
    verdict = verdict_size(
        block_a_calls({2_048: 0.10, 4_096: 0.40, 8_192: 0.60, 16_384: 0.88, 32_768: 0.92})
    )
    assert verdict.startswith("SIZE GOVERNS")
    assert "MEASURED_CURVE" in verdict


def test_size_is_dead_when_sixteen_fold_in_size_moves_nothing():
    verdict = verdict_size(
        block_a_calls({2_048: 0.70, 4_096: 0.71, 8_192: 0.72, 16_384: 0.72, 32_768: 0.73})
    )
    assert verdict.startswith("SIZE IS DEAD")


def test_size_moving_without_clearing_either_threshold_is_reported_as_partial():
    verdict = verdict_size(
        block_a_calls({2_048: 0.35, 4_096: 0.45, 8_192: 0.55, 16_384: 0.70, 32_768: 0.75})
    )
    assert verdict.startswith("PARTIAL")


def test_the_size_verdict_refuses_to_conclude_from_an_incomplete_block():
    partial = block_a_calls({size: 0.5 for size in SIZES_A})
    without_the_biggest = [c for c in partial if c.arm != f"A-{SIZES_A[-1]}"]
    assert verdict_size(without_the_biggest).startswith("INCONCLUSIVE")


def test_volume_governs_when_more_sends_buy_a_warmer_prefix_at_both_sizes():
    verdict = verdict_volume(block_b_calls({"x4": 0.20, "x24": 0.50, "x96": 0.80, "slow": 0.5}))
    assert verdict.startswith("VOLUME GOVERNS")


def test_volume_is_dead_when_twenty_four_times_the_sends_buys_nothing():
    verdict = verdict_volume(block_b_calls({"x4": 0.60, "x24": 0.62, "x96": 0.61, "slow": 0.6}))
    assert verdict.startswith("VOLUME IS DEAD")


def test_rate_governs_when_the_same_sends_at_a_tenth_the_cadence_run_colder():
    verdict = verdict_rate(block_b_calls({"x4": 0.5, "x24": 0.80, "x96": 0.5, "slow": 0.40}))
    assert verdict.startswith("RATE GOVERNS")


def test_rate_is_dead_when_cadence_costs_nothing_at_the_same_send_count():
    verdict = verdict_rate(block_b_calls({"x4": 0.5, "x24": 0.80, "x96": 0.5, "slow": 0.78}))
    assert verdict.startswith("RATE IS DEAD")


def test_heads_compete_when_the_contended_arm_runs_colder_than_the_lonely_one():
    calls = [
        *arm_calls("C1-alone", 0.80, block="C"),
        *arm_calls("C2-contended", 0.30, block="C"),
    ]
    verdict = verdict_head(calls)
    assert verdict.startswith("HEADS COMPETE")
    assert "unique leading bytes" in verdict


def test_heads_do_not_compete_when_a_busier_neighbour_costs_the_arm_nothing():
    calls = [
        *arm_calls("C1-alone", 0.80, block="C"),
        *arm_calls("C2-contended", 0.78, block="C"),
    ]
    assert verdict_head(calls).startswith("HEADS DO NOT COMPETE")


def test_when_every_block_reads_dead_the_overall_verdict_says_so_out_loud():
    """The outcome that ends the request-side search and leaves only explicit caching."""
    calls = [
        *block_a_calls({size: 0.70 for size in SIZES_A}),
        *block_b_calls({"x4": 0.70, "x24": 0.70, "x96": 0.70, "slow": 0.70}),
        *arm_calls("C1-alone", 0.70, block="C"),
        *arm_calls("C2-contended", 0.70, block="C"),
    ]
    assert verdict_overall(calls).startswith("NOTHING MOVES ANYTHING")


def test_one_variable_moving_is_enough_to_stop_the_overall_verdict_giving_up():
    calls = [
        *block_a_calls({2_048: 0.10, 4_096: 0.40, 8_192: 0.60, 16_384: 0.88, 32_768: 0.92}),
        *block_b_calls({"x4": 0.70, "x24": 0.70, "x96": 0.70, "slow": 0.70}),
        *arm_calls("C1-alone", 0.70, block="C"),
        *arm_calls("C2-contended", 0.70, block="C"),
    ]
    assert verdict_overall(calls).startswith("At least one variable moved")
