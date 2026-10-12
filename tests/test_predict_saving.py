"""The arithmetic behind the pre-A/B prediction, checked without a network.

`predict_saving.py` reads BigQuery, but everything it decides is decided by the four pure
functions below. They are what these tests hold. The numbers are the reference agent's, so
a failure here means the prediction moved rather than that a fixture did.
"""

from __future__ import annotations

import pytest
from predict_saving import (
    CACHED_RATE,
    MEASURED_CURVE,
    ORIGINAL_DOMAINS,
    ResidencyCurve,
    SessionFacts,
    SessionShape,
    Traffic,
    predict,
    replay_warmth,
    resolve_prunable,
    routes,
    session_billed,
    surface_mismatch,
    surface_names,
)

# The reference agent, measured. 100 declarations bill 14,324 tokens against a 1,777-token
# system instruction; a 43-tool menu is roughly 6,400 of those declaration tokens.
FULL_PREFIX = 16_101
MENU_PREFIX = 8_177
OPENING = 130
GROWTH = 512.0


# --- session_billed ---------------------------------------------------------------


def shape(turns: int, prefix: int = FULL_PREFIX) -> SessionShape:
    return SessionShape(turns, prefix, OPENING, GROWTH)


def test_a_single_cold_turn_bills_everything():
    assert (
        session_billed(SessionShape(1, FULL_PREFIX, OPENING, 0), warm=0.0) == FULL_PREFIX + OPENING
    )


def test_a_single_warm_turn_bills_the_prefix_at_a_tenth():
    billed = session_billed(SessionShape(1, FULL_PREFIX, OPENING, 0), warm=1.0)
    assert billed == pytest.approx(CACHED_RATE * FULL_PREFIX + OPENING)


def test_the_second_turn_carries_the_first():
    """The whole of turn one is resident when turn two arrives: the prefix is locked."""
    turn_one = FULL_PREFIX + OPENING
    turn_two = turn_one + GROWTH - (1 - CACHED_RATE) * turn_one
    assert session_billed(shape(2), warm=0.0) == pytest.approx(turn_one + turn_two)


def test_warmth_only_ever_helps_and_only_on_turn_one():
    """However long the session, opening warm is worth the same fixed amount."""
    gain = session_billed(shape(6), warm=0.0) - session_billed(shape(6), warm=1.0)
    assert gain == pytest.approx((1 - CACHED_RATE) * FULL_PREFIX)


def test_the_prune_is_worth_less_per_turn_the_longer_the_session_runs():
    """Turn one pays full rate for the declarations; every later turn pays a tenth."""

    def saving_per_turn(turns: int) -> float:
        full = session_billed(shape(turns), warm=0.0)
        pruned = session_billed(shape(turns, MENU_PREFIX), warm=0.0)
        return (full - pruned) / turns

    assert saving_per_turn(1) > saving_per_turn(20) > 0


def test_an_empty_session_costs_nothing():
    assert session_billed(shape(0), warm=1.0) == 0.0


# --- the residency curve ------------------------------------------------------------

CURVE = ResidencyCurve(((10.0, 0.9), (100.0, 0.5), (1_000.0, 0.4)))


def test_the_curve_is_flat_outside_the_range_it_was_measured_over():
    """Nothing was measured past either end, so the ends are what the ends say."""
    assert CURVE.warmth(0.0) == 0.9
    assert CURVE.warmth(1.0) == 0.9
    assert CURVE.warmth(10_000.0) == 0.4 == CURVE.floor


def test_between_knots_the_curve_moves_in_log_gap():
    """31.6 s is halfway from 10 to 100 on a log scale, so it is halfway from 0.9 to 0.5."""
    assert CURVE.warmth(10.0**1.5) == pytest.approx(0.7)
    assert CURVE.warmth(100.0) == pytest.approx(0.5)


def test_an_empty_curve_never_promises_a_cache():
    assert ResidencyCurve(()).warmth(0.0) == 0.0


def test_the_shipped_curve_falls_and_never_reaches_zero():
    """What `measure_residency.py` found: a decay onto a floor, not a cliff at a window."""
    probabilities = [p for _, p in MEASURED_CURVE.knots]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0.4 < MEASURED_CURVE.floor < MEASURED_CURVE.warmth(0.0) < 1.0


# --- replaying the arrivals --------------------------------------------------------

MENU_A = frozenset({"a", "b"})
MENU_B = frozenset({"c", "d"})


def test_an_opening_is_warmer_the_sooner_it_follows_the_same_prefix():
    warm = replay_warmth(
        turns=[(0.0, MENU_A), (100.0, MENU_A)],
        openings=[("first", 0.0, MENU_A), ("soon", 10.0, MENU_A), ("late", 1_100.0, MENU_A)],
        curve=CURVE,
    )
    assert warm["first"] == 0.0  # nothing sent this prefix before, so nothing cached it
    assert warm["soon"] == 0.9
    assert warm["late"] == 0.4


def test_another_menu_running_does_not_warm_this_one():
    warm = replay_warmth(turns=[(0.0, MENU_B)], openings=[("s", 1.0, MENU_A)], curve=CURVE)
    assert warm["s"] == 0.0


# --- predict ------------------------------------------------------------------------
#
# A synthetic run: sessions of three turns, one opening every `period` seconds. These price
# against the shipped curve rather than the toy one above, because half of what `predict`
# does is decided by the curve's shape and the shipped shape is the one that has to hold.


def run(sessions: int, period: float, menu_for) -> tuple[Traffic, dict]:
    facts = []
    menus = {}
    for i in range(sessions):
        sid = f"s{i:04d}"
        start = i * period
        facts.append(SessionFacts(sid, (start, start + 1.0, start + 2.0), OPENING, GROWTH, 1, 0))
        menus[sid] = menu_for(i)
    return Traffic(tuple(facts), FULL_PREFIX), menus


def priced(traffic: Traffic, menus: dict, **kwargs):
    """`predict` with the prefix table taken from whatever menus the fixture handed out."""
    return predict(
        traffic,
        menus,
        full_prefix_tokens=FULL_PREFIX,
        prefix_tokens={menu: MENU_PREFIX for menu in menus.values() if menu is not None},
        curve=MEASURED_CURVE,
        **kwargs,
    )


def test_one_menu_beats_two_at_the_same_traffic():
    """Not because the menus differ. Because splitting the traffic splits the warming."""
    one = priced(*run(60, 30.0, lambda i: MENU_A), route_fraction=1.0)
    two = priced(*run(60, 30.0, lambda i: MENU_A if i % 2 else MENU_B), route_fraction=1.0)
    assert one.rollout_g1 < two.rollout_g1 < 0
    assert one.warm_openings > two.warm_openings


def test_a_busier_agent_keeps_more_of_its_menus_warm():
    """More traffic through a fixed pool is less idle time per menu. No exceptions."""
    traffic, menus = run(60, 80.0, lambda i: MENU_A if i % 2 else MENU_B)
    warmth = [
        priced(traffic, menus, route_fraction=1.0, rate_multiplier=m).warm_openings
        for m in (0.25, 1.0, 4.0, 16.0)
    ]
    assert warmth == sorted(warmth)


def test_traffic_favours_the_router_once_the_full_list_has_no_room_left():
    """Extra traffic warms the menus and the full list alike, so it is not a gain in itself.

    It becomes one when the full list is already saturated and only the menus can improve.
    An unrouted production agent sits exactly there, which is why more traffic helps the
    router in practice while proving nothing in principle.
    """
    traffic, menus = run(240, 2.0, lambda i: frozenset({f"m{i % 24}"}))
    saving = [
        priced(traffic, menus, route_fraction=1.0, rate_multiplier=m).rollout_g1
        for m in (1.0, 2.0, 4.0, 8.0, 16.0)
    ]
    assert saving == sorted(saving, reverse=True)
    assert saving[-1] < saving[0] < 0


def test_an_ab_is_not_the_rollout_it_sits_inside():
    """Always true: the rollout is diluted by the half that is not routed and the A/B is not.

    Which of the two is larger is not a law, so nothing here asserts one. What the script
    reports is both, because a reader who assumes either number rescales into the other
    will be wrong by more than the effect they are chasing.
    """
    traffic, menus = run(80, 15.0, lambda i: MENU_A if i % 2 else MENU_B)
    half = priced(traffic, menus, route_fraction=0.5)
    assert half.ab_g1 is not None
    assert half.ab_g1 < half.rollout_g1 < 0
    assert priced(traffic, menus, route_fraction=1.0).ab_g1 is None  # no holdout arm left


def test_a_half_sized_ab_understates_the_rollout_across_a_realistic_pool():
    """Two dozen menus and a saturated full list, which is where a real agent starts.

    The A/B splits the warming on both sides, so it reads short of the full rollout it
    exists to justify. Worth knowing before reading a marginal experiment as a failure.
    """
    traffic, menus = run(240, 2.0, lambda i: frozenset({f"m{i % 24}"}))
    rollout = priced(traffic, menus, route_fraction=1.0).rollout_g1
    half = priced(traffic, menus, route_fraction=0.5).ab_g1
    assert half is not None
    assert rollout < half < 0


def test_routing_nothing_predicts_nothing():
    traffic, menus = run(20, 30.0, lambda i: MENU_A)
    prediction = priced(traffic, menus, route_fraction=0.0)
    assert prediction.rollout_g1 == pytest.approx(0.0)
    assert prediction.routed_sessions == 0


def test_a_declined_session_ships_the_full_list_and_warms_it():
    traffic, menus = run(40, 30.0, lambda i: None if i % 2 else MENU_A)
    prediction = priced(traffic, menus, route_fraction=1.0)
    assert prediction.routed_sessions == 20
    assert prediction.full_warm_openings > 0.0


def test_the_split_does_not_move_when_the_sweep_does():
    """Same session, same arm, at every route fraction. A sweep must compare like with like."""
    assert routes("s0001", 1.0) and not routes("s0001", 0.0)
    routed_at_quarter = {f"s{i}" for i in range(500) if routes(f"s{i}", 0.25)}
    routed_at_half = {f"s{i}" for i in range(500) if routes(f"s{i}", 0.5)}
    assert routed_at_quarter < routed_at_half
    assert 100 < len(routed_at_quarter) < 150


# --- the residency bands ----------------------------------------------------------
#
# Added when `measure_residency.py` learned to fit each prefix size on its own. The old
# single fit was a mixture: the full list supplied nearly every short-gap observation and
# the menus nearly every long-gap one, so its slope read as decay when it was really the
# gap between two prefixes that never decayed into each other. These tests hold the seam,
# because a caller that silently prices a menu on the full list's curve overstates the
# saving by seventeen points and nothing else would notice.


def test_a_prefix_at_or_above_the_boundary_reads_the_large_band():
    assert MEASURED_CURVE.band(MEASURED_CURVE.band_tokens) == MEASURED_CURVE.large_knots
    assert MEASURED_CURVE.band(MEASURED_CURVE.band_tokens - 1) == MEASURED_CURVE.knots


def test_a_caller_that_does_not_know_the_size_gets_the_menu_band():
    """The prefix a routed session actually ships, which is the conservative reading."""
    assert MEASURED_CURVE.band(None) == MEASURED_CURVE.knots
    assert MEASURED_CURVE.warmth(30.0) == MEASURED_CURVE.warmth(30.0, MENU_PREFIX)


def test_the_full_list_opens_warmer_than_a_menu_at_the_same_idle_time():
    """The deficit the banding exists to expose: 17 points, before any decay starts."""
    menu = MEASURED_CURVE.warmth(2.4, MENU_PREFIX)
    full = MEASURED_CURVE.warmth(2.4, FULL_PREFIX)
    assert full - menu == pytest.approx(0.17, abs=0.02)


def test_one_measured_band_applies_at_every_size():
    """An agent whose prefixes are all one size. Empty `large_knots` must not mean zero."""
    single = ResidencyCurve(knots=((1.0, 0.8), (100.0, 0.5)))
    assert single.warmth(1.0, 99_999) == 0.8
    assert single.band(99_999) == single.knots


def test_a_curve_file_written_before_banding_still_reads(tmp_path):
    from predict_saving import load_curve

    old = tmp_path / "old.json"
    old.write_text('{"curve": [[1.0, 0.8], [100.0, 0.5]]}')
    curve = load_curve(str(old))
    assert curve.knots == ((1.0, 0.8), (100.0, 0.5))
    assert curve.large_knots == ()


def test_a_banded_curve_file_reads_both_bands_and_the_boundary(tmp_path):
    from predict_saving import load_curve

    new = tmp_path / "new.json"
    new.write_text(
        '{"band_tokens": 12000, "bands": ['
        '{"curve": [[1.0, 0.7], [100.0, 0.4]]},'
        '{"curve": [[1.0, 0.9], [100.0, 0.85]]}]}'
    )
    curve = load_curve(str(new))
    assert curve.band_tokens == 12_000
    assert curve.warmth(1.0, 11_999) == 0.7
    assert curve.warmth(1.0, 12_000) == 0.9


def test_two_menus_of_different_widths_price_differently_at_the_same_idle_time():
    """`sizes` is the only thing telling `replay_warmth` which band a menu belongs in."""
    turns = [(0.0, "wide"), (0.0, "narrow"), (10.0, "wide"), (10.0, "narrow")]
    openings = [("wide-session", 10.0, "wide"), ("narrow-session", 10.0, "narrow")]
    sizes = {"wide": FULL_PREFIX, "narrow": MENU_PREFIX}
    warm = replay_warmth(turns, openings, MEASURED_CURVE, sizes=sizes)
    assert warm["wide-session"] > warm["narrow-session"]

    blind = replay_warmth(turns, openings, MEASURED_CURVE)
    assert blind["wide-session"] == pytest.approx(blind["narrow-session"])


# --- which surface, and which setting ------------------------------------------------


def test_the_original_domains_are_the_twenty_five_tool_surface():
    """The named domains have to add up to the agent that ran, or the comparison is empty."""
    assert len(surface_names(",".join(ORIGINAL_DOMAINS))) == 25
    assert surface_names(None) is None


def test_a_domain_nobody_declares_is_refused_rather_than_priced_as_nothing():
    with pytest.raises(SystemExit):
        surface_names("orders,cutlery")


def test_classless_is_every_live_tool_the_artifact_has_no_class_for():
    live = frozenset({"a", "b", "c"})
    assert resolve_prunable("classless", live, frozenset({"a", "ABSTAIN"})) == {"b", "c"}


def test_naming_tools_by_hand_prices_exactly_those_and_the_default_prices_none():
    live = frozenset({"a", "b", "c"})
    assert resolve_prunable("b, c", live, frozenset()) == {"b", "c"}
    assert resolve_prunable(None, live, frozenset()) == frozenset()


def test_a_tool_the_agent_does_not_declare_is_refused():
    """Silently dropping it would report a saving for a setting nobody could ship."""
    with pytest.raises(SystemExit):
        resolve_prunable("b,gone", frozenset({"a", "b"}), frozenset())


# --- which surface the traffic ran on -------------------------------------------------
#
# `opening_tokens` is the turn-one prompt minus the prefix being priced, so traffic from a
# smaller surface makes it negative and every number after it nonsense. This is the check
# that caught corpus-20260831 being pooled with corpus-20260904, whose openings differ by
# 14,000 tokens because the surface grew between them.


def facts_with(opening: int, count: int = 3) -> Traffic:
    return Traffic(
        tuple(SessionFacts(f"s{i}", (float(i),), opening, GROWTH, 1, 0) for i in range(count)),
        FULL_PREFIX,
    )


def test_traffic_from_the_surface_being_priced_passes():
    assert surface_mismatch(facts_with(OPENING), "corpus-20260904", FULL_PREFIX) is None


def test_traffic_from_a_smaller_surface_is_refused():
    complaint = surface_mismatch(facts_with(-14_175), "corpus", FULL_PREFIX)
    assert complaint is not None
    assert "16,101" in complaint
    assert "corpus" in complaint


def test_one_short_question_is_not_a_different_surface():
    """The sum, not any single session: a two-word opening is a two-word opening."""
    mixed = Traffic(
        (
            SessionFacts("s0", (0.0,), -40, GROWTH, 1, 0),
            SessionFacts("s1", (1.0,), 200, GROWTH, 1, 0),
        ),
        FULL_PREFIX,
    )
    assert surface_mismatch(mixed, "corpus-20260904", FULL_PREFIX) is None


def test_no_sessions_is_not_a_mismatch():
    """Empty traffic is a different complaint, and the caller makes it first."""
    assert surface_mismatch(Traffic((), FULL_PREFIX), "corpus-x", FULL_PREFIX) is None
