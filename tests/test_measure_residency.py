"""The residency measurement, checked without a network.

`measure_residency.py` reads BigQuery, but the shape of what it reports is decided by four
pure functions: `observe`, which turns a stream of turns into idle times, `fit_curve`, which
turns those into knots, and `band_boundary` and `fit_bands`, which decide whether the run
holds one population of prefixes or two and fit each on its own. These hold all four.
"""

from __future__ import annotations

import datetime as dt

from measure_residency import (
    BAND_MIN_OPENINGS,
    MIN_GAP_SPREAD,
    Observation,
    band_boundary,
    fit_bands,
    fit_curve,
    observe,
)

EPOCH = dt.datetime(2026, 9, 4, tzinfo=dt.UTC)


def turn(prefix: str, at: float, ordinal: int, prompt: int = 1_000, cached: int = 900) -> dict:
    return {
        "prefix_key": prefix,
        "timestamp": EPOCH + dt.timedelta(seconds=at),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "ordinal": ordinal,
    }


def seen(gap: float, share: float, prefix_tokens: int = 8_000) -> Observation:
    return Observation(gap, share, prefix_tokens)


# --- observe -----------------------------------------------------------------------


def test_an_opening_is_timed_against_its_own_prefix_and_no_other():
    rows = [
        turn("a", 0.0, 1),
        turn("b", 10.0, 1),  # another prefix in between, which must not reset the clock
        turn("a", 30.0, 1),
    ]
    openings, _ = observe(rows)
    assert [o.gap_s for o in openings] == [30.0]


def test_the_first_sighting_of_a_prefix_is_not_an_observation():
    """Nothing sent it before, so it says nothing about how long a cache survives."""
    openings, _ = observe([turn("a", 0.0, 1)])
    assert openings == []


def test_a_session_warms_its_own_prefix_for_the_next_one():
    """Later turns are not observations, but they are still requests and still refresh it."""
    rows = [turn("a", 0.0, 1), turn("a", 100.0, 2), turn("a", 105.0, 1)]
    openings, carry = observe(rows)
    assert [o.gap_s for o in openings] == [5.0]  # from turn two, not from the session start
    assert carry == [0.9]


def test_the_share_is_what_was_cached_against_what_was_sent():
    openings, _ = observe([turn("a", 0.0, 1), turn("a", 1.0, 1, prompt=800, cached=200)])
    assert openings[0].share == 0.25


def test_a_prefix_is_sized_by_the_smallest_prompt_ever_sent_on_it():
    """History only grows, so the smallest prompt is the closest thing to the prefix alone."""
    rows = [
        turn("a", 0.0, 1, prompt=8_100),
        turn("a", 1.0, 2, prompt=9_400),  # the same prefix, carrying a turn of history
        turn("a", 2.0, 1, prompt=8_200),
    ]
    openings, _ = observe(rows)
    assert [o.prefix_tokens for o in openings] == [8_100]


def test_a_prefix_is_sized_from_turns_that_are_not_openings():
    """A menu that opened one session is still sized from all the traffic that mentions it."""
    rows = [turn("a", 0.0, 2, prompt=8_050), turn("a", 1.0, 1, prompt=8_900)]
    openings, _ = observe(rows)
    assert [o.prefix_tokens for o in openings] == [8_050]


def test_two_prefixes_are_sized_apart():
    rows = [
        turn("menu", 0.0, 1, prompt=8_000),
        turn("full", 1.0, 1, prompt=16_000),
        turn("menu", 2.0, 1, prompt=8_000),
        turn("full", 3.0, 1, prompt=16_000),
    ]
    openings, _ = observe(rows)
    assert sorted(o.prefix_tokens for o in openings) == [8_000, 16_000]


def test_rows_out_of_order_are_still_read_in_order():
    rows = [turn("a", 50.0, 1), turn("a", 0.0, 1)]
    openings, _ = observe(rows)
    assert [o.gap_s for o in openings] == [50.0]


# --- fit_curve ---------------------------------------------------------------------


def test_a_curve_falls_even_when_the_bins_do_not():
    """A bin that reads higher than the one before it is noise, so it is pooled away."""
    rising = [seen(gap, share) for gap, share in [(1, 0.9), (10, 0.2), (100, 0.8)]]
    curve = fit_curve(rising, bins=3)
    shares = [share for _, share in curve]
    assert shares == sorted(shares, reverse=True)


def test_knots_are_never_closer_together_than_the_bins_can_resolve():
    clustered = [seen(1.0 + i * 0.01, 1.0 - i / 100) for i in range(60)]
    curve = fit_curve(clustered, bins=12)
    gaps = [gap for gap, _ in curve]
    assert all(b >= a * MIN_GAP_SPREAD for a, b in zip(gaps, gaps[1:], strict=False))


def test_nothing_measured_fits_nothing():
    assert fit_curve([]) == []


def test_observations_that_do_not_decay_fit_a_flat_curve():
    """No decay to find, so the curve says so instead of inventing a slope."""
    flat = [seen(float(i + 1), 0.5) for i in range(60)]
    assert {share for _, share in fit_curve(flat, bins=12)} == {0.5}


# --- band_boundary -------------------------------------------------------------------
#
# The run this was built against: 24 menus between 7,849 and 8,296 tokens, and one full list
# at 16,214. The menus cache about 0.58 on arrival and the full list about 0.88, so pooling
# them fits a slope that is really a difference between two populations.

MENU_SIZES = (7_849, 7_997, 8_072, 8_191, 8_296)


def two_populations(small: int = 60, large: int = 60) -> list[Observation]:
    return [
        *[seen(float(i + 1), 0.58, MENU_SIZES[i % len(MENU_SIZES)]) for i in range(small)],
        *[seen(float(i + 1), 0.88, 16_214) for i in range(large)],
    ]


def test_a_pool_of_menus_and_a_full_list_splits_between_them():
    boundary = band_boundary(two_populations())
    assert boundary is not None
    assert 8_296 < boundary < 16_214


def test_prefixes_that_are_all_about_one_size_are_one_band():
    """A menu pool cut to a single width has no size contrast to fit, so it gets one curve."""
    assert band_boundary(two_populations(large=0)) is None


def test_a_second_population_too_small_to_fit_is_not_split_off():
    """Twelve knots over a handful of openings is noise with a shape, so it is refused."""
    thin = two_populations(large=BAND_MIN_OPENINGS - 1)
    assert band_boundary(thin) is None


def test_the_widest_ratio_wins_and_not_the_widest_token_distance():
    """2k to 8k is a bigger break than 8k to 16k, even though it spans fewer tokens."""
    observations = [
        *[seen(float(i + 1), 0.2, 2_048) for i in range(40)],
        *[seen(float(i + 1), 0.6, 8_192) for i in range(40)],
        *[seen(float(i + 1), 0.9, 16_384) for i in range(40)],
    ]
    boundary = band_boundary(observations)
    assert boundary is not None
    assert 2_048 < boundary < 8_192


def test_nothing_measured_has_no_boundary():
    assert band_boundary([]) is None


# --- fit_bands -----------------------------------------------------------------------


def test_each_band_is_fitted_on_its_own_prefixes_and_no_others():
    bands = fit_bands(two_populations(), bins=4)
    assert [b.openings for b in bands] == [60, 60]
    assert (bands[0].min_tokens, bands[0].max_tokens) == (7_849, 8_296)
    assert (bands[1].min_tokens, bands[1].max_tokens) == (16_214, 16_214)
    assert bands[0].mean_share == 0.58
    assert bands[1].mean_share == 0.88


def test_the_small_band_comes_first_because_that_is_what_a_router_ships():
    bands = fit_bands(two_populations(), bins=4)
    assert bands[0].max_tokens < bands[1].min_tokens


def test_one_population_fits_one_band_covering_every_prefix():
    bands = fit_bands(two_populations(large=0), bins=4)
    assert len(bands) == 1
    assert bands[0].openings == 60


def test_splitting_removes_a_slope_the_pooled_fit_invents():
    """The mixture artifact, in miniature: each band is flat, and pooling them is not.

    The big prefix supplies every short-gap opening and the small one every long-gap
    opening, so a pooled fit reads a decay from 0.9 to 0.5 that neither population has.
    """
    mixed = [
        *[seen(float(i + 1), 0.9, 16_214) for i in range(60)],
        *[seen(1_000.0 + i, 0.5, 8_000) for i in range(60)],
    ]
    pooled = [share for _, share in fit_curve(mixed, bins=6)]
    assert max(pooled) - min(pooled) > 0.3

    bands = fit_bands(mixed, bins=6)
    assert [{share for _, share in band.curve} for band in bands] == [{0.5}, {0.9}]


def test_nothing_measured_fits_no_bands():
    assert fit_bands([]) == []
