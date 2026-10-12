"""Tests for canonical menus: choosing them offline, and snapping to one at serve time.

`menus` is pure, so everything here is arithmetic on lists and frozensets. Two properties
carry the whole design and each has its own section below.

`choose` has to cover as many real tool calls as it can with a handful of menus, because
every call it misses is a turn the model cannot do its job.

`rank` has to be deterministic in the strongest sense: the same prediction vector must
produce the same menu on every replica, forever. A tie broken by dict order would give two
servers two different declaration lists for the same session, which is the exact cost
menus exist to remove (DESIGN.md 18.16).
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import NamedTuple

from adk_smart_router.menus import bucket, choose, pad, popularity, rank


class Pred(NamedTuple):
    """Structural match for menus.PredictionLike."""

    label: str
    probability: float


def menu(*names: str) -> frozenset[str]:
    return frozenset(names)


# --- popularity ---------------------------------------------------------------


def test_popularity_counts_every_call_not_every_session():
    sessions = [["a", "a", "a"], ["b"], ["c"]]
    assert popularity(sessions)[0] == "a"


def test_popularity_breaks_ties_by_name():
    assert popularity([["z", "a", "m"]]) == ["a", "m", "z"]


def test_popularity_of_nothing_is_nothing():
    assert popularity([]) == []
    assert popularity([[], []]) == []


# --- pad ----------------------------------------------------------------------


def test_pad_fills_spare_slots_from_the_popular_list():
    assert pad(["a"], ["x", "y", "z"], 3) == ["a", "x", "y"]


def test_pad_never_repeats_a_tool_it_already_has():
    assert pad(["a", "b"], ["b", "c"], 3) == ["a", "b", "c"]


def test_pad_trims_when_there_are_already_too_many():
    assert pad(["a", "b", "c", "d"], ["x"], 2) == ["a", "b"]


def test_pad_drops_duplicates_in_its_input():
    assert pad(["a", "a", "b"], ["c"], 3) == ["a", "b", "c"]


def test_pad_stays_short_when_there_is_nothing_left_to_add():
    assert pad(["a"], ["a"], 5) == ["a"]


# --- choose -------------------------------------------------------------------


def test_one_menu_for_a_corpus_that_only_ever_wants_one_thing():
    sessions = [["a", "b"], ["a", "b"], ["b", "a"]]
    assert choose(sessions, size=2, count=4) == (menu("a", "b"),)


def test_two_populations_get_one_menu_each():
    """Half the sessions do billing, half do shipping, and no menu of three fits both.

    This is the case a single static menu loses: whichever one it picks, it misses the
    other half's calls entirely (DESIGN.md 18.17).
    """
    billing = [["refund", "invoice", "credit"]] * 5
    shipping = [["track", "reship", "label"]] * 5
    chosen = choose(billing + shipping, size=3, count=2)
    assert set(chosen) == {menu("refund", "invoice", "credit"), menu("track", "reship", "label")}


def test_the_first_menu_chosen_is_the_one_that_covers_most():
    """Greedy, so the biggest population is served first. A budget of one menu has to
    spend it on the majority, and a budget of two adds the minority afterwards."""
    common = [["a", "b"]] * 9
    rare = [["y", "z"]] * 2
    assert choose(common + rare, size=2, count=1) == (menu("a", "b"),)


def test_choosing_stops_once_another_menu_would_add_nothing():
    """Asked for eight menus over a corpus that only contains two shapes, it returns two.

    Returning eight would mean six menus nobody snaps to, each a declaration list the
    cache would have to hold for no coverage in return.
    """
    sessions = [["a", "b"]] * 5 + [["y", "z"]] * 5
    assert len(choose(sessions, size=2, count=8)) == 2


def test_a_menu_is_padded_out_to_its_full_size():
    """A session that called one tool must not mint a one-tool menu. The spare slots are
    free, so they go to whatever else the corpus asks for most."""
    sessions = [["a"]] + [["b", "c"]] * 3
    (first,) = choose(sessions, size=3, count=1)
    assert len(first) == 3


def test_repeated_calls_in_one_session_count_once():
    """`a` is called five times in one session and `b` once in each of two. `b` wins,
    because the router declares a menu per session and a session it gets wrong is one
    mistake however many times the missing tool was wanted (DESIGN.md 19.3).
    """
    sessions = [["a", "a", "a", "a", "a"], ["b"], ["b"]]
    assert choose(sessions, size=1, count=1) == (menu("b"),)


def test_a_session_wanting_many_tools_does_not_outvote_several_wanting_one():
    """Set recall divides by the size of the true set, so every session has one vote. Under
    raw call counting the five-tool session would take the only menu."""
    sessions = [["a", "b", "c", "d", "e"], ["z"], ["z"], ["z"], ["z"]]
    assert choose(sessions, size=1, count=1) == (menu("z"),)


def test_choosing_from_nothing_yields_nothing():
    assert choose([], size=5, count=3) == ()
    assert choose([[], []], size=5, count=3) == ()


def test_a_nonsense_budget_yields_nothing_rather_than_raising():
    """Called at training time from a script, so a bad argument must produce an empty
    artifact the caller can notice, not a traceback halfway through publishing."""
    sessions = [["a", "b"]]
    assert choose(sessions, size=0, count=3) == ()
    assert choose(sessions, size=3, count=0) == ()


def test_an_exact_tie_falls_to_the_alphabetically_first_menu():
    """Two shapes, equally popular, and room for only one menu. Candidates are built from
    a set, whose iteration order is not guaranteed between processes, so without a
    tie-break two machines could publish two different artifacts from the same corpus.
    """
    sessions = [["y", "z"]] * 3 + [["a", "b"]] * 3
    assert choose(sessions, size=2, count=1) == (menu("a", "b"),)
    assert choose(list(reversed(sessions)), size=2, count=1) == (menu("a", "b"),)


# --- bucket -------------------------------------------------------------------


def test_every_tool_the_corpus_called_is_in_some_bucket():
    """The property that makes merging worth doing at all.

    Padding cannot promise this: a candidate is one session's set plus popular filler, and
    a rare tool that no chosen candidate happens to carry is unreachable by any menu in the
    pool. That is the collapse DESIGN.md 19.42 measured, where 264 slots held 91 tools.
    """
    sessions = [["a", "b"], ["c"], ["d", "e"], ["f"], ["g", "h"], ["rare"]]
    buckets = bucket(sessions, size=3)
    assert set().union(*buckets) == {t for calls in sessions for t in calls}


def test_no_bucket_is_wider_than_a_menu():
    sessions = [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["e", "f"]]
    assert all(len(found) <= 3 for found in bucket(sessions, size=3))


def test_sets_that_co_occur_land_together_and_ones_that_do_not_stay_apart():
    """The merge rule itself: overlap decides, and only when the union still fits."""
    sessions = [["a", "b"], ["a", "b"], ["b", "c"], ["x", "y"], ["x", "y"]]
    buckets = {tuple(sorted(found)) for found in bucket(sessions, size=3)}
    assert ("a", "b", "c") in buckets  # b is shared, and three fits
    assert ("x", "y") in buckets  # nothing in common with the others


def test_a_bucket_does_not_grow_past_the_menu_size_to_absorb_an_overlap():
    """`a b` and `c d` both overlap `b c`, but only one union fits in three."""
    sessions = [["b", "c"], ["b", "c"], ["a", "b"], ["c", "d"]]
    buckets = {tuple(sorted(found)) for found in bucket(sessions, size=3)}
    assert ("a", "b", "c") in buckets
    assert ("c", "d") in buckets


def test_a_session_wanting_more_tools_than_a_menu_holds_keeps_its_popular_ones():
    """No menu can serve it, but the tools it shares with sessions that can must survive."""
    sessions = [["a", "b", "c", "d"], ["a"], ["a"], ["b"], ["b"]]
    buckets = bucket(sessions, size=2)
    assert all(len(found) <= 2 for found in buckets)
    assert menu("a", "b") in buckets  # the two the rest of the corpus also wants


def test_the_buckets_do_not_depend_on_the_order_the_sessions_arrive_in():
    """Same reason `choose` has a tie-break: two replicas must publish the same artifact."""
    sessions = [["a", "b"], ["b", "c"], ["d", "e"], ["e", "f"], ["a", "c"]]
    assert bucket(sessions, size=3) == bucket(list(reversed(sessions)), size=3)


def test_bucketing_nothing_yields_nothing():
    assert bucket([], size=3) == ()
    assert bucket([["a"]], size=0) == ()


def test_choose_uses_the_candidates_it_is_given_and_nothing_else():
    """The seam the measurement runs through. If `choose` ever padded on top of what it was
    handed, the merged arm would silently become the padded one and the table would show a
    difference that was not there.
    """
    sessions = [["a", "b"], ["c", "d"]]
    offered = [menu("a", "b")]
    assert choose(sessions, size=4, count=2, candidates=offered) == (menu("a", "b"),)


def test_a_candidate_wider_than_the_menu_is_trimmed_rather_than_shipped():
    """`choose` promises menus of `size`. A caller's candidates must not be able to break it."""
    sessions = [["a", "b", "c"]]
    (chosen,) = choose(sessions, size=2, count=1, candidates=[menu("a", "b", "c")])
    assert len(chosen) == 2


# --- rank ---------------------------------------------------------------------


def test_the_menu_holding_the_most_probability_mass_wins():
    preds = [Pred("a", 0.7), Pred("b", 0.2), Pred("c", 0.1)]
    menus = [menu("a", "z"), menu("b", "c")]
    assert rank(preds, menus) == menu("a", "z")


def test_mass_is_summed_across_the_whole_menu_not_just_the_top_label():
    """`a` alone outweighs `b`, but the menu holding `b` and `c` holds more together."""
    preds = [Pred("a", 0.5), Pred("b", 0.3), Pred("c", 0.25)]
    menus = [menu("a", "z"), menu("b", "c")]
    assert rank(preds, menus) == menu("b", "c")


def test_tools_the_model_has_no_opinion_about_are_worth_nothing():
    preds = [Pred("a", 0.6)]
    menus = [menu("a"), menu("x", "y", "z", "w")]
    assert rank(preds, menus) == menu("a")


def test_a_tie_is_broken_by_membership_so_every_replica_agrees():
    preds = [Pred("a", 0.5), Pred("b", 0.5)]
    menus = [menu("b", "y"), menu("a", "z")]
    assert rank(preds, menus) == menu("a", "z")
    assert rank(preds, list(reversed(menus))) == menu("a", "z")


def test_with_no_menus_there_is_no_menu_to_give():
    assert rank([Pred("a", 0.9)], []) is None


def test_with_no_predictions_the_first_menu_by_name_is_the_answer():
    """Every menu scores zero, so the tie-break decides. Still a menu rather than None:
    an empty prediction vector is the caller's problem, and returning something stable
    keeps the prefix shared."""
    menus = [menu("b"), menu("a")]
    assert rank([], menus) == menu("a")


# --- choose is fast, and fast has to mean the same thing -----------------------


def choose_plainly(sessions, size, count):
    """`choose` written for reading rather than for speed: rescore everything every round.

    This is the specification, which is why it lives here and not in the module. The greedy
    step is one line and obviously right, and the property test below asserts the shipped
    version agrees with it on every corpus it is given.

    Scores are `Fraction`, not `float`, for the reason `_choose_greedily` gives: set recall
    summed in floating point is not associative, so the total depends on the row order and
    the alphabetical tie-break never gets to run. `Fraction` is far too slow to ship and
    exactly right for a specification.
    """
    if size < 1 or count < 1:
        return ()
    ranked = popularity(sessions)
    if not ranked:
        return ()

    def total(menus):
        if not menus:
            return Fraction(0)
        return sum(Fraction(max(len(tools & m) for m in menus), len(tools)) for tools in wanted)

    wanted = [frozenset(calls) for calls in sessions if calls]
    candidates = {frozenset(pad(dict.fromkeys(calls), ranked, size)) for calls in sessions if calls}
    chosen = []
    while len(chosen) < count and candidates:
        best = min(candidates, key=lambda m: (-total([*chosen, m]), tuple(sorted(m))))
        if chosen and total([*chosen, best]) == total(chosen):
            break
        chosen.append(best)
        candidates.discard(best)
    return tuple(chosen)


def random_sessions(rng, count, vocabulary, longest):
    tools = [f"t{n:02d}" for n in range(vocabulary)]
    return [rng.sample(tools, rng.randint(1, min(longest, vocabulary))) for _ in range(count)]


def test_the_fast_choose_and_the_plain_one_agree_on_random_corpora():
    """The whole licence for the heap and the carried score. Menus decide which tools an
    agent declares, so a speedup that quietly picks a different pool would change published
    artifacts without changing a documented number anywhere."""
    rng = random.Random(20260905)
    for trial in range(60):
        sessions = random_sessions(rng, rng.randint(1, 40), rng.randint(2, 12), 5)
        size, count = rng.randint(1, 6), rng.randint(1, 6)
        assert choose(sessions, size, count) == choose_plainly(sessions, size, count), (
            f"trial {trial}: size={size} count={count} sessions={sessions}"
        )


def test_the_two_agree_when_every_session_wants_the_same_thing():
    """The degenerate corpus, where every candidate ties and only the order key separates
    them. Ties are where a reordered search is most likely to diverge."""
    sessions = [["a", "b"], ["b", "a"], ["a", "b"]]
    assert choose(sessions, size=2, count=3) == choose_plainly(sessions, size=2, count=3)


def test_the_two_agree_when_there_is_nothing_left_to_improve():
    """One menu covers everybody, so every later round gains exactly zero and both versions
    have to stop rather than pad the pool out to `count`."""
    sessions = [["a"], ["b"], ["a", "b"]]
    assert choose(sessions, size=4, count=5) == choose_plainly(sessions, size=4, count=5)
    assert len(choose(sessions, size=4, count=5)) == 1


def test_the_pool_does_not_depend_on_the_order_the_sessions_arrive_in():
    """The bug the integer score fixes. Sessions come out of BigQuery with no ORDER BY, so
    two reads of one corpus can hand `choose` the same rows in a different sequence. Under a
    floating-point total that was enough to move two exactly-tied menus an ulp apart and pick
    a different pool, which meant two runs over identical data could publish different
    declaration lists. `DESIGN.md` §19.41.
    """
    sessions = random_sessions(random.Random(1), 32, 7, 5)  # a corpus that did exhibit it
    rng = random.Random(99)
    pools = set()
    for _ in range(25):
        shuffled = sessions[:]
        rng.shuffle(shuffled)
        pools.add(choose(shuffled, size=2, count=6))
    assert len(pools) == 1, f"{len(pools)} different pools from one corpus in 25 row orders"
