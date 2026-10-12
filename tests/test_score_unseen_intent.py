"""The unseen-intent re-score's blocking, splitting and arithmetic, checked without a network.

`score_unseen_intent.py` reads BigQuery and a local generation journal, but everything that
decides what it prints is pure: which neighbours a regime lets a session see, how the
neighbours' votes become a menu, how the corpus is cut into opening-length quartiles, and
how a lift is taken against the static baseline. These hold all four, plus the journal
loading, whose whole job is to fail softly on a clone that has never generated a corpus.

The one thing worth stating up front: at one vector per session, "block the session" and
"block this row" are the same set, so the regime table's top row is a leave-one-out score.
That is what makes it comparable with the published number, and the tests below pin it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from score_unseen_intent import (
    REGIMES,
    SCENARIO_REGIME,
    SESSION_REGIME,
    Case,
    Regime,
    Row,
    build_cases,
    fit_pools,
    fold_of,
    load_journal,
    mean_recall,
    normalise,
    quartiles,
    recall,
    regime_line,
    report_short_opening_fallback,
    scorable,
    score_regime,
    static_row,
    unit,
    voted_menu,
    votes,
)

from adk_smart_router.menus import pad, popularity

RANKED = ["a", "b", "c", "d", "e"]


def case(
    session_id: str,
    tools: set[str],
    opening: str = "hello",
    scenario: str = "s1",
    persona: str = "p1",
) -> Case:
    return Case(
        session_id=session_id,
        tools=frozenset(tools),
        opening=opening,
        scenario=scenario,
        persona=persona,
    )


def regime(name: str) -> Regime:
    return next(r for r in REGIMES if r.name == name)


# --- the blocking regimes ---------------------------------------------------------------


def test_the_four_shipped_regimes_group_a_session_by_what_they_claim_to():
    subject = case("s-1", {"a"}, opening="Where is my order?", scenario="s42", persona="p07")
    keys = {r.name: r.key(subject) for r in REGIMES}
    assert keys[SESSION_REGIME] == "s-1"
    assert keys["block the opening"] == "where is my order"
    assert keys[SCENARIO_REGIME] == "s42"
    assert keys["block the persona"] == "p07"


def test_only_the_two_intent_regimes_need_the_journal():
    """Scenario and persona exist nowhere but the journal, so they are the ones that skip."""
    assert {r.name for r in REGIMES if r.from_journal} == {SCENARIO_REGIME, "block the persona"}


def test_a_session_the_journal_does_not_carry_gets_no_scenario_rather_than_a_shared_one():
    """An empty key would group every unknown session together and invent a fake intent."""
    cases, _ = build_cases(
        [_session("s-1", [1.0, 0.0], {"a"})], {"s-1": "hello"}, journal={"other": ("s9", "p9")}
    )
    assert cases[0].scenario == ""
    assert cases[0].persona == ""


def test_blocking_a_group_hides_every_member_of_it_from_the_vote():
    """Two sessions of one scenario, one of another. Blocking the scenario leaves the other."""
    tools = [frozenset({"a"}), frozenset({"a"}), frozenset({"b"})]
    similarity = np.array([1.0, 0.99, 0.5])
    blocked = np.array([True, True, False])
    assert votes(similarity, blocked, tools, k=3) == {"b": 0.5}


def test_blocking_everything_leaves_no_vote_rather_than_a_negative_one():
    """A regime can block the whole corpus on a tiny fixture, and -2.0 must not become a vote."""
    tools = [frozenset({"a"}), frozenset({"b"})]
    assert votes(np.array([1.0, 0.9]), np.array([True, True]), tools, k=2) == {}


def test_a_neighbour_that_called_no_tool_takes_a_slot_and_casts_no_vote():
    """They are a fifth of the corpus and dropping them would let the vote reach further out."""
    tools = [frozenset(), frozenset({"a"})]
    close_quiet = votes(np.array([0.99, 0.9]), np.array([False, False]), tools, k=1)
    assert close_quiet == {}
    assert votes(np.array([0.99, 0.9]), np.array([False, False]), tools, k=2) == {"a": 0.9}


def test_a_vote_is_weighted_by_how_near_the_neighbour_is():
    tools = [frozenset({"a"}), frozenset({"a", "b"})]
    assert votes(np.array([0.8, 0.5]), np.array([False, False]), tools, k=2) == {
        "a": pytest.approx(1.3),
        "b": pytest.approx(0.5),
    }


def test_the_vote_reaches_exactly_k_neighbours_deep():
    tools = [frozenset({"a"}), frozenset({"b"}), frozenset({"c"})]
    similarity = np.array([0.9, 0.8, 0.7])
    unblocked = np.array([False, False, False])
    assert set(votes(similarity, unblocked, tools, k=2)) == {"a", "b"}
    assert set(votes(similarity, unblocked, tools, k=3)) == {"a", "b", "c"}


# --- turning votes into a menu ------------------------------------------------------------


def test_the_menu_is_the_best_supported_tools_and_no_wider_than_the_menu_size():
    weighted = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert voted_menu(weighted, RANKED, size=2) == frozenset({"b", "c"})


def test_tied_tools_fall_to_the_more_popular_one_so_two_runs_agree():
    """Fifteen neighbours leave a long tail on one weak vote each; without a total order the
    menu would depend on dict ordering."""
    weighted = {"e": 0.5, "b": 0.5, "d": 0.5}
    assert voted_menu(weighted, RANKED, size=1) == frozenset({"b"})
    assert voted_menu(dict(reversed(weighted.items())), RANKED, size=1) == frozenset({"b"})


def test_a_menu_smaller_than_the_baseline_is_what_padding_exists_to_fix():
    """The voted set averages five tools against a static eleven, so the like-for-like arm
    fills it the way the shipped `menus.choose` fills every menu it publishes."""
    voted = voted_menu({"e": 1.0}, RANKED, size=3)
    assert voted == frozenset({"e"})
    assert frozenset(pad(sorted(voted), RANKED, 3)) == frozenset({"e", "a", "b"})


def test_no_vote_at_all_is_an_empty_menu_and_not_a_crash():
    assert voted_menu({}, RANKED, size=3) == frozenset()


# --- the pool folds -------------------------------------------------------------------------


def test_a_group_always_lands_in_the_same_fold():
    assert fold_of("s42", 5) == fold_of("s42", 5)
    assert all(0 <= fold_of(f"s{i}", 5) < 5 for i in range(200))


def test_the_folds_between_them_cover_every_group():
    assert {fold_of(f"s{i}", 4) for i in range(200)} == {0, 1, 2, 3}


def test_the_pool_a_session_is_served_never_saw_that_session():
    """The point of folding the pool fit: a pool fitted over the whole corpus has already
    seen the scenario it is being asked to serve, and scores two points higher for it."""
    cases = [case(f"s{i}", {"a", "b"}, scenario=f"sc{i}") for i in range(8)]
    cases.append(case("odd", {"z", "y"}, scenario="odd"))
    pools = fit_pools(cases, regime(SCENARIO_REGIME), size=2, count=4, folds=3)
    serving = pools[fold_of("odd", 3)]
    assert not any("z" in menu for menu in serving)
    assert any("z" in menu for pool in pools for menu in pool)  # the other folds did see it


def test_there_is_one_pool_per_fold():
    cases = [case(f"s{i}", {"a"}, scenario=f"sc{i}") for i in range(10)]
    assert len(fit_pools(cases, regime(SCENARIO_REGIME), size=2, count=2, folds=4)) == 4


# --- the quartile split ------------------------------------------------------------------


def test_the_quartiles_are_four_equal_groups_shortest_opening_first():
    cases = [case(f"s{i}", {"a"}, opening=" ".join(["w"] * (i + 1))) for i in range(12)]
    groups = quartiles(cases, list(range(12)))
    assert [len(g) for g in groups] == [3, 3, 3, 3]
    assert [cases[i].words for i in groups[0]] == [1, 2, 3]
    assert [cases[i].words for i in groups[-1]] == [10, 11, 12]


def test_the_split_is_by_rank_and_not_by_word_count_so_the_quarters_stay_even():
    """Real openings pile up on a few lengths; cutting at the quantiles gave 435 against 309."""
    cases = [case(f"s{i:02d}", {"a"}, opening="w w w") for i in range(11)]
    cases += [case("long", {"a"}, opening=" ".join(["w"] * 40))]
    groups = quartiles(cases, list(range(12)))
    assert [len(g) for g in groups] == [3, 3, 3, 3]


def test_sessions_of_the_same_opening_length_split_the_same_way_every_run():
    cases = [case(f"s{i:02d}", {"a"}, opening="w w") for i in range(8)]
    assert quartiles(cases, list(range(8))) == quartiles(cases, list(reversed(range(8))))


def test_a_corpus_that_does_not_divide_by_four_still_scores_every_session_once():
    """The quarters differ by one session and no session falls between two of them."""
    cases = [case(f"s{i}", {"a"}, opening=" ".join(["w"] * (i + 1))) for i in range(7)]
    groups = quartiles(cases, list(range(7)))
    assert [len(g) for g in groups] == [1, 2, 2, 2]
    assert sorted(i for group in groups for i in group) == list(range(7))


# --- recall and the lift arithmetic ---------------------------------------------------------


def test_a_session_that_called_no_tool_is_not_scored_because_its_recall_is_undefined():
    cases = [case("s1", {"a"}), case("s2", set())]
    assert scorable(cases) == [0]


def test_a_partly_covered_journal_scores_only_what_it_names():
    """Otherwise the scenario row would score an unnamed session leave-one-out and call it
    unseen, and the rows would stop describing the same traffic."""
    cases = [case("s1", {"a"}, scenario="sc1"), case("s2", {"a"}, scenario="")]
    assert scorable(cases) == [0]


def test_no_journal_at_all_still_scores_everything_it_can():
    """A clone with no journal loses the intent regimes, not the leave-one-out row."""
    cases = [case("s1", {"a"}, scenario=""), case("s2", {"a"}, scenario="")]
    assert scorable(cases) == [0, 1]


def test_recall_is_the_share_of_what_the_session_wanted_that_the_menu_held():
    assert recall(frozenset({"a", "b"}), frozenset({"a", "c"})) == 0.5
    assert recall(frozenset({"a"}), frozenset({"a"})) == 1.0
    assert recall(frozenset(), frozenset({"a"})) == 0.0


def test_a_session_that_wanted_nothing_cannot_be_missed():
    """It is never scored in practice, and it must not divide by zero if it ever is."""
    assert recall(frozenset(), frozenset()) == 1.0


def test_the_static_row_is_the_popularity_ranking_and_nothing_else():
    """The baseline has to be the shipped rule, or the lift measures this script's author."""
    cases = [case("s1", {"a", "b"}), case("s2", {"a"}), case("s3", {"a", "c"})]
    ranked = popularity([sorted(c.tools) for c in cases])
    assert ranked[:2] == ["a", "b"]  # 'b' and 'c' tie at one call each, so name breaks it
    menu = frozenset(pad([], ranked, 2))
    row = static_row(menu, cases, [0, 1, 2])
    assert row.name == "static top-2"
    assert row.voted == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert row.no_miss == pytest.approx(2 / 3)


def test_the_baseline_row_is_scored_the_same_in_all_three_arms():
    """A static menu is not voted, padded or pooled, so the three columns hold one number and
    the lifts below it are all taken against a like quantity."""
    row = static_row(frozenset({"a", "b"}), [case("s1", {"a"})], [0])
    assert row.voted == row.padded == row.pooled == 1.0


def test_the_lift_is_the_row_minus_the_baseline_in_every_arm():
    static = Row("static top-11", voted=0.75, padded=0.75, pooled=0.75)
    row = Row("block the scenario", voted=0.8322, padded=0.8830, pooled=0.8557)
    lifts = [float(cell) for cell in regime_line(row, static).split()[4:9:2]]
    assert lifts == pytest.approx([0.0822, 0.1330, 0.1057], abs=5e-5)


def test_the_baseline_prints_no_lift_against_itself():
    static = Row("static top-11", voted=0.75, padded=0.75, pooled=0.75, no_miss=0.6, width=11.0)
    assert regime_line(static, None).split() == [
        "static",
        "top-11",
        "0.7500",
        "0.7500",
        "0.7500",
        "0.6000",
        "11.0",
    ]


def test_the_mean_over_a_quartile_only_counts_that_quartile():
    cases = [case("s1", {"a"}), case("s2", {"b"})]
    menus = {0: frozenset({"a"}), 1: frozenset({"a"})}
    assert mean_recall(menus, [0], cases) == 1.0
    assert mean_recall(menus, [1], cases) == 0.0
    assert mean_recall(menus, [0, 1], cases) == 0.5


# --- the journal ------------------------------------------------------------------------


def test_no_journal_is_not_an_error_because_a_fresh_clone_has_none(tmp_path):
    assert load_journal(tmp_path / "never-generated") == {}


def test_every_generation_run_on_the_machine_is_read():
    """One corpus can be spread over several seeds, and each run writes its own file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        _write(directory / "seed-a.jsonl", [("s-1", "sc1", "p1")])
        _write(directory / "seed-b.jsonl", [("s-2", "sc2", "p2")])
        assert load_journal(directory) == {"s-1": ("sc1", "p1"), "s-2": ("sc2", "p2")}


def test_a_half_written_line_is_skipped_rather_than_thrown(tmp_path):
    """The journal is appended to while a corpus generates, so a truncated tail is normal."""
    path = tmp_path / "seed.jsonl"
    path.write_text(
        json.dumps({"session_id": "s-1", "scenario_id": "sc1", "persona_id": "p1"})
        + '\n{"session_id": "s-2", "scen\n'
    )
    assert load_journal(tmp_path) == {"s-1": ("sc1", "p1")}


# --- putting it together -------------------------------------------------------------------


def test_a_session_whose_intent_is_blocked_scores_worse_than_one_whose_twin_is_visible():
    """The whole finding, on nine sessions. Two intents: one wants {a, b} and opens near
    (1, 0), the other wants {c, d} and opens near (0, 1). Blocking the session leaves the
    twins in place and the router is perfect. Blocking the scenario takes them away and it
    has nothing to fall back on but the other intent's tools."""
    cases = [case(f"x{i}", {"a", "b"}, scenario="sc-x") for i in range(4)]
    cases += [case(f"y{i}", {"c", "d"}, scenario="sc-y") for i in range(4)]
    vectors = unit(
        np.array([[1.0, 0.01 * i] for i in range(4)] + [[0.01 * i, 1.0] for i in range(4)])
    )
    similarity = vectors @ vectors.T
    ranked = popularity([sorted(c.tools) for c in cases])
    pools = [(frozenset({"a", "b"}), frozenset({"c", "d"}))] * 2
    scored = list(range(8))

    seen, _, _ = score_regime(
        regime(SESSION_REGIME), cases, scored, similarity, ranked, pools, size=2, neighbours=2
    )
    unseen, _, _ = score_regime(
        regime(SCENARIO_REGIME), cases, scored, similarity, ranked, pools, size=2, neighbours=2
    )
    assert seen.voted == 1.0
    assert seen.no_miss == 1.0
    assert unseen.voted == 0.0  # every visible neighbour wanted the other intent's tools
    assert unseen.width == 2.0  # it still declares a full menu, it is just the wrong one


def test_the_voted_and_pooled_menus_both_come_back_for_the_later_tables_to_reuse():
    cases = [case("s1", {"a"}), case("s2", {"a"})]
    similarity = np.array([[1.0, 0.9], [0.9, 1.0]])
    _, menus, pooled = score_regime(
        regime(SESSION_REGIME),
        cases,
        [0, 1],
        similarity,
        ["a"],
        [(frozenset({"a"}),)],
        size=1,
        neighbours=1,
    )
    assert menus == {0: frozenset({"a"}), 1: frozenset({"a"})}
    # The pooled arm is what the router declares, and it is what the short-opening sweep
    # compares against the static menu, so it has to come back too and it has to be a menu
    # from the pool rather than the voted set that chose it.
    assert pooled == {0: frozenset({"a"}), 1: frozenset({"a"})}


def test_the_opening_regime_groups_text_that_differs_only_in_case_and_punctuation():
    assert normalise("Hi!  Where is my ORDER?") == normalise("hi where is my order")
    assert normalise("order 12") != normalise("order 13")


def test_a_zero_vector_does_not_divide_by_zero():
    assert np.isfinite(unit(np.array([[0.0, 0.0], [3.0, 4.0]]))).all()
    assert unit(np.array([[3.0, 4.0]])).tolist() == [[0.6, 0.8]]


def test_a_session_with_no_opening_text_is_dropped_rather_than_scored_on_a_guess():
    """The vectors and the text come from two reads of the same table; a session that is in
    one and not the other has no length to bucket and no text to group on."""
    sessions = [_session("s-1", [1.0, 0.0], {"a"}), _session("s-2", [0.0, 1.0], {"b"})]
    cases, vectors = build_cases(sessions, {"s-1": "hello there"}, {})
    assert [c.session_id for c in cases] == ["s-1"]
    assert vectors.shape == (1, 2)


def _session(session_id: str, vector: list[float], tools: set[str]):
    from score_sets import Session

    return Session(
        session_id=session_id, vector=vector, tools=frozenset(tools), turns=4, billed=1000.0
    )


def _write(path, rows) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"session_id": s, "scenario_id": sc, "persona_id": p}) for s, sc, p in rows
        )
    )


# --- the short-opening fallback ----------------------------------------------------------


def test_the_blended_column_is_the_recall_of_the_mixed_policy_and_not_an_average_of_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sweep prints one number per threshold and a decision would be made on it.

    It is computed as a delta applied to the whole-corpus figure, because that is cheap and
    reads clearly. This checks it against the thing it claims to be: score every session
    under the rule, static menu for the short ones and the router's menu for the rest, and
    take the mean. If those two ever disagree the printed column is a summary of nothing.
    """
    static = frozenset({"a"})
    cases = [
        case("s1", {"a"}, opening="hi"),  # 1 word: the router misses it, the static menu has it
        case("s2", {"a"}, opening="hi there"),  # 2 words: both are right, so the rule is free
        case("s3", {"c"}, opening="a much longer opening than that one"),  # 7 words, untouched
        case("s4", {"d"}, opening="another opening that runs past the limit"),  # 7, untouched
    ]
    pooled = {0: frozenset({"z"}), 1: frozenset({"a"}), 2: frozenset({"c"}), 3: frozenset({"d"})}
    scored = [0, 1, 2, 3]

    report_short_opening_fallback(cases, scored, pooled, static, thresholds=[2])
    row = next(line for line in capsys.readouterr().out.splitlines() if "2 words or less" in line)
    printed = float(row.split()[-2])

    direct = sum(
        recall(static if cases[i].words <= 2 else pooled[i], cases[i].tools) for i in scored
    ) / len(scored)
    base = sum(recall(pooled[i], cases[i].tools) for i in scored) / len(scored)
    assert (base, direct) == (0.75, 1.0), "the fixture has to move, or it proves nothing"
    assert printed == pytest.approx(direct, abs=5e-5)
