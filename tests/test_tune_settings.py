"""The tuner's arithmetic and its choosing, checked without a network.

`tune_settings.py` reads BigQuery, but everything that decides an outcome is pure: the pool
fit, the two recall definitions, the miss-price threshold, and `pick`. These hold all four,
plus the caching that keeps a grid of several dozen candidates cheap.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
import tune_settings
from score_sets import Session
from tune_settings import (
    MIN_MENU_RECALL,
    Candidate,
    Scored,
    agrees,
    decline_threshold,
    fit_pool,
    halves,
    measure_recall,
    pick,
    write_settings,
)

from adk_smart_router.config import RouterConfig

CFG = RouterConfig(miss_cost_tokens=1_600, expected_session_turns=4.6, max_tools_kept=3)
TOKENS = {"a": 100, "b": 100, "c": 100, "d": 100, "e": 100}
LIVE = frozenset(TOKENS)


def session(session_id: str, tools: set[str]) -> Session:
    return Session(
        session_id=session_id,
        vector=[1.0, 0.0],
        tools=frozenset(tools),
        turns=4,
        billed=10_000.0,
    )


def scored(recall: float, g1: float, size: int = 11, count: int = 24, menus: int = 24) -> Scored:
    return Scored(
        candidate=Candidate(size, count, 1_600),
        menus=menus,
        width=float(size),
        recall=recall,
        honest=recall,
        pruned_share=1.0,
        g1=g1,
        warm=0.7,
    )


class Pred:
    """The shape `pruner` reads off a prediction: a label and a probability."""

    def __init__(self, label: str, probability: float) -> None:
        self.label = label
        self.probability = probability


# --- fit_pool ---------------------------------------------------------------------------


def test_the_pool_is_fit_from_the_sessions_it_is_given():
    train = [session(f"s{i}", {"a", "b", "c"}) for i in range(10)]
    pool = fit_pool(train, size=3, count=2)
    assert pool
    assert all(len(menu) <= 3 for menu in pool)
    assert frozenset({"a", "b", "c"}) in pool


def test_a_session_that_called_nothing_contributes_nothing():
    """An empty tool set is not a menu anyone can be served, so it must not shape the pool."""
    real = [session(f"s{i}", {"a", "b"}) for i in range(5)]
    assert fit_pool([*real, session("empty", set())], 2, 2) == fit_pool(real, 2, 2)


def test_no_sessions_means_no_pool_rather_than_a_crash():
    assert fit_pool([], 3, 4) == ()


def test_asking_for_more_menus_than_there_are_distinct_sets_returns_fewer():
    """`choose` is allowed to return a shorter pool, which is why `Scored` records what it got."""
    train = [session(f"s{i}", {"a", "b"}) for i in range(10)]
    assert len(fit_pool(train, size=2, count=20)) < 20


def test_merging_spends_on_what_a_session_wanted_where_padding_spends_on_the_head():
    """The difference the arm exists for, on the smallest corpus that shows it.

    Two sessions want `t` and `u` together and nothing else. Padding gives that candidate a
    third slot and fills it from the global popularity ranking, so the pool ships `q` to
    sessions that never asked for it. Merging leaves the menu at the width the traffic
    justifies. On a wide surface this is what turns 264 slots into 91 distinct tools
    (DESIGN.md 19.42, 19.47).
    """
    train = [session(f"a{i}", {"p", "q"}) for i in range(3)]
    train += [session(f"b{i}", {"q", "r"}) for i in range(3)]
    train += [session(f"c{i}", {"t", "u"}) for i in range(2)]
    padded = fit_pool(train, size=3, count=2)
    merged = fit_pool(train, size=3, count=2, merged=True)
    assert frozenset({"q", "t", "u"}) in padded
    assert frozenset({"t", "u"}) in merged


def test_the_two_arms_agree_when_there_is_nothing_to_merge():
    """One shape in the corpus leaves the two arms nothing to disagree about, which is worth
    pinning: the difference has to come from the candidates and not from the set cover."""
    train = [session(f"s{i}", {"a", "b", "c"}) for i in range(10)]
    assert fit_pool(train, 3, 2) == fit_pool(train, 3, 2, merged=True)


def test_a_merged_pool_is_still_capped_at_the_menu_size():
    train = [session(f"s{i}", {"a", "b"}) for i in range(5)]
    train += [session(f"t{i}", {"b", "c"}) for i in range(5)]
    train += [session(f"u{i}", {"c", "d"}) for i in range(5)]
    assert all(len(menu) <= 2 for menu in fit_pool(train, size=2, count=4, merged=True))


def test_merging_nothing_is_still_nothing():
    assert fit_pool([], 3, 4, merged=True) == ()


# --- measure_recall ---------------------------------------------------------------------


def test_recall_is_the_share_of_wanted_tools_the_menu_held():
    held = [session("s", {"a", "b", "c", "d"})]
    shipped = {"s": frozenset({"a", "b"})}
    gate, honest, pruned = measure_recall(held, shipped, LIVE, LIVE, TOKENS)
    assert gate == pytest.approx(0.5)
    assert honest == pytest.approx(0.5)
    assert pruned == pytest.approx(1.0)


def test_a_declined_session_scores_a_free_one_and_does_not_count_as_pruned():
    """Shipping the full list cannot miss anything, so recall alone flatters a timid pool."""
    held = [session("s", {"a", "b", "c"})]
    gate, honest, pruned = measure_recall(held, {"s": None}, LIVE, LIVE, TOKENS)
    assert (gate, honest) == (1.0, 1.0)
    assert pruned == 0.0


def test_the_gate_forgives_a_tool_the_model_has_no_class_for_and_the_honest_number_does_not():
    """The one case the two columns exist for. `d` is wanted, unreachable, and only one
    of the two definitions charges the pool for it."""
    held = [session("s", {"a", "d"})]
    shipped = {"s": frozenset({"a"})}
    gate, honest, _ = measure_recall(held, shipped, LIVE, frozenset({"a", "b", "c"}), TOKENS)
    assert gate == pytest.approx(1.0)
    assert honest == pytest.approx(0.5)


def test_the_two_agree_when_every_wanted_tool_is_a_class():
    """What the reference agent turns out to be: the pinning removes nothing at all."""
    held = [session("s", {"a", "b"}), session("t", {"b", "c"})]
    shipped = {"s": frozenset({"a"}), "t": frozenset({"b", "c"})}
    gate, honest, _ = measure_recall(held, shipped, LIVE, LIVE, TOKENS)
    assert gate == honest


def test_a_session_with_no_shipped_entry_is_skipped_rather_than_guessed():
    held = [session("s", {"a", "b"}), session("absent", {"c"})]
    gate, _, _ = measure_recall(held, {"s": frozenset({"a"})}, LIVE, LIVE, TOKENS)
    assert gate == pytest.approx(0.5)


def test_nothing_held_out_is_zero_rather_than_a_division_by_zero():
    assert measure_recall([], {}, LIVE, LIVE, TOKENS) == (0.0, 0.0, 0.0)


# --- decline_threshold --------------------------------------------------------------------


def test_the_threshold_is_the_price_that_flips_the_decision():
    """Cross-checked against the rule itself: a hair under keeps the prune, a hair over drops
    it. That is the whole claim the inertness report rests on."""
    from adk_smart_router.pruner import worth_pruning

    ranked = {"s": [Pred("a", 0.6), Pred("b", 0.2)]}
    chosen = {"s": frozenset({"a", "b"})}
    point = decline_threshold(ranked, chosen, LIVE, TOKENS, CFG)[0]
    saved = 300  # c, d, e removed at 100 tokens each
    assert worth_pruning(saved, 0.8, replace(CFG, miss_cost_tokens=int(point * 0.99)))
    assert not worth_pruning(saved, 0.8, replace(CFG, miss_cost_tokens=int(point * 1.01)))


def test_a_session_the_model_is_certain_about_has_no_threshold_at_all():
    """No risk of a miss means no price can make the gamble look bad, so it is left out
    rather than recorded as a very large number that would drag the median."""
    ranked = {"sure": [Pred("a", 1.0)], "unsure": [Pred("a", 0.5)]}
    chosen = {"sure": frozenset({"a"}), "unsure": frozenset({"a"})}
    assert len(decline_threshold(ranked, chosen, LIVE, TOKENS, CFG)) == 1


def test_a_declined_session_has_no_threshold_because_it_already_declined():
    ranked = {"s": [Pred("a", 0.5)]}
    assert decline_threshold(ranked, {"s": None}, LIVE, TOKENS, CFG) == []


def test_a_bigger_prune_takes_a_higher_price_to_give_up():
    ranked = {"wide": [Pred("a", 0.5)], "narrow": [Pred("a", 0.5)]}
    chosen = {"wide": frozenset({"a"}), "narrow": frozenset({"a", "b", "c", "d"})}
    wide, narrow = decline_threshold(ranked, chosen, LIVE, TOKENS, CFG)
    assert wide < narrow  # sorted, so the smaller prune is first


def test_the_thresholds_come_back_sorted_because_the_report_takes_a_median():
    ranked = {f"s{i}": [Pred("a", 0.1 * (i + 1))] for i in range(6)}
    chosen = {f"s{i}": frozenset({"a"}) for i in range(6)}
    got = decline_threshold(ranked, chosen, LIVE, TOKENS, CFG)
    assert got == sorted(got)


# --- pick -------------------------------------------------------------------------------


def test_the_best_saving_that_clears_the_bar_wins():
    """The candidate saving most is below the bar, so it must lose to the one that clears it."""
    best = pick([scored(0.80, -0.40), scored(0.95, -0.30), scored(0.99, -0.10)])
    assert best is not None
    assert best.g1 == pytest.approx(-0.30)


def test_nothing_eligible_is_none_rather_than_the_least_bad():
    assert pick([scored(0.10, -0.40), scored(0.20, -0.30)]) is None


def test_a_candidate_exactly_on_the_bar_is_eligible():
    assert scored(MIN_MENU_RECALL, -0.2).eligible


def test_the_floor_travels_with_the_row_so_the_frontier_can_replay_it():
    """`report_floors` re-picks by swapping the floor onto each row, so eligibility has to
    read the row's own bar and not a module constant."""
    rows = [replace(scored(0.93, -0.30), floor=0.95)]
    assert pick(rows) is None
    assert pick([replace(row, floor=0.90) for row in rows]) is not None


def test_a_tie_on_saving_breaks_towards_the_smaller_pool():
    fewer = scored(0.95, -0.30, count=8, menus=8)
    more = scored(0.95, -0.30, count=32, menus=32)
    assert pick([more, fewer]) is fewer


def test_a_tie_on_saving_and_pool_breaks_towards_the_narrower_menu():
    narrow = scored(0.95, -0.30, size=7)
    wide = scored(0.95, -0.30, size=15)
    assert pick([wide, narrow]) is narrow


def test_nothing_to_choose_from_is_none():
    assert pick([]) is None


# --- write_settings -----------------------------------------------------------------------


def test_the_choice_is_written_with_the_evidence_beside_it(tmp_path):
    out = tmp_path / "settings.json"
    write_settings(str(out), "support", scored(0.9110, -0.3122), {"session_prefix": "corpus-x"})
    payload = json.loads(out.read_text())
    assert (payload["menu_size"], payload["menu_count"]) == (11, 24)
    assert payload["menu_candidates"] == "padded"
    assert payload["miss_cost_tokens"] == 1_600
    assert payload["evidence"]["menu_recall"] == pytest.approx(0.9110)
    assert payload["evidence"]["min_menu_recall"] == MIN_MENU_RECALL
    assert payload["evidence"]["session_prefix"] == "corpus-x"


def test_the_file_names_which_way_the_pool_was_built(tmp_path):
    """`publish_router.py --settings` reads this key, so a tuner that chose merging and
    wrote nothing would silently publish the padded pool it did not choose."""
    out = tmp_path / "settings.json"
    chosen = replace(scored(0.95, -0.4), candidate=Candidate(11, 24, 1_600, merged=True))
    write_settings(str(out), "support", chosen, {})
    assert json.loads(out.read_text())["menu_candidates"] == "merged"


# --- the stability check ------------------------------------------------------------------


def test_every_session_lands_in_exactly_one_half():
    held = [session(f"s{i}", {"a"}) for i in range(400)]
    left, right = halves(held)
    assert len(left) + len(right) == 400
    assert {s.session_id for s in left} & {s.session_id for s in right} == set()


def test_the_halves_are_roughly_even():
    held = [session(f"s{i}", {"a"}) for i in range(400)]
    left, _ = halves(held)
    assert 150 < len(left) < 250


def test_the_split_does_not_move_between_runs():
    """A random split would let a rerun make an unstable recommendation stable by chance."""
    held = [session(f"s{i}", {"a"}) for i in range(200)]
    assert halves(held)[0] == halves(held)[0]


def test_nothing_held_out_splits_into_nothing():
    assert halves([]) == ([], [])


def test_a_half_that_picks_the_same_candidate_agrees():
    best = scored(0.95, -0.30)
    assert agrees([scored(0.95, -0.28), best], best)


def test_a_half_that_picks_a_different_candidate_does_not_agree():
    best = scored(0.95, -0.30, size=11)
    assert not agrees([scored(0.95, -0.40, size=25)], best)


def test_agreement_is_about_the_candidate_and_not_the_predicted_level():
    """A half is under-warmed, so its G1 differs for the same choice. That is not a disagreement."""
    best = scored(0.95, -0.34)
    assert agrees([scored(0.93, -0.30)], best)


def test_a_half_where_nothing_clears_the_bar_does_not_agree():
    assert not agrees([scored(0.10, -0.40)], scored(0.95, -0.30))


def test_nothing_to_compare_against_does_not_agree():
    assert not agrees([scored(0.95, -0.30)], None)


# --- The first cycle of a new agent -------------------------------------------------------
#
# The sweep replays held-out traffic through the published model's menus, so an agent that
# has never published has nothing to sweep against. That has to reach `retrain.sh` as a
# decline and not as a failure, or the first cycle can never finish: tuning comes before
# publishing, so a new agent would need an artifact it cannot have yet.


class _NoModel:
    """A ModelStore that refuses, over a bucket holding the versions it is handed."""

    def __init__(self, versions: tuple[str, ...]) -> None:
        self._versions = versions

    def load(self, agent: str) -> None:
        return None

    def published_versions(self, agent: str) -> tuple[str, ...]:
        return self._versions


def run_main(monkeypatch, versions: tuple[str, ...]) -> int:
    monkeypatch.setattr(
        sys, "argv", ["tune_settings.py", "--project", "p", "--bucket", "b", "--agent", "brand-new"]
    )
    monkeypatch.setattr(tune_settings, "ModelStore", lambda cfg: _NoModel(versions))
    return tune_settings.main()


def test_an_agent_with_nothing_published_declines_rather_than_failing(monkeypatch, capsys):
    assert run_main(monkeypatch, ()) == 0
    assert "nothing to sweep" in capsys.readouterr().out


def test_a_broken_artifact_is_still_a_failure(monkeypatch, capsys):
    """The half of the old behaviour worth keeping. Something was published and cannot be
    read, and publishing on the defaults over the top of it would bury that."""
    assert run_main(monkeypatch, ("20260101T000000Z",)) == 1
    assert "refused to load" in capsys.readouterr().err


def test_a_decline_writes_no_settings_file(monkeypatch, tmp_path):
    """`retrain.sh` reads the size of the output file, not the exit code, to tell a
    recommendation from a decline. Exiting 0 with a file written would publish a pool shape
    nothing measured."""
    out = tmp_path / "settings.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_settings.py",
            "--project",
            "p",
            "--bucket",
            "b",
            "--agent",
            "brand-new",
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(tune_settings, "ModelStore", lambda cfg: _NoModel(()))

    assert tune_settings.main() == 0
    assert not out.exists()


# --- What the sweep leaves in the retrain ledger ----------------------------------


def tune_args(**overrides):
    import argparse

    fields = {
        "out": "settings.json",
        "project": "proj",
        "dataset": "ds",
        "agent": "support",
        "cycle_id": "cyc123",
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def tune_steps(monkeypatch):
    written = []
    monkeypatch.setattr(
        tune_settings.lifecycle,
        "record",
        lambda project, dataset, step, client=None: written.append(step) or True,
    )
    return written


def test_a_sweep_that_recommends_something_records_the_shape_it_chose(monkeypatch):
    written = tune_steps(monkeypatch)
    best = scored(0.95, -0.31, size=15, count=8)
    tune_settings.record_tune(tune_args(), "ok", "chose 15x8", after=best, evidence=best)
    step = written[0]
    assert (step.step, step.outcome) == ("tune", "ok")
    assert step.settings_after == {
        "menu_size": 15,
        "menu_count": 8,
        "menu_candidates": "padded",
        "miss_cost_tokens": 1_600,
    }
    assert step.metrics["predicted_g1"] == -0.31


def test_a_decline_leaves_the_settings_unchanged_rather_than_writing_what_it_refused(monkeypatch):
    """`settings_after` is what stands when the step finishes, and a decline changes nothing."""
    written = tune_steps(monkeypatch)
    tune_settings.record_tune(
        tune_args(),
        "declined",
        "did not replicate",
        before=scored(0.95, -0.30),
        evidence=scored(0.96, -0.33),
    )
    step = written[0]
    assert step.outcome == "declined"
    assert step.settings_after is None
    assert step.settings_before is not None


def test_the_shape_it_declined_to_adopt_is_still_kept_as_evidence(monkeypatch):
    """A recommendation that keeps arriving and keeps failing to replicate is worth seeing."""
    written = tune_steps(monkeypatch)
    tune_settings.record_tune(
        tune_args(), "declined", "did not replicate", evidence=scored(0.96, -0.33, size=7)
    )
    assert written[0].metrics["proposed"]["menu_size"] == 7


def test_a_sweep_with_nothing_to_recommend_records_the_decline_with_no_shape(monkeypatch):
    written = tune_steps(monkeypatch)
    tune_settings.record_tune(tune_args(), "declined", "nothing cleared the bar")
    assert written[0].metrics is None
    assert written[0].settings_after is None


def test_a_report_only_run_records_nothing_because_it_recommends_nothing(monkeypatch):
    """Without --out the sweep hands nobody anything, so it is a rehearsal like --dry-run."""
    written = tune_steps(monkeypatch)
    tune_settings.record_tune(
        tune_args(out=None), "ok", "chose something", after=scored(0.95, -0.3)
    )
    assert written == []


def test_the_tuners_outcomes_are_words_the_ledger_knows():
    for outcome in ("ok", "declined"):
        assert outcome in tune_settings.lifecycle.OUTCOMES
