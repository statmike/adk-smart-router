"""The signal that gates, the two that do not, and the baseline that is not zero.

§19.63 is the only real drift event this project owns, and every threshold here is fitted to it.
These tests pin the two readings it produced — a surface change that moved unseen share 26.1% to
77.3% for 36 points of recall, and a day of new traffic on the same tools where nothing moved — so
that a later change to the rule has to argue with the measurement rather than around it.
"""

from __future__ import annotations

from adk_smart_router import drift

# The two ends of §19.63, as shares.
HEALTHY = 0.261
AFTER_SURFACE_CHANGE = 0.773


# --- measuring ------------------------------------------------------------------------------


def test_unseen_share_counts_declared_tools_the_model_cannot_predict():
    got = drift.signals(declared=["a", "b", "c", "d"], classes=["a", "b"])
    assert got.unseen_share == 0.5
    assert (got.declared, got.unclassed) == (4, 2)


def test_a_class_the_agent_does_not_declare_changes_nothing():
    """The model knowing a retired tool is not drift. Only the declared side is the surface."""
    got = drift.signals(declared=["a", "b"], classes=["a", "b", "retired"])
    assert got.unseen_share == 0.0


def test_an_agent_declaring_nothing_is_not_infinitely_drifted():
    """No tool declared means no tool missing. A guard against dividing by the empty surface."""
    assert drift.signals(declared=[], classes=["a"]).unseen_share == 0.0


def test_abstain_in_the_class_list_makes_no_difference():
    with_abstain = drift.signals(["a", "b"], ["a", "ABSTAIN"])
    without = drift.signals(["a", "b"], ["a"])
    assert with_abstain.unseen_share == without.unseen_share


def test_confidence_quantiles_come_back_ordered():
    got = drift.signals(["a"], ["a"], confidences=[0.9, 0.1, 0.5, 0.7, 0.3])
    assert got.confidence_p10 is not None and got.confidence_p50 is not None
    assert got.confidence_p10 < got.confidence_p50
    assert got.sessions == 5


def test_no_sessions_reads_as_unmeasured_and_not_as_zero_confidence():
    """ "Never confident" and "nothing to measure" are opposite readings of the same agent."""
    got = drift.signals(["a"], ["a"], confidences=[])
    assert got.confidence_p50 is None
    assert got.sessions == 0


# --- the verdict ------------------------------------------------------------------------------


def base(share=HEALTHY):
    return drift.Signals(unseen_share=share, declared=100, unclassed=int(share * 100))


def now(share):
    return drift.Signals(unseen_share=share, declared=100, unclassed=int(share * 100))


def test_the_surface_change_of_19_63_is_due():
    got = drift.verdict(now(AFTER_SURFACE_CHANGE), base())
    assert got.due
    assert got.signal == "unseen_share"


def test_a_healthy_model_on_its_own_baseline_is_not_due():
    """26.1% unclassed is what healthy looked like. Compared to zero it reads as drifted."""
    got = drift.verdict(now(HEALTHY), base())
    assert not got.due
    assert "26.1%" in got.reason


def test_a_model_that_has_improved_is_not_due():
    assert not drift.verdict(now(0.10), base()).due


def test_the_margin_is_the_boundary_and_equalling_it_is_not_due():
    assert not drift.verdict(now(HEALTHY + 0.20), base(), margin=0.20).due
    assert drift.verdict(now(HEALTHY + 0.21), base(), margin=0.20).due


def test_the_reason_names_both_numbers_because_it_answers_why_this_triggered():
    got = drift.verdict(now(AFTER_SURFACE_CHANGE), base())
    assert "77.3%" in got.reason and "26.1%" in got.reason


# --- the clock, which is the backstop and not the headline -------------------------------------


def test_an_old_artifact_with_no_drift_is_due_on_age():
    got = drift.verdict(now(HEALTHY), base(), age_days=40, max_age_days=35)
    assert got.due
    assert got.signal == "age"


def test_a_young_artifact_with_no_drift_is_not_due():
    assert not drift.verdict(now(HEALTHY), base(), age_days=3, max_age_days=35).due


def test_drift_is_reported_ahead_of_age_when_both_fire():
    """The less interesting of two true things is the wrong one to print."""
    got = drift.verdict(now(AFTER_SURFACE_CHANGE), base(), age_days=90, max_age_days=35)
    assert got.signal == "unseen_share"


def test_no_age_limit_means_the_clock_never_fires():
    assert not drift.verdict(now(HEALTHY), base(), age_days=9000).due


# --- no baseline is a state, not an error ------------------------------------------------------


def test_an_artifact_published_before_this_existed_is_not_due_and_says_why():
    got = drift.verdict(now(0.99), baseline=None)
    assert not got.due
    assert "no baseline" in got.reason


def test_an_absent_baseline_never_triggers_on_age_either():
    """Without a baseline nothing here has been measured, so nothing here should act."""
    assert not drift.verdict(now(HEALTHY), None, age_days=900, max_age_days=35).due


# --- round-tripping through metadata.json -------------------------------------------------------


def test_a_baseline_survives_the_trip_through_json():
    original = drift.signals(["a", "b", "c"], ["a"], confidences=[0.4, 0.8])
    assert drift.Signals.from_dict(original.as_dict()) == original


def test_metadata_without_a_baseline_reads_as_no_baseline():
    assert drift.Signals.from_dict({}) is None
    assert drift.Signals.from_dict(None) is None
    assert drift.Signals.from_dict("nonsense") is None


def test_a_malformed_baseline_reads_as_no_baseline_rather_than_raising():
    """`status` runs against whatever is in the bucket, including something hand-edited."""
    assert drift.Signals.from_dict({"unseen_share": "quite a lot"}) is None


def test_a_baseline_from_a_newer_publisher_keeps_the_fields_this_one_knows():
    got = drift.Signals.from_dict({"unseen_share": 0.3, "declared": 10, "something_new": 1})
    assert got is not None and got.unseen_share == 0.3


# --- the ledger's word for it --------------------------------------------------------------------


def test_the_outcome_maps_onto_the_ledger_vocabulary():
    from adk_smart_router import lifecycle

    assert drift.verdict(now(AFTER_SURFACE_CHANGE), base()).outcome in lifecycle.OUTCOMES
    assert drift.verdict(now(HEALTHY), base()).outcome in lifecycle.OUTCOMES
