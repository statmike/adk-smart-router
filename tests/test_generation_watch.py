"""The slow guard: when a published generation is withdrawn, and what withdrawing does.

Two halves, and they are independent on purpose. `generation_watch.verdict` is arithmetic over two
arms and has no idea a bucket exists. The withdrawal marker is a fact about `model_store` and has
no idea a verdict exists. The script in between is thin because both halves are testable without
it (DESIGN.md 21.5).
"""

from __future__ import annotations

import json

from test_model_store import V1, V2, make_store, metadata_dict, version_blobs

from adk_smart_router import generation_watch as watch

# Comfortably over the sample floor in both arms, so a test that means to be about the margin is
# not accidentally about the floor.
PLENTY = watch.DEFAULT_MIN_LATER_TURNS * 4


def arms(routed: float, holdout: float, turns: int = PLENTY) -> watch.Arms:
    return watch.Arms(
        routed_turns=turns,
        routed_billed_per_turn=routed,
        holdout_turns=turns,
        holdout_billed_per_turn=holdout,
    )


# --- The verdict --------------------------------------------------------------


def test_a_generation_cheaper_than_its_holdout_arm_is_left_alone():
    verdict = watch.verdict(arms(routed=874, holdout=1243))

    assert not verdict.withdraw
    assert verdict.outcome == "declined"
    assert "-29.7%" in verdict.reason


def test_a_generation_dearer_than_its_holdout_arm_is_withdrawn():
    verdict = watch.verdict(arms(routed=1400, holdout=1000))

    assert verdict.withdraw
    assert verdict.outcome == "ok"
    assert "+40.0%" in verdict.reason
    assert verdict.metrics["g1_later"] == 0.4


def test_dearer_but_inside_the_margin_is_not_withdrawn():
    """Breaking even is already a failure, but the margin is what stops noise acting on it."""
    verdict = watch.verdict(arms(routed=1040, holdout=1000), margin=0.05)

    assert not verdict.withdraw
    assert verdict.outcome == "declined"


def test_the_margin_is_a_boundary_and_not_a_trigger():
    """Exactly at the margin declines. One token past it withdraws."""
    assert not watch.verdict(arms(routed=1050, holdout=1000), margin=0.05).withdraw
    assert watch.verdict(arms(routed=1051, holdout=1000), margin=0.05).withdraw


def test_too_little_traffic_is_skipped_rather_than_judged():
    thin = watch.Arms(
        routed_turns=watch.DEFAULT_MIN_LATER_TURNS - 1,
        routed_billed_per_turn=5000.0,
        holdout_turns=PLENTY,
        holdout_billed_per_turn=1000.0,
    )
    verdict = watch.verdict(thin)

    assert not verdict.withdraw
    assert verdict.outcome == "skipped"
    assert "not enough traffic" in verdict.reason


def test_the_floor_applies_to_each_arm_separately():
    """A generation with plenty of routed traffic and almost no holdout has no control arm, and
    a percentage computed against one is noise wearing a decimal point."""
    lopsided = watch.Arms(
        routed_turns=PLENTY,
        routed_billed_per_turn=5000.0,
        holdout_turns=3,
        holdout_billed_per_turn=1000.0,
    )

    assert watch.verdict(lopsided).outcome == "skipped"


def test_a_holdout_arm_with_turns_but_no_price_is_a_logging_fault():
    broken = watch.Arms(
        routed_turns=PLENTY,
        routed_billed_per_turn=5000.0,
        holdout_turns=PLENTY,
        holdout_billed_per_turn=None,
    )
    verdict = watch.verdict(broken)

    assert verdict.outcome == "skipped"
    assert "logging fault" in verdict.reason


def test_the_rate_limit_holds_back_a_second_withdrawal_in_one_window():
    verdict = watch.verdict(arms(routed=1400, holdout=1000), withdrawals_in_window=1)

    assert not verdict.withdraw
    assert verdict.outcome == "skipped"


def test_a_held_back_withdrawal_still_records_what_it_would_have_done():
    """The one situation the ledger most needs to show is an agent partway through a cascade.
    A `skipped` row saying only "rate limited" would hide it."""
    verdict = watch.verdict(arms(routed=1400, holdout=1000), withdrawals_in_window=1)

    assert "+40.0%" in verdict.reason
    assert "already been withdrawn 1 time(s)" in verdict.reason
    assert verdict.metrics["g1_later"] == 0.4


def test_the_rate_limit_does_not_hold_back_the_first_withdrawal():
    assert watch.verdict(arms(routed=1400, holdout=1000), withdrawals_in_window=0).withdraw


def test_every_outcome_records_both_thresholds_it_was_judged_against():
    """They are a stated prior rather than a fit, so the first person with two generations of
    history has to be able to see which numbers produced the row in front of them."""
    for verdict in (
        watch.verdict(arms(routed=874, holdout=1243)),
        watch.verdict(arms(routed=1400, holdout=1000)),
        watch.verdict(arms(routed=1400, holdout=1000, turns=2)),
    ):
        assert verdict.metrics["margin"] == watch.DEFAULT_MARGIN
        assert verdict.metrics["min_later_turns"] == watch.DEFAULT_MIN_LATER_TURNS
        assert verdict.metrics["routed_later_billed_per_turn"] == 1400 or verdict.outcome != "ok"


def test_the_slow_guard_and_the_fast_one_agree_about_where_harmful_starts():
    """`cache_watch_margin` and this one are the same quantity. An agent sitting in a gap between
    them would be demoted by one guard and approved by the other, on the same traffic."""
    from adk_smart_router.config import RouterConfig

    assert RouterConfig().cache_watch_margin == watch.DEFAULT_MARGIN


# --- What the marker does -----------------------------------------------------


def withdrawn(version: str) -> dict[str, bytes]:
    marker = json.dumps({"reason": "worse"}).encode()
    return {f"routers/planner/{version}/withdrawn.json": marker}


def two_versions() -> dict[str, bytes]:
    blobs = version_blobs(V1, meta=metadata_dict())
    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2)))
    return blobs


def test_without_a_marker_the_newest_version_serves(tmp_path):
    store = make_store(tmp_path, two_versions())

    assert store.latest_version("planner") == V2
    assert store.published_versions("planner") == (V1, V2)


def test_a_withdrawn_version_falls_back_to_the_one_before_it(tmp_path):
    blobs = two_versions()
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs)

    assert store.latest_version("planner") == V1
    model = store.load("planner")
    assert model is not None
    assert model.metadata.version == V1


def test_a_withdrawn_version_stops_counting_as_published(tmp_path):
    """One marker has to reach every reader of the version list at once. If a retrain cycle still
    saw a withdrawn version as shipped, its no-regression gate would compare against a model that
    was taken out of service for being bad."""
    blobs = two_versions()
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs)

    assert store.published_versions("planner") == (V1,)
    assert store.withdrawn_versions("planner") == (V2,)


def test_withdrawing_every_version_leaves_the_agent_in_passthrough(tmp_path):
    """The cascade run to its end. This is not a failure mode; it is the contract's floor reached
    the long way round, and `load` returning None is passthrough."""
    blobs = two_versions()
    blobs.update(withdrawn(V1))
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs)

    assert store.published_versions("planner") == ()
    assert store.latest_version("planner") is None
    assert store.load("planner") is None


def test_deleting_the_marker_puts_the_version_back(tmp_path):
    """Reversibility is what makes the watch something to trust rather than switch off."""
    blobs = two_versions()
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs)
    assert store.latest_version("planner") == V1

    del blobs[f"routers/planner/{V2}/withdrawn.json"]

    assert store.latest_version("planner") == V2


def test_a_pinned_version_ignores_the_marker(tmp_path):
    """A pin is somebody saying they know which artifact they want. The watch is automatic and a
    pin is not, so the explicit instruction wins and `--restore` is not needed to honour it."""
    blobs = two_versions()
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs, pin_version=V2)

    assert store.latest_version("planner") == V2
    model = store.load("planner")
    assert model is not None
    assert model.metadata.version == V2


def test_a_marker_beside_an_incomplete_directory_changes_nothing(tmp_path):
    """A half-published version was never a candidate. Withdrawing it must not resurrect it into
    any list, which is what a `withdrawn` set computed independently of `complete` would do."""
    blobs = version_blobs(V1, meta=metadata_dict())
    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2), booster=None))
    blobs.update(withdrawn(V2))
    store = make_store(tmp_path, blobs)

    assert store.published_versions("planner") == (V1,)
    assert store.latest_version("planner") == V1


def test_the_marker_name_is_the_one_the_store_reads():
    """The script writes this file and the store decides on it. Two spellings would mean a
    withdrawal that reports success and changes nothing."""
    from adk_smart_router.model_store import WITHDRAWN_NAME

    assert WITHDRAWN_NAME == "withdrawn.json"


def test_the_ledger_can_record_a_withdrawal():
    """`v_router_cycles` pivots `lifecycle.STEPS`, and a step missing from the tuple is a row
    nothing can write. The dashboard side of the same pairing is in test_sql_dashboard.py."""
    from adk_smart_router import lifecycle

    assert "withdraw" in lifecycle.STEPS
