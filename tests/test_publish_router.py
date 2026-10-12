"""The publisher's reconciliation, which decides the label mapping.

`scripts/publish_router.py` is the only thing standing between a correctly
trained booster and a `metadata.json` that mislabels every prediction it ever
serves. The failure is silent by construction: the digest still verifies, the
gate metrics are still true, and `predictor.py` still returns confident answers
for the wrong tools. So the pure parts get tested.

The GCS and BigQuery halves are not tested here. They are thin and they are
exercised end to end by the script's own `--dry-run` against the real bucket.
The format conversion is covered only for its guard, because the conversion
itself needs a second XGBoost that only exists inside the publisher's subprocess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import publish_router
import pytest
from fleet_cache_watch import Arm as FleetArm
from fleet_cache_watch import Stratum as FleetStratum
from fleet_cache_watch import Verdict as FleetVerdict
from publish_router import (
    DEFAULT_MENU_CANDIDATES,
    DEFAULT_MENU_COUNT,
    DEFAULT_MENU_SIZE,
    DEFAULT_MIN_MENU_RECALL,
    DEFAULT_MIN_PREDICTED_SAVING,
    GATE_METRIC_COLUMNS,
    MIN_SAVING_SESSIONS,
    HeldOut,
    Recall,
    Refused,
    Saving,
    TrainingRun,
    apply_settings,
    build_metadata,
    choose_menus,
    convert_booster,
    gate_fleet_cache_watch,
    gate_menu_recall,
    gate_no_regression,
    gate_predicted_saving,
    load_declaration_tokens,
    measure_held_out,
    menu_recall,
    predicted_saving,
    reconcile_classes,
    reconcile_features,
    reconcile_shape,
    split_uri,
)

from adk_smart_router import drift
from adk_smart_router.model_store import parse_metadata

# The real export's order: descending alphabetical, ABSTAIN last (DESIGN.md 18.8).
BOOSTER_ORDER = ["update_email", "track_shipment", "get_invoice", "cancel_order", "ABSTAIN"]
# What BigQuery puts in `router_training_runs.classes`: the same set, ascending.
RECORD_ORDER = sorted(BOOSTER_ORDER)


def run(**overrides: object) -> TrainingRun:
    fields: dict[str, object] = {
        "run_id": "r-1",
        "agent": "support",
        "version": "20260901T160000Z",
        "trained_at": "2026-09-01T16:00:00+00:00",
        "corpus_since": None,
        "environment": "prod",
        "embedding_model": "gemini-embedding-2",
        "embedding_dims": 4,
        "rows": 5373,
        "classes": tuple(RECORD_ORDER),
        "export_uri": "gs://bucket/routers/support/20260901T160000Z",
        "gate_metrics": {"coverage": 0.9852, "miss_reduction": 0.84},
    }
    fields.update(overrides)
    return TrainingRun(**fields)  # type: ignore[arg-type]


def assets(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "class_names": list(BOOSTER_ORDER),
        "feature_names": ["f0", "f1", "f2", "f3"],
        "label_col": "tool_name",
        "model_type": "boosted_tree_classifier",
    }
    base.update(overrides)
    return base


# --- The order comes from the assets, never from the run record ---------------


def test_publishes_the_booster_order_not_the_sorted_one():
    got = reconcile_classes(assets(), run(), allow_unlearned=False)
    assert got == tuple(BOOSTER_ORDER)
    assert got != tuple(RECORD_ORDER), "publishing the run record's order reverses the mapping"


def test_same_set_in_a_different_order_is_not_an_error():
    """The two orders disagreeing is the normal case, not the failure case."""
    assert set(BOOSTER_ORDER) == set(RECORD_ORDER)
    reconcile_classes(assets(), run(), allow_unlearned=False)


def test_refuses_a_label_the_corpus_never_contained():
    bad = assets(class_names=[*BOOSTER_ORDER, "invented_tool"])
    with pytest.raises(Refused, match="labels the corpus never contained"):
        reconcile_classes(bad, run(), allow_unlearned=False)


def test_refuses_a_class_the_booster_cannot_predict_unless_allowed():
    record = run(classes=(*RECORD_ORDER, "rare_tool"))
    with pytest.raises(Refused, match="cannot predict"):
        reconcile_classes(assets(), record, allow_unlearned=False)
    assert reconcile_classes(assets(), record, allow_unlearned=True) == tuple(BOOSTER_ORDER)


def test_refuses_duplicate_class_names():
    with pytest.raises(Refused, match="duplicates"):
        reconcile_classes(assets(class_names=["a", "a"]), run(classes=("a",)), False)


@pytest.mark.parametrize("value", [None, [], "ABSTAIN", ["ok", ""], ["ok", 3]])
def test_refuses_unusable_class_names(value):
    with pytest.raises(Refused, match="no usable class_names"):
        reconcile_classes(assets(class_names=value), run(), False)


# --- Features are positional, so their order is load-bearing too --------------


def test_accepts_features_in_order():
    reconcile_features(assets(), dims=4)


def test_refuses_reordered_features():
    with pytest.raises(Refused, match="not f0..f3 in order"):
        reconcile_features(assets(feature_names=["f0", "f2", "f1", "f3"]), dims=4)


def test_refuses_the_wrong_number_of_features():
    with pytest.raises(Refused, match="not f0..f767 in order"):
        reconcile_features(assets(), dims=768)


# --- Shape --------------------------------------------------------------------


def test_refuses_a_model_type_the_predictor_cannot_read():
    with pytest.raises(Refused, match="model_type"):
        reconcile_shape(assets(model_type="linear_reg"))


def test_refuses_a_different_label_column():
    with pytest.raises(Refused, match="label_col"):
        reconcile_shape(assets(label_col="tool"))


# --- URIs ---------------------------------------------------------------------


def test_splits_a_gs_uri():
    assert split_uri("gs://b/routers/support/v1/") == ("b", "routers/support/v1")


@pytest.mark.parametrize("uri", ["/routers/support", "https://b/x", "gs://bucket", "gs://"])
def test_refuses_a_uri_it_cannot_use(uri):
    with pytest.raises(Refused):
        split_uri(uri)


# --- Format conversion --------------------------------------------------------


def fake_converter(monkeypatch, *, returncode=0, report=None, stderr=""):
    """Stand in for the subprocess, which needs an XGBoost this venv does not have.

    It writes the report file the real converter writes, at the path the command
    line names, so the caller's file handling is exercised rather than skipped.
    """

    class Completed:
        pass

    Completed.returncode = returncode
    Completed.stdout = ""
    Completed.stderr = stderr

    def run(command, **_kwargs):
        Path(command[-1]).write_text(json.dumps(report or {}))
        Path(command[-2]).write_bytes(b"converted-booster-bytes")
        return Completed()

    monkeypatch.setattr(publish_router.subprocess, "run", run)


def test_refuses_when_the_conversion_fails():
    """A failed conversion must be a refusal, not an empty `model.ubj`."""
    fine = {"xgboost_version": "1.7.6", "num_features": 4, "num_class": 5, "objective": "x"}
    with pytest.MonkeyPatch.context() as patch:
        fake_converter(patch, returncode=1, report=fine, stderr="Invalid model format")
        with pytest.raises(Refused, match="Invalid model format"):
            convert_booster(b"not a booster", tuple(BOOSTER_ORDER), 4)
    assert publish_router.LEGACY_XGBOOST_PIN.startswith("xgboost==")


def test_refuses_a_booster_from_a_different_export():
    """Right format, wrong model: the shapes disagree and the pair is rejected."""
    wrong = {"xgboost_version": "1.7.6", "num_features": 768, "num_class": 22, "objective": "x"}
    with pytest.MonkeyPatch.context() as patch:
        fake_converter(patch, report=wrong)
        with pytest.raises(Refused, match="not from the same export"):
            convert_booster(b"anything", tuple(BOOSTER_ORDER), 4)


def test_returns_the_converted_bytes_not_the_exported_ones():
    """What comes back is what gets digested and served."""
    matching = {
        "xgboost_version": "1.7.6",
        "num_features": 4,
        "num_class": len(BOOSTER_ORDER),
        "objective": "multi:softprob",
    }
    with pytest.MonkeyPatch.context() as patch:
        fake_converter(patch, report=matching)
        got = convert_booster(b"legacy-export-bytes", tuple(BOOSTER_ORDER), 4)
    assert got == b"converted-booster-bytes"


# --- The metadata the reader has to accept ------------------------------------


def test_metadata_passes_the_readers_own_validator():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"booster-bytes")
    parsed = parse_metadata(metadata)
    assert parsed.classes == tuple(BOOSTER_ORDER)
    assert parsed.size_bytes == len(b"booster-bytes")
    assert parsed.backend == "vertex"


def test_digest_and_size_describe_the_bytes_passed_in():
    import hashlib

    payload = b"\x00\x01\x02" * 100
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), payload)
    assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["size_bytes"] == len(payload)


# --- Menus --------------------------------------------------------------------
#
# The choosing itself is `menus.choose` and is tested there. What is tested here is the
# publisher's part: reconciling the corpus against the label order, and writing something
# the reader accepts and a human can diff.


def test_menus_hold_only_tools_the_booster_can_predict():
    """`send_gift` was called in the corpus and is not a class, so it was dropped from
    training. A menu slot spent on it can never be chosen on purpose."""
    sessions = [["get_invoice", "send_gift"]] * 3
    (menu,) = choose_menus(sessions, tuple(BOOSTER_ORDER), size=2, count=1)
    assert "send_gift" not in menu


def test_abstain_and_the_control_plane_never_take_a_slot():
    sessions = [["get_invoice", "ABSTAIN", "transfer_to_agent"]] * 3
    (menu,) = choose_menus(sessions, tuple(BOOSTER_ORDER), size=3, count=1)
    assert menu.isdisjoint({"ABSTAIN", "transfer_to_agent"})


def test_a_corpus_with_nothing_recognisable_in_it_yields_no_menus():
    """Publishing goes ahead without menus; the router prunes per session, as it always
    did. Refusing here would block a good booster over an optional field."""
    assert choose_menus([["send_gift"], ["send_gift"]], tuple(BOOSTER_ORDER), 7, 8) == ()


def test_asking_for_merged_candidates_gets_a_merged_pool():
    """The publisher's half of DESIGN.md 19.47: `--menu-candidates merged` has to reach
    `menus.bucket`. Padding fills the third slot of the small menu from the popularity
    ranking; merging leaves it at the width the traffic justifies."""
    sessions = [["get_invoice", "cancel_order"]] * 3
    sessions += [["cancel_order", "update_email"]] * 3
    sessions += [["track_shipment", "get_invoice"]] * 2
    padded = choose_menus(sessions, tuple(BOOSTER_ORDER), 3, 2)
    merged = choose_menus(sessions, tuple(BOOSTER_ORDER), 3, 2, "merged")
    assert max(len(menu) for menu in padded) == 3
    assert frozenset({"track_shipment", "get_invoice"}) in merged


def test_the_shipped_default_is_the_pool_that_was_always_published():
    """Nobody's artifact moves because this option exists. The default has to keep building
    what it built before, and the way to say that is to call it both ways."""
    sessions = [["get_invoice", "cancel_order"]] * 10 + [["update_email"]]
    assert choose_menus(sessions, tuple(BOOSTER_ORDER), 2, 2) == choose_menus(
        sessions, tuple(BOOSTER_ORDER), 2, 2, DEFAULT_MENU_CANDIDATES
    )


def test_merging_drops_the_tools_the_booster_cannot_predict_too():
    """The filter runs before the candidates are built, so it has to hold on both arms."""
    sessions = [["get_invoice", "send_gift"]] * 3
    merged = choose_menus(sessions, tuple(BOOSTER_ORDER), 2, 1, "merged")
    assert all("send_gift" not in menu for menu in merged)


def test_menus_are_written_sorted_so_two_publishes_can_be_diffed():
    menus = (frozenset({"track_shipment", "get_invoice"}),)
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", menus)
    assert metadata["menus"] == [["get_invoice", "track_shipment"]]


def test_published_menus_survive_the_readers_validator():
    menus = (frozenset({"get_invoice"}), frozenset({"cancel_order", "update_email"}))
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", menus)
    assert parse_metadata(metadata).menus == menus


def test_a_publish_with_no_menus_writes_an_empty_list_not_a_missing_key():
    """`model_store` treats an absent key and an empty list the same way, so this is
    about the file being self-describing: it says menus were considered."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes")
    assert metadata["menus"] == []
    assert parse_metadata(metadata).menus == ()


# --- Gate criterion 1: menu recall (DESIGN.md 19.11) ------------------------------

MENU = frozenset({"get_invoice", "cancel_order"})


def held(wanted, kept=MENU, removed=0, turns=None, billed=None):
    """One held-out session. Defaults to routed, unpriced and unbilled."""
    return HeldOut(
        wanted=frozenset(wanted), kept=kept, removed_tokens=removed, turns=turns, billed=billed
    )


def test_a_menu_holding_everything_a_session_wanted_scores_one():
    found = menu_recall([held({"get_invoice"})])
    assert found.blended == 1.0
    assert found.sessions == 1


def test_a_menu_missing_half_of_what_was_wanted_scores_half():
    found = menu_recall([held({"get_invoice", "update_email"})])
    assert found.blended == 0.5


def test_a_declined_session_scores_a_free_one_because_it_ships_the_full_list():
    found = menu_recall([held({"update_email"}, kept=None)])
    assert found.blended == 1.0
    assert found.routed_sessions == 0


def test_declining_lifts_the_criterion_and_leaves_the_routed_number_alone():
    """The reason both numbers are printed. A model can pass criterion 1 by routing less,
    which is why criterion 3 prices the declining (DESIGN.md 19.5)."""
    found = menu_recall([held({"update_email"}, kept=None), held({"update_email"})])
    assert found.blended == 0.5  # one free 1.0 and one miss
    assert found.routed == 0.0
    assert found.routed_sessions == 1


def test_a_session_that_wanted_nothing_is_left_out_of_both_the_sum_and_the_count():
    found = menu_recall([held(()), held({"get_invoice"})])
    assert found.blended == 1.0
    assert found.sessions == 1


def test_every_session_counts_once_however_many_tools_it_wanted():
    """Set recall, not call coverage. A session wanting two tools cannot outvote two
    sessions wanting one each (DESIGN.md 19.4)."""
    found = menu_recall([held({"get_invoice", "update_email"}), held({"get_invoice"})])
    assert found.blended == 0.75  # 0.5 and 1.0, averaged
    assert found.sessions == 2


def test_no_sessions_to_measure_scores_zero_rather_than_dividing_by_zero():
    found = menu_recall([])
    assert (found.blended, found.routed, found.sessions) == (0.0, 0.0, 0)


# --- The measured recall travels with the artifact --------------------------------
#
# Criterion 1 is measured at publish time and used to be printed and discarded. A
# non-regression guard — refuse an artifact whose recall has fallen away from the one it
# replaces — needs the predecessor's number, so the number has to be in the file first.
# Nothing compares it yet; these tests are about it being there, and about the difference
# between "not measured" and "measured badly".


def measured(blended=0.9187, routed=0.8802, sessions=433, routed_sessions=390):
    return Recall(
        blended=blended, routed=routed, sessions=sessions, routed_sessions=routed_sessions
    )


def test_the_measured_menu_recall_is_written_into_the_artifacts_gate_metrics():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert metadata["gate_metrics"]["menu_recall"] == 0.9187


def test_the_routed_recall_is_written_beside_it_so_declining_stays_visible():
    """A model can lift the blended number by declining more. Carrying only the blended
    one would let a later comparison read a decline-happy artifact as an improvement."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert metadata["gate_metrics"]["menu_recall_routed"] == 0.8802


def test_the_run_records_own_gate_metrics_are_kept_alongside_the_recall():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert metadata["gate_metrics"]["coverage"] == 0.9852
    assert metadata["gate_metrics"]["miss_reduction"] == 0.84


def test_an_unmeasured_recall_leaves_the_key_out_rather_than_writing_a_zero():
    """No menus, or --skip-recall-gate. A zero would read as a measured catastrophe and
    a later non-regression guard would refuse the next publish over it."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes")
    assert "menu_recall" not in metadata["gate_metrics"]
    assert "menu_recall_routed" not in metadata["gate_metrics"]


def test_writing_the_recall_does_not_mutate_the_run_record_it_came_from():
    """`build_metadata` is called once per publish today, but `TrainingRun.gate_metrics`
    is a plain dict on a frozen dataclass and a shared reference would leak the recall of
    one publish into the next one built from the same run."""
    training_run = run()
    build_metadata(training_run, tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert "menu_recall" not in training_run.gate_metrics


def test_an_artifact_carrying_a_menu_recall_survives_the_readers_validator():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert parse_metadata(metadata).gate_metrics["menu_recall"] == 0.9187


def test_an_artifact_published_before_this_field_existed_still_loads():
    """Every artifact already in the bucket is one of these, including the one serving."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,))
    assert parse_metadata(metadata).gate_metrics == {"coverage": 0.9852, "miss_reduction": 0.84}


# --- The set size the numbers were measured at (DESIGN.md 19.72) ------------------


def test_the_gate_query_asks_for_the_set_size_the_metrics_were_measured_at():
    """Every other name here is a metric. `top_n` is the condition they hold under, and
    without it in the SELECT nothing downstream can tell a stale promise from a live one."""
    assert "top_n" in GATE_METRIC_COLUMNS


def test_the_set_size_rides_along_into_the_artifact():
    metadata = build_metadata(
        run(gate_metrics={"coverage": 0.9852, "top_n": 11.0}), tuple(BOOSTER_ORDER), b"bytes"
    )
    assert parse_metadata(metadata).gate_metrics["top_n"] == 11.0


def test_a_publish_with_no_menus_has_no_recall_to_carry():
    """No client is touched: with nothing to measure it returns before BigQuery."""
    args = argparse.Namespace(skip_recall_gate=False)
    assert measure_held_out(None, args, run(), tuple(BOOSTER_ORDER), (), b"bytes", None) is None


def test_skipping_the_recall_gate_carries_no_recall_either():
    """--skip-recall-gate means the number was never computed, so the artifact must not
    claim one. It is the same distinction as an absent key versus a zero."""
    args = argparse.Namespace(skip_recall_gate=True)
    assert (
        measure_held_out(None, args, run(), tuple(BOOSTER_ORDER), (MENU,), b"bytes", None) is None
    )


def test_an_unmeasured_holdout_gates_on_neither_criterion():
    """Both gates read the same replay, so both have to survive its absence."""
    args = argparse.Namespace(min_menu_recall=0.90, min_predicted_saving=-0.10)
    assert gate_menu_recall(None, args, None) is None
    assert gate_predicted_saving(None, args, {"get_invoice": 100}) is None


# --- Gate criterion 4: what the fleet has actually been billed (DESIGN.md 19.75) ---
#
# The three criteria above replay a holdout, which is the artifact predicting its own
# future. This one reads the past out of the warehouse. It is the only gate that can refuse
# on evidence the replay cannot produce, so what matters is that it refuses on a real loss
# and stays quiet on everything else.


def fleet_args(**overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {
        "skip_fleet_gate": False,
        "project": "p",
        "dataset": "d",
        "agent": "support",
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def fleet_verdict(pruned: float, whole: float, *, measured: bool = True) -> FleetVerdict:
    strata = (
        FleetStratum("r1", FleetArm(100, 100 * pruned), FleetArm(100, 100 * whole), versions=1),
    )
    return FleetVerdict(
        agent="support",
        days=30,
        min_samples=50,
        margin=0.05,
        strata=strata,
        judged=strata if measured else (),
        pooled_pruned=pruned,
        pooled_whole=whole,
        standardised_pruned=pruned if measured else None,
        standardised_whole=whole if measured else None,
        harmful=measured and pruned >= 1.05 * whole,
    )


def run_fleet_gate(monkeypatch, found: FleetVerdict | None, **overrides: object):
    def fake(client, **kwargs):  # noqa: ARG001
        assert found is not None, "the gate asked the warehouse when it should have skipped"
        return found

    monkeypatch.setattr(publish_router, "measure_fleet", fake)
    return gate_fleet_cache_watch(None, fleet_args(**overrides))


def test_a_fleet_that_is_paying_more_to_prune_refuses_the_publish(monkeypatch):
    """The live guard would have demoted this agent if any one replica had seen enough of
    it. Publishing a new artifact on top would be guessing at a question already answered."""
    with pytest.raises(Refused) as refusal:
        run_fleet_gate(monkeypatch, fleet_verdict(1_200.0, 1_000.0))
    assert "costing support" in str(refusal.value)


def test_a_fleet_where_pruning_is_working_publishes(monkeypatch):
    assert run_fleet_gate(monkeypatch, fleet_verdict(800.0, 1_000.0)).harmful is False


def test_a_loss_inside_the_margin_is_wobble_and_not_a_refusal(monkeypatch):
    """Same margin the live guard demotes at, so the two cannot disagree about a near miss."""
    assert run_fleet_gate(monkeypatch, fleet_verdict(1_040.0, 1_000.0)).harmful is False


def test_an_unmeasured_fleet_publishes_rather_than_refusing(monkeypatch):
    """A first publish has no production history and no holdout arm. Reading that silence
    as a failure would mean the gate could never be passed for the first time."""
    found = run_fleet_gate(monkeypatch, fleet_verdict(9_000.0, 0.0, measured=False))
    assert found.measured is False


def test_skipping_the_fleet_gate_asks_the_warehouse_nothing(monkeypatch):
    """The escape hatch has to save the BigQuery read, not just ignore its answer."""
    assert run_fleet_gate(monkeypatch, None, skip_fleet_gate=True) is None


# --- The declaration tokens the gate needs to price a prune (DESIGN.md 19.13) ------


def write_tokens(tmp_path: Path, payload: object) -> str:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_no_path_means_no_token_counts_and_the_gate_says_so_elsewhere():
    assert load_declaration_tokens(None) is None


def test_a_token_file_is_read_as_written(tmp_path):
    path = write_tokens(tmp_path, {"get_invoice": 120, "update_email": 90})
    assert load_declaration_tokens(path) == {"get_invoice": 120, "update_email": 90}


@pytest.mark.parametrize(
    "payload",
    [
        {},  # an empty file is a mistake, not an agent with no tools
        [],
        {"get_invoice": "120"},
        {"get_invoice": -1},
        {"get_invoice": 1.5},
    ],
)
def test_a_file_that_is_not_tool_names_and_token_counts_is_refused(tmp_path, payload):
    with pytest.raises(Refused):
        load_declaration_tokens(write_tokens(tmp_path, payload))


# --- Gate criterion 3: predicted steady-state saving (DESIGN.md 19.5) --------------
#
# Criterion 1 can only refuse in one direction. The cheapest route to a perfect recall is
# to decline every session, so the pair only works if something prices the declining.


def priced(removed, turns, billed, kept=MENU):
    return held({"get_invoice"}, kept=kept, removed=removed, turns=turns, billed=billed)


def test_the_saving_is_removed_tokens_at_the_cached_rate_over_what_was_billed():
    """1,000 tokens off every one of 4 turns, priced at 0.10, against 2,000 billed."""
    found = predicted_saving([priced(removed=1000, turns=4, billed=2000.0)])
    assert found.predicted == pytest.approx(-0.20)
    assert found.removed_billed == pytest.approx(400.0)


def test_a_declined_session_pays_into_the_denominator_and_never_the_numerator():
    """The whole point of the criterion. Declining is free on recall and costs here."""
    found = predicted_saving(
        [
            priced(removed=1000, turns=4, billed=2000.0),
            priced(removed=0, turns=4, billed=2000.0, kept=None),
        ]
    )
    assert found.predicted == pytest.approx(-0.10)
    assert (found.sessions, found.routed) == (2, 1)


def test_declining_everything_predicts_no_saving_at_all():
    found = predicted_saving([priced(removed=0, turns=4, billed=2000.0, kept=None)])
    assert found.predicted == 0.0
    assert found.routed == 0


def test_a_longer_session_saves_more_because_the_prune_is_paid_on_every_turn():
    short = predicted_saving([priced(removed=1000, turns=2, billed=2000.0)])
    long = predicted_saving([priced(removed=1000, turns=8, billed=2000.0)])
    assert long.predicted == pytest.approx(4 * short.predicted)


def test_a_session_the_response_log_never_saw_is_dropped_rather_than_counted_free():
    """A LEFT JOIN miss with billed None. Counting it as zero-billed would divide by a
    denominator that is missing its own sessions and overstate the saving."""
    found = predicted_saving([priced(removed=1000, turns=4, billed=2000.0), held({"get_invoice"})])
    assert found.predicted == pytest.approx(-0.20)
    assert (found.sessions, found.unpriced) == (1, 1)


def test_nothing_billed_anywhere_measures_nothing_rather_than_dividing_by_zero():
    assert predicted_saving([held({"get_invoice"})]) is None
    assert predicted_saving([]) is None


def enough(removed, turns=4, billed=2000.0, kept=MENU):
    """`MIN_SAVING_SESSIONS` identical sessions, so the gate has enough of them to vote."""
    return [priced(removed=removed, turns=turns, billed=billed, kept=kept)] * MIN_SAVING_SESSIONS


def test_an_artifact_that_reaches_the_bar_publishes(capsys):
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=False)
    found = gate_predicted_saving(enough(removed=1000), args, {"update_email": 1000})
    assert found.predicted == pytest.approx(-0.20)
    assert "-20.0% predicted" in capsys.readouterr().out


def test_an_artifact_that_is_safe_and_saves_too_little_is_refused():
    """0.10 * 100 * 4 over 2,000 is -2%, well short of the -10% bar."""
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=False)
    with pytest.raises(Refused, match="criterion 3"):
        gate_predicted_saving(enough(removed=100), args, {"a": 100})


def test_the_saving_gate_can_be_overridden_and_says_that_it_was(capsys):
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=True)
    found = gate_predicted_saving(enough(removed=100), args, {"a": 100})
    assert found.predicted == pytest.approx(-0.02)
    assert "--skip-saving-gate" in capsys.readouterr().out


def test_too_few_priced_sessions_reports_the_number_without_refusing_on_it(capsys):
    """A corpus that spans a tool surface change leaves only a slice of the holdout billed
    for the whole list. That slice is worth printing and too thin to refuse a release on."""
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=False)
    found = gate_predicted_saving(enough(removed=100)[:20], args, {"a": 100})
    assert found.predicted == pytest.approx(-0.02)
    assert "not gated on" in capsys.readouterr().out


def test_without_declaration_tokens_the_criterion_is_absent_not_passed(capsys):
    """Every session would measure as removing nothing, which is a 0% saving and would
    refuse every artifact for a reason that is about the publish step, not the model."""
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=False)
    assert gate_predicted_saving(enough(removed=0), args, None) is None
    assert "--declaration-tokens" in capsys.readouterr().out


def test_a_holdout_with_no_billing_anywhere_is_absent_too(capsys):
    args = argparse.Namespace(min_predicted_saving=-0.10, skip_saving_gate=False)
    assert gate_predicted_saving([held({"get_invoice"})], args, {"a": 100}) is None
    assert "no held-out session appears" in capsys.readouterr().out


def test_the_default_bar_is_the_one_design_states():
    assert DEFAULT_MIN_PREDICTED_SAVING == -0.10


# --- The predicted saving travels with the artifact -------------------------------


def saved(predicted=-0.3090, sessions=907, routed=600):
    return Saving(
        predicted=predicted,
        removed_billed=1.0,
        billed=1.0,
        sessions=sessions,
        routed=routed,
        unpriced=0,
    )


def test_the_predicted_saving_is_written_into_the_artifacts_gate_metrics():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured(), saved())
    assert metadata["gate_metrics"]["predicted_g1"] == -0.3090


def test_the_routed_share_is_written_beside_it_because_it_explains_the_saving():
    """Two artifacts can predict the same saving at very different routed shares, and
    the one routing less of its traffic has more room and more risk left in it."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured(), saved())
    assert metadata["gate_metrics"]["routed_share"] == pytest.approx(600 / 907)


def test_an_unmeasured_saving_leaves_the_key_out_rather_than_writing_a_zero():
    """A zero predicted saving is a real and damning measurement. Not having measured one
    must not be recorded as having measured that."""
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    assert "predicted_g1" not in metadata["gate_metrics"]
    assert "routed_share" not in metadata["gate_metrics"]


def test_an_artifact_carrying_a_predicted_saving_survives_the_readers_validator():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured(), saved())
    assert parse_metadata(metadata).gate_metrics["predicted_g1"] == -0.3090


# --- --settings, the tuner's output reaching publishing -----------------------------------


def settings_args(tmp_path, payload=None, **flags):
    """A namespace shaped like `parse_args` produces: the three merged flags start unset."""
    path = None
    if payload is not None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(payload))
    fields = {
        "settings": str(path) if path else None,
        "agent": "support",
        "menu_size": None,
        "menu_count": None,
        "menu_candidates": None,
        "min_menu_recall": None,
    }
    fields.update(flags)
    return argparse.Namespace(**fields)


def tuned(**overrides):
    payload = {
        "agent": "support",
        "menu_size": 15,
        "menu_count": 32,
        "miss_cost_tokens": 1_600,
        "evidence": {"min_menu_recall": 0.92},
    }
    payload.update(overrides)
    return payload


def test_no_settings_file_leaves_the_shipped_defaults():
    args = settings_args(None)
    apply_settings(args)
    assert (args.menu_size, args.menu_count) == (DEFAULT_MENU_SIZE, DEFAULT_MENU_COUNT)
    assert args.menu_candidates == DEFAULT_MENU_CANDIDATES
    assert args.min_menu_recall == DEFAULT_MIN_MENU_RECALL


def test_the_tuner_can_ask_for_a_merged_pool(tmp_path):
    args = settings_args(tmp_path, tuned(menu_candidates="merged"))
    apply_settings(args)
    assert args.menu_candidates == "merged"


def test_a_file_that_does_not_mention_candidates_gets_the_shipped_default(tmp_path):
    """A settings file written before 2026-09-05 has no such key and describes a padded
    artifact, so this used to assert `padded` to stop a republish silently changing the pool.
    The default moved to `merged` on 2026-09-07 (DESIGN.md 19.66) and that risk moved with
    it: a file that old now republishes a different pool from the same evidence. It is left
    that way rather than special-cased, because `tune_settings.py` has written the key since
    2026-09-05 and no such file exists in this project. An adopter carrying one forward
    should add `"menu_candidates": "padded"` to it before the next retrain."""
    payload = tuned()
    payload.pop("menu_candidates", None)
    args = settings_args(tmp_path, payload)
    apply_settings(args)
    assert args.menu_candidates == DEFAULT_MENU_CANDIDATES == "merged"


def test_a_way_of_building_the_pool_that_does_not_exist_is_refused(tmp_path):
    """Typos here are silent otherwise: anything that is not `merged` builds a padded pool,
    so `menu_candidates: "merge"` would publish the arm the operator did not ask for."""
    args = settings_args(tmp_path, tuned(menu_candidates="merge"))
    with pytest.raises(Refused, match="merge"):
        apply_settings(args)


def test_a_flag_given_without_a_settings_file_survives():
    args = settings_args(None, menu_size=7)
    apply_settings(args)
    assert args.menu_size == 7
    assert args.menu_count == DEFAULT_MENU_COUNT


def test_the_tuners_choice_is_taken_when_no_flag_says_otherwise(tmp_path):
    args = settings_args(tmp_path, tuned())
    apply_settings(args)
    assert (args.menu_size, args.menu_count) == (15, 32)
    assert args.min_menu_recall == 0.92


def test_an_explicit_flag_beats_the_file(tmp_path):
    """The whole reason the tuner writes a file instead of publishing for you."""
    args = settings_args(tmp_path, tuned(), menu_size=7, min_menu_recall=0.99)
    apply_settings(args)
    assert args.menu_size == 7  # the flag
    assert args.menu_count == 32  # the file, untouched by the other flag
    assert args.min_menu_recall == 0.99


def test_a_file_missing_a_value_falls_through_to_the_default(tmp_path):
    payload = tuned()
    del payload["menu_count"]
    del payload["evidence"]["min_menu_recall"]
    args = settings_args(tmp_path, payload)
    apply_settings(args)
    assert args.menu_size == 15
    assert args.menu_count == DEFAULT_MENU_COUNT
    assert args.min_menu_recall == DEFAULT_MIN_MENU_RECALL


def test_another_agents_settings_are_refused(tmp_path):
    """A tuned shape is a claim about one tool surface, so it cannot travel."""
    args = settings_args(tmp_path, tuned(agent="billing"))
    with pytest.raises(Refused, match="billing"):
        apply_settings(args)


def test_a_file_that_is_not_json_is_refused_rather_than_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    args = settings_args(None)
    args.settings = str(path)
    with pytest.raises(Refused):
        apply_settings(args)


def test_a_settings_file_that_is_not_there_is_refused(tmp_path):
    args = settings_args(None)
    args.settings = str(tmp_path / "absent.json")
    with pytest.raises(Refused):
        apply_settings(args)


def test_the_miss_price_is_reported_and_not_applied(tmp_path, capsys):
    """It is a runtime config value, so publishing says its piece and changes nothing."""
    args = settings_args(tmp_path, tuned(miss_cost_tokens=9_999))
    apply_settings(args)
    assert not hasattr(args, "miss_cost_tokens")
    assert "9999" in capsys.readouterr().out.replace(",", "")


def test_what_came_from_where_is_printed(tmp_path, capsys):
    """ROADMAP's condition on V2-7: the chosen values have to be visible."""
    args = settings_args(tmp_path, tuned(), menu_size=7)
    apply_settings(args)
    out = capsys.readouterr().out
    assert "settings.json" in out
    assert "command line" in out


# --- The non-regression guard -------------------------------------------------
#
# Criterion 1 is an absolute bar, so it cannot see an artifact that clears the bar and is
# still worse than the one it replaces. These check the second refusal: that it fires on a
# real drop, that it does not fire on the ordinary wobble between two retrains, and that it
# cannot be fooled by a model that lifts the blend by declining more.


class FakeBlob:
    def __init__(self, name: str, body: bytes = b"") -> None:
        self.name = name
        self._body = body

    def download_as_bytes(self) -> bytes:
        return self._body


class FakeBucket:
    """Just enough of a GCS bucket for `incumbent_metrics`: listing and one download."""

    def __init__(self, versions: dict[str, dict | None]) -> None:
        self._blobs: list[FakeBlob] = []
        self._bodies: dict[str, bytes] = {}
        for version, metrics in versions.items():
            root = f"routers/support/{version}"
            body = json.dumps({"gate_metrics": metrics} if metrics is not None else {}).encode()
            self._bodies[f"{root}/metadata.json"] = body
            self._blobs += [FakeBlob(f"{root}/metadata.json"), FakeBlob(f"{root}/model.ubj")]

    def list_blobs(self, prefix: str) -> list[FakeBlob]:
        return [b for b in self._blobs if b.name.startswith(prefix)]

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self._bodies[name])


def metrics(blended: float, routed: float) -> dict:
    return {"menu_recall": blended, "menu_recall_routed": routed, "coverage": 0.97}


def regression_args(**overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {
        "allow_recall_regression": False,
        "max_recall_regression": 0.02,
        "version_suffix": None,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def guard(before: dict | None, now: Recall | None, **overrides: object) -> None:
    versions = {"20260905T120000Z": before} if before is not None else {}
    gate_no_regression(
        FakeBucket(versions),
        regression_args(**overrides),
        run(version="20260906T120000Z"),
        now,
    )


def test_an_artifact_better_than_the_one_it_replaces_publishes():
    guard(metrics(0.9600, 0.9600), Recall(0.9659, 0.9659, 1545, 1545))


def test_ordinary_wobble_between_two_retrains_is_not_a_regression():
    guard(metrics(0.9600, 0.9600), Recall(0.9450, 0.9450, 1545, 1545))


def test_a_real_drop_is_refused_even_though_it_clears_criterion_one():
    with pytest.raises(Refused, match="menu_recall falls"):
        guard(metrics(0.9600, 0.9600), Recall(0.9300, 0.9300, 1545, 1545))


def test_declining_more_cannot_buy_its_way_past_the_guard():
    """The blend rises because declined sessions score a free 1.0; routed recall collapsed."""
    with pytest.raises(Refused, match="menu_recall_routed falls"):
        guard(metrics(0.9600, 0.9600), Recall(0.9700, 0.8000, 1545, 700))


def test_the_first_artifact_for_an_agent_has_nothing_to_compare_against():
    guard(None, Recall(0.5000, 0.5000, 10, 10))


def test_an_incumbent_from_before_the_field_existed_is_not_a_comparison():
    guard({"coverage": 0.97}, Recall(0.5000, 0.5000, 10, 10))


def test_an_unmeasured_recall_is_not_scored_as_zero():
    """`--skip-recall-gate` leaves one number, and one number is not a comparison."""
    guard(metrics(0.9600, 0.9600), None)


def test_the_operator_can_override_it_deliberately():
    guard(metrics(0.9600, 0.9600), Recall(0.5000, 0.5000, 10, 10), allow_recall_regression=True)


def test_a_sibling_version_is_an_experiment_and_not_a_replacement():
    guard(metrics(0.9600, 0.9600), Recall(0.5000, 0.5000, 10, 10), version_suffix="one-menu")


def test_it_compares_against_the_newest_published_version_not_the_first():
    versions = {
        "20260901T120000Z": metrics(0.5000, 0.5000),
        "20260905T120000Z": metrics(0.9600, 0.9600),
    }
    with pytest.raises(Refused, match="20260905T120000Z"):
        gate_no_regression(
            FakeBucket(versions),
            regression_args(),
            run(version="20260906T120000Z"),
            Recall(0.9300, 0.9300, 1545, 1545),
        )


def test_the_version_being_published_is_not_its_own_incumbent(capsys):
    """--overwrite republishes into a directory that already holds a metadata.json."""
    gate_no_regression(
        FakeBucket({"20260906T120000Z": metrics(0.9900, 0.9900)}),
        regression_args(),
        run(version="20260906T120000Z"),
        Recall(0.9300, 0.9300, 1545, 1545),
    )
    assert "nothing published yet" in capsys.readouterr().out


# --- The drift baseline, recorded at publish because it is not zero --------------------


def replayed(top1=None, wanted=("get_invoice",)):
    return HeldOut(wanted=frozenset(wanted), kept=None, top1=top1)


TOKENS = {"update_email": 40, "track_shipment": 40, "get_invoice": 40, "cancel_order": 40}


def test_the_baseline_measures_declared_tools_the_model_has_no_class_for():
    baseline = publish_router.drift_baseline(
        tuple(BOOSTER_ORDER), {**TOKENS, "brand_new_tool": 40}, []
    )
    assert baseline is not None
    assert baseline.unseen_share == 0.2
    assert (baseline.declared, baseline.unclassed) == (5, 1)


def test_abstain_is_not_counted_as_a_class_the_agent_could_declare():
    """It is in the booster's label list and is never a tool. Counting it would flatter
    the baseline by one every time."""
    baseline = publish_router.drift_baseline(tuple(BOOSTER_ORDER), {**TOKENS, "ABSTAIN": 1}, [])
    assert baseline is not None
    assert baseline.unclassed == 1


def test_a_model_that_covers_its_whole_surface_baselines_at_zero():
    baseline = publish_router.drift_baseline(tuple(BOOSTER_ORDER), TOKENS, [])
    assert baseline is not None and baseline.unseen_share == 0.0


def test_without_a_declaration_tokens_file_there_is_no_baseline_to_record():
    """The alternative is measuring the model against its own class list, which is zero by
    construction and would read later as a surface that has never moved."""
    assert publish_router.drift_baseline(tuple(BOOSTER_ORDER), None, []) is None
    assert publish_router.drift_baseline(tuple(BOOSTER_ORDER), {}, []) is None


def test_confidence_quantiles_come_from_the_held_out_replay():
    baseline = publish_router.drift_baseline(
        tuple(BOOSTER_ORDER), TOKENS, [replayed(0.9), replayed(0.5), replayed(0.1)]
    )
    assert baseline is not None
    assert baseline.confidence_p50 == 0.5
    assert baseline.sessions == 3


def test_a_skipped_recall_gate_still_baselines_the_surface_without_confidences():
    """--skip-recall-gate replays nothing, and the half of the baseline that needs no
    replay is the half that gates."""
    baseline = publish_router.drift_baseline(tuple(BOOSTER_ORDER), {**TOKENS, "new": 1}, [])
    assert baseline is not None
    assert baseline.unseen_share > 0 and baseline.confidence_p50 is None


# --- and travels in metadata.json ------------------------------------------------------


def test_the_baseline_is_written_into_the_artifact():
    baseline = publish_router.drift_baseline(
        tuple(BOOSTER_ORDER), {**TOKENS, "new": 1}, [replayed(0.8)]
    )
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), None, None, baseline)
    assert drift.Signals.from_dict(metadata["drift_baseline"]) == baseline


def test_an_artifact_published_without_one_carries_no_key_rather_than_a_null():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,))
    assert "drift_baseline" not in metadata


def test_the_readers_validator_ignores_the_thirteenth_field():
    """`parse_metadata` reads by key, so an older reader loads a newer artifact."""
    baseline = publish_router.drift_baseline(tuple(BOOSTER_ORDER), TOKENS, [])
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), None, None, baseline)
    assert parse_metadata(metadata).version == "20260901T160000Z"


def test_the_baseline_survives_being_written_as_json():
    baseline = publish_router.drift_baseline(tuple(BOOSTER_ORDER), TOKENS, [replayed(0.4)])
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), None, None, baseline)
    written = json.loads(json.dumps(metadata))
    assert drift.Signals.from_dict(written["drift_baseline"]) == baseline


# --- The ledger row a publish leaves behind ---------------------------------------------


def publish_args(**overrides):
    fields = {
        "dry_run": False,
        "project": "proj",
        "dataset": "ds",
        "cycle_id": "cyc123",
        "agent": "support",
        "environment": "prod",
        "run_id": None,
        "menu_size": 11,
        "menu_count": 24,
        "menu_candidates": DEFAULT_MENU_CANDIDATES,
        "min_menu_recall": 0.90,
        "no_menus": False,
        "settings": "/tmp/settings.json",
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def steps(monkeypatch):
    """Capture what `record_publish` hands the ledger, without a warehouse."""
    written = []

    def fake_record(project, dataset, step, client=None):
        written.append((project, dataset, step))
        return True

    monkeypatch.setattr(publish_router.lifecycle, "record", fake_record)
    return written


def test_a_publish_writes_one_publish_step_against_its_cycle(monkeypatch):
    written = steps(monkeypatch)
    publish_router.record_publish(publish_args(), "ok", "every gate passed", run())
    ((project, dataset, step),) = written
    assert (project, dataset) == ("proj", "ds")
    assert (step.step, step.cycle_id, step.outcome) == ("publish", "cyc123", "ok")
    assert step.version == "20260901T160000Z" and step.run_id == "r-1"
    assert step.source == "publish_router.py"


def test_a_dry_run_leaves_no_trace_because_it_shipped_nothing(monkeypatch):
    written = steps(monkeypatch)
    publish_router.record_publish(publish_args(dry_run=True), "ok", "rehearsed", run())
    assert written == []


def test_a_refusal_before_the_run_is_known_still_records_the_agent(monkeypatch):
    written = steps(monkeypatch)
    publish_router.record_publish(publish_args(run_id="r-9"), "rejected", "settings unreadable")
    step = written[0][2]
    assert (step.agent, step.outcome, step.version) == ("support", "rejected", None)
    assert step.run_id == "r-9" and step.reason == "settings unreadable"


def test_the_shape_that_reached_the_artifact_is_recorded_beside_it(monkeypatch):
    """The settings file the tuner wrote is overwritten next cycle. This is the durable copy."""
    written = steps(monkeypatch)
    publish_router.record_publish(publish_args(), "ok", "fine", run())
    settings = written[0][2].settings_after
    assert settings["menu_size"] == 11 and settings["menu_count"] == 24
    assert settings["min_menu_recall"] == 0.90


def test_a_ledger_row_that_does_not_land_is_ignored_rather_than_raised(monkeypatch):
    """`lifecycle.record` swallows its own failures and answers False; this checks nothing
    downstream of it decides that a missing row is worth stopping for."""
    monkeypatch.setattr(
        publish_router.lifecycle, "record", lambda project, dataset, step, client=None: False
    )
    assert publish_router.record_publish(publish_args(), "ok", "fine", run()) is None


def test_the_recorded_metrics_are_the_gate_numbers_plus_the_baseline():
    baseline = publish_router.drift_baseline(
        tuple(BOOSTER_ORDER), {**TOKENS, "new": 1}, [replayed(0.7)]
    )
    metadata = build_metadata(
        run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured(), saved(), baseline
    )
    metrics = publish_router.ledger_metrics(metadata, baseline)
    assert metrics["menu_recall"] == 0.9187
    assert metrics["unseen_share_baseline"] == 0.2
    assert metrics["confidence_p50"] == 0.7


def test_recorded_metrics_without_a_baseline_are_just_the_gate_numbers():
    metadata = build_metadata(run(), tuple(BOOSTER_ORDER), b"bytes", (MENU,), measured())
    metrics = publish_router.ledger_metrics(metadata, None)
    assert "unseen_share_baseline" not in metrics
    assert metrics["menu_recall"] == 0.9187


def test_the_publish_outcomes_are_words_the_ledger_knows():
    for outcome in ("ok", "rejected", "failed"):
        assert outcome in publish_router.lifecycle.OUTCOMES
