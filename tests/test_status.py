"""The day-two command, and the two ways it could lie.

Everything here guards one of two things. The reads have to stay cheap and bind their arguments,
because this is a command people will run on a schedule against production. And every line has
to degrade to a stated absence, because the interesting first run is against a fresh install
with no ledger, no baseline and often no holdout arm at all — and a status command that renders
those as zeroes is worse than one that refuses to print (DESIGN.md 20.6, 20.10).
"""

from __future__ import annotations

from types import SimpleNamespace

from google.api_core import exceptions as api

from adk_smart_router import status
from adk_smart_router.config import RouterConfig
from adk_smart_router.drift import Signals

CFG = RouterConfig(bq_project="proj", bq_dataset="ds", gcs_bucket="bucket", max_model_age_days=35)

GATES = {"miss_reduction": 0.41, "menu_recall": 0.9187, "predicted_g1": -0.182}


class FakeJob:
    def __init__(self, rows, missing=False):
        self._rows = rows
        self._missing = missing
        self.total_bytes_processed = 1024**3

    def result(self):
        if self._missing:
            raise api.NotFound("no such view")
        return self._rows


class FakeClient:
    """Answers by table name, because `read` runs five different queries in one pass."""

    def __init__(self, tables=None, missing=()):
        self.tables = tables or {}
        self.missing = missing
        self.sql = []
        self.params = []

    def query(self, sql, job_config=None):
        self.sql.append(sql)
        self.params.append({p.name: p.value for p in (job_config.query_parameters or [])})
        for name in self.missing:
            if name in sql:
                return FakeJob([], missing=True)
        for name, rows in self.tables.items():
            if name in sql:
                return FakeJob(rows)
        return FakeJob([])


def model(menus=(frozenset({"a", "b"}),) * 24, gates=None, baseline=None, trained="2026-09-04"):
    meta = SimpleNamespace(
        version="20260904T200906Z",
        trained_at=f"{trained}T20:09:06+00:00",
        rows=34_912,
        gate_metrics=GATES if gates is None else gates,
        drift_baseline=baseline,
    )
    return SimpleNamespace(metadata=meta, menus=menus, age_days=4.0)


def store(loaded):
    return SimpleNamespace(load=lambda agent: loaded)


def ledger(*rows):
    """Newest first, the way `lifecycle.read` returns them."""
    return [{"cycle_id": "c1", "environment": "prod", **row} for row in rows]


def built(**overrides) -> status.Status:
    fields = {
        "agent": "support",
        "days": 7,
        "max_age_days": 35,
        "model": model(),
        "traffic": status.Traffic({"routed": status.Arm(4118, 9000, 100.0)}),
        "mix": status.Mix((("reused", 82), ("fresh", 14))),
        "cache": status.CacheUse(),
        "ledger": (),
    }
    fields.update(overrides)
    return status.Status(**fields)


def rendered(**overrides) -> str:
    return "\n".join(status.lines(built(**overrides)))


# --- The reads stay cheap, and bind what they filter on ---------------------------------


def test_the_cost_read_is_turn_weighted_rather_than_an_average_of_daily_percentages():
    """A mean over days weights a quiet Sunday like a Monday."""
    assert "SUM(billed_per_turn * turns)" in status.TRAFFIC_SQL


def test_the_decision_mix_filters_on_timestamp_so_partitions_prune():
    """`day` is computed inside the view. Only `timestamp` reaches the partitioning column."""
    assert "timestamp >= TIMESTAMP_SUB" in status.MIX_SQL
    assert "day >=" not in status.MIX_SQL


def test_the_run_lookup_is_pinned_to_the_version_that_is_actually_serving():
    assert "version = @version" in status.RUN_SQL


def test_the_agent_name_is_bound_and_never_pasted_into_the_sql():
    client = FakeClient()
    status.read(CFG, "support", client=client, store=store(model()))
    assert all("support" not in sql for sql in client.sql)
    assert all(params["agent"] == "support" for params in client.params)


def test_every_read_is_summed_into_one_reported_scan():
    """Five reads, and a number an operator can hold against an invoice."""
    client = FakeClient()
    report = status.read(CFG, "support", client=client, store=store(model()))
    assert report.scanned == 4 * 1024**3  # four warehouse reads; the ledger read is not priced
    assert "read 4.00 GB" in "\n".join(status.lines(report))


def test_the_artifact_is_not_read_from_when_nothing_is_published():
    """No version means no run to look up, so the query is not sent at all."""
    client = FakeClient()
    report = status.read(CFG, "support", client=client, store=store(None))
    assert all("router_training_runs" not in sql for sql in client.sql)
    assert report.model is None


def test_the_window_reaches_the_queries_that_take_one():
    client = FakeClient()
    status.read(CFG, "support", days=30, client=client, store=store(model()))
    assert {params["days"] for params in client.params} == {30, 90}  # 90 is the ledger's floor


def test_the_reads_are_assembled_into_the_shape_the_readout_prints():
    client = FakeClient(
        {
            "v_router_cost_daily": [
                {"arm": "routed", "sessions": 4118, "turns": 9000, "billed_per_turn": 100.0},
                {"arm": "holdout", "sessions": 462, "turns": 1000, "billed_per_turn": 120.0},
            ],
            "v_router_decisions": [{"label": "reused", "decisions": 82}],
            "v_router_tool_cache_daily": [{"tools": 2, "completions": 100, "from_cache": 12}],
            "router_training_runs": [{"run_id": "a1f3c2", "n_rows": 34912, "n_sessions": 5204}],
        }
    )
    report = status.read(CFG, "support", client=client, store=store(model()))
    assert report.traffic.arm("holdout").sessions == 462
    assert report.mix.counts == (("reused", 82),)
    assert report.cache.from_cache == 12
    assert report.run["run_id"] == "a1f3c2"


# --- Absences, which is what a first run is made of --------------------------------------


def test_no_published_artifact_says_so_rather_than_printing_an_empty_version():
    out = rendered(model=None)
    assert "live artifact      none" in out
    assert "nothing published for this agent" in out


def test_views_that_were_never_deployed_name_the_script_that_deploys_them():
    """An install with traffic and no `deploy_sql.sh` is a normal state, not a stack trace."""
    client = FakeClient(missing=("v_router_cost_daily",))
    report = status.read(CFG, "support", client=client, store=store(model()))
    assert report.traffic is None
    out = "\n".join(status.lines(report))
    assert "the dashboard views are not deployed" in out
    assert "deploy_sql.sh" in out


def test_an_agent_with_no_traffic_says_so_instead_of_dividing_by_it():
    assert "no sessions logged" in rendered(traffic=status.Traffic({}))


def test_a_missing_control_arm_is_printed_as_a_missing_control_arm():
    """A sequential A/B runs its control with the router uninstalled, so it writes nothing.

    This is the one line where the command could ship a confident wrong number (19.60).
    """
    out = rendered()
    assert "no control arm" in out
    assert "G1" not in out


def test_a_control_arm_that_logged_no_cost_is_also_no_control_arm():
    traffic = status.Traffic(
        {"routed": status.Arm(10, 20, 100.0), "holdout": status.Arm(2, 4, None)}
    )
    assert "no control arm" in rendered(traffic=traffic)


def test_a_real_control_arm_reports_the_saving_with_its_denominator_beside_it():
    traffic = status.Traffic(
        {"routed": status.Arm(4118, 9000, 100.0), "holdout": status.Arm(462, 1000, 120.0)}
    )
    out = rendered(traffic=traffic)
    assert "G1 -16.7%" in out
    assert "462-session control arm" in out


def test_an_artifact_published_before_the_gates_recorded_anything_says_not_recorded():
    assert "not recorded on this artifact" in rendered(model=model(gates={}))


def test_an_artifact_with_no_menus_describes_what_that_means():
    assert "no menus; a bespoke set per session" in rendered(model=model(menus=()))


def test_a_cache_nobody_configured_reads_as_nothing_served_rather_than_as_a_hit_rate_of_zero():
    out = rendered(cache=status.CacheUse(tools=0, completions=400, from_cache=0))
    assert "nothing served from cache, over 400 tool calls" in out


def test_fail_open_is_shown_however_small_it_is():
    """Everything else in the mix is ranked. This one is an endpoint problem and always prints."""
    counts = tuple((f"source_{i}", 100) for i in range(6)) + (("fail-open", 1),)
    assert "fail-open 0.2%" in rendered(mix=status.Mix(counts))


# --- Drift, which is read back rather than re-measured ------------------------------------


def test_an_agent_that_has_never_been_checked_names_the_script_that_checks_it():
    out = rendered()
    assert "drift              never checked" in out
    assert "check_drift.py --agent support" in out


def test_a_check_against_an_artifact_with_no_baseline_says_there_was_no_baseline():
    rows = ledger({"step": "drift", "outcome": "ok", "metrics": '{"unseen_share": 0.274}'})
    assert "against no baseline" in rendered(ledger=rows)


def test_a_check_with_a_baseline_prints_the_move_in_points():
    rows = ledger(
        {
            "step": "drift",
            "outcome": "ok",
            "metrics": '{"unseen_share": 0.274, "unseen_share_baseline": 0.261}',
        }
    )
    out = rendered(ledger=rows)
    assert "unseen share 27.4%  against a 26.1% baseline   (+1.3)" in out
    assert "not due" in out


def test_confidence_is_shown_against_publish_and_never_gates():
    rows = ledger(
        {
            "step": "drift",
            "outcome": "ok",
            "metrics": '{"unseen_share": 0.2, "confidence_p50": 0.79}',
        }
    )
    baseline = Signals(unseen_share=0.26, declared=100, unclassed=26, confidence_p50=0.81)
    out = rendered(ledger=rows, model=model(baseline=baseline))
    assert "confidence p50 0.79  against 0.81" in out


def test_a_due_verdict_is_shouted_and_becomes_the_reason_to_retrain():
    rows = ledger(
        {
            "step": "drift",
            "outcome": "due",
            "reason": "unseen share 0.773 is 0.512 above baseline",
            "metrics": '{"unseen_share": 0.773, "unseen_share_baseline": 0.261}',
        }
    )
    out = rendered(ledger=rows)
    assert "RETRAIN DUE" in out
    assert "next retrain       now, on drift" in out
    assert "0.512 above baseline" in out


def test_metrics_that_cannot_be_read_do_not_take_the_section_down_with_them():
    """The column is JSON in a string, written by a version of this that may not be installed."""
    assert status.metrics_of({"metrics": "{not json"}) == {}
    assert status.metrics_of({"metrics": "[1, 2]"}) == {}
    assert status.metrics_of(None) == {}


# --- The cycle, and when to go round again -------------------------------------------------


def test_with_no_drift_reading_the_clock_is_what_says_when():
    out = rendered()
    assert "next retrain       2026-10-09, on age (max_model_age_days 35)" in out


def test_a_cycle_reads_as_one_line_in_the_order_the_steps_run():
    rows = ledger(
        {"step": "publish", "outcome": "ok", "recorded_at": "2026-09-04T21:00:00+00:00"},
        {"step": "tune", "outcome": "declined", "recorded_at": "2026-09-04T20:30:00+00:00"},
        {"step": "train", "outcome": "ok", "recorded_at": "2026-09-04T20:00:00+00:00"},
    )
    out = rendered(ledger=rows)
    assert "last cycle         2026-09-04   train ok  ·  tune declined  ·  publish ok" in out


def test_a_cycle_is_grouped_by_its_id_rather_than_by_its_date():
    """A cycle that starts before midnight and publishes after it is still one cycle."""
    rows = [
        {
            "cycle_id": "new",
            "step": "publish",
            "outcome": "ok",
            "recorded_at": "2026-09-05T00:10:00+00:00",
        },
        {
            "cycle_id": "new",
            "step": "train",
            "outcome": "ok",
            "recorded_at": "2026-09-04T23:50:00+00:00",
        },
        {
            "cycle_id": "old",
            "step": "train",
            "outcome": "ok",
            "recorded_at": "2026-09-01T10:00:00+00:00",
        },
    ]
    cycle = status.newest_cycle(rows)
    assert [row["step"] for row in cycle] == ["train", "publish"]


def test_a_tuner_that_declined_says_where_the_settings_it_kept_came_from():
    rows = ledger(
        {"step": "tune", "outcome": "declined", "recorded_at": "2026-09-04T20:30:00+00:00"},
        {"step": "tune", "outcome": "ok", "recorded_at": "2026-08-28T20:30:00+00:00"},
    )
    out = rendered(ledger=rows)
    assert "24 menus of 2   (tuner declined 2026-09-04; carried from 2026-08-28)" in out


def test_a_tuner_that_has_never_agreed_with_itself_says_the_settings_are_the_defaults():
    rows = ledger(
        {"step": "tune", "outcome": "declined", "recorded_at": "2026-09-04T20:30:00+00:00"}
    )
    assert "still on the defaults" in rendered(ledger=rows)


def test_an_agent_that_has_never_tuned_says_so_rather_than_crediting_the_tuner():
    assert "(no tuning recorded)" in rendered()


def test_the_training_run_carries_its_denominators_and_its_window():
    run = {
        "run_id": "a1f3c2",
        "n_rows": 34_912,
        "n_sessions": 5_204,
        "corpus_since": "2026-07-31T20:09:06+00:00",
        "trained_at": "2026-09-04T20:09:06+00:00",
    }
    out = rendered(run=run)
    assert "a1f3c2  ·  34,912 rows / 5,204 sessions over 35 days" in out


def test_a_run_that_has_aged_out_of_the_log_still_reports_what_the_artifact_knows():
    out = rendered(run=None)
    assert "34,912 rows; the training run itself is not in the log" in out
