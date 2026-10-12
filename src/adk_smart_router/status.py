"""One screen answering "is it working, and is it paying".

`qualify` and `doctor` are day-zero commands: they answer whether pruning can pay for an agent
and whether it is wired up. Everything after the first deploy used to be a BI tool somebody had
to build first, which made the adopter's day two a cliff (DESIGN.md 20.10). This is the answer
to that day, and it is a command rather than a ninth view because no view can join the artifact
in GCS to the traffic in BigQuery, and because the case it serves is somebody with one terminal
and five seconds.

Five small reads and one artifact download, all of them read-only. No model call, nothing on the
hot path, and nothing here can change what the router does.

**Every section degrades to a stated absence.** A fresh install has no ledger, no baseline and
possibly no holdout arm, and a status command that renders those as zeroes is worse than one
that refuses to print. The specific trap is the cost line: a sequential A/B runs its control
period with the router uninstalled, so the control arm writes nothing at all and `v_router_g1_daily`
returns NULL rather than a number (DESIGN.md 19.60). That has to reach the reader as "no control
arm" and not as a confident 0.0%.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from adk_smart_router import lifecycle
from adk_smart_router.config import RouterConfig
from adk_smart_router.model_store import ModelStore, RouterModel

__all__ = ["Status", "lines", "read"]

# Aliased for the reason `openings` aliases it: the BigQuery types stay out of this module's
# signatures, so a test passes a fake and nothing here imports the client at import time.
BigQueryClient = Any

DEFAULT_DAYS = 7

# The window's two arms and what a turn cost in each.
#
# Turn-weighted rather than an average of the daily percentages. `v_router_cost_daily` already
# reports a per-turn cost per arm per day, and a mean over those days weights a quiet Sunday like
# a Monday; multiplying back by `turns` recovers the totals the ratio should be taken over. Per
# turn throughout, because sessions differ in length between arms for reasons that have nothing
# to do with routing.
TRAFFIC_SQL = """
SELECT
  arm,
  SUM(sessions) AS sessions,
  SUM(turns) AS turns,
  SAFE_DIVIDE(SUM(billed_per_turn * turns), SUM(turns)) AS billed_per_turn
FROM `{project}.{dataset}.v_router_cost_daily`
WHERE agent = @agent
  AND day >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
GROUP BY arm
"""

# Why sessions landed where they did. The three fail-open sources are collapsed into their class
# because the operational question is "did the embedder answer", and three separate tenths of a
# percent read as noise rather than as the one line worth acting on.
#
# Filtered on `timestamp` rather than on `day`, which matters: the view passes `timestamp` through
# from the partitioning column, so the predicate prunes partitions, and the same filter written
# against `day` would scan the view's whole 90-day floor.
MIX_SQL = """
SELECT
  IF(decision_class = 'fail-open', 'fail-open', decision_source) AS label,
  COUNT(*) AS decisions
FROM `{project}.{dataset}.v_router_decisions`
WHERE agent = @agent
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
GROUP BY label
ORDER BY decisions DESC
"""

# How much of the agent's tool demand the tool cache absorbed. `tools` counts only the tools that
# served something from cache, so a cache nobody configured reads as zero tools rather than as
# every tool the agent called at a hit rate of nothing.
CACHE_SQL = """
SELECT
  COUNT(DISTINCT IF(from_cache > 0, tool_name, NULL)) AS tools,
  SUM(completions) AS completions,
  SUM(from_cache) AS from_cache
FROM `{project}.{dataset}.v_router_tool_cache_daily`
WHERE agent = @agent
  AND day >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
"""

# What the live artifact was fitted on. `metadata.json` carries the row count and not the session
# count or the corpus window, and those are the two numbers that say whether the model saw enough
# and how far back it looked.
RUN_SQL = """
SELECT run_id, n_rows, n_sessions, corpus_since, trained_at
FROM `{project}.{dataset}.router_training_runs`
WHERE agent = @agent AND version = @version
ORDER BY trained_at DESC
LIMIT 1
"""


# --- What was read ----------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One side of the A/B over the window."""

    sessions: int = 0
    turns: int = 0
    billed_per_turn: float | None = None


@dataclass(frozen=True)
class Traffic:
    """Both arms, plus the sessions that belong to neither."""

    arms: Mapping[str, Arm] = field(default_factory=dict)

    def arm(self, name: str) -> Arm:
        return self.arms.get(name, Arm())

    @property
    def sessions(self) -> int:
        return sum(arm.sessions for arm in self.arms.values())

    @property
    def g1(self) -> float | None:
        """Billed input tokens per turn, routed against holdout. None when there is no control.

        Negative is cheaper. None here is the reading a sequential A/B produces, and the caller
        has to print it as an absent control arm rather than as a saving of zero.
        """
        routed = self.arm("routed").billed_per_turn
        holdout = self.arm("holdout").billed_per_turn
        if not routed or not holdout:
            return None
        return (routed - holdout) / holdout


@dataclass(frozen=True)
class Mix:
    """Decision sources over the window, commonest first."""

    counts: Sequence[tuple[str, int]] = ()

    @property
    def total(self) -> int:
        return sum(count for _, count in self.counts)


@dataclass(frozen=True)
class CacheUse:
    """What the tool cache absorbed over the window."""

    tools: int = 0
    completions: int = 0
    from_cache: int = 0


@dataclass(frozen=True)
class Status:
    """Everything the readout prints, gathered but not yet formatted.

    A field is None when the thing it describes could not be read, and the three warehouse
    sections are None specifically when the dashboard views are not deployed, because that is a
    normal state on an install that has traffic and has not run `deploy_sql.sh` yet.
    """

    agent: str
    days: int
    max_age_days: int
    model: RouterModel | None = None
    run: Mapping[str, Any] | None = None
    traffic: Traffic | None = None
    mix: Mix | None = None
    cache: CacheUse | None = None
    ledger: Sequence[Mapping[str, Any]] = ()
    scanned: int = 0


# --- Reading ------------------------------------------------------------------------


def read(
    cfg: RouterConfig,
    agent: str,
    days: int = DEFAULT_DAYS,
    client: BigQueryClient = None,
    store: ModelStore | None = None,
) -> Status:
    """Gather the readout. Read-only, and safe to run against production.

    The artifact is loaded through `ModelStore` rather than by reading `metadata.json` directly,
    so what this reports is the artifact a serving process would accept: same digest check, same
    environment check, same pin. A model that would be refused at serving time reads as no model
    here too, which is the answer worth having.
    """
    from google.cloud import bigquery

    reader = client or bigquery.Client(project=cfg.bq_project)
    model = (store or ModelStore(cfg)).load(agent)

    scanned = 0
    run: Mapping[str, Any] | None = None
    if model is not None:
        rows, read_bytes = _query(reader, RUN_SQL, cfg, agent, days, version=model.metadata.version)
        scanned += read_bytes
        run = rows[0] if rows else None

    traffic_rows, read_bytes = _query(reader, TRAFFIC_SQL, cfg, agent, days)
    scanned += read_bytes
    traffic = (
        None
        if traffic_rows is None
        else Traffic(
            {
                str(row["arm"]): Arm(
                    sessions=int(row["sessions"] or 0),
                    turns=int(row["turns"] or 0),
                    billed_per_turn=_maybe_float(row["billed_per_turn"]),
                )
                for row in traffic_rows
            }
        )
    )

    mix_rows, read_bytes = _query(reader, MIX_SQL, cfg, agent, days)
    scanned += read_bytes
    mix = (
        None
        if mix_rows is None
        else Mix(tuple((str(row["label"]), int(row["decisions"])) for row in mix_rows))
    )

    cache_rows, read_bytes = _query(reader, CACHE_SQL, cfg, agent, days)
    scanned += read_bytes
    cache = None if cache_rows is None else _cache_use(cache_rows)

    return Status(
        agent=agent,
        days=days,
        max_age_days=cfg.max_model_age_days,
        model=model,
        run=run,
        traffic=traffic,
        mix=mix,
        cache=cache,
        ledger=lifecycle.read(cfg.bq_project, cfg.bq_dataset, agent, client=reader),
        scanned=scanned,
    )


def _query(
    reader: BigQueryClient,
    sql: str,
    cfg: RouterConfig,
    agent: str,
    days: int,
    version: str | None = None,
) -> tuple[list[Mapping[str, Any]] | None, int]:
    """Run one read. Returns None for the rows when the table or view is not there.

    A missing view is not a failure to report. An install that has traffic but has never run
    `deploy_sql.sh` is a real and recoverable state, and it needs a line saying so rather than a
    stack trace that buries the four sections that did read.
    """
    from google.api_core import exceptions as api
    from google.cloud import bigquery

    params = [
        bigquery.ScalarQueryParameter("agent", "STRING", agent),
        bigquery.ScalarQueryParameter("days", "INT64", days),
    ]
    if version is not None:
        params.append(bigquery.ScalarQueryParameter("version", "STRING", version))
    job = reader.query(
        sql.format(project=cfg.bq_project, dataset=cfg.bq_dataset),
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    )
    try:
        rows: list[Mapping[str, Any]] = [dict(row) for row in job.result()]
    except api.NotFound:
        return None, 0
    return rows, int(job.total_bytes_processed or 0)


def _cache_use(rows: list[Mapping[str, Any]]) -> CacheUse:
    """One aggregate row, which BigQuery returns even when nothing matched."""
    if not rows:
        return CacheUse()
    row = rows[0]
    return CacheUse(
        tools=int(row["tools"] or 0),
        completions=int(row["completions"] or 0),
        from_cache=int(row["from_cache"] or 0),
    )


# --- Reading the ledger back ---------------------------------------------------------


def last_step(ledger: Sequence[Mapping[str, Any]], step: str) -> Mapping[str, Any] | None:
    """The newest row for one step, or None. `lifecycle.read` returns newest first."""
    return next((row for row in ledger if row.get("step") == step), None)


def newest_cycle(ledger: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Every row of the most recent cycle, in the order the steps run.

    By cycle id rather than by date, because a cycle that starts before midnight and publishes
    after it is one cycle and should read as one line.
    """
    if not ledger:
        return []
    cycle_id = ledger[0].get("cycle_id")
    rows = [row for row in ledger if row.get("cycle_id") == cycle_id]
    return sorted(rows, key=lambda row: lifecycle.STEPS.index(str(row.get("step") or "train")))


def metrics_of(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """A ledger row's `metrics` column, decoded. Empty when it is absent or unreadable.

    Unreadable is possible: the column is JSON held as a string, and this project has to keep
    working against rows written by a version of itself that is no longer installed.
    """
    if row is None:
        return {}
    raw = row.get("metrics")
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# --- Rendering -----------------------------------------------------------------------


def lines(status: Status) -> list[str]:
    """The readout, one string per line. Pure, so the shape is testable without a warehouse."""
    out = [""]
    out += _artifact_lines(status)
    out += [""] + _traffic_lines(status)
    out += [""] + _drift_lines(status)
    out += [""] + _plan_lines(status)
    if status.scanned:
        out += ["", f"  read {status.scanned / 1024**3:.2f} GB"]
    return out + [""]


def _artifact_lines(status: Status) -> list[str]:
    model = status.model
    if model is None:
        return [
            _row("live artifact", "none"),
            _sub("", "nothing published for this agent, or what is published cannot be loaded"),
        ]

    meta = model.metadata
    out = [
        _row(
            "live artifact",
            f"{meta.version}    trained {_day(meta.trained_at)}, {_age(model.age_days)}",
        )
    ]

    run = status.run
    if run is None:
        out.append(
            _sub("from run", f"{meta.rows:,} rows; the training run itself is not in the log")
        )
    else:
        window = _corpus_days(run)
        sessions = run.get("n_sessions")
        counted = f"{int(run['n_rows']):,} rows" if run.get("n_rows") else f"{meta.rows:,} rows"
        if sessions:
            counted += f" / {int(sessions):,} sessions"
        out.append(_sub("from run", f"{run['run_id']}  ·  {counted}{window}"))

    out.append(_sub("gates", _gates(meta.gate_metrics)))
    out.append(_sub("settings", _settings(status)))
    return out


def _gates(metrics: Mapping[str, float]) -> str:
    """The three numbers a publish is judged on, each omitted when it was never measured."""
    parts = []
    if "miss_reduction" in metrics:
        parts.append(f"miss reduction {metrics['miss_reduction']:.2f}")
    if "menu_recall" in metrics:
        parts.append(f"menu recall {metrics['menu_recall']:.4f}")
    if "predicted_g1" in metrics:
        parts.append(f"predicted saving {metrics['predicted_g1']:+.1%}")
    return "  ·  ".join(parts) if parts else "not recorded on this artifact"


def _settings(status: Status) -> str:
    """The menu shape that shipped, and who chose it.

    The shape is read from the artifact rather than from the settings file, because the file is
    overwritten by the next cycle and the artifact is what is actually serving. The tuner's part
    comes from the ledger, which is the only place a refusal is written down at all.
    """
    menus = status.model.menus if status.model is not None else ()
    shape = (
        f"{len(menus)} menus of {max(len(menu) for menu in menus)}"
        if menus
        else "no menus; a bespoke set per session"
    )

    tune = last_step(status.ledger, "tune")
    if tune is None:
        return f"{shape}   (no tuning recorded)"
    when = _day(tune.get("recorded_at"))
    if tune.get("outcome") == "ok":
        return f"{shape}   (tuner set them on {when})"
    carried = next(
        (row for row in status.ledger if row.get("step") == "tune" and row.get("outcome") == "ok"),
        None,
    )
    since = (
        f"carried from {_day(carried.get('recorded_at'))}" if carried else "still on the defaults"
    )
    return f"{shape}   (tuner declined {when}; {since})"


def _traffic_lines(status: Status) -> list[str]:
    traffic = status.traffic
    if traffic is None:
        return [
            _row(f"last {status.days} days", "the dashboard views are not deployed"),
            _sub("", "scripts/deploy_sql.sh src/adk_smart_router/sql/dashboard.sql"),
        ]
    if not traffic.sessions:
        return [_row(f"last {status.days} days", "no sessions logged")]

    routed, holdout = traffic.arm("routed"), traffic.arm("holdout")
    neither = traffic.arm("neither")
    counted = f"{routed.sessions:,} sessions routed  ·  {holdout.sessions:,} holdout"
    if neither.sessions:
        counted += f"  ·  {neither.sessions:,} in neither arm"
    return [
        _row(f"last {status.days} days", counted),
        _sub("cost", _cost(traffic)),
        _sub("decisions", _mix(status.mix)),
        _sub("tool cache", _cache(status.cache)),
    ]


def _cost(traffic: Traffic) -> str:
    """G1 over the window, or the reason there is no G1 to report.

    The holdout arm is the whole measurement. Without it the routed arm's cost per turn is a
    number with nothing to compare it to, and reporting it as a saving is the one way this
    command could ship a confident wrong answer (DESIGN.md 19.60, 20.6).
    """
    g1 = traffic.g1
    if g1 is None:
        return "no control arm, so there is no saving to report"
    holdout = traffic.arm("holdout")
    return f"G1 {g1:+.1%}   over a {holdout.sessions:,}-session control arm"


def _mix(mix: Mix | None) -> str:
    """The commonest decision sources, with fail-open kept whatever its size."""
    if mix is None or not mix.total:
        return "none recorded"
    shown = list(mix.counts[:5])
    fail_open = next((pair for pair in mix.counts if pair[0] == "fail-open"), None)
    if fail_open is not None and fail_open not in shown:
        shown.append(fail_open)
    return "  ·  ".join(f"{label} {count / mix.total:.1%}" for label, count in shown)


def _cache(cache: CacheUse | None) -> str:
    if cache is None or not cache.completions:
        return "no tool calls logged"
    if not cache.from_cache:
        return f"nothing served from cache, over {cache.completions:,} tool calls"
    share = cache.from_cache / cache.completions
    return (
        f"{share:.1%} of {cache.completions:,} tool calls served from cache, "
        f"across {cache.tools} tool(s)"
    )


def _drift_lines(status: Status) -> list[str]:
    row = last_step(status.ledger, "drift")
    if row is None:
        return [
            _row("drift", "never checked"),
            _sub("", "uv run scripts/check_drift.py --agent " + status.agent),
        ]

    signal = metrics_of(row)
    out = [_row("drift", _unseen(signal))]
    confidence = _confidence(signal, status)
    if confidence:
        out.append(_row("", confidence))
    due = row.get("outcome") == "due"
    out.append(
        _sub(
            "verdict",
            f"{'RETRAIN DUE' if due else 'not due'}, checked {_day(row.get('recorded_at'))}",
        )
    )
    return out


def _unseen(signal: Mapping[str, Any]) -> str:
    now = _maybe_float(signal.get("unseen_share"))
    if now is None:
        return "the last check recorded no unseen share"
    baseline = _maybe_float(signal.get("unseen_share_baseline"))
    if baseline is None:
        return f"unseen share {now:.1%}  against no baseline; published before drift tracking"
    moved = (now - baseline) * 100
    return f"unseen share {now:.1%}  against a {baseline:.1%} baseline   ({moved:+.1f})"


def _confidence(signal: Mapping[str, Any], status: Status) -> str:
    """Reported, never gating. §19.63 measured it moving 0.25 for 1.3 points of recall."""
    now = _maybe_float(signal.get("confidence_p50"))
    if now is None:
        return ""
    baseline = status.model.metadata.drift_baseline if status.model is not None else None
    at_publish = baseline.confidence_p50 if baseline is not None else None
    if at_publish is None:
        return f"confidence p50 {now:.2f}  with none recorded at publish"
    return f"confidence p50 {now:.2f}  against {at_publish:.2f}"


def _plan_lines(status: Status) -> list[str]:
    out = [_row("next retrain", _next_retrain(status))]
    cycle = newest_cycle(status.ledger)
    if not cycle:
        return out + [_row("last cycle", "none recorded yet")]
    steps = "  ·  ".join(f"{row.get('step')} {row.get('outcome')}" for row in cycle)
    return out + [_row("last cycle", f"{_day(cycle[0].get('recorded_at'))}   {steps}")]


def _next_retrain(status: Status) -> str:
    """When the clock says to go round again, unless drift already said so.

    Drift is reported ahead of age when both are true, for the reason `check_drift.py` gives:
    the less interesting of two true things is the wrong one to print.
    """
    drift_row = last_step(status.ledger, "drift")
    if drift_row is not None and drift_row.get("outcome") == "due":
        return f"now, on drift  ·  {drift_row.get('reason') or 'see the drift row'}"
    if status.model is None:
        return "when there is a model; nothing is published"
    trained = _parse(status.model.metadata.trained_at)
    if trained is None:
        return (
            f"unknown; the artifact's trained_at is unreadable ({status.model.metadata.trained_at})"
        )
    due = trained + dt.timedelta(days=status.max_age_days)
    return f"{due:%Y-%m-%d}, on age (max_model_age_days {status.max_age_days})"


# --- Plumbing -------------------------------------------------------------------------

# Both indents put the value in the same column, so the sub-rows read as a hanging list under
# the row above rather than as a second table.
_LABEL = 19


def _row(label: str, value: str) -> str:
    return f"{label:<{_LABEL}}{value}"


def _sub(label: str, value: str) -> str:
    return f"  {label:<{_LABEL - 2}}{value}"


def _age(days: float) -> str:
    whole = round(days)
    if whole == 0:
        return "today"
    return f"{whole} day{'' if whole == 1 else 's'} ago"


def _corpus_days(run: Mapping[str, Any]) -> str:
    """How far back the training corpus reached, when both ends of it were recorded."""
    since, trained = _parse(run.get("corpus_since")), _parse(run.get("trained_at"))
    if since is None or trained is None:
        return ""
    return f" over {(trained - since).days} days"


def _day(value: Any) -> str:  # noqa: ANN401
    parsed = _parse(value)
    return f"{parsed:%Y-%m-%d}" if parsed else "an unrecorded date"


def _parse(value: Any) -> dt.datetime | None:  # noqa: ANN401
    """A timestamp from BigQuery or from `metadata.json`, whichever this row came from."""
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _maybe_float(value: Any) -> float | None:  # noqa: ANN401
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
