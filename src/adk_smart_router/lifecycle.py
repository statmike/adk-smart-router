"""The retrain ledger: one row per step of one cycle, so a published version can explain itself.

`router_training_runs` records the train step and nothing else. The tuner's decision, including
the refusal that means "keep last cycle's answer", lands in a local JSON file the next cycle
overwrites. The publisher's verdict lands in a GCS object and on stdout. So *why did this version
ship with these settings, and what did the tuner reject to get here* is not answerable from the
warehouse, and the runner that knew is usually gone (DESIGN.md 20.1).

This is the missing record. Each step writes its own row, rather than `retrain.sh` collecting
them at the end, because the three steps are usable standalone and frequently are: a publish run
outside a cycle must still leave a trace. `retrain.sh` mints a `cycle_id` and passes it down; a
script run on its own mints one and becomes a single-step cycle, which is honest about what
happened.

**This table is authoritative for nothing.** `router_training_runs` still gates promotion,
`metadata.json` is still what the loader verifies its `sha256` against, and the settings file is
still what the publisher reads. It follows that a failed ledger write must never fail the step it
describes, so `record` catches everything and returns False. An observability system that can take
a retrain down is a worse trade than a gap in a chart. Same reasoning as `@fail_open` on the hot
path, for the same reason.

Nothing here runs at request time. The plugin does not import this module, and BigQuery is
imported lazily so that installing the package does not drag a query client into a serving path.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DDL",
    "OUTCOMES",
    "STEPS",
    "TABLE",
    "Step",
    "ensure_table",
    "new_cycle_id",
    "read",
    "record",
]

logger = logging.getLogger(__name__)

# One method of google-cloud-bigquery, aliased the way `openings` and `model_store` alias their
# clients: the type stays out of this module's signatures, so a test passes a fake and nothing
# here imports BigQuery at import time.
BigQueryClient = Any

TABLE = "router_lifecycle"

# What a row can describe. `deploy` is written by whatever puts a new artifact in front of
# traffic, which this project does not own; the value exists so an adopter's deploy hook has
# somewhere to say so, and a cycle whose artifact never shipped is visible as its absence.
#
# `withdraw` is the only step that undoes another one, and it is the only step nothing asked a
# person for. `watch_generations.py` writes it when a generation costs more per later turn than
# its own concurrent holdout arm, and the row carries the two arms and the margin it was judged
# against, because a rollback nobody can explain the next morning is not an improvement on a
# rollback that never happened (DESIGN.md 21.5).
STEPS = ("train", "tune", "publish", "backtest", "drift", "deploy", "withdraw")

# `declined` and `rejected` are kept apart on purpose. The tuner declines, which is a normal
# answer meaning "no recommendation beat the last one" and leaves the previous settings standing.
# A gate rejects, which means a model was measured and found wanting. Collapsing them would make
# a healthy cycle and a failed promotion read the same in a chart.
OUTCOMES = ("ok", "declined", "rejected", "failed", "skipped", "due")

# All columns nullable, matching `router_training_runs` and `router_backtest`. A step that cannot
# measure something writes NULL, which is honest; REQUIRED would make a partial row unwritable and
# a partial row is exactly what a failed step has to leave behind.
#
# `recorded_at` rather than the obvious `at`, because `AT` is a reserved word in GoogleSQL and a
# column called that can only ever be read through backticks. Every future hand-written query
# against this table would have to know that, and the first one that does not fails at parse time.
#
# `settings_before`, `settings_after` and `metrics` are STRING holding JSON rather than BigQuery's
# JSON type. The fields differ per step, and a typed column each is a table that grows a column
# every time a script learns to measure something — `router_training_runs` shows that cost, with
# five ADD COLUMN IF NOT EXISTS statements guarding one insert. JSON as a string loses nothing to
# the views, because `JSON_VALUE` reads a STRING path expression just as happily, and it survives
# `insert_rows_json` without depending on how a client version spells a JSON literal.
DDL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  cycle_id STRING,
  step STRING,
  recorded_at TIMESTAMP,
  agent STRING,
  environment STRING,
  outcome STRING,
  reason STRING,
  run_id STRING,
  version STRING,
  settings_before STRING,
  settings_after STRING,
  metrics STRING,
  source STRING
)
PARTITION BY DATE(recorded_at)
CLUSTER BY agent, step
"""

# 90 days, the floor every view in dashboard.sql carries, for the same reason: the worst a
# careless read can do is scan a quarter.
READ_SQL = """
SELECT
  cycle_id, step, recorded_at, agent, environment, outcome, reason, run_id, version,
  settings_before, settings_after, metrics, source
FROM `{project}.{dataset}.{table}`
WHERE recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
  AND (@agent IS NULL OR agent = @agent)
ORDER BY recorded_at DESC
LIMIT @limit
"""


def new_cycle_id() -> str:
    """An id for one pass of train, tune and publish. Twelve hex characters, like `replica_id`."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Step:
    """One step of one cycle: what ran, what it decided, and what it decided it with."""

    cycle_id: str
    step: str
    agent: str
    outcome: str
    environment: str | None = None
    reason: str | None = None
    run_id: str | None = None
    version: str | None = None
    settings_before: dict[str, Any] | None = None
    settings_after: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    source: str | None = None

    def as_row(self) -> dict[str, Any]:
        """The row as `insert_rows_json` wants it.

        Field by field rather than `dataclasses.asdict()`, for the reason `provenance` gives
        about its own records: adding a field to the dataclass must not silently add it to what
        gets written. This one holds operational metadata and no user content, and it stays that
        way by being spelled out.
        """
        return {
            "cycle_id": self.cycle_id,
            "step": self.step,
            "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
            "agent": self.agent,
            "environment": self.environment,
            "outcome": self.outcome,
            "reason": self.reason,
            "run_id": self.run_id,
            "version": self.version,
            "settings_before": _json(self.settings_before),
            "settings_after": _json(self.settings_after),
            "metrics": _json(self.metrics),
            "source": self.source,
        }


def record(
    project: str,
    dataset: str,
    step: Step,
    client: BigQueryClient = None,
) -> bool:
    """Append one row. Returns whether it landed, and never raises.

    Creates the table on the first write that finds it missing, so the ledger appears wherever
    the scripts already run rather than needing a provisioning step of its own. That retry is
    the only one: a second failure is logged and the caller carries on, because the step this
    row describes has already happened and refusing to finish it would be the worse outcome.
    """
    try:
        from google.cloud import bigquery

        writer = client or bigquery.Client(project=project)
        errors = _insert(project, dataset, step, writer)
        if errors:
            # A missing table is the one failure worth another round trip. Everything else is a
            # schema or permission problem, which retrying does not fix.
            ensure_table(project, dataset, writer)
            errors = _insert(project, dataset, step, writer)
        if errors:
            logger.warning("smart_router: lifecycle row not recorded: %s", errors)
            return False
    except Exception:
        logger.exception("smart_router: lifecycle row not recorded")
        return False
    return True


def ensure_table(project: str, dataset: str, client: BigQueryClient = None) -> None:
    """Create the ledger if it is not there. Idempotent, and free."""
    from google.cloud import bigquery

    writer = client or bigquery.Client(project=project)
    writer.query(DDL.format(project=project, dataset=dataset, table=TABLE)).result()


def read(
    project: str,
    dataset: str,
    agent: str | None = None,
    days: int = 90,
    limit: int = 200,
    client: BigQueryClient = None,
) -> list[dict[str, Any]]:
    """Recent steps, newest first. An empty list when the table does not exist yet.

    A missing table is the normal state on an agent that has never retrained, and `status` has
    to be useful in exactly that state (DESIGN.md 20.10), so it is not an error here.
    """
    from google.api_core import exceptions as api
    from google.cloud import bigquery

    reader = client or bigquery.Client(project=project)
    job = reader.query(
        READ_SQL.format(project=project, dataset=dataset, table=TABLE),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("days", "INT64", days),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        ),
    )
    try:
        return [dict(row) for row in job.result()]
    except api.NotFound:
        return []


def _insert(
    project: str,
    dataset: str,
    step: Step,
    writer: BigQueryClient,
) -> list[dict[str, Any]]:
    table = f"{project}.{dataset}.{TABLE}"
    errors: list[dict[str, Any]] = writer.insert_rows_json(table, [step.as_row()])
    return errors


def _json(value: dict[str, Any] | None) -> str | None:
    """JSON text, or NULL. Sorted keys so two identical settings diff as identical."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)
