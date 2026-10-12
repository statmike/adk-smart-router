#!/usr/bin/env python
"""Build-order task 9 — turn a finished training run into a published artifact.

`train_router.sql` trains, gates, and writes the booster to
`gs://{bucket}/routers/{agent}/{version}/`. It cannot finish the job, for two
reasons.

The booster it writes is unreadable. `model.bst` is XGBoost's pre-1.0 binary
format, and XGBoost dropped support for reading that in 2.1 (DESIGN.md 18.9), so
this project's own XGBoost raises on it. `scripts/convert_booster.py` rewrites it
as UBJSON in a throwaway environment holding an old XGBoost, and `model.ubj` is
what gets published and served.

And three of the twelve fields `model_store` reads are not properties of the
query at all. `sha256` and `size_bytes` describe the exported object, which
BigQuery has no way to measure. `menus` is a second pass over the corpus,
choosing the handful of fixed tool sets the router snaps sessions to, and it
belongs here because it has to agree with the label order settled below
(DESIGN.md 18.17).

It is also where gate criterion 1 lives. The training procedure gates on the
model's raw top-N ranking, which is the harshest measurement available to a
query that has no menus; the criterion is defined on the menu the router ships,
and the menu is chosen here (DESIGN.md 19.11). So this refuses to publish an
artifact whose menus, driven by the booster it is about to write, do not hold
enough of what held-out sessions asked for.

That bar is absolute, and an absolute bar cannot see a slow decay. So there is a
second refusal beside it: an artifact may not fall more than a couple of points
of recall below the one it is replacing, even while clearing the bar.

Recall alone would only ever refuse in one direction, and the cheapest route to
a perfect recall is to decline every session and save nothing. So gate criterion
3 sits beside it, on the same replay of the holdout: the steady-state saving the
artifact predicts, in the same unit as G1, which has to reach -10%. Pricing a
prune needs to know what a declaration costs, which is an agent's property and
not the router's, so it is measured only when `--declaration-tokens` says
(DESIGN.md 19.5, 19.13).

So this script converts, then writes `metadata.json` LAST. A version directory
counts as publishable only once both `metadata.json` and `model.ubj` are present,
so completeness is the atomicity mechanism (DESIGN.md D11). There is no pointer
file to update and no window where a half-written version looks whole.

The reconciliation is the reason this is a script and not three lines of shell.
`predictor.py` zips class names against the booster's probability vector by
position. BigQuery emits `router_training_runs.classes` in ASCENDING order and
the exported `assets/model_metadata.json` lists `class_names` in DESCENDING
order (DESIGN.md 18.8). Publishing the run record's array would reverse the
label mapping: every prediction wrong, every digest valid, no exception
anywhere. The exported assets are therefore the only source for the order, and
the run record is used only as a set to check it against.

Cost is a few cents of BigQuery metadata reads plus one 200 KB download.
Authentication is ADC only. No service-account key is created, read or accepted.

Usage:
    uv run python scripts/publish_router.py --dry-run
    uv run python scripts/publish_router.py --confirm
    uv run python scripts/publish_router.py --confirm --run-id <uuid>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.cloud import bigquery, storage

# The package is not installed into the venv; pytest reaches it through
# `pythonpath` in pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from fleet_cache_watch import Verdict  # noqa: E402
from fleet_cache_watch import measure as measure_fleet  # noqa: E402

from adk_smart_router import drift, lifecycle  # noqa: E402
from adk_smart_router import menus as menu_rules  # noqa: E402
from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore, parse_metadata  # noqa: E402
from adk_smart_router.predictor import Prediction  # noqa: E402
from adk_smart_router.pruner import contained_mass, decide, worth_pruning  # noqa: E402

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# The shipped pool shape and recall bar, used when neither a flag nor a `--settings` file
# says otherwise. Named rather than inlined as argparse defaults because `--settings` has to
# tell an explicit flag apart from an unspecified one, which argparse defaults make
# impossible: the flags below default to None and land on these only after the merge.
#
# `--menu-size` is parity with RouterConfig.max_tools_kept and with `top_n` in
# train_router.sql: a menu larger than the pruner keeps is a menu the pruner would trim, and
# a smaller one gives away coverage the gate was calibrated for. 8 menus was where coverage
# flattened on the 25-tool corpus (DESIGN.md 18.17); on the 100-tool surface menu COUNT is
# what buys recall and menu size mostly costs saving, so the pool is 24 x 11 (19.16, 19.21).
DEFAULT_MENU_SIZE = 11
DEFAULT_MENU_COUNT = 24
# How the pool's candidate menus are built. This was `padded` until 2026-09-07, on the
# grounds that it was what every artifact this project had published was built with and that
# DESIGN.md 19.47 found the two arms genuinely trade on a narrow tool surface. E1 then put a
# hundred tools nobody here wrote in front of it: `padded` reached menu recall 0.6942 and this
# script refused to publish it, while `merged` reached 0.9095 at the same size and count
# (19.66). A default that fails the gate on the shape the product is sold for is the wrong
# default, and the trade 19.47 measured is narrow enough that `tune_settings.py` can price it
# per agent from the second retrain cycle onward. Narrow surfaces lose little; wide ones stop
# hitting a refusal they have no way to diagnose.
DEFAULT_MENU_CANDIDATES = "merged"
MENU_CANDIDATES = ("padded", "merged")
# Gate criterion 1 (DESIGN.md 19.5). This was 0.98 until 2026-09-05, which sounds like the
# safe direction for a default and was not: 0.98 refuses every artifact this project has
# ever shipped, so in practice it was either overridden by hand or it stopped the release.
# A gate nobody can pass teaches nobody anything.
#
# 0.90 is the bar the live A/B actually cleared, at menus of 11 for -23.4% billed tokens
# (DESIGN.md 19.21). Setting the default there turns the criterion into the thing it was
# meant to be, which is a non-regression check against what we know works, rather than an
# aspiration. `tune_settings.py` has assumed 0.90 all along; this is also where the two
# constants stop disagreeing, so the effective bar no longer depends on whether an
# operator remembered to pass --settings.
DEFAULT_MIN_MENU_RECALL = 0.90

# How far recall may fall against the artifact this one replaces. The absolute bar above
# cannot see a slow decay: an artifact three points below its predecessor and still over
# 0.90 passes, and so does the next one. Two points is roughly the spread between
# consecutive retrains on the same corpus, so it lets ordinary run-to-run wobble through
# and catches a step change.
DEFAULT_MAX_RECALL_REGRESSION = 0.02

# Gate criterion 3 (DESIGN.md 19.5): predicted steady-state saving, at most -10%. Negative
# is the good direction, so the comparison is `predicted > bar` refuses. The shipped
# artifact measures around -31% against this bar, which is the point: criterion 3 is not
# meant to be a near miss for a healthy model, it is meant to catch a model that declines
# so much of its traffic that it saves nothing worth the risk it takes.
DEFAULT_MIN_PREDICTED_SAVING = -0.10

# What a cached input token bills at, relative to an uncached one. Criterion 3 prices a
# removed declaration at this rate because in steady state the declarations sit inside the
# cached prefix, so removing one saves a tenth of a token, not a token. The same 0.10 is in
# `RouterConfig.cached_token_multiplier` and in every offline scorer in `scripts/`.
CACHED_TOKEN_WEIGHT = 0.10

# How many held-out sessions criterion 3 needs before it is allowed to refuse anything. The
# sessions it can use are only those billed for the whole tool list, which on a corpus that
# spans a surface change or an A/B is a fraction of the holdout, and a gate resting on
# twenty sessions refuses good artifacts for noise. Below this the number is still measured,
# printed and written into the artifact; it just does not get a vote.
MIN_SAVING_SESSIONS = 100

# What `EXPORT MODEL` writes, and what this script publishes beside it. They are
# the same trees in two formats; only the second one can be read by an XGBoost
# built this decade (DESIGN.md 18.9).
EXPORTED_BOOSTER_NAME = "model.bst"
SERVED_BOOSTER_NAME = "model.ubj"
METADATA_NAME = "metadata.json"
ASSET_NAME = "assets/model_metadata.json"

# The newest XGBoost that can still read the legacy binary format. 2.1 removed
# it. Pinned exactly because this is a format-compatibility shim, and a floating
# version would eventually float past the thing it exists to do.
LEGACY_XGBOOST_PIN = "xgboost==1.7.6"
CONVERTER = Path(__file__).resolve().parent / "convert_booster.py"

# What the exported assets must say about themselves. Anything else means the
# procedure trained a model this serving stack cannot read.
EXPECTED_MODEL_TYPE = "boosted_tree_classifier"
EXPECTED_LABEL_COL = "tool_name"

# `backend` in metadata.json. model_store refuses anything else, because the
# value records which embedding stack produced the vectors the model was fit on.
BACKEND = "vertex"

# Numbers copied from the run record into `gate_metrics`. Every one is a float
# or NULL in BigQuery; NULLs are dropped, because the contract wants numbers.
# Not the whole of `gate_metrics`: `menu_recall` and `predicted_g1` are measured here,
# not in BigQuery, and join them in `gate_metrics_with_recall`.
GATE_METRIC_COLUMNS = (
    "coverage",
    "baseline_coverage",
    "baseline_coverage_corpus",
    "lift",
    "miss_reduction",
    "macro_recall",
    "abstain_precision",
    # Coverage under the rule the pruner serves when the artifact ships no menus:
    # `top_k_by_mass`, which stops at `keep_probability_mass` and so usually keeps
    # fewer than `top_n`. `coverage` above is a strict top-N cut and is therefore an
    # upper bound on it, by 6.3 points on the reference corpus (DESIGN.md 19.74).
    "coverage_mass",
    # Not metrics, the settings the metrics above were measured at. `top_n` and
    # `keep_probability_mass` in `train_router.sql` have to equal
    # `RouterConfig.max_tools_kept` and `RouterConfig.keep_probability_mass`, or the gate
    # is calibrated for a set size that is never served, and until these were published
    # nothing could tell. `startup_checks.artifact_skew` compares them at serve time
    # (DESIGN.md 19.72, 19.74).
    "top_n",
    "keep_probability_mass",
)

# The label the model uses for "no opinion". Never a tool, so never in a menu.
ABSTAIN = "ABSTAIN"

# ADK's control plane. Always declared and never prunable, so a menu slot spent on it
# would be a slot bought and thrown away.
TRANSFER = "transfer_to_agent"

# One session's tool calls, in order, repeats included. The window ends at the run's
# `trained_at` so the menus describe the same traffic the booster was fit on, and the
# exclusions are `train_router.sql`'s: successful calls only, no control plane, and no
# cache hits, because a cache hit is the router reading back its own echo (DESIGN.md 11.5).
CORPUS_SQL = """
SELECT session_id, ARRAY_AGG(tool_name ORDER BY timestamp) AS tools
FROM `{project}.{dataset}.v_tool_completed`
WHERE agent = @agent
  AND status = 'OK'
  AND tool_name != '{transfer}'
  AND timestamp >= @since
  AND timestamp <= @until
  AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
GROUP BY session_id
"""


# The held-out half of the flattened training table: one row per session carrying its
# first message's vector, plus the set of tools that session went on to want. The eval
# split is the training procedure's own `split_col`, so this is the same holdout the
# booster never saw. One row per session because a session's rows all carry the same
# vector, one per tool it wanted (DESIGN.md 19.2).
#
# `spend` is what criterion 3 is denominated in: how many turns the session ran, and what
# its prompts actually billed, cached tokens weighted at `{rate}`.
#
# The HAVING is the load-bearing line. A denominator for "what does pruning save" has to be
# what the agent bills when it prunes nothing, and a training corpus does not consist only
# of such sessions. This warehouse holds three tool surfaces: a 25-tool agent, the 100-tool
# agent shipping now, and the routed arms of every A/B in between, whose prompts were
# already pruned. Averaging those together answered a question nobody asked and, measured,
# claimed -76% for an artifact worth about -27%.
#
# `{min_prompt}` is the total size of the tool list this artifact would prune, so a session
# every one of whose prompts cleared it was demonstrably billed for the whole list. That
# test is self-calibrating: it excludes the older surface because its prompts are too small
# to have carried these declarations, and it excludes an already-pruned session for exactly
# the same reason, without needing to know which experiment either came from. A session
# that declined and shipped the whole list stays, which is right, because it was billed for
# the whole list.
#
# It is a LEFT JOIN, so a session the filter drops still measures for criterion 1 and is
# counted out of criterion 3 rather than counted into it as free.
HELD_OUT_SQL = """
WITH held_out AS (
  SELECT * FROM `{project}.{dataset}.router_flat_{agent}`
  WHERE split_col
  QUALIFY ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY session_id) = 1
),
wanted AS (
  SELECT session_id, ARRAY_AGG(DISTINCT tool_name) AS labels
  FROM `{project}.{dataset}.router_flat_{agent}`
  WHERE split_col
  GROUP BY session_id
),
spend AS (
  SELECT
    session_id,
    COUNT(*) AS turns,
    SUM(usage_prompt_tokens - LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens)
        + LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens) * {rate}) AS billed
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = '{agent}' AND usage_prompt_tokens IS NOT NULL
  GROUP BY session_id
  HAVING MIN(usage_prompt_tokens) >= {min_prompt}
)
SELECT h.* EXCEPT (tool_name, split_col), w.labels, s.turns, s.billed
FROM held_out h
JOIN wanted w USING (session_id)
LEFT JOIN spend s USING (session_id)
"""


class Refused(Exception):
    """A refusal to publish. The message is the whole explanation."""


def apply_settings(args: argparse.Namespace) -> None:
    """Fill in the numbers `tune_settings.py` chose, wherever a flag did not.

    Precedence is explicit flag, then the settings file, then the shipped default. That order
    is the point of the feature: a tuner that cannot be overridden at the moment of publishing
    is one nobody can take the wheel from, and the whole reason the numbers are written to a
    file rather than applied automatically is that a human stays in the loop.

    Every value taken from the file is printed with where it came from. ROADMAP's condition on
    V2-7 was that the chosen values be visible and overridable, and a self-tuning router that
    publishes a shape nobody can see in the log fails it.

    `miss_cost_tokens` is read and checked but deliberately not applied. It is a runtime config
    value rather than a publish parameter, so honouring it here would mean threading it through
    ArtifactMetadata into RouterConfig, which makes loading an artifact quietly change runtime
    policy. DESIGN.md 19.21 measured that price as inert on any surface this wide, so that is
    real plumbing for a number that currently changes nothing. Mismatches warn instead.
    """
    if not args.settings:
        args.menu_size = args.menu_size if args.menu_size is not None else DEFAULT_MENU_SIZE
        args.menu_count = args.menu_count if args.menu_count is not None else DEFAULT_MENU_COUNT
        if args.menu_candidates is None:
            args.menu_candidates = DEFAULT_MENU_CANDIDATES
        if args.min_menu_recall is None:
            args.min_menu_recall = DEFAULT_MIN_MENU_RECALL
        return

    path = Path(args.settings)
    try:
        chosen = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as broken:
        raise Refused(f"could not read settings from {path}: {broken}") from broken

    # Publishing agent A with agent B's tuned shape would be silent and wrong, and the file
    # records which agent it was tuned on precisely so this can be caught.
    tuned_for = chosen.get("agent")
    if tuned_for != args.agent:
        raise Refused(
            f"{path} was tuned for agent {tuned_for!r}, and this is publishing {args.agent!r}. "
            "Tuned settings do not carry across agents: they are a statement about one tool "
            "surface and one traffic shape."
        )

    # The two the tuner chose sit at the top level; the bar it chose them under is evidence,
    # because it is an input to the choice rather than an output of it.
    evidence = chosen.get("evidence", {})
    offered_candidates = chosen.get("menu_candidates")
    if offered_candidates is not None and offered_candidates not in MENU_CANDIDATES:
        raise Refused(
            f"{path} asks for menu_candidates={offered_candidates!r}, and the only ways to "
            f"build a pool are {', '.join(MENU_CANDIDATES)}."
        )

    print(f"==> settings from {path}")
    for flag, offered, fallback in (
        ("menu_size", chosen.get("menu_size"), DEFAULT_MENU_SIZE),
        ("menu_count", chosen.get("menu_count"), DEFAULT_MENU_COUNT),
        ("menu_candidates", offered_candidates, DEFAULT_MENU_CANDIDATES),
        ("min_menu_recall", evidence.get("min_menu_recall"), DEFAULT_MIN_MENU_RECALL),
    ):
        if getattr(args, flag) is not None:
            print(f"    {flag}: {getattr(args, flag)} from the command line, over {offered}")
        elif offered is not None:
            setattr(args, flag, offered)
            print(f"    {flag}: {offered} from {path.name}")
        else:
            setattr(args, flag, fallback)
            print(f"    {flag}: {fallback}, the shipped default; the file did not say")

    priced = chosen.get("miss_cost_tokens")
    running = load_config().miss_cost_tokens
    if priced is not None and priced != running:
        print(
            f"    note: tuned against a miss priced at {priced}, and this environment runs "
            f"{running}. Publishing does not change that; set SMART_ROUTER_MISS_COST_TOKENS "
            "if the tuned price is the one you want."
        )


@dataclass(frozen=True)
class TrainingRun:
    """One row of `router_training_runs`, the fields publishing needs."""

    run_id: str
    agent: str
    version: str
    trained_at: str
    corpus_since: str | None
    environment: str
    embedding_model: str
    embedding_dims: int
    rows: int
    classes: tuple[str, ...]
    export_uri: str
    gate_metrics: dict[str, float]


# --- BigQuery -----------------------------------------------------------------


RUN_COLUMNS = (
    "run_id",
    "agent",
    "version",
    "trained_at",
    "corpus_since",
    "environment",
    "embedding_model",
    "embedding_dims",
    "n_rows",
    "classes",
    "export_uri",
)


def available_metrics(client: bigquery.Client, table: str) -> tuple[str, ...]:
    """The names from `GATE_METRIC_COLUMNS` this particular runs table actually has.

    `router_training_runs` is created with CREATE IF NOT EXISTS and upgraded in place by an
    ALTER near the end of `train_router.sql`, so a table written by an older procedure is
    missing whatever columns have been added since, and a SELECT that names one fails
    outright. Asking the table first turns that into the case the rest of this script already
    handles: a metric nobody measured is absent from `gate_metrics` and gated on by nothing.
    """
    present = {field.name for field in client.get_table(table).schema}
    return tuple(name for name in GATE_METRIC_COLUMNS if name in present)


def fetch_run(client: bigquery.Client, args: argparse.Namespace) -> TrainingRun:
    """The run to publish: a named one, or the newest that passed its gate."""
    table = f"{args.project}.{args.dataset}.router_training_runs"
    metrics = available_metrics(client, table)
    sql = f"""
        SELECT {", ".join((*RUN_COLUMNS, *metrics))}
        FROM `{table}`
        WHERE agent = @agent
          AND environment = @environment
          AND gate_passed
          AND export_uri IS NOT NULL
          AND (@run_id IS NULL OR run_id = @run_id)
        ORDER BY trained_at DESC
        LIMIT 1
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", args.agent),
                bigquery.ScalarQueryParameter("environment", "STRING", args.environment),
                bigquery.ScalarQueryParameter("run_id", "STRING", args.run_id),
            ]
        ),
    )
    rows = list(job.result())
    if not rows:
        raise Refused(
            f"no gate-passing run with an export for agent {args.agent!r} "
            f"in environment {args.environment!r}"
        )
    row = rows[0]
    return TrainingRun(
        run_id=row["run_id"],
        agent=row["agent"],
        version=row["version"],
        # model_store parses this with datetime.fromisoformat, and an
        # unparseable value reads as infinitely old rather than raising.
        trained_at=row["trained_at"].isoformat(),
        corpus_since=row["corpus_since"].isoformat() if row["corpus_since"] else None,
        environment=row["environment"],
        embedding_model=row["embedding_model"],
        embedding_dims=int(row["embedding_dims"]),
        rows=int(row["n_rows"]),
        classes=tuple(row["classes"]),
        export_uri=row["export_uri"],
        gate_metrics={c: float(row[c]) for c in metrics if row[c] is not None},
    )


def menu_window(args: argparse.Namespace, run: TrainingRun) -> tuple[datetime, datetime]:
    """The traffic window the menus are fitted on: lookback, floored by the run's own."""
    until = datetime.fromisoformat(run.trained_at)
    since = until - timedelta(days=args.menu_lookback_days)
    if run.corpus_since:
        since = max(since, datetime.fromisoformat(run.corpus_since).replace(tzinfo=since.tzinfo))
    return since, until


def fetch_sessions(
    client: bigquery.Client, args: argparse.Namespace, run: TrainingRun
) -> list[list[str]]:
    """One list of tool calls per session, over the window the booster was trained on.

    Costs a few cents. The window ends at the run's `trained_at` and starts at
    `--menu-lookback-days` back from it, so the menus and the model describe the same
    traffic. It does not have to match exactly: a menu chosen from slightly different
    weeks is a slightly worse menu, not a wrong one.

    The exception is a run given an explicit `corpus_since`, which is how a training set
    is fenced off from history that no longer applies. V2-4 replaced a 25-tool surface
    with a 100-tool one, and the 35-day lookback would have fitted menus for the new
    model on four times more sessions from the old surface than the new (DESIGN.md
    19.16). When a run names a floor, the menus honour it.
    """
    since, until = menu_window(args, run)
    job = client.query(
        CORPUS_SQL.format(project=args.project, dataset=args.dataset, transfer=TRANSFER),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", run.agent),
                bigquery.ScalarQueryParameter("since", "TIMESTAMP", since),
                bigquery.ScalarQueryParameter("until", "TIMESTAMP", until),
            ]
        ),
    )
    return [list(row["tools"]) for row in job.result()]


def choose_menus(
    sessions: list[list[str]],
    classes: tuple[str, ...],
    size: int,
    count: int,
    built: str = DEFAULT_MENU_CANDIDATES,
) -> tuple[frozenset[str], ...]:
    """The canonical menus to publish, chosen from the corpus and pinned to the label set.

    Calls to tools the booster cannot predict are dropped first. A menu slot spent on
    one is a slot that can never be chosen deliberately, only stumbled into, and the
    whole point of a fixed menu is that its contents were paid for on purpose.

    Why menus at all: pruning to the best tools for each session gave the cache one
    declaration list per session, and a declaration list is part of Gemini's implicit
    cache key. That cost more than the prune saved (DESIGN.md 18.16). A handful of menus
    keeps the saving and gives the cache a handful of prefixes to keep warm (18.17).

    `built` decides what the chooser gets to choose between. `padded` gives each session its
    own set with the spare slots filled from the popularity ranking, and `merged` fuses
    co-occurring sets into candidates instead. The set cover afterwards is identical, and
    DESIGN.md 19.47 has the measurement of which is worth taking on which surface.
    """
    known = set(classes) - {ABSTAIN, TRANSFER}
    filtered = [[call for call in calls if call in known] for calls in sessions]
    corpus = [calls for calls in filtered if calls]
    offered = menu_rules.bucket(corpus, size) if built == "merged" else None
    return menu_rules.choose(corpus, size=size, count=count, candidates=offered)


# --- GCS ----------------------------------------------------------------------


def split_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/a/b` -> `("bucket", "a/b")`."""
    if not uri.startswith("gs://"):
        raise Refused(f"export_uri is not a gs:// URI: {uri!r}")
    bucket, _, prefix = uri[len("gs://") :].partition("/")
    if not bucket or not prefix:
        raise Refused(f"export_uri has no object prefix: {uri!r}")
    return bucket, prefix.rstrip("/")


def read_assets(bucket: Any, prefix: str) -> dict[str, Any]:  # noqa: ANN401
    """The exported `assets/model_metadata.json`, decoded."""
    blob = bucket.blob(f"{prefix}/{ASSET_NAME}")
    if not blob.exists():
        raise Refused(f"{prefix}/{ASSET_NAME} is missing; the export did not complete")
    return json.loads(blob.download_as_bytes())


def read_booster(bucket: Any, prefix: str) -> bytes:  # noqa: ANN401
    """The exported booster, in the legacy format BigQuery wrote it in."""
    blob = bucket.blob(f"{prefix}/{EXPORTED_BOOSTER_NAME}")
    if not blob.exists():
        raise Refused(f"{prefix}/{EXPORTED_BOOSTER_NAME} is missing; the export did not complete")
    return blob.download_as_bytes()


# --- Format conversion ----------------------------------------------------------


def convert_booster(legacy: bytes, classes: tuple[str, ...], dims: int) -> bytes:
    """Rewrite the legacy booster as UBJSON, in a throwaway old-XGBoost environment.

    `uv run --isolated --no-project` builds that environment from nothing, so it
    cannot pick up this project's XGBoost and the conversion cannot be affected
    by whatever the venv happens to hold. It costs a few seconds and no network
    after the first run, because uv caches the wheel.

    The converter reports what it found inside the booster. Checking that against
    the reconciled class list catches a mismatched pair of files — an assets JSON
    from one export beside a booster from another — which every other check here
    would happily wave through.
    """
    with tempfile.TemporaryDirectory(prefix="publish_router_") as workdir:
        source = Path(workdir) / EXPORTED_BOOSTER_NAME
        destination = Path(workdir) / SERVED_BOOSTER_NAME
        report_path = Path(workdir) / "report.json"
        source.write_bytes(legacy)

        command = [
            "uv", "run", "--isolated", "--no-project", "--quiet",
            "--with", LEGACY_XGBOOST_PIN,
            "python", str(CONVERTER), str(source), str(destination), str(report_path),
        ]  # fmt: skip
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise Refused(
                f"converting {EXPORTED_BOOSTER_NAME} failed "
                f"({LEGACY_XGBOOST_PIN}, exit {done.returncode}):\n{done.stderr.strip()}"
            )

        report = json.loads(report_path.read_text())
        if report["num_features"] != dims or report["num_class"] != len(classes):
            raise Refused(
                f"the booster holds {report['num_features']} features and "
                f"{report['num_class']} classes, but the export describes {dims} features "
                f"and {len(classes)} classes. These files are not from the same export."
            )
        print(
            f"    converted to {SERVED_BOOSTER_NAME} with {report['xgboost_version']}: "
            f"{report['num_class']} classes, {report['objective']}"
        )
        return destination.read_bytes()


# --- Reconciliation -----------------------------------------------------------


def reconcile_classes(
    assets: dict[str, Any], run: TrainingRun, allow_unlearned: bool
) -> tuple[str, ...]:
    """Return the booster's label order, having checked it against the run record.

    The order comes from the assets and only from the assets. The run record's
    array is a set to compare against, because its order is wrong for this
    purpose (DESIGN.md 18.8).
    """
    names = assets.get("class_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise Refused(f"{ASSET_NAME} has no usable class_names")
    if len(set(names)) != len(names):
        raise Refused(f"{ASSET_NAME} class_names contains duplicates")

    exported, recorded = set(names), set(run.classes)
    unknown = sorted(exported - recorded)
    if unknown:
        raise Refused(
            f"the booster predicts labels the corpus never contained: {unknown}. "
            "The run record and the artifact do not describe the same training."
        )
    unlearned = sorted(recorded - exported)
    if unlearned and not allow_unlearned:
        raise Refused(
            f"the corpus contains labels the booster cannot predict: {unlearned}. "
            "Usually the per-class floor sent every row of those classes to the eval "
            "side. Re-run with --allow-unlearned-classes to publish anyway."
        )
    if unlearned:
        print(
            f"    warning: {len(unlearned)} corpus labels the booster cannot predict: {unlearned}"
        )
    return tuple(names)


def reconcile_features(assets: dict[str, Any], dims: int) -> None:
    """The booster's features must be `f0` … `f{dims-1}`, in that order.

    `embedding.py` hands the hot path a bare vector and `predictor.py` names the
    columns positionally. If the export reordered or renamed them, every served
    prediction reads the embedding through a permutation.
    """
    names = assets.get("feature_names")
    expected = [f"f{i}" for i in range(dims)]
    if names != expected:
        got = names if not isinstance(names, list) else f"{len(names)} names starting {names[:3]}"
        raise Refused(f"feature_names is not f0..f{dims - 1} in order: {got}")


def reconcile_shape(assets: dict[str, Any]) -> None:
    """The model type and label column the rest of the pipeline assumes."""
    if assets.get("model_type") != EXPECTED_MODEL_TYPE:
        raise Refused(f"model_type is {assets.get('model_type')!r}, not {EXPECTED_MODEL_TYPE!r}")
    if assets.get("label_col") != EXPECTED_LABEL_COL:
        raise Refused(f"label_col is {assets.get('label_col')!r}, not {EXPECTED_LABEL_COL!r}")


def probe_booster(booster_bytes: bytes, classes: tuple[str, ...], dims: int) -> None:
    """Load the converted booster with the serving XGBoost and predict once.

    Deliberately the installed XGBoost, not the one that did the conversion. That
    makes this the check that the conversion produced something *this* project
    can read, which is the entire reason the conversion exists.

    It also checks the width. `predictor.py` refuses a mismatch at serving time
    by passing through, which is safe and silent. Catching it here turns a
    permanent invisible degradation into a refusal to publish.
    """
    import numpy as np
    import xgboost

    booster = xgboost.Booster()
    booster.load_model(bytearray(booster_bytes))
    matrix = xgboost.DMatrix(
        np.zeros((1, dims), dtype=np.float32),
        feature_names=[f"f{i}" for i in range(dims)],
    )
    width = np.asarray(booster.predict(matrix)).reshape(1, -1).shape[1]
    if width != len(classes):
        raise Refused(
            f"the booster emits {width} probabilities but the export names "
            f"{len(classes)} classes; the label mapping cannot be trusted"
        )


@dataclass(frozen=True)
class HeldOut:
    """One held-out session, as the router would have handled it and as it was billed.

    `wanted` is what the session went on to call. `kept` is the menu the router would have
    declared, or None for a decline, which ships the whole list. `removed_tokens` is what
    the prune took out of every prompt in the session, and is zero on a decline and zero
    when no `--declaration-tokens` file said what a declaration costs.

    `turns` and `billed` come from the response log and are what criterion 3 divides by.
    They are None together when the log has no row for the session, and also when it has
    rows that show the session was never billed for the whole tool list, which makes its
    cost the wrong thing to compare a prune against.

    `top1` is the probability the booster put on its first choice, kept so publishing can
    record what confidence looked like on a model nobody has complained about yet. It
    gates nothing, here or later (DESIGN.md 20.4).
    """

    wanted: frozenset[str]
    kept: frozenset[str] | None
    removed_tokens: int = 0
    turns: int | None = None
    billed: float | None = None
    top1: float | None = None


@dataclass(frozen=True)
class Recall:
    """Gate criterion 1, and the same number over the sessions that were actually routed.

    Two numbers because they answer different questions and only one of them is the
    criterion. `blended` is what §19.5 defines and what the agent experiences: a declined
    session ships the whole tool list, so nothing can be missing from it. `routed` drops
    those free ones and asks what the menus are worth when they are used. A model can lift
    `blended` by declining more, which is why the pair is printed and why criterion 3
    exists to price the declining.
    """

    blended: float
    routed: float
    sessions: int
    routed_sessions: int


def menu_recall(decisions: list[HeldOut]) -> Recall:
    """Fold the held-out sessions into criterion 1.

    Sessions that wanted nothing have no recall to measure (DESIGN.md 19.4) and are left
    out of both the sum and the count. Wanted sets are already pinned to the label set
    here, because they come out of the training table, so a tool the booster cannot
    predict cannot count against a menu it was never allowed to hold.
    """
    total, measured = 0.0, 0
    routed_total, routed_measured = 0.0, 0
    for session in decisions:
        tools, menu = session.wanted, session.kept
        if not tools:
            continue
        recall = 1.0 if menu is None else len(menu & tools) / len(tools)
        total += recall
        measured += 1
        if menu is not None:
            routed_total += recall
            routed_measured += 1
    return Recall(
        blended=total / measured if measured else 0.0,
        routed=routed_total / routed_measured if routed_measured else 0.0,
        sessions=measured,
        routed_sessions=routed_measured,
    )


@dataclass(frozen=True)
class Saving:
    """Gate criterion 3: the steady-state saving this artifact is predicted to deliver.

    `predicted` is negative when the artifact saves, in the same unit as G1, so it can be
    read straight against the A/B's headline number. `routed` and `sessions` are printed
    beside it because the same predicted saving means very different things at a 40% and a
    90% routed share, and because a model can always trade one for the other.

    `removed_billed` and `billed` are kept so the number can be checked by hand: it is
    just the first divided by the second, negated.
    """

    predicted: float
    removed_billed: float
    billed: float
    sessions: int
    routed: int
    unpriced: int


def predicted_saving(decisions: list[HeldOut], rate: float = CACHED_TOKEN_WEIGHT) -> Saving | None:
    """Fold the held-out sessions into criterion 3 (DESIGN.md 19.5).

    `-rate x sum(tokens removed x turns) / sum(billed)`. A declined session keeps the whole
    list, removes nothing, and so contributes to the denominator and not the numerator.
    That is the whole reason the criterion exists: it is the one number that prices
    declining, and a model that buys its recall by passing 72% of sessions through cannot
    hide behind a good recall figure here.

    A session the response log has never heard of is dropped from both sums rather than
    counted as free, and the count of those is returned so the caller can say how much of
    the holdout the number rests on. None when that leaves nothing to divide by.

    It predicts the **steady state** and is optimistic by whatever turn one costs, because
    a removed declaration is priced at the cached rate on every turn including the first,
    where nothing is cached yet. Turn-one residency is its own open question (DESIGN.md
    18.24) and is deliberately not smuggled in here.
    """
    removed_billed, billed = 0.0, 0.0
    sessions, routed, unpriced = 0, 0, 0
    for session in decisions:
        if session.turns is None or session.billed is None:
            unpriced += 1
            continue
        sessions += 1
        routed += session.kept is not None
        billed += session.billed
        removed_billed += rate * session.removed_tokens * session.turns
    if not billed:
        return None
    return Saving(
        predicted=-removed_billed / billed,
        removed_billed=removed_billed,
        billed=billed,
        sessions=sessions,
        routed=routed,
        unpriced=unpriced,
    )


def load_declaration_tokens(path: str | None) -> dict[str, int] | None:
    """The agent's tool declaration sizes, so the gate can price a prune (DESIGN.md 19.13).

    A JSON object of tool name to billed tokens, as `measure_menus.py --dump-tokens` writes
    it. Optional, and the gate says which rule it measured either way, because most agents
    are not this repository's reference agent and publishing must not require one.
    """
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise Refused(f"{path} is not a non-empty JSON object of tool name to token count")
    tokens = {}
    for name, size in raw.items():
        if not isinstance(name, str) or not isinstance(size, int) or size < 0:
            raise Refused(f"{path}: {name!r} does not map a tool name to a token count")
        tokens[name] = size
    return tokens


def rank_held_out(
    client: bigquery.Client,
    args: argparse.Namespace,
    run: TrainingRun,
    classes: tuple[str, ...],
    booster_bytes: bytes,
    menus: tuple[frozenset[str], ...],
    tokens: dict[str, int] | None = None,
) -> list[HeldOut]:
    """Run the booster that is about to be published over the held-out sessions.

    Returns one `HeldOut` per session: what it wanted, what the router would have declared
    for it, what that prune removed, and what the session cost. `pruner.decide` makes the
    menu call, with no prior, because the prefix lock means turn one with no prior is the
    only decision the router ever gets to make (18.22). A None is a decline, and a declined
    session ships the full list.

    `tokens` completes the picture. Choosing a menu is only half of what the router does;
    the other half is `pruner.worth_pruning` deciding whether sending it beats the risk of
    a miss, and that half is arithmetic about declaration tokens. Without them this
    measures every session as pruned, which is strictly harsher than what ships, because a
    declined session gets the whole tool list and cannot miss anything (DESIGN.md 19.13).

    Deliberately the converted booster and the reconciled class order, not the BQML model
    still sitting in BigQuery. Those trees have already been checked by the training gate;
    what has not been checked is that *these bytes*, read by *this* XGBoost, with *this*
    label mapping, rank the way the training run believed. A reversed class list produces
    a perfectly valid artifact that is wrong on every prediction (DESIGN.md 18.8), and
    this is the only measurement in the pipeline that would notice.

    The live tool set stands in as the label set, because publishing has no agent to ask.
    That makes the measurement blind to tools the booster never learned, which is the same
    blindness `choose_menus` has and for the same reason: a menu cannot hold them either.
    A `tokens` file names the agent's real declarations, so when one is given it is the
    better answer to the same question and is used instead.

    The corpus is whatever the flattened table holds now, which for the usual case of
    publishing the run that just trained is that run's own holdout. Publishing an older
    run scores it against the current holdout instead, which is a fair question to ask of
    an artifact and not the question the training gate asked.
    """
    import numpy as np
    import xgboost

    job = client.query(
        HELD_OUT_SQL.format(
            project=args.project,
            dataset=args.dataset,
            agent=run.agent,
            rate=CACHED_TOKEN_WEIGHT,
            # Without a token file nothing is priced, so nothing is filtered and criterion
            # 3 reports itself unmeasurable rather than measuring the wrong sessions.
            min_prompt=sum(tokens.values()) if tokens else 0,
        )
    )
    rows = list(job.result())
    if not rows:
        raise Refused(f"router_flat_{run.agent} has no held-out rows to measure recall on")

    dims = run.embedding_dims
    booster = xgboost.Booster()
    booster.load_model(bytearray(booster_bytes))
    matrix = xgboost.DMatrix(
        np.asarray([[row[f"f{i}"] for i in range(dims)] for row in rows], dtype=np.float32),
        feature_names=[f"f{i}" for i in range(dims)],
    )
    probs = np.asarray(booster.predict(matrix)).reshape(len(rows), -1)

    cfg = load_config(gcs_bucket=args.bucket, environment=run.environment)
    known = frozenset(classes) - {ABSTAIN}
    live = frozenset(tokens) if tokens else known
    decisions = []
    for row, session_probs in zip(rows, probs, strict=True):
        # Sorted, because `decide` reads `preds[0]` as the model's top choice and the
        # booster emits probabilities in class order. `Predictor.predict` sorts for the
        # same reason; this is the one place that has to do it by hand.
        ranked = sorted(
            (Prediction(label, float(p)) for label, p in zip(classes, session_probs, strict=True)),
            key=lambda pred: (-pred.probability, pred.label),
        )
        kept = decide(ranked, live, known, None, cfg, menus).kept
        removed = 0
        if kept is not None and tokens is not None:
            removed = sum(tokens.get(tool, 0) for tool in live - kept)
            if not worth_pruning(removed, contained_mass(ranked, kept), cfg):
                kept, removed = None, 0
        decisions.append(
            HeldOut(
                wanted=frozenset(row["labels"]) - {ABSTAIN},
                kept=kept,
                removed_tokens=removed,
                turns=row["turns"],
                billed=row["billed"],
                top1=ranked[0].probability if ranked else None,
            )
        )
    return decisions


def drift_baseline(
    classes: tuple[str, ...],
    tokens: dict[str, int] | None,
    decisions: list[HeldOut],
) -> drift.Signals | None:
    """What this model looks like against its own surface on the day it ships.

    A healthy model has no class for a good share of the tools an agent declares, because
    the rare-class floor drops tools that never appear often enough to learn. §19.63 put
    that at 26.1% on a model that was scoring 0.9849 recall. So a later drift check has to
    compare against this number rather than against zero, or it reads a healthy model as a
    quarter drifted on its first day (DESIGN.md 20.4).

    None without a `--declaration-tokens` file. Without one there is no declared surface to
    be missing tools from, and a baseline computed against the model's own class list is
    zero by construction, which is a worse answer than admitting there is not one.
    """
    if not tokens:
        return None
    confidences = [held.top1 for held in decisions if held.top1 is not None]
    return drift.signals(tokens, frozenset(classes) - {ABSTAIN}, confidences)


# --- Publishing ---------------------------------------------------------------


def gate_metrics_with_recall(
    run: TrainingRun, recall: Recall | None, saving: Saving | None = None
) -> dict[str, float]:
    """The run record's numbers, plus criteria 1 and 3, which BigQuery never saw.

    Menu recall is measured here and only here, because menus are chosen here (19.11).
    Until now it was printed and thrown away, which is why the gate can only be an
    absolute floor: refusing an artifact whose recall has fallen away from the one it
    replaces needs the predecessor's number, and no published artifact carries it. This
    writes it so a later change can make that comparison. Nothing compares it yet.

    Absent, not zero, when there was nothing to measure — no menus, or `--skip-recall-gate`.
    A zero would read as a measured catastrophe and refuse the next publish for free.
    `model_store._as_metrics` takes any string-to-number map, so old artifacts without the
    key still load and a reader has to treat it as optional.
    """
    metrics = dict(run.gate_metrics)
    if recall is not None:
        metrics["menu_recall"] = recall.blended
        metrics["menu_recall_routed"] = recall.routed
    if saving is not None:
        metrics["predicted_g1"] = saving.predicted
        metrics["routed_share"] = saving.routed / saving.sessions if saving.sessions else 0.0
    return metrics


def build_metadata(
    run: TrainingRun,
    classes: tuple[str, ...],
    booster: bytes,
    menus: tuple[frozenset[str], ...] = (),
    recall: Recall | None = None,
    saving: Saving | None = None,
    baseline: drift.Signals | None = None,
) -> dict[str, Any]:
    """The twelve fields `model_store.parse_metadata` validates, plus the drift baseline.

    Menu membership is written sorted. A frozenset has no order, so serialising it as it
    comes out would give two publishes of the same menus two different JSON files and
    make the artifact impossible to diff.

    `drift_baseline` is the thirteenth key and the only optional one. It is written beside
    the artifact rather than only into the ledger because the question it answers — has
    this model's surface moved since it shipped — is about these bytes, and the answer has
    to survive the ledger being unreadable or the table not existing yet. `parse_metadata`
    reads by key and ignores the rest, so an older reader loads a newer artifact and a
    newer reader loads an older one as having no baseline, which is the truth.
    """
    metadata: dict[str, Any] = {
        "version": run.version,
        "sha256": hashlib.sha256(booster).hexdigest(),
        "size_bytes": len(booster),
        "trained_at": run.trained_at,
        "rows": run.rows,
        "classes": list(classes),
        "backend": BACKEND,
        "environment": run.environment,
        "embedding_model": run.embedding_model,
        "embedding_dims": run.embedding_dims,
        "gate_metrics": gate_metrics_with_recall(run, recall, saving),
        "menus": [sorted(menu) for menu in menus],
    }
    if baseline is not None:
        metadata["drift_baseline"] = baseline.as_dict()
    return metadata


def plan_menus(
    client: bigquery.Client,
    args: argparse.Namespace,
    run: TrainingRun,
    classes: tuple[str, ...],
) -> tuple[frozenset[str], ...]:
    """Choose the menus and say what was chosen, or explain why there are none.

    Publishing without menus is a supported outcome, not a failure. The router routes
    either way; without them it builds a bespoke tool set per session, which is what it
    did before menus existed. A quiet empty list would hide a corpus query that found
    nothing, so this prints either way.
    """
    if args.no_menus:
        print("    menus: skipped (--no-menus); the router will build a set per session")
        return ()

    sessions = fetch_sessions(client, args, run)
    since, until = menu_window(args, run)
    window = f"{(until - since).days}d from {since.date()}"
    menus = choose_menus(sessions, classes, args.menu_size, args.menu_count, args.menu_candidates)
    if not menus:
        print(
            f"    menus: none. {len(sessions)} sessions over {window} held no call "
            "to a tool the booster predicts"
        )
        return ()

    calls = sum(len(calls) for calls in sessions)
    print(
        f"    menus: {len(menus)} of {args.menu_size} tools, {args.menu_candidates} candidates, "
        f"from {len(sessions)} sessions and {calls} calls over {window}"
    )
    for menu in menus:
        print(f"      {', '.join(sorted(menu))}")
    return menus


def measure_held_out(
    client: bigquery.Client,
    args: argparse.Namespace,
    run: TrainingRun,
    classes: tuple[str, ...],
    menus: tuple[frozenset[str], ...],
    booster: bytes,
    tokens: dict[str, int] | None,
) -> list[HeldOut] | None:
    """Replay the holdout once, for both of the criteria measured here.

    One BigQuery read and one booster pass answer criterion 1 and criterion 3, so they
    share it. Without menus there is nothing to measure. The router then prunes per session,
    which is the pre-menu behaviour, and 18.16 measured what that costs the cache; what it
    scores is not these gates' business.

    None when nothing was measured. That is not the same as measuring zero and the two must
    not collapse: an unmeasured artifact is not one that scored badly.
    """
    if not menus:
        print("    holdout: not measured; no menus were chosen")
        return None
    if args.skip_recall_gate:
        print("    holdout: skipped (--skip-recall-gate)")
        return None
    return rank_held_out(client, args, run, classes, booster, menus, tokens)


def gate_menu_recall(
    decisions: list[HeldOut] | None, args: argparse.Namespace, tokens: dict[str, int] | None
) -> Recall | None:
    """Gate criterion 1, here rather than in the training procedure (DESIGN.md 19.11).

    The procedure gates on the model's raw top-N ranking because BigQuery has no menus.
    That number is harsher than what ships: a menu holds tools that were seen travelling
    together, so it catches a session's second tool when the ranker put it eighth. The
    criterion is defined on the menu, and the menu exists here, three lines after it was
    chosen and one line after the booster that will serve it was converted.

    With `--declaration-tokens` the decline rule runs too and the measurement is the whole
    shipped path. Without it, every session is measured as pruned, which understates the
    recall an agent sees and can refuse a model the product would be happy with. The line
    it prints says which of the two it did (DESIGN.md 19.13).

    Returns what it measured, so `build_metadata` can put the number in the artifact.
    """
    if decisions is None:
        return None

    found = menu_recall(decisions)
    rule = "with the decline rule" if tokens else "pruning every session, no decline rule"
    print(
        f"    recall: {found.blended:.4f} over {found.sessions} held-out sessions that "
        f"wanted a tool, {found.routed:.4f} over the {found.routed_sessions} it routed"
    )
    print(f"            measured {rule}")
    # E1 measured this number against an answer key for the first time and found that one minus
    # it lands within a point of the share of requests the routed arm then failed to serve
    # (DESIGN.md 19.66). So say it in those terms rather than leaving the reader to subtract:
    # the gate is a policy dial, and the policy is how many users you are willing to disappoint.
    print(f"            expect to miss about {(1 - found.blended) * 100:.1f}% of requests")
    if found.blended < args.min_menu_recall:
        raise Refused(
            f"menu recall {found.blended:.4f} is below {args.min_menu_recall:.4f}. The menus "
            "this artifact would ship do not hold enough of what sessions ask for, so "
            "publishing it would break agents to save tokens (DESIGN.md 19.5, criterion 1)."
        )
    return found


def gate_mass_coverage(
    menus: tuple[frozenset[str], ...], run: TrainingRun, args: argparse.Namespace
) -> None:
    """Criterion 1 for an artifact with no menus, which until now had no criterion 1 at all.

    `gate_menu_recall` above is defined on the menu, so when no menu is chosen it returns
    early and nothing checks that the model can still find the tools sessions ask for. That
    was the hole: a menu-less artifact passed on `coverage`, which is a strict top-N cut, and
    then served `top_k_by_mass`, which stops earlier. On the reference corpus the two were
    0.9889 and 0.9263, so the number the publish printed overstated the served one by six
    points and nobody could see it (DESIGN.md 19.74).

    `coverage_mass` closes that. `train_router.sql` now measures the same holdout under the
    exact rule the menu-less path serves, and this compares it to the same threshold
    `--min-menu-recall` sets, for the same reason: the threshold is a policy dial for how
    many requests you are willing to leave unserved.

    The two numbers are not the same statistic and the difference is worth knowing. Menu
    recall is per session, blended over routed and declined; this is per (session, tool)
    pair, so a session that wanted four tools counts four times and a partial hit scores
    partial credit. Held to one threshold they answer the same question at different
    resolutions, and the harsher of the two moves first.

    Silent when the artifact has menus, because then `gate_menu_recall` is the criterion and
    this cut never runs. Silent too when the run predates the column, which reads as NULL and
    is dropped from `gate_metrics`: an unmeasured run is not one that scored badly.
    """
    if menus:
        return
    if args.skip_recall_gate:
        print("    mass coverage: skipped (--skip-recall-gate)")
        return

    found = run.gate_metrics.get("coverage_mass")
    if found is None:
        print("    mass coverage: not measured; this run predates coverage_mass, nothing to gate")
        return

    print(
        f"    mass coverage: {found:.4f} at keep_probability_mass="
        f"{run.gate_metrics.get('keep_probability_mass', float('nan')):.2f}, against "
        f"{run.gate_metrics.get('coverage', float('nan')):.4f} at the strict top-N the "
        "run record calls coverage"
    )
    if found < args.min_menu_recall:
        raise Refused(
            f"mass coverage {found:.4f} is below {args.min_menu_recall:.4f}. This artifact "
            "ships no menus, so the router serves top_k_by_mass, and that cut does not hold "
            "enough of what sessions ask for. Publishing it would break agents to save "
            "tokens. Raise keep_probability_mass and retrain, or choose menus "
            "(DESIGN.md 19.74, criterion 1)."
        )


def gate_fleet_cache_watch(client: bigquery.Client, args: argparse.Namespace) -> Verdict | None:
    """Criterion 4: has pruning been costing this agent money in production?

    The three criteria above are all measured on a holdout replay, which is the artifact
    predicting its own future. This one is the past, measured on what the fleet actually
    billed. They can disagree, and when they do the fleet is right: a replay does not know
    about the implicit cache, and the cache is the mechanism by which a good prune turns into
    a bad bill (DESIGN.md 18.21).

    The live guard already asks this question, in memory, per process. It cannot answer it for
    a thinly routed agent, because no single replica sees the fifty warm calls in each arm
    that it needs, and it cannot carry an answer across a restart. `fleet_cache_watch.py`
    asks the same question of the warehouse, where every replica's calls are pooled and
    nothing is forgotten, and it uses the same `cache_watch.costs_more` to decide
    (DESIGN.md 19.75).

    Refusing here is a strong claim, so it is made only on a stratified comparison and only
    when a replica has enough of both arms to support one. An agent with no holdout, or a new
    agent with no history, is unmeasured, and unmeasured passes. That is the `coverage_mass`
    lesson from the gate above, applied in the other direction: an unmeasured run is not one
    that scored badly.
    """
    if args.skip_fleet_gate:
        print("    fleet cache watch: skipped (--skip-fleet-gate)")
        return None

    verdict = measure_fleet(client, project=args.project, dataset=args.dataset, agent=args.agent)
    if not verdict.measured:
        print(
            f"    fleet cache watch: not measured; no replica of {args.agent} has both arms "
            "warm in the last 30 days, nothing to gate"
        )
        return verdict

    assert verdict.standardised_pruned is not None  # narrowed by `measured`
    assert verdict.standardised_whole is not None
    ratio = verdict.standardised_pruned / verdict.standardised_whole - 1.0
    print(
        f"    fleet cache watch: {ratio:+.1%} billed per warm call against holdout, over "
        f"{len(verdict.judged)} of {len(verdict.strata)} replicas, 30 days"
    )
    # Printed beside it, never gated on. Where the two disagree, the difference is the
    # replicas' traffic mixes, and seeing it is how an operator knows which number to trust.
    pooled = verdict.pooled_pruned / verdict.pooled_whole - 1.0
    print(f"                       pooled, unstratified, would say {pooled:+.1%}")
    if verdict.harmful:
        raise Refused(
            f"pruning is costing {args.agent} {ratio:+.1%} per warm call against its holdout, "
            f"past the {verdict.margin:.0%} margin the live guard demotes at. Publishing a new "
            "artifact on top of that ships a second guess at a question production has "
            "already answered. Find out why the cache is breaking first (DESIGN.md 19.75)."
        )
    return verdict


def gate_predicted_saving(
    decisions: list[HeldOut] | None, args: argparse.Namespace, tokens: dict[str, int] | None
) -> Saving | None:
    """Gate criterion 3, which until now was a number a human read off `score_sets.py`.

    Criterion 1 asks whether the artifact is safe. This asks whether it is worth shipping,
    and the two pull in opposite directions: the cheapest way to a perfect recall is to
    decline every session, and that artifact saves nothing. Without this gate the pipeline
    could only refuse the dangerous direction (DESIGN.md 19.13).

    It needs declaration token counts, which the publish step has no agent to ask for, so
    without `--declaration-tokens` the criterion is unmeasurable and is reported as such
    rather than assumed. That is why it says so rather than passing quietly: a gate that
    silently does nothing is worse than one that is visibly absent.
    """
    if decisions is None:
        return None
    if not tokens:
        print("    saving: not measured; criterion 3 needs --declaration-tokens")
        return None

    found = predicted_saving(decisions)
    if found is None:
        print("    saving: not measured; no held-out session appears in the response log")
        return None

    print(
        f"    saving: {found.predicted:+.1%} predicted, over {found.sessions} held-out "
        f"sessions the router would route {found.routed / found.sessions:.1%} of"
    )
    if found.unpriced:
        print(
            f"            {found.unpriced} more were not billed for the whole tool list "
            "and were dropped"
        )
    if found.sessions < MIN_SAVING_SESSIONS:
        print(
            f"            fewer than {MIN_SAVING_SESSIONS} sessions, so this is reported "
            "and not gated on"
        )
        return found
    if found.predicted > args.min_predicted_saving:
        if args.skip_saving_gate:
            print("            below the bar, allowed by --skip-saving-gate")
            return found
        raise Refused(
            f"predicted saving {found.predicted:+.1%} does not reach "
            f"{args.min_predicted_saving:+.1%}. This artifact is safe and not worth "
            "shipping: it declines too much of its traffic, or the menus it does send are "
            "too close to the whole list to pay for themselves (DESIGN.md 19.5, "
            "criterion 3). Pass --skip-saving-gate if you are publishing it anyway."
        )
    return found


def incumbent_metrics(
    bucket: Any, agent: str, version: str
) -> tuple[str, Mapping[str, Any]] | None:  # noqa: ANN401
    """The artifact this publish would replace, and what it recorded.

    The same rule `ModelStore._latest_complete_version` uses, because the point is to compare
    against what serving is actually holding: the highest version directory with both files,
    excluding the one about to be written. Version strings sort chronologically.
    """
    root = f"routers/{agent}/"
    seen: dict[str, set[str]] = {}
    for blob in bucket.list_blobs(prefix=root):
        parts = blob.name[len(root) :].split("/")
        if len(parts) == 2:
            seen.setdefault(parts[0], set()).add(parts[1])

    complete = [
        v
        for v, files in seen.items()
        if v != version and {METADATA_NAME, SERVED_BOOSTER_NAME} <= files
    ]
    if not complete:
        return None
    previous = max(complete)
    return previous, json.loads(
        bucket.blob(f"{root}{previous}/{METADATA_NAME}").download_as_bytes()
    )


def gate_no_regression(
    bucket: Any,  # noqa: ANN401
    args: argparse.Namespace,
    run: TrainingRun,
    recall: Recall | None,
) -> None:
    """Refuse a publish that would make the agent measurably worse than it is today.

    Criterion 1 asks whether an artifact is good enough in absolute terms. It cannot see a
    model that is still above the bar and three points below the one it replaces, which is
    the shape a slow decay takes: every individual publish passes and the agent gets worse
    anyway. This compares against the incumbent instead.

    Both recall figures have to hold, not just the blended one. A model can lift `blended`
    by declining more often, because a declined session ships the full list and scores a
    free 1.0, so a guard reading the blend alone would wave through an artifact that got
    worse at the only job it has and hid it by doing that job less.

    Nothing to compare against is not a failure. The first artifact for an agent, an
    artifact published before this field existed, and a `--skip-recall-gate` run all leave
    the guard with one number, and one number is not a comparison.
    """
    if recall is None:
        return
    if args.allow_recall_regression:
        print("    non-regression: skipped (--allow-recall-regression)")
        return
    if args.version_suffix:
        # A sibling version is an experiment publishing a deliberately different pool
        # beside the live one (19.7). Holding it to the incumbent's recall would refuse
        # exactly the comparison it exists to make.
        print("    non-regression: skipped; a sibling version is not a replacement")
        return

    found = incumbent_metrics(bucket, run.agent, run.version)
    if found is None:
        print("    non-regression: nothing published yet for this agent")
        return
    previous, metadata = found
    before = metadata.get("gate_metrics") or {}
    if "menu_recall" not in before:
        print(f"    non-regression: {previous} predates menu_recall, nothing to compare")
        return

    floor = args.max_recall_regression
    pairs = (
        ("menu_recall", float(before["menu_recall"]), recall.blended),
        ("menu_recall_routed", float(before.get("menu_recall_routed", 0.0)), recall.routed),
    )
    print(f"    non-regression: against {previous}")
    for name, was, now in pairs:
        print(f"      {name:<19} {was:.4f} -> {now:.4f}  ({now - was:+.4f})")
    for name, was, now in pairs:
        if was - now > floor:
            raise Refused(
                f"{name} falls {was - now:.4f} from {was:.4f} to {now:.4f}, past the "
                f"{floor:.4f} allowed against {previous}. This artifact clears criterion 1 "
                "and is still worse than the one serving holds. Publish it with "
                "--allow-recall-regression if the surface changed and the drop is expected."
            )


def apply_suffix(run: TrainingRun, prefix: str, suffix: str | None) -> tuple[TrainingRun, str]:
    """Publish this run under a sibling version, so one run can ship two pools.

    A training run owns one export directory, and `metadata.json` lives in it, so publishing
    the same run twice normally means overwriting the artifact serving already holds. That
    is right for the everyday case and wrong for an experiment: V2-3 has to put one menu and
    eight menus side by side, from the same booster, and pin traffic to each in turn
    (DESIGN.md 19.7). The suffix gives the second pool its own version and its own directory.

    The version string and the directory name have to stay in step, because `ModelStore`
    finds an artifact by directory and then checks what the metadata says it is.
    """
    if not suffix:
        return run, prefix
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", suffix):
        raise Refused(f"version suffix {suffix!r} is not a safe path segment")
    if not prefix.endswith(run.version):
        raise Refused(f"export prefix {prefix!r} does not end in version {run.version!r}")
    print(f"    publishing as a sibling version, suffix {suffix!r}")
    return replace(run, version=f"{run.version}-{suffix}"), f"{prefix}-{suffix}"


def verify_served(run: TrainingRun, classes: tuple[str, ...], args: argparse.Namespace) -> None:
    """Load the artifact back through `ModelStore`, the way serving will.

    Writing a file the reader rejects is the one outcome this script must not
    have, and the reader is right here, so there is no reason to guess. This
    also exercises the sha256 and size checks against the real blob, so a
    digest computed over different bytes than serving downloads shows up now.
    """
    cfg = load_config(
        gcs_bucket=args.bucket,
        environment=run.environment,
        embedding_model=run.embedding_model,
        embedding_dims=run.embedding_dims,
        pin_version=run.version,
        local_cache_dir=args.local_cache_dir,
    )
    model = ModelStore(cfg).load(run.agent)
    if model is None:
        raise Refused("published, but ModelStore refused to load it; see the warning above")
    if model.classes != classes:
        raise Refused("ModelStore read back a different label order than was published")
    print(
        f"    ModelStore loaded version {model.metadata.version}, "
        f"{len(model.classes)} classes, {len(model.menus)} menus"
    )


# --- Entry point --------------------------------------------------------------


def published_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The menu shape this publish ran with, whatever chose it.

    Recorded as `settings_after` rather than `settings_before` because publishing does not
    change any setting. It acts on the ones the tuner left behind, and this is the record of
    which ones actually reached an artifact, which is the thing the settings file cannot say
    on its own: the next cycle overwrites it.
    """
    return {
        "menu_size": args.menu_size,
        "menu_count": args.menu_count,
        "menu_candidates": args.menu_candidates,
        "min_menu_recall": args.min_menu_recall,
        "no_menus": args.no_menus,
        "settings_file": args.settings,
    }


def ledger_metrics(metadata: dict[str, Any], baseline: drift.Signals | None) -> dict[str, Any]:
    """What the gates measured, flattened into one map for the ledger's `metrics` column.

    The baseline is written here as well as into `metadata.json`, so a chart can follow
    unseen share across versions without opening every artifact in the bucket.
    """
    metrics = dict(metadata["gate_metrics"])
    if baseline is not None:
        metrics["unseen_share_baseline"] = baseline.unseen_share
        metrics["confidence_p50"] = baseline.confidence_p50
        metrics["confidence_p10"] = baseline.confidence_p10
        metrics["sessions"] = baseline.sessions
    return metrics


def record_publish(
    args: argparse.Namespace,
    outcome: str,
    reason: str,
    run: TrainingRun | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Write this publish into the retrain ledger. Best effort, and never fatal.

    A dry run records nothing, for the same reason it writes nothing to the bucket. It is a
    rehearsal, and a ledger holding rehearsals cannot be read as a history of what shipped.

    `lifecycle.record` swallows its own failures, so there is nothing to catch here. The
    ledger is authoritative for nothing and must not be able to fail a publish that has
    already happened (DESIGN.md 20.3).
    """
    if args.dry_run:
        return
    lifecycle.record(
        args.project,
        args.dataset,
        lifecycle.Step(
            cycle_id=args.cycle_id,
            step="publish",
            agent=args.agent,
            outcome=outcome,
            environment=run.environment if run else args.environment,
            reason=reason,
            run_id=run.run_id if run else args.run_id,
            version=run.version if run else None,
            settings_after=published_settings(args),
            metrics=metrics,
            source="publish_router.py",
        ),
    )


def main() -> int:
    args = parse_args()

    try:
        apply_settings(args)
    except Refused as refusal:
        print(f"\nREFUSED: {refusal}", file=sys.stderr)
        record_publish(args, "failed", str(refusal))
        return 1

    bq = bigquery.Client(project=args.project)

    # None until the run is known, which is what the ledger row records when a publish
    # refuses before it has one.
    run: TrainingRun | None = None
    try:
        run = fetch_run(bq, args)
        print(f"==> run {run.run_id}")
        print(f"    agent {run.agent}, version {run.version}, environment {run.environment}")
        print(f"    trained {run.trained_at} on {run.rows} rows")
        print(f"    export {run.export_uri}")

        bucket_name, prefix = split_uri(run.export_uri)
        args.bucket = args.bucket or bucket_name
        bucket = storage.Client(project=args.project).bucket(bucket_name)

        # Reads always come from the export directory; writes may go to a sibling.
        run, out_prefix = apply_suffix(run, prefix, args.version_suffix)
        target = bucket.blob(f"{out_prefix}/{METADATA_NAME}")
        if target.exists() and not args.overwrite:
            raise Refused(
                f"{prefix}/{METADATA_NAME} already exists. This version is published; "
                "re-publishing it would change an artifact serving may already hold. "
                "Pass --overwrite only if you know why."
            )

        assets = read_assets(bucket, prefix)
        reconcile_shape(assets)
        reconcile_features(assets, run.embedding_dims)
        classes = reconcile_classes(assets, run, args.allow_unlearned_classes)
        print(f"    reconciled {len(classes)} classes, {run.embedding_dims} features")

        menus = plan_menus(bq, args, run, classes)

        booster = convert_booster(read_booster(bucket, prefix), classes, run.embedding_dims)
        probe_booster(booster, classes, run.embedding_dims)
        tokens = load_declaration_tokens(args.declaration_tokens)
        decisions = measure_held_out(bq, args, run, classes, menus, booster, tokens)
        recall = gate_menu_recall(decisions, args, tokens)
        gate_mass_coverage(menus, run, args)
        gate_fleet_cache_watch(bq, args)
        saving = gate_predicted_saving(decisions, args, tokens)
        gate_no_regression(bucket, args, run, recall)
        baseline = drift_baseline(classes, tokens, decisions)
        if baseline is not None:
            print(
                f"    drift baseline: {baseline.unseen_share:.1%} of {baseline.declared} "
                "declared tools have no class"
            )
        # The digest covers the converted bytes, because those are the bytes
        # serving downloads. Digesting the export instead would fail every load.
        metadata = build_metadata(run, classes, booster, menus, recall, saving, baseline)
        parse_metadata(metadata)  # the reader's own validator, before anything is written
        print(f"    booster {metadata['size_bytes']} bytes, sha256 {metadata['sha256'][:16]}…")

        if args.dry_run:
            print("\n" + json.dumps(metadata, indent=2))
            print("\n    dry run. Nothing was written.")
            return 0

        # Booster first, metadata last. Between these two lines the directory has
        # a loadable booster and no metadata, which reads as unpublished.
        bucket.blob(f"{out_prefix}/{SERVED_BOOSTER_NAME}").upload_from_string(
            booster, content_type="application/octet-stream"
        )
        print(f"    wrote gs://{bucket_name}/{out_prefix}/{SERVED_BOOSTER_NAME}")
        target.upload_from_string(
            json.dumps(metadata, indent=2) + "\n", content_type="application/json"
        )
        print(f"    wrote gs://{bucket_name}/{out_prefix}/{METADATA_NAME}")
        verify_served(run, classes, args)
        record_publish(
            args,
            "ok",
            f"published {run.version} from run {run.run_id}; every gate passed",
            run,
            ledger_metrics(metadata, baseline),
        )
        print("\n    published.")
        return 0

    except Refused as refusal:
        print(f"\nREFUSED: {refusal}", file=sys.stderr)
        record_publish(args, "rejected", str(refusal), run)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--environment", default="prod")
    parser.add_argument("--run-id", default=None, help="publish a specific run, not the newest")
    parser.add_argument("--bucket", default=None, help="defaults to the bucket in export_uri")
    parser.add_argument("--local-cache-dir", default="/tmp/adk_routers_publish_check")
    parser.add_argument(
        "--cycle-id",
        default=lifecycle.new_cycle_id(),
        help="the retrain cycle this publish belongs to, as passed down by retrain.sh. "
        "Run on its own, publishing mints one and becomes a single-step cycle, which is "
        "an honest record of what happened",
    )
    parser.add_argument(
        "--version-suffix",
        default=None,
        help="publish this run under `<version>-<suffix>` instead of `<version>`, so one "
        "booster can ship two menu pools side by side for an experiment",
    )
    # Menu shape and the recall bar. These four default to None so `--settings` can tell an
    # explicit flag from an unspecified one; `apply_settings` fills in the rest. See the
    # DEFAULT_* constants at the top for what the numbers mean and where they came from.
    parser.add_argument(
        "--settings",
        default=None,
        help="JSON written by `tune_settings.py --out`. Supplies --menu-size, --menu-count, "
        "--menu-candidates and --min-menu-recall wherever the flag was not given; an "
        "explicit flag always wins",
    )
    parser.add_argument("--menu-size", type=int, default=None)
    parser.add_argument("--menu-count", type=int, default=None)
    parser.add_argument(
        "--menu-candidates",
        choices=MENU_CANDIDATES,
        default=None,
        help="how the pool's candidate menus are built before the set cover picks between "
        f"them; defaults to {DEFAULT_MENU_CANDIDATES}. DESIGN.md 19.47.",
    )
    parser.add_argument("--menu-lookback-days", type=int, default=35)
    parser.add_argument(
        "--no-menus", action="store_true", help="publish without menus; prune per session"
    )
    parser.add_argument("--min-menu-recall", type=float, default=None)
    parser.add_argument(
        "--declaration-tokens",
        default=None,
        help="JSON of tool name to declaration tokens, from `measure_menus.py "
        "--dump-tokens`. Given one, criterion 1 measures the shipped path including the "
        "decline rule and criterion 3 can be measured at all; without one criterion 1 "
        "measures every session as pruned, which is harsher, and criterion 3 is skipped",
    )
    parser.add_argument(
        "--skip-recall-gate",
        action="store_true",
        help="publish without replaying the holdout at all; skips criteria 1 and 3 both, "
        "and saves one BigQuery read",
    )
    parser.add_argument(
        "--skip-fleet-gate",
        action="store_true",
        help="publish without asking the warehouse whether pruning has been costing this "
        "agent money in production; skips criterion 4 and saves one BigQuery read",
    )
    parser.add_argument(
        "--min-predicted-saving",
        type=float,
        default=DEFAULT_MIN_PREDICTED_SAVING,
        help="gate criterion 3: the steady-state saving this artifact must predict, as a "
        f"negative fraction of billed input tokens. Defaults to "
        f"{DEFAULT_MIN_PREDICTED_SAVING}. Needs --declaration-tokens to be measurable",
    )
    parser.add_argument(
        "--skip-saving-gate",
        action="store_true",
        help="publish an artifact that predicts too little saving; the number is still "
        "measured, printed and written into the metadata",
    )
    parser.add_argument(
        "--max-recall-regression",
        type=float,
        default=DEFAULT_MAX_RECALL_REGRESSION,
        help="how far menu recall may fall against the artifact being replaced",
    )
    parser.add_argument(
        "--allow-recall-regression",
        action="store_true",
        help="publish an artifact worse than the one it replaces; say why in the commit",
    )
    parser.add_argument(
        "--allow-unlearned-classes",
        action="store_true",
        help="publish when the corpus holds labels the booster never learned",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing metadata.json"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="print the metadata, write nothing")
    group.add_argument("--confirm", action="store_true", help="write metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
