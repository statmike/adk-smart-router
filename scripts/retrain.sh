#!/usr/bin/env bash
#
# retrain.sh - one retrain cycle: train, tune, publish. The thing a scheduler runs.
#
# The three steps already exist and each one is safe on its own. What did not exist is the
# order they go in and what each one does when the step before it declines, which is the
# part a human was still supplying by hand every time (ROADMAP, V2-7c).
#
#   0. check    Is the deployed train_router the one in this repository? A stale procedure
#               runs happily and reports numbers from code nobody has read since.
#   1. train    CALL train_router(...). Fits a model, runs criteria 2 and 3, and exports a
#               booster to GCS only if both pass. A failed gate exports nothing, and this
#               script notices by watching the newest gate-passing run id.
#   2. tune     tune_settings.py sweeps the pool shape against held-out traffic and writes
#               its choice, but only if the same choice wins on both halves of that traffic.
#               A refusal is normal and leaves the previous settings in place.
#   3. publish  publish_router.py builds the artifact from the exported booster, applying
#               criterion 1 and whatever settings step 2 left behind.
#
# Four things worth knowing before scheduling this.
#
# The tuner reads the currently published model and the publisher ships the newly trained
# one, so a recommendation applies to the generation after the one it was measured on. That
# is a lag, not a leak: the pool shape is fit from traffic, and the model only decides which
# menu a session lands in. On a steady cadence it costs one cycle.
#
# The settings file has to survive between cycles, because a refusal means "keep last time's
# answer" and there is nowhere else that answer is written down. On a throwaway runner, point
# --settings at something durable.
#
# Nothing here restarts the agent, and nothing needs to. Serving processes re-read the bucket
# every `refresh_interval_s` and swap a new artifact in by themselves (DESIGN.md 21.4). Set
# `max_model_age_days` to a little more than this cadence anyway: it is the backstop for a
# process whose poll cannot reach the bucket at all.
#
# Every step of a confirmed cycle writes a row to router_lifecycle under one cycle id, so a
# published version can later explain what trained it, what the tuner recommended and what
# the gates said. `smart-router status` reads it back. A dry run writes nothing.
#
# Authentication is ADC only. No service-account key is created, read or referenced.
#
# Usage:
#   ./scripts/retrain.sh --confirm                 # the full cycle
#   ./scripts/retrain.sh                           # dry run: trains nothing, publishes nothing
#   ./scripts/retrain.sh --confirm --skip-train    # tune and publish an existing run
#   ./scripts/retrain.sh --confirm --agent support --lookback 35
#   ./scripts/retrain.sh --confirm --agent xlam --since '2026-09-06 20:00:00'
#   ./scripts/retrain.sh --confirm --cycle-id abc123def456   # join a cycle already under way

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"
BQ_LOCATION="${SMART_ROUTER_BQ_LOCATION:-US}"
GCS_BUCKET="${SMART_ROUTER_GCS_BUCKET:-}"

AGENT=support
ENVIRONMENT=prod
LOOKBACK_DAYS=35
SESSION_PREFIX=""
SINCE=""
SETTINGS=""
CYCLE_ID=""
CONFIRM=false
SKIP_TRAIN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm) CONFIRM=true ;;
    --skip-train) SKIP_TRAIN=true ;;
    --agent) AGENT="${2:?--agent needs a value}" && shift ;;
    --environment) ENVIRONMENT="${2:?--environment needs a value}" && shift ;;
    --lookback) LOOKBACK_DAYS="${2:?--lookback needs a value}" && shift ;;
    --since) SINCE="${2:?--since needs a value}" && shift ;;
    --session-prefix) SESSION_PREFIX="${2:?--session-prefix needs a value}" && shift ;;
    --settings) SETTINGS="${2:?--settings needs a value}" && shift ;;
    --cycle-id) CYCLE_ID="${2:?--cycle-id needs a value}" && shift ;;
    -h | --help)
      sed -n '3,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) printf 'ERROR: unknown argument %s\n' "$1" >&2 && exit 1 ;;
  esac
  shift
done

require_env SMART_ROUTER_BQ_PROJECT SMART_ROUTER_GCS_BUCKET

# Per agent, because the tuner refuses settings measured on a different one.
SETTINGS="${SETTINGS:-$REPO_ROOT/settings/$AGENT.json}"

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

command -v bq >/dev/null 2>&1 || die "bq not found. Install the Google Cloud CLI."
command -v uv >/dev/null 2>&1 || die "uv not found. See GETTING_STARTED.md."

# The newest run that passed its gate and left a booster behind. This is exactly the row
# publish_router.py would pick, so comparing it before and after training answers "did this
# cycle produce anything new" without second-guessing the gate.
latest_run() {
  bq query --nouse_legacy_sql --format=csv --project_id="$PROJECT_ID" \
    --parameter="agent:STRING:$AGENT" --parameter="env:STRING:$ENVIRONMENT" \
    "SELECT run_id FROM \`$PROJECT_ID.$BQ_DATASET.router_training_runs\`
     WHERE agent = @agent AND environment = @env AND gate_passed AND export_uri IS NOT NULL
     ORDER BY trained_at DESC LIMIT 1" | tail -n +2
}

# One id for the whole cycle, so the three steps read as one story afterwards. Minted here
# and passed down; a script run on its own mints its own and becomes a single-step cycle.
#
# `run_cycle.sh` passes one in, because the drift check that decided to run this is part of the
# same story and should be part of the same row in v_router_cycles.
if [[ -z "$CYCLE_ID" ]]; then
  CYCLE_ID="$(cd "$REPO_ROOT" && PYTHONPATH=src uv run python -c \
    'from adk_smart_router import lifecycle; print(lifecycle.new_cycle_id())')"
fi

# Every step of a confirmed cycle leaves a row. A dry run leaves none, for the same reason it
# publishes nothing: it is a rehearsal, and a ledger holding rehearsals cannot be read as a
# history of what shipped. The ledger is authoritative for nothing, so this never fails.
record() {
  $CONFIRM || return 0
  (cd "$REPO_ROOT" && uv run scripts/record_step.py \
    --project "$PROJECT_ID" --dataset "$BQ_DATASET" --agent "$AGENT" \
    --environment "$ENVIRONMENT" --cycle-id "$CYCLE_ID" --source retrain.sh "$@") || true
}

say "retrain $AGENT in $ENVIRONMENT ($PROJECT_ID.$BQ_DATASET)"
note "cycle $CYCLE_ID"
if ! $CONFIRM; then
  note "dry run. Pass --confirm to train and publish."
fi

before="$(latest_run)"
note "currently publishable: ${before:-nothing}"

# --- 0. is the warehouse running this repository's SQL? -------------------------------

# A procedure body is accepted without being evaluated, so an edit that was never deployed
# costs nothing until a run reports metrics computed by code no longer in the repository.
# That happened here, undetected, for a commit that changed the eval split. Checking takes a
# second and turns a silent wrong number into a refusal to start.
if $SKIP_TRAIN; then
  : # nothing below calls the procedure
elif ! "$SCRIPT_DIR/deploy_sql.sh" --check; then
  die "the deployed train_router is not this repository's. Deploy it, or pass --skip-train."
fi

# --- 1. train ------------------------------------------------------------------------

# train_router takes either a lookback in days or an explicit floor, and the floor is the
# one to reach for when some of the agent's logged traffic should not be trained on: a run
# that was abandoned halfway, or one sent through a fixture that has since changed. A
# lookback cannot express "everything after this afternoon", and the alternative is deleting
# telemetry, which this project does not do.
if [[ -n "$SINCE" ]]; then
  window="everything since $SINCE"
  since_arg="TIMESTAMP '$SINCE'"
else
  window="the last $LOOKBACK_DAYS days"
  since_arg="NULL"
fi

if $SKIP_TRAIN; then
  say "train: skipped"
  record --step train --outcome skipped --reason 'run with --skip-train'
elif ! $CONFIRM; then
  say "train: would CALL train_router over $window"
else
  say "train: CALL train_router over $window"
  # The row is written from here rather than from inside train_router. The procedure knows
  # the outcome, but teaching SQL to append to a second table for the sake of a chart is
  # more of the procedure's blast radius than an observability feature has earned. This
  # already compares the newest gate-passing run before and after, which is the same answer.
  if ! bq query --nouse_legacy_sql --location="$BQ_LOCATION" --project_id="$PROJECT_ID" \
    "CALL \`$PROJECT_ID.$BQ_DATASET.train_router\`(
       '$PROJECT_ID', '$BQ_DATASET', '$AGENT', '$GCS_BUCKET', '$ENVIRONMENT', $since_arg, $LOOKBACK_DAYS)" \
    >/dev/null; then
    record --step train --outcome failed --reason 'the CALL to train_router failed'
    die "training failed. Nothing was tuned or published."
  fi

  after="$(latest_run)"
  if [[ "$after" == "$before" ]]; then
    # The gate rejected the model, which is the gate working. Publishing anyway would
    # re-ship the booster that is already live under a new version number, so it does not.
    record --step train --outcome rejected \
      --reason 'the training gate rejected this model; router_training_runs says which criterion'
    note "the gate rejected this model. Nothing new to publish; stopping."
    note "router_training_runs records which criterion it missed."
    exit 0
  fi
  record --step train --outcome ok --run-id "$after" --reason 'the training gate passed'
  note "new run $after"
fi

# --- 2. tune -------------------------------------------------------------------------

say "tune: sweep the pool shape against held-out traffic"
mkdir -p "$(dirname -- "$SETTINGS")"

# Written to a scratch path and moved only on success. The tuner exits 0 when it declines to
# recommend anything, because declining is an answer, so a refusal must not look like a
# failure here either. It must also not blank the settings from the last cycle that did
# agree with itself: falling back to the shipped defaults would silently undo a real tuning.
scratch="$(mktemp)"
trap 'rm -f "$scratch"' EXIT

tune_args=(--project "$PROJECT_ID" --dataset "$BQ_DATASET" --agent "$AGENT"
  --bucket "$GCS_BUCKET" --out "$scratch" --cycle-id "$CYCLE_ID")
if [[ -n "$SESSION_PREFIX" ]]; then
  tune_args+=(--session-prefix "$SESSION_PREFIX")
fi

(cd "$REPO_ROOT" && uv run scripts/tune_settings.py "${tune_args[@]}") ||
  die "the tuner failed. Nothing was published."

if [[ -s "$scratch" ]]; then
  mv "$scratch" "$SETTINGS"
  note "wrote $SETTINGS"
elif [[ -f "$SETTINGS" ]]; then
  note "the tuner declined. Keeping the settings from the last cycle that agreed with itself."
else
  note "the tuner declined and there are no previous settings. Publishing on the defaults."
fi

# --- 3. publish ----------------------------------------------------------------------

publish_args=(--project "$PROJECT_ID" --dataset "$BQ_DATASET" --agent "$AGENT"
  --environment "$ENVIRONMENT" --bucket "$GCS_BUCKET" --cycle-id "$CYCLE_ID")
if [[ -f "$SETTINGS" ]]; then
  publish_args+=(--settings "$SETTINGS")
fi
if $CONFIRM; then
  publish_args+=(--confirm)
else
  publish_args+=(--dry-run)
fi

say "publish: build the artifact and apply the recall criterion"
if ! (cd "$REPO_ROOT" && uv run scripts/publish_router.py "${publish_args[@]}"); then
  # A full cycle only reaches here with a run id that did not exist a moment ago, so
  # "already published" is impossible and every refusal is a real one. Without a fresh
  # train it is the expected answer, and worth saying so rather than leaving a bare error.
  if $SKIP_TRAIN || ! $CONFIRM; then
    note "nothing new was trained, so the newest run is the one already live."
  fi
  die "publishing was refused. The live artifact is untouched."
fi

say "done"
if $CONFIRM; then
  note "serving picks this up within refresh_interval_s. No redeploy needed."
  note "smart-router status --agent $AGENT shows what this cycle did."
fi
