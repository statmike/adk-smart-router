#!/usr/bin/env bash
#
# run_cycle.sh - check whether a retrain is due, and run one if it is. The thing a timer runs.
#
# `check_drift.py` writes `due` or `ok` and returns, and `retrain.sh` retrains whatever it is
# pointed at. What did not exist is the piece between them, which was a person reading the
# verdict and deciding (DESIGN.md 21.6). This is that piece and it is deliberately thin:
#
#   1. drift     check_drift.py --exit-code. Exit 10 means due, 0 means not, anything else
#                means the check itself broke and nothing further happens.
#   2. interval  How long since this agent last paid for a training run, read from the ledger.
#                Under the minimum, the cycle stops and says so. Seven days by default.
#   3. retrain   retrain.sh --confirm, under the same cycle id as the drift check, so the
#                verdict and what it caused are one row in v_router_cycles.
#
# Three things worth knowing before putting this on a timer.
#
# Retraining on a spurious trigger is safe and always was. Gate 1 refuses to publish a model
# that does not beat the incumbent, gates 2 through 4 refuse to train one that does not beat
# its own baseline, and the tuner declining means "keep last cycle's answer". The worst outcome
# is a training bill and a cycle that publishes nothing.
#
# The minimum interval exists because that bill is real, not because a cycle is risky. It is
# read from `router_lifecycle` rather than from a file on the runner, so a throwaway container
# and a long-lived VM reach the same answer, and two triggers in one day retrain once.
#
# This does not run the generation watch. `watch_generations.py` is the rollback and belongs on
# its own timer at its own cadence; see docs/operations.md for both jobs. Ordering them the
# other way round, a retrain trigger without a rollback beside it, is the one arrangement that
# can leave an agent worse off.
#
# Authentication is ADC only. No service-account key is created, read or referenced.
#
# Usage:
#   ./scripts/run_cycle.sh --confirm                        # check, and retrain if due
#   ./scripts/run_cycle.sh                                  # dry run: checks, retrains nothing
#   ./scripts/run_cycle.sh --confirm --min-interval-days 14
#   ./scripts/run_cycle.sh --confirm --force                # retrain whatever the check says

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"

# What check_drift.py exits with when a retrain is due. Kept in step by a test.
DUE=10

AGENT=support
ENVIRONMENT=prod
MIN_INTERVAL_DAYS=7
CONFIRM=false
FORCE=false
RETRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm) CONFIRM=true ;;
    --force) FORCE=true ;;
    --agent) AGENT="${2:?--agent needs a value}" && shift ;;
    --environment) ENVIRONMENT="${2:?--environment needs a value}" && shift ;;
    --min-interval-days) MIN_INTERVAL_DAYS="${2:?--min-interval-days needs a value}" && shift ;;
    --lookback) RETRAIN_ARGS+=(--lookback "${2:?--lookback needs a value}") && shift ;;
    --settings) RETRAIN_ARGS+=(--settings "${2:?--settings needs a value}") && shift ;;
    -h | --help)
      sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) printf 'ERROR: unknown argument %s\n' "$1" >&2 && exit 1 ;;
  esac
  shift
done

require_env SMART_ROUTER_BQ_PROJECT SMART_ROUTER_GCS_BUCKET

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

command -v bq >/dev/null 2>&1 || die "bq not found. Install the Google Cloud CLI."
command -v uv >/dev/null 2>&1 || die "uv not found. See GETTING_STARTED.md."

CYCLE_ID="$(cd "$REPO_ROOT" && PYTHONPATH=src uv run python -c \
  'from adk_smart_router import lifecycle; print(lifecycle.new_cycle_id())')"

# Unguarded by --confirm, unlike the one in retrain.sh. A dry run here still runs the drift
# check for real and still gets a `due` or `ok` row out of it, so leaving the interval decision
# unrecorded would file a `due` verdict with nothing after it, which is what a dead runner looks
# like. Nothing this writes claims a step ran that did not.
record() {
  (cd "$REPO_ROOT" && uv run scripts/record_step.py \
    --project "$PROJECT_ID" --dataset "$BQ_DATASET" --agent "$AGENT" \
    --environment "$ENVIRONMENT" --cycle-id "$CYCLE_ID" --source run_cycle.sh "$@") || true
}

say "cycle $AGENT in $ENVIRONMENT ($PROJECT_ID.$BQ_DATASET)"
note "cycle $CYCLE_ID"
if ! $CONFIRM; then
  note "dry run. Pass --confirm to retrain if the check says due."
fi

# --- 1. is a retrain due? ---------------------------------------------------------------

# The drift row is written by check_drift.py itself, under this cycle id, so the verdict and
# whatever it causes below share a row in v_router_cycles. A dry run still records it: reading
# the surface is the expensive part and it has been paid for either way, so throwing the answer
# away would make a rehearsal cost the same as a real run and leave nothing behind.
say "drift: does the published model still fit the surface it is serving?"
set +e
(cd "$REPO_ROOT" && uv run scripts/check_drift.py \
  --project "$PROJECT_ID" --dataset "$BQ_DATASET" --agent "$AGENT" \
  --cycle-id "$CYCLE_ID" --exit-code)
verdict=$?
set -e

if [[ $verdict -ne 0 && $verdict -ne $DUE ]]; then
  # Neither due nor ok means the check did not finish, and a timer that cannot tell the
  # difference retrains on a broken query. Stopping is the safe direction; --force is a person
  # saying they have read the error and want the cycle anyway.
  $FORCE || die "the drift check exited $verdict, which is neither due nor ok. Nothing was retrained."
  note "the check exited $verdict. --force, so continuing on it anyway."
elif [[ $verdict -eq 0 ]]; then
  $FORCE || {
    say "done"
    note "not due. Nothing was trained and nothing was published."
    exit 0
  }
  note "not due. --force, so retraining anyway."
fi

# --- 2. has it been long enough? ---------------------------------------------------------

# From the ledger, not from local state, because the runner is usually a container that will not
# exist tomorrow. `skipped` is excluded: it is what --skip-train and a held-off cycle write, and
# neither one paid for a training run. The 365-day floor is there to prune partitions.
last_train_days() {
  bq query --nouse_legacy_sql --format=csv --project_id="$PROJECT_ID" \
    --parameter="agent:STRING:$AGENT" --parameter="env:STRING:$ENVIRONMENT" \
    "SELECT IFNULL(CAST(TIMESTAMP_DIFF(
       CURRENT_TIMESTAMP(), MAX(recorded_at), HOUR) / 24 AS INT64), -1)
     FROM \`$PROJECT_ID.$BQ_DATASET.router_lifecycle\`
     WHERE agent = @agent AND environment = @env
       AND step = 'train' AND outcome != 'skipped'
       AND recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)" | tail -n +2
}

if ! since="$(last_train_days)"; then
  # Almost always a missing table on an agent that has never retrained. Refusing is the safe
  # direction: without the ledger there is no interval, and a timer with no interval retrains
  # on every tick. `record_step.py --ensure-table` creates it empty.
  die "cannot read $PROJECT_ID.$BQ_DATASET.router_lifecycle, so there is no interval to respect. Create it with: uv run scripts/record_step.py --ensure-table"
fi

say "interval: at least $MIN_INTERVAL_DAYS days between cycles"
if [[ "$since" -lt 0 ]]; then
  note "no training run on record, so this is the first cycle."
elif [[ "$since" -ge "$MIN_INTERVAL_DAYS" ]]; then
  note "the last one was $since days ago."
elif $FORCE; then
  note "the last one was $since days ago, which is under the minimum. --force, so going anyway."
else
  note "the last one was $since days ago. Holding off."
  # The verdict said due and no retrain followed, which without a row is indistinguishable from
  # a runner that died. `v_router_cycles` reads this shape back as `held off`.
  record --step train --outcome skipped \
    --reason "drift says due, but the last training run was $since days ago and the minimum interval is $MIN_INTERVAL_DAYS days"
  say "done"
  note "nothing was trained. Lower --min-interval-days, or pass --force."
  exit 0
fi

# --- 3. retrain ---------------------------------------------------------------------------

retrain=("$SCRIPT_DIR/retrain.sh" --agent "$AGENT" --environment "$ENVIRONMENT"
  --cycle-id "$CYCLE_ID" "${RETRAIN_ARGS[@]+"${RETRAIN_ARGS[@]}"}")

# A dry run stops here rather than running retrain.sh in its own dry-run mode. That mode still
# tunes for real and then asks the publisher to refuse, which is a fair rehearsal of a retrain
# and a poor one of this: the question here is whether the trigger fires, and it has answered.
if ! $CONFIRM; then
  say "done"
  note "would run: ${retrain[*]} --confirm"
  exit 0
fi

say "retrain: train, tune, publish"
"${retrain[@]}" --confirm ||
  die "the retrain cycle stopped. router_lifecycle records which step, and the live artifact is untouched."

say "done"
note "smart-router status --agent $AGENT shows what this cycle did."
