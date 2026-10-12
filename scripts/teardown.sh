#!/usr/bin/env bash
#
# teardown.sh - destroy exactly what scripts/provision.sh creates, and nothing else.
#
#   1. BigQuery dataset smart_router, with every table and view inside it, including
#      agent_events and the BQAA v_* views.
#   2. The roles/aiplatform.user binding provision.sh granted to the connection's
#      service agent, removed before the connection so the member still resolves.
#   3. BigQuery connection smart_router_conn.
#   4. GCS bucket gs://<project>-smart-router, all live objects and all noncurrent
#      versions. Versioning is on, so the noncurrent generations have to go explicitly
#      or the bucket delete fails on a bucket that looks empty.
#
# Requires --confirm. Without it the script prints the plan and exits non-zero.
# It refuses to run at all if the names do not match the project's naming patterns,
# so a typo in SMART_ROUTER_BQ_DATASET cannot delete someone's production dataset.
#
# Authentication is ADC only. No service-account key is created, read or referenced.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"
BQ_CONNECTION_ID="${SMART_ROUTER_BQ_CONNECTION_ID:-smart_router_conn}"
BQ_CONNECTION_LOCATION="${SMART_ROUTER_BQ_CONNECTION_LOCATION:-us}"
CONNECTION_ROLE="${SMART_ROUTER_CONNECTION_ROLE:-roles/aiplatform.user}"
GCS_BUCKET="${SMART_ROUTER_GCS_BUCKET:-}"

GCS_BUCKET="${GCS_BUCKET#gs://}"
GCS_BUCKET="${GCS_BUCKET%/}"

CONNECTION_FQN="${PROJECT_ID}.${BQ_CONNECTION_LOCATION}.${BQ_CONNECTION_ID}"
DATASET_FQN="${PROJECT_ID}:${BQ_DATASET}"

# Guard rails. These are deliberately not configurable: their whole job is to be the thing
# a mistyped environment variable cannot get past.
DATASET_PATTERN='^smart_router[a-z0-9_]*$'
CONNECTION_PATTERN='^smart_router[a-z0-9_]*$'
BUCKET_PATTERN='^[a-z0-9][a-z0-9._-]*-smart-router(-[a-z0-9-]+)?$'

CONFIRMED=false

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/teardown.sh [--confirm]

  --confirm   Actually delete. Without it the plan is printed and nothing is touched.
  --help      This message.

Environment (same as provision.sh):
  SMART_ROUTER_BQ_PROJECT              required, no default
  SMART_ROUTER_BQ_DATASET              default smart_router
  SMART_ROUTER_BQ_CONNECTION_ID        default smart_router_conn
  SMART_ROUTER_BQ_CONNECTION_LOCATION  default us
  SMART_ROUTER_CONNECTION_ROLE         default roles/aiplatform.user
  SMART_ROUTER_GCS_BUCKET              required, no default
EOF
}

parse_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
    --confirm) CONFIRMED=true ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: ${arg}"
      ;;
    esac
  done
}

require_commands() {
  local missing=()
  local cmd
  for cmd in gcloud bq jq; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
  done
  if ((${#missing[@]} > 0)); then
    die "missing required command(s): ${missing[*]}"
  fi
}

check_names() {
  [[ "$BQ_DATASET" =~ $DATASET_PATTERN ]] ||
    die "refusing to run: dataset '${BQ_DATASET}' does not match ${DATASET_PATTERN}. This script only ever deletes datasets this project provisions."
  [[ "$BQ_CONNECTION_ID" =~ $CONNECTION_PATTERN ]] ||
    die "refusing to run: connection '${BQ_CONNECTION_ID}' does not match ${CONNECTION_PATTERN}."
  [[ "$GCS_BUCKET" =~ $BUCKET_PATTERN ]] ||
    die "refusing to run: bucket '${GCS_BUCKET}' does not match ${BUCKET_PATTERN}."
}

# ---------------------------------------------------------------------------
# Discovery, so the plan lists what is really there rather than what we assume
# ---------------------------------------------------------------------------

DATASET_PRESENT=false
CONNECTION_PRESENT=false
BUCKET_PRESENT=false
SERVICE_AGENT=""
OBJECT_COUNT="0"
TABLE_COUNT="0"

discover() {
  if bq --project_id="$PROJECT_ID" show --dataset --format=none "$DATASET_FQN" >/dev/null 2>&1; then
    DATASET_PRESENT=true
    TABLE_COUNT="$(bq --project_id="$PROJECT_ID" ls --format=json --max_results=1000 "$DATASET_FQN" 2>/dev/null | jq -r 'length' 2>/dev/null || true)"
    # An empty dataset lists as nothing at all rather than as an empty array, so say
    # "unknown" instead of quietly claiming zero.
    [[ "$TABLE_COUNT" =~ ^[0-9]+$ ]] || TABLE_COUNT="an unknown number of"
  fi

  if bq --project_id="$PROJECT_ID" show --connection --format=none "$CONNECTION_FQN" >/dev/null 2>&1; then
    CONNECTION_PRESENT=true
    # Resolve the service agent now. After the connection is gone the IAM member name is
    # unrecoverable, and the binding would be left behind pointing at nothing.
    SERVICE_AGENT="$(bq --project_id="$PROJECT_ID" show --connection --format=prettyjson "$CONNECTION_FQN" 2>/dev/null |
      jq -r '.cloudResource.serviceAccountId // empty' || true)"
  fi

  if gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(name)' >/dev/null 2>&1; then
    BUCKET_PRESENT=true
    OBJECT_COUNT="$(gcloud storage ls --all-versions "gs://${GCS_BUCKET}/**" 2>/dev/null | grep -c . || true)"
    [[ "$OBJECT_COUNT" =~ ^[0-9]+$ ]] || OBJECT_COUNT="0"
  fi
}

print_plan() {
  say "Teardown plan for project ${PROJECT_ID}"
  printf '\nThese resources WILL BE DESTROYED:\n\n'
  if [[ "$DATASET_PRESENT" == true ]]; then
    printf '  [x] BigQuery dataset    %s  (%s tables/views, all dropped)\n' "$DATASET_FQN" "$TABLE_COUNT"
  else
    printf '  [ ] BigQuery dataset    %s  (not present)\n' "$DATASET_FQN"
  fi
  if [[ -n "$SERVICE_AGENT" ]]; then
    printf '  [x] IAM binding         %s on serviceAccount:%s\n' "$CONNECTION_ROLE" "$SERVICE_AGENT"
  else
    printf '  [ ] IAM binding         %s (no service agent resolved)\n' "$CONNECTION_ROLE"
  fi
  if [[ "$CONNECTION_PRESENT" == true ]]; then
    printf '  [x] BigQuery connection %s\n' "$CONNECTION_FQN"
  else
    printf '  [ ] BigQuery connection %s  (not present)\n' "$CONNECTION_FQN"
  fi
  if [[ "$BUCKET_PRESENT" == true ]]; then
    printf '  [x] GCS bucket          gs://%s  (%s objects including noncurrent versions)\n' "$GCS_BUCKET" "$OBJECT_COUNT"
    printf '      Every router artifact under routers/{agent}/{version}/ goes with it,\n'
    printf '      including every previous generation. There is no undo.\n'
  else
    printf '  [ ] GCS bucket          gs://%s  (not present)\n' "$GCS_BUCKET"
  fi
  printf '\nNothing else is touched: no other dataset, no API enablement, no other IAM binding,\n'
  printf 'no local file, and no credential.\n'
}

# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

delete_dataset() {
  say "BigQuery dataset ${DATASET_FQN}"
  if [[ "$DATASET_PRESENT" != true ]]; then
    note "not present, nothing to do"
    return
  fi
  note "deleting the dataset and every table and view inside it (${TABLE_COUNT} of them)"
  bq --project_id="$PROJECT_ID" rm --recursive --force --dataset "$DATASET_FQN"
  note "deleted"
}

remove_iam() {
  say "IAM ${CONNECTION_ROLE} on the connection service agent"
  if [[ -z "$SERVICE_AGENT" ]]; then
    note "no service agent resolved, nothing to do"
    return
  fi
  local member="serviceAccount:${SERVICE_AGENT}"
  local existing
  existing="$(gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten='bindings[].members' \
    --filter="bindings.role=${CONNECTION_ROLE} AND bindings.members=${member}" \
    --format='value(bindings.members)' 2>/dev/null || true)"
  if [[ -z "$existing" ]]; then
    note "binding not present, nothing to do"
    return
  fi
  note "removing ${CONNECTION_ROLE} from ${member}"
  gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
    --member="$member" \
    --role="$CONNECTION_ROLE" \
    --condition=None \
    --quiet >/dev/null
  note "removed"
}

delete_connection() {
  say "BigQuery connection ${CONNECTION_FQN}"
  if [[ "$CONNECTION_PRESENT" != true ]]; then
    note "not present, nothing to do"
    return
  fi
  note "deleting"
  bq --project_id="$PROJECT_ID" rm --connection --force "$CONNECTION_FQN"
  note "deleted"
}

delete_bucket() {
  say "GCS bucket gs://${GCS_BUCKET}"
  if [[ "$BUCKET_PRESENT" != true ]]; then
    note "not present, nothing to do"
    return
  fi
  if [[ "$OBJECT_COUNT" != "0" ]]; then
    # --all-versions is the load-bearing flag. A bare 'rm gs://bucket/**' deletes only the
    # live versions, the noncurrent generations stay, and the bucket delete then fails on a
    # bucket that lists as empty.
    note "deleting ${OBJECT_COUNT} objects including every noncurrent version"
    gcloud storage rm --all-versions "gs://${GCS_BUCKET}/**"
  else
    note "bucket is already empty"
  fi
  note "deleting the bucket"
  gcloud storage buckets delete "gs://${GCS_BUCKET}" --quiet
  note "deleted"
}

verify() {
  say "Verification"
  printf '  %-22s %-46s %s\n' "RESOURCE" "NAME" "STATE"
  printf '  %-22s %-46s %s\n' "----------------------" "----------------------------------------------" "------"
  local state
  local remaining=0

  if bq --project_id="$PROJECT_ID" show --dataset --format=none "$DATASET_FQN" >/dev/null 2>&1; then
    state="STILL PRESENT"
    remaining=$((remaining + 1))
  else
    state="GONE"
  fi
  printf '  %-22s %-46s %s\n' "BQ dataset" "$DATASET_FQN" "$state"

  if bq --project_id="$PROJECT_ID" show --connection --format=none "$CONNECTION_FQN" >/dev/null 2>&1; then
    state="STILL PRESENT"
    remaining=$((remaining + 1))
  else
    state="GONE"
  fi
  printf '  %-22s %-46s %s\n' "BQ connection" "$CONNECTION_FQN" "$state"

  if gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(name)' >/dev/null 2>&1; then
    state="STILL PRESENT"
    remaining=$((remaining + 1))
  else
    state="GONE"
  fi
  printf '  %-22s %-46s %s\n' "GCS bucket" "gs://${GCS_BUCKET}" "$state"

  # Exit non-zero on a partial teardown, so a caller cannot read success from a bucket that
  # is still billing.
  if ((remaining > 0)); then
    die "${remaining} resource(s) survived teardown. Re-run, or inspect them by hand."
  fi
}

main() {
  parse_args "$@"
  require_env SMART_ROUTER_BQ_PROJECT SMART_ROUTER_GCS_BUCKET
  require_commands
  check_names
  discover
  print_plan

  if [[ "$CONFIRMED" != true ]]; then
    printf '\nNothing was deleted. Re-run with --confirm to destroy the resources listed above.\n'
    exit 1
  fi

  say "--confirm given, proceeding"
  delete_dataset
  remove_iam
  delete_connection
  delete_bucket
  verify
  printf '\nTeardown complete.\n'
}

main "$@"
