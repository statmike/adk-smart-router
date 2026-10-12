#!/usr/bin/env bash
#
# provision.sh - create the infrastructure in DESIGN.md 3.1, idempotently.
#
#   1. BigQuery dataset            smart_router            (location US)
#   2. BigQuery connection         smart_router_conn       (location us, CLOUD_RESOURCE)
#      plus roles/aiplatform.user for the connection's service agent. AI.EMBED refuses to
#      run against a multimodal endpoint without the connection, and the IAM grant on its
#      service agent is the step people forget, so it is a first-class step here.
#   3. GCS bucket                  gs://<project>-smart-router
#      Object versioning ON: that is what lets a bad router artifact roll back by
#      generation (11.6). Uniform bucket-level access and public access prevention ON.
#   4. Optional load of the committed corpus into <dataset>.agent_events (3.3).
#
# Every step checks before it creates, so re-running is safe and prints what already exists.
# Authentication is ADC only. No service-account key is created, downloaded or referenced.
#
# Undo with scripts/teardown.sh --confirm.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

# Every resource name is overridable, and the shapes are the ones documented in DESIGN.md
# 3.1. SMART_ROUTER_BQ_PROJECT / _BQ_DATASET / _GCS_BUCKET are the same variables the
# runtime package reads (config.py), so provisioning and serving cannot drift apart. The
# project and the bucket have no default at all; see deployment.sh.
PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"
BQ_LOCATION="${SMART_ROUTER_BQ_LOCATION:-US}"
BQ_CONNECTION_ID="${SMART_ROUTER_BQ_CONNECTION_ID:-smart_router_conn}"
BQ_CONNECTION_LOCATION="${SMART_ROUTER_BQ_CONNECTION_LOCATION:-us}"
CONNECTION_ROLE="${SMART_ROUTER_CONNECTION_ROLE:-roles/aiplatform.user}"
GCS_BUCKET="${SMART_ROUTER_GCS_BUCKET:-}"
GCS_LOCATION="${SMART_ROUTER_GCS_LOCATION:-US}"
GCS_STORAGE_CLASS="${SMART_ROUTER_GCS_STORAGE_CLASS:-STANDARD}"
EVENTS_TABLE="${SMART_ROUTER_EVENTS_TABLE:-agent_events}"
CORPUS_FILE="${SMART_ROUTER_CORPUS_FILE:-$REPO_ROOT/examples/reference_agent/corpus/corpus.jsonl.gz}"
CORPUS_RELOAD="${SMART_ROUTER_CORPUS_RELOAD:-false}"

# A bucket may be given as a bare name or as a gs:// URL. Normalise to the bare name.
GCS_BUCKET="${GCS_BUCKET#gs://}"
GCS_BUCKET="${GCS_BUCKET%/}"

CONNECTION_FQN="${PROJECT_ID}.${BQ_CONNECTION_LOCATION}.${BQ_CONNECTION_ID}"
DATASET_FQN="${PROJECT_ID}:${BQ_DATASET}"
EVENTS_FQN="${PROJECT_ID}:${BQ_DATASET}.${EVENTS_TABLE}"

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

require_commands() {
  local missing=()
  local cmd
  for cmd in gcloud bq jq; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
  done
  if ((${#missing[@]} > 0)); then
    die "missing required command(s): ${missing[*]}. gcloud and bq come from the Google Cloud CLI; jq reads the connection's service agent out of 'bq show' JSON."
  fi
}

require_credentials() {
  local account
  account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
  if [[ -z "$account" ]]; then
    die "no active gcloud credential. Run 'gcloud auth login' and 'gcloud auth application-default login'. This script uses ambient credentials only, never a key file."
  fi
  note "authenticated as ${account}"
}

# ---------------------------------------------------------------------------
# 1. BigQuery dataset
# ---------------------------------------------------------------------------

ensure_dataset() {
  say "BigQuery dataset ${DATASET_FQN} (location ${BQ_LOCATION})"
  if bq --project_id="$PROJECT_ID" show --dataset --format=none "$DATASET_FQN" >/dev/null 2>&1; then
    note "already exists, leaving it alone"
    return
  fi
  note "creating"
  bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" mk \
    --dataset \
    --description="adk-smart-router: BQAA agent_events, preflight and training (DESIGN.md 3.1)" \
    "$DATASET_FQN"
  note "created"
}

# ---------------------------------------------------------------------------
# 2. BigQuery connection + IAM on its service agent
# ---------------------------------------------------------------------------

connection_exists() {
  bq --project_id="$PROJECT_ID" show --connection --format=none "$CONNECTION_FQN" >/dev/null 2>&1
}

connection_service_agent() {
  # The service agent is minted with the connection. It can lag by a second or two on a
  # fresh create, so poll briefly rather than failing the whole run on a race.
  local attempt json agent
  for attempt in 1 2 3 4 5; do
    json="$(bq --project_id="$PROJECT_ID" show --connection --format=prettyjson "$CONNECTION_FQN" 2>/dev/null || true)"
    agent="$(printf '%s' "$json" | jq -r '.cloudResource.serviceAccountId // empty' 2>/dev/null || true)"
    if [[ -n "$agent" ]]; then
      printf '%s' "$agent"
      return 0
    fi
    sleep 2
  done
  return 1
}

ensure_connection() {
  say "BigQuery connection ${CONNECTION_FQN} (CLOUD_RESOURCE, location ${BQ_CONNECTION_LOCATION})"
  if connection_exists; then
    note "already exists, leaving it alone"
  else
    note "creating"
    bq --project_id="$PROJECT_ID" mk \
      --connection \
      --location="$BQ_CONNECTION_LOCATION" \
      --connection_type=CLOUD_RESOURCE \
      "$BQ_CONNECTION_ID"
    note "created"
  fi

  local agent
  if ! agent="$(connection_service_agent)"; then
    die "connection ${CONNECTION_FQN} exists but exposes no cloudResource.serviceAccountId. Without it AI.EMBED cannot reach Vertex. Inspect with: bq show --connection --format=prettyjson ${CONNECTION_FQN}"
  fi
  note "service agent ${agent}"
  ensure_connection_iam "$agent"
}

ensure_connection_iam() {
  local agent="$1"
  local member="serviceAccount:${agent}"
  say "IAM ${CONNECTION_ROLE} for the connection service agent on ${PROJECT_ID}"
  local existing
  existing="$(gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten='bindings[].members' \
    --filter="bindings.role=${CONNECTION_ROLE} AND bindings.members=${member}" \
    --format='value(bindings.members)' 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    note "binding already present"
    return
  fi
  note "granting ${CONNECTION_ROLE} to ${member}"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="$member" \
    --role="$CONNECTION_ROLE" \
    --condition=None \
    --quiet >/dev/null
  note "granted (IAM propagation can take up to a minute)"
}

# ---------------------------------------------------------------------------
# 3. GCS bucket
# ---------------------------------------------------------------------------

bucket_exists() {
  gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(name)' >/dev/null 2>&1
}

ensure_bucket() {
  say "GCS bucket gs://${GCS_BUCKET} (location ${GCS_LOCATION})"
  if bucket_exists; then
    note "already exists, leaving its contents alone"
  else
    note "creating"
    gcloud storage buckets create "gs://${GCS_BUCKET}" \
      --project="$PROJECT_ID" \
      --location="$GCS_LOCATION" \
      --default-storage-class="$GCS_STORAGE_CLASS" \
      --uniform-bucket-level-access \
      --public-access-prevention
    note "created"
  fi

  # Applied on every run, so a bucket that predates this script, or one someone loosened
  # by hand, converges to the settings 11.6 depends on. All three are idempotent.
  note "enforcing versioning, uniform bucket-level access, public access prevention"
  gcloud storage buckets update "gs://${GCS_BUCKET}" \
    --versioning \
    --uniform-bucket-level-access \
    --public-access-prevention >/dev/null
  note "enforced"
}

# ---------------------------------------------------------------------------
# 4. Corpus load (optional)
# ---------------------------------------------------------------------------

table_exists() {
  bq --project_id="$PROJECT_ID" show --format=none "$EVENTS_FQN" >/dev/null 2>&1
}

table_row_count() {
  # COUNT(*) rather than `bq show`'s numRows. BQAA streams its rows, and numRows
  # excludes whatever is still in the streaming buffer, so a table freshly written
  # by the corpus generator reports far fewer rows than it holds. That undercount
  # reads as a failed load. BigQuery answers this from metadata, so it scans 0 bytes.
  bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" query \
    --use_legacy_sql=false --format=csv --quiet \
    "SELECT COUNT(*) FROM \`${PROJECT_ID}.${BQ_DATASET}.${EVENTS_TABLE}\`" 2>/dev/null |
    tail -n 1 | tr -d ' ' || echo "0"
}

load_corpus() {
  say "Corpus load into ${EVENTS_FQN}"
  if [[ ! -f "$CORPUS_FILE" ]]; then
    note "no corpus at ${CORPUS_FILE}, skipping."
    note "The corpus is generated separately by scripts/generate_corpus.py and committed"
    note "under examples/reference_agent/corpus/. Everything else above is provisioned."
    return
  fi

  # Decide what kind of load this is BEFORE touching the file, so the common case of
  # "the table is already populated" does not pay to unpack 167 MB it will not use.
  local mode
  if ! table_exists; then
    mode=create
  else
    local existing
    existing="$(table_row_count)"
    if [[ "$existing" != "0" && "$CORPUS_RELOAD" != "true" ]]; then
      note "${EVENTS_TABLE} already holds ${existing} rows, skipping the load."
      note "Set SMART_ROUTER_CORPUS_RELOAD=true to replace the table contents."
      return
    fi
    if [[ "$CORPUS_RELOAD" == "true" ]]; then mode=replace; else mode=append; fi
  fi

  # The committed corpus is gzipped, because uncompressed it is 167 MB and GitHub
  # refuses anything over 100 MB. `bq load` does NOT decompress a local .gz. It fails
  # with "Failed to parse JSON: No active field found", which reads like a corrupt
  # corpus rather than a compressed one, so unpack to a temp file and load that.
  local load_file="$CORPUS_FILE"
  if [[ "$CORPUS_FILE" == *.gz ]]; then
    load_file="$(mktemp -t smart-router-corpus.XXXXXX.jsonl)"
    trap 'rm -f "$load_file"' RETURN
    note "unpacking $(basename "$CORPUS_FILE")"
    gunzip -c "$CORPUS_FILE" >"$load_file"
  fi
  note "found ${CORPUS_FILE} ($(wc -l <"$load_file" | tr -d ' ') lines)"

  case "$mode" in
    replace)
      note "SMART_ROUTER_CORPUS_RELOAD=true, replacing table contents"
      bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" load \
        --replace \
        --source_format=NEWLINE_DELIMITED_JSON \
        "$EVENTS_FQN" "$load_file"
      ;;
    append)
      note "table exists and is empty, appending with its existing schema"
      bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" load \
        --source_format=NEWLINE_DELIMITED_JSON \
        "$EVENTS_FQN" "$load_file"
      ;;
    create)
      # Match the shape BQAA itself writes (18.2): DAY partitioned on timestamp, clustered on
      # (event_type, agent, user_id). Every preflight and training query filters on timestamp
      # and agent, so getting this wrong here costs money on every later query.
      note "creating ${EVENTS_TABLE} from the corpus, partitioned and clustered like BQAA writes it"
      bq --project_id="$PROJECT_ID" --location="$BQ_LOCATION" load \
        --source_format=NEWLINE_DELIMITED_JSON \
        --autodetect \
        --time_partitioning_field=timestamp \
        --time_partitioning_type=DAY \
        --clustering_fields=event_type,agent,user_id \
        "$EVENTS_FQN" "$load_file"
      ;;
  esac
  note "loaded, table now holds $(table_row_count) rows"
}

# ---------------------------------------------------------------------------
# Verification summary
# ---------------------------------------------------------------------------

row() { printf '  %-22s %-46s %s\n' "$1" "$2" "$3"; }

# Whoever can write this bucket chooses which tools every session of every agent is shown
# (docs/threat-model.md, "Who can write the bucket"). This reports it rather than enforcing
# it, because the publisher and the serving identities are deployment-specific and this
# script cannot invent them. Bucket-level bindings are only half the answer: a project-level
# roles/editor or roles/storage.admin also writes here and does not appear in this policy,
# which is why the count below is a floor and the note says so.
report_writers() {
  local writers count
  writers="$(gcloud storage buckets get-iam-policy "gs://${GCS_BUCKET}" \
    --flatten='bindings[].members' \
    --filter='bindings.role:(roles/storage.admin OR roles/storage.objectAdmin OR roles/storage.objectCreator OR roles/storage.objectUser OR roles/storage.legacyBucketWriter)' \
    --format='value(bindings.members)' 2>/dev/null | sort -u || true)"
  count="$(printf '%s' "$writers" | grep -c . || true)"
  row "  bucket writers" "${count} bound at the bucket (project roles add more)" "CHECK"
  if [[ -n "$writers" ]]; then
    while IFS= read -r member; do
      [[ -n "$member" ]] && note "    ${member}"
    done <<<"$writers"
  fi
}

verify() {
  say "Verification"
  printf '  %-22s %-46s %s\n' "RESOURCE" "NAME" "STATE"
  printf '  %-22s %-46s %s\n' "----------------------" "----------------------------------------------" "------"

  local state

  if bq --project_id="$PROJECT_ID" show --dataset --format=none "$DATASET_FQN" >/dev/null 2>&1; then
    state="OK"
  else
    state="MISSING"
  fi
  row "BQ dataset" "${DATASET_FQN} (${BQ_LOCATION})" "$state"

  if connection_exists; then
    state="OK"
  else
    state="MISSING"
  fi
  row "BQ connection" "${CONNECTION_FQN}" "$state"

  local agent member granted
  agent=""
  if connection_exists; then
    agent="$(connection_service_agent 2>/dev/null || true)"
  fi
  if [[ -n "$agent" ]]; then
    member="serviceAccount:${agent}"
    granted="$(gcloud projects get-iam-policy "$PROJECT_ID" \
      --flatten='bindings[].members' \
      --filter="bindings.role=${CONNECTION_ROLE} AND bindings.members=${member}" \
      --format='value(bindings.members)' 2>/dev/null || true)"
    if [[ -n "$granted" ]]; then state="OK"; else state="MISSING"; fi
    row "IAM binding" "${CONNECTION_ROLE} -> ${agent}" "$state"
  else
    row "IAM binding" "${CONNECTION_ROLE} (no service agent resolved)" "MISSING"
  fi

  if bucket_exists; then
    local versioning uba pap
    versioning="$(gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(versioning_enabled)' 2>/dev/null || true)"
    uba="$(gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(uniform_bucket_level_access)' 2>/dev/null || true)"
    pap="$(gcloud storage buckets describe "gs://${GCS_BUCKET}" --format='value(public_access_prevention)' 2>/dev/null || true)"
    row "GCS bucket" "gs://${GCS_BUCKET}" "OK"
    row "  versioning" "${versioning:-unknown}" "$([[ "$versioning" == "True" ]] && echo OK || echo CHECK)"
    row "  uniform access" "${uba:-unknown}" "$([[ "$uba" == "True" ]] && echo OK || echo CHECK)"
    row "  public access" "${pap:-unknown}" "$([[ "$pap" == "enforced" ]] && echo OK || echo CHECK)"
    report_writers
  else
    row "GCS bucket" "gs://${GCS_BUCKET}" "MISSING"
  fi

  if table_exists; then
    row "Corpus table" "${EVENTS_FQN}" "$(table_row_count) rows"
  else
    row "Corpus table" "${EVENTS_FQN}" "ABSENT"
  fi

  say "Export these for the runtime package and the SQL"
  note "export SMART_ROUTER_BQ_PROJECT=${PROJECT_ID}"
  note "export SMART_ROUTER_BQ_DATASET=${BQ_DATASET}"
  note "export SMART_ROUTER_BQ_CONNECTION=${CONNECTION_FQN}"
  note "export SMART_ROUTER_GCS_BUCKET=${GCS_BUCKET}"
}

main() {
  require_env SMART_ROUTER_BQ_PROJECT SMART_ROUTER_GCS_BUCKET
  say "adk-smart-router provisioning in project ${PROJECT_ID}"
  require_commands
  require_credentials
  ensure_dataset
  ensure_connection
  ensure_bucket
  load_corpus
  verify
  printf '\nDone. Re-running this script is safe.\n'
}

main "$@"
