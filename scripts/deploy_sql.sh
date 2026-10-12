#!/usr/bin/env bash
#
# deploy_sql.sh - substitute the deploy-time placeholders in a .sql file and create the
# stored procedure in BigQuery.
#
# The SQL under src/adk_smart_router/sql/ ships with two placeholders, because a scheduled
# query cannot template its own SQL and everything else is a runtime argument:
#
#   @project.@dataset               where the procedure and its tables live
#   @project.@location.@connection  the CLOUD_RESOURCE connection AI.EMBED runs through
#
# Substituting those by hand works exactly once. This script exists so that redeploying
# after a change to the pipeline is one repeatable command rather than a remembered sed.
#
# It handles both kinds of file the project ships: one holding a stored procedure, and
# dashboard.sql, which holds views. --check knows the difference and compares whichever it
# finds, view by view for the second kind.
#
# Authentication is ADC only. No service-account key is created, read or referenced.
#
# Usage:
#   ./scripts/deploy_sql.sh                       # deploy train_router.sql
#   ./scripts/deploy_sql.sh --dry-run             # validate only, create nothing
#   ./scripts/deploy_sql.sh --check               # is the deployed copy this file?
#   ./scripts/deploy_sql.sh path/to/other.sql
#   ./scripts/deploy_sql.sh --print > rendered.sql

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"
BQ_LOCATION="${SMART_ROUTER_BQ_LOCATION:-US}"
BQ_CONNECTION_ID="${SMART_ROUTER_BQ_CONNECTION_ID:-smart_router_conn}"
BQ_CONNECTION_LOCATION="${SMART_ROUTER_BQ_CONNECTION_LOCATION:-us}"

SQL_FILE="$REPO_ROOT/src/adk_smart_router/sql/train_router.sql"
MODE=deploy

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run ;;
    --print) MODE=print ;;
    --check) MODE=check ;;
    -h | --help)
      sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*) printf 'ERROR: unknown flag %s\n' "$1" >&2 && exit 1 ;;
    *) SQL_FILE="$1" ;;
  esac
  shift
done

require_env SMART_ROUTER_BQ_PROJECT

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "$SQL_FILE" ]] || die "no such file: $SQL_FILE"
command -v bq >/dev/null 2>&1 || die "bq not found. Install the Google Cloud CLI."

# Order matters. @project.@location.@connection contains @project, so the longer pattern
# has to be replaced first or it is left holding a half-substituted string.
render() {
  sed \
    -e "s|@project\.@location\.@connection|${PROJECT_ID}.${BQ_CONNECTION_LOCATION}.${BQ_CONNECTION_ID}|g" \
    -e "s|@project\.@dataset|${PROJECT_ID}.${BQ_DATASET}|g" \
    "$SQL_FILE"
}

# A placeholder that survives rendering means a new one was added to the SQL and not taught
# to this script. Deploying it would produce a procedure that fails at CALL time with a
# "dataset @project not found" error, which reads like a permissions problem.
leftover="$(render | grep -n '@project\|@dataset\|@location\|@connection' || true)"
if [[ -n "$leftover" ]]; then
  printf '\nERROR: unsubstituted placeholder in %s:\n%s\n' "$(basename "$SQL_FILE")" "$leftover" >&2
  exit 1
fi

if [[ "$MODE" == "print" ]]; then
  render
  exit 0
fi

# The body of a procedure is stored verbatim, so it can be compared byte for byte. The
# signature above it cannot: BigQuery re-prints the argument list in its own style. So the
# comparison starts at the first BEGIN, which is where the two agree or do not.
body() { sed -n '/^BEGIN/,$p'; }

# A view body is stored verbatim too, comments included, so the same comparison works on a file
# of views. Split the rendered file into one file per view, named for the view, holding exactly
# what BigQuery stores: everything after the `AS` line, with the statement's trailing semicolon
# removed. A semicolon inside a string literal would end a body early here; none of the shipped
# SQL has one, and if one arrives the diff fails loudly rather than passing quietly.
split_views() {
  render | awk -v dir="$1" '
    /^CREATE OR REPLACE VIEW `.*` AS$/ {
      name = $0
      sub(/^CREATE OR REPLACE VIEW `/, "", name)
      sub(/` AS$/, "", name)
      sub(/^.*\./, "", name)
      out = dir "/" name ".sql"
      printf "" > out
      print name
      collecting = 1
      next
    }
    collecting {
      line = $0
      if (line ~ /;[ \t]*$/) {
        sub(/;[ \t]*$/, "", line)
        print line >> out
        close(out)
        collecting = 0
        next
      }
      print line >> out
    }
  '
}

# Drift here is quiet and expensive. CREATE OR REPLACE PROCEDURE accepts a body without
# evaluating it, and nothing else in the pipeline reads it, so an edit that was never
# deployed leaves the warehouse a commit behind and every run afterwards reports numbers
# produced by code that is no longer in the repository. This ran for a week on this project.
# A stale view is the same fault seen from a dashboard instead of from a report.
if [[ "$MODE" == "check" ]] && render | grep -q '^CREATE OR REPLACE VIEW'; then
  command -v jq >/dev/null 2>&1 || die "jq not found. --check reads the deployed DDL out of bq's JSON output."

  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  mapfile -t view_names < <(split_views "$work")
  [[ ${#view_names[@]} -gt 0 ]] || die "no CREATE OR REPLACE VIEW blocks parsed out of $(basename "$SQL_FILE")"

  say "$(basename "$SQL_FILE") vs deployed ${PROJECT_ID}.${BQ_DATASET} (${#view_names[@]} views)"

  quoted="$(printf "'%s'," "${view_names[@]}")"
  bq query --nouse_legacy_sql --format=json --location="$BQ_LOCATION" \
    "SELECT table_name, view_definition
     FROM \`${PROJECT_ID}.${BQ_DATASET}\`.INFORMATION_SCHEMA.VIEWS
     WHERE table_name IN (${quoted%,})" >"$work/deployed.json"

  drifted=0
  for name in "${view_names[@]}"; do
    jq -r --arg n "$name" '.[] | select(.table_name == $n) | .view_definition' \
      <"$work/deployed.json" >"$work/$name.deployed"
    if [[ ! -s "$work/$name.deployed" ]]; then
      printf '    MISSING  %s is not deployed\n' "$name" >&2
      drifted=1
      continue
    fi
    if diff -u "$work/$name.deployed" "$work/$name.sql" >"$work/$name.diff" 2>&1; then
      note "ok       $name"
    else
      printf '    DRIFTED  %s\n' "$name" >&2
      head -30 "$work/$name.diff" >&2
      drifted=1
    fi
  done

  if [[ "$drifted" -eq 0 ]]; then
    note "in sync."
    exit 0
  fi
  printf '\n    Run ./scripts/deploy_sql.sh %s to make the warehouse match the repository.\n' \
    "$SQL_FILE" >&2
  exit 1
fi

if [[ "$MODE" == "check" ]]; then
  command -v jq >/dev/null 2>&1 || die "jq not found. --check reads the deployed DDL out of bq's JSON output."

  routine="$(render | sed -n 's/^CREATE OR REPLACE PROCEDURE `[^`]*\.\([A-Za-z0-9_]*\)`.*/\1/p' | head -1)"
  [[ -n "$routine" ]] || die "no CREATE OR REPLACE PROCEDURE line in $(basename "$SQL_FILE")"

  say "$(basename "$SQL_FILE") vs deployed ${PROJECT_ID}.${BQ_DATASET}.${routine}"

  deployed="$(bq query --nouse_legacy_sql --format=json --location="$BQ_LOCATION" \
    "SELECT ddl FROM \`${PROJECT_ID}.${BQ_DATASET}\`.INFORMATION_SCHEMA.ROUTINES
     WHERE routine_name = '${routine}'" | jq -r '.[0].ddl // empty')"

  if [[ -z "$deployed" ]]; then
    printf '\nERROR: %s is not deployed in %s.%s. Run this script with no flags.\n' \
      "$routine" "$PROJECT_ID" "$BQ_DATASET" >&2
    exit 1
  fi

  if diff -u <(printf '%s\n' "$deployed" | body) <(render | body) >/tmp/deploy_sql_drift.$$ 2>&1; then
    rm -f /tmp/deploy_sql_drift.$$
    note "in sync."
    exit 0
  fi

  printf '\nERROR: the deployed %s is not this file. Left is deployed, right is the repository.\n\n' \
    "$routine" >&2
  head -60 /tmp/deploy_sql_drift.$$ >&2
  lines="$(wc -l </tmp/deploy_sql_drift.$$)"
  [[ "$lines" -gt 60 ]] && printf '\n    ... %s more diff lines\n' "$((lines - 60))" >&2
  rm -f /tmp/deploy_sql_drift.$$
  printf '\n    Run ./scripts/deploy_sql.sh to make the warehouse match the repository.\n' >&2
  exit 1
fi

say "$(basename "$SQL_FILE") -> ${PROJECT_ID}.${BQ_DATASET} (${BQ_LOCATION})"
note "connection ${PROJECT_ID}.${BQ_CONNECTION_LOCATION}.${BQ_CONNECTION_ID}"

if [[ "$MODE" == "dry-run" ]]; then
  render | bq query --dry_run --nouse_legacy_sql --location="$BQ_LOCATION" >/dev/null
  note "valid. Nothing was created."
  exit 0
fi

render | bq query --nouse_legacy_sql --location="$BQ_LOCATION" >/dev/null
note "deployed."
