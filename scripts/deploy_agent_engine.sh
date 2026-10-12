#!/usr/bin/env bash
#
# deploy_agent_engine.sh - put the reference agent, router and all, onto Agent Platform.
#
# Every other measurement in this project was taken on a workstation. That is the one thing
# an adopter cannot check for themselves from the README, because the interesting questions
# about a managed runtime are not about the router's logic at all. They are: does the image
# build with a private dependency in it, does the runtime's identity have the two roles the
# serving path needs, and does the artifact cache survive between requests on one instance.
# This script exists to answer those three by doing it, and scripts/verify_agent_engine.py
# is what reads the answers back. docs/deploying.md holds the results.
#
# What it deploys is examples/deployed_support: the same 100-tool support agent, the same
# published `support` artifact, wired the way DESIGN.md 6.3 says a deployment has to be.
#
# What it stages, and why staging is needed at all:
#
#   agents/deployed_support   the agent folder, from examples/deployed_support
#   /app/reference_agent      the agent's tools and instruction, minus the 7 MB training
#                             corpus and the scenario files, which the deployment does not
#                             read. ADK's ignore-file handling only applies to the agent
#                             folder, so trimming an extra package means copying it first.
#   /app/adk_smart_router     the router itself, as source. While the repository is private
#                             `pip install git+https://...` cannot resolve from Cloud Build,
#                             so the package rides along on PYTHONPATH instead. Once the
#                             repository is public this drops out and requirements.txt grows
#                             one line; the README in the agent folder says which.
#
# Configuration arrives as a .env written into the staged agent folder, never into the
# repository, because it names a project. ADK reads it at deploy time and turns it into the
# runtime's environment.
#
# Deploying creates a billable Agent Platform resource that keeps running until it is
# deleted, so --confirm is required and --delete is provided. --list shows what exists.
#
# Authentication is ADC only. No service-account key is created, read or referenced.
#
# Usage:
#   ./scripts/deploy_agent_engine.sh                  # print the plan, deploy nothing
#   ./scripts/deploy_agent_engine.sh --confirm        # create or update the deployment
#   ./scripts/deploy_agent_engine.sh --stage-only     # show what would ship, no cloud calls
#   ./scripts/deploy_agent_engine.sh --list
#   ./scripts/deploy_agent_engine.sh --delete --confirm

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
# shellcheck source=scripts/deployment.sh
source "$SCRIPT_DIR/deployment.sh"

AGENT_SRC="$REPO_ROOT/examples/deployed_support"
REFERENCE_SRC="$REPO_ROOT/examples/reference_agent"
ROUTER_SRC="$REPO_ROOT/src/adk_smart_router"
APP_NAME="deployed_support"

PROJECT_ID="${SMART_ROUTER_BQ_PROJECT:-}"
BQ_DATASET="${SMART_ROUTER_BQ_DATASET:-smart_router}"
BQ_LOCATION="${SMART_ROUTER_BQ_LOCATION:-US}"
GCS_BUCKET="${SMART_ROUTER_GCS_BUCKET:-}"
# Where the embedding model is served, which has nothing to do with where the agent runs.
# gemini-embedding-2 is not published to us-central1: the call 404s, the embedder fails open,
# the breaker latches, and the agent looks perfectly healthy while routing nothing. RouterConfig
# says all of this in a comment and defaults to `global`; this script used to override it with
# the deployment's own region and reproduced the failure the comment describes.
VERTEX_LOCATION="${SMART_ROUTER_VERTEX_LOCATION:-global}"
REGION="${SMART_ROUTER_AGENT_ENGINE_REGION:-us-central1}"
# 1.0 by default, which is not what a production rollout would do. This deployment exists to
# be measured, and a fraction below 1.0 means a smoke test can come back clean having routed
# nothing. Holdout stays at the shipped default so the deployed traffic still carries a
# control arm (DESIGN.md 12.2).
ROUTE_FRACTION="${SMART_ROUTER_ROUTE_FRACTION:-1.0}"
HOLDOUT_FRACTION="${SMART_ROUTER_HOLDOUT_FRACTION:-0.05}"
# A gate, not a label. The router refuses an artifact whose published environment is not this
# one, so it has to match what `smart-router publish` stamped rather than describe where the
# agent happens to run. Naming it after the runtime is what made the first deploy of this
# script route nothing: every session declined with
#   artifact environment 'prod' != configured 'agent-platform'
# which is the gate doing its job, on a value that had no business being different.
ENVIRONMENT="${SMART_ROUTER_ENVIRONMENT:-prod}"

GCS_BUCKET="${GCS_BUCKET#gs://}"
GCS_BUCKET="${GCS_BUCKET%/}"

CONFIRMED=false
ACTION=deploy
STAGE=""

# An EXIT trap rather than a RETURN one, because the failure paths below exit rather than
# return and a staging directory that outlives a failed deploy holds a .env naming a project.
cleanup() {
  [[ -n "$STAGE" ]] && rm -rf "$STAGE"
  return 0
}
trap cleanup EXIT

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/deploy_agent_engine.sh [--confirm] [--list | --delete]

  --confirm   Actually do it. Without it the plan is printed and nothing is created.
  --list      Show the Agent Platform deployments in the project and exit.
  --delete    Delete the deployment this script creates. Needs --confirm too.
  --stage-only  Build the image's source tree, print what is in it, and stop. No cloud calls.
  --help      This message.

Environment:
  SMART_ROUTER_BQ_PROJECT             required, no default
  SMART_ROUTER_GCS_BUCKET             required, no default. Bare bucket name, not a gs:// URL
  SMART_ROUTER_BQ_DATASET             default smart_router
  SMART_ROUTER_BQ_LOCATION            default US
  SMART_ROUTER_VERTEX_LOCATION        default global, where the embedding model is served
  SMART_ROUTER_AGENT_ENGINE_REGION    default us-central1, where the deployment lives
  SMART_ROUTER_ROUTE_FRACTION         default 1.0, higher than a real rollout on purpose
  SMART_ROUTER_HOLDOUT_FRACTION       default 0.05, the shipped control arm
  SMART_ROUTER_ENVIRONMENT            default prod. Must match the artifact, not the runtime
EOF
}

parse_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
    --confirm) CONFIRMED=true ;;
    --list) ACTION=list ;;
    --delete) ACTION=delete ;;
    --stage-only) ACTION=stage ;;
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
  command -v uv >/dev/null 2>&1 || die "uv is not on PATH"
}

# Both of these run out of a dependency group that `uv sync` leaves out, because
# google-cloud-aiplatform is a large install that only the deploying machine needs.
adk() { uv run --project "$REPO_ROOT" --group deploy adk "$@"; }
engines() {
  uv run --project "$REPO_ROOT" --group deploy python "$SCRIPT_DIR/agent_engines.py" \
    "$1" --project "$PROJECT_ID" --region "$REGION" --name "$APP_NAME"
}

list_deployments() {
  say "Agent Platform deployments in ${PROJECT_ID} / ${REGION}"
  engines list || die "could not list deployments. Is the Agent Platform API enabled?"
}

# The resource id is not something this script chooses; Agent Platform assigns it at create
# time. It is recovered by display name so that a second run updates rather than duplicates.
# Empty output means no match, which is why agent_engines.py prints nothing rather than a
# message when it finds none.
find_existing_id() { engines id; }

# A trimmed copy of the reference agent. `cp -r` of the whole thing would ship a 7 MB corpus
# of synthetic conversations and 64 KB of scenario files into a serving image that never
# opens either, and ADK applies no ignore patterns to extra packages.
stage_reference_agent() {
  local dst="$1"
  mkdir -p "$dst"
  cp "$REFERENCE_SRC"/*.py "$dst/"
  cp -r "$REFERENCE_SRC/tools" "$dst/tools"
  find "$dst" -name '__pycache__' -type d -prune -exec rm -rf {} +
  # A local experiment runner with a command line and an argparse block. Nothing in the
  # deployed app imports it, and leaving it out keeps the image honest about what runs.
  rm -f "$dst/run_agent.py"
}

write_env_file() {
  local path="$1"
  # Written into the staging copy under /tmp, never into the repository. ADK reads it with
  # dotenv at deploy time and hands the values to the runtime as environment variables.
  cat >"$path" <<EOF
SMART_ROUTER_BQ_PROJECT=${PROJECT_ID}
SMART_ROUTER_BQ_DATASET=${BQ_DATASET}
SMART_ROUTER_BQ_LOCATION=${BQ_LOCATION}
SMART_ROUTER_GCS_BUCKET=${GCS_BUCKET}
SMART_ROUTER_VERTEX_LOCATION=${VERTEX_LOCATION}
SMART_ROUTER_MODE=active
SMART_ROUTER_ROUTE_FRACTION=${ROUTE_FRACTION}
SMART_ROUTER_HOLDOUT_FRACTION=${HOLDOUT_FRACTION}
SMART_ROUTER_ENVIRONMENT=${ENVIRONMENT}
EOF
}

print_plan() {
  say "Plan"
  note "project          ${PROJECT_ID}"
  note "region           ${REGION}"
  note "app name         ${APP_NAME}   (the folder basename, which becomes the resource name)"
  note "agent folder     examples/deployed_support"
  note "extra packages   reference_agent (trimmed), adk_smart_router (source)"
  note "artifact bucket  gs://${GCS_BUCKET}"
  note "analytics        ${PROJECT_ID}.${BQ_DATASET} in ${BQ_LOCATION}"
  note "routing          mode=active route_fraction=${ROUTE_FRACTION} holdout=${HOLDOUT_FRACTION}"
  note "environment      ${ENVIRONMENT}   (must match what the artifact was published under)"
  note "embedding        ${VERTEX_LOCATION}   (where the model is served, not where this runs)"
  local existing
  existing="$(find_existing_id)"
  if [[ -n "$existing" ]]; then
    note "existing         ${existing}, which will be updated in place"
  else
    note "existing         none, so a new instance will be created"
  fi
}

delete_deployment() {
  local existing
  existing="$(find_existing_id)"
  [[ -n "$existing" ]] || {
    say "Nothing to delete: no deployment named ${APP_NAME} in ${PROJECT_ID} / ${REGION}"
    return 0
  }
  say "Deleting ${existing}"
  if [[ "$CONFIRMED" != true ]]; then
    note "dry run. Pass --confirm to delete it."
    return 0
  fi
  engines delete || die "delete failed"
}

stage_everything() {
  STAGE="$(mktemp -d)"
  say "Staging"
  cp -r "$AGENT_SRC" "$STAGE/$APP_NAME"
  find "$STAGE/$APP_NAME" -name '__pycache__' -type d -prune -exec rm -rf {} +
  stage_reference_agent "$STAGE/reference_agent"
  cp -r "$ROUTER_SRC" "$STAGE/adk_smart_router"
  find "$STAGE/adk_smart_router" -name '__pycache__' -type d -prune -exec rm -rf {} +
  write_env_file "$STAGE/$APP_NAME/.env"
  note "$(du -sh "$STAGE" | cut -f1) total"
  local part
  for part in "$APP_NAME" reference_agent adk_smart_router; do
    note "$(du -sh "$STAGE/$part" | cut -f1)  ${part}"
  done
}

deploy() {
  stage_everything
  local existing
  local args=()
  existing="$(find_existing_id)"
  args=(
    deploy agent_engine
    --project "$PROJECT_ID"
    --region "$REGION"
    --display_name "$APP_NAME"
    --description "adk-smart-router reference agent, deployed for end-to-end verification"
    --extra_packages "$STAGE/reference_agent"
    --extra_packages "$STAGE/adk_smart_router"
  )
  [[ -n "$existing" ]] && args+=(--agent_engine_id "$existing")
  args+=("$STAGE/$APP_NAME")

  say "Deploying"
  note "adk ${args[*]}"
  adk "${args[@]}" || die "deploy failed"

  say "Deployed"
  note "verify it with: uv run scripts/verify_agent_engine.py"
  note "delete it with: ./scripts/deploy_agent_engine.sh --delete --confirm"
}

main() {
  parse_args "$@"
  require_commands
  require_env SMART_ROUTER_BQ_PROJECT SMART_ROUTER_GCS_BUCKET

  [[ -d "$AGENT_SRC" ]] || die "no agent folder at ${AGENT_SRC}"

  case "$ACTION" in
  list)
    list_deployments
    exit 0
    ;;
  delete)
    delete_deployment
    exit 0
    ;;
  stage)
    # What would ship, without a control-plane call or a bill. The staging directory is
    # removed on exit like every other run, so this is a listing rather than a build output.
    stage_everything
    say "Contents"
    find "$STAGE" -type f | sed "s|^${STAGE}/|    |" | sort
    exit 0
    ;;
  esac

  print_plan
  if [[ "$CONFIRMED" != true ]]; then
    say "Dry run"
    note "Nothing was created. Pass --confirm to deploy."
    note "This creates a billable Agent Platform resource that runs until it is deleted."
    exit 0
  fi
  deploy
}

main "$@"
