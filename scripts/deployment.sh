# Which cloud deployment the shell scripts talk to. Sourced, not run.
#
# The bash half of scripts/deployment.py, and it exists for the same reason: provision.sh,
# teardown.sh, deploy_sql.sh and retrain.sh create, alter and delete real resources, and none
# of them should have an opinion about whose. There is no fallback project and no fallback
# bucket. A default is a teardown that deletes somebody else's dataset the first morning
# somebody forgets an export.
#
#   export SMART_ROUTER_BQ_PROJECT=your-gcp-project
#   export SMART_ROUTER_GCS_BUCKET=your-bucket
#
# Callers source this, read the variables with `${VAR:-}` so that `--help` still answers, and
# call `require_env` once after their arguments are parsed.

require_env() {
  local name missing=false
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      printf 'ERROR: %s is not set, and this script has no default.\n' "$name" >&2
      missing=true
    fi
  done
  if [[ "$missing" == true ]]; then
    printf '\nPoint these scripts at your own deployment first:\n\n' >&2
    printf '  export SMART_ROUTER_BQ_PROJECT=your-gcp-project\n' >&2
    printf '  export SMART_ROUTER_GCS_BUCKET=your-bucket\n\n' >&2
    printf 'See README.md for the rest of the setup.\n' >&2
    exit 2
  fi
}
