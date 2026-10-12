#!/usr/bin/env bash
#
# check_install.sh - build the wheel, install it into an empty environment, and run the
# commands a new user runs first.
#
# Every other gate in this repo points at the source tree. That leaves packaging, entry
# points and data files with no coverage at all: the CLI shipped before anything confirmed
# an installed copy could reach it, and the wheel once failed to build for a week while
# lint, types and 877 tests stayed green.
#
# So this runs the actual first minute. A fresh virtualenv, the wheel and nothing else, an
# agent that does not import the router, and:
#
#   smart-router qualify          must answer, and must say pruning does not pay for a
#                                 two-tool agent
#   smart-router doctor           must fail all three checks on the unwired app and pass all
#                                 three on the wired one
#   smart-router readiness        must be registered, and must name the missing configuration
#                                 rather than raise
#   smart-router check-arguments  must be registered, and must name the missing configuration
#                                 rather than raise
#   the five SQL files            must be readable through importlib.resources from the
#                                 installed copy
#
# No cloud, no credentials, no model call, so it costs nothing and runs in CI on every push.
#
# Usage:
#   ./scripts/check_install.sh
#   ./scripts/check_install.sh --keep    # leave the environment behind for poking at

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
AGENT_DIR="$REPO_ROOT/examples/tiny_agent"

WORK="$(mktemp -d)"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1
cleanup() { [[ $KEEP -eq 1 ]] && echo "left behind: $WORK" || rm -rf "$WORK"; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "building the wheel and sdist"
uv build -o "$WORK/dist" --project "$REPO_ROOT" >/dev/null
WHEEL="$(echo "$WORK"/dist/*.whl)"
SDIST="$(echo "$WORK"/dist/*.tar.gz)"
echo "  $(basename "$WHEEL")"
echo "  $(basename "$SDIST")"

echo "installing it into an empty environment"
uv venv --python "$(cat "$REPO_ROOT/.python-version")" "$WORK/venv" >/dev/null 2>&1
uv pip install --quiet --python "$WORK/venv/bin/python" "$WHEEL"
BIN="$WORK/venv/bin"
export PYTHONPATH="$AGENT_DIR"

# The console script has to exist as a script, not just as an importable module. A user who
# follows the README types `smart-router`, not `python -m adk_smart_router.cli`.
[[ -x "$BIN/smart-router" ]] || fail "no smart-router console script in the installed environment"

echo "smart-router qualify, on an agent that does not import the router"
QUALIFY="$("$BIN/smart-router" qualify --app agent:app)" || fail "qualify exited nonzero"
grep -q "declared tools *2" <<<"$QUALIFY" || fail "qualify did not find the agent's two tools"
grep -q "context-cache floor" <<<"$QUALIFY" || fail "qualify printed no verdict"
grep -q "nothing to prune" <<<"$QUALIFY" || fail "qualify did not say a two-tool agent has nothing to prune"
echo "  answered, $(wc -l <<<"$QUALIFY") lines"

echo "smart-router doctor, on the unwired app"
DOCTOR="$("$BIN/smart-router" doctor --app agent:app)" && fail "doctor passed an unwired app"
grep -q "0 of 3 ok" <<<"$DOCTOR" || fail "doctor did not fail all three checks"
echo "  0 of 3, as it should be"

echo "smart-router doctor, on the wired app"
WIRED="$("$BIN/smart-router" doctor --app wired:app)" || fail "doctor failed a correctly wired app"
grep -q "3 of 3 ok" <<<"$WIRED" || fail "doctor did not pass all three checks"
echo "  3 of 3"

echo "smart-router readiness, which needs a cloud but no provisioning"
# Same shape as the check below: the subcommand has to exist in the wheel, and a missing
# project has to come back as a sentence. The project check runs before any client is built,
# so nothing here touches the network.
READY="$(env -u SMART_ROUTER_BQ_PROJECT "$BIN/smart-router" readiness --agent support 2>&1)" \
  && fail "readiness passed with no BigQuery project configured"
grep -q "no BigQuery project" <<<"$READY" || fail "readiness did not explain the missing project: $READY"
echo "  registered, and says what is missing"

echo "smart-router check-arguments, which also needs a cloud"
# It cannot be run for real here, so this checks the two things that can break without a
# project: that the subcommand is registered in the installed wheel at all, and that missing
# configuration comes back as a sentence rather than a traceback. It reaches the project check
# before it reaches credentials, so nothing here touches the network.
ARGS="$(env -u SMART_ROUTER_BQ_PROJECT "$BIN/smart-router" check-arguments --app agent:app 2>&1)" \
  && fail "check-arguments passed with no BigQuery project configured"
grep -q "no BigQuery project" <<<"$ARGS" || fail "check-arguments did not explain the missing project: $ARGS"
echo "  registered, and says what is missing"

echo "the SQL ships and is readable from the installed copy"
"$BIN/python" - <<'PY' || fail "the SQL is not reachable through importlib.resources"
from importlib.resources import files

import adk_smart_router

sql = files(adk_smart_router) / "sql"
names = sorted(p.name for p in sql.iterdir())
expected = [
    "backtest.sql",
    "dashboard.sql",
    "preflight.sql",
    "tool_ttl_report.sql",
    "train_router.sql",
]
assert names == expected, f"expected {expected}, found {names}"
for name in names:
    assert (sql / name).read_text().strip(), f"{name} is empty"
print(f"  {len(names)} files, {sum(len((sql / n).read_bytes()) for n in names):,} bytes")
PY

echo "the PyPI description rewrites all relative links and sdist stays lean"
"$BIN/python" - "$SDIST" <<'PY' || fail "packaging metadata or sdist check failed"
import importlib.metadata
import os
import re
import sys
import tarfile

import adk_smart_router

meta = importlib.metadata.metadata("adk-smart-router")
desc = meta["Description"]
assert desc and len(desc) > 1000, "PyPI Description metadata is missing or truncated"
rel_links = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", desc)
assert not rel_links, f"un-rewritten relative links in PyPI Description: {rel_links}"
assert "https://github.com/statmike/adk-smart-router/blob/main/docs/evidence.md" in desc

sdist_path = sys.argv[1]
sdist_bytes = os.path.getsize(sdist_path)
assert sdist_bytes < 1_500_000, f"sdist is too large ({sdist_bytes:,} bytes > 1.5 MB)"
with tarfile.open(sdist_path, "r:gz") as tf:
    members = tf.getnames()
for forbidden in (".github/", "corpus.jsonl.gz", "/uv.lock", "/plugin.json", "/skills/"):
    offenders = [m for m in members if forbidden in m]
    assert not offenders, f"sdist contains excluded path {forbidden}: {offenders[:3]}"
print(f"  0 relative links in PyPI metadata; sdist {sdist_bytes:,} bytes ({len(members)} files)")
PY

echo
echo "install ok"
