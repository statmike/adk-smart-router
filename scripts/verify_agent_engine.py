#!/usr/bin/env python
"""Drive the Agent Platform deployment, and close the three facts nobody had checked.

`scripts/deploy_agent_engine.sh` puts the reference agent onto Agent Platform. This asks it
questions and reads the routing provenance back out of the answers, which turns three
paragraphs of "not verified" in docs/deploying.md into either a yes or a specific no.

The three:

1. **Does the runtime's identity reach both services the serving path needs?** The router
   reads one artifact from Cloud Storage and makes one embedding call per session, and it
   fails open on either, so a missing role looks like an agent that simply never routes. A
   `fresh` decision carrying a model version proves both worked. A `disabled` decision means
   no artifact was in hand; `timeout`, `error` and `breaker` mean the embedding did not
   answer. That is the difference between "grant roles/storage.objectViewer" and "grant
   roles/aiplatform.user", and guessing between them is a bad afternoon.

   `disabled` has one cause that is not a permission at all: `SMART_ROUTER_ENVIRONMENT` is
   matched against the environment the artifact was published under, so a deployment that
   names it after its runtime declines every session with an artifact sitting right there.
   That is the first thing this script found when it was run.

2. **Did the build resolve a router that is not on PyPI?** If it did not, the app does not
   import and nothing answers at all. Any provenance record answers this one.

3. **Does one process serve more than one session?** The artifact is cached in the process
   and on local disk, so the download cost is amortised across everything an instance
   handles, and an instance that dies between requests pays it every time. Every record
   carries `replica_id`, minted once per process, so counting distinct replica ids across
   N sessions measures exactly this.

Reading the answers out of the event stream rather than out of BigQuery is deliberate. The
provenance rides on `custom_metadata` of the event BQAA logs, so it is in the response
before it is in the warehouse, and a verification that does not have to wait on a streaming
insert is one that can be run right after a deploy. The warehouse gets the same rows a
minute later; `smart-router status` is the command for reading them there.

Cost is one model turn per session, plus one embedding call per routed session. The default
of six sessions is a few cents. Authentication is ADC only. No service-account key is
created, read or accepted.

Usage:
    uv run --group deploy python scripts/verify_agent_engine.py
    uv run --group deploy python scripts/verify_agent_engine.py --sessions 12
    uv run --group deploy python scripts/verify_agent_engine.py --region us-central1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import agent_engines
from deployment import project_arg

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = REPO_ROOT / "examples" / "reference_agent" / "scenarios" / "scenarios.json"

# The display name deploy_agent_engine.sh gives the deployment, which is the folder basename
# ADK derives it from. Changing one means changing both.
APP_NAME = "deployed_support"

# The key the router stamps its records under, and the one the deployed app puts in BQAA's
# allowlist. Imported rather than copied would be better, but this script runs in the deploy
# dependency group and should not need the package on its path to report a missing one.
PROVENANCE_KEY = "smart_router"

# Which decision sources mean the router did its job, and what each of the others points at.
# Straight out of the docstring on `RoutingRecord.decision_source`.
ROUTED = {"fresh", "reused"}
DIAGNOSIS = {
    "disabled": "no artifact in hand: check SMART_ROUTER_ENVIRONMENT, the bucket, and "
    "roles/storage.objectViewer. ModelStore logs which",
    "timeout": "the embedding call timed out: check roles/aiplatform.user and the region",
    "error": "the embedding call failed: check roles/aiplatform.user and the region",
    "breaker": "the embedding failed enough times to trip the breaker",
    "holdout": "the control arm, working as designed",
    "cheap": "the router declined: this menu was not worth the risk of pruning it",
    "lowconf": "the model was not confident enough to prune",
    "floor": "the prompt is below the context-cache floor, so pruning would not pay",
    "watch": "the cache watchdog switched this agent off for the rest of the process",
    "late": "the session had already sent a full tool list",
    "noop": "there was nothing worth pruning",
    "small": "too few tools to bother",
}


@dataclass
class Turn:
    """One session's worth of what came back."""

    scenario_id: str
    opening: str
    replied: bool
    record: dict[str, Any] | None


def _find_engine(api: Any, name: str) -> Any:
    """The deployment, or a sentence saying to deploy one. The lookup itself is shared."""
    engine = agent_engines.find(api, name)
    if engine is None:
        sys.exit(
            f"no Agent Platform deployment named {name!r} in this project and region.\n"
            "Deploy one first:\n\n"
            "    ./scripts/deploy_agent_engine.sh --confirm\n"
        )
    return engine


def _openings(count: int) -> list[tuple[str, str]]:
    """First messages from the reference agent's scenario file, taken in order.

    In order, not sampled, so that two runs of this script ask the same questions and a
    change in the answers is a change in the deployment rather than in the draw.
    """
    scenarios = json.loads(SCENARIOS.read_text())["scenarios"]
    return [(s["id"], s["opening"]) for s in scenarios[:count]]


def _record_from(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The routing record, dug out of whichever event carried the metadata.

    `carry_onto` moves the record onto the last event of the turn, but which event that is
    depends on whether the turn called a tool, so this looks at all of them rather than
    assuming. The record itself is a list, because a turn can hold a routing decision and
    one or more cache serves; the routing decision is the one with a `decision_source`.
    """
    for event in events:
        bag = (event.get("custom_metadata") or {}).get(PROVENANCE_KEY)
        for entry in bag if isinstance(bag, list) else [bag] if bag else []:
            if isinstance(entry, dict) and "decision_source" in entry:
                return entry
    return None


def _ask(engine: Any, user_id: str, opening: str) -> tuple[bool, dict[str, Any] | None]:
    """One session, one turn. Returns whether it replied and what it decided."""
    events: list[dict[str, Any]] = []
    for event in engine.stream_query(message=opening, user_id=user_id):
        events.append(event if isinstance(event, dict) else json.loads(event))
    replied = any(
        part.get("text") for e in events for part in (e.get("content") or {}).get("parts", [])
    )
    return replied, _record_from(events)


def _report(turns: list[Turn]) -> int:
    """Print the three verdicts. Returns the exit code."""
    records = [t.record for t in turns if t.record is not None]
    sources = Counter(str(r.get("decision_source")) for r in records)
    replicas = {str(r.get("replica_id")) for r in records}
    routed = [r for r in records if r.get("decision_source") in ROUTED]
    versions = {str(r.get("model_version")) for r in routed if r.get("model_version")}

    print()
    print(
        f"{len(turns)} sessions, {sum(t.replied for t in turns)} replied, "
        f"{len(records)} carried a routing record"
    )
    print()
    for source, n in sources.most_common():
        note = DIAGNOSIS.get(source, "")
        print(f"  {source:<10} {n:>3}   {note}")

    print()
    print("1. the runtime's identity")
    if routed:
        print(
            f"   ok. {len(routed)} sessions routed, model version "
            f"{', '.join(sorted(versions)) or 'unreported'}."
        )
        print("   The artifact was read from Cloud Storage and the embedding call succeeded,")
        print("   so the runtime holds both roles/storage.objectViewer and roles/aiplatform.user.")
    elif records:
        worst = sources.most_common(1)[0][0]
        print(f"   no. Nothing routed. The commonest outcome was {worst!r}:")
        print(f"   {DIAGNOSIS.get(worst, 'see the decision_source docstring in provenance.py')}")
    else:
        print("   unknown. No session carried a routing record at all, which means either the")
        print("   plugin is not wired in or the analytics allowlist is missing the key.")

    print()
    print("2. the build resolved the router")
    if records:
        print("   ok. Provenance records exist, so adk_smart_router imported in the container.")
    elif any(t.replied for t in turns):
        print("   partly. The agent replied but stamped nothing, so it came up without the")
        print("   plugin. Check the staged extra package and examples/deployed_support/agent.py.")
    else:
        print("   no. Nothing replied. Read the Cloud Build log for the deployment.")

    print()
    print("3. instance reuse")
    if len(replicas) == 0:
        print("   unknown. No records to count replicas from.")
    elif len(replicas) == len(records):
        print(f"   no reuse observed. {len(records)} records, {len(replicas)} distinct replicas,")
        print("   so every session landed on a fresh process and paid the artifact download.")
    else:
        print(f"   yes. {len(records)} records across {len(replicas)} distinct replicas, so a")
        print("   process serves several sessions and the artifact cache is amortised.")

    print()
    ok = bool(routed)
    print("verified" if ok else "not verified: see above")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument(
        "--region",
        default="us-central1",
        help="where the deployment lives (default: %(default)s)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=6,
        help="how many sessions to open, one turn each (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default=APP_NAME,
        help="display name of the deployment (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    api = agent_engines.client(args.project, args.region)
    found = _find_engine(api, args.name)
    print(f"{args.name} ({agent_engines.resource_id(found)}) in {args.project} / {args.region}")

    # Found through the control plane, driven through the data plane. The record the control
    # plane returns knows what the deployment is; only the data-plane object can ask it
    # anything. agent_engines.runnable() says why.
    engine = agent_engines.runnable(found.api_resource.name, args.project, args.region)

    turns: list[Turn] = []
    for i, (scenario_id, opening) in enumerate(_openings(args.sessions)):
        # A distinct user per session, so nothing shares a session by accident and the
        # replica count below is measuring what it claims to.
        replied, record = _ask(engine, f"verify-{i:03d}", opening)
        source = (record or {}).get("decision_source", "-")
        print(f"  {scenario_id}  {'replied' if replied else 'silent ':<8}  {source}")
        turns.append(Turn(scenario_id, opening, replied, record))

    return _report(turns)


if __name__ == "__main__":
    raise SystemExit(main())
