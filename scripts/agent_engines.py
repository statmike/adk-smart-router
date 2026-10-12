#!/usr/bin/env python
"""Reach an Agent Platform deployment by display name: find it, list, delete, or drive it.

One capability, in one place, because three callers need it and none of them should grow its
own copy: `deploy_agent_engine.sh` looks a deployment up so that a second deploy updates
rather than duplicates, the same script lists and deletes, and `verify_agent_engine.py` looks
the same deployment up to drive it.

Display name rather than resource id, because Agent Platform assigns the id at create time
and nobody writes it down. `adk deploy` defaults the display name to the agent folder's
basename, so the folder is the handle.

It is not `gcloud`. `gcloud alpha agent-engines` does not exist in every gcloud, and the
control plane is already reachable through the client ADK's own deploy path uses, which is in
the `deploy` dependency group along with everything else that only the deploying machine
needs.

Usable as a module or from a shell:

    uv run --group deploy python scripts/agent_engines.py list --project P --region us-central1
    uv run --group deploy python scripts/agent_engines.py id --name deployed_support ...
    uv run --group deploy python scripts/agent_engines.py delete --name deployed_support ...

`id` prints the resource id and nothing else, or nothing at all when there is no match, so a
shell can read it with a command substitution and test it for emptiness. Authentication is
ADC only. No service-account key is created, read or accepted.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any

from deployment import project_arg


def _vertexai() -> Any:
    """The library, imported late so that a missing install explains itself in one sentence."""
    try:
        import vertexai
    except ImportError:
        sys.exit(
            "google-cloud-aiplatform is not installed. It is only needed on the machine that\n"
            "deploys, so it lives in its own dependency group:\n\n"
            "    uv run --group deploy python scripts/agent_engines.py ...\n"
        )
    return vertexai


def client(project: str, region: str) -> Any:
    """The control-plane client: create, list, update, delete.

    `vertexai.Client`, deliberately, even though it prints a `FutureWarning` pointing at
    `agentplatform.Client`. Two reasons to stay on the old one for now. It is the client ADK
    2.8's own deploy path builds, so this script and the deploy cannot disagree about what
    they are looking at. And the new client renames the collection: `client.agent_engines`
    becomes `client.runtimes`, which is a rename this can follow the day the deploy path
    follows it, and not a day earlier. The warning is suppressed rather than printed at
    somebody who cannot act on it.
    """
    vertexai = _vertexai()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*vertexai.Client.*", category=FutureWarning)
        return vertexai.Client(project=project, location=region)


def runnable(resource_name: str, project: str, region: str) -> Any:
    """The same deployment, as something whose registered operations can be called.

    Two surfaces, because they do two jobs. The `AgentEngine` the control-plane client hands
    back is a pydantic record of the resource, so it has `api_resource` and `delete` and
    nothing else; asking it to `stream_query` raises an attribute error even though the
    deployment plainly registers one, which `operation_schemas()` will confirm. The data
    plane is `vertexai.agent_engines.get`, which binds every registered operation as a
    method. Managing a deployment needs the first, driving one needs the second, and the
    resource name is what carries you from one to the other.
    """
    vertexai = _vertexai()
    from vertexai import agent_engines as data_plane

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        vertexai.init(project=project, location=region)
        return data_plane.get(resource_name)


def find(api: Any, name: str) -> Any | None:
    """The one deployment with this display name, or None. Two is an error worth stopping on.

    Duplicates are the failure this exists to prevent, so it refuses to pick one rather than
    guessing which of them a deploy should update or a verifier should drive.
    """
    matches = [e for e in api.agent_engines.list() if e.api_resource.display_name == name]
    if not matches:
        return None
    if len(matches) > 1:
        ids = ", ".join(resource_id(e) for e in matches)
        sys.exit(f"{len(matches)} deployments are named {name!r}: {ids}. Delete the spares.")
    return matches[0]


def resource_id(engine: Any) -> str:
    """The trailing id out of `projects/.../locations/.../reasoningEngines/<id>`."""
    return str(engine.api_resource.name).rsplit("/", 1)[-1]


def _list(api: Any) -> int:
    engines = list(api.agent_engines.list())
    if not engines:
        print("no Agent Platform deployments in this project and region")
        return 0
    print(f"{'ID':<24} {'DISPLAY NAME':<28} UPDATED")
    for engine in engines:
        resource = engine.api_resource
        print(
            f"{resource_id(engine):<24} {resource.display_name or '-':<28} "
            f"{getattr(resource, 'update_time', '-')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["list", "id", "delete"])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--region", default="us-central1", help="default: %(default)s")
    parser.add_argument("--name", default="deployed_support", help="default: %(default)s")
    args = parser.parse_args(argv)

    api = client(args.project, args.region)
    if args.command == "list":
        return _list(api)

    engine = find(api, args.name)
    if args.command == "id":
        # Silence rather than a message, so `$(... id ...)` is empty when there is no match.
        if engine is not None:
            print(resource_id(engine))
        return 0

    if engine is None:
        print(f"nothing to delete: no deployment named {args.name!r}")
        return 0
    api.agent_engines.delete(name=engine.api_resource.name, force=True)
    print(f"deleted {resource_id(engine)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
