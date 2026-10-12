"""The reference agent wired the way a deployment wires it, for Agent Platform.

`examples/reference_agent/run_agent.py` builds an `App` too, but it builds it from command
line flags because it is a local experiment runner: routing off by default, analytics
optional, every knob a switch. A deployment has no command line. It gets environment
variables and it has to come up correctly or not at all, so the wiring here is the wiring
an adopter would copy, and it is deliberately short.

Three things make it a deployment rather than an example:

1. **It exports `app`.** ADK's agent loader looks for a module attribute named `app` that
   is an `App` instance before it looks for `root_agent`, so this is the shape that carries
   plugins through `adk web`, `adk deploy cloud_run` and `adk deploy agent_engine` alike.
   A folder that only exports `root_agent` loses its plugins on the way to production, and
   nothing warns you.
2. **It configures the router from the environment**, through the public `load_config()`.
   That is what lets an operator turn routing off, or pin a version, on a running
   deployment without a code change (DESIGN.md 11.1).
3. **Plugin order is fixed here and not left to a flag.** The router is registered before
   the analytics plugin, because the other order persists the event before the router has
   stamped its provenance onto it and every routing decision is thrown away silently
   (DESIGN.md 6.3, 11.5). `smart-router doctor` checks it, and this file is what it checks.

The agent itself is the reference agent, unchanged, and that is the point: its 100 tools
and its name are what the published `support` artifact was trained on, so a deployment of
this folder is the first end-to-end test of the router on somebody else's runtime rather
than on a workstation. `docs/deploying.md` covers what that deploy proved.
"""

from __future__ import annotations

import os

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from reference_agent.agent import root_agent

from adk_smart_router import SmartRouterPlugin, load_config
from adk_smart_router.provenance import PROVENANCE_KEY


def _required(name: str) -> str:
    """The variable's value, or the name of the variable that is missing.

    No defaults. A deployment that guesses a project id points at somebody else's
    warehouse, and the failure arrives as a permission error with no clue in it.
    """
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"set {name}: this deployment has no default project.")
    return value


BQ_PROJECT = _required("SMART_ROUTER_BQ_PROJECT")
BQ_DATASET = os.environ.get("SMART_ROUTER_BQ_DATASET", "smart_router")
BQ_LOCATION = os.environ.get("SMART_ROUTER_BQ_LOCATION", "US")

app = App(
    name="deployed_support",
    root_agent=root_agent,
    # Session-stable pruning saves almost nothing without explicit caching, because the
    # declarations it removes would otherwise bill at a tenth of the uncached rate. 2048
    # is the 2.5 Flash floor. `doctor` fails an App that has no cache config for exactly
    # this reason (DESIGN.md 6.3).
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800),
    plugins=[
        # Router first. See the module docstring.
        SmartRouterPlugin(load_config()),
        BigQueryAgentAnalyticsPlugin(
            project_id=BQ_PROJECT,
            dataset_id=BQ_DATASET,
            location=BQ_LOCATION,
            # Without this the analytics plugin drops the router's provenance key and the
            # warehouse has no record of which model shaped which turn, which is the same
            # outcome as the wrong plugin order and a different cause.
            custom_metadata_allowlist=[PROVENANCE_KEY],
        ),
    ],
)
