"""Local runner for the xLAM agent. A runner, not a library.

    uv run python -m examples.xlam_agent.run_agent --prompt "what is the zip code of 8.8.8.8?"
    uv run python -m examples.xlam_agent.run_agent --prompt "hi" --no-analytics

This is `reference_agent/run_agent.py` with one agent swapped and one app name changed.
Kept as its own file rather than parameterised, because the two agents are two fixtures
and a shared runner would make it possible to point one at the other's telemetry by
accident. The app name is the identity every row in BigQuery carries.

Authentication is ADC (`gcloud auth application-default login`). No key files, ever.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent import root_agent

APP_NAME = "xlam"

# No fallback deployment. `_required` below turns a missing one into the name of the
# variable to export, rather than a permission error against a project you do not own.
BQ_PROJECT = os.environ.get("SMART_ROUTER_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
BQ_DATASET = os.environ.get("SMART_ROUTER_BQ_DATASET", "smart_router")
BQ_LOCATION = os.environ.get("SMART_ROUTER_BQ_LOCATION", "US")

# The same string as `adk_smart_router.provenance.PROVENANCE_KEY`, hardcoded so this
# runner does not depend on the router package. Inert until something writes it.
PROVENANCE_KEY = "smart_router"


def _required(name: str, value: str) -> str:
    """The value, or a message naming the variable instead of somebody else's project id."""
    if not value:
        raise SystemExit(f"set {name}: this runner has no default deployment.")
    return value


def build_app(
    *,
    analytics: bool,
    route_fraction: float = 0.0,
    holdout_fraction: float = 0.05,
    cache_watch: bool = True,
) -> App:
    """Assemble the App the xLAM agent runs under.

    Router before analytics, for the reason DESIGN.md 6.3 and 11.5 give: the other order
    orphans a span and corrupts the `total_ms` on every tool row.
    """
    if not root_agent.tools:
        raise SystemExit(
            "examples/xlam_agent/tools.py is missing. "
            "Run `uv run python scripts/build_xlam_agent.py --fetch` first."
        )
    plugins: list[BasePlugin] = []

    if route_fraction > 0:
        from adk_smart_router import RouterConfig, SmartRouterPlugin

        plugins.append(
            SmartRouterPlugin(
                RouterConfig(
                    bq_project=_required("SMART_ROUTER_BQ_PROJECT", BQ_PROJECT),
                    bq_dataset=BQ_DATASET,
                    gcs_bucket=_required(
                        "SMART_ROUTER_GCS_BUCKET",
                        os.environ.get("SMART_ROUTER_GCS_BUCKET", ""),
                    ),
                    bq_connection=os.environ.get("SMART_ROUTER_BQ_CONNECTION")
                    or f"{BQ_PROJECT}.us.smart_router_conn",
                    pin_version=os.environ.get("SMART_ROUTER_PIN_VERSION") or None,
                    mode="active",
                    route_fraction=route_fraction,
                    holdout_fraction=holdout_fraction,
                    cache_watch_enabled=cache_watch,
                )
            )
        )

    if analytics:
        plugins.append(
            BigQueryAgentAnalyticsPlugin(
                project_id=_required("SMART_ROUTER_BQ_PROJECT", BQ_PROJECT),
                dataset_id=BQ_DATASET,
                location=BQ_LOCATION,
                custom_metadata_allowlist=[PROVENANCE_KEY],
            )
        )

    return App(
        name=APP_NAME,
        root_agent=root_agent,
        # Same cache regime as the reference agent, because the comparison between the two
        # fixtures is only meaningful if the caching floor is the same. 2048 is 2.5 Flash's.
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800),
        plugins=plugins,
    )


async def warmup_router(app: App) -> None:
    """Warm the router's embedding client, if a router is installed."""
    for plugin in app.plugins or ():
        warmup = getattr(plugin, "warmup", None)
        if callable(warmup):
            await warmup()


async def run_turn(runner: InMemoryRunner, user_id: str, session_id: str, text: str) -> None:
    """Send one user message and print the tools called and the final answer."""
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        for call in event.get_function_calls():
            print(f"  [tool] {call.name}")
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text for part in event.content.parts if part.text)
            if answer:
                print(answer.strip())


async def main_async(args: argparse.Namespace) -> None:
    """Own the Runner's lifecycle: the analytics plugin only flushes on close."""
    app = build_app(
        analytics=not args.no_analytics,
        route_fraction=args.route_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    runner = InMemoryRunner(app=app)
    session_id = uuid.uuid4().hex
    try:
        await warmup_router(app)
        await runner.session_service.create_session(
            app_name=APP_NAME, user_id=args.user_id, session_id=session_id
        )
        await run_turn(runner, args.user_id, session_id, args.prompt)
    finally:
        await runner.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--user-id", default="local")
    parser.add_argument("--route-fraction", type=float, default=0.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--no-analytics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
