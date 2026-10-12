"""Local runner for the reference agent. A runner, not a library: nothing imports this.

Run it from the repo root so the `examples` namespace package resolves:

    uv run python -m examples.reference_agent.run_agent --prompt "where is order A-1042?"
    uv run python -m examples.reference_agent.run_agent --interactive
    uv run python -m examples.reference_agent.run_agent --prompt "hi" --no-analytics

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

APP_NAME = "support"

# No fallback deployment. `_required` below turns a missing one into the name of the
# variable to export, rather than a permission error against a project you do not own.
BQ_PROJECT = os.environ.get("SMART_ROUTER_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
BQ_DATASET = os.environ.get("SMART_ROUTER_BQ_DATASET", "smart_router")
BQ_LOCATION = os.environ.get("SMART_ROUTER_BQ_LOCATION", "US")

# Same string as `adk_smart_router.provenance.PROVENANCE_KEY`, hardcoded rather than
# imported: this runner must not depend on the router package. Allowlisting it now keeps
# the corpus schema identical before and after the router is installed. The key is inert
# until something writes it (DESIGN.md 11.3).
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
    """Assemble the App the reference agent runs under."""
    plugins: list[BasePlugin] = []

    # The router goes here, BEFORE the BQAA plugin. The order is load-bearing (DESIGN.md
    # 6.3, 11.5): router-first keeps BQAA's span stack paired; BQAA-first produces an
    # orphaned TOOL_COMPLETED whose `total_ms` is borrowed from the enclosing agent span,
    # which silently corrupts training data.
    #
    # It ships off. `--route-fraction 0` is the default, so a plain run behaves exactly as
    # it did while the corpus was being generated. That default is not politeness: routing
    # while generating training data would prune the menu Gemini chooses from, so the
    # labels would record the router's guesses and the next model would train on its own
    # output (DESIGN.md 3.3).
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
                    # Normally unset, and then serving takes the newest complete version.
                    # A sequential A/B pins each period to the artifact it is testing, which
                    # is the only way two menu pools can be compared on the same traffic.
                    pin_version=os.environ.get("SMART_ROUTER_PIN_VERSION") or None,
                    mode="active",
                    route_fraction=route_fraction,
                    holdout_fraction=holdout_fraction,
                    # On everywhere except a sequential A/B, where there is no holdout
                    # and so no unpruned arm for the watch to compare against.
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
        # Session-stable routing has no benefit without explicit caching on, so the
        # corpus must be generated under the same cache regime the router will run in
        # (DESIGN.md 6.3). 2048 tokens is the 2.5 Flash floor.
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800),
        plugins=plugins,
    )


async def warmup_router(app: App) -> None:
    """Warm the router's embedding client, if a router is installed.

    Duck-typed rather than imported, for the same reason `PROVENANCE_KEY` is hardcoded:
    this runner works whether or not the router package is installed.
    """
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
    """Own the Runner's lifecycle: BQAA only flushes its buffered rows on close."""
    app = build_app(
        analytics=not args.no_analytics,
        route_fraction=args.route_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    runner = InMemoryRunner(app=app)
    session_id = uuid.uuid4().hex
    try:
        # Before any traffic. A cold embedding client overruns the hot-path budget, and
        # one embedding failure pins a whole session to passthrough (DESIGN.md 18.10).
        await warmup_router(app)
        await runner.session_service.create_session(
            app_name=APP_NAME, user_id=args.user_id, session_id=session_id
        )
        if args.prompt:
            await run_turn(runner, args.user_id, session_id, args.prompt)
        else:
            await repl(runner, args.user_id, session_id)
    finally:
        await runner.close()


async def repl(runner: InMemoryRunner, user_id: str, session_id: str) -> None:
    """One session, many turns. Multi-turn is the point: turn 1 sets the cached prefix."""
    print(f"session {session_id} — empty line or Ctrl-D to quit")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            return
        if not text:
            return
        await run_turn(runner, user_id, session_id, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reference support agent locally.")
    turn = parser.add_mutually_exclusive_group(required=True)
    turn.add_argument("--prompt", help="run a single turn and exit")
    turn.add_argument("--interactive", action="store_true", help="multi-turn REPL")
    parser.add_argument(
        "--no-analytics",
        action="store_true",
        help="skip the BigQuery analytics plugin, so the turn writes no telemetry",
    )
    parser.add_argument(
        "--user-id",
        default=os.environ.get("REFERENCE_AGENT_USER_ID", "local-dev"),
        help="user id recorded on every event; vary it to get per-user corpus rows",
    )
    parser.add_argument(
        "--route-fraction",
        type=float,
        default=0.0,
        help="install the router and route this share of sessions. 0 leaves it uninstalled.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.05,
        help="share of sessions held out as the permanent control arm",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
