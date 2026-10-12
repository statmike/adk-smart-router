#!/usr/bin/env python
"""G3 — how much latency does the router add to a session?

`DESIGN.md` 2.4 sets the bar at 250 ms p95 per session and names "plugin timing" as
the source, and nothing had ever produced that timing. What existed was
`check_embedding_parity.py`'s number for the embedding call alone, 214 ms p95, quoted
as though it were the whole of G3. It is the largest term but it is not the only one,
and the rest had never been measured at all.

This measures the rest. It runs the real hot path over the real reference agent, 100
tools and a 1,777-token instruction, against the published artifact, with the network
call stubbed out. What is left is everything the plugin itself does:

  * the decide path, once per session — startup checks, rollout hashes, the classless
    inventory, the prefix lock, the booster, the pruning rule, the prompt measurement
    both guards need, and the edit to `config.tools`.
  * the reuse path, on every later turn — recall the session's decision and re-apply
    it. This is what session-stable routing bought and it should be nearly free.

A session's added latency is one decide plus one reuse per later turn, plus the one
embedding call. The reference run averaged 4.67 turns per session (`DESIGN.md` 19.17),
so that is the default here, and `--turns` changes it.

The embedding call is added from a flag rather than measured, because measuring it
means real Vertex traffic and `check_embedding_parity.py` already does that properly.
Its p95 is the default. Pass `--embed-p95 0` to see the in-process cost alone.

Cost: one GCS read for the artifact. No BigQuery, no model traffic, no embedding calls.
Authentication is ADC only. No service-account key is created, read or accepted.

Usage:
    uv run python scripts/bench_overhead.py
    uv run python scripts/bench_overhead.py --iterations 500 --turns 4.67
    uv run python scripts/bench_overhead.py --embed-p95 0
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# The package is not installed into the venv; pytest reaches it through
# `pythonpath` in pyproject.toml and scripts have to do the same by hand.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from deployment import require_bucket, require_project  # noqa: E402
from google.adk.agents.run_config import StreamingMode  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.tools.base_tool import BaseTool  # noqa: E402
from google.genai import types  # noqa: E402

from adk_smart_router.config import RouterConfig, load_config  # noqa: E402
from adk_smart_router.embedding import OK  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.plugin import SmartRouterPlugin  # noqa: E402

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# The gate, and the two measurements it has to be read against.
G3_BAR_MS = 250.0
# `check_embedding_parity.py`, 616 warm calls over 19 fresh processes, 2026-09-05
# (DESIGN.md 19.23). The network term of G3, not measured here.
EMBED_P95_MS = 214.0
# Turns per session on the reference A/B run, routed arm (DESIGN.md 19.17, 19.51).
TURNS_PER_SESSION = 4.67

# Enough iterations that the p95 is a p95 and not the second-worst sample, few enough
# that a run finishes while you watch it.
DEFAULT_ITERATIONS = 300
# Thrown away before timing starts. The first call through the decide path pays for
# every import xgboost defers, the startup diagnostics, and a cold branch predictor.
WARMUP = 5


# --- Stand-ins ----------------------------------------------------------------
#
# Small and local on purpose. The plugin reads a handful of attributes off ADK's
# context objects and building real ones needs a Runner, a session service and an
# event loop doing real work, all of which would be timed alongside the thing under
# test. Duck-typed the same way `tests/test_plugin.py` does it.


class _Session:
    def __init__(self, session_id: str) -> None:
        self.app_name = DEFAULT_AGENT
        self.id = session_id


class _RunConfig:
    streaming_mode = StreamingMode.NONE


class _PluginManager:
    plugins: list[Any] = []


class _InvocationContext:
    def __init__(self, agent: Any) -> None:  # noqa: ANN401 - duck-typed agent
        self.agent = agent
        self.plugin_manager = _PluginManager()
        self.context_cache_config = object()  # present, so the diagnostic stays quiet
        self.custom_metadata: dict[str, Any] = {}


class _Context:
    """Whatever `_route` reads off a `CallbackContext`, and nothing else."""

    def __init__(self, session_id: str, agent: Any) -> None:  # noqa: ANN401 - duck-typed agent
        self.agent_name = DEFAULT_AGENT
        self.user_id = "bench"
        self.session = _Session(session_id)
        self.run_config = _RunConfig()
        self.state: dict[str, Any] = {}
        self.invocation_id = session_id
        self._ictx = _InvocationContext(agent)
        self.custom_metadata = self._ictx.custom_metadata

    def get_invocation_context(self) -> _InvocationContext:
        return self._ictx


class _Agent:
    def __init__(self) -> None:
        self.name = DEFAULT_AGENT
        self.parent_agent = None
        self.sub_agents: list[Any] = []
        self.before_tool_callback = None


class _StubEmbedder:
    """Returns instantly. The network term is added back from `--embed-p95`.

    Timing the real call here would measure Vertex rather than the plugin, and would
    do it worse than `check_embedding_parity.py` does, which runs fresh processes to
    keep the cold start out of the steady-state number.
    """

    def __init__(self, dims: int) -> None:
        # Unit-norm, like anything `Embedder` returns. The booster does not care what
        # the vector says, only how long it is.
        self._vector = [1.0 / (dims**0.5)] * dims

    async def embed(self, text: str) -> list[float]:
        return list(self._vector)

    async def embed_with_reason(self, text: str) -> tuple[list[float], str]:
        return list(self._vector), OK

    async def warmup(self) -> bool:
        return True

    @property
    def stats(self) -> dict[str, int]:
        return {}


# --- The request under test ---------------------------------------------------


def reference_request() -> LlmRequest:
    """The real reference agent's request: 100 declarations and its real instruction.

    Built through the agent rather than by hand, because the thing being timed walks
    every declaration and every character of the instruction. A fixture of empty
    declarations would measure an agent nobody runs.
    """
    from examples.reference_agent.agent import MODEL, root_agent
    from examples.reference_agent.instruction import INSTRUCTION

    with warnings.catch_warnings():
        # ADK warns on every `_get_declaration` that its JSON-schema path is experimental.
        # A hundred of those before the first number is noise, and the CLI silences the
        # same warning for the same reason.
        warnings.simplefilter("ignore", UserWarning)
        tools = asyncio.run(root_agent.canonical_tools(None))
        # `_get_declaration` is ADK's only synchronous accessor for a tool's schema, and
        # is what `BaseTool.process_llm_request` calls to build the request this imitates.
        pairs = [(tool, tool._get_declaration()) for tool in tools]  # noqa: SLF001
    declarations = [decl for _, decl in pairs if decl is not None]
    tools_dict: dict[str, BaseTool] = {}
    for orig_tool, decl in pairs:
        if decl is None or not decl.name:
            continue
        bt = BaseTool(name=decl.name, description=decl.description or "")
        if hasattr(orig_tool, "func"):
            bt.func = orig_tool.func  # type: ignore[attr-defined]
        tools_dict[decl.name] = bt
    return LlmRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="my order arrived damaged, how do I send it back")],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            tools=[types.Tool(function_declarations=declarations)],
        ),
        tools_dict=tools_dict,
    )


def build_plugin(cfg: RouterConfig, dims: int) -> SmartRouterPlugin:
    """The real plugin, with the network call stubbed and the rollout pinned on."""
    routed = replace(
        cfg,
        mode="active",
        route_fraction=1.0,
        holdout_fraction=0.0,
        max_model_age_days=max(cfg.max_model_age_days, 3650),
    )
    return SmartRouterPlugin(
        routed,
        model_store=ModelStore(routed),
        embedder=_StubEmbedder(dims),  # type: ignore[arg-type]
    )


def declared(request: LlmRequest) -> int:
    """How many function declarations this request would send."""
    return sum(len(tool.function_declarations or []) for tool in request.config.tools or [])


async def probe(plugin: SmartRouterPlugin, base: LlmRequest) -> tuple[int, int]:
    """Run the path once and report what it did, before any of it is timed.

    Without this the script has a silent failure that looks like a good result. If the
    artifact will not load, or the agent name does not match the model's, `_route`
    returns early and the decide path measures a few hash lookups. That prints a
    reassuringly small number and means nothing at all.
    """
    request = base.model_copy(deep=True)
    before = declared(request)
    await plugin.before_model_callback(
        callback_context=_Context("probe", _Agent()), llm_request=request
    )
    return before, declared(request)


# --- Timing -------------------------------------------------------------------


@dataclass(frozen=True)
class Timing:
    """One path's per-call cost, in milliseconds."""

    name: str
    samples: list[float]

    @property
    def p50(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def p95(self) -> float:
        ordered = sorted(self.samples)
        # Nearest-rank, so the number is always a sample that really happened.
        rank = max(1, round(0.95 * len(ordered)))
        return ordered[rank - 1]

    @property
    def worst(self) -> float:
        return max(self.samples)


async def time_decide(plugin: SmartRouterPlugin, base: LlmRequest, runs: int) -> Timing:
    """Turn one, once per session. A fresh session id each time, so nothing is reused."""
    agent = _Agent()
    samples: list[float] = []
    for index in range(runs + WARMUP):
        # Deep, because the plugin edits `config.tools` in place and a shared config
        # would leave every run after the first pruning an already-pruned menu.
        request = base.model_copy(deep=True)
        ctx = _Context(f"decide-{index}", agent)
        start = time.perf_counter()
        await plugin.before_model_callback(callback_context=ctx, llm_request=request)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= WARMUP:
            samples.append(elapsed)
    return Timing("decide (turn one)", samples)


async def time_reuse(plugin: SmartRouterPlugin, base: LlmRequest, runs: int) -> Timing:
    """Every later turn. One session, decided once, then replayed."""
    agent = _Agent()
    ctx = _Context("reuse", agent)
    await plugin.before_model_callback(callback_context=ctx, llm_request=base.model_copy(deep=True))
    samples: list[float] = []
    for index in range(runs + WARMUP):
        request = base.model_copy(deep=True)
        start = time.perf_counter()
        await plugin.before_model_callback(callback_context=ctx, llm_request=request)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= WARMUP:
            samples.append(elapsed)
    return Timing("reuse (later turns)", samples)


def per_session_p95(decide: Timing, reuse: Timing, turns: float, embed_p95: float) -> float:
    """G3's unit: what one whole session pays, not what one turn pays.

    The three terms are added at their own p95 rather than convolved, which overstates
    the total: a session that draws the worst decide is not the session that also draws
    the worst reuse on every one of its later turns. Overstating is the right direction
    for a bar you have to stay under.
    """
    return decide.p95 + embed_p95 + max(0.0, turns - 1.0) * reuse.p95


def report(decide: Timing, reuse: Timing, turns: float, embed_p95: float) -> None:
    print(f"\nRouter overhead, {len(decide.samples)} iterations of each path\n")
    print(f"{'path':<22}{'p50 ms':>10}{'mean ms':>10}{'p95 ms':>10}{'worst ms':>10}")
    for timing in (decide, reuse):
        print(
            f"{timing.name:<22}{timing.p50:>10.2f}{timing.mean:>10.2f}"
            f"{timing.p95:>10.2f}{timing.worst:>10.2f}"
        )

    in_process = per_session_p95(decide, reuse, turns, 0.0)
    total = per_session_p95(decide, reuse, turns, embed_p95)
    print(f"\nA session of {turns:.2f} turns, at p95 on every term:\n")
    print(f"  in-process        {in_process:>8.2f} ms   this script")
    print(f"  embedding call    {embed_p95:>8.2f} ms   check_embedding_parity.py, 19.23")
    print(f"  {'-' * 46}")
    print(f"  G3                {total:>8.2f} ms   against a {G3_BAR_MS:.0f} ms bar")

    verdict = "PASS" if total <= G3_BAR_MS else "FAIL"
    headroom = G3_BAR_MS - total
    print(f"\n{verdict}: {abs(headroom):.0f} ms {'under' if headroom >= 0 else 'over'} the bar.")
    share = 100.0 * embed_p95 / total if total else 0.0
    print(
        f"The embedding call is {share:.0f}% of it, so the number an operator can move is the "
        f"endpoint, not the plugin."
    )


# --- Entry point --------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--turns",
        type=float,
        default=TURNS_PER_SESSION,
        help="turns per session; G3 is a per-session bar",
    )
    parser.add_argument(
        "--embed-p95",
        type=float,
        default=EMBED_P95_MS,
        help="the network term, added rather than measured; 0 for the in-process cost alone",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Same deployment as every other script here (DESIGN.md 3.1). The router's own config
    # wins if it is set; otherwise deployment.py asks the environment, and says which variable
    # is missing if it has to give up. Without an artifact there is no booster to time.
    cfg = load_config()
    cfg = replace(
        cfg,
        gcs_bucket=require_bucket(cfg.gcs_bucket),
        bq_project=require_project(cfg.bq_project),
    )

    request = reference_request()
    plugin = build_plugin(cfg, cfg.embedding_dims)

    before, after = asyncio.run(probe(plugin, request))
    if after >= before:
        print(
            f"The router left all {before} declarations in place, so this run would time an "
            f"early return rather than the hot path. Check that an artifact is published for "
            f"agent {DEFAULT_AGENT!r} in gs://{cfg.gcs_bucket}.",
            file=sys.stderr,
        )
        return 2
    print(f"Pruned {before} declarations to {after}. Timing that.", file=sys.stderr)

    decide = asyncio.run(time_decide(plugin, request, args.iterations))
    reuse = asyncio.run(time_reuse(plugin, request, args.iterations))
    report(decide, reuse, args.turns, args.embed_p95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
