#!/usr/bin/env python3
"""Generate the reference agent's telemetry corpus with a Gemini-simulated customer.

DESIGN.md 3.3. A second Gemini call plays the customer, the reference agent answers
with its real tools, and the BigQuery Agent Analytics plugin records what happened.
The label used for training is then whatever Gemini actually chose when the agent
ran, which is exactly the prediction target in production.

The honesty constraint this script exists to protect
----------------------------------------------------
**The generator is blind to the tool list.** The customer simulator is given a
persona and a scenario and nothing else: no tool name, no schema, no tool
description, no capability list, no expected label. If a tool name reached the
customer prompt the corpus would be worthless, because the preflight queries would
then measure how well we can predict a tool from a prompt we wrote to name that
tool. `assert_scenarios_are_blind()` fails the run before any money is spent, and
`assert_prompt_is_blind()` re-checks the finished prompt string at the API
boundary. `KNOWN_TOOL_NAMES` below exists **only** so those two checks have
something to compare against, and it must never be passed into a prompt.

The corpus this produces is synthetic. `corpus.meta.json` records
`corpus_source: "synthetic"`, every downstream preflight output carries it, and no
synthetic run may satisfy a gate silently (DESIGN.md 3.3, R12).

Cost and restartability
-----------------------
About 1,700 sessions of about 3 turns is roughly 5,000 turns and roughly $15 of
Flash. The script refuses to start without `--confirm`, prints a running estimate,
and appends one journal line per finished session so a restart skips the sessions
already paid for.

Usage
-----
    uv run python scripts/generate_corpus.py --dry-run
    uv run python scripts/generate_corpus.py --confirm --limit 5
    uv run python scripts/generate_corpus.py --confirm
    uv run python scripts/generate_corpus.py --export-only    # rebuild the artifact

The router plugin is deliberately NOT installed here. The corpus must record what
an unrouted agent did, or the classifier learns from its own predecessor's pruning
(DESIGN.md 11.4, build order task 3 before task 11).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deployment import DEFAULT_PROJECT, require_project  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "examples" / "reference_agent" / "scenarios"
CORPUS_DIR = REPO_ROOT / "examples" / "reference_agent" / "corpus"
DEFAULT_JOURNAL_DIR = REPO_ROOT / ".corpus_runs"

# --- Defaults. Every one of these is overridable by environment variable. -------
# No fallback project: this writes a corpus into somebody's BigQuery and bills somebody's
# Vertex quota, so it asks rather than guesses. main() is where the missing case is reported,
# so that `--help` still answers.
PROJECT = DEFAULT_PROJECT
DATASET = os.environ.get("SMART_ROUTER_BQ_DATASET", "smart_router")
BQ_LOCATION = os.environ.get("SMART_ROUTER_BQ_LOCATION", "US")
VERTEX_LOCATION = os.environ.get("SMART_ROUTER_VERTEX_LOCATION", "us-central1")
CUSTOMER_MODEL = os.environ.get("SMART_ROUTER_CUSTOMER_MODEL", "gemini-2.5-flash")
SESSIONS = int(os.environ.get("SMART_ROUTER_CORPUS_SESSIONS", "1700"))
SEED = int(os.environ.get("SMART_ROUTER_CORPUS_SEED", "20260831"))
CONCURRENCY = int(os.environ.get("SMART_ROUTER_CORPUS_CONCURRENCY", "4"))

# Event types the committed artifact leaves out. See export_corpus() for why.
EXPORT_EXCLUDED_EVENTS = ("LLM_REQUEST",)

# Gemini 2.5 Flash list prices, USD per million tokens, as of the design date.
# Used for the running estimate only. This is not a billing record.
PRICE_INPUT = float(os.environ.get("SMART_ROUTER_PRICE_INPUT_PER_M", "0.30"))
PRICE_OUTPUT = float(os.environ.get("SMART_ROUTER_PRICE_OUTPUT_PER_M", "2.50"))

# Pre-run estimate only; the running total uses measured usage_metadata.
# Calibrated against a 12-session smoke run on the 100-tool surface of
# DESIGN.md 19.15, which measured 28,400 input tokens per user turn. A user turn
# is more than one LLM call whenever a tool is used, which is why this is much
# larger than the 10,565-token declaration surface. The figure below is 16%
# under the smoke run because a long run warms the shared prefix and a 12-session
# one does not; that is the same margin the 25-tool constant carried.
EST_INPUT_TOKENS_PER_TURN = 24000
EST_OUTPUT_TOKENS_PER_TURN = 180

# ADC only. No key files, no GOOGLE_APPLICATION_CREDENTIALS (CODE_STANDARDS 4).

# --- The negative check, and nothing else ---------------------------------------
# Mirrors DESIGN.md 3.2. Read by assert_scenarios_are_blind() and
# assert_prompt_is_blind() so a leaked tool name fails the run loudly. Never
# rendered into a prompt, never passed to a model, never written to the corpus.
KNOWN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_order_status",
        "list_orders",
        "cancel_order",
        "modify_shipping_address",
        "start_return",
        "get_return_status",
        "get_return_policy",
        "track_shipment",
        "get_shipping_options",
        "estimate_delivery",
        "get_payment_methods",
        "add_payment_method",
        "get_invoice",
        "request_refund",
        "search_products",
        "get_product_details",
        "check_stock",
        "get_price",
        "get_account",
        "update_email",
        "get_loyalty_points",
        "list_addresses",
        "get_faq",
        "create_ticket",
        "escalate_to_human",
        "transfer_to_agent",
    }
)

# Vocabulary that only someone who has seen the tool list would write. A persona
# that talks about tools has stopped being a customer.
JARGON_PHRASES: tuple[str, ...] = (
    "tool call",
    "tool name",
    "tool list",
    "toolset",
    "function call",
    "function calling",
    "function declaration",
    "available tools",
    "api endpoint",
    "json schema",
)


class LeakError(RuntimeError):
    """A tool name reached text that a model is about to see."""


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with camelCase split apart first.

    Splitting camelCase means `getOrderStatus` tokenizes the same as
    `get_order_status` and `get order status`, so all three spellings of a leak
    are caught by one comparison.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.findall(r"[a-z0-9]+", spaced.lower())


def contains_phrase(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    """True if phrase_tokens appears as a contiguous run inside tokens."""
    n = len(phrase_tokens)
    if n == 0 or n > len(tokens):
        return False
    return any(list(tokens[i : i + n]) == list(phrase_tokens) for i in range(len(tokens) - n + 1))


def find_leaks(text: str, tool_names: Iterable[str]) -> list[str]:
    """Every tool name or tool-shop jargon phrase present in text."""
    tokens = tokenize(text)
    hits = [name for name in sorted(tool_names) if contains_phrase(tokens, tokenize(name))]
    hits += [p for p in JARGON_PHRASES if contains_phrase(tokens, tokenize(p))]
    return hits


def assert_scenarios_are_blind(
    personas: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    tool_names: Iterable[str],
) -> None:
    """Fail the whole run if any persona or scenario text names a tool.

    This is the assertion the corpus depends on. A leak here would make the
    preflight gates self-fulfilling, so it is a hard stop and not a warning, and
    it runs in --dry-run too.
    """
    names = frozenset(tool_names)
    problems: list[str] = []
    for persona in personas:
        text = json.dumps(persona, ensure_ascii=False)
        for hit in find_leaks(text, names):
            problems.append(f"persona {persona.get('id')}: {hit!r}")
    for scenario in scenarios:
        text = json.dumps(scenario, ensure_ascii=False)
        for hit in find_leaks(text, names):
            problems.append(f"scenario {scenario.get('id')}: {hit!r}")
    if problems:
        raise LeakError(
            "Tool names leaked into the scenario files. The corpus would be circular "
            "(DESIGN.md 3.3), so nothing will run until these are rewritten in customer "
            "language:\n  " + "\n  ".join(problems)
        )


def assert_prompt_is_blind(prompt: str, tool_names: Iterable[str]) -> None:
    """Last gate before the customer prompt leaves the process."""
    hits = find_leaks(prompt, tool_names)
    if hits:
        raise LeakError(f"Customer prompt names tools {hits}. Refusing to call the model.")


# --- Inputs ---------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_root_agent(*, required: bool) -> Any | None:
    """Import the reference agent. The only contract: `root_agent` in agent.py.

    A dry run is allowed to proceed without it so the scenario files can be
    reviewed on their own, but then only the committed tool-name list is
    available for the blindness check and the run says so.
    """
    examples = str(REPO_ROOT / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    try:
        from reference_agent.agent import root_agent
    except Exception as exc:  # pragma: no cover - depends on a sibling deliverable
        if required:
            raise SystemExit(
                f"Could not import examples/reference_agent/agent.py: {exc}\n"
                "The generator drives the real agent, so it cannot run without it."
            ) from exc
        print(f"WARNING: reference agent not importable ({exc}). Checking against the committed")
        print("WARNING: tool-name list only; a tool added to the agent would not be checked.")
        return None
    return root_agent


def live_tool_names(root_agent: Any) -> frozenset[str]:
    """Tool names as the running agent declares them, for the negative check only."""
    names: set[str] = set()
    for tool in getattr(root_agent, "tools", None) or []:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if isinstance(name, str):
            names.add(name)
    return frozenset(names)


# --- Planning -------------------------------------------------------------------


@dataclass(frozen=True)
class SessionPlan:
    """One planned conversation. Deterministic given the seed."""

    index: int
    session_id: str
    user_id: str
    persona: Mapping[str, Any]
    scenario: Mapping[str, Any]
    turns: int


def build_plan(
    personas: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    sessions: int,
    seed: int,
) -> list[SessionPlan]:
    """Sample (persona, scenario) pairs with a seeded RNG.

    The persona is the customer identity, so a persona's sessions share a user_id
    and P2's per-user baseline has history to work with.
    """
    by_id = {p["id"]: p for p in personas}
    weights = [float(s.get("weight", 1)) for s in scenarios]
    rng = random.Random(seed)
    plan: list[SessionPlan] = []
    for index in range(sessions):
        scenario = rng.choices(scenarios, weights=weights, k=1)[0]
        persona = by_id[scenario["persona_id"]]
        plan.append(
            SessionPlan(
                index=index,
                session_id=session_id_for(seed, index),
                user_id=f"corpus-{persona['id']}",
                persona=persona,
                scenario=scenario,
                turns=int(scenario.get("turns", 3)),
            )
        )
    return plan


def session_prefix(seed: int) -> str:
    return f"corpus-{seed}-"


def session_id_for(seed: int, index: int) -> str:
    return f"{session_prefix(seed)}{index:05d}"


# --- The simulated customer ------------------------------------------------------


def customer_system_prompt(persona: Mapping[str, Any], scenario: Mapping[str, Any]) -> str:
    """Build the customer's instructions from persona and scenario text alone.

    Everything this function can see is customer-side. It takes no tool list, no
    agent description and no expected answer, and it must stay that way.
    """
    facts = "\n".join(f"- {fact}" for fact in persona.get("facts", []))
    return (
        "You are a customer messaging an online shop's support chat. You are not an "
        "assistant and you never help; you are the person with the problem.\n\n"
        f"Your situation: {persona['situation']}\n"
        f"Your mood: {persona['mood']}\n"
        f"What you know:\n{facts}\n\n"
        f"What you came here for: {scenario['opening']}\n"
        f"How you carry on: {scenario['follow_up_style']}\n\n"
        "Rules:\n"
        "- Write one short message at a time, the way a real person types into a chat "
        "box. One or two sentences, sometimes a fragment.\n"
        "- Say what you want. Never say how the shop should do it, and never mention "
        "software, systems or lookups.\n"
        "- Stay in character. Do not narrate, do not use quotation marks, do not "
        "explain yourself to anyone but the shop.\n"
        "- Reply only with the message text."
    )


def customer_contents(history: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """History from the customer's point of view: the customer is the model.

    Roles are inverted relative to the agent's transcript, because to the
    simulator the shop's replies are the incoming side of the conversation.
    """
    contents: list[dict[str, Any]] = []
    if not history:
        contents.append({"role": "user", "parts": [{"text": "(the chat has just opened)"}]})
    for speaker, text in history:
        role = "model" if speaker == "customer" else "user"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


@dataclass
class Spend:
    """Running token and dollar totals. Measured where possible, estimated where not."""

    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    sessions: int = 0
    errors: int = 0

    def add_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        self.output_tokens += int(getattr(usage, "thoughts_token_count", 0) or 0)

    @property
    def usd(self) -> float:
        return (self.input_tokens * PRICE_INPUT + self.output_tokens * PRICE_OUTPUT) / 1_000_000


def estimate_usd(turns: int) -> float:
    """Pre-run estimate, printed before the confirmation gate."""
    tokens_in = turns * EST_INPUT_TOKENS_PER_TURN
    tokens_out = turns * EST_OUTPUT_TOKENS_PER_TURN
    return (tokens_in * PRICE_INPUT + tokens_out * PRICE_OUTPUT) / 1_000_000


# --- Driving one session ---------------------------------------------------------


async def next_customer_message(
    client: Any,
    model: str,
    system_prompt: str,
    history: Sequence[tuple[str, str]],
    spend: Spend,
) -> str:
    """One turn of the simulated customer."""
    from google.genai import types

    response = await client.aio.models.generate_content(
        model=model,
        contents=customer_contents(history),
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            max_output_tokens=200,
        ),
    )
    spend.add_usage(getattr(response, "usage_metadata", None))
    text = (response.text or "").strip()
    return text


async def run_session(
    plan: SessionPlan,
    runner: Any,
    client: Any,
    model: str,
    tool_names: frozenset[str],
    spend: Spend,
) -> dict[str, Any]:
    """Drive one conversation end to end and return its journal record."""
    from google.genai import types

    system_prompt = customer_system_prompt(plan.persona, plan.scenario)
    assert_prompt_is_blind(system_prompt, tool_names)

    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=plan.user_id, session_id=plan.session_id
    )

    history: list[tuple[str, str]] = []
    prompts: list[str] = []
    observed_tools: list[str] = []
    started = time.monotonic()

    for _ in range(plan.turns):
        message = await next_customer_message(client, model, system_prompt, history, spend)
        if not message:
            break
        history.append(("customer", message))
        prompts.append(message)

        reply_parts: list[str] = []
        async for event in runner.run_async(
            user_id=plan.user_id,
            session_id=plan.session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            spend.add_usage(getattr(event, "usage_metadata", None))
            observed_tools.extend(call.name for call in event.get_function_calls())
            if event.is_final_response() and event.content and event.content.parts:
                reply_parts.extend(part.text for part in event.content.parts if part.text)
        history.append(("agent", "\n".join(reply_parts).strip() or "(no reply)"))
        spend.turns += 1

    spend.sessions += 1
    return {
        "session_id": plan.session_id,
        "user_id": plan.user_id,
        "persona_id": plan.persona["id"],
        "scenario_id": plan.scenario["id"],
        "kind": plan.scenario.get("kind"),
        "turns": len(prompts),
        "prompts": prompts,
        # Diagnostic only. The authoritative label is what BQAA recorded; this is
        # here so a failed run can be triaged without a BigQuery round trip.
        "observed_tools": observed_tools,
        "wall_ms": int((time.monotonic() - started) * 1000),
        "finished_at": datetime.now(UTC).isoformat(),
    }


# --- Journal (resume) -------------------------------------------------------------


def journal_path(explicit: str | None, seed: int) -> Path:
    """Where per-session progress is recorded. Nothing is created until a write."""
    if explicit:
        return Path(explicit).resolve()
    return DEFAULT_JOURNAL_DIR / f"seed-{seed}.jsonl"


def window_start(records: Sequence[Mapping[str, Any]], fallback: datetime) -> datetime:
    """Earliest moment this corpus could have been written.

    A resumed run spans days, so the export window has to start at the first
    session in the journal, not at this invocation. One minute of slack covers the
    gap between a session starting and its journal line landing.
    """
    stamps = [str(r["finished_at"]) for r in records if r.get("finished_at")]
    if not stamps:
        return fallback
    earliest = min(datetime.fromisoformat(s) for s in stamps)
    return min(earliest, fallback) - timedelta(minutes=1)


def read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_journal(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Progress is a local run artifact, not a deliverable. Keep it out of git
    # without touching the shared .gitignore.
    ignore = path.parent / ".gitignore"
    if path.parent == DEFAULT_JOURNAL_DIR and not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- The corpus artifact ----------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Make a BigQuery row value JSON-serialisable without changing its meaning."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def export_corpus(agent_name: str, seed: int, since: datetime, out_path: Path) -> int:
    """Export the rows BQAA wrote for this run to corpus.jsonl.gz.

    The artifact is the rows themselves, not a summary, so `bq load` puts them
    back into `agent_events` (DESIGN.md 3.3). Filtering on `timestamp` and `agent`
    keeps the read inside the partition and cluster pruning that DESIGN.md 18.2
    records for this table.

    One event type is left out. An `LLM_REQUEST` row carries the whole tool list
    serialized into its attributes, so on the 100-tool surface it averages 94 KB
    and accounts for 89% of the export's bytes: 280 MB of a 312 MB sample. Nothing
    in this project reads those rows. The SQL reads `USER_MESSAGE_RECEIVED`,
    `TOOL_STARTING` and `TOOL_COMPLETED` from the base table, and the A/B report
    reads BQAA's `v_llm_response` view. Dropping them takes the artifact from
    197 MB to something a repository can hold, and what a fresh clone reloads is
    every row the project actually queries. The full set stays in BigQuery.

    Written gzipped because the point of committing it is that a fresh clone does
    not pay the generator's bill again, and GitHub refuses anything over 100 MB.
    `bq load` will NOT decompress a local .gz, verified, so `provision.sh` unpacks
    it to a temp file first.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT)
    sql = f"""
        SELECT *
        FROM `{PROJECT}.{DATASET}.agent_events`
        WHERE timestamp >= @since
          AND agent = @agent
          AND STARTS_WITH(session_id, @prefix)
          AND event_type NOT IN UNNEST(@excluded)
        ORDER BY timestamp
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("since", "TIMESTAMP", since),
            bigquery.ScalarQueryParameter("agent", "STRING", agent_name),
            bigquery.ScalarQueryParameter("prefix", "STRING", session_prefix(seed)),
            bigquery.ArrayQueryParameter("excluded", "STRING", list(EXPORT_EXCLUDED_EVENTS)),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        for row in client.query(sql, job_config=job_config, location=BQ_LOCATION).result():
            handle.write(json.dumps(json_safe(dict(row)), ensure_ascii=False) + "\n")
            rows += 1
    return rows


def git_commit() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def write_meta(
    path: Path,
    *,
    seed: int,
    sessions: int,
    turns: int,
    rows: int | None,
    agent_name: str,
    agent_model: str | None,
    customer_model: str,
    estimated_cost_usd: float,
    started_at: datetime,
) -> None:
    """Record what made this corpus, and that it is synthetic."""
    meta = {
        # Read by every preflight query's output. A synthetic run must never
        # satisfy a gate silently (DESIGN.md 3.3).
        "corpus_source": "synthetic",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_started_at": started_at.isoformat(),
        "generator": {
            "script": "scripts/generate_corpus.py",
            "git": git_commit(),
            "personas_sha256": sha256_file(SCENARIO_DIR / "personas.json"),
            "scenarios_sha256": sha256_file(SCENARIO_DIR / "scenarios.json"),
        },
        "models": {"agent": agent_model, "customer_simulator": customer_model},
        "seed": seed,
        "sessions": sessions,
        "turns": turns,
        "rows": rows,
        "bigquery": {
            "dataset": DATASET,
            "table": "agent_events",
            "location": BQ_LOCATION,
            "agent": agent_name,
            "session_prefix": session_prefix(seed),
        },
        "estimated_cost_usd": round(estimated_cost_usd, 2),
        "caveat": (
            "Gemini is nondeterministic, so re-running the generator produces a different "
            "corpus. Preflight numbers are reproducible from this artifact, not from the "
            "seed. P4 is a pipeline test on this data, not a gate, and no lift number "
            "measured here may be published (DESIGN.md 3.3, R12, R13)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Orchestration -----------------------------------------------------------------


@dataclass
class RunInputs:
    """Everything the async run needs, resolved once by main()."""

    plan: list[SessionPlan]
    tool_names: frozenset[str]
    root_agent: Any
    journal: Path
    args: argparse.Namespace
    spend: Spend = field(default_factory=Spend)


async def run_all(inputs: RunInputs) -> None:
    """Drive the planned sessions, bounded concurrency, journalling as they land.

    The App comes from the reference agent's own runner rather than being rebuilt
    here. Plugin registration order and the cache config are load-bearing
    (DESIGN.md 6.3, 11.5), and a second copy of that wiring would drift.
    """
    from google import genai
    from google.adk.runners import InMemoryRunner
    from reference_agent.run_agent import build_app

    args = inputs.args
    runner = InMemoryRunner(app=build_app(analytics=True))
    client = genai.Client(vertexai=True, project=PROJECT, location=VERTEX_LOCATION)

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    total = len(inputs.plan)

    async def one(plan: SessionPlan) -> None:
        async with semaphore:
            try:
                record = await run_session(
                    plan, runner, client, args.customer_model, inputs.tool_names, inputs.spend
                )
            except LeakError:
                raise
            except Exception as exc:  # one bad session must not end the run
                inputs.spend.errors += 1
                print(f"  session {plan.session_id} failed: {type(exc).__name__}: {exc}")
                return
            async with write_lock:
                append_journal(inputs.journal, record)
                done = inputs.spend.sessions
                if done % 25 == 0 or done == total:
                    print(
                        f"  {done}/{total} sessions, {inputs.spend.turns} turns, "
                        f"~${inputs.spend.usd:.2f} spent, "
                        f"~${inputs.spend.usd / max(done, 1) * total:.2f} projected"
                    )

    try:
        await asyncio.gather(*(one(p) for p in inputs.plan))
    finally:
        # BQAA only flushes its buffered rows on close. Skipping this loses the tail
        # of the run, and the tail is the part that was paid for most recently.
        await runner.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="cap sessions run this invocation")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true", help="plan and price it, call nothing")
    parser.add_argument("--confirm", action="store_true", help="required: this spends real money")
    parser.add_argument("--journal", default=None, help="resume file (default .corpus_runs/)")
    parser.add_argument("--customer-model", default=CUSTOMER_MODEL)
    # The GCP target is env-only on purpose. The reference agent's runner reads the
    # same variables to wire BQAA, so a flag here could point the export at a
    # dataset the run did not write to.
    parser.add_argument("--skip-export", action="store_true", help="do not rebuild corpus.jsonl.gz")
    parser.add_argument("--export-only", action="store_true", help="rebuild the artifact and stop")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_project(PROJECT)
    started_at = datetime.now(UTC)

    personas = load_json(SCENARIO_DIR / "personas.json")["personas"]
    scenarios = load_json(SCENARIO_DIR / "scenarios.json")["scenarios"]

    root_agent = load_root_agent(required=not args.dry_run)
    agent_name = str(getattr(root_agent, "name", "support"))
    agent_model = getattr(root_agent, "model", None)
    agent_model = agent_model if isinstance(agent_model, str) else str(agent_model or "")

    # Union of the committed list and what the agent actually declares, so a tool
    # added to the agent is checked even before DESIGN.md 3.2 is updated.
    tool_names = KNOWN_TOOL_NAMES | live_tool_names(root_agent)
    assert_scenarios_are_blind(personas, scenarios, tool_names)
    print(
        f"blindness check passed: {len(personas)} personas, {len(scenarios)} scenarios, "
        f"{len(tool_names)} tool names checked"
    )

    journal = journal_path(args.journal, args.seed)
    done = {record["session_id"] for record in read_journal(journal)}

    if args.export_only:
        records = read_journal(journal)
        # Re-exporting spends nothing, so carry forward what the paid run cost
        # rather than overwriting it with zero.
        meta_path = CORPUS_DIR / "corpus.meta.json"
        prior_cost = 0.0
        if meta_path.exists():
            prior_cost = float(load_json(meta_path).get("estimated_cost_usd") or 0.0)
        exported = export_corpus(
            agent_name, args.seed, window_start(records, started_at), CORPUS_DIR / "corpus.jsonl.gz"
        )
        write_meta(
            meta_path,
            seed=args.seed,
            sessions=len(records),
            turns=sum(int(r.get("turns", 0)) for r in records),
            rows=exported,
            agent_name=agent_name,
            agent_model=agent_model,
            customer_model=args.customer_model,
            estimated_cost_usd=prior_cost,
            started_at=started_at,
        )
        print(f"exported {exported} rows to {CORPUS_DIR / 'corpus.jsonl.gz'}")
        return 0

    plan = [
        p
        for p in build_plan(personas, scenarios, args.sessions, args.seed)
        if p.session_id not in done
    ]
    if args.limit is not None:
        plan = plan[: args.limit]
    turns = sum(p.turns for p in plan)
    print(
        f"{len(plan)} sessions to run ({len(done)} already in {journal}), "
        f"~{turns} turns, estimated ~${estimate_usd(turns):.2f}"
    )

    if args.dry_run:
        for p in plan[:5]:
            print(f"  {p.session_id} {p.persona['id']}/{p.scenario['id']} {p.turns} turns")
        return 0
    if not args.confirm:
        print("Refusing to start: this spends real money. Re-run with --confirm.")
        return 2
    if not plan:
        print("Nothing to do.")
        return 0

    inputs = RunInputs(
        plan=plan, tool_names=tool_names, root_agent=root_agent, journal=journal, args=args
    )
    asyncio.run(run_all(inputs))

    records = read_journal(journal)
    rows: int | None = None
    if not args.skip_export:
        rows = export_corpus(
            agent_name, args.seed, window_start(records, started_at), CORPUS_DIR / "corpus.jsonl.gz"
        )
    write_meta(
        CORPUS_DIR / "corpus.meta.json",
        seed=args.seed,
        sessions=len(records),
        turns=sum(int(r.get("turns", 0)) for r in records),
        rows=rows,
        agent_name=agent_name,
        agent_model=agent_model,
        customer_model=args.customer_model,
        estimated_cost_usd=inputs.spend.usd,
        started_at=started_at,
    )
    print(
        f"done: {inputs.spend.sessions} sessions, {inputs.spend.turns} turns, "
        f"{inputs.spend.errors} failed, ~${inputs.spend.usd:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
