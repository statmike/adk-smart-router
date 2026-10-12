#!/usr/bin/env python
"""E8 — price speculative execution without building it.

Prefetch is cut from v1 and deferred to v1.2 over a data-leakage concern, and that decision
stands. This does not reopen it. What it does is settle the other half of the question, which
has never been answered: **if the leakage concern were solved tomorrow, would prefetch be worth
building?** That is arithmetic over data we already have, and answering it now means the v1.2
conversation starts from a number instead of an intuition.

The idea being priced: at turn one the router already has a ranked prediction, so it could run
the top-1 tool in parallel with the model call instead of waiting for the model to ask. A hit
hides the tool's latency inside the model's. A miss runs a tool nobody asked for.

    saving per session   = P(hit) * min(tool latency, model latency)
    cost per session     = P(1 - hit) * one unwanted tool execution

`P(hit)` has two factors and only one of them is the classifier. Getting the tool *name* right
is top-1 accuracy, which is measured here. Getting the *arguments* right is a separate problem
the router does not currently attempt, and DESIGN.md 19.56 measured how often the first call's
arguments are even present in the message the router sees: 25.2%. The rest are recalled from
history, normalised, or invented by the model. A prefetch cannot run a tool it cannot supply
arguments to, so that fraction is a ceiling on the whole idea and it is applied here rather
than mentioned in a footnote.

Cost is one BigQuery read. No embedding calls: the scoring uses the vectors BigQuery already
computed, because top-1 accuracy is the same either way and `check_router.py` already proves
the served path matches. Authentication is ADC only.

Usage:
    uv run python scripts/price_prefetch.py
    uv run python scripts/price_prefetch.py --grounded 0.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Predictor  # noqa: E402

DEFAULT_AGENT = "support"

# The training split, copied out of train_router.sql the same way check_router.py copies it.
# Repeated rather than imported: two scripts sharing a constant through a third module is a
# worse trade than two copies with a comment, and a mismatch shows up as an accuracy number
# that disagrees with the run record, which is a signal worth having.
EVAL_PCT = 25
OPENING_KEY = r"IFNULL(TRIM(REGEXP_REPLACE(LOWER(f.prompt_text), r'\s+', ' ')), f.session_id)"

# DESIGN.md 19.56: of 5,253 first-call arguments, 1,325 appear literally in the message the
# router had to work from. The rest the model recalled, normalised, or invented. A prefetch
# has to supply arguments before it can run anything, so this bounds the hit rate.
GROUNDED_ARGUMENTS = 0.252

# What fraction of a turn's latency prefetch has to remove before anyone would ship it. Chosen
# rather than derived; it is the reporting threshold, not a result.
WORTH_SHIPPING = 0.05

# Tool latencies to price the counterfactual at. The reference agent's tools are deterministic
# stubs, so the observed latency measures the corpus rather than the idea. These are the
# latencies a tool that talks to something would have.
HYPOTHETICAL_MS = (10, 50, 200, 1000, 3000)


@dataclass(frozen=True)
class Traffic:
    """What the agent's own logs say about the shape of a turn."""

    sessions: int
    tool_p50_ms: float
    tool_mean_ms: float
    model_p50_ms: float


@dataclass(frozen=True)
class Row:
    """One held-out session's first call, with the vector the router would have seen."""

    tool_name: str
    vector: list[float]


TRAFFIC_SQL = """
WITH first_call AS (
  SELECT session_id,
    ARRAY_AGG(STRUCT(tool_name, total_ms) ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS fc
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent
    -- A cache hit is the router's own work, not the tool's, and it takes no time to run.
    -- Including them would report a tool latency the tool never had (DESIGN.md 9.1, 11.5).
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  GROUP BY session_id
),
model AS (
  SELECT APPROX_QUANTILES(total_ms, 100)[OFFSET(50)] AS p50
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
)
SELECT
  (SELECT COUNT(*) FROM first_call) AS sessions,
  (SELECT APPROX_QUANTILES(fc.total_ms, 100)[OFFSET(50)] FROM first_call) AS tool_p50_ms,
  (SELECT AVG(fc.total_ms) FROM first_call) AS tool_mean_ms,
  (SELECT p50 FROM model) AS model_p50_ms
"""


def fetch_traffic(client: bigquery.Client, project: str, dataset: str, agent: str) -> Traffic:
    job = client.query(
        TRAFFIC_SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("agent", "STRING", agent)]
        ),
    )
    row = next(iter(job.result()))
    return Traffic(
        sessions=int(row.sessions),
        tool_p50_ms=float(row.tool_p50_ms),
        tool_mean_ms=float(row.tool_mean_ms),
        model_p50_ms=float(row.model_p50_ms),
    )


def fetch_rows(
    client: bigquery.Client, project: str, dataset: str, agent: str, dims: int, limit: int
) -> list[Row]:
    """Held-out first calls, one row per session.

    Prefetch fires once, at the opening of a session, so the unit here is the session's first
    tool call rather than every call it makes. The split is the one training used and it is
    decided per session by the opening message, so the query has to find each session's first
    message before it can tell which side the session is on. That is the `opening` CTE, copied
    from `check_router.py`, which copied it from `train_router.sql`.
    """
    table = f"{project}.{dataset}.router_features_{agent}"
    sql = f"""
      WITH opening AS (
        SELECT
          session_id,
          ARRAY_AGG(invocation_id ORDER BY prompt_ts, invocation_id LIMIT 1)[OFFSET(0)]
            AS invocation_id
        FROM `{table}`
        WHERE ARRAY_LENGTH(v) = @dims
        GROUP BY session_id
      ),
      held_out AS (
        SELECT f.session_id
        FROM `{table}` f
        JOIN opening o USING (invocation_id)
        WHERE ABS(MOD(FARM_FINGERPRINT({OPENING_KEY}), 100)) < {EVAL_PCT}
      ),
      first_call AS (
        SELECT
          session_id,
          ARRAY_AGG(STRUCT(tool_name, v) ORDER BY prompt_ts, invocation_id LIMIT 1)[OFFSET(0)]
            AS fc
        FROM `{table}`
        WHERE ARRAY_LENGTH(v) = @dims
          AND tool_name IS NOT NULL
          AND tool_name != 'ABSTAIN'
          AND session_id IN (SELECT session_id FROM held_out)
        GROUP BY session_id
      )
      SELECT fc.tool_name AS tool_name, fc.v AS v
      FROM first_call
      ORDER BY FARM_FINGERPRINT(session_id)
      LIMIT @limit
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dims", "INT64", dims),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        ),
    )
    return [Row(tool_name=r["tool_name"], vector=list(r["v"])) for r in job.result()]


def top1_accuracy(predictor: Predictor, rows: list[Row]) -> tuple[int, int]:
    """How often the router's first guess is the tool the session actually called first."""
    hits = scored = 0
    for row in rows:
        ranked = predictor.predict(row.vector)
        if not ranked:
            continue  # an empty ranking is a passthrough, which prefetches nothing.
        scored += 1
        hits += ranked[0].label == row.tool_name
    return hits, scored


def hidden_ms(hit: float, tool_ms: float, model_ms: float) -> float:
    """Latency prefetch removes from a turn, in expectation.

    A prefetch runs the tool alongside the model call, so what it hides is the tool's own
    time, and only up to the model's: a tool slower than the model finishes late no matter
    when it started, and the turn waits for it either way.
    """
    return hit * min(tool_ms, model_ms)


def required_tool_ms(hit: float, model_ms: float, worth: float) -> float:
    """How slow a tool has to be before prefetching it is worth `worth` of a turn.

    Infinite when the hit rate is zero, and also when the hit rate alone is below `worth`:
    `hidden_ms` is capped at `hit * model_ms`, so a hit rate under the threshold cannot reach
    it at any tool latency. That is a ceiling on the idea rather than on this corpus.
    """
    if hit <= worth:
        return float("inf")
    return worth * model_ms / hit


def report(traffic: Traffic, hits: int, scored: int, grounded: float) -> None:
    name_right = hits / scored if scored else 0.0
    hit = name_right * grounded

    print("\n--- What a turn looks like ----------------------------------------")
    print(f"  sessions                {traffic.sessions:,}")
    print(f"  first tool call, median {traffic.tool_p50_ms:,.0f} ms")
    print(f"  model call, median      {traffic.model_p50_ms:,.0f} ms")
    share = traffic.tool_p50_ms / traffic.model_p50_ms
    print(f"  the tool is {share:.2%} of the turn")

    print("\n--- Could prefetch fire? ------------------------------------------")
    print(f"  top-1 tool name correct   {name_right:.1%}  ({hits:,} of {scored:,} held out)")
    print(f"  arguments in the message  {grounded:.1%}  (DESIGN.md 19.56)")
    print(f"  both, so P(hit)           {hit:.1%}")
    print(f"  wasted tool runs          {1 - hit:.1%} of sessions prefetched")

    print("\n--- What it would be worth ----------------------------------------")
    print("  Priced at the observed latency, then at latencies a real tool would have.\n")
    print(f"  {'tool latency':>14}  {'hidden per turn':>16}  {'share of turn':>14}")
    observed = round(traffic.tool_p50_ms)
    for tool_ms in (observed, *HYPOTHETICAL_MS):
        hidden = hidden_ms(hit, float(tool_ms), traffic.model_p50_ms)
        share_of_turn = hidden / traffic.model_p50_ms
        mark = "  <- observed" if tool_ms == observed else ""
        print(f"  {tool_ms:>11,} ms  {hidden:>13,.0f} ms  {share_of_turn:>13.2%}{mark}")

    need = required_tool_ms(hit, traffic.model_p50_ms, WORTH_SHIPPING)
    print("\n--- The verdict ---------------------------------------------------")
    if need == float("inf"):
        print(f"  No tool latency reaches {WORTH_SHIPPING:.0%} of a turn. Prefetch hides at most")
        print(f"  P(hit) of the turn, which is {hit:.1%}, and that is below the bar. The hit")
        print("  rate is the binding constraint here, not the tool.")
    else:
        print(f"  To hide {WORTH_SHIPPING:.0%} of a turn at P(hit) = {hit:.1%}, a tool must take")
        print(f"  {need:,.0f} ms. This agent's tools take {traffic.tool_p50_ms:,.0f} ms.")
    print()
    if traffic.tool_p50_ms >= need:
        print("  Prefetch would pay here. The v1.2 leakage question is the only thing in")
        print("  the way, and it is now worth the argument.")
    else:
        print("  Prefetch cannot pay on this corpus, and the reason is the corpus rather")
        print("  than the idea: these tools are deterministic stubs with no I/O, so there")
        print("  is no latency to hide. This is the same gap that blocks the tool-response")
        print("  channel, and E2's document-returning corpus should close both at once by")
        print("  giving its tools realistic latency as well as realistic payloads.")
        print()
        print("  Note what does not change with the corpus: the argument ceiling. Even with")
        print(f"  a perfect classifier, prefetch fires on {grounded:.1%} of sessions until the")
        print("  router can predict arguments, which is a v2 item. Prefetch is blocked on")
        print("  argument prediction at least as hard as it is blocked on leakage.")


def main() -> int:
    args = parse_args()
    cfg = load_config(gcs_bucket=args.bucket, bq_project=args.project)

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.classes)} classes")

    client = bigquery.Client(project=args.project)
    traffic = fetch_traffic(client, args.project, args.dataset, args.agent)
    rows = fetch_rows(
        client, args.project, args.dataset, args.agent, cfg.embedding_dims, args.limit
    )
    if not rows:
        print("\nNo held-out first calls to score.", file=sys.stderr)
        return 1

    hits, scored = top1_accuracy(Predictor(model), rows)
    if not scored:
        print("\nEvery held-out row came back a passthrough; nothing to price.", file=sys.stderr)
        return 1
    report(traffic, hits, scored, args.grounded)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument("--limit", type=int, default=2000, help="held-out first calls to score")
    parser.add_argument(
        "--grounded",
        type=float,
        default=GROUNDED_ARGUMENTS,
        help="fraction of first calls whose arguments appear in the message (DESIGN.md 19.56)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
