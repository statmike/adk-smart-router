#!/usr/bin/env python
"""Score a trained-but-unpublished model under DESIGN.md 19.4, before promoting anything.

`score_sets.py` scores the artifact `ModelStore` would serve. That is the right thing to
measure and the wrong thing to measure first, because a model only becomes an artifact by
passing a gate, and V2-2 changes what the gate should be asking. Judging a candidate by
publishing it is how a bad gate promotes a bad model.

So this reads the BQML model straight out of BigQuery. `ML.PREDICT` over the held-out
split gives the same ranking `predictor.Predictor` would produce from the exported
booster, one row per session, and everything downstream is `score_sets`: the same menus,
the same `pruner.decide`, the same arms, the same report. Nothing is written anywhere.

Two differences from `score_sets.py`, both deliberate:

* the menu pool is chosen here rather than read from an artifact, because a candidate has
  no artifact. It is chosen from the **training** half of the same corpus the booster was
  fit on, so the pool never sees an eval session's tools.
* sessions come from the flattened training table rather than a fresh query, so the tool
  sets are exactly the labels the model was trained against, cache-hit and routed-session
  exclusions included (DESIGN.md 19.2).

Cost is one `ML.PREDICT` over a few thousand rows and 25 token counts. No embedding calls
and no writes. Authentication is ADC only.

Usage:
    uv run python scripts/score_candidate.py
    uv run python scripts/score_candidate.py --sweep 500 1000 2278
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_tokens  # noqa: E402
from score_sets import (  # noqa: E402
    CACHED_RATE,
    DEFAULT_AGENT,
    Arm,
    Session,
    menu_always,
    menu_best,
    menu_router,
    menu_true,
    report,
    score,
)

from adk_smart_router import menus as menu_rules  # noqa: E402
from adk_smart_router.config import FRAMEWORK_TOOLS, load_config  # noqa: E402
from adk_smart_router.predictor import Prediction  # noqa: E402
from adk_smart_router.pruner import ABSTAIN  # noqa: E402

# One row per held-out session: its ranking, the tools it wanted, and what it cost.
#
# `one_row` exists because a session has one row per tool it wanted and they all carry the
# same vector, so predicting all of them would pay for the same inference several times and
# then have to deduplicate an ARRAY<STRUCT>, which BigQuery will not group by.
CANDIDATE_SQL = """
WITH one_row AS (
  SELECT * EXCEPT (tool_name)
  FROM `{project}.{dataset}.router_flat_{agent}`
  WHERE split_col AND STARTS_WITH(session_id, @prefix)
  QUALIFY ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY session_id) = 1
),
predicted AS (
  SELECT session_id, predicted_tool_name_probs AS probs
  FROM ML.PREDICT(MODEL `{project}.{dataset}.router_{agent}`, (SELECT * FROM one_row))
),
wanted AS (
  SELECT session_id, ARRAY_AGG(DISTINCT tool_name) AS labels
  FROM `{project}.{dataset}.router_flat_{agent}`
  WHERE split_col
  GROUP BY session_id
),
spend AS (
  SELECT
    session_id,
    COUNT(*) AS turns,
    SUM(usage_prompt_tokens - LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens)
        + LEAST(COALESCE(usage_cached_tokens, 0), usage_prompt_tokens) * {rate}) AS billed
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent AND usage_prompt_tokens IS NOT NULL
  GROUP BY session_id
)
SELECT p.session_id, p.probs, w.labels, IFNULL(s.turns, 0) AS turns, IFNULL(s.billed, 0.0) AS billed
FROM predicted p
JOIN wanted w USING (session_id)
LEFT JOIN spend s USING (session_id)
"""

# The training half, as tool sets. This is the menu pool's corpus, and it is the one place
# the eval split must not leak into.
TRAIN_SETS_SQL = """
SELECT session_id, ARRAY_AGG(DISTINCT tool_name) AS labels
FROM `{project}.{dataset}.router_flat_{agent}`
WHERE NOT split_col
GROUP BY session_id
"""


@dataclass(frozen=True)
class Candidate:
    """One held-out session and the ranking the unpublished model gives it."""

    session: Session
    ranked: list[Prediction]


def fetch_candidates(project: str, dataset: str, agent: str, prefix: str) -> list[Candidate]:
    job = bigquery.Client(project=project).query(
        CANDIDATE_SQL.format(project=project, dataset=dataset, agent=agent, rate=CACHED_RATE),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            ]
        ),
    )
    found = []
    for row in job.result():
        tools = frozenset(row["labels"]) - {ABSTAIN}
        found.append(
            Candidate(
                session=Session(
                    session_id=row["session_id"],
                    vector=[],
                    tools=tools,
                    turns=int(row["turns"]),
                    billed=float(row["billed"]),
                ),
                ranked=sorted(
                    (Prediction(p["label"], float(p["prob"])) for p in row["probs"]),
                    key=lambda pred: (-pred.probability, pred.label),
                ),
            )
        )
    return found


def fetch_train_sets(project: str, dataset: str, agent: str) -> list[list[str]]:
    job = bigquery.Client(project=project).query(
        TRAIN_SETS_SQL.format(project=project, dataset=dataset, agent=agent)
    )
    return [[t for t in row["labels"] if t != ABSTAIN] for row in job.result()]


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    train_sets = fetch_train_sets(args.project, args.dataset, args.agent)
    pool = menu_rules.choose(
        [calls for calls in train_sets if calls], size=args.menu_size, count=args.menu_count
    )
    print(f"{len(train_sets)} training sessions -> {len(pool)} menus of {args.menu_size}")
    for menu in pool:
        print(f"       {', '.join(sorted(menu))}")

    tokens = asyncio.run(declaration_tokens())
    live = frozenset(tokens)
    protected = (FRAMEWORK_TOOLS | cfg.never_prune) & live
    known = frozenset({tool for calls in train_sets for tool in calls})
    print(f"       {len(live)} live tools, {sum(tokens.values())} declaration tokens")

    candidates = fetch_candidates(args.project, args.dataset, args.agent, args.session_prefix)
    if not candidates:
        print("\nno held-out sessions; has the model been trained?", file=sys.stderr)
        return 1
    sessions = [c.session for c in candidates]
    turns = sum(s.turns for s in sessions) / len(sessions)
    print(f"       {len(sessions)} held-out sessions, {turns:.2f} model calls each on average")

    # The corpus's own measured session length replaces the config's prior, because here it
    # is not a prior: these sessions have already finished and their turns are counted.
    costs = args.sweep or [args.miss_cost or cfg.miss_cost_tokens]
    priced_cfgs = [
        replace(cfg, miss_cost_tokens=cost, expected_session_turns=turns) for cost in costs
    ]
    fixed = [Arm("full list"), Arm("always prune")]
    priced = [Arm(f"candidate @{cost}") for cost in costs]
    oracles = [Arm("best menu"), Arm("true set")]

    for candidate in candidates:
        session, ranked = candidate.session, candidate.ranked
        menu = menu_always(ranked, live, known, cfg, pool)
        score(fixed[0], session, live, live, tokens)
        score(fixed[1], session, menu, live, tokens)
        for arm, priced_cfg in zip(priced, priced_cfgs, strict=True):
            score(arm, session, menu_router(ranked, menu, live, tokens, priced_cfg), live, tokens)
        score(oracles[0], session, menu_best(session, pool, live, protected), live, tokens)
        score(oracles[1], session, menu_true(session, live, protected), live, tokens)

    report([*fixed, *priced, *oracles], sessions, sum(1 for s in sessions if not s.tools))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--session-prefix",
        default="",
        help="score only held-out sessions whose id starts with this. The menu pool is "
        "chosen from the whole training half either way, because that is what publishing "
        "does. Use `corpus` to compare against score_sets.py, which defaults to it.",
    )
    parser.add_argument("--menu-size", type=int, default=7)
    parser.add_argument("--menu-count", type=int, default=8)
    parser.add_argument(
        "--miss-cost",
        type=int,
        default=None,
        help="billed tokens a missing tool costs. Defaults to the shipped miss_cost_tokens.",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=None,
        metavar="TOKENS",
        help="score one candidate arm per miss price, to see the recall/saving frontier",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
