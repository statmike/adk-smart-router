#!/usr/bin/env python
"""E5 — does the router poison its own training data, and does the fix cost more than the bug?

The worry is a feedback loop. The model only sees labels from tools it chose to keep, so a tool
it stops predicting stops appearing in training, and the omission compounds until the tool is
gone. That is the case for ε-greedy exploration, and it has sat parked since the design review
because nobody measured whether the loop actually runs.

It does not run today, and the reason is a filter rather than an accident. `train_router.sql`
drops every routed session from the corpus (the ratchet, 11.4): a pruned session could only call
the tools it was left with, so its label set is censored by the router's own decision. Only the
holdout arm's uncensored sessions train the next model. So the real question is not "does
censoring compound" in the abstract. It is which of two failure modes we would rather have:

    ratchet on   uncensored labels, and a training corpus that shrinks as the rollout ramps.
                 At route_fraction 0.9 nine tenths of traffic teaches nothing. Starvation.
    ratchet off  every session teaches, and the routed ones teach the model to agree with
                 itself. Poisoning.

ε-greedy is the third option: ratchet off, but every menu has an ε chance of carrying one extra
tool picked uniformly at random, so a tool the model has given up on can still be observed. Note
that ε here is exploration *inside* the menu, not a chance of skipping the prune. Skipping the
prune with probability ε is arithmetically the same as lowering `route_fraction`, which is a knob
we already have, and it pays the full prompt every time it fires. One extra tool costs one tool's
worth of prompt.

How the simulation works
------------------------
The corpus is the real one: opening embeddings and per-session tool sets from
`router_features_support`, split by the shipped opening-text hash. The held-out quarter is
permanent and never trains anything, so every generation is scored against the same fixed exam.

Generation 0 trains on the training pool uncensored, which is the honest starting state: before
the router shipped, nothing was pruned. Each later generation serves the pool with the previous
generation's model, collects whatever labels that serving produced, and trains on those. The
corpus is replaced rather than accumulated, because `train_router.sql` reads a rolling `@since`
window and the compounding worry is about what a window contains, not about the whole archive.

Two deliberate infidelities, both in the pessimistic direction. The learner is a local xgboost
softmax rather than BQML's boosted tree, same family and different implementation. And serving
takes the model's top `max_tools_kept` rather than picking from the menu catalogue; a shared
catalogue keeps a tool alive whenever any popular menu carries it, so top-k censors harder than
the real server does. If the loop does not run here it does not run there.

Cost is one BigQuery read. Everything after it is local. Authentication is ADC only.

Usage:
    uv run python scripts/simulate_censoring.py
    uv run python scripts/simulate_censoring.py --generations 8 --route-fraction 0.5
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery

DEFAULT_AGENT = "support"
DEFAULT_DIMS = 768

# The split, copied from train_router.sql the same way check_router.py and price_prefetch.py
# copy it. Hashed on the opening text rather than the session id, so two sessions that begin
# with the same question land on the same side.
EVAL_PCT = 25
OPENING_KEY = r"IFNULL(TRIM(REGEXP_REPLACE(LOWER(f.prompt_text), r'\s+', ' ')), f.session_id)"

ABSTAIN = "ABSTAIN"
MAX_TOOLS_KEPT = 11  # config.RouterConfig.max_tools_kept
ROUTE_FRACTION = 0.9  # a fully ramped rollout, which is where both failure modes are worst
GENERATIONS = 5
SEED = 20260906

# Matched to train_router.sql's CREATE MODEL options, so the local surrogate is the same shape
# of model rather than an arbitrary one. `max_tree_depth = 6` is set there explicitly; 20 rounds
# is BQML's default `max_iterations`, which the trainer takes.
TREES = 20
DEPTH = 6
LEARNING_RATE = 0.3

# `auto_class_weights = TRUE` in the trainer, reproduced here because it is not a detail. Class
# weighting is what keeps a starved tool visible to the loss, so a simulation of starvation that
# left it out would be measuring a model we do not run.
BALANCE_CLASSES = True

# `min_class_rows` from train_router.sql, the per-class floor. A class with fewer rows than this
# has its sessions dropped from training, and 11.4 calls it the first ratchet: a tool the router
# stops predicting loses rows, and once it falls under the floor it loses the rest at once. This
# is the amplifier the censoring worry is really about, so it is modelled rather than assumed
# away.
MIN_CLASS_ROWS = 10

# The four things a deployment can do with a generation of routed traffic. `ratchet` is what
# train_router.sql does today; the rest are what it would do with the filter removed and varying
# amounts of exploration.
# Names are comma-free because --arms splits on commas.
ARMS = {
    "ratchet (shipped)": (True, 0.0),
    "censored": (False, 0.0),
    "explore 0.01": (False, 0.01),
    "explore 0.05": (False, 0.05),
}


@dataclass(frozen=True)
class Case:
    """One session: what it opened with, and every tool it went on to try."""

    session_id: str
    vector: list[float]
    wanted: frozenset[str]
    held_out: bool


@dataclass
class Generation:
    """What one arm looked like after one round of train, serve, and re-train."""

    index: int
    training_rows: int
    classes: int
    recall: float
    proposed: int
    forgotten: int


CORPUS_SQL = """
WITH opening AS (
  SELECT
    session_id,
    ARRAY_AGG(invocation_id ORDER BY prompt_ts, invocation_id LIMIT 1)[OFFSET(0)] AS invocation_id
  FROM `{table}`
  WHERE ARRAY_LENGTH(v) = @dims
  GROUP BY session_id
),
first_message AS (
  SELECT
    f.session_id,
    f.v,
    ABS(MOD(FARM_FINGERPRINT({key}), 100)) < {eval_pct} AS held_out
  FROM `{table}` f
  JOIN opening o USING (invocation_id)
),
attempted AS (
  -- The v2 label set (19.2): every tool the session tried, errors included, because a tool
  -- that errored still had to be declared for the call to happen.
  SELECT session_id, tool_name
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent AND status = 'OK'
    AND tool_name != 'transfer_to_agent'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  UNION DISTINCT
  SELECT session_id, tool_name
  FROM `{project}.{dataset}.v_tool_error`
  WHERE agent = @agent AND tool_name != 'transfer_to_agent'
),
dropped AS (
  -- Cache hits only. The routed-session filter is deliberately NOT applied here: this
  -- simulation needs the uncensored truth for every session so it can censor them itself,
  -- one arm at a time.
  SELECT DISTINCT session_id
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
)
SELECT
  f.session_id,
  ANY_VALUE(f.v) AS v,
  ANY_VALUE(f.held_out) AS held_out,
  ARRAY_AGG(DISTINCT a.tool_name IGNORE NULLS) AS wanted
FROM first_message f
LEFT JOIN attempted a USING (session_id)
WHERE f.session_id NOT IN (SELECT session_id FROM dropped)
GROUP BY f.session_id
"""


def fetch_corpus(project: str, dataset: str, agent: str, dims: int) -> list[Case]:
    """Every session's opening vector and uncensored tool set.

    Uncensored is the point. The live trainer throws routed sessions away because it cannot
    tell what they would have called; here we know, because we are the ones doing the pruning.
    """
    sql = CORPUS_SQL.format(
        table=f"{project}.{dataset}.router_features_{agent}",
        project=project,
        dataset=dataset,
        key=OPENING_KEY,
        eval_pct=EVAL_PCT,
    )
    job = bigquery.Client(project=project).query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("dims", "INT64", dims),
            ]
        ),
    )
    return [
        Case(
            session_id=row["session_id"],
            vector=list(row["v"]),
            wanted=frozenset(row["wanted"]),
            held_out=bool(row["held_out"]),
        )
        for row in job.result()
    ]


def rows(
    cases: Sequence[Case], observed: dict[str, frozenset[str]]
) -> tuple[list[list[float]], list[str]]:
    """One row per (session, tool it wanted), which is the shape train_router.sql builds.

    The vector repeats across a session's tools on purpose: a softmax fitted on the repetition
    learns a distribution over the whole set rather than a single winner (19.3). A session that
    wanted nothing is one ABSTAIN row and is kept, because those are the sessions where every
    removal is free.
    """
    vectors, labels = [], []
    for case in cases:
        tools = observed[case.session_id] or frozenset({ABSTAIN})
        for tool in sorted(tools):
            vectors.append(case.vector)
            labels.append(tool)
    return floor_rare(vectors, labels)


def floor_rare(
    vectors: Sequence[Sequence[float]], labels: Sequence[str]
) -> tuple[list[list[float]], list[str]]:
    """Drop rows whose class has fewer than `MIN_CLASS_ROWS` of them.

    train_router.sql drops the whole session, not the row; here a session's rows are already
    independent of each other, so dropping the row is the same edit. ABSTAIN is never floored,
    because a corpus can legitimately be almost all abstentions and flooring it away would leave
    a model that cannot say "keep nothing".
    """
    counts = Counter(labels)
    kept = [
        (list(vector), label)
        for vector, label in zip(vectors, labels, strict=True)
        if label == ABSTAIN or counts[label] >= MIN_CLASS_ROWS
    ]
    return [vector for vector, _ in kept], [label for _, label in kept]


def balanced(labels: Sequence[str]) -> list[float]:
    """BQML's `auto_class_weights`: every class gets the same total weight.

    A rare tool with ten rows and a common one with a thousand pull on the loss equally, which
    is the only reason a starved class has a chance of climbing back. Without it the simulation
    would show a collapse that the shipped trainer's own options prevent.
    """
    counts = Counter(labels)
    total = len(labels)
    return [total / (len(counts) * counts[label]) for label in labels]


class Model:
    """A generation's router: an xgboost softmax over whatever labels its corpus contained.

    A tool absent from the corpus is absent from the model, and asking for it back is not
    possible. That is the whole mechanism under test, so it is left as a real consequence
    rather than papered over with a full class list.
    """

    def __init__(self, vectors: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
        self.classes = sorted(set(labels))
        index = {name: i for i, name in enumerate(self.classes)}
        # The native booster rather than the sklearn wrapper: it is what `model_store` loads at
        # serving time, and it does not drag scikit-learn into a project that does not use it.
        self.booster = (
            xgb.train(
                {
                    "objective": "multi:softprob",
                    "num_class": len(self.classes),
                    "max_depth": DEPTH,
                    "eta": LEARNING_RATE,
                    "verbosity": 0,
                },
                xgb.DMatrix(
                    np.asarray(vectors, dtype=np.float32),
                    label=[index[name] for name in labels],
                    weight=balanced(labels) if BALANCE_CLASSES else None,
                ),
                num_boost_round=TREES,
            )
            if len(self.classes) > 1
            else None
        )

    def probabilities(self, cases: Sequence[Case]) -> list[dict[str, float]]:
        """One name-to-probability map per case, over whatever classes the model has.

        Exposed rather than kept inside `menus` because the drift experiment reads the shape
        of the distribution and not just its top few names: a model whose traffic has moved
        away from it goes flat before it goes wrong.
        """
        if not cases or self.booster is None:
            flat = len(self.classes) or 1
            return [dict.fromkeys(self.classes, 1.0 / flat) for _ in cases]
        rows = self.booster.predict(
            xgb.DMatrix(np.asarray([c.vector for c in cases], dtype=np.float32))
        )
        return [dict(zip(self.classes, (float(p) for p in row), strict=True)) for row in rows]

    def menus(self, cases: Sequence[Case], width: int) -> list[list[str]]:
        """The top `width` tools for each case, ABSTAIN dropped rather than counted.

        ABSTAIN means "this session wants nothing", which is not a tool the prompt can carry.
        Letting it occupy a slot would silently narrow the menu for exactly the sessions the
        model is least sure about.

        A model down to one class cannot rank anything, so it serves that class to everyone.
        That is a total collapse and the tables should show it as one rather than crash.
        """
        keep = [i for i, name in enumerate(self.classes) if name != ABSTAIN]
        if not cases or not keep:
            return [[] for _ in cases]
        if self.booster is None:
            return [[self.classes[keep[0]]] for _ in cases]
        probs = self.booster.predict(
            xgb.DMatrix(np.asarray([c.vector for c in cases], dtype=np.float32))
        )
        return [
            [self.classes[i] for i in sorted(keep, key=lambda i: -row[i])[:width]] for row in probs
        ]


def serve(
    model: Model,
    pool: Sequence[Case],
    route_fraction: float,
    epsilon: float,
    width: int,
    catalogue: Sequence[str],
    rng: random.Random,
) -> tuple[dict[str, frozenset[str]], set[str]]:
    """Run one generation of traffic and return what it recorded, plus which sessions were routed.

    A routed session can only call tools that were on its menu, so its recorded label set is the
    intersection. An unrouted session records the truth. This is the only place censoring
    happens; every arm differs only in what it does with the result.
    """
    observed: dict[str, frozenset[str]] = {}
    routed: set[str] = set()
    for case, menu in zip(pool, model.menus(pool, width), strict=True):
        if rng.random() >= route_fraction:
            observed[case.session_id] = case.wanted
            continue
        routed.add(case.session_id)
        kept = set(menu)
        if epsilon > 0 and rng.random() < epsilon:
            kept.add(rng.choice(catalogue))
        observed[case.session_id] = case.wanted & kept
    return observed, routed


def score(model: Model, exam: Sequence[Case], width: int) -> tuple[float, int]:
    """Recall against the permanent holdout, and how many distinct tools the model proposes.

    Recall is per session and all-or-nothing: a session is covered when every tool it wanted
    survives the menu, because one missing tool costs the turn regardless of how many were
    right. Sessions that wanted nothing are excluded rather than scored as free wins.
    """
    wanting = [case for case in exam if case.wanted]
    menus = model.menus(wanting, width)
    covered = sum(1 for case, menu in zip(wanting, menus, strict=True) if case.wanted <= set(menu))
    proposed = {tool for menu in menus for tool in menu}
    return (covered / len(wanting) if wanting else 0.0, len(proposed))


def run_arm(
    pool: Sequence[Case],
    exam: Sequence[Case],
    catalogue: Sequence[str],
    ratchet: bool,
    epsilon: float,
    route_fraction: float,
    width: int,
    generations: int,
) -> list[Generation]:
    """One arm's whole history, generation 0 through `generations`.

    Every arm starts from the same generation 0, trained on the uncensored pool, because that
    is the state the router actually shipped into: nothing had been pruned yet. The arms only
    diverge once the first generation of routed traffic exists to disagree about.
    """
    rng = random.Random(SEED)
    truth = {case.session_id: case.wanted for case in pool}
    vectors, labels = rows(pool, truth)
    history = []

    for index in range(generations + 1):
        model = Model(vectors, labels)
        recall, proposed = score(model, exam, width)
        history.append(
            Generation(
                index=index,
                training_rows=len(labels),
                classes=len(model.classes),
                recall=recall,
                proposed=proposed,
                forgotten=len(catalogue) - len([c for c in model.classes if c != ABSTAIN]),
            )
        )
        if index == generations:
            break

        observed, routed = serve(model, pool, route_fraction, epsilon, width, catalogue, rng)
        teaching = [c for c in pool if not (ratchet and c.session_id in routed)]
        if not teaching:
            break
        vectors, labels = rows(teaching, observed)

    return history


def report(name: str, history: Sequence[Generation]) -> None:
    print(f"\n**{name}**\n")
    print("| generation | training rows | classes | holdout recall | tools proposed | forgotten |")
    print("| --- | --- | --- | --- | --- | --- |")
    for gen in history:
        print(
            f"| {gen.index} | {gen.training_rows:,} | {gen.classes} |"
            f" {gen.recall:.4f} | {gen.proposed} | {gen.forgotten} |"
        )


def verdict(arms: dict[str, list[Generation]]) -> None:
    print("\n--- Verdict ---------------------------------------------------------")
    start = next(iter(arms.values()))[0]
    print(f"  Every arm starts at recall {start.recall:.4f} on {start.training_rows:,} rows.")
    print()
    for name, history in arms.items():
        last = history[-1]
        drift = last.recall - start.recall
        print(
            f"  {name:<22} recall {last.recall:.4f} ({drift:+.4f}),"
            f" {last.forgotten} tools forgotten, {last.training_rows:,} rows"
        )
    print()
    ranked = sorted(arms.items(), key=lambda item: -item[1][-1].recall)
    if len(ranked) < 2:
        print("  One arm, so there is nothing to compare it against. Read the drift above:")
        print("  a flat column means the loop did not run under these conditions.")
        return
    best, second = ranked[0], ranked[1]
    margin = best[1][-1].recall - second[1][-1].recall
    if margin < 0.005:
        print(f"  {best[0]} and {second[0]} finish within {margin:.4f} of each other, which is")
        print("  not a separation. On this corpus the arms are the same arm.")
    else:
        print(f"  {best[0]} finishes {margin:.4f} ahead of {second[0]}.")


def main() -> int:
    args = parse_args()
    corpus = fetch_corpus(args.project, args.dataset, args.agent, args.dims)
    pool = [case for case in corpus if not case.held_out]
    exam = [case for case in corpus if case.held_out]
    catalogue = sorted({tool for case in corpus for tool in case.wanted})

    print(f"{len(corpus):,} sessions: {len(pool):,} training pool, {len(exam):,} permanent holdout")
    print(f"{len(catalogue)} tools in the catalogue, menus of {args.width}")
    print(f"route_fraction {args.route_fraction}, {args.generations} generations after the first")

    wanted = set(args.arms.split(",")) if args.arms else set(ARMS)
    histories = {}
    for name, (ratchet, epsilon) in ARMS.items():
        if name not in wanted:
            continue
        histories[name] = run_arm(
            pool,
            exam,
            catalogue,
            ratchet=ratchet,
            epsilon=epsilon,
            route_fraction=args.route_fraction,
            width=args.width,
            generations=args.generations,
        )
        report(name, histories[name])

    verdict(histories)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--width", type=int, default=MAX_TOOLS_KEPT)
    parser.add_argument("--route-fraction", type=float, default=ROUTE_FRACTION)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument(
        "--arms",
        default="",
        help=(
            "comma-separated subset of "
            + ", ".join(ARMS)
            + ". Each arm costs one model fit per generation, so a sensitivity sweep wants few."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
