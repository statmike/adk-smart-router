#!/usr/bin/env python
"""E3 — move the tool surface under the model, and find the signal that says so.

The router ships with two ways of noticing it has gone stale, and neither one is a
detector. `cache_watch` reads the bill, so it fires after the money is spent.
`max_model_age_days` reads the clock, so it fires whether or not anything has changed.
What is missing is a signal that says "this model no longer describes this traffic"
before either of those, and nothing has been tuned against a real drift event because
nobody has driven one.

This project owns one. On 2026-09-04 the reference agent went from 25 tools to 100,
which is the same event an adopter causes when they ship a new integration. Everything
before that day is one surface and everything on it is another, so a model trained on
the old days and served the new day's traffic is a real drift case with a real answer
sitting behind it.

Three conditions, because "drift" is two different failures wearing one word:

    control          the old model on old traffic. What healthy looks like.
    surface drift    the old model on new traffic. Tools it has never heard of.
    traffic drift    the old model on the new day's traffic, restricted to sessions
                     whose tools all existed in the old surface. Same tools, different
                     questions. This is the hard one, because nothing about the request
                     looks unusual.

Three signals, all of them readable at serving time with no labels, which is the whole
point: a detector that needs to know what the session went on to call is a report, not
a detector.

    unseen share     the fraction of declared tools the model has no class for
    top-1 mass       how much probability the model puts on its best guess
    novelty          cosine distance from the opening to the nearest training opening

The learner and the corpus shape come from `simulate_censoring`, deliberately. Two
experiments that disagree about what the model is cannot be read against each other,
and the alternative is a second copy of the same sixty lines drifting away from the
first.

Cost is one BigQuery read. Everything after it is local. Authentication is ADC only.

Usage:
    uv run python scripts/simulate_drift.py
    uv run python scripts/simulate_drift.py --changeover 2026-09-04
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery
from simulate_censoring import (
    EVAL_PCT,
    MAX_TOOLS_KEPT,
    OPENING_KEY,
    Case,
    Model,
    rows,
)

DEFAULT_AGENT = "support"
DEFAULT_DIMS = 768

# The day the reference agent's surface went from 25 tools to 100 (19.15). Sessions before
# it could only call the original 25; sessions on it and after could call any of the 100.
CHANGEOVER = "2026-09-04"

# Topics to split the openings into for the constructed intent shift. Eight is a guess at
# the number of things a support agent gets asked about, and the experiment is not sensitive
# to it: any partition that withholds a coherent chunk of questions serves the same purpose.
CLUSTERS = 8


@dataclass(frozen=True)
class Era:
    """One side of the changeover."""

    name: str
    cases: tuple[Case, ...]

    @property
    def tools(self) -> frozenset[str]:
        return frozenset(tool for case in self.cases for tool in case.wanted)


@dataclass(frozen=True)
class Reading:
    """What one condition looked like: the outcome, and the three signals beside it."""

    name: str
    sessions: int
    recall: float
    unseen_share: float
    top1_mass: float
    novelty: float


CORPUS_SQL = """
WITH opening AS (
  SELECT
    session_id,
    ARRAY_AGG(invocation_id ORDER BY prompt_ts, invocation_id LIMIT 1)[OFFSET(0)] AS invocation_id,
    MIN(prompt_ts) AS opened_at
  FROM `{table}`
  WHERE ARRAY_LENGTH(v) = @dims
  GROUP BY session_id
),
first_message AS (
  SELECT
    f.session_id,
    f.v,
    o.opened_at,
    ABS(MOD(FARM_FINGERPRINT({key}), 100)) < {eval_pct} AS held_out
  FROM `{table}` f
  JOIN opening o USING (invocation_id)
),
attempted AS (
  -- The v2 label set (19.2): every tool the session tried, errors included.
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
  SELECT DISTINCT session_id
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
)
SELECT
  f.session_id,
  ANY_VALUE(f.v) AS v,
  ANY_VALUE(f.held_out) AS held_out,
  DATE(ANY_VALUE(f.opened_at)) AS day,
  ARRAY_AGG(DISTINCT a.tool_name IGNORE NULLS) AS wanted
FROM first_message f
LEFT JOIN attempted a USING (session_id)
WHERE f.session_id NOT IN (SELECT session_id FROM dropped)
  AND DATE(f.opened_at) <= @last_day
GROUP BY f.session_id
"""


def fetch(
    project: str, dataset: str, agent: str, dims: int, last_day: str
) -> list[tuple[Case, str]]:
    """Every session's opening vector, tool set, and the day it opened.

    `last_day` exists so a run that is still landing traffic cannot move the answer. The
    new-surface era has to be a closed window, or rerunning this next week compares against
    a different population and the numbers stop reproducing.
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
                bigquery.ScalarQueryParameter("last_day", "DATE", last_day),
            ]
        ),
    )
    return [
        (
            Case(
                session_id=row["session_id"],
                vector=list(row["v"]),
                wanted=frozenset(row["wanted"]),
                held_out=bool(row["held_out"]),
            ),
            str(row["day"]),
        )
        for row in job.result()
    ]


def split_eras(loaded: Sequence[tuple[Case, str]], changeover: str) -> tuple[Era, Era]:
    """Before the changeover and on or after it."""
    before = tuple(case for case, day in loaded if day < changeover)
    after = tuple(case for case, day in loaded if day >= changeover)
    return Era("old surface", before), Era("new surface", after)


# --- The three signals, all label-free ---------------------------------------------


def unseen_share(model: Model, declared: frozenset[str]) -> float:
    """The fraction of the tools on offer that the model has no class for.

    The cheapest possible detector and the one that answers surface drift outright. It
    needs nothing but the request, so a plugin can compute it on the first call after a
    deployment and never has to wait for a bill.
    """
    if not declared:
        return 0.0
    return sum(1 for tool in declared if tool not in model.classes) / len(declared)


def top1_mass(model: Model, cases: Sequence[Case]) -> float:
    """Mean probability on the model's best guess.

    A model whose traffic has moved out from under it does not usually start guessing
    wrong with conviction. It goes flat first, because the opening resembles nothing it
    was fitted on and the softmax spreads.
    """
    if not cases:
        return 0.0
    return statistics.fmean(max(probs.values()) for probs in model.probabilities(cases))


def novelty(train: Sequence[Case], cases: Sequence[Case]) -> float:
    """Mean cosine distance from each opening to the nearest opening the model trained on.

    The signal for traffic drift, where the surface has not changed and the model still has
    a class for everything on offer. It is the only one of the three that can see a shift
    in what people are asking about, and it is also the only one that costs anything to
    compute, since it wants the training openings kept around beside the model.
    """
    if not train or not cases:
        return 0.0
    fitted = _unit(np.asarray([c.vector for c in train], dtype=np.float32))
    probe = _unit(np.asarray([c.vector for c in cases], dtype=np.float32))
    nearest = (probe @ fitted.T).max(axis=1)
    return float(np.mean(1.0 - nearest))


def cluster(cases: Sequence[Case], k: int, rounds: int = 20) -> list[int]:
    """k-means over the openings, returning one cluster index per case.

    Written out rather than imported because scikit-learn is not a dependency of this
    project and one experiment is not a reason to make it one. Seeding is the first `k`
    cases in corpus order, which is deterministic and good enough: nothing downstream
    depends on the clustering being optimal, only on it being a partition by topic.
    """
    points = _unit(np.asarray([c.vector for c in cases], dtype=np.float32))
    centres = points[:k].copy()
    assigned = np.zeros(len(points), dtype=int)
    for _ in range(rounds):
        assigned = np.argmax(points @ centres.T, axis=1)
        for i in range(k):
            members = points[assigned == i]
            if len(members):
                centres[i] = members.mean(axis=0)
        centres = _unit(centres)
    return [int(i) for i in assigned]


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def recall_of(model: Model, cases: Sequence[Case], width: int) -> float:
    """Share of sessions whose whole tool set survives the menu.

    All or nothing per session, the same way `simulate_censoring` scores: one missing tool
    costs the turn no matter how many of the others were right. Sessions that wanted
    nothing are left out, because every menu covers the empty set.
    """
    wanting = [case for case in cases if case.wanted]
    if not wanting:
        return 0.0
    menus = model.menus(wanting, width)
    return sum(
        1 for case, menu in zip(wanting, menus, strict=True) if case.wanted <= set(menu)
    ) / len(wanting)


def read(
    name: str, model: Model, train: Sequence[Case], cases: Sequence[Case], declared: frozenset[str]
) -> Reading:
    return Reading(
        name=name,
        sessions=len(cases),
        recall=recall_of(model, cases, MAX_TOOLS_KEPT),
        unseen_share=unseen_share(model, declared),
        top1_mass=top1_mass(model, cases),
        novelty=novelty(train, cases),
    )


# How far a signal has to move before this script calls it a move rather than noise. Not a
# tuned threshold and not proposed as one: it is here so the table has a verdict column
# instead of leaving the reader to eyeball four decimal places.
MOVED = 0.02


def report(pairs: Sequence[tuple[Reading, Sequence[Reading]]]) -> None:
    print("\n| condition | sessions | recall | unseen tools | top-1 mass | novelty |")
    print("| --- | --- | --- | --- | --- | --- |")
    for control, drifted in pairs:
        for r in (control, *drifted):
            print(
                f"| {r.name} | {r.sessions:,} | {r.recall:.4f} | {r.unseen_share:.1%} |"
                f" {r.top1_mass:.4f} | {r.novelty:.4f} |"
            )

    print("\n--- What a detector would have seen ---------------------------------")
    for control, drifted in pairs:
        for r in drifted:
            change = r.recall - control.recall
            print(f"\n  {r.name}: recall {r.recall:.4f}, {change:+.4f} against its control.")
            for label, moved, base in (
                ("unseen tools", r.unseen_share, control.unseen_share),
                ("top-1 mass", r.top1_mass, control.top1_mass),
                ("novelty", r.novelty, control.novelty),
            ):
                shift = moved - base
                verdict = "moves" if abs(shift) > MOVED else "flat"
                print(f"    {label:<14} {moved:.4f} against {base:.4f}, {shift:+.4f}  {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--changeover", default=CHANGEOVER, help="first day of the new surface")
    parser.add_argument(
        "--clusters", type=int, default=CLUSTERS, help="topics to split the openings into"
    )
    parser.add_argument(
        "--last-day",
        default=CHANGEOVER,
        help="last day to read, so a run still landing traffic cannot move the answer",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = fetch(args.project, args.dataset, args.agent, args.dims, args.last_day)
    old, new = split_eras(loaded, args.changeover)
    if not old.cases or not new.cases:
        raise SystemExit(
            f"one side of {args.changeover} is empty: "
            f"{len(old.cases)} before, {len(new.cases)} on or after. Nothing to compare."
        )

    train = [case for case in old.cases if not case.held_out]
    vectors, labels = rows(train, {case.session_id: case.wanted for case in train})
    model = Model(vectors, labels)

    # The new surface is everything either era ever called. An adopter's plugin sees this
    # list on every request, which is why `unseen_share` is computable at serving time.
    old_surface, new_surface = old.tools, old.tools | new.tools

    print(f"{len(loaded):,} sessions read, changeover {args.changeover}")
    print(f"  old surface: {len(old.cases):,} sessions, {len(old_surface)} tools")
    print(f"  new surface: {len(new.cases):,} sessions, {len(new_surface)} tools")
    print(f"  model trained on {len(train):,} old-surface sessions, {len(model.classes)} classes")

    old_exam = [case for case in old.cases if case.held_out]
    new_exam = list(new.cases)
    # Traffic drift on its own: the new day's sessions that only wanted tools the old
    # surface already had. Nothing about these requests is unfamiliar except the questions.
    familiar = [case for case in new_exam if case.wanted and case.wanted <= old_surface]

    surface = (
        read("control, old model on old traffic", model, train, old_exam, old_surface),
        [
            read("surface drift, new tools on offer", model, train, new_exam, new_surface),
            read("traffic drift, new day and old tools", model, train, familiar, old_surface),
        ],
    )

    # A constructed intent shift, because the corpus does not contain a natural one. Cluster
    # the old surface's openings and train a second model with the largest cluster withheld,
    # then serve it that cluster. Same tool surface, same period, questions the model has
    # never been fitted on. This is the condition `novelty` exists for, and without building
    # it the signal would go untested: the new day's familiar sessions ask the old day's
    # questions, so nothing in the real data moves it.
    assignments = cluster(train, args.clusters)
    held = max(set(assignments), key=assignments.count)
    partial = [case for case, i in zip(train, assignments, strict=True) if i != held]
    withheld = [case for case, i in zip(train, assignments, strict=True) if i == held]
    partial_vectors, partial_labels = rows(
        partial, {case.session_id: case.wanted for case in partial}
    )
    narrow = Model(partial_vectors, partial_labels)
    print(
        f"  intent shift: {args.clusters} clusters, largest holds {len(withheld):,} of "
        f"{len(train):,}; the second model trains on {len(partial):,} and has "
        f"{len(narrow.classes)} classes"
    )

    intent = (
        read("intent control, familiar openings", narrow, partial, old_exam, old_surface),
        [read("intent drift, a whole cluster unseen", narrow, partial, withheld, old_surface)],
    )

    report([surface, intent])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
