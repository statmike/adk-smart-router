#!/usr/bin/env python
"""E6 — would an automatic promoter have made the right calls?

Ramping the rollout is a human decision in v1. The operator reads G1 against the holdout arm
and turns `route_fraction` up. ROADMAP has wanted to automate that since the design review and
has always answered the same way: not until there is operational history to check a policy
against. There is now. Twelve A/B windows ran between 2026-09-01 and 2026-09-11, and every one
of them has a known outcome.

So this replays a candidate promoter over that history and reports what it would have done.
The question it is really asking is not "what is the right threshold" but **"can a window this
size tell a real win from noise at all?"**, because a promoter that acts on a point estimate
will ramp on a window that got lucky and demote on one that did not.

The policy
----------
One decision per window, from a confidence interval rather than a point estimate:

    ramp    the interval lies entirely below the G1 target. The win is real at this size.
    demote  the interval lies entirely above zero. Routing is costing money, not saving it.
    hold    the interval straddles either line. Not enough evidence to move.

The interval is bootstrapped **over sessions, not turns**. Turns inside one session are not
independent: the routing decision is made once and frozen for the conversation (DESIGN.md
11.4), so a session's turns all carry the same treatment and the same scenario. Resampling
turns would treat one session's six turns as six pieces of evidence and shrink the interval to
about a third of its honest width, which is exactly the error that would make a promoter
confident. Sessions are the unit of randomisation, so sessions are the unit of resampling.

G1 itself is `report_ab.py`'s definition, unchanged: billed-equivalent input tokens per turn,
`uncached + 0.10 * cached`, routed against holdout, arms read from the router's own provenance
rather than from a hash recomputed here.

What this can and cannot conclude
---------------------------------
It can say whether a policy would have moved, and how large a window has to be before it
moves. It cannot say whether moving would have been correct in the counterfactual, because
nobody ran the other branch. Where a window's true effect is known from elsewhere, the
comparison is in DESIGN.md rather than asserted here.

It also cannot see a sequential A/B. The policy reads the router's own provenance, and a
sequential control period runs with the router uninstalled, so it writes none. Those sessions
are counted under "no provenance" and left unscored rather than dropped, because a promoter
that silently ignores an experimental design is worse than one that says it cannot read it.

Cost is one BigQuery read over the A/B sessions. Authentication is ADC only.

Usage:
    uv run python scripts/replay_promotion.py
    uv run python scripts/replay_promotion.py --target -0.10 --confidence 0.90
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery

DEFAULT_AGENT = "support"

# report_ab.py's constants, repeated rather than imported: they are part of the definition of
# the gate, and a promoter that scored on a different definition than the operator reads would
# be answering a different question.
CACHED_TOKEN_WEIGHT = 0.10
ROUTED_SOURCES = ("fresh", "reused")
G1_TARGET = -0.10

BOOTSTRAP_DRAWS = 2000
CONFIDENCE = 0.90
SEED = 20260906

# The window sizes to ask "how big would this have had to be?" at. The largest is well past
# anything we have run, on purpose: the answer is allowed to be "bigger than you can afford".
HYPOTHETICAL_SESSIONS = (200, 400, 600, 1000, 2000, 5000, 10000, 20000)


@dataclass(frozen=True)
class Session:
    """One session's contribution to G1: how many turns it ran and what they billed."""

    arm: str
    turns: int
    billed: float


@dataclass(frozen=True)
class Window:
    """One A/B run, as the promoter would see it."""

    prefix: str
    sessions: tuple[Session, ...]

    def arm(self, name: str) -> list[Session]:
        """The scoreable sessions in one arm.

        A session with no turns billed nothing, so it cannot go in an average of billing. It
        is still counted, by `silent`, because a run that recorded a routing decision and then
        produced no turn failed at something and should not leave the denominator quietly.
        """
        return [s for s in self.sessions if s.arm == name and s.turns > 0]

    @property
    def silent(self) -> int:
        """Sessions the router decided on that then billed nothing."""
        return sum(1 for s in self.sessions if s.turns == 0)

    @property
    def unseen(self) -> int:
        """Sessions that billed turns and left the router no provenance to read them by.

        `ab-20260910b` is 300 of these. It is the control period of the sequential run in
        DESIGN 18.24, and the router was uninstalled for it on purpose, so there is nothing to
        read and no arm to assign. An inner join dropped the window silently; it is kept here
        and printed, because "this window is invisible to the policy" is a result about the
        policy rather than a gap in the data.
        """
        return sum(1 for s in self.sessions if s.arm == "unseen")


@dataclass(frozen=True)
class Verdict:
    """What the policy decided and what it decided from."""

    prefix: str
    routed: int
    holdout: int
    neither: int
    silent: int
    unseen: int
    effect: float
    low: float
    high: float
    action: str

    @property
    def two_armed(self) -> bool:
        return self.routed > 0 and self.holdout > 0

    @property
    def unreadable(self) -> bool:
        """The window billed turns and left no provenance, so the policy cannot see it at all.

        Distinct from a one-armed window, which the policy sees and declines to act on. This
        one it never sees, and the difference matters: a sequential A/B writes no provenance
        during its control period by design, so an automatic promoter is structurally blind to
        half of that design.
        """
        return self.routed + self.holdout + self.neither == 0 and self.unseen > 0

    @property
    def power(self) -> int:
        """The smaller arm, which is what the interval's width is really set by."""
        return min(self.routed, self.holdout)


SESSIONS_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, 'ab-')
    AND JSON_VALUE(record, '$.tool_name') IS NULL
),
arms AS (
  SELECT
    session_id,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
      ELSE 'neither'
    END AS arm
  FROM decisions
  GROUP BY session_id
),
turns AS (
  SELECT
    session_id,
    COUNT(*) AS turns,
    SUM(
      GREATEST(COALESCE(usage_prompt_tokens, 0) - COALESCE(usage_cached_tokens, 0), 0)
      + {weight} * COALESCE(usage_cached_tokens, 0)
    ) AS billed
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, 'ab-')
    AND usage_prompt_tokens IS NOT NULL
  GROUP BY session_id
)
SELECT
  session_id,
  REGEXP_EXTRACT(session_id, r'^(.*?)[0-9]{{5}}$') AS prefix,
  COALESCE(arms.arm, 'unseen') AS arm,
  COALESCE(turns.turns, 0) AS turns,
  COALESCE(turns.billed, 0.0) AS billed
FROM arms
FULL OUTER JOIN turns USING (session_id)
"""


def fetch_windows(project: str, dataset: str, agent: str) -> list[Window]:
    """Every A/B window on record, one row per session.

    A window is whatever precedes `run_ab.py`'s five-digit session counter. The first version
    of this matched `ab-<date>` instead and silently dropped `ab-v2-100tools`, which is the
    run every published number comes from. An id that does not parse now stops the script
    rather than disappearing from the denominator.
    """
    job = bigquery.Client(project=project).query(
        SESSIONS_SQL.format(project=project, dataset=dataset, weight=CACHED_TOKEN_WEIGHT),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES)),
            ]
        ),
    )
    grouped: dict[str, list[Session]] = {}
    for row in job.result():
        if row["prefix"] is None:
            raise SystemExit(f"{row['session_id']} does not end in a session counter.")
        grouped.setdefault(row["prefix"], []).append(
            Session(row["arm"], int(row["turns"]), float(row["billed"]))
        )
    return [Window(prefix, tuple(sessions)) for prefix, sessions in sorted(grouped.items())]


def per_turn(sessions: Sequence[Session]) -> float:
    """Billed tokens per turn over a set of sessions.

    Summed then divided, not averaged per session and then averaged again. Sessions run to
    different lengths and the router changes none of that, so weighting a two-turn session
    equally with a ten-turn one would measure the scenario mix (report_ab.py's docstring).
    """
    turns = sum(s.turns for s in sessions)
    return sum(s.billed for s in sessions) / turns if turns else 0.0


def effect(routed: Sequence[Session], holdout: Sequence[Session]) -> float:
    """G1 for one window. Negative is cheaper."""
    baseline = per_turn(holdout)
    return (per_turn(routed) - baseline) / baseline if baseline > 0 else 0.0


def interval(
    routed: Sequence[Session],
    holdout: Sequence[Session],
    draws: int,
    confidence: float,
    rng: random.Random,
) -> tuple[float, float]:
    """A bootstrap interval on G1, resampling whole sessions within each arm.

    Percentile bootstrap rather than a normal approximation: G1 is a ratio of two sums with a
    long-tailed numerator, and the sessions inside an arm are not identically distributed
    because scenarios differ in length. The percentile version makes no claim about the shape
    of the sampling distribution, which is the only honest option here.
    """
    if not routed or not holdout:
        return (float("-inf"), float("inf"))
    scores = sorted(
        effect(
            [routed[rng.randrange(len(routed))] for _ in routed],
            [holdout[rng.randrange(len(holdout))] for _ in holdout],
        )
        for _ in range(draws)
    )
    tail = (1.0 - confidence) / 2.0
    return (scores[int(tail * draws)], scores[min(int((1.0 - tail) * draws), draws - 1)])


def decide(low: float, high: float, target: float) -> str:
    """Ramp on evidence, demote on evidence, hold on anything else.

    Both bars are one-sided and neither is the other's complement. An interval below the
    target is a win worth ramping on; an interval above zero is a loss worth backing out of;
    everything between is a window that has not decided anything, including the common case of
    a real but unproven win.
    """
    if high < target:
        return "ramp"
    if low > 0.0:
        return "demote"
    return "hold"


def replay(
    windows: Sequence[Window], target: float, confidence: float, draws: int
) -> list[Verdict]:
    rng = random.Random(SEED)
    verdicts = []
    for window in windows:
        routed, holdout = window.arm("routed"), window.arm("holdout")
        low, high = interval(routed, holdout, draws, confidence, rng)
        verdicts.append(
            Verdict(
                prefix=window.prefix,
                routed=len(routed),
                holdout=len(holdout),
                neither=len(window.arm("neither")),
                silent=window.silent,
                unseen=window.unseen,
                effect=effect(routed, holdout),
                low=low,
                high=high,
                action=decide(low, high, target),
            )
        )
    return verdicts


def best_powered(windows: Sequence[Window]) -> Window | None:
    """The window the size question should be asked of.

    Not a pool of every window. Pooling would mix treatments: these runs used different
    artifacts at different settings and one of them was deliberately built to lose, so the
    pooled effect is an average of things that were never meant to be averaged, and a sample
    size derived from it would describe no real deployment. The best-powered two-armed window
    is a real deployment, and its effect is the one the project publishes.
    """
    two_armed = [w for w in windows if w.arm("routed") and w.arm("holdout")]
    if not two_armed:
        return None
    return max(two_armed, key=lambda w: min(len(w.arm("routed")), len(w.arm("holdout"))))


def size_to_decide(
    window: Window, target: float, confidence: float, draws: int, rng: random.Random
) -> int | None:
    """The smallest hypothetical window at which this window's effect would clear the bar.

    Resamples the window's own sessions up or down to each candidate size and asks the same
    policy the same question. The effect stays where it was measured; only the sample size
    moves, which is the whole point: it separates "the win is too small to see" from "the
    window was too small to see it".
    """
    routed, holdout = window.arm("routed"), window.arm("holdout")
    for size in HYPOTHETICAL_SESSIONS:
        half = size // 2
        low, high = interval(
            [routed[rng.randrange(len(routed))] for _ in range(half)],
            [holdout[rng.randrange(len(holdout))] for _ in range(half)],
            draws,
            confidence,
            rng,
        )
        if decide(low, high, target) == "ramp":
            return size
    return None


def report(verdicts: Sequence[Verdict], target: float, confidence: float) -> None:
    two_armed = [v for v in verdicts if v.two_armed]
    print(f"\n--- {len(verdicts)} windows, {confidence:.0%} interval, target {target:+.0%} ---\n")
    print(
        f"| window | routed | holdout | unrouted | no turns | no provenance | G1 |"
        f" {confidence:.0%} interval | policy |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for v in verdicts:
        if v.unreadable:
            band = "invisible to the policy"
        elif v.two_armed:
            band = f"{v.low:+.1%} to {v.high:+.1%}"
        else:
            band = "no second arm"
        effect_cell = f"{v.effect:+.1%}" if v.two_armed else "—"
        print(
            f"| {v.prefix} | {v.routed} | {v.holdout} | {v.neither} | {v.silent} | {v.unseen} |"
            f" {effect_cell} | {band} | {v.action} |"
        )

    actions = {
        name: sum(1 for v in two_armed if v.action == name) for name in ("ramp", "hold", "demote")
    }
    unreadable = [v for v in verdicts if v.unreadable]
    one_armed = len(verdicts) - len(two_armed) - len(unreadable)
    print(f"\n  Of {len(two_armed)} two-armed windows: ramp {actions['ramp']},", end="")
    print(f" hold {actions['hold']}, demote {actions['demote']}.")
    if one_armed:
        print(f"  {one_armed} more ran without a holdout and are unscoreable, not held.")
    if unreadable:
        sessions = sum(v.unseen for v in unreadable)
        print(f"  {len(unreadable)} left no provenance at all, over {sessions:,} sessions.")
        print("  A sequential A/B uninstalls the router for its control period, so it writes")
        print("  nothing to read, and an automatic promoter is blind to that whole design.")

    stalled = sum(v.silent for v in verdicts)
    if stalled:
        print(f"  {stalled} sessions across the run were routed and then billed no turn.")

    beat = sum(1 for v in two_armed if v.effect < target)
    print(f"  {beat} beat the target on the point estimate and {actions['ramp']} on the interval.")


def report_size(window: Window, size: int | None, target: float, confidence: float) -> None:
    routed, holdout = window.arm("routed"), window.arm("holdout")
    print("\n--- How big does a window have to be? ------------------------------")
    print(f"  Asked of {window.prefix}, the best-powered two-armed window:")
    print(f"  {len(routed):,} routed, {len(holdout):,} holdout, G1 {effect(routed, holdout):+.1%}.")
    print("  Its own sessions are resampled to each size, so the effect is held and only the")
    print("  sample moves.")
    print()
    if size is None:
        print(f"  At this effect size, no window up to {max(HYPOTHETICAL_SESSIONS):,} sessions")
        print(f"  clears {target:+.0%} on a {confidence:.0%} interval. An automatic promoter")
        print("  cannot act on a window an operator would be willing to run.")
    else:
        print(f"  {size:,} sessions, split evenly between the arms, is the smallest window at")
        print("  which the policy ramps.")


def main() -> int:
    args = parse_args()
    windows = fetch_windows(args.project, args.dataset, args.agent)
    if not windows:
        print("No A/B windows found. Nothing to replay.", file=sys.stderr)
        return 1

    verdicts = replay(windows, args.target, args.confidence, args.draws)
    report(verdicts, args.target, args.confidence)

    best = best_powered(windows)
    if best is None:
        print("\nNo window ran both arms, so there is no size question to ask.")
        return 0
    rng = random.Random(SEED)
    report_size(
        best,
        size_to_decide(best, args.target, args.confidence, args.draws, rng),
        args.target,
        args.confidence,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--target", type=float, default=G1_TARGET, help="G1, negative is cheaper")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
