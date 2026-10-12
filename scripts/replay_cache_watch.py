#!/usr/bin/env python
"""E4 — make the safety net fire, and find out what the cool-off is worth.

`cache_watch` demotes an agent when pruning is measurably costing it money. It has
demoted once for real, in the false positive of §18.23, and it has never been calibrated
against a deliberate loss because nobody ran one on purpose. ROADMAP's E4 asked for a
losing agent and a live sweep of `cooloff_calls`.

It does not need one. Twelve A/B windows are already logged, one of them (`ab-20260906`,
§18.22) is a real 17.6% loss, and every model call in all of them carries its prompt
size, its cached share, its position in the session and the arm it belonged to. Streaming
those calls back through the shipped class in timestamp order is the same experiment
without the second bill, and it is reproducible, which a live run is not. §18.22's table
was produced this way by hand; this is that harness, committed.

What a sweep of `cooloff_calls` can and cannot move
---------------------------------------------------
Not detection. The first demotion happens when the evidence arrives, and the cool-off has
not started yet. Every setting detects at the same call.

What it moves is what happens next: how long the agent runs unpruned before it is allowed
to try again, therefore how many demote and re-promote cycles fit in a run, therefore how
much of the traffic is spent on retries. So the three questions this answers are how fast
it detects, how much it flaps, and what a demotion costs when it was wrong.

Pricing a demotion
------------------
A replay cannot change what was billed, so the cost of a demotion is a counterfactual and
is stated as one. While an agent is demoted its routed sessions would have run unpruned,
so each routed call in that period would have billed the unpruned rate instead of the
pruned one. The difference, summed over those calls, is what the demotion cost. It is
positive when pruning was winning, which is exactly when a demotion is a mistake.

Cost is one BigQuery read. Everything after it is local. Authentication is ADC only.

Usage:
    uv run python scripts/replay_cache_watch.py
    uv run python scripts/replay_cache_watch.py --cooloffs 100,500,2000
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery

from adk_smart_router.cache_watch import CacheWatch

DEFAULT_AGENT = "support"

# Kept identical to report_ab.py and replay_promotion.py: a session is routed only if the
# router acted on it. Everything else it saw is an unpruned call the watch counts for free.
ROUTED_SOURCES = ("fresh", "reused")

# The same weight the gate is defined in (DESIGN.md 10), and the same one CacheWatch uses.
CACHED_TOKEN_WEIGHT = 0.10

# run_ab.py names sessions `{prefix}{index:05d}`, so a window is everything before a
# trailing five-digit counter. replay_promotion.py says why this is a `raise` and not a
# fallback: a replay whose failure mode is a smaller, tidier table is the wrong shape.
WINDOW = re.compile(r"^(.*?)[0-9]{5}$")

COOLOFFS = (100, 250, 500, 1_000, 2_000)

# CacheWatch's own default, repeated here only so the report can say which windows never
# gathered enough evidence to be judged. Not passed in: the sweep runs the shipped values.
MIN_SAMPLES = 50


@dataclass(frozen=True)
class Call:
    """One model response, as the watch would have seen it."""

    session_id: str
    window: str
    arm: str  # routed, holdout, or neither, exactly as report_ab.py labels them
    first_call: bool
    prompt_tokens: int
    cached_tokens: int
    seconds: float  # since the first call of its window

    @property
    def pruned(self) -> bool:
        """What the watch sees. Only a routed session had tools taken off it."""
        return self.arm == "routed"

    @property
    def billed(self) -> float:
        uncached = max(self.prompt_tokens - self.cached_tokens, 0)
        return uncached + CACHED_TOKEN_WEIGHT * self.cached_tokens


@dataclass(frozen=True)
class Outcome:
    """One window replayed at one setting."""

    window: str
    calls: int
    truth: float | None  # measured G1: negative means pruning was winning
    warm_truth: float | None  # the same, over the warm calls the watch counts
    warm_routed: int  # warm routed calls, against a min_samples of 50
    detected_at: int | None  # call index of the first demotion
    detected_seconds: float | None
    demotions: int
    repromotions: int
    demoted_calls: int  # routed calls that happened while the agent was demoted
    cost: float | None  # billed tokens the demotions cost, negative if they saved money

    @property
    def right(self) -> bool | None:
        """Whether demoting was the correct call, or None if there is nothing to judge by."""
        if self.detected_at is None or self.truth is None:
            return None
        return self.truth > 0


CALLS_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
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
calls AS (
  SELECT
    session_id,
    timestamp,
    COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
    COALESCE(usage_cached_tokens, 0) AS cached_tokens,
    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) = 1 AS first_call
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND usage_prompt_tokens IS NOT NULL
)
SELECT
  c.session_id,
  c.timestamp,
  c.prompt_tokens,
  c.cached_tokens,
  c.first_call,
  a.arm
FROM calls c
JOIN arms a USING (session_id)
ORDER BY c.timestamp
"""


def fetch(project: str, dataset: str, agent: str, prefix: str) -> list[Call]:
    """Every priced model call that carries an arm, oldest first.

    The join is inner on purpose here, unlike `replay_promotion.py`'s. There the missing
    side was a whole window the policy needed to be told about; here a call with no
    provenance is a call the live watch would not have been able to label either, so
    dropping it reproduces the process rather than hiding from it.
    """
    sql = CALLS_SQL.format(project=project, dataset=dataset)
    job = bigquery.Client(project=project).query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
                bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES)),
            ]
        ),
    )
    rows = list(job.result())
    starts: dict[str, float] = {}
    calls = []
    for row in rows:
        match = WINDOW.match(row["session_id"])
        if not match:
            raise SystemExit(
                f"session id {row['session_id']!r} does not end in a five-digit counter, "
                "so it cannot be grouped into a window. Fix the pattern rather than the data."
            )
        window = match.group(1)
        stamp = row["timestamp"].timestamp()
        starts.setdefault(window, stamp)
        calls.append(
            Call(
                session_id=row["session_id"],
                window=window,
                arm=str(row["arm"]),
                first_call=bool(row["first_call"]),
                prompt_tokens=int(row["prompt_tokens"]),
                cached_tokens=int(row["cached_tokens"]),
                seconds=stamp - starts[window],
            )
        )
    return calls


def per_turn(calls: Sequence[Call], arm: str, warm_only: bool = False) -> float | None:
    """Billed-equivalent input tokens per turn for one arm, or None if it did not run.

    This is G1's own arithmetic (`report_ab.py`, gate G1 in DESIGN.md 10) with a turn read
    as a model call, which is what a turn is. `warm_only` drops the call that opens each
    session, which is the split the watch itself keeps its counters on.
    """
    rows = [c for c in calls if c.arm == arm and not (warm_only and c.first_call)]
    return sum(c.billed for c in rows) / len(rows) if rows else None


def warm_truth(calls: Sequence[Call]) -> float | None:
    """The same comparison over warm calls only, which is the one the watch is judging.

    Worth printing next to `truth` because the two can disagree in sign, and when they do
    the disagreement is the finding rather than a bug in either. A window whose arms were
    warmed by different amounts of traffic prices its first calls very differently from its
    later ones, and DESIGN.md 18.23 is that window.
    """
    routed = per_turn(calls, "routed", warm_only=True)
    holdout = per_turn(calls, "holdout", warm_only=True)
    if routed is None or not holdout:
        return None
    return (routed - holdout) / holdout


def truth(calls: Sequence[Call]) -> float | None:
    """What pruning actually did to this window, as G1 would score it. None if unscoreable.

    Routed against holdout, and nothing else. The watch is allowed to learn from any
    unpruned call it sees, including the declined sessions of an agent that never enrolled
    them, because at serving time that is all the evidence there is. A verdict on the watch
    is a different question and needs the real control arm: the declined sessions are the
    ones the router refused, so they are a biased sample of the traffic by construction,
    and pooling them into the baseline moves the number by tens of points. A window with no
    holdout has no baseline here, and saying so is better than substituting one.
    """
    routed, holdout = per_turn(calls, "routed"), per_turn(calls, "holdout")
    if routed is None or not holdout:
        return None
    return (routed - holdout) / holdout


def replay(window: str, calls: Sequence[Call], cooloff: int, agent: str) -> Outcome:
    """Stream one window through a fresh watch and record what it did.

    The shipped class, not a model of it. Anything reimplemented here would be a second
    opinion about the code rather than a measurement of it, and the whole value of a
    replay is that the thing under test is the thing that ships.
    """
    watch = CacheWatch(cooloff_calls=cooloff)
    detected_at: int | None = None
    detected_seconds: float | None = None
    repromotions = 0
    demoted_calls = 0
    was_demoted = False

    for index, call in enumerate(calls, start=1):
        watch.observe(
            agent,
            pruned=call.pruned,
            prompt_tokens=call.prompt_tokens,
            cached_tokens=call.cached_tokens,
            first_call=call.first_call,
        )
        now = watch.suppressed(agent)
        if now and not was_demoted and detected_at is None:
            detected_at, detected_seconds = index, call.seconds
        if was_demoted and not now:
            repromotions += 1
        if now and call.pruned:
            # A call that was pruned in the log but would not have been in a run where the
            # demotion was live. This is the exposure the demotion buys or wastes.
            demoted_calls += 1
        was_demoted = now

    return Outcome(
        window=window,
        calls=len(calls),
        truth=truth(calls),
        warm_truth=warm_truth(calls),
        warm_routed=sum(1 for c in calls if c.pruned and not c.first_call),
        detected_at=detected_at,
        detected_seconds=detected_seconds,
        demotions=watch.demotions(agent),
        repromotions=repromotions,
        demoted_calls=demoted_calls,
        cost=price(calls, demoted_calls),
    )


def price(calls: Sequence[Call], demoted_calls: int) -> float | None:
    """What those demoted calls would have cost unpruned, minus what they cost pruned.

    A counterfactual, and it is one whether or not the demotion was right: the same
    arithmetic reads as a saving when pruning was losing and as a loss when it was winning.
    The sign is carried by the window, not by the policy. Priced off the holdout for the
    same reason `truth` is scored off it.
    """
    routed, holdout = per_turn(calls, "routed"), per_turn(calls, "holdout")
    if routed is None or holdout is None or not demoted_calls:
        return None
    return demoted_calls * (holdout - routed)


def report(cooloff: int, outcomes: Sequence[Outcome]) -> None:
    print(f"\n--- cooloff {cooloff} calls " + "-" * 48)
    print(
        "| window | G1 says | warm calls say | warm routed | detects at |"
        " demotions | back | exposed | cost |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for o in outcomes:
        if o.truth is None:
            actual = "no holdout"
        else:
            actual = f"{'lost' if o.truth > 0 else 'fine'}, {o.truth:+.1%}"
        warm = "—" if o.warm_truth is None else f"{o.warm_truth:+.1%}"
        detected = (
            f"call {o.detected_at:,}, {o.detected_seconds:,.0f}s"
            if o.detected_at is not None
            else "never"
        )
        cost = f"{o.cost:+,.0f}" if o.cost is not None else "—"
        print(
            f"| {o.window} | {actual} | {warm} | {o.warm_routed:,} | {detected} |"
            f" {o.demotions} | {o.repromotions} | {o.demoted_calls:,} | {cost} |"
        )

    fired = [o for o in outcomes if o.detected_at is not None]
    judged = [o for o in fired if o.right is not None]
    right = [o for o in judged if o.right]
    unjudged = len(fired) - len(judged)
    print(
        f"\n  fired on {len(fired)} of {len(outcomes)} windows, "
        f"right on {len(right)}, wrong on {len(judged) - len(right)}"
        + (f", unscoreable on {unjudged}." if unjudged else ".")
    )
    missed = [o for o in outcomes if o.truth and o.truth > 0 and o.detected_at is None]
    if missed:
        worst = max(missed, key=lambda o: o.truth or 0.0)
        print(
            f"  it let {len(missed)} losing window(s) through, the worst of them "
            f"{worst.window} at {worst.truth:+.1%}."
        )
    starved = [o for o in outcomes if o.warm_routed < MIN_SAMPLES]
    if starved:
        print(
            f"  {len(starved)} window(s) never reached {MIN_SAMPLES} warm routed calls, "
            "so the watch had nothing to judge them on."
        )
    flapping = [o for o in fired if o.repromotions]
    print(
        f"  {len(flapping)} window(s) came back at least once; "
        f"the most cycles any window ran was {max((o.demotions for o in fired), default=0)}."
    )
    wrong_cost = sum(o.cost or 0.0 for o in fired if o.right is False)
    print(f"  wrong demotions cost {wrong_cost:+,.0f} billed tokens across the run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--prefix", default="ab-", help="session id prefix to sweep")
    parser.add_argument(
        "--cooloffs",
        default=",".join(str(c) for c in COOLOFFS),
        help="comma-separated cooloff_calls values to replay",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calls = fetch(args.project, args.dataset, args.agent, args.prefix)
    if not calls:
        raise SystemExit(f"no priced calls with provenance under {args.prefix!r}.")

    windows: dict[str, list[Call]] = {}
    for call in calls:
        windows.setdefault(call.window, []).append(call)

    print(f"{len(calls):,} priced calls with provenance across {len(windows)} windows")
    print(f"  agent {args.agent}, min_samples 50, max_repromotions 2, both at their shipped values")
    print("  exposed = routed calls that landed while the agent was demoted, which a live run")
    print("  would have sent whole. cost prices those calls at the gap between the two arms.")

    for cooloff in (int(c) for c in args.cooloffs.split(",")):
        report(
            cooloff,
            [
                replay(window, window_calls, cooloff, args.agent)
                for window, window_calls in sorted(windows.items())
            ],
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
