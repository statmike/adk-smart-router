#!/usr/bin/env python
"""How long a prefix stays cached, measured instead of assumed.

`predict_saving.py` cannot price a routed agent without knowing how likely a session is to
open on a prefix that is still resident. That probability is the largest single term in the
prediction: DESIGN.md 19.17 measured a routed first turn caching 54.8% against a holdout's
88.6%, and that gap was worth more G1 than the prune itself.

The obvious place to measure it is the agent's own unrouted traffic, and that does not
work. An unrouted agent has exactly one prefix, so every request refreshes it and turn one
is warm essentially always. The reference agent's 1,700-session corpus reports 95.6%
residency over a two-hour run at roughly one request a second, which is consistent with a
two-second cache and with an hour-long one. Unrouted traffic gives a lower bound and
nothing more.

A *routed* run does answer it, because the router splits traffic across menus and so
produces prefixes that idle for seconds and prefixes that idle for minutes. This script
reads such a run, works out how long each session's prefix had been sitting unused when
the session opened, and reports how often it was still cached.

What comes back is not a cliff. Warmth falls off gradually and settles on a floor rather
than reaching zero, so the answer is a curve and not a window. But it is not *one* curve:
pooling every prefix into a single fit was wrong, and wrong in a way that flattered the
prediction. In the one routed run there is a 16.2k-token full list running 25 times a
minute and 24 menus of about 8k running once or twice a minute, so the short-gap end of a
pooled fit is almost entirely full-list observations and the long-gap end almost entirely
menu ones. The pooled curve then reads 0.886 at a two-second gap, which no menu ever
achieves: a menu measures 0.725 at the same gap. Seventeen points of the routed arm's
deficit are present at zero idle and no decay model can produce them.

So this splits the openings by prefix size and fits each band on its own:

    gap to the previous request     ~8k menu prefix      16.2k full-list prefix
    under 2 seconds                 0.73                 0.89
    about a minute                  0.62                 0.89
    ten minutes or more             0.49                 0.89

Read across, not down. The big prefix does not decay over the range anyone measured; the
menu starts lower and then decays onto a floor. Implicit caching is best-effort, so even a
prefix used a moment ago misses some of the time, and residency does not decay to nothing,
so a menu that runs at all runs mostly warm. What separates the two bands is not settled —
size and a shared system-instruction head are confounded in every run logged so far, which
is what `probe_residency.py` exists to separate. Until it has run, prefix size is the
variable that orders the observations, so prefix size is what the bands use.

    uv run python scripts/measure_residency.py --session-prefix ab-v2-100tools
    uv run python scripts/measure_residency.py --session-prefix ab-v2-100tools --out curve.json

Cost is one BigQuery read. No model calls. Authentication is ADC only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
)

# How many equal-count bins the curve is cut into. Twelve over a few hundred openings keeps
# roughly fifty samples a bin, which is enough for the bin to mean something and fine
# enough to show the shape.
BINS = 12

# The least a knot's gap may exceed the one before it. Where arrivals cluster, equal-count
# binning puts two bins a second apart and the noise between them reads as a cliff, which a
# caller interpolating across it would take seriously. Requiring each knot to sit at twice
# the idle time of the last keeps the curve no finer than the data can actually resolve.
MIN_GAP_SPREAD = 2.0

# How far apart two prefix sizes must sit before they are treated as two populations rather
# than as spread within one. A routed agent's menus all come out within a few hundred tokens
# of each other and its full list is roughly twice that, so the break is large and obvious
# where it exists; requiring half again as many tokens stops the split from chasing the
# 8,296-vs-7,849 spread inside the menu pool, which is real but is not a second regime.
BAND_RATIO = 1.5

# Neither band is worth fitting on a handful of openings, and a curve with twelve knots over
# twenty points is noise with a shape. Below this the run reports one pooled band and says so.
BAND_MIN_OPENINGS = 30

# Which prefix each session shipped is recoverable because `provenance.py` logs `kept_tools`
# on every routing decision. Records with a `tool_name` are cache serves rather than routing
# decisions, so they are filtered out.
_SQL = """
WITH shipped AS (
  SELECT
    e.session_id,
    ANY_VALUE(TO_JSON_STRING(JSON_QUERY(record, '$.kept_tools'))) AS prefix_key
  FROM `{project}.{dataset}.agent_events` e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
  GROUP BY e.session_id
),
turns AS (
  SELECT
    session_id,
    timestamp,
    COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
    COALESCE(usage_cached_tokens, 0) AS cached_tokens,
    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) AS ordinal
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND usage_prompt_tokens IS NOT NULL
)
SELECT shipped.prefix_key, turns.timestamp, turns.prompt_tokens, turns.cached_tokens, turns.ordinal
FROM turns JOIN shipped USING (session_id)
ORDER BY turns.timestamp
"""


@dataclass(frozen=True)
class Observation:
    """One session opening: how long its prefix had idled, and how much of it was cached.

    `share` is the fraction of the prompt billed at the cached rate, which is the quantity
    the billing model multiplies by and so the quantity worth measuring. It is close to
    all-or-nothing per request, 110 openings under 0.1 against 409 over 0.9 out of 575, but
    a hit is not always the whole prefix: implicit caching matches the longest common
    prefix, and two menus that differ early share less of one than two full lists do.
    Averaging the share carries that through where a hit rate would throw it away.

    `prefix_tokens` is how big the prefix was, estimated as the smallest prompt ever sent on
    it. Every request on a prefix carries the prefix plus at least one user message, and
    history only grows, so the minimum is the prefix plus the shortest opening message the
    run happened to contain. That over-states the prefix by ~100 tokens and is stable to
    within a token across the prefixes being compared, which is all the banding needs.
    """

    gap_s: float
    share: float
    prefix_tokens: int


def fetch_rows(project: str, dataset: str, agent: str, prefix: str) -> list[dict]:
    job = bigquery.Client(project=project).query(
        _SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            ]
        ),
    )
    return [dict(row) for row in job.result()]


def cached_share(row: dict) -> float:
    return int(row["cached_tokens"]) / max(1, int(row["prompt_tokens"]))


def observe(rows: Sequence[dict]) -> tuple[list[Observation], list[float]]:
    """Split the run into session openings, timed against their prefix, and later turns.

    Returns the openings that had a predecessor on the same prefix, and the cached share of
    every turn after the first. The second list is the *carry*: how much of what a session
    already sent is still there when it takes another turn.

    Each prefix is sized once, over every turn it ever carried and not only its openings, so
    a menu that opened few sessions is still sized from all the traffic that mentions it.
    """
    size: dict[str, int] = {}
    for row in rows:
        key = str(row["prefix_key"])
        size[key] = min(size.get(key, 1 << 30), int(row["prompt_tokens"]))

    last_seen: dict[str, float] = {}
    openings: list[Observation] = []
    carry: list[float] = []
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        key = str(row["prefix_key"])
        at = row["timestamp"].timestamp()
        if int(row["ordinal"]) == 1:
            if key in last_seen:
                openings.append(Observation(at - last_seen[key], cached_share(row), size[key]))
        else:
            carry.append(cached_share(row))
        last_seen[key] = at
    return openings, carry


def fit_curve(observations: Sequence[Observation], bins: int = BINS) -> list[tuple[float, float]]:
    """Cached share against gap, as knots a reader can check against the raw bins.

    Equal-count bins keep every point equally supported. Two passes then clean them up.
    Pooling adjacent violators forces the result to fall monotonically, because a longer
    idle cannot make a prefix more likely to be cached and any bin that says otherwise is
    sampling noise. Pooling bins that sit too close together in time stops the curve from
    claiming detail the binning cannot support.
    """
    if not observations:
        return []
    ordered = sorted(observations, key=lambda o: o.gap_s)
    size = max(1, len(ordered) // bins)
    pooled = [
        [statistics.median(o.gap_s for o in group), sum(o.share for o in group), len(group)]
        for group in (
            ordered[i : i + size] if i + 2 * size <= len(ordered) else ordered[i:]
            for i in range(0, len(ordered), size)
        )
        if group
    ]

    def collapse(violates) -> None:
        merged = True
        while merged:
            merged = False
            for i in range(len(pooled) - 1):
                left, right = pooled[i], pooled[i + 1]
                if violates(left, right):
                    pooled[i : i + 2] = [
                        [max(left[0], right[0]), left[1] + right[1], left[2] + right[2]]
                    ]
                    merged = True
                    break

    collapse(lambda left, right: left[1] / left[2] < right[1] / right[2])
    collapse(lambda left, right: right[0] < left[0] * MIN_GAP_SPREAD)
    return [(round(gap, 1), round(hits / count, 3)) for gap, hits, count in pooled]


@dataclass(frozen=True)
class Band:
    """One population of prefix sizes, and the curve fitted to that population alone."""

    min_tokens: int
    max_tokens: int
    openings: int
    mean_share: float
    curve: list[tuple[float, float]]

    @property
    def label(self) -> str:
        if self.min_tokens == self.max_tokens:
            return f"{self.min_tokens:,}-token prefixes"
        return f"{self.min_tokens:,}-{self.max_tokens:,}-token prefixes"


def band_boundary(observations: Sequence[Observation]) -> int | None:
    """The prefix size that splits these openings into two populations, or None.

    Cut at the widest *multiplicative* gap between two prefix sizes that anyone actually
    sent, because what matters is the ratio and not the absolute distance: 8k against 16k is
    a different regime, 7.8k against 8.3k is the same one measured twice. The boundary
    itself is the geometric mean of the two sizes either side of the cut, which leaves room
    for a caller whose idea of a prefix differs from this one by an opening message.

    Returning None is the normal answer for an agent with one prefix, or with a pool of
    menus all cut to the same width. One band is then the honest fit and the caller says so.
    """
    if not observations:
        return None
    sizes = sorted({o.prefix_tokens for o in observations})
    best: tuple[float, int] | None = None
    for low, high in zip(sizes, sizes[1:], strict=False):
        ratio = high / max(1, low)
        if ratio < BAND_RATIO:
            continue
        cut = round(math.sqrt(max(1, low) * high))
        below = sum(1 for o in observations if o.prefix_tokens < cut)
        if min(below, len(observations) - below) < BAND_MIN_OPENINGS:
            continue
        if best is None or ratio > best[0]:
            best = (ratio, cut)
    return best[1] if best else None


def fit_bands(observations: Sequence[Observation], bins: int = BINS) -> list[Band]:
    """One curve per prefix-size band, smallest band first.

    Splitting before fitting is the whole point. A pooled fit mixes prefixes whose warmth
    differs by seventeen points at the same idle time, and because the big prefix is also
    the busy one, the mix is not even constant along the curve: it is nearly all big prefix
    at the short-gap end and nearly all small at the long-gap end. That gives a pooled fit a
    slope it inherited from the traffic mix rather than from any decay.
    """
    boundary = band_boundary(observations)
    groups = (
        [list(observations)]
        if boundary is None
        else [
            [o for o in observations if o.prefix_tokens < boundary],
            [o for o in observations if o.prefix_tokens >= boundary],
        ]
    )
    return [
        Band(
            min_tokens=min(o.prefix_tokens for o in group),
            max_tokens=max(o.prefix_tokens for o in group),
            openings=len(group),
            mean_share=statistics.mean(o.share for o in group),
            curve=fit_curve(group, bins),
        )
        for group in groups
        if group
    ]


def report(observations: Sequence[Observation], carry: Sequence[float], prefixes: int) -> None:
    bands = fit_bands(observations)
    print(f"\n{len(observations)} session openings across {prefixes} distinct prefixes")
    if observations:
        mean = statistics.mean(o.share for o in observations)
        print(f"  cached share on arrival   {mean:.3f} mean")
    if carry:
        print(f"  later turns, cached share {statistics.mean(carry):.3f} mean")
        print(f"                            {statistics.median(carry):.3f} median")

    for band in bands:
        print(f"\nresidency curve, isotonic over equal-count bins: {band.label}")
        print(f"  {band.openings} openings, {band.mean_share:.3f} mean cached share")
        print(f"  {'gap (s)':>10}   {'cached share':>12}")
        for gap, share in band.curve:
            print(f"  {gap:>10,.1f}   {share:>12.3f}")

    if len(bands) == 1:
        print(
            "\n  One band: every prefix in this run is about the same size, so there is no"
            "\n  size contrast to fit and this curve covers all of them."
        )
    else:
        print(
            "\n  Read each band as a decay onto a floor rather than a window, and read the"
            "\n  bands against each other rather than pooling them. The gap between them at"
            "\n  the *shortest* idle is the part no decay model can explain, and it is the"
            "\n  part that costs a routed agent most."
        )


def main() -> int:
    args = parse_args()
    rows = fetch_rows(args.project, args.dataset, args.agent, args.session_prefix)
    if not rows:
        print(f"no routed turns under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    observations, carry = observe(rows)
    if not observations:
        print(
            f"{len(rows)} turns under {args.session_prefix!r}, but no session opened on a prefix"
            " another session had already used. That is unrouted traffic, or a run too short"
            " to reuse a menu, and it cannot measure residency.",
            file=sys.stderr,
        )
        return 1

    prefixes = len({str(row["prefix_key"]) for row in rows})
    report(observations, carry, prefixes)

    if args.out:
        bands = fit_bands(observations)
        # `curve` is the smallest band and not a pooled fit. It is what a reader who ignores
        # the bands should get, because the smallest band is the one a routed agent ships on
        # every routed session, and because pooling is the mistake this script stopped making.
        payload = {
            "session_prefix": args.session_prefix,
            "openings": len(observations),
            "prefixes": prefixes,
            "carry": round(statistics.mean(carry), 4) if carry else None,
            "band_tokens": band_boundary(observations),
            "bands": [
                {
                    "min_tokens": band.min_tokens,
                    "max_tokens": band.max_tokens,
                    "openings": band.openings,
                    "curve": band.curve,
                }
                for band in bands
            ],
            "curve": bands[0].curve if bands else [],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote the curve to {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument(
        "--session-prefix",
        default="ab-",
        help="only read sessions whose id starts with this. Must be ROUTED traffic: an"
        " unrouted agent has one prefix and it is always hot, which measures nothing.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the curve to this JSON file. `predict_saving.py --residency-curve` reads it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
