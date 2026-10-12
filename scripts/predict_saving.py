#!/usr/bin/env python
"""V2-5 — what the router will save on this agent, worked out before anyone spends.

An A/B is the only way to *prove* a saving and a poor way to *find* one. The reference
agent's first A/B measured +8.3% and cost real money to learn that 25 tools is too few
(DESIGN.md 18.24). The second measured -23.4% on 100 tools. Neither answer needed a model
call to predict: both were latent in traffic the agent had already logged. This script
reads that traffic and returns the expected G1.

What it needs, and where each piece comes from:

    declaration sizes     `measure_menus.declaration_tokens`, one `count_tokens` call
    the menu per session  the published artifact, replayed through `pruner.decide`
    session shape         `v_llm_response`: turns, opening prompt, history growth
    when every turn ran   the same table's timestamps

The last one is the difference between this and arithmetic anyone could do on a napkin.
§19.17 measured a routed first turn caching 54.8% against a holdout's 88.6%, and that
34-point gap was worth 22 points of G1, more than the prune itself. A prediction that
prices declarations and ignores the cache is not off by a little.

Why the clock matters
---------------------
Implicit caching keeps a prefix resident for a while after it is used. An unrouted agent
has one prefix, so every request refreshes it and a new session almost always opens warm.
A routed agent has one prefix per menu, so a menu is warm on arrival only if some other
session used that same menu recently. Whether it did is a question about timing, and the
agent's own logs answer it exactly: every turn has a timestamp, so the arrivals can simply
be replayed.

That is what this does. It replays the real arrival pattern under the menus the router
would have chosen, works out how long each menu had been idle when a session opened on it,
and prices the result. No assumption about traffic being smooth, which matters, because
real traffic is not and a burst can keep a menu warm that an average rate says is cold.

Turning an idle time into a cost takes a residency curve, and that curve is measured.
`measure_residency.py` reads a routed run, where menus genuinely idle for different lengths
of time, and reports how much of a prefix survives a given gap. The answer is a decay onto
a floor rather than a cliff, so this prices each opening at its own expected cached share
instead of calling it warm or cold. The curve belongs to the Gemini service rather than to
any agent, which is why one measured default ships here.

Idle time is not the whole story, though, and pretending it was is what made the first
version of this script optimistic. A menu opens about 17 points colder than the full list
*at the same idle time*, before any decay begins, so the curve is banded by prefix size and
a routed opening is priced on the band it will actually ship on. Why size separates them is
still open — `probe_residency.py` is what settles it — but that the two bands differ is
measured, and pricing a menu on the full list's warmth is the single largest error this
script can make.

Two different G1s, and they are both real
-----------------------------------------
Turning the router on for half the traffic is not half of turning it on for all of it. At
half, the menus get half the warming, and so does the full list the other arm is still
sending. So this reports two numbers:

    rollout    total spend at this route fraction against the router being uninstalled.
               What the bill actually does, holdout traffic included.
    as an A/B  the routed arm against the holdout arm inside that same split world.
               What `report_ab.py` will print when the experiment finishes.

Neither is a rescaling of the other. At 50% the rollout is diluted by the half that is not
routed at all, so it reads much smaller. The A/B ignores that half but pays for splitting
the warming on both sides, so it usually reads short of the full rollout it exists to
justify. On the reference agent the three numbers are -11.9%, -25.0% and -27.4%. Plan
against the A/B one; the full rollout will normally beat it, though how far depends on
where both arms sit on the residency curve and not on the split alone.

Does it work
------------
It was checked against the one A/B that has run. Reading only the unrouted `corpus-20260904`
traffic, and asked for the 50/50 split at the rate `ab-v2-100tools` actually ran at, it
predicts a routed arm opening 60.9% cached against a holdout at 89.1%, and G1 at -25.0%.
The A/B measured 58.4%, 88.4% and -23.4%. So the prediction runs optimistic by about a
point and a half of G1 and would have called that experiment correctly, which is the
decision it exists to support. Do not read it as better than a point or two.

Banding the curve is what bought most of that. Pooled into one curve, the same replay
predicted 63.6% / 87.3% / -28.8%: five points of G1 optimistic, and optimistic precisely
because it priced menus off a curve whose short-gap end was measured on the full list.

    uv run python scripts/predict_saving.py --session-prefix corpus-20260904
    uv run python scripts/predict_saving.py --session-prefix corpus-20260904 --route-fraction 0.5

Cost is two BigQuery reads and two `count_tokens` calls. No model calls, no A/B, no agent
traffic. Authentication is ADC only.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import math
import sys
import zlib
from collections.abc import Mapping, Sequence
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

# Which menu a session would get is `score_sets`' question, already answered there against
# the shipped `pruner.decide`. Importing it is what stops the two scripts from drifting
# into two different opinions about what the router does.
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Menu,
    fetch_sessions,
    menu_always,
    menu_router,
    tok,
)

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Predictor  # noqa: E402

# Vertex bills a cached input token at a tenth of an uncached one (DESIGN.md 18.16).
CACHED_RATE = 0.10

# The reference agent's system instruction, measured 2026-09-01. It is the part of the
# prefix a prune cannot touch, so it sets how much of the prefix survives pruning.
SYSTEM_TOKENS = 1777

# The key standing for "this request carried the whole tool list". Menus are keyed by
# themselves, so one namespace holds both and the simulator does not care which is which.
FULL_LIST: Menu = frozenset({"\x00full"})

# The seven domains the reference agent shipped before DESIGN.md 19.15 added twelve more.
# Together they are the twenty-five-tool surface `corpus-20260831` ran on, and naming them
# is how a run gets priced against that surface instead of today's hundred.
ORIGINAL_DOMAINS = ("account", "catalog", "orders", "payments", "returns", "shipping", "support")


# --- Which surface, and which setting -------------------------------------------------


def surface_names(domains: str | None) -> frozenset[str] | None:
    """Tool names in the named domains, or None for everything the agent declares today.

    A domain nobody declares is an error rather than an empty contribution. Silently
    ignoring a typo prices a smaller agent than the one asked for and says nothing.
    """
    if not domains:
        return None
    from reference_agent.tools import ALL_TOOLS

    wanted = {name.strip() for name in domains.split(",") if name.strip()}
    by_domain: dict[str, set[str]] = {}
    for fn in ALL_TOOLS:
        by_domain.setdefault(fn.__module__.rsplit(".", 1)[-1], set()).add(fn.__name__)
    missing = sorted(wanted - set(by_domain))
    if missing:
        raise SystemExit(
            f"no reference-agent tool lives in {missing}; the domains are "
            f"{', '.join(sorted(by_domain))}"
        )
    return frozenset().union(*(by_domain[domain] for domain in wanted))


def resolve_prunable(
    setting: str | None, live: frozenset[str], known: frozenset[str]
) -> frozenset[str]:
    """The `prunable_by_default` set this run prices: named tools, or every classless one.

    `classless` is the setting worth pricing rather than a shortcut for typing. A tool with
    no training class is kept in every menu forever, so it is the whole of what the knob can
    reach, and the answer it gives is the ceiling on what naming tools by hand could buy
    (DESIGN.md 19.27).
    """
    if not setting:
        return frozenset()
    if setting == "classless":
        return live - known
    named = frozenset(name.strip() for name in setting.split(",") if name.strip())
    unknown = named - live
    if unknown:
        raise SystemExit(f"--prunable names tools this agent does not declare: {sorted(unknown)}")
    return named


# --- The billing model ------------------------------------------------------------
#
# Pure arithmetic, no network, unit-tested in tests/test_predict_saving.py. `cost_model.py`
# holds a sibling of this that prices *policies* against ADK's explicit cache, where the
# trigger is two matching tool fingerprints. This one prices *measured sessions* against
# the implicit cache, where the trigger is recency. They share the tenth-rate rule and
# nothing else, which is why neither calls the other.


@dataclass(frozen=True)
class SessionShape:
    """One session's billing shape: how long it ran and what it carried each turn."""

    turns: int
    prefix_tokens: int  # system instruction plus declarations: shared, so cacheable
    opening_tokens: int  # the first user message: this session's alone
    growth_tokens: float  # how much history each further turn adds


def session_billed(shape: SessionShape, warm: float, rate: float = CACHED_RATE) -> float:
    """Billed-equivalent input tokens for a whole session.

    Two sources of cache. Turn one may open on a prefix another session left resident,
    which is what `warm` reports. Every turn after that carries everything the previous
    turn already sent, which is certain, because the prefix is locked for the session and
    history only grows.
    """
    turns = shape.turns
    if turns <= 0:
        return 0.0
    prefix, opening, growth = shape.prefix_tokens, shape.opening_tokens, shape.growth_tokens

    gross = turns * (prefix + opening) + growth * turns * (turns - 1) / 2
    warm_start = warm * prefix
    carried = (turns - 1) * (prefix + opening) + growth * (turns - 1) * (turns - 2) / 2
    return gross - (1.0 - rate) * (warm_start + carried)


# --- Cache residency, replayed rather than assumed ----------------------------------


# The prefix size that separates the two measured residency bands, in tokens. It is the
# geometric mean of the two sizes either side of the break `measure_residency.py` found in
# the `ab-v2-100tools` run: 24 menus at 7,849-8,296 tokens against a 16,214-token full list.
# The geometric mean and not a cut at either edge, because a caller's idea of a prefix
# differs from the warehouse's by an opening message and there is no reason to be precise
# about a boundary drawn across an empty region.
BAND_TOKENS = 11_598


@dataclass(frozen=True)
class ResidencyCurve:
    """How much of a prefix is still cached, given how long it has sat unused and how big it is.

    Knots are (gap in seconds, cached share), rising in gap and falling in share. Between
    them the share moves linearly in log gap, because the measured decay is spread over
    three orders of magnitude and a linear reading of it would be dominated by the tail.
    Outside them it is flat, which is the honest extrapolation: nothing was measured past
    the ends and the curve is already close to level at both.

    A share rather than a hit rate, because that is what the billing formula multiplies by.
    The two differ: a routed session that hits often matches only the leading part of its
    prefix, since implicit caching matches the longest common prefix and two menus diverge
    earlier than two full lists do.

    Two sets of knots, not one, because gap does not explain the data on its own. A 16k
    prefix opens 0.89 cached and a 8k prefix opens 0.73 cached *at the same idle time*, and
    the big one then shows no decay at all over the range anyone has measured. Fitting one
    curve through both hid that behind a slope, and the slope was an artifact of the traffic
    mix rather than of any decay: the big prefix is also the busy one, so it supplied nearly
    every short-gap observation and the menus nearly every long-gap one.

    The band is chosen inside `warmth` rather than by handing callers two curves. A caller
    that knows nothing about prefix size gets `knots`, the small band, which is the prefix a
    routed session actually ships. `large_knots` empty means one band was measured, which is
    the normal result for an agent whose prefixes are all one size.
    """

    knots: tuple[tuple[float, float], ...]
    large_knots: tuple[tuple[float, float], ...] = ()
    band_tokens: int = BAND_TOKENS

    def band(self, prefix_tokens: int | None = None) -> tuple[tuple[float, float], ...]:
        """The knots that apply to a prefix of this size."""
        if self.large_knots and prefix_tokens is not None and prefix_tokens >= self.band_tokens:
            return self.large_knots
        return self.knots

    def warmth(self, gap_s: float, prefix_tokens: int | None = None) -> float:
        knots = self.band(prefix_tokens)
        if not knots:
            return 0.0
        if gap_s <= knots[0][0]:
            return knots[0][1]
        if gap_s >= knots[-1][0]:
            return knots[-1][1]
        index = bisect.bisect_left([gap for gap, _ in knots], gap_s)
        (left_gap, left_p), (right_gap, right_p) = knots[index - 1], knots[index]
        span = math.log(right_gap) - math.log(left_gap)
        if span <= 0:
            return right_p
        along = (math.log(gap_s) - math.log(left_gap)) / span
        return left_p + along * (right_p - left_p)

    @property
    def floor(self) -> float:
        """What a small prefix nobody has touched in a long time is still worth."""
        return self.knots[-1][1] if self.knots else 0.0


# Measured 2026-09-04 by `measure_residency.py` over the `ab-v2-100tools` run: 575 session
# openings across 25 distinct prefixes, split into 261 on a menu of about 8k tokens and 314
# on the 16,214-token full list. Three readings matter and none is what a window model would
# say. Implicit caching is best-effort, so even the full list opens only 89% cached two
# seconds after it last ran; residency does not decay to nothing, so a menu that runs at all
# runs mostly warm; and the menu opens 17 points behind the full list at the same idle time,
# before any decay starts, which is a deficit no amount of traffic can warm away. That last
# one is why routing costs more than the declaration arithmetic says it should.
#
# What separates the bands is not established. Size and a shared 1,600-token system-instruction
# head are confounded in every run logged so far; `probe_residency.py` separates them. Until
# it has run, size is the variable that orders the observations, so size is what selects here.
#
# 2026-09-05: it has run, and it does not agree with this curve's tail (DESIGN 19.30). Under
# control, at every size from 4k to 32k, nothing at all is still cached after a 600-second idle;
# this curve reads 0.535 there on the small band and 0.860 flat on the large one. The probe also
# kills the two mechanisms that could have explained a long tail: 24 times the send volume buys
# no residency, and neither does a tenth of the cadence. What is left is a short TTL, near
# certain at 5 seconds and gone by 600.
#
# The curve is deliberately not refitted from it. Twenty-five controlled first-probe observations
# cannot replace 575 live openings, and filler declarations are not an agent. The likeliest
# reconciliation is that `measure_residency.py` groups by `kept_tools` while implicit caching keys
# on leading bytes, so the fit's long-gap tail may be warmed by traffic the grouping cannot see.
# Read this curve as a description of the traffic it was measured on. Do not read its floor as
# evidence that an idle prefix retains anything.
MEASURED_CURVE = ResidencyCurve(
    knots=(
        (0.9, 0.725),
        (6.1, 0.694),
        (23.7, 0.634),
        (309.9, 0.583),
        (1357.5, 0.475),
    ),
    large_knots=(
        (2.4, 0.894),
        (4.9, 0.882),
        (43.3, 0.860),
    ),
)


def load_curve(path: str) -> ResidencyCurve:
    """Read a curve `measure_residency.py --out` wrote, for an agent that has routed data.

    A file with one band, and a file written before banding existed, both read as a single
    curve that applies at every prefix size. That is what the file says, so it is what the
    curve should do.
    """
    payload = json.loads(Path(path).read_text())
    bands = payload.get("bands") or [{"curve": payload["curve"]}]
    knots = [tuple((float(gap), float(p)) for gap, p in band["curve"]) for band in bands]
    return ResidencyCurve(
        knots=knots[0],
        large_knots=knots[1] if len(knots) > 1 else (),
        band_tokens=int(payload.get("band_tokens") or BAND_TOKENS),
    )


def replay_warmth(
    turns: Sequence[tuple[float, Menu]],
    openings: Sequence[tuple[str, float, Menu]],
    curve: ResidencyCurve,
    sizes: Mapping[Menu, int] | None = None,
) -> dict[str, float]:
    """How warm each session opening lands, as a cached share the curve supplies.

    `turns` is every request in the run, tagged with the prefix it carried. `openings` is
    the subset that started a session. How warm a session opens is set by how long it has
    been since anything else carried the same prefix, and by how big that prefix is, which
    `sizes` supplies. An opening with no predecessor at all gets nothing: a prefix that has
    never been sent cannot be cached.

    The result is an expected share rather than a verdict, and it goes straight into the
    billing formula, where warmth already enters linearly. So the expected bill is right
    even though no individual session's fate is known.
    """
    by_prefix: dict[Menu, list[float]] = {}
    for at, prefix in turns:
        by_prefix.setdefault(prefix, []).append(at)
    for times in by_prefix.values():
        times.sort()

    warm: dict[str, float] = {}
    for session_id, at, prefix in openings:
        times = by_prefix.get(prefix, ())
        index = bisect.bisect_left(times, at)
        tokens = None if sizes is None else sizes.get(prefix)
        warm[session_id] = curve.warmth(at - times[index - 1], tokens) if index > 0 else 0.0
    return warm


# --- What the agent's own traffic says ---------------------------------------------

_TURNS_SQL = """
SELECT
  session_id,
  timestamp,
  ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) AS ordinal,
  COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
  LEAST(COALESCE(usage_cached_tokens, 0), COALESCE(usage_prompt_tokens, 0)) AS cached_tokens
FROM `{project}.{dataset}.v_llm_response`
WHERE agent = @agent
  AND STARTS_WITH(session_id, @prefix)
  AND usage_prompt_tokens IS NOT NULL
ORDER BY session_id, timestamp
"""


@dataclass(frozen=True)
class SessionFacts:
    """One session as it was actually billed, plus when each of its turns happened."""

    session_id: str
    times: tuple[float, ...]  # seconds from the start of the run, one per turn
    opening_tokens: int  # turn-one prompt less the prefix it carried
    growth_tokens: float  # per-turn history growth, this session's own where it has one
    first_prompt: int
    first_cached: int

    @property
    def turns(self) -> int:
        return len(self.times)

    @property
    def opened_at(self) -> float:
        return self.times[0]


@dataclass(frozen=True)
class Traffic:
    sessions: tuple[SessionFacts, ...]
    prefix_tokens: int  # the prefix these sessions actually carried

    @property
    def turns(self) -> int:
        return sum(session.turns for session in self.sessions)

    @property
    def span_hours(self) -> float:
        times = [at for session in self.sessions for at in session.times]
        return (max(times) - min(times)) / 3600.0 if times else 0.0

    @property
    def residency(self) -> float:
        """Cached share of turn one, against the prefix that was available to cache."""
        cached = sum(session.first_cached for session in self.sessions)
        offered = sum(min(self.prefix_tokens, session.first_prompt) for session in self.sessions)
        return cached / offered if offered else 0.0


def fetch_traffic(
    project: str, dataset: str, agent: str, prefix: str, prefix_tokens: int
) -> Traffic:
    """Read every turn under `prefix` and fold it into one record per session."""
    job = bigquery.Client(project=project).query(
        _TURNS_SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            ]
        ),
    )
    rows = [dict(row) for row in job.result()]
    if not rows:
        return Traffic((), prefix_tokens)

    origin = min(row["timestamp"] for row in rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"]), []).append(row)

    partial: list[tuple[str, tuple[float, ...], int, float | None, int, int]] = []
    for session_id, turns in grouped.items():
        turns.sort(key=lambda row: row["timestamp"])  # type: ignore[arg-type,return-value]
        times = tuple((row["timestamp"] - origin).total_seconds() for row in turns)  # type: ignore[operator]
        first = int(turns[0]["prompt_tokens"])  # type: ignore[arg-type]
        last = int(turns[-1]["prompt_tokens"])  # type: ignore[arg-type]
        growth = (last - first) / (len(turns) - 1) if len(turns) > 1 else None
        partial.append(
            (
                session_id,
                times,
                first - prefix_tokens,
                growth,
                first,
                int(turns[0]["cached_tokens"]),
            )  # type: ignore[arg-type]
        )

    growths = [growth for *_, growth, _, _ in partial if growth is not None]
    mean_growth = sum(growths) / len(growths) if growths else 0.0
    # A session that never reached turn two has no growth of its own, so it inherits the
    # agent's. Fixing it up here keeps the per-session path free of the special case.
    return Traffic(
        tuple(
            SessionFacts(sid, times, opening, mean_growth if g is None else g, first, cached)
            for sid, times, opening, g, first, cached in partial
        ),
        prefix_tokens,
    )


def surface_mismatch(traffic: Traffic, session_prefix: str, full_prefix: int) -> str | None:
    """Did this traffic run on the tool surface being priced? The complaint, or None.

    `opening_tokens` is the turn-one prompt minus the prefix the agent declares today, so
    traffic from a smaller surface makes it negative and every downstream number nonsense:
    a session billed on a negative opening can come out cheaper than free, and a G1 built
    from it prints in the hundreds of per cent. The sum rather than any single session,
    because one short opening is a short question and a whole corpus of them is a different
    agent.

    Two ways to trip it, and the message names both because they need different fixes. The
    traffic predates a change to the declarations or the system instruction, in which case
    no prefix will do and a fresh corpus is the only answer. Or the prefix is loose enough
    to match two corpora at once, in which case a dated prefix fixes it (DEV_NOTES,
    2026-09-05).
    """
    if sum(session.opening_tokens for session in traffic.sessions) >= 0:
        return None
    return (
        f"turn-one prompts under {session_prefix!r} average less than the {full_prefix:,}-token"
        " prefix the agent declares today, so this traffic ran on a different tool surface and"
        " cannot price this one. Pick a prefix from after the surface last changed, and make it"
        f" specific enough not to pool two runs: {len(traffic.sessions):,} sessions matched,"
        " which may be more than one corpus."
    )


# --- Prediction ---------------------------------------------------------------------


def routes(session_id: str, fraction: float) -> bool:
    """Whether this session is the router's to touch. Stable across processes and sweeps."""
    return zlib.crc32(session_id.encode()) % 10_000 < fraction * 10_000


@dataclass(frozen=True)
class Prediction:
    route_fraction: float
    rate_multiplier: float
    rollout_g1: float
    ab_g1: float | None  # undefined when the split leaves one arm empty
    routed_sessions: int
    warm_openings: float  # share of routed openings that landed on a resident prefix
    full_warm_openings: float


def predict(
    traffic: Traffic,
    menus: dict[str, Menu | None],
    full_prefix_tokens: int,
    prefix_tokens: dict[Menu, int],
    curve: ResidencyCurve,
    route_fraction: float,
    rate_multiplier: float = 1.0,
) -> Prediction:
    """Price the deployment, and separately price what an A/B inside it would report."""
    facts = {
        session.session_id: session for session in traffic.sessions if session.session_id in menus
    }
    if not facts:
        raise ValueError("no session appears in both the billing table and the replay")

    def scaled(at: float) -> float:
        return at / rate_multiplier

    # Who ships what, in the world with this route fraction.
    prefix_of: dict[str, Menu] = {}
    for session_id in facts:
        menu = menus[session_id]
        prefix_of[session_id] = (
            menu if menu is not None and routes(session_id, route_fraction) else FULL_LIST
        )

    # How big each prefix is, so the curve can pick the band. Without this every prefix
    # would be priced on the small band, which reads the full list 17 points too cold.
    sizes: dict[Menu, int] = {FULL_LIST: full_prefix_tokens, **prefix_tokens}

    turns = [
        (scaled(at), prefix_of[session_id])
        for session_id, session in facts.items()
        for at in session.times
    ]
    openings = [
        (session_id, scaled(session.opened_at), prefix_of[session_id])
        for session_id, session in facts.items()
    ]
    warm = replay_warmth(turns, openings, curve, sizes)

    # The counterfactual: the router uninstalled, so every request carries the full list.
    off_turns = [(scaled(at), FULL_LIST) for session in facts.values() for at in session.times]
    off_openings = [
        (session_id, scaled(session.opened_at), FULL_LIST) for session_id, session in facts.items()
    ]
    off_warm = replay_warmth(off_turns, off_openings, curve, sizes)

    def bill(session: SessionFacts, prefix: int, is_warm: float) -> float:
        shape = SessionShape(session.turns, prefix, session.opening_tokens, session.growth_tokens)
        return session_billed(shape, is_warm)

    # How warm the full list runs in this world. A routed session priced as though it had
    # been holdout would have opened on the full list, and the honest estimate of how warm
    # that would have been is how warm it was for the sessions that really did.
    on_full = [warm[sid] for sid, prefix in prefix_of.items() if prefix == FULL_LIST]
    full_warmth = sum(on_full) / len(on_full) if on_full else 0.0

    on_total = off_total = 0.0
    routed_billed = counterfactual_billed = 0.0
    routed_warm = 0.0
    for session_id, session in facts.items():
        prefix = prefix_of[session_id]
        tokens = full_prefix_tokens if prefix == FULL_LIST else prefix_tokens[prefix]
        on_total += bill(session, tokens, warm[session_id])
        off_total += bill(session, full_prefix_tokens, off_warm[session_id])
        if prefix == FULL_LIST:
            continue
        routed_warm += warm[session_id]
        routed_billed += bill(session, tokens, warm[session_id])
        # The same session, in the same world, had it fallen on the holdout side. That is
        # the comparison `report_ab.py` makes, with the session mix held fixed.
        counterfactual_billed += bill(session, full_prefix_tokens, full_warmth)

    routed_sessions = sum(1 for prefix in prefix_of.values() if prefix != FULL_LIST)
    return Prediction(
        route_fraction=route_fraction,
        rate_multiplier=rate_multiplier,
        rollout_g1=(on_total / off_total - 1.0) if off_total else 0.0,
        ab_g1=(routed_billed / counterfactual_billed - 1.0)
        if counterfactual_billed and routed_sessions and on_full
        else None,
        routed_sessions=routed_sessions,
        warm_openings=routed_warm / routed_sessions if routed_sessions else 0.0,
        full_warm_openings=full_warmth,
    )


# --- Report ---------------------------------------------------------------------------


def report(
    traffic: Traffic,
    full_prefix_tokens: int,
    declaration_total: int,
    menus: dict[str, Menu | None],
    prefix_tokens: dict[Menu, int],
    curve: ResidencyCurve,
    curve_source: str,
    base: Prediction,
    rate_sweep: Sequence[Prediction],
    fraction_sweep: Sequence[Prediction],
) -> None:
    sessions = len(traffic.sessions)
    opening = sum(s.opening_tokens for s in traffic.sessions) / sessions
    growth = sum(s.growth_tokens for s in traffic.sessions) / sessions
    print(f"\n{sessions} unrouted sessions, {traffic.turns} turns over {traffic.span_hours:.2f} h")
    print(f"  full prefix              {full_prefix_tokens:,} tokens")
    print(f"    system instruction     {SYSTEM_TOKENS:,}")
    print(f"    declarations           {declaration_total:,}, measured against Vertex")
    shortest = min(s.opening_tokens for s in traffic.sessions)
    print(f"  opening message          {opening:,.0f} tokens, shortest {shortest:,}")
    print(f"  history growth per turn  {growth:+,.0f} tokens")
    print(f"  turns per session        {traffic.turns / sessions:.2f}")
    print(f"  arrivals                 {traffic.turns / traffic.span_hours:,.0f} turns/hour")

    print(f"\nresidency curve, {curve_source}")
    if curve.large_knots:
        print(f"  a menu, and a prefix of {curve.band_tokens:,}+ tokens, are priced separately")
    for label, knots in (("menu", curve.knots), ("full list", curve.large_knots)):
        for gap, probability in knots:
            print(f"  {label:<10} idle {gap:>8,.1f} s   {probability:.1%} still cached")
    print(f"  this corpus opened warm  {traffic.residency:.1%}, on its one unrouted prefix")

    pool = {menu for menu in menus.values() if menu is not None}
    declined = sum(1 for menu in menus.values() if menu is None)
    kept = [prefix_tokens[m] - SYSTEM_TOKENS for m in menus.values() if m is not None]
    print(f"\nreplaying the published pool: {len(pool)} menus used over {len(menus)} sessions")
    print(f"  declined, ships full     {declined}")
    if kept:
        print(f"  declarations shipped     {sum(kept) / len(kept):,.0f} tokens on average")

    print(f"\npredicted at {base.route_fraction:.0%} routed and the measured arrival rate")
    print(f"  routed openings warm     {base.warm_openings:.1%}")
    print(f"  full-list openings warm  {base.full_warm_openings:.1%}")
    print(f"  G1, rollout              {base.rollout_g1:+.1%}")
    print(f"  G1, as an A/B reports it {_pct(base.ab_g1)}")

    print("\nif the agent were busier or quieter")
    print("  rate      turns/hour   routed warm   G1 rollout")
    for prediction in rate_sweep:
        rate = traffic.turns / traffic.span_hours * prediction.rate_multiplier
        print(
            f"  {prediction.rate_multiplier:>5.2f}x   {rate:>10,.0f}"
            f"   {prediction.warm_openings:>11.1%}   {prediction.rollout_g1:>+10.1%}"
        )

    print("\nif only part of the traffic is routed")
    print("  routed share   G1 rollout   G1 as an A/B reports it")
    for prediction in fraction_sweep:
        print(
            f"  {prediction.route_fraction:>12.0%}   {prediction.rollout_g1:>+10.1%}"
            f"   {_pct(prediction.ab_g1):>23}"
        )
    print(
        "\n  An A/B splits the traffic and so splits the warming, on both sides, which is"
        "\n  why it usually reads short of the full rollout it exists to justify. The gap is"
        "\n  a property of the split rather than noise in it, so do not rescale one into the"
        "\n  other: read the column that matches what you are about to run."
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


# --- Entry point ------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.menus)} menus")

    tokens = asyncio.run(declaration_tokens(surface_names(args.domains)))
    live = frozenset(tokens)
    declaration_total = sum(tokens.values())
    full_prefix = SYSTEM_TOKENS + declaration_total
    if args.domains:
        print(f"pricing a {len(live)}-tool surface: domains {args.domains}")

    prunable = resolve_prunable(args.prunable, live, frozenset(model.classes))
    if prunable:
        cfg = replace(cfg, prunable_by_default=prunable)
        weight = sum(tokens[name] for name in prunable)
        print(
            f"prunable_by_default names {len(prunable)} of {len(live)} tools,"
            f" {weight:,} of {declaration_total:,} declaration tokens"
        )

    traffic = fetch_traffic(
        args.project, args.dataset, args.agent, args.session_prefix, full_prefix
    )
    if not traffic.sessions:
        print(f"no sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1
    complaint = surface_mismatch(traffic, args.session_prefix, full_prefix)
    if complaint:
        print(f"\n{complaint}", file=sys.stderr)
        return 1

    if args.residency_curve:
        curve = load_curve(args.residency_curve)
        curve_source = f"read from {args.residency_curve}"
    else:
        curve = MEASURED_CURVE
        curve_source = "the shipped default, measured on a routed run"

    scored = fetch_sessions(
        args.project,
        args.dataset,
        args.agent,
        cfg.embedding_dims,
        args.session_prefix,
        eval_only=False,
    )
    billed = {session.session_id for session in traffic.sessions}
    predictor = Predictor(model)
    known = frozenset(model.classes)
    menus: dict[str, Menu | None] = {}
    for session in scored:
        if session.session_id not in billed:
            continue  # scored but never billed, so there is nothing to price
        ranked = predictor.predict(session.vector)
        chosen = menu_always(ranked, live, known, cfg, model.menus)
        menus[session.session_id] = menu_router(ranked, chosen, live, tokens, cfg)
    if not menus:
        print(
            "no session appears in both the feature table and the billing table.", file=sys.stderr
        )
        return 1

    prefix_tokens = {
        menu: SYSTEM_TOKENS + tok(menu, tokens) for menu in menus.values() if menu is not None
    }

    def at(fraction: float, multiplier: float = 1.0) -> Prediction:
        return predict(traffic, menus, full_prefix, prefix_tokens, curve, fraction, multiplier)

    base = at(args.route_fraction)
    rate_sweep = [at(args.route_fraction, m) for m in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)]
    fraction_sweep = [at(f) for f in (0.25, 0.5, 0.75, 1.0)]
    report(
        traffic,
        full_prefix,
        declaration_total,
        menus,
        prefix_tokens,
        curve,
        curve_source,
        base,
        rate_sweep,
        fraction_sweep,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--session-prefix",
        # Dated, not the bare "corpus" it used to be. That matched corpus-20260830 as well,
        # and the two corpora ran on different tool surfaces, so the default invocation
        # pooled traffic it then correctly refused to price. A default that always fails
        # reads as a broken script.
        default="corpus-20260904",
        help="only read sessions whose id starts with this. Must be UNROUTED traffic that"
        " ran on the tool surface the agent declares today.",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="comma-separated reference-agent tool modules, e.g."
        f" {','.join(ORIGINAL_DOMAINS)} for the 25-tool surface. Default is every tool the"
        " agent declares today. The traffic must have run on the surface you name.",
    )
    parser.add_argument(
        "--prunable",
        default=None,
        help="what to put in config.prunable_by_default before replaying: comma-separated"
        " tool names, or the word `classless` for every tool the artifact has no class for."
        " Default is the empty set the router ships with.",
    )
    parser.add_argument(
        "--route-fraction",
        type=float,
        default=1.0,
        help="share of traffic the router touches. 1.0 is a full rollout; use an A/B's own"
        " split to predict what that A/B will report.",
    )
    parser.add_argument(
        "--residency-curve",
        default=None,
        help="a curve `measure_residency.py --out` wrote. Worth passing once this agent has"
        " routed traffic of its own; until then the shipped default is the better estimate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
