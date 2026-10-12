#!/usr/bin/env python
"""How much of the router's set-recall lift is recognising an intent it has already seen?

DESIGN.md 19.16 reports the router at 0.9225 set recall against a static top-11 menu's
0.7530, a lift of +0.170, on the `corpus-20260904` corpus. That number is measured with the
train/eval split hashed on `session_id`. This corpus is not 1,700 independent sessions: it
is 106 authored scenarios replayed about sixteen times each, so a session-hashed split puts
every eval scenario in the training half as well, and 21.5% of eval sessions have a
byte-identical opening on the other side. A split like that asks the model "which of these
106 intents is this", which is not the question an adopter's traffic will ask.

So this script re-scores the same corpus under **neighbour-blocking regimes**. A regime
names a group, and a session is scored using only the sessions outside its own group:

    block the session    the published frontier. Leave-one-out, the leakage everyone
                         already defends against, and the row that has to reproduce the
                         published number for the rest of the table to mean anything.
    block the opening    every session whose first message normalises to the same string.
                         This is exactly what a split fingerprinted on the opening text
                         buys, and the gap to the row above is what that fix is worth.
    block the scenario   every replay of the same authored intent. The honest frontier for
                         "an intent the router has never seen".
    block the persona    every session from the same synthetic customer. The frontier for
                         "a new customer with a new problem", which is stricter than any
                         real deployment with repeat users would face.

The predictor here is a k-nearest-neighbour vote over the same first-message embedding the
booster reads, not the booster itself. That is deliberate: blocking a group means refitting,
and refitting a BQML booster 106 times is not an offline script. The proxy earns its place
on the top row. It has to land on the published in-distribution number, and it does, which
is what licenses reading the rows below it as comparable.

Three arms, because menu width is half of a recall number and the published lift compares a
five-tool prediction against an eleven-tool baseline:

    voted set    the tools the neighbours actually named, at most `--menu-size` of them.
                 It averages about five tools, so it wins its lift while carrying half the
                 declarations the baseline does.
    padded       the same set filled to `--menu-size` with popular tools, by `menus.pad`,
                 which is what the shipped `menus.choose` does to every menu it publishes.
                 This is the like-for-like comparison against a static top-N.
    pooled       snapped to one of `--menu-count` canonical menus by `menus.rank`, which is
                 what the router really ships. The pool is refit per fold so that it, too,
                 is blind to the blocked group. A pool fitted over the whole corpus has
                 already seen the scenario it is being asked to serve, which is worth two
                 points on its own: 0.8755 against 0.8557 under scenario blocking.

The baseline is `menus.popularity` and the pool is `menus.choose`, both the shipped code, so
what is measured is the rule this project publishes rather than a restatement of it here.

Scenario and persona come from the corpus generation journal under `.corpus_runs/`, which is
local and untracked. Without it the two intent regimes cannot run; the script says so and
reports the rest. The warehouse has no scenario id and never will on real traffic.

Cost is two BigQuery reads, about $0.002, and a minute or so of arithmetic, most of it the
pool fits. No embedding calls, no model load, no LLM traffic. Authentication is ADC only.

Usage:
    uv run python scripts/score_unseen_intent.py
    uv run python scripts/score_unseen_intent.py --neighbours 25 --menu-size 15
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Session,
    fetch_sessions,
)

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.menus import choose, pad, popularity, rank  # noqa: E402

Menu = frozenset[str]

# How many neighbours vote. Fifteen is the value at which the leave-one-out row reproduces
# the published in-distribution recall to within 0.001, and that agreement is the whole
# licence for using a proxy at all, so it is not a free parameter. More neighbours flatter
# every regime: at k=50 the leave-one-out row reaches 0.9630, well past what the booster
# scores. The conclusion is not sensitive to the choice. Blocking the scenario costs 0.127
# of recall at k=5 and 0.064 at k=50, and the scenario-blocked lift is below the
# session-blocked one at every k in that range.
NEIGHBOURS = 15

# The published pool shape, `publish_router.DEFAULT_MENU_COUNT`. Restated rather than
# imported because importing that module drags in xgboost and a GCS client for one integer.
MENU_COUNT = 24

# How many folds the menu pool is refit over. A session is served the pool fitted without
# its whole fold, which blocks strictly more than its own group and so cannot leak it. Five
# is the cheapest number that still leaves 80% of the corpus in every fit, and the pool fit
# is the only expensive step here.
POOL_FOLDS = 5

# The corpus run to score. Not the `corpus` prefix the other scripts default to, because
# `corpus-20260831` is the 25-tool reference agent and `corpus-20260904` is the 100-tool one
# that replaced it. Mixing them would put two different tool catalogues into one popularity
# ranking and score a baseline no agent ever shipped.
DEFAULT_PREFIX = "corpus-20260904"

# Where `generate_corpus.py` writes what it generated. Untracked, local, and the only place
# `session_id -> scenario_id` exists: the warehouse records what a session did, not which
# authored intent it was replaying.
JOURNAL_DIR = REPO_ROOT / ".corpus_runs"

# The first user message of each session, as it was embedded. `score_sets.fetch_sessions`
# reads the same rows but drops the text from its `Session`, and the two things this script
# needs text for, the opening-length quartiles and the identical-opening regime, have to be
# the string the vector was computed from. The journal's copy is not: 10 of the 1,700
# sessions differ, because `router_features` skips messages under three characters and those
# sessions opened with "hi".
OPENINGS_SQL = """
SELECT
  session_id,
  ARRAY_AGG(prompt_text ORDER BY prompt_ts LIMIT 1)[OFFSET(0)] AS prompt_text
FROM `{project}.{dataset}.router_features_{agent}`
WHERE ARRAY_LENGTH(v) = @dims
  AND prompt_text IS NOT NULL
  AND STARTS_WITH(session_id, @prefix)
GROUP BY session_id
"""

SESSION_REGIME = "block the session"
SCENARIO_REGIME = "block the scenario"


@dataclass(frozen=True)
class Case:
    """One session, reduced to what a blocking regime and a recall need.

    `scenario` and `persona` are the empty string when the journal does not carry the
    session, which is how a regime discovers it cannot run rather than silently grouping
    every unknown session together.
    """

    session_id: str
    tools: Menu
    opening: str
    scenario: str
    persona: str

    @property
    def words(self) -> int:
        return len(self.opening.split())


@dataclass(frozen=True)
class Vote:
    """A tool and the weight its neighbours gave it, in the shape `menus.rank` reads."""

    label: str
    probability: float


@dataclass(frozen=True)
class Regime:
    """A rule for which other sessions a session is allowed to learn from."""

    name: str
    key: Callable[[Case], str]
    from_journal: bool = False


@dataclass
class Row:
    """One regime's scores, all over the same sessions so the rows are comparable."""

    name: str
    voted: float = 0.0
    padded: float = 0.0
    pooled: float = 0.0
    no_miss: float = 0.0
    width: float = 0.0


# Word counts the short-opening fallback is swept over. The first quartile of this corpus
# ends at 8 words, so the sweep brackets it on both sides rather than assuming it.
SHORT_OPENINGS = (4, 6, 8, 10, 12)

REGIMES = (
    Regime(SESSION_REGIME, lambda case: case.session_id),
    Regime("block the opening", lambda case: normalise(case.opening)),
    Regime(SCENARIO_REGIME, lambda case: case.scenario, from_journal=True),
    Regime("block the persona", lambda case: case.persona, from_journal=True),
)


# --- Data -----------------------------------------------------------------------------


def fetch_openings(
    project: str, dataset: str, agent: str, dims: int, prefix: str
) -> dict[str, str]:
    """Each session's first message. Same table and same ordering as `fetch_sessions`."""
    job = bigquery.Client(project=project).query(
        OPENINGS_SQL.format(project=project, dataset=dataset, agent=agent),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dims", "INT64", dims),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            ]
        ),
    )
    return {row["session_id"]: row["prompt_text"] for row in job.result()}


def load_journal(directory: Path) -> dict[str, tuple[str, str]]:
    """`session_id -> (scenario_id, persona_id)` from every generation run on this machine.

    A missing directory, a missing file and an unparseable line all come back as "no
    journal" rather than an exception. This directory is deliberately untracked, so a clone
    that has never generated a corpus is the normal case and not an error.
    """
    if not directory.is_dir():
        return {}
    journal: dict[str, tuple[str, str]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = entry.get("session_id")
            if session_id:
                journal[session_id] = (entry.get("scenario_id", ""), entry.get("persona_id", ""))
    return journal


def build_cases(
    sessions: Sequence[Session], openings: dict[str, str], journal: dict[str, tuple[str, str]]
) -> tuple[list[Case], np.ndarray]:
    """Cases and their unit vectors, in the same order, for sessions that have an opening.

    Sorted by session id, which is not cosmetic. BigQuery does not promise a row order, and
    `menus.choose` sums a float recall over the corpus in the order it is handed, so an
    unsorted corpus decides a near-tie between two candidate menus differently from one run
    to the next and the pooled column wobbles in its fourth decimal. A measurement artifact
    has to give the same answer twice.
    """
    cases = []
    vectors = []
    for session in sorted(sessions, key=lambda s: s.session_id):
        opening = openings.get(session.session_id)
        if opening is None:
            continue
        scenario, persona = journal.get(session.session_id, ("", ""))
        cases.append(
            Case(
                session_id=session.session_id,
                tools=session.tools,
                opening=opening,
                scenario=scenario,
                persona=persona,
            )
        )
        vectors.append(session.vector)
    return cases, unit(np.array(vectors, dtype=np.float64))


def normalise(text: str) -> str:
    """Case, punctuation and whitespace removed, so "Hi!  Hi" and "hi hi" are one opening.

    This is the normalisation a split key fingerprinted on the opening text would need, and
    it is why the identical-opening regime is worth a row: it measures what such a split
    removes from the promotion gate before anyone changes the gate.
    """
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.casefold()).split())


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


# --- The proxy predictor ---------------------------------------------------------------


def votes(
    similarity: np.ndarray, blocked: np.ndarray, tools: Sequence[Menu], k: int
) -> dict[str, float]:
    """Similarity-weighted votes from the nearest `k` sessions the regime leaves visible.

    Sessions that called no tool stay in the pool and cast no vote. They are a fifth of this
    corpus and they are real traffic: a session whose neighbours all went quiet is evidence
    that nothing is needed, and dropping them would silently let the vote reach further out
    for exactly those sessions.

    Blocked sessions are pushed below every real cosine rather than deleted, which keeps the
    row indexable by the caller's own ordering. `-2.0` is unreachable for a cosine.
    """
    visible = np.where(blocked, -2.0, similarity)
    nearest = np.argsort(-visible)[:k]
    weighted: dict[str, float] = {}
    for index in nearest:
        if visible[index] <= -2.0:
            continue  # the regime blocked more sessions than the corpus has neighbours
        for tool in tools[index]:
            weighted[tool] = weighted.get(tool, 0.0) + float(visible[index])
    return weighted


def voted_menu(weighted: dict[str, float], ranked: Sequence[str], size: int) -> Menu:
    """The `size` best-supported tools. Ties fall to the more popular tool, then to name.

    The tie-break matters more than it looks. At fifteen neighbours a long tail of tools
    scores one weak vote each, and without a total order the menu would depend on dict
    ordering and stop being reproducible.
    """
    order = {tool: place for place, tool in enumerate(ranked)}
    by_support = sorted(
        weighted, key=lambda tool: (-weighted[tool], order.get(tool, len(order)), tool)
    )
    return frozenset(by_support[:size])


def fold_of(key: str, folds: int) -> int:
    """Which pool fit serves this group.

    Any stable hash will do. This one is not `FARM_FINGERPRINT`, which is BigQuery's, and
    nothing here is compared with the training run's split, so the two need not agree.
    """
    return zlib.crc32(key.encode()) % folds


def fit_pools(
    cases: Sequence[Case], regime: Regime, size: int, count: int, folds: int
) -> list[tuple[Menu, ...]]:
    """One `menus.choose` pool per fold, each fitted without that fold's sessions.

    Folds rather than leave-one-group-out because the pool fit is greedy over every distinct
    tool set in the corpus and costs seconds, and 106 of them would not be an offline
    script. A fold blocks about a fifth of the intents, which is strictly more than the one
    the regime is hiding, so no pool can have seen the session it is asked to serve.
    """
    return [
        tuple(
            choose(
                [
                    sorted(case.tools)
                    for case in cases
                    if case.tools and fold_of(regime.key(case), folds) != fold
                ],
                size=size,
                count=count,
            )
        )
        for fold in range(folds)
    ]


# --- Scoring ---------------------------------------------------------------------------


def recall(menu: Menu, tools: Menu) -> float:
    """DESIGN.md 19.4's set recall for one session: the share of what it wanted, kept."""
    return len(menu & tools) / len(tools) if tools else 1.0


def mean_recall(menus: dict[int, Menu], group: Sequence[int], cases: Sequence[Case]) -> float:
    return sum(recall(menus[i], cases[i].tools) for i in group) / len(group) if group else 0.0


def score_regime(
    regime: Regime,
    cases: Sequence[Case],
    scored: Sequence[int],
    similarity: np.ndarray,
    ranked: Sequence[str],
    pools: Sequence[Sequence[Menu]],
    size: int,
    neighbours: int,
) -> tuple[Row, dict[int, Menu], dict[int, Menu]]:
    """Score every session in `scored` under one regime, and keep the menus it chose.

    Both sets of menus come back because the tables below ask further questions of the same
    predictions, and voting twice would be a slow way to get the same answer. The voted set
    is what the quartile table splits; the pooled set is what the router actually declares,
    and it is the one the short-opening sweep has to reason about.
    """
    keys = np.array([regime.key(case) for case in cases])
    tools = [case.tools for case in cases]
    row = Row(regime.name)
    menus: dict[int, Menu] = {}
    pooled_menus: dict[int, Menu] = {}

    for index in scored:
        weighted = votes(similarity[index], keys == keys[index], tools, neighbours)
        voted = voted_menu(weighted, ranked, size)
        padded = frozenset(pad(sorted(voted), ranked, size))
        pooled = rank(
            [Vote(t, w) for t, w in weighted.items()], pools[fold_of(keys[index], len(pools))]
        )

        menus[index] = voted
        pooled_menus[index] = pooled or frozenset()
        row.voted += recall(voted, tools[index])
        row.padded += recall(padded, tools[index])
        row.pooled += recall(pooled or frozenset(), tools[index])
        row.no_miss += float(tools[index] <= voted)
        row.width += len(voted)

    n = len(scored)
    return (
        Row(
            row.name, row.voted / n, row.padded / n, row.pooled / n, row.no_miss / n, row.width / n
        ),
        menus,
        pooled_menus,
    )


def static_row(menu: Menu, cases: Sequence[Case], scored: Sequence[int]) -> Row:
    """The baseline: one popularity-ranked menu for everybody, scored the same way."""
    held = sum(recall(menu, cases[i].tools) for i in scored) / len(scored)
    no_miss = sum(1 for i in scored if cases[i].tools <= menu) / len(scored)
    return Row(f"static top-{len(menu)}", held, held, held, no_miss, float(len(menu)))


def scorable(cases: Sequence[Case]) -> list[int]:
    """Which sessions a recall is taken over: the ones that called a tool, and, when the
    journal names only part of the corpus, only the ones it names.

    Every row of the table has to describe the same traffic or the differences between them
    are not lifts. A session the journal does not name has no intent to block, so scoring it
    under the scenario regime would be scoring it leave-one-out and calling it unseen. It
    still votes, and still counts towards the pool fit: an unnamed session's group key is
    the empty string, which no named session shares, so nothing has to be blind to it.
    """
    partial = 0 < sum(1 for case in cases if case.scenario) < len(cases)
    return [i for i, case in enumerate(cases) if case.tools and (case.scenario or not partial)]


def quartiles(cases: Sequence[Case], scored: Sequence[int]) -> list[list[int]]:
    """`scored` split into four equal groups by opening length, shortest first.

    Split on rank rather than on a word count, because word counts pile up on a few values
    and cutting at the quantiles gives quarters of 435 and 309 on this corpus. Ties are
    ordered by session id so the split is deterministic, which costs nothing: two sessions
    whose openings are the same length are exchangeable for this question.
    """
    order = sorted(scored, key=lambda i: (cases[i].words, cases[i].session_id))
    return [order[q * len(order) // 4 : (q + 1) * len(order) // 4] for q in range(4)]


# --- Reporting ---------------------------------------------------------------------------


def report_regimes(static: Row, rows: Sequence[Row], n: int) -> None:
    print(f"\nset recall over {n} tool-calling sessions, at menu size {static.width:.0f}\n")
    header = f"  {'regime':<20}" + "".join(
        f"{name:>9}" for name in ("voted", "lift", "padded", "lift", "pooled", "lift")
    )
    print(header + f"{'no-miss':>9}{'tools':>7}")
    print(regime_line(static, None))
    for row in rows:
        print(regime_line(row, static))
    print(
        "\n  voted     the tools the neighbours named, at most one menu's worth of them.\n"
        "  padded    the same, filled to a full menu with popular tools by `menus.pad`.\n"
        "  pooled    snapped to a canonical menu by `menus.rank`, the shipped shape. Its\n"
        "            pool is refit per fold, so the pool is blind to the blocked group too.\n"
        "  no-miss   sessions whose voted set held every tool they wanted.\n"
        "  tools     mean size of the voted set. The baseline always carries a full menu,\n"
        "            which is why the padded column is the like-for-like comparison."
    )


def regime_line(row: Row, static: Row | None) -> str:
    """One line of the regime table. `static` is None for the baseline, which has no lift."""
    cells = ""
    for value, base in (
        (row.voted, static.voted if static else 0.0),
        (row.padded, static.padded if static else 0.0),
        (row.pooled, static.pooled if static else 0.0),
    ):
        cells += f"{value:>9.4f}" + (f"{'':>9}" if static is None else f"{value - base:>+9.4f}")
    return f"  {row.name:<20}{cells}{row.no_miss:>9.4f}{row.width:>7.1f}"


def report_quartiles(
    cases: Sequence[Case],
    groups: Sequence[Sequence[int]],
    arms: dict[str, dict[int, Menu]],
    pooled: dict[int, Menu],
    static: Menu,
    strictest: str,
) -> None:
    """Recall by opening length, with the shipped width beside the voted one.

    The last column is the one to read and the others are there to stop it being misread.
    A voted set averages 4.7 tools against the static menu's 11, so a quartile where the
    voted column trails the baseline may be a quartile where the router is narrower rather
    than wrong. Only the pooled column is a like-for-like comparison, and on the shortest
    openings the two columns disagree about the sign (19.29).
    """
    print("\nthe same, split by how much the opening gave the router to work with\n")
    names = list(arms)
    labels = "".join(f"{name.replace('block the ', ''):>10}" for name in names)
    print(f"  {'opening words':<15}{'n':>6}{'static':>9}{labels}{'pooled':>10}{'miss%':>8}")
    for group in groups:
        span = f"{cases[group[0]].words}-{cases[group[-1]].words}"
        baseline = sum(recall(static, cases[i].tools) for i in group) / len(group)
        cells = "".join(f"{mean_recall(arms[name], group, cases):>10.4f}" for name in names)
        through = mean_recall(pooled, group, cases)
        missed = sum(1 for i in group if not cases[i].tools <= arms[strictest][i]) / len(group)
        print(f"  {span:<15}{len(group):>6}{baseline:>9.4f}{cells}{through:>10.4f}{missed:>8.1%}")
    print(
        f"\n  The named columns are voted sets, about 4.7 tools each. 'pooled' is the"
        f" {strictest!r}\n  arm snapped to a published menu, which is the only column the same"
        f" width as\n  'static' and so the only one comparable to it. miss% is the"
        f" {strictest!r} arm:\n  the share of sessions it left short of a tool they went on"
        " to want."
    )


def report_short_opening_fallback(
    cases: Sequence[Case],
    scored: Sequence[int],
    pooled: dict[int, Menu],
    static: Menu,
    thresholds: Sequence[int],
) -> None:
    """What a short-opening fallback to the static menu would be worth, and at what width.

    The quartile table says the router is net negative on the shortest openings. The reflex
    is to decline them, and that would be the wrong lever: declining sends the whole tool
    list, so it buys recall by giving up the entire saving on the largest quartile of the
    traffic. The static popularity menu is the same width as a routed one, so swapping to it
    costs nothing in tokens and only changes which tools are on the list.

    So the question is not "decline or not" but "is there a word count below which the
    popular menu beats the predicted one, and by enough to be worth a rule". This sweeps it.
    Every column is measured on the sessions the run already scored, with no re-vote: the
    fallback is a choice between two menus that have both already been priced.
    """
    print("\nwhat a fallback to the static menu on short openings would be worth\n")
    print(f"  {'openings of':<14}{'share':>8}{'router':>9}{'static':>9}{'blended':>9}{'delta':>9}")
    base = sum(recall(pooled[i], cases[i].tools) for i in scored) / len(scored)
    for limit in thresholds:
        short = [i for i in scored if cases[i].words <= limit]
        if not short:
            continue
        routed = sum(recall(pooled[i], cases[i].tools) for i in short) / len(short)
        fallback = sum(recall(static, cases[i].tools) for i in short) / len(short)
        blended = base + (fallback - routed) * len(short) / len(scored)
        print(
            f"  {f'{limit} words or less':<14}{len(short) / len(scored):>8.1%}{routed:>9.4f}"
            f"{fallback:>9.4f}{blended:>9.4f}{blended - base:>+9.4f}"
        )
    print(
        f"\n  Router, over every scored session and through the pool: {base:.4f}.\n"
        "  'router' and 'static' are that session subset only; 'blended' is what the whole\n"
        "  corpus would have scored under the rule. The saving is unchanged in every row,\n"
        "  because both menus are the same width."
    )


def report_verdict(
    static: Row,
    rows: Sequence[Row],
    cases: Sequence[Case],
    groups: Sequence[Sequence[int]],
    pooled: dict[str, dict[int, Menu]],
    static_menu: Menu,
) -> None:
    """The answer, in the numbers this run measured rather than the ones it expected."""
    by_name = {row.name: row for row in rows}
    print("\nverdict")
    if SCENARIO_REGIME not in by_name:
        print("  The intent regimes did not run, so this corpus cannot answer the question.")
        print("  What is above is a leave-one-out score and it is not an unseen-intent one.")
        return

    seen, unseen = by_name[SESSION_REGIME], by_name[SCENARIO_REGIME]
    kept = (unseen.voted - static.voted) / (seen.voted - static.voted)
    scenarios = len({case.scenario for case in cases if case.scenario})
    print(
        f"  Blocking only the session leaves a lift of {seen.voted - static.voted:+.4f} over the"
        f" static menu.\n  Blocking the whole scenario leaves {unseen.voted - static.voted:+.4f},"
        f" so {kept:.0%} of the measured lift survives\n  being asked about an intent this corpus"
        f" has not already shown. The rest is\n  recognising one of the {scenarios} scenarios"
        " this project authored."
    )
    if "block the opening" in by_name:
        text = by_name["block the opening"].voted - seen.voted
        print(
            f"  Blocking every identical opening as well moves recall by only {text:+.4f}, so a\n"
            "  split key fingerprinted on the opening text removes the verbatim leak and almost\n"
            "  none of the intent leak. The duplicates that carry an intent are not byte-identical."
        )
    print(
        f"  At the width the router really ships the same comparison is"
        f" {unseen.padded - static.padded:+.4f} padded\n  and {unseen.pooled - static.pooled:+.4f}"
        " through the menu pool. Quoting the voted-set lift understates\n  the router, because"
        f" it scores {seen.width:.1f} declarations against the baseline's {static.width:.0f}."
    )

    # The quartile where the unseen-intent router does worst *against the baseline*, which
    # is not the quartile where it scores lowest: a hard quartile is hard for both of them.
    #
    # Through the pool, not on the voted set, and the distinction decides the answer. The
    # voted set averages 4.7 tools against the baseline's 11, so a quartile table built on it
    # charges the router for being narrow, which is the same width artifact the paragraph
    # above warns about for the headline number. Read on the voted set the shortest quartile
    # is the one place the router loses; read at the width it ships, it wins there too.
    def against_static(group: Sequence[int]) -> float:
        baseline = sum(recall(static_menu, cases[i].tools) for i in group) / len(group)
        return mean_recall(pooled[SCENARIO_REGIME], group, cases) - baseline

    worst = min(groups, key=against_static)
    delta = against_static(worst)
    print(
        f"  Its worst quartile against the static menu is the {cases[worst[0]].words}-to-"
        f"{cases[worst[-1]].words} word openings,\n  at {delta:+.4f} through the menu pool."
        + (
            " Even there the popularity ranking is not the\n  better predictor, so there is"
            " nothing here to decline on."
            if delta >= 0
            else " On a vague opening there is nothing to\n  recognise and the popularity"
            " ranking is the better predictor."
        )
    )
    print(
        "  So a recall or lift figure published from this corpus has to be measured with the\n"
        "  split grouped by intent. A scenario list is a finite intent space, and a session\n"
        "  hash does not separate it."
    )


# --- Entry point --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset)
    size = args.menu_size or cfg.max_tools_kept

    sessions = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, False
    )
    openings = fetch_openings(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix
    )
    if not sessions or not openings:
        print(f"no sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    cases, vectors = build_cases(sessions, openings, load_journal(args.journal))
    scored = scorable(cases)
    known = sum(1 for case in cases if case.scenario)
    scenarios = len({case.scenario for case in cases if case.scenario})
    personas = len({case.persona for case in cases if case.persona})
    print(f"{len(cases)} sessions under {args.session_prefix!r}, {len(scored)} of them called")
    print(f"       a tool. {scenarios} scenarios and {personas} personas in the journal at")
    print(f"       {args.journal}, covering {known} of the sessions.")

    if 0 < known < len(cases):
        print(f"       {len(cases) - known} of them are not in the journal, so they vote but")
        print("       are not scored. Every row then describes the same sessions.")
    elif not known:
        print("       No journal covers this corpus, so the intent regimes cannot run.")

    if not scored:
        print("no session left to score; recall is undefined.", file=sys.stderr)
        return 1

    similarity = vectors @ vectors.T
    ranked = popularity([sorted(case.tools) for case in cases])
    static_menu = frozenset(pad([], ranked, size))
    static = static_row(static_menu, cases, scored)

    rows: list[Row] = []
    arms: dict[str, dict[int, Menu]] = {}
    pooled_arms: dict[str, dict[int, Menu]] = {}
    for regime in REGIMES:
        if regime.from_journal and not known:
            print(f"\nskipping {regime.name!r}: no generation journal to group on.")
            continue
        pools = fit_pools(cases, regime, size, args.menu_count, args.folds)
        row, menus, pooled_menus = score_regime(
            regime, cases, scored, similarity, ranked, pools, size, args.neighbours
        )
        rows.append(row)
        arms[regime.name] = menus
        pooled_arms[regime.name] = pooled_menus

    groups = quartiles(cases, scored)
    strictest = SCENARIO_REGIME if SCENARIO_REGIME in arms else SESSION_REGIME
    shown = {name: arms[name] for name in (SESSION_REGIME, SCENARIO_REGIME) if name in arms}
    report_regimes(static, rows, len(scored))
    report_quartiles(cases, groups, shown, pooled_arms[strictest], static_menu, strictest)
    report_short_opening_fallback(
        cases, scored, pooled_arms[strictest], static_menu, args.short_openings
    )
    report_verdict(static, rows, cases, groups, pooled_arms, static_menu)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument(
        "--session-prefix",
        default=DEFAULT_PREFIX,
        help="only score sessions whose id starts with this. Must be UNROUTED traffic, and"
        " must be one tool surface: two corpus runs against different catalogues do not"
        " share a popularity ranking.",
    )
    parser.add_argument("--journal", type=Path, default=JOURNAL_DIR)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--menu-size", type=int, default=None, help="defaults to max_tools_kept")
    parser.add_argument("--menu-count", type=int, default=MENU_COUNT)
    parser.add_argument("--folds", type=int, default=POOL_FOLDS)
    parser.add_argument(
        "--short-openings",
        type=int,
        nargs="+",
        default=SHORT_OPENINGS,
        help="word counts to sweep the short-opening fallback over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
