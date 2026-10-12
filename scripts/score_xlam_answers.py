#!/usr/bin/env python
"""Did the routed arm answer as well as the holdout arm? — E1's quality check.

Every other A/B this project has run could measure what a session cost and could not measure
whether it was right, because we wrote the conversations and nothing said which tools they
were supposed to reach for. The xLAM corpus ships that answer key: each query comes with the
calls its authors say it needs. `run_xlam_traffic.py` writes them into the journal as
`wanted`, beside the `called` list the agent actually produced.

So this compares the two arms on the one thing a token saving can be quietly bought with.
A pruned menu that is missing a tool does not produce an error. It produces a session that
calls nothing, answers from the model's own knowledge or declines, ends a turn early, and
looks cheap in every ledger this project has. Recall against the answer key is what tells
that apart from a real saving.

Three numbers per arm:

  **hit rate** — the share of sessions that called every tool the key names. The strictest
  reading and the one closest to "did the user get what they asked for".
  **recall** — the share of wanted calls that happened, averaged over sessions. Partial
  credit, so a session that needed two tools and made one is not scored the same as one that
  made neither.
  **silence** — the share of sessions that wanted a tool and called nothing at all. This is
  the failure a prune causes, as distinct from calling the wrong thing.

The arms come from the router's own provenance, by the same rule `report_ab.py` and
`report_ledger.py` use, so the three scripts cannot disagree about who was in which arm.

Cost is one BigQuery read. No model calls.

Usage:
    uv run python scripts/score_xlam_answers.py --prefix xlam-ab1-
    uv run python scripts/score_xlam_answers.py --prefix xlam-ab1- --journal .ab_runs/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402

DEFAULT_AGENT = "xlam"
DEFAULT_JOURNAL_DIR = REPO_ROOT / ".ab_runs"

ROUTED_SOURCES = ("fresh", "reused")

_ARMS_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
)
SELECT
  session_id,
  CASE
    WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
    WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
    ELSE 'neither'
  END AS arm
FROM decisions
GROUP BY session_id
"""


@dataclass(frozen=True)
class Session:
    """One journalled session, with the answer key beside what the agent did."""

    session_id: str
    wanted: frozenset[str]
    called: frozenset[str]

    @property
    def hit(self) -> bool:
        return self.wanted <= self.called

    @property
    def recall(self) -> float:
        return len(self.wanted & self.called) / len(self.wanted)

    @property
    def silent(self) -> bool:
        return not self.called


@dataclass(frozen=True)
class Score:
    """One arm's answer quality. Sessions the key says wanted nothing are not in here."""

    arm: str
    sessions: int
    hits: int
    silences: int
    mean_recall: float
    sd_recall: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.sessions if self.sessions else 0.0

    @property
    def silent_rate(self) -> float:
        return self.silences / self.sessions if self.sessions else 0.0


def read_journal(path: Path) -> list[Session]:
    """Sessions from one run's journal, dropping the ones with no tool in the key.

    A query the corpus answers with no call has no recall to measure, the same exclusion
    `publish_router.py` makes when it gates on menu recall (DESIGN.md 19.4). Counting them
    would score both arms a free one and shrink every difference toward zero.
    """
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wanted = frozenset(row.get("wanted") or ())
        if not wanted:
            continue
        out.append(
            Session(
                session_id=str(row["session_id"]),
                wanted=wanted,
                called=frozenset(row.get("called") or ()),
            )
        )
    return out


def score(arm: str, sessions: Sequence[Session]) -> Score:
    recalls = [s.recall for s in sessions]
    mean = sum(recalls) / len(recalls) if recalls else 0.0
    spread = sum((r - mean) ** 2 for r in recalls) / (len(recalls) - 1) if len(recalls) > 1 else 0.0
    return Score(
        arm=arm,
        sessions=len(sessions),
        hits=sum(1 for s in sessions if s.hit),
        silences=sum(1 for s in sessions if s.silent),
        mean_recall=mean,
        sd_recall=math.sqrt(spread),
    )


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def welch(routed: Score, holdout: Score) -> tuple[float, float] | None:
    """Welch's t on mean recall. None when there is not enough of either arm to test."""
    if routed.sessions < 2 or holdout.sessions < 2:
        return None
    variance = routed.sd_recall**2 / routed.sessions + holdout.sd_recall**2 / holdout.sessions
    if variance <= 0.0:
        return None
    t = (routed.mean_recall - holdout.mean_recall) / math.sqrt(variance)
    return t, 2.0 * (1.0 - normal_cdf(abs(t)))


def missed_tools(sessions: Iterable[Session], limit: int) -> list[tuple[str, int]]:
    """Which tools the arm was asked for and did not call, commonest first."""
    tally: dict[str, int] = {}
    for s in sessions:
        for tool in s.wanted - s.called:
            tally[tool] = tally.get(tool, 0) + 1
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def fetch_arms(project: str, dataset: str, agent: str, prefix: str) -> dict[str, str]:
    from google.cloud import bigquery  # imported here so --help costs nothing

    client = bigquery.Client(project=project)  # no credentials argument: ADC only
    job = client.query(
        _ARMS_SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
                bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES)),
            ]
        ),
    )
    return {row["session_id"]: row["arm"] for row in job.result()}


def report(routed: Score, holdout: Score, sessions: dict[str, list[Session]]) -> None:
    print(f"\nxLAM answer quality — {routed.sessions} routed, {holdout.sessions} holdout")
    print("(sessions whose answer key names no tool are excluded; they have no recall)\n")

    print(f"  {'':<26}{'holdout':>12}{'routed':>12}{'change':>10}")
    for label, r, h, places in (
        ("called every wanted tool", routed.hit_rate, holdout.hit_rate, 4),
        ("recall of wanted calls", routed.mean_recall, holdout.mean_recall, 4),
        ("called nothing at all", routed.silent_rate, holdout.silent_rate, 4),
    ):
        delta = f"{(r - h) / h:+.1%}" if h else "n/a"
        print(f"  {label:<26}{h:>12.{places}f}{r:>12.{places}f}{delta:>10}")

    print("\n  is the routed arm answering worse?")
    tested = welch(routed, holdout)
    if tested is None:
        print("    not enough sessions in one arm to test.")
    else:
        t, p = tested
        verdict = (
            "no measurable difference in answer quality"
            if p >= 0.05
            else "the arms answer differently, and the saving is not free"
        )
        print(f"    Welch t = {t:+.2f}, two-sided p = {p:.3f} — {verdict}.")

    for arm in ("routed", "holdout"):
        missed = missed_tools(sessions[arm], 8)
        if missed:
            joined = ", ".join(f"{tool} x{n}" for tool, n in missed)
            print(f"\n  most missed by {arm}: {joined}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--prefix", required=True, help="session prefix, e.g. xlam-ab1-")
    parser.add_argument(
        "--journal",
        default=None,
        help="the run's journal; defaults to .ab_runs/<prefix without its trailing dash>.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = (
        Path(args.journal)
        if args.journal
        else DEFAULT_JOURNAL_DIR / f"{args.prefix.rstrip('-')}.jsonl"
    )
    if not path.exists():
        print(f"no journal at {path}.", file=sys.stderr)
        return 1

    journalled = read_journal(path)
    arms = fetch_arms(args.project, args.dataset, args.agent, args.prefix)
    sessions: dict[str, list[Session]] = {"routed": [], "holdout": []}
    for s in journalled:
        arm = arms.get(s.session_id)
        if arm in sessions:
            sessions[arm].append(s)

    if not sessions["routed"] or not sessions["holdout"]:
        print("a run needs both arms to compare and at least one is missing.", file=sys.stderr)
        return 1

    report(score("routed", sessions["routed"]), score("holdout", sessions["holdout"]), sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
