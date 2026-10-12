#!/usr/bin/env python
"""Build-order task 10 — does the published artifact actually predict?

Everything up to this point has been checked in the place it was produced. The
gate metrics were measured inside BigQuery, where the model and the label mapping
are both correct by construction. The publisher checked the artifact against the
export. `check_embedding_parity.py` checked that serving embeds the way training
did. None of that runs the thing end to end.

This does. It loads the published artifact through `ModelStore`, hands it to
`Predictor`, and scores it on held-out rows whose true `tool_name` is known,
using the same split the training procedure used: the hash of the session's
normalised opening message, not of its id (DESIGN.md 9.4, 19.33). Then it
compares what it measured against what the run record claims.

It scores twice, and the difference between the two is the point:

  * **stored vectors** — the embeddings BigQuery already computed, fed straight
    to the booster. This isolates the model and the label order. If the class
    list were published in the wrong order, top-1 accuracy here collapses to
    roughly chance and the failure is unmissable.
  * **re-embedded prompts** — the same prompts through `Embedder`, the real hot
    path. Any gap between this and the first number is a train/serve difference
    that the stored vectors cannot show.

Cost is one BigQuery read plus one embedding call per sampled prompt. The default
sample of 500 is fractions of a cent. Authentication is ADC only. No
service-account key is created, read or accepted.

Usage:
    uv run python scripts/check_router.py
    uv run python scripts/check_router.py --limit 300 --agent support
    uv run python scripts/check_router.py --skip-embedding
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through
# `pythonpath` in pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.embedding import Embedder  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Predictor  # noqa: E402

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# The training procedure's split, repeated here rather than parameterised. If it
# changes there it must change here, and a mismatch shows up as a coverage number
# that disagrees with the run record, which is exactly the signal this wants.
EVAL_PCT = 25

# The split key, copied character for character out of train_router.sql. Two sessions that
# open with the same question, modulo case and whitespace, land on the same side; hashing
# the session id instead put byte-identical openings on both sides and leaked the answer
# into training. Lowercase, collapse whitespace, trim, and fall back to the session id when
# the opening is NULL, exactly as the procedure does.
OPENING_KEY = r"IFNULL(TRIM(REGEXP_REPLACE(LOWER(f.prompt_text), r'\s+', ' ')), f.session_id)"

# Embedding sequentially against a cold client blows the production hot-path
# budget, so this script uses its own timeout for the same reason
# check_embedding_parity.py does. Serving latency is that script's subject.
TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class Row:
    """One held-out prompt, its stored vector, and the tool the agent really called."""

    prompt: str
    vector: list[float]
    tool_name: str


@dataclass(frozen=True)
class Score:
    """How a ranking did over the sampled rows."""

    label: str
    scored: int
    top1: int
    covered: int
    empty: int

    @property
    def accuracy(self) -> float:
        return self.top1 / self.scored if self.scored else 0.0

    @property
    def coverage(self) -> float:
        return self.covered / self.scored if self.scored else 0.0


# --- Data ---------------------------------------------------------------------


def fetch_eval_rows(project: str, dataset: str, agent: str, dims: int, limit: int) -> list[Row]:
    """Held-out tool-calling rows, sampled evenly across the corpus.

    ABSTAIN rows are excluded because coverage is defined over turns that
    actually called a tool (DESIGN.md 10), which is what the run record measured.

    Ordered by a fingerprint rather than by time, for the same reason
    `check_embedding_parity.py` is: the table is written oldest-first, so a bare
    LIMIT would sample one corner of one day.

    The split is decided per session by its opening message, so the query has to
    find each session's first message before it can tell which side the session
    is on. `opening` is the same CTE the training procedure opens with.
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
      )
      SELECT prompt_text, v, tool_name
      FROM `{table}`
      WHERE ARRAY_LENGTH(v) = @dims
        AND prompt_text IS NOT NULL
        AND tool_name != 'ABSTAIN'
        AND session_id IN (SELECT session_id FROM held_out)
      ORDER BY FARM_FINGERPRINT(invocation_id)
      LIMIT @limit
    """
    job = bigquery.Client(project=project).query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dims", "INT64", dims),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        ),
    )
    return [Row(r["prompt_text"], list(r["v"]), r["tool_name"]) for r in job.result()]


async def embed_rows(embedder: Embedder, rows: list[Row]) -> list[list[float] | None]:
    """Re-embed every prompt through the serving path, sequentially."""
    vectors: list[list[float] | None] = []
    for index, row in enumerate(rows, start=1):
        vectors.append(await embedder.embed(row.prompt))
        if index % 25 == 0:
            print(f"  [{index}/{len(rows)}] embedded")
    return vectors


# --- Scoring --------------------------------------------------------------------


def score(
    label: str, predictor: Predictor, rows: list[Row], vectors: list[list[float] | None], top_n: int
) -> Score:
    """Rank each row and count top-1 hits and top-N coverage."""
    scored = top1 = covered = empty = 0
    for row, vector in zip(rows, vectors, strict=True):
        if vector is None:
            empty += 1
            continue
        ranked = predictor.predict(vector)
        if not ranked:
            # An empty ranking is passthrough at serving time, which keeps every
            # tool. Counting it as covered would flatter the model, so it is not
            # scored at all and is reported separately.
            empty += 1
            continue
        scored += 1
        kept = [p.label for p in ranked[:top_n]]
        top1 += ranked[0].label == row.tool_name
        covered += row.tool_name in kept
    return Score(label, scored, top1, covered, empty)


def report(scores: list[Score], recorded_coverage: float | None, top_n: int) -> bool:
    """Print the comparison and return whether it holds together."""
    print(f"\nScored on held-out rows, keeping the top {top_n}.\n")
    for s in scores:
        print(
            f"  {s.label:<18} n={s.scored:<5} top-1 {s.accuracy:.4f}   "
            f"top-{top_n} coverage {s.coverage:.4f}"
            + (f"   ({s.empty} unranked)" if s.empty else "")
        )

    if not scores or not scores[0].scored:
        print("\nFAIL: nothing could be scored.")
        return False

    passed = True
    stored = scores[0]

    # The sharpest signal available: a reversed label mapping cannot survive this.
    if stored.accuracy < 0.20:
        print(
            f"\nFAIL: top-1 accuracy of {stored.accuracy:.4f} on stored training vectors is "
            "near chance. The booster is being read through the wrong label order, or the "
            "artifact does not match the run record. Check `classes` in metadata.json against "
            "assets/model_metadata.json (DESIGN.md 18.8)."
        )
        passed = False

    if recorded_coverage is not None:
        gap = abs(stored.coverage - recorded_coverage)
        print(
            f"\n  run record coverage {recorded_coverage:.4f}, measured here {stored.coverage:.4f}"
        )
        # A sample of a few hundred rows will not reproduce the full eval split
        # exactly. It should not be off by ten points either.
        if gap > 0.05:
            print(
                f"\nFAIL: {gap:.4f} apart. BigQuery and this process disagree about what the "
                "same model does with the same vectors."
            )
            passed = False

    if len(scores) > 1:
        served = scores[1]
        gap = abs(stored.coverage - served.coverage)
        print(f"  stored vectors vs re-embedded: {gap:.4f} coverage apart")
        if gap > 0.02:
            print(
                "\nFAIL: re-embedding the same prompts changes the ranking. That is a "
                "train/serve embedding difference; run scripts/check_embedding_parity.py."
            )
            passed = False

    print(
        "\nPASS: the published artifact predicts the way the run record says it does."
        if passed
        else "\nDo not route traffic to this artifact."
    )
    return passed


# --- Entry point ----------------------------------------------------------------


def fetch_recorded_coverage(project: str, dataset: str, agent: str, version: str) -> float | None:
    """What the training run claimed, for the version actually loaded."""
    sql = f"""
      SELECT coverage FROM `{project}.{dataset}.router_training_runs`
      WHERE agent = @agent AND version = @version LIMIT 1
    """
    job = bigquery.Client(project=project).query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("version", "STRING", version),
            ]
        ),
    )
    rows = list(job.result())
    return float(rows[0]["coverage"]) if rows and rows[0]["coverage"] is not None else None


def main() -> int:
    args = parse_args()
    cfg = load_config(
        bq_project=args.project,
        bq_dataset=args.dataset,
        gcs_bucket=args.bucket,
        embed_timeout_ms=TIMEOUT_MS,
        embed_breaker_threshold=args.limit + 1,
    )

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.classes)} classes")
    print(f"       {model.booster_path}")

    predictor = Predictor(model)
    rows = fetch_eval_rows(args.project, args.dataset, args.agent, cfg.embedding_dims, args.limit)
    if not rows:
        print("\nNo held-out tool-calling rows to score.", file=sys.stderr)
        return 1
    print(f"       {len(rows)} held-out rows, {len({r.tool_name for r in rows})} distinct tools")

    top_n = args.top_n or cfg.max_tools_kept
    scores = [score("stored vectors", predictor, rows, [r.vector for r in rows], top_n)]

    if not args.skip_embedding:
        print(f"\nRe-embedding {len(rows)} prompts through the serving path...")
        served = asyncio.run(embed_rows(Embedder(cfg), rows))
        scores.append(score("re-embedded", predictor, rows, served, top_n))

    recorded = fetch_recorded_coverage(
        args.project, args.dataset, args.agent, model.metadata.version
    )
    return 0 if report(scores, recorded, top_n) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    # 500 rather than 100, because the coverage comparison below is a sample against a
    # full-split number and the tolerance is fixed at 0.05. At n=100 and coverage near
    # 0.93 the sampling error alone is 2.5 points, so a healthy artifact failed this
    # check on the 100-tool surface. At n=500 it is 1.1 points. The cost is 500
    # embeddings instead of 100, which is still fractions of a cent.
    parser.add_argument("--limit", type=int, default=500, help="held-out rows to score")
    parser.add_argument("--top-n", type=int, default=None, help="defaults to max_tools_kept")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="score stored vectors only; no embedding calls, no cost",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
