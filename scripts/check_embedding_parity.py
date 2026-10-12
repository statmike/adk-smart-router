#!/usr/bin/env python
"""Build-order task 7 — does serving embed a prompt the same way training did?

Training embeds with BigQuery `AI.EMBED`. Serving embeds with `google-genai`
through `embedding.py`. If those two produce different vectors, the model clears
every offline gate and predicts garbage in production, and nothing anywhere
raises. It is the quietest failure in the system, and no test of the model can
see it, because both halves are individually correct.

`tests/test_embedding.py` proves the serving code path: truncate before
normalize, `vertexai=True`, the right width. It cannot prove the vectors agree,
because it runs against a fake client. This script is the other half. It reads
prompts whose vectors are already sitting in the training features table, embeds
the same strings through the real serving path, and compares.

Three things it can catch that unit tests cannot:

  * a different endpoint. `AI.EMBED` resolves through a BigQuery connection in
    the `us` multi-region; `embedding.py` calls Vertex at `vertex_location`.
    Nothing forces those to be the same model build, and the first run of this
    script found `vertex_location` pointing at a region where the model does
    not exist at all (DESIGN.md 18.7).
  * a different normalization. `AI.EMBED` is asked for 768 dimensions directly
    and its output is stored unmodified, while `embedding.py` truncates and then
    L2-normalizes. That is only a no-op while `AI.EMBED` returns unit vectors,
    which is measured here rather than assumed.
  * silent model drift. `gemini-embedding-2` is one name over a serving stack
    that can be updated underneath a stored corpus.

Cost is fractions of a cent: fifty embedding calls on a text endpoint.
Authentication is ADC only. No service-account key is created, read or accepted.

Usage:
    uv run python scripts/check_embedding_parity.py
    uv run python scripts/check_embedding_parity.py --limit 200 --min-cosine 0.9999
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through
# `pythonpath` in pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.embedding import Embedder  # noqa: E402

# Same default as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# Two vectors from the same model on the same text should agree to floating-point
# noise. 0.9999 leaves room for that and for a reduced-precision transport, and
# still fails loudly on a different model build or a normalization mismatch.
DEFAULT_MIN_COSINE = 0.9999

# This script's own per-call timeout, not the serving one. See --timeout-ms.
DEFAULT_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class Pair:
    """One prompt, its stored training vector, and the vector serving produces."""

    invocation_id: str
    prompt: str
    trained: list[float]
    served: list[float]

    @property
    def cosine(self) -> float:
        return cosine_similarity(self.trained, self.served)

    @property
    def max_abs_delta(self) -> float:
        return max(abs(a - b) for a, b in zip(self.trained, self.served, strict=True))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def fetch_trained(project: str, dataset: str, agent: str, dims: int, limit: int) -> list[tuple]:
    """Read prompts and their stored vectors out of the training features table.

    Ordered by a hash of the invocation id rather than by time. The table is
    written oldest-first, so `LIMIT` without an order would sample one corner of
    one day and would miss a drift that started midway through the corpus.
    """
    table = f"{project}.{dataset}.router_features_{agent}"
    sql = f"""
      SELECT invocation_id, prompt_text, v
      FROM `{table}`
      WHERE ARRAY_LENGTH(v) = @dims AND prompt_text IS NOT NULL
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
    return [(r["invocation_id"], r["prompt_text"], list(r["v"])) for r in job.result()]


async def embed_all(embedder: Embedder, rows: list[tuple]) -> tuple[list[Pair], list[float]]:
    """Embed every prompt through the real serving path, sequentially.

    Sequential on purpose. The point is to exercise `Embedder` exactly as the hot
    path does, and a burst of concurrent calls would additionally be testing the
    endpoint's rate limiter.

    Returns the pairs and the per-call latencies. The latencies matter because
    the production `embed_timeout_ms` is a hot-path budget, and a script that
    silently ran under a relaxed one would hide an endpoint too slow to serve.
    """
    pairs: list[Pair] = []
    latencies: list[float] = []
    for index, (invocation_id, prompt, trained) in enumerate(rows, start=1):
        started = time.monotonic()
        served = await embedder.embed(prompt)
        latencies.append((time.monotonic() - started) * 1000.0)
        if served is None:
            # Embedder never raises; it returns None and logs. A None here is a
            # failed comparison, not a passed one, so say so and keep going.
            print(f"  [{index}/{len(rows)}] {invocation_id}: serving returned no vector")
            continue
        pairs.append(Pair(invocation_id, prompt, trained, served))
        if index % 10 == 0:
            print(f"  [{index}/{len(rows)}] embedded")
    return pairs, latencies


def report_latency(latencies: list[float], hot_path_budget_ms: int) -> None:
    """Print call latency against the production timeout. Never gates.

    This script relaxes `embed_timeout_ms` so that a slow endpoint fails as a
    slow endpoint rather than as a parity mismatch. That relaxation would hide
    something worth knowing, so the real numbers are printed beside the budget
    the hot path actually enforces.
    """
    if not latencies:
        return
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(f"\n  latency  p50 {p50:.0f} ms   p95 {p95:.0f} ms   max {ordered[-1]:.0f} ms")
    print(f"           production embed_timeout_ms is {hot_path_budget_ms}")
    if p50 > hot_path_budget_ms:
        over = sum(1 for value in ordered if value > hot_path_budget_ms)
        print(
            f"\n  NOTE: {over}/{len(ordered)} calls exceeded the hot-path budget, median "
            f"included. At that rate the breaker opens and the router serves every turn "
            f"unrouted. Parity is unaffected; serving is not. Raise embed_timeout_ms or "
            f"accept that routing rarely fires."
        )


def report(pairs: list[Pair], rows: list[tuple], min_cosine: float) -> bool:
    """Print the comparison and return whether it passed."""
    if not pairs:
        print("\nFAIL: no prompt could be embedded through the serving path.")
        return False

    cosines = sorted(p.cosine for p in pairs)
    trained_norms = [l2_norm(p.trained) for p in pairs]
    served_norms = [l2_norm(p.served) for p in pairs]
    worst = min(pairs, key=lambda p: p.cosine)

    print(f"\nCompared {len(pairs)} of {len(rows)} prompts.\n")
    print(
        f"  cosine  min {cosines[0]:.9f}   p50 {statistics.median(cosines):.9f}   "
        f"max {cosines[-1]:.9f}"
    )
    print(f"  |delta| max across all components  {max(p.max_abs_delta for p in pairs):.3e}")
    print(f"  L2 norm training  min {min(trained_norms):.9f}  max {max(trained_norms):.9f}")
    print(f"  L2 norm serving   min {min(served_norms):.9f}  max {max(served_norms):.9f}")

    passed = cosines[0] >= min_cosine
    missing = len(rows) - len(pairs)
    if missing:
        print(f"\n  {missing} prompt(s) produced no serving vector. Counted as a failure.")
        passed = False

    if passed:
        print(f"\nPASS: every pair is at or above {min_cosine}.")
        return True

    print(f"\nFAIL: worst cosine {worst.cosine:.9f} is below {min_cosine}.")
    print(f"  invocation {worst.invocation_id}")
    print(f"  prompt     {worst.prompt[:110]!r}")
    print("\nThe two sides are not embedding the same way. Check, in this order:")
    print("  1. vertex_location against the region the BigQuery connection resolves to")
    print("  2. embedding_model and embedding_dims in config.py against train_router.sql")
    print("  3. whether AI.EMBED still returns unit-norm vectors (the norms above)")
    print("\nDo not publish a model until this passes. DESIGN.md 9.1.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument(
        "--dataset", default=os.environ.get("SMART_ROUTER_BQ_DATASET", DEFAULT_DATASET)
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--limit", type=int, default=50, help="Prompts to compare. Default 50.")
    parser.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=(
            "Per-call timeout for this script. Deliberately far above the production "
            "embed_timeout_ms: under the hot-path budget a cold client trips the breaker "
            "and every prompt reads as a parity failure when nothing is wrong with the "
            "vectors. The real latencies are reported separately."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset)
    hot_path_budget_ms = cfg.embed_timeout_ms
    # Also lift the breaker out of the way. Five slow calls would otherwise open it
    # and every prompt after that returns None without being tried.
    cfg = load_config(
        bq_project=args.project,
        bq_dataset=args.dataset,
        embed_timeout_ms=args.timeout_ms,
        embed_breaker_threshold=len(range(args.limit)) + 1,
    )
    print(f"training side  {args.project}.{args.dataset}.router_features_{args.agent}")
    print(
        f"serving side   {cfg.embedding_model} @ {cfg.vertex_location}, "
        f"{cfg.embedding_dims} dims, task_type={cfg.task_type}"
    )

    rows = fetch_trained(args.project, args.dataset, args.agent, cfg.embedding_dims, args.limit)
    if not rows:
        print("\nNothing to compare: the features table holds no vectors of that width.")
        return 1

    print(f"\nEmbedding {len(rows)} prompts through the serving path...")
    pairs, latencies = asyncio.run(embed_all(Embedder(cfg), rows))
    report_latency(latencies, hot_path_budget_ms)
    return 0 if report(pairs, rows, args.min_cosine) else 1


if __name__ == "__main__":
    sys.exit(main())
