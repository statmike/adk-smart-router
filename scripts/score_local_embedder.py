#!/usr/bin/env python
"""Could a CPU model on the box replace the remote embedding call?

Every routed session pays one network hop before the agent's first token. `Embedder` calls
Vertex, waits, and prunes on what comes back, and DESIGN.md 19.21 measured that hop at a
median 170 ms. It is the only remote dependency on the critical path, it is the thing that
timed out 34 times in 19.67, and it is the reason the router needs a breaker at all. A model
running in-process would delete all three problems at once.

The question is whether it can. The router does not need a good embedding, it needs an
embedding whose *neighbourhoods* are the same, because the prediction is a similarity-weighted
vote over the nearest sessions. So this scores the local candidates the only way that
matters: swap the vectors, change nothing else, and read the recall.

The fixture is `score_xlam.py`'s. Same corpus, same surfaces, same regimes, same menu pools,
same `menus.choose`, `menus.pad` and `menus.rank`. Both arms are scored by the same function
on the same cases, so the only difference between the rows of any table below is which model
produced the 60,000 vectors underneath them. The remote arm reuses the vectors already cached
at `.xlam/vectors.npy`, so this script spends nothing at Vertex and nothing at BigQuery.

Menu pools are fitted once per surface and shared across the arms. They are built from tool
sets rather than from vectors, so re-fitting them per embedder would cost minutes and change
nothing.

What "good enough" means
------------------------
Recall is the gate and it is not a close call. A local embedder that is a point behind is
worth taking for the latency; one that is five points behind is a worse product sold as a
faster one, because 19.66 measured what a recall gap does to the user. Latency and money are
tiebreakers, reported below the recall tables and never above them.

The candidates are three ONNX models spanning the size range an adopter would consider: the
smallest credible one, the strongest tiny retrieval model, and one matched to the remote
endpoint's own width so the comparison is not confounded by dimensionality.

Usage:
    uv run --group experiments python scripts/score_local_embedder.py
    uv run --group experiments python scripts/score_local_embedder.py --surfaces 25 100 400
    uv run --group experiments python scripts/score_local_embedder.py --models BAAI/bge-small-en
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from score_unseen_intent import Case, Row, fit_pools, recall, unit
from score_xlam import (
    CACHE_DIR,
    CORPUS_PATH,
    ID_PATH,
    MENU_COUNT,
    NEIGHBOURS,
    POOL_FOLDS,
    REGIMES,
    VECTOR_PATH,
    Query,
    load_corpus,
    say,
    score_surface,
    surface_of,
)

from adk_smart_router.config import load_config
from adk_smart_router.menus import pad, popularity

# The remote arm, for labelling. Its vectors come from the cache rather than from a call.
REMOTE_DIMS = 768

# Three ONNX models, chosen to span the decision rather than to win it. MiniLM is the floor:
# 90 MB, and if it holds there is no argument for anything larger. bge-small is the strongest
# model at that size on retrieval benchmarks. bge-base matches the remote endpoint's 768 dims,
# so if the small ones lose we can tell whether they lost on width or on quality.
MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
)

# Enough to make the median stable without making the script slow. These are single-query
# calls, not batches, because one query at a time is the shape the critical path has.
LATENCY_SAMPLES = 200

# The surfaces worth reading. `score_xlam.py` also scores all 3,179 tools, which is 60,000
# cases against 60,000 and hours of Python per arm. Top 100 is the like-for-like read against
# every other number in this project, and 400 is already wider than any agent measured here.
SURFACES = (25, 100, 400)

BATCH = 256


def cache_path(model: str) -> Path:
    """Where one model's vectors live. The slug keeps the vendor prefix out of the filename."""
    return CACHE_DIR / f"vectors-{model.split('/')[-1]}.npy"


def embed_local(model: str, texts: Sequence[str]) -> tuple[np.ndarray, float]:
    """Every query through one ONNX model, cached. Returns the vectors and the seconds spent.

    A cache hit reports 0.0 seconds, which the throughput line then declines to print rather
    than reporting a made-up number.
    """
    path = cache_path(model)
    if path.exists():
        print(f"  reusing cached vectors at {path}")
        return np.load(path), 0.0

    from fastembed import TextEmbedding

    say(f"loading {model}")
    embedder = TextEmbedding(model_name=model)
    say(f"embedding {len(texts):,} queries")
    started = time.monotonic()
    vectors = unit(np.array(list(embedder.embed(texts, batch_size=BATCH)), dtype=np.float32))
    elapsed = time.monotonic() - started
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, vectors)
    say(f"cached {vectors.shape[0]:,} x {vectors.shape[1]} vectors at {path}")
    return vectors, elapsed


def single_query_latency(model: str, texts: Sequence[str], samples: int) -> float:
    """Median milliseconds to embed one query, warm, which is what the critical path pays.

    Batched throughput is the wrong number for this decision. The router embeds one message
    and blocks on it, so the tail of a batch is irrelevant and the per-call cost is not the
    batch cost divided by the batch.
    """
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=model)
    list(embedder.embed(texts[:8]))  # warm the session and the tokenizer
    timings = []
    for text in texts[:samples]:
        started = time.perf_counter()
        list(embedder.embed([text]))
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def score_arm(
    name: str,
    dims: int,
    cases: Sequence[Case],
    vectors: np.ndarray,
    pools: dict[str, list[tuple[frozenset[str], ...]]],
    ranked: Sequence[str],
    size: int,
    neighbours: int,
) -> list[tuple[str, int, Row]]:
    """One embedder's rows for one surface, one per regime."""
    rows = []
    for regime in REGIMES:
        say(f"{name}: scoring {len(cases):,} cases, {regime.name}")
        rows.append(
            (
                name,
                dims,
                score_surface(cases, vectors, regime, ranked, pools[regime.name], size, neighbours),
            )
        )
    return rows


def report(
    width: int, tools: int, cases: Sequence[Case], static: float, rows: Sequence, size: int
) -> None:
    """One table per surface. The remote arm is first so every other row reads as a delta."""
    label = "every tool" if width == 0 else f"top {width}"
    print(f"\n### surface: {label} ({tools:,} tools, {len(cases):,} queries)\n")
    print("| embedder | dims | regime | voted | pooled | no miss | width |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    print(f"| static top-{size} | — | — | {static:.4f} | — | — | {size} |")
    for name, dims, row in rows:
        print(
            f"| {name} | {dims} | {row.name} | {row.voted:.4f} | {row.pooled:.4f} |"
            f" {row.no_miss:.4f} | {row.width:.1f} |"
        )


def main() -> int:
    args = parse_args()
    cfg = load_config()
    size = args.menu_size or cfg.max_tools_kept

    if not VECTOR_PATH.exists() or not ID_PATH.exists():
        raise SystemExit(
            f"{VECTOR_PATH} is missing. Run score_xlam.py --confirm first; this script "
            "compares against those vectors and will not pay to make them."
        )

    queries = load_corpus(CORPUS_PATH)
    ids = json.loads(ID_PATH.read_text())
    remote = np.load(VECTOR_PATH)

    # The remote pass dropped whatever came back unembedded, so its ids are the corpus the
    # comparison runs on. Every local arm embeds exactly these texts in exactly this order.
    by_id = {q.row_id: q for q in queries}
    embedded: list[Query] = [by_id[i] for i in ids if i in by_id]
    remote = remote[[n for n, i in enumerate(ids) if i in by_id]]
    lookup = {q.row_id: n for n, q in enumerate(embedded)}
    texts = [q.text for q in embedded]

    print(f"xlam-function-calling-60k: {len(embedded):,} queries embedded by both arms")
    print(f"  menu size    {size} (max_tools_kept)")
    print(f"  remote arm   {cfg.embedding_model} at {remote.shape[1]} dims, from cache")
    print(f"  local arms   {', '.join(args.models)}")

    arms: list[tuple[str, np.ndarray]] = [(cfg.embedding_model, remote)]
    throughput: dict[str, float] = {}
    for model in args.models:
        vectors, elapsed = embed_local(model, texts)
        if vectors.shape[0] != len(embedded):
            raise SystemExit(f"{model} returned {vectors.shape[0]} vectors, not {len(embedded)}")
        if elapsed:
            throughput[model] = len(embedded) / elapsed
        arms.append((model, vectors))

    for width in args.surfaces:
        surface = surface_of(embedded, width)
        keep = [q for q in embedded if q.tools and q.tools <= surface]
        if len(keep) < args.neighbours * 2:
            print(f"\nskipping the top-{width} surface: only {len(keep)} queries fit it.")
            continue

        cases = [Case(q.row_id, q.tools, q.text, "", "") for q in keep]
        index = [lookup[q.row_id] for q in keep]
        ranked = popularity([sorted(c.tools) for c in cases])
        static_menu = frozenset(pad([], ranked, size))
        static = sum(recall(static_menu, c.tools) for c in cases) / len(cases)

        say(f"surface {width or 'all'}: {len(surface):,} tools, {len(cases):,} queries")
        pools = {}
        for regime in REGIMES:
            say(f"fitting {args.folds} pools, {regime.name}")
            pools[regime.name] = fit_pools(cases, regime, size, args.menu_count, args.folds)

        rows: list[tuple[str, int, Row]] = []
        for name, vectors in arms:
            rows += score_arm(
                name, vectors.shape[1], cases, vectors[index], pools, ranked, size, args.neighbours
            )
        report(width, len(surface), cases, static, rows, size)

    print("\n### the tiebreakers\n")
    print("| embedder | median ms, one query | queries/second, batched |")
    print("| --- | --- | --- |")
    print(f"| {cfg.embedding_model} | 170 (DESIGN.md 19.21) | — |")
    for model in args.models:
        rate = f"{throughput[model]:,.0f}" if model in throughput else "cached"
        print(f"| {model} | {single_query_latency(model, texts, args.latency):.1f} | {rate} |")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--surfaces", nargs="+", type=int, default=list(SURFACES))
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--menu-size", type=int, default=None, help="defaults to max_tools_kept")
    parser.add_argument("--menu-count", type=int, default=MENU_COUNT)
    parser.add_argument("--folds", type=int, default=POOL_FOLDS)
    parser.add_argument("--latency", type=int, default=LATENCY_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
