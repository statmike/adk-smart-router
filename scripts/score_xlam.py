#!/usr/bin/env python
"""Does the routing rule work on somebody else's data?

Every recall number this project publishes comes from a corpus it wrote itself. DESIGN.md
19.24 already showed how much that flatters us: the same rule scores 0.9232 when the split
blocks the session and 0.8322 when it blocks the authored scenario, so about half the
measured lift was the model recognising our own 106 intents. That fixed the *split*. It did
not answer the question underneath it, which is whether the rule works at all on traffic
nobody here designed.

This script answers that against `xlam-function-calling-60k` (Salesforce, CC-BY-4.0,
arXiv:2406.18518): 60,000 natural-language queries, each labelled with the function calls
that answer it, over 3,179 distinct tools drawn largely from real RapidAPI endpoints. It is
about as far from our corpus as a public dataset gets.

    corpus-20260904                     xlam-function-calling-60k
    1,700 sessions, 106 intents         60,000 queries, 12,704 distinct candidate sets
    100 tools, written for this test    3,179 tools, mostly real third-party APIs
    static top-11 covers 75.3%          static top-11 covers 5.4%
    mean 2.6 tools per session          mean 1.4 tools per query

That last pair of rows is the important one, and it cuts both ways. xLAM's tool
distribution is almost flat, so the static baseline this project has been beating is
nearly worthless there and any lift over it would look spectacular for the wrong reason.
Flat also makes the prediction genuinely harder: there is no popular core to fall back on.

So this script does not report lift over a degenerate baseline as its headline. It reports
**how wide a menu has to be to reach a given recall**, which is the economic question
anyway, on surfaces of four different widths.

The surfaces
------------
xLAM is not one agent and cannot honestly be sliced into agents: every row carries a
candidate tool list of about three, and the co-occurrence graph over those lists collapses
into one component of 3,510 tools, so there is no grouping to find that we would not have
invented. Instead the surfaces are nested by popularity, which is a rule with no free
parameters:

    top 25      the widest surface where a static menu is still a real competitor
    top 100     matched to the reference agent in DESIGN.md 19.15, for a like-for-like read
    top 400     wider than any agent this project has measured
    everything  all 3,179, the stress case

A row belongs to a surface when *every* tool it called is on it. A row calling something
off-surface is not a miss, it is a different agent, and scoring it would measure the
sampling rule rather than the router.

The regimes
-----------
Two of `score_unseen_intent.py`'s four transfer here. Blocking the row itself is
leave-one-out. Blocking the normalised opening removes the 2,118 rows that share their
query text with another row. There is no scenario and no persona, because nobody authored
these; that absence is the point of the exercise.

Everything else is imported rather than reimplemented: `menus.popularity`, `menus.choose`,
`menus.pad` and `menus.rank` are the shipped code, and `votes`, `voted_menu`, `recall` and
`fit_pools` come from `score_unseen_intent.py` so the two tables are computed by the same
arithmetic and can be read side by side.

Cost and credentials
--------------------
One embedding pass over 60,000 queries through the same BigQuery `AI.EMBED` call the
training pipeline uses, at the same endpoint and the same width, so what is measured is our
embedding and not a stand-in. About 1.6M tokens. The vectors are cached in a BigQuery table
and in a local `.npy`, so a second run costs nothing. Authentication is Application Default
Credentials.

Usage:
    uv run python scripts/score_xlam.py --fetch          # download the corpus, ~96 MB
    uv run python scripts/score_xlam.py                  # dry run: prints the plan
    uv run python scripts/score_xlam.py --confirm        # embeds, then scores
    uv run python scripts/score_xlam.py --menu-size 25   # after the vectors are cached
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery
from score_unseen_intent import (
    Case,
    Menu,
    Regime,
    Row,
    Vote,
    fit_pools,
    fold_of,
    normalise,
    recall,
    unit,
    voted_menu,
    votes,
)

from adk_smart_router.config import load_config
from adk_smart_router.menus import pad, popularity, rank

# Salesforce gates the canonical repo behind an access form. This is a byte-identical public
# mirror of the same file, which CC-BY-4.0 expressly permits; the gate is Salesforce's access
# control and not a licence term. Attribution stays with Salesforce (arXiv:2406.18518).
SOURCE_URL = (
    "https://huggingface.co/datasets/lockon/xlam-function-calling-60k"
    "/resolve/main/xlam_function_calling_60k.json"
)
CACHE_DIR = Path(".xlam")
CORPUS_PATH = CACHE_DIR / "xlam.json"
VECTOR_PATH = CACHE_DIR / "vectors.npy"
ID_PATH = CACHE_DIR / "ids.json"

STAGING_TABLE = "xlam_queries"
EMBEDDING_TABLE = "xlam_embeddings"

# Matched to score_unseen_intent.py so the two tables are comparable.
NEIGHBOURS = 15
MENU_COUNT = 24
POOL_FOLDS = 5

# Nested by call count, no free parameters. See the module docstring.
SURFACES = (25, 100, 400, 0)  # 0 means every tool

REGIMES = (
    Regime("block the row", lambda case: case.session_id),
    Regime("block the opening", lambda case: normalise(case.opening)),
)

# Every stage below announces itself. An earlier version of this script printed nothing
# between the embedding and the first table, and then spent two hours in a pool fit that
# could not have finished in a week. Nobody could tell, because stdout redirected to a file
# is block buffered and no block had filled. `DESIGN.md` 19.41.
STARTED = time.monotonic()


def say(message: str) -> None:
    """A progress line, stamped with elapsed time and flushed so a redirect shows it."""
    print(f"  [{time.monotonic() - STARTED:6.0f}s] {message}", flush=True)


# --- The corpus ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    """One xLAM row, reduced to the two fields a router would see and be graded on."""

    row_id: str
    text: str
    tools: Menu


def download(path: Path) -> None:
    """Fetch the corpus once. 96 MB, no credentials, no cost."""
    path.parent.mkdir(exist_ok=True)
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, path)  # noqa: S310 - constant https URL
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


def load_corpus(path: Path) -> list[Query]:
    """Read the corpus into queries, dropping rows we cannot grade.

    `answers` and `tools` are JSON *strings* inside the JSON, which is the dataset's own
    shape and not a mistake here. A row whose answer list is empty is kept: a query that
    correctly calls nothing is real traffic and a router that declares a menu for it is
    still spending tokens.
    """
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run with --fetch first.")

    queries = []
    for raw in json.loads(path.read_text()):
        text = raw["query"].strip()
        if not text:
            continue
        called = {call["name"] for call in json.loads(raw["answers"])}
        queries.append(Query(str(raw["id"]), text, frozenset(called)))
    return queries


def surface_of(queries: Sequence[Query], width: int) -> Menu:
    """The `width` most-called tools, or every tool when `width` is 0."""
    ranked = popularity([sorted(q.tools) for q in queries])
    return frozenset(ranked if width == 0 else ranked[:width])


# --- Embedding ------------------------------------------------------------------------------


def estimate_tokens(queries: Sequence[Query]) -> int:
    """Roughly what the embedding pass will bill, at the 4 chars/token English rule."""
    return sum(len(q.text) for q in queries) // 4


def embed(
    queries: Sequence[Query], project: str, dataset: str, model: str, dims: int, connection: str
) -> tuple[list[str], np.ndarray]:
    """Embed every query through BigQuery `AI.EMBED`, and cache the result locally.

    The same endpoint, width and `model_params` key the training pipeline uses. Going
    through BigQuery rather than `embedding.Embedder` is deliberate: 60,000 sequential
    calls at 170 ms each is three hours, and `check_embedding_parity.py` already proves the
    two paths agree bit for bit, so there is nothing to learn from doing it the slow way.
    """
    if VECTOR_PATH.exists() and ID_PATH.exists():
        print(f"  reusing cached vectors at {VECTOR_PATH}")
        return json.loads(ID_PATH.read_text()), np.load(VECTOR_PATH)

    client = bigquery.Client(project=project)
    staging = f"{project}.{dataset}.{STAGING_TABLE}"
    target = f"{project}.{dataset}.{EMBEDDING_TABLE}"

    print(f"  loading {len(queries):,} queries into {staging}")
    client.load_table_from_json(
        [{"row_id": q.row_id, "query_text": q.text} for q in queries],
        staging,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            schema=[
                bigquery.SchemaField("row_id", "STRING"),
                bigquery.SchemaField("query_text", "STRING"),
            ],
        ),
    ).result()

    # A width filter rather than a status check, for the reason preflight.sql spells out:
    # AI.EMBED ignores a model_params key it does not recognise, returns the endpoint's
    # native 3072-wide vector, and reports status = '' either way.
    print(f"  embedding into {target}, this is the part that spends")
    client.query(
        f"""
        CREATE OR REPLACE TABLE `{target}` AS
        SELECT row_id, e.result AS v
        FROM (
          SELECT
            row_id,
            AI.EMBED(
              content       => query_text,
              endpoint      => '{model}',
              connection_id => '{connection}',
              model_params  => JSON '{{"outputDimensionality": {dims}}}'
            ) AS e
          FROM `{staging}`
        )
        WHERE e.status = '' AND ARRAY_LENGTH(e.result) = {dims}
        """
    ).result()

    rows = list(client.query(f"SELECT row_id, v FROM `{target}`").result())
    if len(rows) < len(queries):
        print(f"  {len(queries) - len(rows):,} queries came back unembedded and are dropped")

    ids = [row["row_id"] for row in rows]
    vectors = unit(np.array([row["v"] for row in rows], dtype=np.float32))
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(VECTOR_PATH, vectors)
    ID_PATH.write_text(json.dumps(ids))
    print(f"  cached {len(ids):,} vectors at {VECTOR_PATH}")
    return ids, vectors


# --- Scoring ---------------------------------------------------------------------------------

# 60,000 squared float32s is 14 GB, so the similarity matrix is never materialised. Rows are
# scored in blocks against the whole corpus, which is 60,000 x BLOCK and fits comfortably.
BLOCK = 512


def score_surface(
    cases: Sequence[Case],
    vectors: np.ndarray,
    regime: Regime,
    ranked: Sequence[str],
    pools: Sequence[Sequence[Menu]],
    size: int,
    neighbours: int,
) -> Row:
    """One regime's scores over every case, in blocks of `BLOCK` rows at a time."""
    keys = np.array([regime.key(case) for case in cases])
    tools = [case.tools for case in cases]
    row = Row(regime.name)

    say(f"scoring {len(cases):,} cases, {regime.name}")
    for start in range(0, len(cases), BLOCK):
        if start and start % (BLOCK * 20) == 0:
            say(f"  {start:,} of {len(cases):,}")
        block = vectors[start : start + BLOCK] @ vectors.T
        for offset, similarity in enumerate(block):
            index = start + offset
            weighted = votes(similarity, keys == keys[index], tools, neighbours)
            voted = voted_menu(weighted, ranked, size)
            pooled = rank(
                [Vote(t, w) for t, w in weighted.items()], pools[fold_of(keys[index], len(pools))]
            )
            row.voted += recall(voted, tools[index])
            row.padded += recall(frozenset(pad(sorted(voted), ranked, size)), tools[index])
            row.pooled += recall(pooled or frozenset(), tools[index])
            row.no_miss += float(tools[index] <= voted)
            row.width += len(voted)

    n = len(cases)
    return Row(
        row.name, row.voted / n, row.padded / n, row.pooled / n, row.no_miss / n, row.width / n
    )


def static_recall(menu: Menu, cases: Sequence[Case]) -> tuple[float, float]:
    """The popularity baseline: one menu for everybody. Mean recall and no-miss rate."""
    scores = [recall(menu, case.tools) for case in cases]
    return sum(scores) / len(scores), sum(1.0 for c in cases if c.tools <= menu) / len(cases)


# --- Reporting ---------------------------------------------------------------------------------


def report(
    width: int,
    tools: int,
    cases: Sequence[Case],
    static: tuple[float, float],
    rows: Sequence[Row],
    size: int,
) -> None:
    label = "every tool" if width == 0 else f"top {width}"
    print(f"\n### surface: {label} ({tools:,} tools, {len(cases):,} queries)\n")
    print("| arm | recall | no miss | menu width | prune |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| static top-{size} | {static[0]:.4f} | {static[1]:.4f} | {size} |"
        f" {1 - size / tools:.1%} |"
    )
    for row in rows:
        print(
            f"| voted, {row.name} | {row.voted:.4f} | {row.no_miss:.4f} | {row.width:.1f} |"
            f" {1 - row.width / tools:.1%} |"
        )
        print(
            f"| padded to {size}, {row.name} | {row.padded:.4f} | — | {size} |"
            f" {1 - size / tools:.1%} |"
        )
        print(f"| pooled, {row.name} | {row.pooled:.4f} | — | {size} | {1 - size / tools:.1%} |")


def main() -> int:
    args = parse_args()
    if args.fetch:
        download(CORPUS_PATH)

    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset)
    size = args.menu_size or cfg.max_tools_kept
    queries = load_corpus(CORPUS_PATH)
    tokens = estimate_tokens(queries)

    print(f"xlam-function-calling-60k: {len(queries):,} queries")
    print(f"  distinct tools called   {len(surface_of(queries, 0)):,}")
    print(f"  menu size               {size} (max_tools_kept)")
    print(f"  embedding               {cfg.embedding_model} at {cfg.embedding_dims} dims")
    print(f"  estimated               {tokens:,} tokens, about ${tokens * 0.15 / 1e6:.2f}")

    if not args.confirm and not VECTOR_PATH.exists():
        print("\nNothing was called. Re-run with --confirm to embed and score.")
        return 2

    connection = cfg.bq_connection or f"{args.project}.us.smart_router_conn"
    ids, vectors = embed(
        queries, args.project, args.dataset, cfg.embedding_model, cfg.embedding_dims, connection
    )

    by_id = {q.row_id: q for q in queries}
    embedded = [by_id[i] for i in ids if i in by_id]
    lookup = {q.row_id: n for n, q in enumerate(embedded)}
    vectors = vectors[[n for n, i in enumerate(ids) if i in by_id]]

    for width in SURFACES:
        surface = surface_of(embedded, width)
        keep = [q for q in embedded if q.tools and q.tools <= surface]
        if len(keep) < NEIGHBOURS * 2:
            print(f"\nskipping the top-{width} surface: only {len(keep)} queries fit it.")
            continue

        cases = [Case(q.row_id, q.tools, q.text, "", "") for q in keep]
        subset = vectors[[lookup[q.row_id] for q in keep]]
        ranked = popularity([sorted(c.tools) for c in cases])
        static = static_recall(frozenset(pad([], ranked, size)), cases)
        say(f"surface {width or 'all'}: {len(surface):,} tools, {len(cases):,} queries")

        rows = []
        for regime in REGIMES:
            say(f"fitting {args.folds} pools, {regime.name}")
            pools = fit_pools(cases, regime, size, args.menu_count, args.folds)
            rows.append(score_surface(cases, subset, regime, ranked, pools, size, args.neighbours))
        report(width, len(surface), cases, static, rows, size)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--fetch", action="store_true", help="download the corpus and exit early")
    parser.add_argument("--confirm", action="store_true", help="required to spend on embeddings")
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--menu-size", type=int, default=None, help="defaults to max_tools_kept")
    parser.add_argument("--menu-count", type=int, default=MENU_COUNT)
    parser.add_argument("--folds", type=int, default=POOL_FOLDS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
