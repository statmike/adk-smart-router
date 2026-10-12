#!/usr/bin/env python
"""What a local embedder costs a replica that scales to zero.

E9 answered the recall half and DESIGN §19.76 recorded it: `all-MiniLM-L6-v2` is under a point of
voted recall behind `gemini-embedding-2` at every surface from 25 to 400 tools, at 15.7 ms a query
against 170. That was a corpus study against cached vectors, so it could not touch the cost side,
and the cost side is the whole of what is left: cold-start load time, resident memory in a replica,
and image growth. This measures those three.

Why it needs its own script and not a flag on `score_local_embedder.py`
-----------------------------------------------------------------------
Because every number here is destroyed by a warm process. Import time is paid once per
interpreter, an ONNX session is constructed once, and RSS never comes back down. A script that has
already embedded 60,000 queries cannot then report a cold start. So each arm runs in a fresh
subprocess, this file re-executes itself with `--phase`, and the parent only collects.

What the numbers are for
------------------------
A Cloud Run replica at zero pays the whole of a cold start on somebody's first request. Today that
is a `google-genai` import, a client construct, and a 170 ms call. Locally it would be an
`onnxruntime` import, a session construct, and 15.7 ms. The swap is worth taking only if the load
is amortised over enough sessions to beat what it saves, so the readout is a break-even in
sessions per replica lifetime rather than a verdict.

Two honesty notes carried forward from §19.76, because they apply here too. This box is shared and
has no VNNI, so wall clocks are an ordering rather than absolute costs, and int8 models are
emulated. And a model that is downloaded on first use rather than baked into the image pays the
download on the cold start too, which is the difference between two of the rows below.

Local only. No cloud calls, no credentials, and nothing is written outside the fastembed cache.

Usage:
    uv run --group experiments scripts/measure_embedder_footprint.py
    uv run --group experiments scripts/measure_embedder_footprint.py --model BAAI/bge-small-en-v1.5
    uv run --group experiments scripts/measure_embedder_footprint.py --embeds 500
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

# The model E9 chose. The others stay reachable through --model so the ordering can be rechecked,
# but the decision this script informs is about this one.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# What the router actually embeds: one short opening message.
SAMPLE = "I need to check the status of order 88213 and then update the shipping address."

# Enough embeds to let the ONNX arena reach its steady size. Growth after a few hundred is what
# tells a one-off allocation apart from a leak, which is the thing a long-lived replica cares about.
DEFAULT_EMBEDS = 300

# Fresh processes per arm. Five is enough for a median to stop moving between runs on this box,
# and each one costs about a second.
DEFAULT_REPEATS = 5

# The remote arm's measured per-call latency, DESIGN §19.54. Used for the break-even only; this
# script makes no network calls and does not re-measure it.
REMOTE_CALL_MS = 214.0

# This script's own weights cache, deliberately not fastembed's default. Measuring the download
# means deleting the cache first, and the default is shared with `score_local_embedder.py`, which
# paid minutes to fill it. Owning a directory keeps a footprint run from costing a scoring run.
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "smart-router-footprint-cache"


def rss_kb() -> int:
    """Resident set size of this process, in KB. Linux only, and deliberately dependency-free."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS is not in /proc/self/status, so this is not Linux")


def dir_bytes(path: Path) -> int:
    """Bytes on disk under `path`, counting each blob once.

    Symlinks are skipped rather than followed. Hugging Face's cache stores weights once under
    `blobs/` and links them into a `snapshots/` tree, so following links reports 174 MB for an
    87 MB model, and 174 MB is a plausible enough number to go unquestioned.
    """
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


# --- The phases, each run in a fresh process ----------------------------------


def phase_local(model: str, embeds: int, cache: Path) -> dict[str, float]:
    """Cold start and memory for the in-process arm, from a clean interpreter."""
    baseline = rss_kb()

    started = time.perf_counter()
    from fastembed import TextEmbedding

    import_ms = (time.perf_counter() - started) * 1000
    imported = rss_kb()

    started = time.perf_counter()
    embedder = TextEmbedding(model_name=model, cache_dir=str(cache))
    construct_ms = (time.perf_counter() - started) * 1000
    loaded = rss_kb()

    started = time.perf_counter()
    list(embedder.embed([SAMPLE]))
    first_ms = (time.perf_counter() - started) * 1000
    after_first = rss_kb()

    started = time.perf_counter()
    for _ in range(embeds):
        list(embedder.embed([SAMPLE]))
    steady_ms = (time.perf_counter() - started) * 1000 / embeds
    settled = rss_kb()

    return {
        "import_ms": import_ms,
        "construct_ms": construct_ms,
        "first_embed_ms": first_ms,
        "steady_embed_ms": steady_ms,
        "rss_baseline_kb": baseline,
        "rss_imported_kb": imported,
        "rss_loaded_kb": loaded,
        "rss_after_first_kb": after_first,
        "rss_settled_kb": settled,
    }


def phase_remote() -> dict[str, float]:
    """The same cold start for what ships today, up to but not including the network call.

    No client is constructed against a project and nothing is sent. What is being measured is the
    import and the object graph, which is the part of the remote arm's cold start that a local
    embedder would replace rather than add to.
    """
    baseline = rss_kb()

    started = time.perf_counter()
    from google import genai  # noqa: F401

    import_ms = (time.perf_counter() - started) * 1000
    imported = rss_kb()

    return {
        "import_ms": import_ms,
        "rss_baseline_kb": baseline,
        "rss_imported_kb": imported,
    }


PHASES = ("local", "remote")


# --- The parent ---------------------------------------------------------------


def run_phase(name: str, model: str, embeds: int, cache: Path) -> dict[str, float]:
    """Run one phase in a fresh interpreter and read its JSON back off stdout."""
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            name,
            "--model",
            model,
            "--embeds",
            str(embeds),
            "--cache-dir",
            str(cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"the {name} phase failed:\n{result.stderr}")
    return dict(json.loads(result.stdout))


def repeat_phase(
    name: str, model: str, embeds: int, cache: Path, repeats: int
) -> dict[str, list[float]]:
    """Run a phase `repeats` times and keep every reading, not just a summary.

    One reading is not enough for any of this. An import time on a shared box moved 645 to 721 ms
    across two runs of an earlier draft, which is a tenth of the cold start and enough to move the
    break-even from 1.4 sessions to 2.5. The spread is part of the answer, so it is kept and
    printed rather than averaged away.
    """
    readings: dict[str, list[float]] = {}
    for _ in range(repeats):
        for key, value in run_phase(name, model, embeds, cache).items():
            readings.setdefault(key, []).append(value)
    return readings


PACKAGE = re.compile(r"([a-z0-9][a-z0-9._-]*) v[0-9]")


def named_packages(*extra: str) -> set[str]:
    """Every package `uv tree` resolves, with the given extra arguments."""
    result = subprocess.run(
        ["uv", "tree", "--depth", "100", *extra],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"uv tree failed:\n{result.stderr}")
    return {m.group(1) for m in PACKAGE.finditer(result.stdout)}


def measure_wheels() -> tuple[int, int]:
    """Bytes the experiments group adds to an installed environment, and how many packages.

    The list is computed rather than written down, because a hand-kept list of "what onnxruntime
    drags in" is wrong by the next lockfile. Sizes come from each distribution's own RECORD, so a
    package that shares a directory with another is still counted once and only for its own files.
    """
    added = named_packages("--group", "experiments") - named_packages()
    site = Path(sys.prefix) / "lib" / f"python3.{sys.version_info.minor}" / "site-packages"

    total = 0
    for info in site.glob("*.dist-info"):
        name = info.name.split("-")[0].replace("_", "-").lower()
        if name not in added:
            continue
        record = info / "RECORD"
        if not record.exists():
            continue
        for line in record.read_text().splitlines():
            path = site / line.split(",")[0]
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total, len(added)


def measure_download(model: str, cache: Path) -> tuple[float, int]:
    """Seconds to fetch the weights, and their size on disk, with the cache cleared first.

    A replica that bakes the model into its image never pays this and a replica that does not pays
    it on every cold start, so the two belong in different rows.
    """
    if cache.exists():
        shutil.rmtree(cache)

    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fastembed import TextEmbedding; "
            f"TextEmbedding(model_name={model!r}, cache_dir={str(cache)!r})",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    if result.returncode != 0:
        sys.exit(f"the download failed:\n{result.stderr}")
    return seconds, dir_bytes(cache)


def mb(kb: float) -> str:
    return f"{kb / 1024:,.0f} MB"


def ms_row(label: str, readings: list[float]) -> str:
    """One timing row: the median, and the spread beside it so nobody reads the median alone."""
    low, high = min(readings), max(readings)
    return f"  {label:<30} {median(readings):7.0f}   {low:.0f} to {high:.0f}"


def report(
    local: dict[str, list[float]],
    remote: dict[str, list[float]],
    weights: int,
    download: float,
    wheels: int,
    packages: int,
):
    load = [
        sum(parts)
        for parts in zip(
            local["import_ms"], local["construct_ms"], local["first_embed_ms"], strict=True
        )
    ]
    added_ms = median(load) - median(remote["import_ms"])
    saved_ms = REMOTE_CALL_MS - median(local["steady_embed_ms"])
    settled = median(local["rss_settled_kb"])
    rss_added = settled - median(remote["rss_imported_kb"])

    print("\nCold start, milliseconds from a fresh interpreter        median   range")
    print(ms_row("remote, import only", remote["import_ms"]))
    print(ms_row("local, import", local["import_ms"]))
    print(ms_row("local, session construct", local["construct_ms"]))
    print(ms_row("local, first embed", local["first_embed_ms"]))
    print(ms_row("local, all three", load))
    print(f"  {'download, if not baked in':<30} {download * 1000:7.0f}   one reading")

    print("\nResident memory")
    print(f"  interpreter alone              {mb(median(local['rss_baseline_kb'])):>10}")
    print(f"  remote, after import           {mb(median(remote['rss_imported_kb'])):>10}")
    print(f"  local, after import            {mb(median(local['rss_imported_kb'])):>10}")
    print(f"  local, session loaded          {mb(median(local['rss_loaded_kb'])):>10}")
    print(f"  local, first embed             {mb(median(local['rss_after_first_kb'])):>10}")
    print(f"  local, settled                 {mb(settled):>10}")
    print(f"  added over remote              {mb(rss_added):>10}")

    print("\nImage")
    print(f"  model weights on disk          {mb(weights / 1024):>10}")
    print(f"  wheels, {packages} added packages     {mb(wheels / 1024):>10}")
    print(f"  both                           {mb((weights + wheels) / 1024):>10}")

    print("\nPer call, milliseconds")
    print(f"  remote, DESIGN 19.54           {REMOTE_CALL_MS:8.0f}")
    print(ms_row("local, steady state", local["steady_embed_ms"]))

    print("\nBreak-even")
    if saved_ms <= 0:
        print("  the local call is not faster, so there is nothing to amortise.")
        return
    sessions = added_ms / saved_ms
    print(f"  the swap adds {added_ms:,.0f} ms to a cold start and saves {saved_ms:,.0f} ms a")
    print(f"  session, so it pays for itself after {sessions:.1f} sessions on a replica.")
    print("  The router embeds once a session, so that is sessions and not turns.")
    print("  Both terms are medians, and the cold-start one is the noisy half. Read the range")
    print("  column before treating a break-even under a session or two as a precise figure.")
    if settled > median(local["rss_after_first_kb"]) * 1.05:
        print("\n  Memory kept growing past the first embed. Read the settled row before")
        print("  sizing a replica; the arena has not finished allocating at one call.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the ONNX model to measure")
    parser.add_argument(
        "--embeds", type=int, default=DEFAULT_EMBEDS, help="embeds after the first, for the arena"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="fresh processes per arm. The cold-start numbers are noisy on a shared box and the "
        "break-even is a difference of two of them, so one reading is not an answer",
    )
    parser.add_argument("--phase", choices=sorted(PHASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"where to keep the weights, default {DEFAULT_CACHE}. This script deletes it to time "
        "the download, so do not point it at a cache you want to keep",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="leave the cache alone and report no download time. Use this on a metered "
        "connection, or to re-read the other rows without a re-fetch",
    )
    args = parser.parse_args()

    if args.phase == "local":
        print(json.dumps(phase_local(args.model, args.embeds, args.cache_dir)))
        return 0
    if args.phase == "remote":
        print(json.dumps(phase_remote()))
        return 0

    print(f"measuring {args.model}, {args.embeds} embeds after the first")
    print(f"{args.repeats} fresh processes per arm, because a warm one cannot report a cold start")

    download = 0.0
    weights = dir_bytes(args.cache_dir)
    if not args.skip_download:
        print(f"\nclearing {args.cache_dir} and re-fetching")
        download, weights = measure_download(args.model, args.cache_dir)

    print("running the local arm")
    local = repeat_phase("local", args.model, args.embeds, args.cache_dir, args.repeats)
    print("running the remote arm")
    remote = repeat_phase("remote", args.model, args.embeds, args.cache_dir, args.repeats)

    print("sizing the wheels the experiments group adds")
    wheels, packages = measure_wheels()

    report(local, remote, weights, download, wheels, packages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
