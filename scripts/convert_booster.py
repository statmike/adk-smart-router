#!/usr/bin/env python
"""Convert a BQML-exported booster from XGBoost's legacy binary format to UBJSON.

`EXPORT MODEL` writes `model.bst` in the pre-1.0 XGBoost binary format. XGBoost
removed the ability to read that format in 2.1, so the version this project
serves with cannot open it at all — not "with a warning", not "degraded", it
raises `Invalid model format` (DESIGN.md 18.9).

The only thing that can still read it is an XGBoost old enough to predate the
removal, which is not a dependency worth carrying into a serving process. So the
conversion happens once, at publish time, in a throwaway environment:

    uv run --isolated --no-project --with xgboost==1.7.6 \
        python scripts/convert_booster.py model.bst model.ubj report.json

`scripts/publish_router.py` runs exactly that and then verifies the result with
the *installed* XGBoost, which is the one serving will use.

`report.json` describes what was inside the booster, so the caller can check the
conversion against the run record instead of trusting it. It is a third path
rather than stdout because XGBoost's C++ layer prints its own warnings there —
loading a pre-1.0 model always emits one — and a report that shares a channel
with library chatter is a report that eventually fails to parse.

This file is deliberately standalone. It runs in an environment where the only
thing installed is XGBoost, so it must not import anything from this project,
and it takes every path it touches as an argument.
"""

from __future__ import annotations

import json
import sys

import xgboost


def convert(source: str, destination: str) -> dict[str, object]:
    """Read a legacy booster, write it as UBJSON, and describe what was in it."""
    booster = xgboost.Booster()
    booster.load_model(source)
    booster.save_model(destination)

    config = json.loads(booster.save_config())
    learner = config["learner"]
    return {
        "xgboost_version": xgboost.__version__,
        "num_features": int(booster.num_features()),
        "num_class": int(learner["learner_model_param"]["num_class"]),
        "objective": learner["objective"]["name"],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(f"usage: {argv[0]} <source.bst> <destination.ubj> <report.json>", file=sys.stderr)
        return 2
    # A conversion failure must not be mistaken for an empty result, so let the
    # XGBoost exception propagate with its own message rather than summarising it.
    report = convert(argv[1], argv[2])
    with open(argv[3], "w") as handle:
        json.dump(report, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
