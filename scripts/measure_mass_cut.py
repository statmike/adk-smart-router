#!/usr/bin/env python
"""H9 — the gate scores a cut the pruner does not make. How far apart are the two?

`sql/train_router.sql` reports coverage at a strict top-N: strip ABSTAIN, take the best
`top_n` labels, ask whether the tool the session wanted is in there. That number is the one
`gate_passed` is decided on and the one published in `gate_metrics`.

Without menus the pruner does something else. `pruner.top_k_by_mass` walks the ranking in
order, keeping labels until their cumulative probability reaches `keep_probability_mass`,
and it walks over ABSTAIN too. So the served set can be smaller than N for two unrelated
reasons: the mass rule stopped early, or ABSTAIN spent one of the slots. Either way the
gate's coverage is an upper bound on what the process serves, and nobody has measured the
distance.

This measures it, on held-out sessions, with the shipped booster and the shipped functions.
Four arms:

    gate cut         ABSTAIN stripped, best `top_n`. What train_router.sql scores.
    mass cut         `top_k_by_mass` exactly as `pruner.decide` calls it. What is served.
    mass, no abstain the same rule with ABSTAIN removed first, which separates the two
                     causes: whatever this recovers over `mass cut` is the slot ABSTAIN took.
    served set       the whole of `pruner.decide`, so protected and classless tools are
                     unioned back in. The bottom line, and the only arm a user experiences.

The first three are cuts and are directly comparable to the SQL. The fourth is there because
if `protected` rescues the coverage then the gap between the first two is real arithmetic and
not a real problem, and reporting only the cuts would overstate it.

Menu artifacts are out of scope by construction: a menu is taken whole, neither rule runs,
and `publish_router.py` already measures their recall against the rule that serves.

Cost is one BigQuery read and no embedding calls: the vectors are the ones BigQuery already
computed, and the tool surface is read from the reference agent on disk. ADC only.

Usage:
    uv run python scripts/measure_mass_cut.py
    uv run python scripts/measure_mass_cut.py --eval-only --top-n 11
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_chars  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Session,
    fetch_sessions,
)

from adk_smart_router.config import FRAMEWORK_TOOLS, load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Prediction, Predictor  # noqa: E402
from adk_smart_router.pruner import ABSTAIN, decide, top_k_by_mass  # noqa: E402


@dataclass
class Arm:
    """Coverage and width for one way of cutting the ranking."""

    name: str
    covered: int = 0  # (session, wanted tool) pairs the set held
    wanted: int = 0  # (session, wanted tool) pairs in total
    complete: int = 0  # sessions whose set held every tool they wanted
    scored: int = 0  # sessions with at least one wanted tool
    widths: list[int] = field(default_factory=list)

    def add(self, chosen: frozenset[str], session: Session) -> None:
        self.widths.append(len(chosen))
        if not session.tools:
            return  # a session that called nothing has no coverage to measure
        self.scored += 1
        held = len(session.tools & chosen)
        self.covered += held
        self.wanted += len(session.tools)
        self.complete += held == len(session.tools)

    @property
    def coverage(self) -> float:
        """Per-tool, the same denominator `train_router.sql` uses."""
        return self.covered / self.wanted if self.wanted else 0.0

    @property
    def completeness(self) -> float:
        """Per-session, which is the quantity a user of the agent actually feels."""
        return self.complete / self.scored if self.scored else 0.0


# --- The cuts -------------------------------------------------------------------


def gate_cut(preds: Sequence[Prediction], top_n: int) -> frozenset[str]:
    """ABSTAIN stripped, best `top_n`. A transcription of the SQL, not a call into it."""
    live = [p for p in preds if p.label != ABSTAIN]
    return frozenset(p.label for p in live[:top_n])


def mass_cut(preds: Sequence[Prediction], mass: float, cap: int) -> frozenset[str]:
    """What `pruner.decide` calls when the artifact has no menus. The shipped function."""
    return frozenset(top_k_by_mass(preds, mass, cap))


def mass_cut_without_abstain(preds: Sequence[Prediction], mass: float, cap: int) -> frozenset[str]:
    """The same rule, ABSTAIN removed first. The difference is the slot ABSTAIN took."""
    return frozenset(top_k_by_mass([p for p in preds if p.label != ABSTAIN], mass, cap))


# --- Report ---------------------------------------------------------------------


def report(arms: list[Arm], sessions: int, toolless: int, top_n: int, mass: float) -> None:
    print()
    print(f"{sessions} held-out sessions, {toolless} of which called no tool at all")
    print(f"top_n={top_n}, keep_probability_mass={mass}")
    print()
    header = f"{'arm':<20} {'coverage':>9} {'complete':>9} {'mean w':>7} {'median':>7} {'max':>4}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        mean = statistics.mean(arm.widths) if arm.widths else 0.0
        median = statistics.median(arm.widths) if arm.widths else 0
        widest = max(arm.widths) if arm.widths else 0
        print(
            f"{arm.name:<20} {arm.coverage:>9.4f} {arm.completeness:>9.4f} "
            f"{mean:>7.2f} {median:>7.0f} {widest:>4}"
        )

    gate, served = arms[0], arms[1]
    gap = gate.coverage - served.coverage
    print()
    print(f"the gap H9 is about: {gap:+.4f} coverage, gate cut minus mass cut")
    if len(arms) > 3:
        rescued = arms[3].coverage - served.coverage
        print(f"protected and classless tools rescue {rescued:+.4f} of it")


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)
    top_n = args.top_n or cfg.max_tools_kept
    mass = args.mass or cfg.keep_probability_mass

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.classes)} classes")
    if model.menus:
        print(
            f"note: this artifact ships {len(model.menus)} menus, so neither cut runs in "
            "production. The numbers below are about the menu-less path only."
        )

    live = frozenset(declaration_chars())
    known = frozenset(model.classes)
    print(f"       {len(live)} live tools on the reference agent")

    sessions = fetch_sessions(
        args.project,
        args.dataset,
        args.agent,
        cfg.embedding_dims,
        args.session_prefix,
        args.eval_only,
    )
    if not sessions:
        print(f"\nno sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    predictor = Predictor(model)
    protected = (FRAMEWORK_TOOLS | cfg.never_prune) & live
    arms = [Arm("gate cut"), Arm("mass cut"), Arm("mass, no abstain"), Arm("served set")]

    for session in sessions:
        preds = predictor.predict(session.vector)
        arms[0].add(gate_cut(preds, top_n), session)
        arms[1].add(mass_cut(preds, mass, top_n), session)
        arms[2].add(mass_cut_without_abstain(preds, mass, top_n), session)
        # `menus=()` on purpose: this is the path being measured, whatever the artifact ships.
        served = decide(preds, live, known, None, cfg, menus=())
        arms[3].add(served.kept if served.kept is not None else live, session)

    report(arms, len(sessions), sum(1 for s in sessions if not s.tools), top_n, mass)
    print(f"({len(protected)} protected tools ride through every cut)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--session-prefix",
        default="corpus",
        help="only score sessions whose id starts with this. Must be UNROUTED traffic.",
    )
    parser.add_argument(
        "--top-n", type=int, default=None, help="defaults to the shipped max_tools_kept"
    )
    parser.add_argument(
        "--mass", type=float, default=None, help="defaults to the shipped keep_probability_mass"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="restrict to the training split's held-out sessions (DESIGN.md 9.4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
