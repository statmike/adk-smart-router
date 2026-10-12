#!/usr/bin/env python
"""V2-6 — is a session's length worth predicting?

The router already predicts what a session will *do*. ROADMAP's V2-6 asks whether it should
also predict what a session will *be*, and names session length as the first candidate. This
script answers that before any model is trained, because the answer might be no.

Length has exactly one live consumer. `pruner.worth_pruning` prices a prune against a miss:

    prune iff  cached_rate * tokens removed * turns  >  P(miss) * what a miss costs

`turns` there is `RouterConfig.expected_session_turns`, one number for every session on the
agent. A session that will run ten turns earns ten times the saving of a session that will
run one, so in principle the constant is wrong for almost every session and a prediction
would fix it. Whether that is worth anything is an arithmetic question, and it is settled
here in three steps.

    leverage    what perfect knowledge of the length would be worth, in billed tokens. The
                rule is its own judge: it prunes when it expects a prune to pay, so the
                regret of the constant is what the rule would have gained by knowing.
    reach       the same, for agents unlike this one. Leverage depends on how many tokens
                the prune removes, and the reference agent removes a great many. Sweeping
                that shows the surface at which the constant starts to cost something.
    signal      whether the opening message predicts the length at all, k-nearest-neighbour
                over the same embedding the tool-set model reads.

The order matters. If leverage is nil the signal is irrelevant, however strong it is.

It is nil. On the reference agent a perfect oracle for the length changes one decision in
1,700 and is worth 0.0015% of what the corpus billed, because the prune removes about eight
thousand declaration tokens and so pays back its worst-case miss after 0.69 turns. No session
is that short. The constant is not accurate, it is merely on the right side of a threshold
nothing comes near, and that is all the rule ever asks of it.

The reach sweep says this is not luck, and also not universal. Regret peaks at 0.199% of
billed for an agent whose prune removes 2,000 tokens, where the break-even lands at 2.8 turns
and sits inside the distribution. Above 4,000 it goes back to nothing. So the decline rule's
length constant is worth tuning on a small tool surface and inert on a large one, which is
worth knowing before V2-7 spends a sweep on it.

The signal is real and does not rescue any of this. Nearest neighbours on the opening
embedding remove about a fifth of the error in the length, so the length is genuinely
somewhat predictable. A fifth of a ceiling of 0.199% is not a feature.

One caveat is load-bearing and belongs on the reader's first pass. This corpus's *user*
turn count is a fixture constant: `scenarios.json` fixes it at 3 or 4 per scenario, and the
opening message comes from that same scenario, so predicting it would be memorising the
generator. What is measured here instead is *model calls*, which the generator does not fix.
Those run 1 to 10 as the agent chooses to call tools or not, and that is also the quantity
the decline rule multiplies by, since every model call re-sends the declarations.

    uv run python scripts/study_session_length.py
    uv run python scripts/study_session_length.py --session-prefix corpus-20260904

Cost is two BigQuery reads and one count_tokens call. No training, no model calls, no
writes. Authentication is ADC only.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_tokens  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Session,
    fetch_sessions,
    menu_always,
    tok,
)

from adk_smart_router.config import RouterConfig, load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Predictor  # noqa: E402
from adk_smart_router.pruner import contained_mass  # noqa: E402

# How many neighbours vote on a length. Ten is the best of the sweep 5, 10, 25, 50, 100, and
# the best is the right default: the conclusion below is that the signal is not worth acting
# on, and that has to be argued against the strongest reading of it rather than a middling
# one. The choice does not decide anything either way. Across that whole 20x range the
# embedding removes between 12% and 21% of the error, so `--neighbours` moves the third
# significant figure of an answer that is already clear at the first.
NEIGHBOURS = 10

# The share of sessions held out. The training run's own split (DESIGN.md 9.4), repeated so
# a number measured here is comparable with one measured there.
EVAL_PCT = 25

# Prunes of this many declaration tokens, for the reach sweep. The reference agent removes
# about eight thousand; an agent with a dozen small tools removes a few hundred.
REACH_TOKENS = (250, 500, 1_000, 2_000, 4_000, 8_000)


@dataclass(frozen=True)
class Case:
    """One session, reduced to what the decline rule actually reads.

    `saved` is the declaration tokens the router's menu would remove and `mass` is the
    probability the session wants nothing that was removed. Together with a turn count they
    are the whole of `worth_pruning`, so a session that has been through here needs no
    predictor and no menu pool to re-price.
    """

    session_id: str
    turns: int
    saved: int
    mass: float
    vector: tuple[float, ...]


# --- Leverage -----------------------------------------------------------------------


def prune_value(case: Case, turns: float, cfg: RouterConfig) -> float:
    """Billed tokens a prune is worth at this length, positive when it pays.

    This is `pruner.worth_pruning` with the comparison taken out: the rule prunes exactly
    when this is above zero. Keeping the value rather than the boolean is what makes regret
    measurable, because a decision that was wrong by one token and one that was wrong by a
    thousand are the same boolean and very different mistakes.
    """
    saving = cfg.cached_token_multiplier * case.saved * turns
    return saving - (1.0 - case.mass) * cfg.miss_cost_tokens


def break_even(case: Case, cfg: RouterConfig) -> float:
    """The length at which this session's prune stops paying and starts costing."""
    earned = cfg.cached_token_multiplier * case.saved
    if earned <= 0:
        return float("inf")
    return (1.0 - case.mass) * cfg.miss_cost_tokens / earned


def regret(cases: Sequence[Case], cfg: RouterConfig) -> tuple[float, int]:
    """What the constant costs against knowing each session's true length.

    Regret per session is what the best decision would have earned minus what the constant's
    decision did earn, both priced at the length the session turned out to run. It is zero
    wherever the constant and the truth agree, which is most places, so the total is a count
    of mistakes weighted by how much each one cost.
    """
    total = 0.0
    flips = 0
    for case in cases:
        truth = prune_value(case, case.turns, cfg)
        chose_prune = prune_value(case, cfg.expected_session_turns, cfg) > 0
        earned = truth if chose_prune else 0.0
        total += max(truth, 0.0) - earned
        flips += chose_prune != (truth > 0)
    return total, flips


# --- Signal -------------------------------------------------------------------------


def held_out(session_id: str) -> bool:
    """A deterministic quarter of sessions, split on the id and nothing else.

    Not `FARM_FINGERPRINT`, which is BigQuery's and cannot be computed here, so a session in
    this eval set is not necessarily in the training run's. That is fine for a question
    about whether a signal exists and would not be fine for comparing two models.
    """
    return zlib.crc32(session_id.encode()) % 100 < EVAL_PCT


def knn_lengths(train: Sequence[Case], test: Sequence[Case], k: int) -> np.ndarray:
    """Each test session's length, guessed as the mean of its nearest neighbours.

    Cosine similarity over the opening embedding, which is the signal the tool-set model
    reads, so a failure here is a statement about the signal and not about a model that was
    given less than the real one gets.
    """
    train_v = _unit(np.array([c.vector for c in train], dtype=np.float64))
    test_v = _unit(np.array([c.vector for c in test], dtype=np.float64))
    lengths = np.array([c.turns for c in train], dtype=np.float64)

    k = min(k, len(train))
    similarity = test_v @ train_v.T
    nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    return lengths[nearest].mean(axis=1)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


# --- Reporting ----------------------------------------------------------------------


def report_spread(cases: Sequence[Case]) -> None:
    lengths = [c.turns for c in cases]
    counts = {n: lengths.count(n) for n in sorted(set(lengths))}
    print(f"\n{len(cases)} sessions, model calls per session")
    print(
        f"  mean {statistics.mean(lengths):.2f}   median {statistics.median(lengths):.0f}"
        f"   sd {statistics.pstdev(lengths):.2f}   range {min(lengths)}-{max(lengths)}"
    )
    print("  " + "  ".join(f"{n}:{count}" for n, count in counts.items()))


def report_leverage(cases: Sequence[Case], cfg: RouterConfig, billed: float) -> None:
    points = [break_even(case, cfg) for case in cases]
    finite = [p for p in points if p != float("inf")]
    cost, flips = regret(cases, cfg)

    print(f"\nleverage: what knowing the true length would be worth, at {cfg.miss_cost_tokens}")
    print(f"  the constant in use              {cfg.expected_session_turns:.2f} turns")
    if finite:
        inside = sum(1 for p in finite if min(c.turns for c in cases) <= p <= max_turns(cases))
        print(f"  break-even length, median        {statistics.median(finite):.2f} turns")
        print(f"  sessions whose break-even lands  {inside} of {len(cases)}")
        print("    inside the observed range      (the only ones a prediction could move)")
    print(f"  decisions a perfect length flips {flips}")
    print(f"  regret against perfect knowledge {cost:,.0f} billed tokens")
    if billed:
        print(f"                                   {cost / billed:.4%} of what the corpus billed")


def max_turns(cases: Sequence[Case]) -> int:
    return max(c.turns for c in cases)


def report_reach(cases: Sequence[Case], cfg: RouterConfig, billed: float) -> None:
    """The same regret for agents whose prune removes less than this one's.

    Leverage is not a property of session length, it is a property of length against prune
    size. A prune worth thousands of tokens a turn pays on the shortest session anyone runs,
    so the constant is never wrong in a way that matters. Shrink the prune and the break-even
    walks into the middle of the distribution, which is where a prediction would earn.

    Every session is given the same prune size, which is the point: only that one quantity is
    hypothetical. The lengths and the miss probabilities stay exactly as measured, so each row
    reads as this corpus's own sessions on an agent with a smaller tool surface.
    """
    print("\nreach: the same regret if the prune removed fewer tokens")
    print(f"  {'removed':>9} {'break-even':>11} {'flips':>7} {'regret':>12} {'of billed':>10}")
    for removed in REACH_TOKENS:
        scaled = [replace(case, saved=removed) for case in cases]
        cost, flips = regret(scaled, cfg)
        median = statistics.median(break_even(case, cfg) for case in scaled)
        share = f"{cost / billed:.3%}" if billed else "n/a"
        print(f"  {removed:>9,} {median:>11.2f} {flips:>7} {cost:>12,.0f} {share:>10}")


def report_signal(cases: Sequence[Case], k: int) -> None:
    train = [c for c in cases if not held_out(c.session_id)]
    test = [c for c in cases if held_out(c.session_id)]
    print(f"\nsignal: does the opening message predict the length? {len(train)}/{len(test)} split")
    if not train or not test:
        print("  not enough sessions on one side of the split to say.")
        return

    truth = np.array([c.turns for c in test], dtype=np.float64)
    baseline = float(np.mean([c.turns for c in train]))
    guessed = knn_lengths(train, test, k)

    flat_error = float(np.mean(np.abs(truth - baseline)))
    knn_error = float(np.mean(np.abs(truth - guessed)))
    spread = float(np.std(guessed))
    print(f"  predict the training mean ({baseline:.2f})  {flat_error:.3f} turns of error")
    print(f"  {k} nearest neighbours by embedding    {knn_error:.3f} turns of error")
    print(f"  the error the embedding removes      {1 - knn_error / flat_error:+.2%}")
    print(f"  spread of what it guessed            {spread:.3f} against {np.std(truth):.3f} true")


# --- Main ---------------------------------------------------------------------------


def build_cases(
    sessions: Sequence[Session],
    predictor: Predictor,
    pool: Sequence[frozenset[str]],
    live: frozenset[str],
    known: frozenset[str],
    tokens: dict[str, int],
    cfg: RouterConfig,
) -> list[Case]:
    """Run the shipped router over each session and keep what the decline rule reads.

    Sessions the router has no menu for are dropped rather than counted as declines. They
    never reach `worth_pruning`, so their length cannot change anything and including them
    would dilute every rate below with sessions that were never in play.
    """
    cases = []
    for session in sessions:
        ranked = predictor.predict(session.vector)
        menu = menu_always(ranked, live, known, cfg, pool)
        if menu is None:
            continue
        cases.append(
            Case(
                session_id=session.session_id,
                turns=session.turns,
                saved=tok(live - menu, tokens),
                mass=contained_mass(ranked, menu),
                vector=tuple(session.vector),
            )
        )
    return cases


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1

    tokens = asyncio.run(declaration_tokens())
    live = frozenset(tokens)
    sessions = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, False
    )
    if not sessions:
        print(f"no sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.menus)} menus")
    print(f"       {len(live)} live tools, {sum(tokens.values())} declaration tokens")

    cases = build_cases(
        sessions, Predictor(model), model.menus, live, frozenset(model.classes), tokens, cfg
    )
    if not cases:
        print("the router produced no menu for any session; nothing to price.", file=sys.stderr)
        return 1

    billed = sum(s.billed for s in sessions)
    report_spread(cases)
    report_leverage(cases, cfg, billed)
    report_reach(cases, cfg, billed)
    report_signal(cases, args.neighbours)
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
        help="only read sessions whose id starts with this. Must be UNROUTED traffic: a"
        " routed session's tool set is censored by the prune it received.",
    )
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
