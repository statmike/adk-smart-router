#!/usr/bin/env python
"""Does `1 - contained_mass` mean what the decline rule reads it as: the chance of a miss?

`pruner.worth_pruning` is the rule that decides whether a prune ships, and it reads the
booster's probabilities as literal odds:

    prune iff  cached_rate * tokens removed * turns  >  (1 - mass) * what a miss costs

The left side is arithmetic on token counts and is as true as the counts are. The right side
is a probability multiplied by a price, and the probability comes out of a model trained with
`auto_class_weights=TRUE`, which deliberately distorts the output distribution so rare classes
are learnable (DESIGN.md R7). A distorted distribution is exactly what you want for ranking
tools and exactly what you must not read as odds. So the rule may be comparing tokens against
a number that is not the quantity it is named after, and nobody has checked.

This checks it, on held-out sessions, through the shipped path. For each session it takes the
menu `pruner.decide` picks, the mass that menu contains, and then the truth: did the session
go on to call a tool the menu did not hold. Bucketed, that is a reliability diagram for the
one probability the router spends money on.

Three questions, in the order that decides whether anything should be built.

    does it discriminate   if sessions the model calls risky miss no more often than sessions
                           it calls safe, the score carries no information and calibrating it
                           is polishing noise. This is asked first because a calibration map
                           is monotone and cannot create an ordering that is not there.
    is it calibrated       the reliability table and one summary number, the expected
                           calibration error, weighted by how many sessions sit in each bucket.
    does it matter here    every session has a miss probability at which `worth_pruning` flips.
                           If no session's true rate is anywhere near its own flip point, the
                           miscalibration is real and inert on this agent, and the honest thing
                           is to say so rather than ship a calibrator that changes nothing.

The third question is the one that decides. `report_miss_price` in `tune_settings.py` already
found the miss *price* inert on this surface for the same structural reason: the prune is so
large that no honest probability declines it. Calibration moves the other factor in the same
product, so the same arithmetic applies and has to be checked rather than assumed.

Cost is two BigQuery reads and one count_tokens call. No training, no model calls, no writes.
Authentication is ADC only.

    uv run python scripts/check_calibration.py
    uv run python scripts/check_calibration.py --buckets 5 --session-prefix corpus-20260904
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
from adk_smart_router.pruner import PredictionLike, contained_mass  # noqa: E402

Menu = frozenset[str]

# How many equal-width buckets the reliability table uses. Ten is the convention and it is a
# bad fit for a corpus where most sessions land in one of them, so the table prints every
# bucket's population and the summary is weighted by it.
DEFAULT_BUCKETS = 10

# Multiples of the shipped miss price the hazard sweep asks about. The point is to find
# the price at which the miscalibration stops being inert, so it runs well past anything
# an operator would plausibly set.
PRICE_SWEEP = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class Case:
    """One held-out session, what the router predicted about it, and what it did."""

    session_id: str
    menu: Menu | None  # None when the router would not prune at all
    predicted_miss: float  # 1 - contained_mass, the number `worth_pruning` reads
    missed: bool  # the session called a tool the menu did not hold
    saving: float  # billed tokens the prune removes over the session, `worth_pruning`'s left side

    def flip_at(self, price: float) -> float:
        """The miss probability at which pruning this session stops paying.

        `worth_pruning` is `rate * saved * turns > miss * price`, so the boundary is the
        quotient. Above 1.0 means no probability declines this prune, which is not a
        degenerate case here: it is the normal one on a wide surface, and it is why the
        third question in the module docstring exists.

        It takes the price rather than reading it off the config so the hazard sweep can ask
        the same question at prices the operator has not set yet.
        """
        return self.saving / price if price else float("inf")


def billed_saving(saved_tokens: int, cfg: RouterConfig) -> float:
    """What removing these tokens is worth over a session, in billed-equivalent tokens.

    Cached input bills at a tenth, and the saving repeats on every turn of the session, so
    both factors are in here. This is `worth_pruning`'s left side, lifted out so it can be
    weighed against more than one price.
    """
    return cfg.cached_token_multiplier * saved_tokens * cfg.expected_session_turns


def build_cases(
    sessions: Sequence[Session],
    ranked: dict[str, Sequence[PredictionLike]],
    pool: Sequence[Menu],
    live: Menu,
    known: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
) -> list[Case]:
    """One case per session, through `pruner.decide` rather than a restatement of it.

    Sessions the router would not prune are kept with `menu=None` and counted separately.
    Dropping them would measure the calibration of a rule on the population the rule already
    accepted, which is the selection the diagram exists to avoid.
    """
    cases = []
    for session in sessions:
        preds = ranked.get(session.session_id, ())
        menu = menu_always(preds, live, known, cfg, pool)
        if menu is None:
            cases.append(Case(session.session_id, None, 0.0, False, 0.0))
            continue
        cases.append(
            Case(
                session_id=session.session_id,
                menu=menu,
                predicted_miss=1.0 - contained_mass(preds, menu),
                missed=bool((session.tools & live) - menu),
                saving=billed_saving(tok(live - menu, tokens), cfg),
            )
        )
    return cases


@dataclass(frozen=True)
class Bucket:
    """One row of the reliability table."""

    low: float
    high: float
    n: int
    predicted: float  # mean predicted miss probability in this bucket
    observed: float  # share of these sessions that actually missed


def reliability(cases: Sequence[Case], buckets: int) -> list[Bucket]:
    """Predicted against observed miss rate, in equal-width buckets of the prediction.

    Equal width rather than equal count, because the question is whether a stated probability
    is that probability. Equal-count buckets would hide a pile-up at one value by spreading it
    over several rows that all report the same number.
    """
    routed = [case for case in cases if case.menu is not None]
    rows = []
    for index in range(buckets):
        low, high = index / buckets, (index + 1) / buckets
        # The last bucket is closed on the right so a prediction of exactly 1.0 has a home.
        inside = [
            case
            for case in routed
            if low <= case.predicted_miss < high
            or (index == buckets - 1 and case.predicted_miss == 1.0)
        ]
        if not inside:
            continue
        rows.append(
            Bucket(
                low=low,
                high=high,
                n=len(inside),
                predicted=sum(case.predicted_miss for case in inside) / len(inside),
                observed=sum(case.missed for case in inside) / len(inside),
            )
        )
    return rows


def calibration_error(rows: Sequence[Bucket]) -> float:
    """Expected calibration error: the gap per bucket, weighted by the bucket's population."""
    total = sum(row.n for row in rows)
    if not total:
        return 0.0
    return sum(row.n * abs(row.predicted - row.observed) for row in rows) / total


def discrimination(cases: Sequence[Case]) -> float | None:
    """The chance a session that missed was scored riskier than one that did not.

    This is the area under the ROC curve, computed the way it is defined rather than
    approximated: every missing/not-missing pair, counting ties as half. Quadratic, and on a
    few thousand sessions that is a few million comparisons, which is nothing.

    0.5 is a coin. Below 0.5 means the score is backwards. Either way a calibration map cannot
    help, because every calibration map is monotone and monotone maps preserve the ordering
    this number is measuring.
    """
    routed = [case for case in cases if case.menu is not None]
    missed = [case.predicted_miss for case in routed if case.missed]
    fine = [case.predicted_miss for case in routed if not case.missed]
    if not missed or not fine:
        return None
    wins = sum((m > f) + 0.5 * (m == f) for m in missed for f in fine)
    return wins / (len(missed) * len(fine))


def headroom(cases: Sequence[Case], price: float) -> tuple[int, float]:
    """How many pruned sessions could ever be declined, and by how far the rest clear it.

    A session whose flip point is above 1.0 declines at no probability at all, so no
    recalibration of any kind can change what the router does with it. Returns the count that
    could flip and the median ratio of flip point to predicted miss over the rest, which is
    the factor a calibrator would have to move the probability by to matter.
    """
    routed = [case for case in cases if case.menu is not None]
    reachable = [case for case in routed if case.flip_at(price) <= 1.0]
    ratios = sorted(
        case.flip_at(price) / case.predicted_miss for case in routed if case.predicted_miss > 0.0
    )
    median = ratios[len(ratios) // 2] if ratios else float("inf")
    return len(reachable), median


def calibrate(rows: Sequence[Bucket], predicted: float) -> float:
    """What the observed rate was, for sessions the model said this about.

    This is the crudest possible calibrator, a lookup into the reliability table, and it is
    deliberately crude: the point is not to ship it but to price what a real one would buy.
    Isotonic regression would smooth these steps and would not move them far, so if the crude
    map changes no decision, neither does the good one.
    """
    for row in rows:
        # The same rule `reliability` bucketed by, including its closed right edge on the
        # top bucket, so a prediction cannot be scored into one bucket and mapped by another.
        if row.low <= predicted < row.high or (predicted == 1.0 and row.high >= 1.0):
            return row.observed
    return predicted


def hazard(
    cases: Sequence[Case], rows: Sequence[Bucket], price: float, multiples: Sequence[float]
) -> list[tuple[float, int, int]]:
    """How many prunes the miscalibration would talk the router out of, at prices above today's.

    Today's answer can be "none", and on a wide surface it will be, because the prune is worth
    more than any miss. That is worth knowing and it is not the whole answer, because the miss
    price is an operator setting and the surface is whatever the agent grows into. This sweeps
    the price upward and reports, at each one, how many sessions the raw probability declines
    against how many the observed rate declines. The gap is the cost of not calibrating, and
    the first price where it opens is the setting at which this finding expires.
    """
    routed = [case for case in cases if case.menu is not None]
    table = []
    for multiple in multiples:
        at = price * multiple
        raw = sum(case.predicted_miss > case.flip_at(at) for case in routed)
        true = sum(calibrate(rows, case.predicted_miss) > case.flip_at(at) for case in routed)
        table.append((multiple, raw, true))
    return table


def report(cases: Sequence[Case], rows: Sequence[Bucket], price: float) -> None:
    routed = [case for case in cases if case.menu is not None]
    declined = len(cases) - len(routed)
    missed = sum(case.missed for case in routed)
    print(f"\n{len(cases)} held-out sessions, {len(routed)} the router would prune")
    if declined:
        print(f"  {declined} declined before the probability was consulted, and are left out")
    if not routed:
        print("nothing to calibrate: the router prunes none of these sessions.")
        return

    predicted = sum(case.predicted_miss for case in routed) / len(routed)
    observed = missed / len(routed)
    print(f"\n  predicted miss rate      {predicted:.4f}, over the pruned sessions")
    print(f"  observed miss rate       {observed:.4f}, {missed} of {len(routed)}")
    print(
        f"  the model is {'pessimistic' if predicted > observed else 'optimistic'} by"
        f" {abs(predicted - observed):.4f}"
    )

    print("\nreliability, by the probability the router acted on\n")
    print(f"  {'bucket':>12} {'sessions':>9} {'predicted':>10} {'observed':>9} {'gap':>8}")
    for row in rows:
        gap = row.predicted - row.observed
        print(
            f"  {row.low:>5.2f}-{row.high:<6.2f} {row.n:>9} {row.predicted:>10.4f}"
            f" {row.observed:>9.4f} {gap:>+8.4f}"
        )
    print(f"\n  expected calibration error {calibration_error(rows):.4f}")

    auc = discrimination(cases)
    if auc is None:
        print(
            "\n  discrimination cannot be measured: every pruned session fell the same way, so\n"
            "  there is no pair to order. That is the answer to the first question too."
        )
    else:
        print(f"  discrimination (AUC)       {auc:.4f}")
        if auc <= 0.55:
            print(
                "  which is a coin. The score does not separate the sessions that miss from the\n"
                "  ones that do not, and no calibration map can put an ordering there, because\n"
                "  every calibration map is monotone. Calibrating this would move the numbers\n"
                "  and change nothing about which sessions are at risk."
            )

    reachable, median = headroom(cases, price)
    print(f"\n  pruned sessions whose prune could ever be declined: {reachable} of {len(routed)}")
    if not reachable:
        print(
            f"  none of them at the shipped miss price of {price:,.0f} tokens, so the\n"
            "  probability is not deciding anything on this agent today. The prune removes more\n"
            "  billed tokens than a miss can cost at any odds, which is `report_miss_price`'s\n"
            "  finding about the other factor in the same product."
        )
    else:
        print(
            f"  the median pruned session would need its miss probability multiplied by"
            f" {median:,.1f}\n  before `worth_pruning` declined it."
        )

    print("\nwhat the miscalibration would cost at a higher miss price\n")
    print(f"  {'miss price':>12} {'raw declines':>13} {'true declines':>14} {'needless':>9}")
    opens_at = None
    for multiple, raw, true in hazard(cases, rows, price, PRICE_SWEEP):
        if opens_at is None and raw > true:
            opens_at = multiple
        print(f"  {price * multiple:>12,.0f} {raw:>13} {true:>14} {raw - true:>9}")
    if opens_at is None:
        print(
            "\n  the two columns never part, so on this corpus the miscalibration is inert at\n"
            "  every price swept, not only at today's."
        )
    else:
        print(
            f"\n  the columns part at {price * opens_at:,.0f} tokens, {opens_at:g} times the\n"
            f"  shipped price. Below that this finding holds and calibration buys nothing.\n"
            "  At or above it the router starts declining prunes that would have paid, and\n"
            f"  the gap in the last column is how many, out of these {len(routed)}."
        )


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1

    tokens = asyncio.run(declaration_tokens())
    live = frozenset(tokens)
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.menus)} menus")
    print(f"       {len(live)} live tools, miss priced at {cfg.miss_cost_tokens} tokens")

    held_out = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, True
    )
    if not held_out:
        print(f"no held-out sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    predictor = Predictor(model)
    ranked = {s.session_id: predictor.predict(s.vector) for s in held_out}
    cases = build_cases(held_out, ranked, model.menus, live, frozenset(model.classes), tokens, cfg)
    report(cases, reliability(cases, args.buckets), float(cfg.miss_cost_tokens))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument("--session-prefix", default="corpus-")
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
