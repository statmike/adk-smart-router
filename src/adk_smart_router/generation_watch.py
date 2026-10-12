"""Whether a published generation is worse than not routing at all, and may be withdrawn.

The contract in DESIGN.md 21.1 is that installing this plugin means, at worst, no improvement, and
never degradation. Everything else that enforces it is per-process and fast: a callback fails open,
`cache_watch` demotes an agent on its own bill within minutes. None of them can answer a
generation-level question, because a generation's traffic is spread across every replica and no
single process sees enough of it.

This is the slow half. It runs offline, reads `v_router_generations`, and compares the current
generation's routed arm against its own concurrent holdout arm. Not against the generation before
it, which ran on different traffic at a different time; against what not routing costs on the same
traffic, right now, which is the guarantee stated as arithmetic.

**It reads `g1_later` and not `g1`.** A session's first model call has no prefix of its own to hit
and bills about three times a continuation, and pruning pushes that one call the wrong way while
making every later call much cheaper. On this project's own warehouse the two came out nineteen
points apart, -10.6% pooled against -29.7% on later turns. Judging on the pooled figure would make
this a measurement of how well `prefix_lock.py` is working rather than of the model (DESIGN.md
18.22, 21.5).

**Withdrawing is subtraction, not deletion.** The caller writes a `withdrawn.json` into the
version's directory, `model_store` stops counting that directory as published, and every serving
process falls back to the generation before it on its next re-poll. Deleting the marker puts it
back. Nothing is destroyed and no redeploy is involved.

Pure arithmetic. No cloud, no client, and nothing here is imported at request time.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_MAX_WITHDRAWALS",
    "DEFAULT_MIN_LATER_TURNS",
    "DEFAULT_WINDOW_HOURS",
    "Arms",
    "Verdict",
    "verdict",
]

# How many priced later turns each arm needs before the comparison says anything. Six times
# `cache_watch_min_samples`, which it can afford because it pools every replica rather than
# reading one process's own traffic, and because the action is fleet-wide where a demotion is not.
#
# **This is a stated prior and not a fit.** There is one generation of production history, so
# there is no run-to-run spread to measure this against. DESIGN.md 21.5 says a few hundred; this
# is that, and it is recorded on every ledger row precisely so the first person with two
# generations of history can argue with it using their own numbers.
DEFAULT_MIN_LATER_TURNS = 300

# How much more expensive routing has to be, as a fraction of the holdout arm's later-turn cost,
# before the generation is called harmful. The same 0.05 as `cache_watch_margin`, deliberately:
# the fast guard and the slow guard are measuring the same quantity, and if they disagreed about
# where harmful starts, an agent could sit in the gap where one demotes and the other approves.
DEFAULT_MARGIN = 0.05

# The rate limit. Falling back is allowed to cascade — withdrawing generation 4 makes 3 current,
# and 3 is judged on its own next window — and running that to its end leaves the agent in
# passthrough, which is the guarantee reached the long way round. What must not happen is all of
# it before anyone reads the ledger, so at most one withdrawal per agent per window. A floor on
# how far back it may walk would be the wrong guard: "keep serving the least bad model" is not a
# guarantee anybody asked for.
DEFAULT_MAX_WITHDRAWALS = 1
DEFAULT_WINDOW_HOURS = 24


@dataclass(frozen=True)
class Arms:
    """One generation's two arms, counted over later turns only.

    A rate is None when its arm had no later turns at all, which is not zero: a generation with no
    holdout traffic has no control to be judged against, and a zero there would read as free.
    """

    routed_turns: int
    routed_billed_per_turn: float | None
    holdout_turns: int
    holdout_billed_per_turn: float | None

    @property
    def g1_later(self) -> float | None:
        """Routed cost per later turn against holdout, as a fraction. Negative is cheaper."""
        if self.routed_billed_per_turn is None or not self.holdout_billed_per_turn:
            return None
        return (self.routed_billed_per_turn - self.holdout_billed_per_turn) / (
            self.holdout_billed_per_turn
        )


@dataclass(frozen=True)
class Verdict:
    """What the watch decided, and the sentence explaining it.

    `outcome` is the ledger's word. `ok` means a version was withdrawn, `declined` means the
    generation was measured and is fine, and `skipped` means it could not be judged. All three are
    recorded, every time the watch looks, because a watch that writes a row only when it acts is
    indistinguishable from a watch that is not running.
    """

    withdraw: bool
    outcome: str
    reason: str
    metrics: dict[str, float | int | None]


def verdict(
    arms: Arms,
    *,
    min_later_turns: int = DEFAULT_MIN_LATER_TURNS,
    margin: float = DEFAULT_MARGIN,
    withdrawals_in_window: int = 0,
    max_withdrawals: int = DEFAULT_MAX_WITHDRAWALS,
) -> Verdict:
    """Judge one generation. Returns a verdict; withdrawing it is the caller's job.

    The rate limit is applied last rather than first, so that a withdrawal it holds back is still
    recorded as the withdrawal it would have been. Checking it up front would write `skipped` over
    an agent that is actually cascading, which is the one situation the ledger most needs to show.
    """
    measured = _measured(arms, margin, min_later_turns)

    if arms.routed_turns < min_later_turns or arms.holdout_turns < min_later_turns:
        return Verdict(
            withdraw=False,
            outcome="skipped",
            reason=(
                f"not enough traffic to judge: {arms.routed_turns} routed and "
                f"{arms.holdout_turns} holdout later turns against a {min_later_turns} floor "
                "in each arm."
            ),
            metrics=measured,
        )

    g1_later = arms.g1_later
    if g1_later is None:
        return Verdict(
            withdraw=False,
            outcome="skipped",
            reason=(
                "one arm has later turns but no priced cost, so there is nothing to compare. "
                "This is a logging fault rather than a model one."
            ),
            metrics=measured,
        )

    if g1_later <= margin:
        return Verdict(
            withdraw=False,
            outcome="declined",
            reason=(
                f"routing costs {g1_later:+.1%} per later turn against its own holdout arm, "
                f"inside the {margin:.0%} margin, over {arms.routed_turns} routed and "
                f"{arms.holdout_turns} holdout later turns."
            ),
            metrics=measured,
        )

    harm = (
        f"routing costs {g1_later:+.1%} per later turn against its own holdout arm, "
        f"{arms.routed_billed_per_turn:.0f} against {arms.holdout_billed_per_turn:.0f} billed "
        f"tokens, over the {margin:.0%} margin"
    )

    if withdrawals_in_window >= max_withdrawals:
        return Verdict(
            withdraw=False,
            outcome="skipped",
            reason=(
                f"{harm}, and this agent has already been withdrawn {withdrawals_in_window} "
                f"time(s) this window against a limit of {max_withdrawals}. Held back so a bad "
                "measurement cannot walk it back through several generations unread."
            ),
            metrics=measured,
        )

    return Verdict(
        withdraw=True,
        outcome="ok",
        reason=f"{harm}. Withdrawn, so serving falls back to the generation before it.",
        metrics=measured,
    )


def _measured(arms: Arms, margin: float, min_later_turns: int) -> dict[str, float | int | None]:
    """Everything the verdict was drawn from, for the ledger row.

    Recorded on every outcome including the skipped ones, so that a rollback can be argued with
    the next morning and a watch that keeps declining can be checked for whether it is looking at
    anything at all.
    """
    return {
        "g1_later": arms.g1_later,
        "routed_later_turns": arms.routed_turns,
        "routed_later_billed_per_turn": arms.routed_billed_per_turn,
        "holdout_later_turns": arms.holdout_turns,
        "holdout_later_billed_per_turn": arms.holdout_billed_per_turn,
        "margin": margin,
        "min_later_turns": min_later_turns,
    }
