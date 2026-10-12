"""Whether a model still describes the traffic it is serving, and which signal says so.

The router ships two ways of noticing it has gone stale and neither one is a detector.
`cache_watch` reads the bill, so it fires after the money is spent. `max_model_age_days` reads the
clock, so it fires whether or not anything has changed. This is the third: a signal readable from
the request alone, before either of those (DESIGN.md 19.63, 20.5).

**Unseen share gates. Confidence is reported and never gates.** §19.63 drove a real drift event and
measured both. The 2026-09-04 surface change from 25 tools to 100 cost 36 points of recall, 0.9849
to 0.6249, and moved unseen share from 26.1% to 77.3% at the same instant: the signal tracks the
harm. A constructed intent shift, withholding one of eight opening clusters from training, moved
top-1 confidence by 0.25 for 1.3 points of recall. That is sensitive to change rather than to harm,
so it belongs on a prompt and not on a trigger. The project makes this distinction elsewhere too —
`coverage_mass`, `lift` and `baseline_coverage_corpus` are all logged and none of them gate.

**The baseline is not zero, which is the whole reason it is recorded at publish.** §19.63 measured
a healthy model with no class for 26.1% of its declared surface, because the rare-class floor drops
tools that never appear often enough to learn. A detector comparing against a constant reads a
healthy model as a quarter drifted on its first day.

Definitions live here rather than in the two callers so that the number `publish_router.py` writes
as a baseline and the number `check_drift.py` measures against it are the same number. Two
measurements that disagree about what they measure cannot be read against each other.

Pure arithmetic. No cloud, no client, nothing imported at request time.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_UNSEEN_SHARE_MARGIN",
    "Signals",
    "Verdict",
    "signals",
    "verdict",
]

# How far over its own baseline the unseen share may drift before a retrain is due, in absolute
# share. **This is a two-point fit and should be read as one.** §19.63 offers one real event, a
# surface change worth +51.2 points of unseen share and −36 points of recall, and one real
# non-event, a new day's traffic on the same tools where every signal stayed flat. Twenty points
# sits comfortably between them and is not otherwise earned. `check_drift.py --backfill` replays an
# agent's own history against a candidate, and this wants re-tuning the first time a second surface
# change exists to fit against.
DEFAULT_UNSEEN_SHARE_MARGIN = 0.20


@dataclass(frozen=True)
class Signals:
    """How well a model fits a tool surface and the traffic over it, at one moment.

    `unseen_share` is the share of the *declared* tools the model has no class for, so it moves
    when the agent's tool list moves underneath a model that was trained before it. `confidence_*`
    are quantiles of the model's top-1 probability mass, reported for a human and never gating.

    A field is None when it could not be measured rather than zero, because "no sessions to take a
    quantile over" and "the model is never confident" are opposite readings of the same agent.
    """

    unseen_share: float
    declared: int
    unclassed: int
    confidence_p50: float | None = None
    confidence_p10: float | None = None
    sessions: int = 0

    def as_dict(self) -> dict[str, Any]:
        """The shape written into `metadata.json` and into a ledger row's `metrics`."""
        return {
            "unseen_share": self.unseen_share,
            "declared": self.declared,
            "unclassed": self.unclassed,
            "confidence_p50": self.confidence_p50,
            "confidence_p10": self.confidence_p10,
            "sessions": self.sessions,
        }

    @staticmethod
    def from_dict(raw: Any) -> Signals | None:  # noqa: ANN401
        """Read a baseline back, or None if there is not one to read.

        Deliberately forgiving. An artifact published before this existed has no baseline, an
        artifact published by a newer version may have fields this one does not know, and neither
        is an error: `status` has to work on both (DESIGN.md 20.10).
        """
        if not isinstance(raw, dict) or "unseen_share" not in raw:
            return None
        try:
            return Signals(
                unseen_share=float(raw["unseen_share"]),
                declared=int(raw.get("declared", 0)),
                unclassed=int(raw.get("unclassed", 0)),
                confidence_p50=_maybe_float(raw.get("confidence_p50")),
                confidence_p10=_maybe_float(raw.get("confidence_p10")),
                sessions=int(raw.get("sessions", 0)),
            )
        except (TypeError, ValueError):
            return None


def signals(
    declared: Collection[str],
    classes: Collection[str],
    confidences: Sequence[float] = (),
) -> Signals:
    """Measure the three signals over one surface and one set of sessions.

    `declared` is the agent's live tool list. `classes` is what the model can predict, `ABSTAIN`
    included or not; it makes no difference, because a class that is not a declared tool cannot
    be in the difference either way.

    An empty `declared` gives an unseen share of 0.0. That is the right answer rather than a
    division to guard: an agent declaring no tools has no tool the model is missing.
    """
    live = frozenset(declared)
    unclassed = live - frozenset(classes)
    return Signals(
        unseen_share=len(unclassed) / len(live) if live else 0.0,
        declared=len(live),
        unclassed=len(unclassed),
        confidence_p50=_quantile(confidences, 0.50),
        confidence_p10=_quantile(confidences, 0.10),
        sessions=len(confidences),
    )


@dataclass(frozen=True)
class Verdict:
    """Whether a retrain is due, and the sentence explaining why.

    `reason` is written to be printed. It is the answer to "why did this trigger", which is the
    question the whole ledger exists to make answerable, so it names the signal and both numbers
    rather than saying `drift`.
    """

    due: bool
    reason: str
    signal: str | None = None

    @property
    def outcome(self) -> str:
        """The ledger's word for it."""
        return "due" if self.due else "ok"


def verdict(
    now: Signals,
    baseline: Signals | None,
    margin: float = DEFAULT_UNSEEN_SHARE_MARGIN,
    age_days: float | None = None,
    max_age_days: int | None = None,
) -> Verdict:
    """Read the signals against the model's own baseline, and against the clock.

    Both triggers are evaluated here, so that a retrain due on age is as explicable as one due on
    drift and both leave the same shape of record. Neither is evaluated at serving time: the clock
    check lives here rather than in the plugin because a serving process that writes to BigQuery to
    explain itself is a serving dependency this project does not want (DESIGN.md 20.8).

    Drift is checked before age. A model that no longer fits its surface should say so, rather than
    reporting the less interesting of two true things.
    """
    if baseline is None:
        return Verdict(
            due=False,
            reason=(
                "no baseline recorded at publish, so drift cannot be measured. "
                f"Unseen share is {now.unseen_share:.1%} today. Republish to set one."
            ),
        )

    moved = now.unseen_share - baseline.unseen_share
    if moved > margin:
        return Verdict(
            due=True,
            signal="unseen_share",
            reason=(
                f"unseen share is {now.unseen_share:.1%} against a {baseline.unseen_share:.1%} "
                f"baseline, {moved:+.1%}, over the {margin:.0%} margin. Tools have been declared "
                "that this model has no class for."
            ),
        )

    if max_age_days is not None and age_days is not None and age_days > max_age_days:
        return Verdict(
            due=True,
            signal="age",
            reason=(
                f"the artifact is {age_days:.0f} days old against a {max_age_days}-day limit. "
                f"Nothing has drifted: unseen share is {now.unseen_share:.1%} against "
                f"{baseline.unseen_share:.1%}."
            ),
        )

    return Verdict(
        due=False,
        reason=(
            f"unseen share is {now.unseen_share:.1%} against a {baseline.unseen_share:.1%} "
            f"baseline, {moved:+.1%}, inside the {margin:.0%} margin."
        ),
    )


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank quantile. None on an empty sequence, which is not the same as zero.

    Nearest-rank rather than interpolated because these are reported to a human beside a
    baseline computed the same way, and the only thing that matters is that both are computed
    the same way.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


def _maybe_float(value: Any) -> float | None:  # noqa: ANN401
    return None if value is None else float(value)
