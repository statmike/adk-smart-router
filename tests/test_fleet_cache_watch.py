"""Tests for the fleet-wide cache comparison.

The rule itself is `cache_watch.costs_more` and is tested with the live guard. What this
file pins is the part that is new here: how a fleet of replicas is combined into one number.

The test that matters is the Simpson's paradox one. Pooling every replica's calls together
is the obvious way to write this and it is wrong in exactly the way DESIGN.md 18.22 was
wrong, because replicas do not hold the same mix of routed and holdout sessions. A version
that pooled would pass that test's data and ship a router that is costing money.

Everything else here is guarding the edges around it: what counts as enough evidence, what
happens with no evidence at all, and that the SQL asks for the calls the live guard counts.
"""

from __future__ import annotations

import pytest
from fleet_cache_watch import (
    STRATA_SQL,
    Arm,
    Stratum,
    combine,
    judge,
    standardise,
)

from adk_smart_router.cache_watch import costs_more

MARGIN = 0.05
MIN_SAMPLES = 50


def stratum(replica: str, pruned: tuple[int, float], whole: tuple[int, float]) -> Stratum:
    return Stratum(
        replica_id=replica,
        pruned=Arm(pruned[0], pruned[1]),
        whole=Arm(whole[0], whole[1]),
        versions=1,
    )


def verdict(*strata: Stratum, min_samples: int = MIN_SAMPLES, margin: float = MARGIN):
    return judge("support", 30, strata, min_samples, margin)


# --- The rule, borrowed from the live guard ----------------------------------


def test_the_verdict_is_the_shipped_rule_and_not_a_restatement_of_it():
    """Same function, same threshold. A gate that scored its own version of the comparison
    could refuse a publish the live guard would have allowed, and then nobody knows which
    one to believe (DESIGN.md 19.74)."""
    found = verdict(stratum("a", (100, 110_000), (100, 100_000)))
    assert found.harmful is costs_more(1_100.0, 1, 1_000.0, 1, MARGIN)


def test_a_margin_sized_loss_is_not_harmful_but_a_hair_more_is():
    """The boundary is inclusive on the refusing side, exactly as `costs_more` writes it."""
    assert verdict(stratum("a", (100, 104_000), (100, 100_000))).harmful is False
    assert verdict(stratum("a", (100, 105_000), (100, 100_000))).harmful is True


def test_pruning_that_saves_money_is_never_harmful():
    assert verdict(stratum("a", (100, 80_000), (100, 100_000))).harmful is False


# --- Combining replicas ------------------------------------------------------


def test_one_replica_standardises_to_exactly_what_pooling_says():
    """The degenerate case has to degrade to the old answer. A fleet of one has no mix to
    correct for, and a number that moved when the correction was added would mean the
    correction was doing something other than correcting."""
    found = verdict(stratum("a", (100, 110_000), (200, 200_000)))
    assert found.standardised_pruned == pytest.approx(found.pooled_pruned)
    assert found.standardised_whole == pytest.approx(found.pooled_whole)


def test_pooling_across_replicas_hides_a_loss_that_every_replica_is_taking():
    """Simpson's paradox, the same one that cost this project a wrong read on turn position
    (DESIGN.md 18.22), wearing `replica_id` as its grouping variable.

    Two replicas. One serves short sessions cheaply and happens to hold almost all of the
    routed traffic; the other serves long expensive sessions and holds almost all of the
    holdout. Inside each replica pruning costs 20% more. Pooled, the cheap replica's routed
    calls drag the routed average below the holdout average and pruning looks free.
    """
    cheap = stratum("cheap", pruned=(900, 900 * 1_200.0), whole=(100, 100 * 1_000.0))
    dear = stratum("dear", pruned=(100, 100 * 12_000.0), whole=(900, 900 * 10_000.0))

    pooled_pruned, pooled_whole = combine((cheap, dear))
    assert pooled_pruned < pooled_whole  # the trap: pruning looks cheaper than not pruning

    found = verdict(cheap, dear)
    assert found.standardised_pruned > found.standardised_whole
    assert found.harmful is True


def test_a_replica_is_weighed_by_how_much_of_the_fleets_traffic_it_carried():
    """Direct standardisation. A replica handling nine calls in ten sets nine tenths of the
    answer, so a tiny replica with an extreme ratio cannot swing the verdict on its own."""
    big = stratum("big", pruned=(900, 900 * 1_000.0), whole=(900, 900 * 1_000.0))
    tiny = stratum("tiny", pruned=(100, 100 * 5_000.0), whole=(100, 100 * 1_000.0))

    pruned, whole = standardise((big, tiny))
    assert pruned == pytest.approx(0.9 * 1_000.0 + 0.1 * 5_000.0)
    assert whole == pytest.approx(1_000.0)


# --- What counts as evidence -------------------------------------------------


def test_a_replica_short_of_samples_in_either_arm_does_not_vote():
    """The same floor the live guard applies, applied per replica rather than per process.
    A replica with four holdout calls has an opinion; it does not have evidence."""
    thin = stratum("thin", pruned=(1_000, 1_000 * 5_000.0), whole=(4, 4 * 1_000.0))
    fat = stratum("fat", pruned=(100, 100 * 1_000.0), whole=(100, 100 * 1_000.0))

    found = verdict(thin, fat)
    assert [s.replica_id for s in found.judged] == ["fat"]
    assert found.harmful is False


def test_no_replica_with_both_arms_warm_is_not_measured_and_does_not_refuse():
    """Unmeasured is not zero. An agent with no holdout arm has told us nothing about
    whether pruning is hurting it, and a gate that reads silence as a failure would block
    every first publish."""
    found = verdict(stratum("a", (1_000, 1_000 * 9_000.0), (0, 0.0)))
    assert found.measured is False
    assert found.standardised_pruned is None
    assert found.harmful is False


def test_the_pooled_number_is_kept_even_when_no_replica_is_judgeable():
    """It is what the operator has left to look at, so it is computed over every stratum
    and not only over the judged ones."""
    found = verdict(stratum("a", (10, 10 * 2_000.0), (10, 10 * 1_000.0)))
    assert found.pooled_pruned == pytest.approx(2_000.0)
    assert found.pooled_whole == pytest.approx(1_000.0)


def test_a_fleet_with_nothing_in_it_at_all_measures_nothing():
    found = verdict()
    assert found.measured is False
    assert found.harmful is False
    assert found.pooled_pruned == 0.0
    assert found.versions == 0


def test_an_empty_arm_prices_at_zero_rather_than_dividing_by_zero():
    assert Arm().per_call == 0.0


# --- The query ---------------------------------------------------------------


def test_the_query_counts_later_calls_only():
    """Turn one is where a session's prompt is smallest and where the two arms differ most
    in how many of them they hold. The live guard keeps a separate set of counters for it;
    this drops it from both arms instead, which is the same decision (DESIGN.md 18.22)."""
    assert "ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) AS position" in (
        STRATA_SQL
    )
    assert "WHERE position > 1" in STRATA_SQL


def test_the_query_compares_routed_against_holdout_and_nothing_else():
    """A declined session shipped the full tool list, so it is not evidence about pruning.
    It is also not the control arm, because it was never randomised into one."""
    assert "WHERE s.arm IN ('routed', 'holdout')" in STRATA_SQL


def test_the_query_clamps_cached_tokens_to_the_prompt():
    """BQAA has been seen reporting the two out of order on a truncated response, and an
    unclamped subtraction turns that into a negative bill."""
    assert "LEAST(COALESCE(usage_cached_tokens, 0), COALESCE(usage_prompt_tokens, 0))" in (
        STRATA_SQL
    )


def test_a_record_with_no_replica_id_still_lands_in_a_stratum():
    """Warehouses hold decisions written before the field existed. Bucketing them together
    gives an unstratified verdict, which is the answer this had before it could stratify;
    dropping them would give no verdict at all."""
    assert "IFNULL(replica_id, '{unknown}')" in STRATA_SQL
