"""Tests for the closed loop: is pruning actually cheaper for this agent?

`CacheWatch` is a pair of running totals and one comparison, so everything here is
arithmetic. What the tests are really pinning down is when it refuses to answer, because
a watch that demotes an agent on thin evidence is expensive: the agent stops pruning for
a cool-off measured in hundreds of calls, and a wrong demotion costs a whole run.

Three scenarios are worth reading first. `test_a_prune_that_removes_tokens_can_still_be_worse`
is the G1 run of DESIGN.md 18.16 in miniature, the case the hit-rate version waved through.
`test_the_pooled_average_of_run_18_22_says_pruning_is_cheaper_and_it_is_not` is the case the
warm-gate version waved through, and it is the reason first calls are excluded. The
re-promotion section at the bottom is about a different question entirely: what stops a
demotion and a re-promotion from arguing with each other forever.
"""

from __future__ import annotations

from adk_smart_router.cache_watch import CacheWatch, costs_more


def feed(
    watch: CacheWatch,
    agent: str,
    *,
    pruned: bool,
    calls: int,
    prompt: int,
    cached: int,
    first_call: bool = False,
) -> None:
    for _ in range(calls):
        watch.observe(
            agent,
            pruned=pruned,
            prompt_tokens=prompt,
            cached_tokens=cached,
            first_call=first_call,
        )


def demote(watch: CacheWatch, agent: str = "root", *, min_samples: int = 10) -> None:
    """Feed the smallest arms that earn a demotion: 800 billed against 3000."""
    feed(watch, agent, pruned=False, calls=min_samples, prompt=4000, cached=3800)
    feed(watch, agent, pruned=True, calls=min_samples, prompt=3000, cached=0)


def test_a_fresh_watch_has_no_opinion_about_anyone():
    assert CacheWatch().suppressed("root") is False


# --- The rule the watch asks, on its own -------------------------------------
#
# `costs_more` was lifted out of `CacheWatch._harmful` so `scripts/fleet_cache_watch.py`
# could ask the same question of a warehouse aggregate over every replica. Two copies of
# this comparison would be two thresholds, and an offline gate that refused what the live
# guard allows is a second opinion rather than a wider view of the same one
# (DESIGN.md 19.75).


def test_the_rule_is_a_ratio_and_not_a_difference_of_averages():
    """Cross-multiplied, so the threshold does not move with how the calls happen to divide
    between the two arms. Ten pruned calls against a thousand unpruned ones is the same 20%
    loss as a thousand against a thousand."""
    assert costs_more(10 * 1_200.0, 10, 1_000 * 1_000.0, 1_000, 0.05) is True
    assert costs_more(1_000 * 1_200.0, 1_000, 10 * 1_000.0, 10, 0.05) is True


def test_an_arm_with_no_calls_in_it_is_never_harmful():
    """There is nothing to compare against. Unmeasured is not a failure, in the guard for
    the same reason it is not one in the publish gate."""
    assert costs_more(0.0, 0, 1_000.0, 1, 0.05) is False
    assert costs_more(1_000_000.0, 1, 0.0, 0, 0.05) is False


def test_the_watch_and_the_gate_reach_the_same_verdict_from_the_same_numbers():
    """The live guard, fed two arms, must agree with the rule the publish gate calls
    directly. This is the seam where the two could drift apart."""
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2000, cached=0)
    feed(watch, "root", pruned=True, calls=10, prompt=2100, cached=0)

    assert watch.suppressed("root") is costs_more(10 * 2_100.0, 10, 10 * 2_000.0, 10, 0.05)


def test_a_prune_that_removes_tokens_can_still_be_worse():
    """DESIGN.md 18.16, scaled down. Pruning cut the prompt by 28% and cost 7.5% more,
    because the tokens it removed were already billing at a tenth."""
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=3916, cached=3325)  # 923 billed
    feed(watch, "root", pruned=True, calls=10, prompt=2808, cached=2016)  # 993 billed
    assert watch.suppressed("root") is True


def test_the_same_prune_is_fine_if_it_holds_its_cache():
    """Same prune, same prompt sizes, cache hit rate held at the control's. -28%."""
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=3916, cached=3325)  # 923 billed
    feed(watch, "root", pruned=True, calls=10, prompt=2808, cached=2384)  # 662 billed
    assert watch.suppressed("root") is False


def test_pruning_that_kills_caching_outright_demotes_the_agent():
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=4000, cached=3800)
    feed(watch, "root", pruned=True, calls=10, prompt=3000, cached=0)
    assert watch.suppressed("root") is True


def test_breaking_even_is_already_a_failure():
    """The router exists to spend less. Costing the same is not a reason to keep paying
    for the embedding call, so only the noise margin protects it."""
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2000, cached=0)
    feed(watch, "root", pruned=True, calls=10, prompt=2101, cached=0)
    assert watch.suppressed("root") is True


def test_a_difference_inside_the_margin_is_treated_as_noise():
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2000, cached=0)
    feed(watch, "root", pruned=True, calls=10, prompt=2080, cached=0)  # 4% worse
    assert watch.suppressed("root") is False


def test_the_margin_is_inclusive():
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2000, cached=0)
    feed(watch, "root", pruned=True, calls=10, prompt=2100, cached=0)  # exactly 5%
    assert watch.suppressed("root") is True


def test_the_cached_multiplier_decides_the_verdict():
    """The same two arms, priced two ways. At a 10% multiplier the removed tokens were
    nearly free and the prune loses; at full price they were worth removing and it wins."""
    cheap = CacheWatch(min_samples=10, margin=0.05, cached_multiplier=0.10)
    dear = CacheWatch(min_samples=10, margin=0.05, cached_multiplier=1.00)
    for watch in (cheap, dear):
        feed(watch, "root", pruned=False, calls=10, prompt=4000, cached=3800)
        feed(watch, "root", pruned=True, calls=10, prompt=3000, cached=2400)
    assert cheap.suppressed("root") is True
    assert dear.suppressed("root") is False


def test_one_arm_alone_is_never_enough():
    """200 ruinous pruned calls still say nothing without a control to compare against."""
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=True, calls=200, prompt=3000, cached=0)
    assert watch.suppressed("root") is False
    # The holdout supplies the other arm, and the verdict lands as soon as it does.
    feed(watch, "root", pruned=False, calls=10, prompt=4000, cached=3900)
    assert watch.suppressed("root") is True


def test_a_thin_control_arm_cannot_demote_a_busy_agent():
    watch = CacheWatch(min_samples=25, margin=0.05)
    feed(watch, "root", pruned=True, calls=500, prompt=3000, cached=0)
    feed(watch, "root", pruned=False, calls=3, prompt=4000, cached=3900)
    assert watch.suppressed("root") is False


def test_a_response_with_no_prompt_count_is_not_a_sample():
    watch = CacheWatch(min_samples=1, margin=0.05)
    watch.observe("root", pruned=True, prompt_tokens=0, cached_tokens=0, first_call=False)
    watch.observe("root", pruned=False, prompt_tokens=0, cached_tokens=0, first_call=False)
    assert watch.billed("root") == (0.0, 0.0, 0, 0)


def test_more_cached_than_prompt_is_clamped_not_believed():
    """Defensive. A cached count above the prompt count would price the call negative and
    make an arm look free, which is the one direction that silently keeps a bad prune on."""
    watch = CacheWatch(min_samples=1, margin=0.05)
    watch.observe("root", pruned=True, prompt_tokens=1000, cached_tokens=9999, first_call=False)
    pruned_cost, _, _, _ = watch.billed("root")
    assert pruned_cost == 100.0


def test_agents_are_judged_separately():
    watch = CacheWatch(min_samples=10, margin=0.05)
    for agent, cached_when_pruned in (("billing", 0), ("search", 2900)):
        feed(watch, agent, pruned=False, calls=10, prompt=4000, cached=3800)
        feed(watch, agent, pruned=True, calls=10, prompt=3000, cached=cached_when_pruned)
    assert watch.suppressed("billing") is True
    assert watch.suppressed("search") is False


def test_billed_reports_both_arms_for_the_operator():
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=4, prompt=1000, cached=1000)  # 100 each
    feed(watch, "root", pruned=True, calls=2, prompt=1000, cached=0)  # 1000 each
    assert watch.billed("root") == (1000.0, 100.0, 2, 4)


def test_hit_rates_are_still_reported_as_context():
    watch = CacheWatch(min_samples=10, margin=0.05)
    feed(watch, "root", pruned=False, calls=8, prompt=1000, cached=900)
    feed(watch, "root", pruned=False, calls=2, prompt=1000, cached=0)
    feed(watch, "root", pruned=True, calls=10, prompt=800, cached=0)
    pruned_hits, whole_hits = watch.hit_rates("root")
    assert (pruned_hits, whole_hits) == (0.0, 0.8)


def test_an_unseen_agent_reports_zeroes_not_an_error():
    assert CacheWatch().billed("nobody") == (0.0, 0.0, 0, 0)
    assert CacheWatch().hit_rates("nobody") == (0.0, 0.0)
    assert CacheWatch().coverage("nobody") == (0, 0)
    assert CacheWatch().demotions("nobody") == 0
    assert CacheWatch().cooloff("nobody") == 0


# --- Excluding a session's first call -----------------------------------------
#
# A first call has no prefix of its own to hit, so it bills about three times a
# continuation. Pooling first and later calls together makes an arm's average a statement
# about what fraction of it was turn one, and the two arms never have the same fraction.
# DESIGN.md 18.22 is the case where that inverted the verdict outright.
#
# The tests below feed billed-equivalent tokens directly, as uncached prompts, because the
# split of a call's price between cached and uncached tokens is not what is under test here.

# DESIGN.md 18.22, from BigQuery: (pruned, first call, how many, billed each).
RUN_18_22 = (
    (False, True, 446, 1941),
    (False, False, 1063, 660),
    (True, True, 148, 2443),
    (True, False, 897, 782),
)


def interleave(buckets: tuple[tuple[bool, bool, int, int], ...]) -> list[tuple[bool, bool, int]]:
    """Spread each bucket evenly over one timeline, in the order a run's calls arrive.

    Feeding the buckets one after another would test something else. The watch re-checks its
    verdict on every call, so a replay that hands it every first call before any later one
    lets a transient decide the outcome, and the whole point of these two runs is what the
    arms say once the mix has settled.
    """
    stream = [
        ((rank + 0.5) / calls, pruned, first, billed)
        for pruned, first, calls, billed in buckets
        for rank in range(calls)
    ]
    stream.sort(key=lambda row: row[0])
    return [(pruned, first, billed) for _, pruned, first, billed in stream]


def test_a_session_first_call_is_not_evidence_about_pruning():
    """Both arms, not just the pruned one. Excluding first calls from one side only would
    be worse than excluding neither: the mix difference is the whole confound."""
    watch = CacheWatch(min_samples=3, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2261, cached=2100, first_call=True)
    feed(watch, "root", pruned=True, calls=10, prompt=2404, cached=0, first_call=True)
    assert watch.suppressed("root") is False
    assert watch.billed("root") == (0.0, 0.0, 0, 0)
    assert watch.coverage("root") == (0, 20)


def test_later_calls_in_the_same_sessions_still_demote():
    """The exclusion delays the verdict by one call per session, it does not remove it."""
    watch = CacheWatch(min_samples=3, margin=0.05)
    feed(watch, "root", pruned=False, calls=10, prompt=2261, cached=2100, first_call=True)
    feed(watch, "root", pruned=True, calls=10, prompt=2404, cached=0, first_call=True)
    assert watch.suppressed("root") is False
    feed(watch, "root", pruned=False, calls=3, prompt=2261, cached=2100)
    feed(watch, "root", pruned=True, calls=3, prompt=2404, cached=0)
    assert watch.suppressed("root") is True


def test_coverage_reports_both_comparisons():
    """Which one is carrying the agent is the operator's question, so both are reported."""
    watch = CacheWatch(min_samples=3, margin=0.05)
    feed(watch, "root", pruned=True, calls=7, prompt=2404, cached=0, first_call=True)
    feed(watch, "root", pruned=True, calls=2, prompt=2404, cached=0)
    assert watch.coverage("root") == (2, 7)


# --- The single-call agent -----------------------------------------------------------


def open_sessions(watch: CacheWatch, agent: str = "root", *, calls: int = 20) -> None:
    """Feed only session-opening calls, the pruned side losing its cache outright."""
    feed(watch, agent, pruned=False, calls=calls, prompt=4000, cached=3800, first_call=True)
    feed(watch, agent, pruned=True, calls=calls, prompt=3000, cached=0, first_call=True)


def test_an_agent_whose_sessions_are_one_call_long_now_gets_a_verdict():
    """The hole the exclusion left. Every call is a first call, and pruning is losing."""
    watch = CacheWatch(min_samples=3, margin=0.05, first_call_min_samples=20)
    open_sessions(watch)
    assert watch.suppressed("root") is True
    assert watch.verdict_basis("root") == "first"


def test_the_fallback_needs_its_own_larger_sample_before_it_will_speak():
    watch = CacheWatch(min_samples=3, margin=0.05, first_call_min_samples=20)
    open_sessions(watch, calls=19)
    assert watch.suppressed("root") is False
    open_sessions(watch, calls=1)
    assert watch.suppressed("root") is True


def test_the_fallback_holds_pruning_to_a_wider_margin():
    """1000 against 920 is 8.7% dearer: past the main margin, inside the first-call one."""
    watch = CacheWatch(min_samples=3, first_call_min_samples=5, margin=0.05, first_call_margin=0.15)
    feed(watch, "root", pruned=False, calls=5, prompt=1000, cached=100, first_call=True)
    feed(watch, "root", pruned=True, calls=5, prompt=1000, cached=0, first_call=True)
    assert watch.suppressed("root") is False


def test_a_multi_turn_agent_never_reaches_the_fallback():
    """Once the main comparison has its samples it owns the verdict, whatever turn one says."""
    watch = CacheWatch(min_samples=3, margin=0.05, first_call_min_samples=20)
    feed(watch, "root", pruned=False, calls=3, prompt=1000, cached=900)
    feed(watch, "root", pruned=True, calls=3, prompt=1000, cached=900)
    open_sessions(watch)
    assert watch.suppressed("root") is False


def test_the_main_comparison_is_still_asked_first_when_both_could_fire():
    watch = CacheWatch(min_samples=3, margin=0.05, first_call_min_samples=3)
    open_sessions(watch, calls=3)
    demote(watch, min_samples=3)
    assert watch.verdict_basis("root") == "first"


def test_a_first_call_verdict_cools_off_on_first_calls():
    """A single-call agent has no later calls, so a cool-off counted in them never ends."""
    watch = CacheWatch(
        min_samples=3, margin=0.05, first_call_min_samples=20, cooloff_calls=3, max_repromotions=1
    )
    open_sessions(watch)
    assert watch.suppressed("root") is True
    feed(watch, "root", pruned=False, calls=100, prompt=1000, cached=900)
    assert watch.suppressed("root") is True  # later calls are not this demotion's currency
    feed(watch, "root", pruned=False, calls=20, prompt=1000, cached=900, first_call=True)
    assert watch.suppressed("root") is False


def test_the_first_call_cooloff_is_stretched_to_cost_the_same_share_of_traffic():
    """3 calls at min_samples 3 becomes 30 at first_call_min_samples 30: the same 9%."""
    watch = CacheWatch(
        min_samples=3, margin=0.05, first_call_min_samples=30, cooloff_calls=3, max_repromotions=1
    )
    open_sessions(watch, calls=30)
    assert watch.cooloff("root") == 30


def test_a_re_promoted_single_call_agent_starts_from_empty_arms():
    watch = CacheWatch(
        min_samples=3, margin=0.05, first_call_min_samples=20, cooloff_calls=1, max_repromotions=1
    )
    open_sessions(watch)
    # The cool-off is stretched by the same 20/3 the sample requirement is, so it is 6.
    feed(watch, "root", pruned=False, calls=6, prompt=1000, cached=900, first_call=True)
    assert watch.suppressed("root") is False
    assert watch.coverage("root") == (0, 0)


def test_the_first_call_arms_are_reported_for_the_operator():
    watch = CacheWatch(min_samples=3, margin=0.05, first_call_min_samples=20)
    open_sessions(watch)
    pruned_cost, whole_cost, n_pruned, n_whole = watch.first_call_billed("root")
    assert (n_pruned, n_whole) == (20, 20)
    assert (pruned_cost, whole_cost) == (3000.0, 580.0)
    assert watch.hit_rates("root", first_call=True) == (0.0, 1.0)
    assert watch.hit_rates("root") == (0.0, 0.0)


def test_an_agent_nobody_demoted_has_no_verdict_basis():
    assert CacheWatch().verdict_basis("nobody") == ""


def test_the_pooled_average_of_run_18_22_says_pruning_is_cheaper_and_it_is_not():
    """The bug this exclusion exists for, at the real call counts of DESIGN.md 18.22.

    Pruning was 25.9% worse on first calls and 18.4% worse on later ones, and the pooled
    average came out 2.4% *cheaper*, because 29.6% of the unpruned arm was first calls
    against 14.2% of the pruned arm. Simpson's paradox: more samples would not have helped,
    because the confound is the composition of the arms and the router sets that itself.
    """
    # `max_repromotions=0` because this test is about the comparison, not the retry: a whole
    # run arrives inside one call to `feed`, so a cool-off would elapse mid-replay.
    stratified = CacheWatch(min_samples=25, margin=0.05, max_repromotions=0)
    pooled = CacheWatch(min_samples=25, margin=0.05, max_repromotions=0)
    for pruned, first, billed in interleave(RUN_18_22):
        stratified.observe(
            "root", pruned=pruned, prompt_tokens=billed, cached_tokens=0, first_call=first
        )
        # The same stream with the exclusion switched off, which is what the watch did to
        # every call whose menu had been sent ten times.
        pooled.observe(
            "root", pruned=pruned, prompt_tokens=billed, cached_tokens=0, first_call=False
        )
    assert stratified.suppressed("root") is True
    assert pooled.suppressed("root") is False
    # 1017 billed per pruned call against 1039 unpruned: the pooled average really does read
    # 2.1% cheaper over the whole run, which is what the shipped rule reported at the time.
    pooled_pruned, pooled_whole, _, _ = pooled.billed("root")
    assert pooled_pruned < pooled_whole


def test_run_18_23_demoted_an_agent_that_was_not_losing_and_no_longer_does():
    """DESIGN.md 18.23, replayed. The pooled arms read +10.4% and the guard fired on its
    minimum sample; restricted to later calls the pruned arm is 14% cheaper and there is
    nothing to fire on. That demotion cost the run and DESIGN 18.23 calls it worthless."""
    watch = CacheWatch(min_samples=50, margin=0.05)
    feed(watch, "root", pruned=False, calls=2013, prompt=709, cached=0)
    feed(watch, "root", pruned=True, calls=50, prompt=610, cached=0)
    assert watch.suppressed("root") is False


def test_fifty_samples_is_the_smallest_arm_that_does_not_false_fire_on_run_18_24():
    """Why `cache_watch_min_samples` is 50 and not 25. Replaying ab-20260910a, at 25 the
    guard fires on an early window reading 695 against 595, on a run whose steady state was
    645 against 702 — pruning winning by 8.3%. Fifty samples outlast that window."""
    early = ((True, 25, 695), (False, 32, 595))
    hasty = CacheWatch(min_samples=25, margin=0.05)
    patient = CacheWatch(min_samples=50, margin=0.05)
    for pruned, calls, billed in early:
        feed(hasty, "root", pruned=pruned, calls=calls, prompt=billed, cached=0)
        feed(patient, "root", pruned=pruned, calls=calls, prompt=billed, cached=0)
    assert hasty.suppressed("root") is True
    assert patient.suppressed("root") is False
    # And the rest of the run, which is where the truth was all along.
    feed(patient, "root", pruned=True, calls=319, prompt=641, cached=0)
    feed(patient, "root", pruned=False, calls=722, prompt=707, cached=0)
    assert patient.suppressed("root") is False


def test_an_arm_made_entirely_of_first_calls_waits_for_the_fallback_and_not_the_main_rule():
    """The exclusion's hole, now the fallback's job. Pruning is costing this agent ten times
    the unpruned price and the main comparison still will not say so at any sample count; what
    finally speaks is the first-call rule, once it has the 500 calls per arm it asks for."""
    watch = CacheWatch(min_samples=3, margin=0.05)
    feed(watch, "root", pruned=False, calls=100, prompt=500, cached=0, first_call=True)
    feed(watch, "root", pruned=True, calls=100, prompt=5000, cached=0, first_call=True)
    assert watch.suppressed("root") is False
    assert watch.coverage("root") == (0, 200)
    feed(watch, "root", pruned=False, calls=400, prompt=500, cached=0, first_call=True)
    feed(watch, "root", pruned=True, calls=400, prompt=5000, cached=0, first_call=True)
    assert watch.suppressed("root") is True
    assert watch.verdict_basis("root") == "first"


# --- Coming back from a demotion ----------------------------------------------
#
# A demotion used to be permanent, on the argument that re-promoting on a thin sample
# oscillates and every oscillation is a cache break. Both halves of that are still true;
# what changed is that "permanent" is also wrong, because the conditions that earn a
# demotion are not permanent — a model version moves, a menu set is republished — and an
# agent stuck in passthrough until someone notices is an open-ended silent loss.
#
# So the demotion lifts, a bounded number of times, on a cool-off counted in the agent's
# own calls. The bound is what makes this a mechanism rather than a flap: the cool-off
# doubles each cycle and the retries run out.


def test_a_demoted_agent_prunes_again_once_it_has_paid_its_cooloff():
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=20)
    demote(watch)
    assert watch.suppressed("root") is True
    feed(watch, "root", pruned=False, calls=19, prompt=1000, cached=900)
    assert watch.suppressed("root") is True
    feed(watch, "root", pruned=False, calls=1, prompt=1000, cached=900)
    assert watch.suppressed("root") is False


def test_the_cooloff_is_counted_in_calls_so_a_silent_agent_never_comes_back():
    """No clock, on purpose. An agent seeing no traffic has nothing to re-measure, and a
    timer would spend its retry on conditions nobody has observed since the demotion."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=20)
    demote(watch)
    assert watch.cooloff("root") == 20
    feed(watch, "other", pruned=False, calls=10_000, prompt=1000, cached=900)
    assert watch.suppressed("root") is True
    assert watch.cooloff("root") == 20


def test_re_promotion_throws_away_the_evidence_that_earned_the_demotion():
    """Otherwise the old arms re-fire the instant the pruned side reaches `min_samples`
    again, and the retry proves nothing because it was outvoted before it started."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=20)
    demote(watch)
    feed(watch, "root", pruned=False, calls=20, prompt=1000, cached=900)
    assert watch.suppressed("root") is False
    assert watch.billed("root") == (0.0, 0.0, 0, 0)
    assert watch.coverage("root") == (0, 0)


def test_a_re_promoted_agent_can_be_demoted_again_on_its_own_fresh_evidence():
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=20)
    demote(watch)
    feed(watch, "root", pruned=False, calls=20, prompt=1000, cached=900)
    assert watch.suppressed("root") is False
    demote(watch)
    assert watch.suppressed("root") is True
    assert watch.demotions("root") == 2


def test_a_demotion_that_is_already_in_force_is_not_counted_twice():
    """Ruinous pruned calls keep arriving after a demotion — requests already in flight,
    and any test feeding the class directly. None of them is a second demotion, so none of
    them may shorten the retry budget or double the next cool-off."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=1000)
    demote(watch)
    feed(watch, "root", pruned=True, calls=200, prompt=9999, cached=0)
    assert watch.demotions("root") == 1
    assert watch.cooloff("root") == 800


def test_each_demotion_doubles_the_cooloff_before_the_next_retry():
    """The retry costs the same absolute number of harmed calls every time, so doubling
    the gap halves the share of traffic an unprunable agent spends on retries."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=4, max_repromotions=5)
    for expected in (4, 8, 16):
        demote(watch)
        assert watch.cooloff("root") == expected
        feed(watch, "root", pruned=False, calls=expected, prompt=1000, cached=900)
        assert watch.suppressed("root") is False


def test_an_agent_that_keeps_earning_demotion_eventually_stops_coming_back():
    """The bound. Two rules that can each undo the other have to terminate, or the agent
    flaps between routed and passthrough for the life of the process. After
    `max_repromotions` retries the demotion is permanent and no amount of traffic lifts it."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=4, max_repromotions=2)
    for _ in range(3):
        demote(watch)
        feed(watch, "root", pruned=False, calls=1000, prompt=1000, cached=900)
    assert watch.demotions("root") == 3
    assert watch.suppressed("root") is True
    assert watch.cooloff("root") == -1
    feed(watch, "root", pruned=False, calls=100_000, prompt=1000, cached=900)
    assert watch.suppressed("root") is True


def test_retries_can_be_switched_off_entirely():
    """`max_repromotions=0` is the old behaviour: one way, until redeploy."""
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=4, max_repromotions=0)
    demote(watch)
    assert watch.cooloff("root") == -1
    feed(watch, "root", pruned=False, calls=10_000, prompt=1000, cached=900)
    assert watch.suppressed("root") is True


# --- A new artifact ---------------------------------------------------------------
#
# The cool-off above answers "was the measurement wrong?". This answers a different
# question: the measurement was right, and then the thing it measured was replaced. Every
# counter the watch holds is about one artifact's menus and declared sets, so when the
# plugin swaps a new version in they are evidence about something that is no longer
# running. Worse, a republish is usually the *repair* — someone saw the demotion, retrained
# with menus that fit under the cache floor, and published. Carrying the verdict across
# that swap keeps the agent in passthrough for the life of the process exactly when it has
# been fixed.


def test_a_new_artifact_lifts_a_demotion_the_old_one_earned():
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=10_000)
    demote(watch)
    assert watch.suppressed("root") is True

    watch.reset("root")

    assert watch.suppressed("root") is False
    assert watch.billed("root") == (0.0, 0.0, 0, 0), "the old version's evidence came along"
    assert watch.verdict_basis("root") == ""


def test_a_new_artifact_is_one_more_chance_and_not_a_fresh_budget_of_them():
    """The bound has to survive the swap or it is not a bound.

    Otherwise an agent that loses money under every version it is given flaps once per
    publish forever. Keeping the lifetime count means the first artifact gets two retries,
    the third gets one demotion and stops, and the total is finite in publishes.
    """
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=4, max_repromotions=2)
    for _ in range(3):
        demote(watch)
        feed(watch, "root", pruned=False, calls=1000, prompt=1000, cached=900)
    assert watch.suppressed("root") is True, "this demotion is permanent, which is the setup"

    watch.reset("root")
    assert watch.suppressed("root") is False

    demote(watch)
    assert watch.demotions("root") == 4
    assert watch.suppressed("root") is True
    assert watch.cooloff("root") == -1, "the new version got its chance and lost it too"


def test_a_new_artifact_for_one_agent_says_nothing_about_another():
    watch = CacheWatch(min_samples=10, margin=0.05, cooloff_calls=10_000)
    demote(watch, "root")
    demote(watch, "billing")

    watch.reset("root")

    assert watch.suppressed("root") is False
    assert watch.suppressed("billing") is True
