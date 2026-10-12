"""Tests for the xLAM traffic driver.

The driver spends money and its live behaviour cannot be tested here, so what is tested is
the planning it does before it spends any: which queries each pass draws, and whether the
two passes can ever touch the same one. Contamination between the corpus pass and the A/B
would not fail loudly, it would show up as an unusually good G1, which is the failure mode
worth a test.
"""

from __future__ import annotations

from run_xlam_traffic import EVAL_SHARE, journal_path, plan_for, pool_of, session_prefix


def rows(count: int) -> list[dict[str, object]]:
    return [{"id": f"row-{i}", "query": f"q{i}", "tools": ["x"]} for i in range(count)]


def test_the_two_pools_share_no_query():
    """The one that matters. If a corpus query reached the A/B, the router would be graded
    partly on traffic it memorised, and nothing downstream would notice."""
    corpus = plan_for(rows(4_000), 500, 1, "corpus", "train")
    evaluation = plan_for(rows(4_000), 500, 1, "ab1", "eval")

    assert {s.row_id for s in corpus}.isdisjoint({s.row_id for s in evaluation})


def test_the_split_does_not_move_between_runs():
    """`hash()` is salted per process, so the obvious implementation would reshuffle the
    split on every invocation and quietly leak last week's training queries into today's
    evaluation."""
    assert pool_of({"id": "row-7"}) == pool_of({"id": "row-7"})


def test_the_split_is_roughly_the_share_it_claims():
    held_out = sum(1 for row in rows(4_000) if pool_of(row) == "eval")

    assert 0.8 <= held_out / (4_000 / EVAL_SHARE) <= 1.2


def test_the_same_seed_sends_the_same_queries():
    first = plan_for(rows(4_000), 50, 20260906, "ab1", "eval")
    second = plan_for(rows(4_000), 50, 20260906, "ab1", "eval")

    assert [s.row_id for s in first] == [s.row_id for s in second]


def test_a_different_seed_sends_different_queries():
    first = plan_for(rows(4_000), 50, 1, "ab1", "eval")
    second = plan_for(rows(4_000), 50, 2, "ab1", "eval")

    assert [s.row_id for s in first] != [s.row_id for s in second]


def test_no_query_is_sent_twice_in_one_run():
    """A repeat would warm its own prefix and hand the second arm to draw it a discount it
    did not earn."""
    plan = plan_for(rows(4_000), 400, 3, "ab1", "eval")

    assert len({s.row_id for s in plan}) == len(plan)


def test_asking_for_more_sessions_than_the_pool_holds_gives_the_whole_pool():
    plan = plan_for(rows(100), 10_000, 4, "corpus", "eval")

    assert 0 < len(plan) < 100


def test_session_ids_are_numbered_so_a_prefix_query_can_find_them():
    """`report_ab.py` matches on the prefix and `run_ab.py`'s window rule reads a trailing
    five-digit counter. This has to keep that shape or the A/B is unreportable."""
    plan = plan_for(rows(4_000), 3, 5, "ab1", "eval")

    assert [s.session_id for s in plan] == ["xlam-ab1-00000", "xlam-ab1-00001", "xlam-ab1-00002"]


def test_the_prefix_cannot_collide_with_the_reference_agents_runs():
    """The reference agent's windows are all `ab-` or `corpus-`. Sharing a namespace would
    let one report pool two different agents into one population."""
    assert session_prefix("ab1").startswith("xlam-")


def test_user_ids_vary_so_the_rollout_hash_has_something_to_work_with():
    plan = plan_for(rows(4_000), 200, 6, "ab1", "eval")

    assert len({s.user_id for s in plan}) > 1


def test_each_label_journals_to_its_own_file():
    assert journal_path(None, "corpus").name == "xlam-corpus.jsonl"
    assert journal_path(None, "ab1").name == "xlam-ab1.jsonl"
