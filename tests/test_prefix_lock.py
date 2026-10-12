"""Tests for the prefix lock: a session that has sent a tool list cannot change it.

`PrefixLock` is a bounded set, so there is not much arithmetic to get wrong. What these
pin down is the direction of the rule. It is one-way and it is per scope, because both of
those are what make it safe: a lock that could lift, or that leaked between sessions,
would reintroduce exactly the mid-session prefix change it exists to prevent.
"""

from __future__ import annotations

from adk_smart_router.prefix_lock import PrefixLock


def test_a_session_nobody_has_sent_anything_for_is_free_to_prune():
    assert PrefixLock().locked("scope-a") is False


def test_sending_a_full_list_locks_that_session():
    lock = PrefixLock()
    lock.sent_unpruned("scope-a")
    assert lock.locked("scope-a") is True


def test_the_lock_is_one_way():
    """There is no unlock, on purpose. The prefix a session established is not undone by
    anything that happens later in it."""
    lock = PrefixLock()
    lock.sent_unpruned("scope-a")
    for _ in range(50):
        lock.sent_unpruned("scope-a")
    assert lock.locked("scope-a") is True
    assert not hasattr(lock, "unlock")


def test_sessions_are_locked_separately():
    """The prefix belongs to one session on one agent. A lock that spread would stop the
    router routing anything at all after the first unrouted turn in the process."""
    lock = PrefixLock()
    lock.sent_unpruned("scope-a")
    assert lock.locked("scope-b") is False


def test_recording_the_same_send_twice_is_harmless():
    lock = PrefixLock(capacity=4)
    for _ in range(10):
        lock.sent_unpruned("scope-a")
    assert lock.locked("scope-a") is True


def test_the_table_is_bounded_and_evicts_the_least_recent():
    lock = PrefixLock(capacity=3)
    for scope in ("a", "b", "c", "d"):
        lock.sent_unpruned(scope)
    assert lock.locked("a") is False  # evicted
    assert [lock.locked(s) for s in ("b", "c", "d")] == [True, True, True]


def test_an_evicted_scope_reads_as_unlocked():
    """The failure mode of eviction, stated so it is a decision and not a surprise: a very
    long-running session in a very busy process can lose its lock and prune late. That is
    the old behaviour for one session, not a new failure, and the bound is what keeps the
    table from growing without limit."""
    lock = PrefixLock(capacity=1)
    lock.sent_unpruned("old")
    lock.sent_unpruned("new")
    assert lock.locked("old") is False
