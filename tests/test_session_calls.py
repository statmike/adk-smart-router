"""`session_calls` remembers which sessions have already sent a model call.

Three lines of logic, and every one of them is an off-by-one waiting to happen. The
question is always about the prefix a request will *meet*, so the answer has to be read
before the send is recorded, and an evicted scope has to read as never-sent rather than
as a later turn — losing a sample is cheap and counting a cold call as warm is the whole
failure this replaced.
"""

from __future__ import annotations

from adk_smart_router.session_calls import SessionCalls

A = "scope-a"
B = "scope-b"


def test_a_scope_nobody_has_sent_for_is_on_its_first_call():
    assert SessionCalls().is_first(A) is True


def test_only_the_first_call_of_a_session_reads_as_first():
    calls = SessionCalls()
    assert calls.is_first(A) is True
    calls.sent(A)
    for _ in range(5):
        assert calls.is_first(A) is False
        calls.sent(A)


def test_sessions_are_tracked_independently():
    calls = SessionCalls()
    calls.sent(A)
    assert calls.is_first(A) is False
    assert calls.is_first(B) is True


def test_an_evicted_scope_reads_as_a_first_call_again():
    """The conservative direction. A wrongly-first call is dropped from the comparison;
    a wrongly-later one would put a cold, prefix-less call into it as if it were warm."""
    calls = SessionCalls(capacity=2)
    calls.sent(A)
    calls.sent(B)
    calls.sent("scope-c")
    assert calls.is_first(A) is True
    assert calls.is_first(B) is False


def test_recording_a_send_refreshes_a_scope_against_eviction():
    """A session in the middle of a long conversation must not be evicted by the sessions
    that started after it."""
    calls = SessionCalls(capacity=2)
    calls.sent(A)
    calls.sent(B)
    calls.sent(A)
    calls.sent("scope-c")  # evicts B, the least recently sent
    assert calls.is_first(A) is False
    assert calls.is_first(B) is True
