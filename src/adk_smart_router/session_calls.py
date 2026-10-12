"""Which sessions have already made a model call, and so whether the next one is their first.

Pure, total, deterministic. No clock, no I/O, no ADK. It remembers scopes and answers one
question: is the request about to go out the first this process has sent for this session?

**Why anyone needs to ask.** `cache_watch` decides whether pruning is costing an agent money
by comparing what pruned and unpruned requests were billed. A session's first model call is
the one call in the conversation with no prefix of its own to hit, and it bills about three
times a continuation as a result. Leaving those calls in the comparison does not add noise, it
adds a confound: the two arms have different mixes of first calls, and the pooled average
moves with the mix rather than with the prune. Measured on the run in DESIGN.md 18.22:

    unpruned  first calls   446   1941 billed
    unpruned  later calls  1063    660 billed
    pruned    first calls   148   2443 billed
    pruned    later calls   897    782 billed

Pruning is worse in both strata and cheaper pooled, because 29.6% of the unpruned arm was
first calls against 14.2% of the pruned arm. That is Simpson's paradox, and the router's own
behaviour is what set the mix, so it is not a transient that waiting out would fix.

Excluding first calls from both arms removes the confound. It does not throw them away:
`cache_watch` keeps them in a second pair of arms and compares first to first, which is the
only comparison an agent whose sessions are one call long can ever have. Both trades are
argued there.

**Sends, not `len(llm_request.contents)`.** Content length is a fact about the conversation,
not about what this process has sent. An app that seeds a session with prior history would
read as a later turn on the very call that establishes its prefix, which is exactly backwards.
Counting our own sends cannot be wrong about our own sends.

The key is the same `rollout.scope_key` a routing decision is filed under, because the prefix
being asked about is per agent as well as per session (DESIGN.md 7.6, D9). In-process and not
in ADK session state, for the reason in DESIGN.md D8: a replica that did not send the first
call does not know what prefix that call established.
"""

from __future__ import annotations

from collections import OrderedDict


class SessionCalls:
    """The scopes that have already sent a model call.

    Not thread-safe, for the same reason `PrefixLock` and `CacheWatch` are not: the operation
    is one insertion, and a lost update costs one call its exclusion rather than corrupting
    anything.
    """

    def __init__(self, *, capacity: int = 4096) -> None:
        """
        Args:
            capacity: how many scopes to remember, least recently sent evicted first. An
                evicted scope reads as never-sent, so its next call is excluded from the
                comparison as if it were a first call. That is the conservative direction:
                the bound loses samples, it never invents bad ones. Size it above the
                sessions in flight; the default matches the plugin's decision cache.
        """
        self._capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()

    def is_first(self, scope: str) -> bool:
        """Would the next call for this scope be its first?

        Ask before recording the send, not after: the question is about the prefix this
        request will meet, which is the one the previous sends left behind.
        """
        return scope not in self._seen

    def sent(self, scope: str) -> None:
        """Record one model call for this scope. Idempotent apart from the LRU order."""
        self._seen[scope] = None
        self._seen.move_to_end(scope)
        while len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
