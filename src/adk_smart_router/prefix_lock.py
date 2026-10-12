"""Which sessions have already committed to a declaration list, and so may no longer change it.

Pure, total, deterministic. No clock, no I/O, no ADK. It remembers scopes and answers one
question: has this session already sent a tool list, making any different list a new prefix?

**Why the router needs a moment, not just an opinion.** Gemini caches the prefix of a request
and the tool declarations sit inside it. The first model call of a session is therefore the
only free moment to choose that list: nothing is cached yet, so any list costs the same. Every
later turn is caching against what the first one sent, and a different list there is not a
smaller prompt, it is a new prefix at full price.

The router used to treat low confidence on turn one as "ask again next turn". That reads as
patience and bills as a cache break. Measured over 245 routed sessions in DESIGN.md 18.22:

    pruned from turn one          turn 2 cached 99.0%      301 billed
    full list turn one, pruned    turn 2 cached  6.7%     2579 billed
    holdout (never pruned)        turn 2 cached 99.3%      392 billed

Same menus, same model, same traffic. The only difference is whether the session had already
sent something else. A session that prunes from its first turn beats the control by 23%; one
that changes its mind on turn two pays 6.6 times what it would have paid by doing nothing. In
that run 97 of 245 routed sessions took the second path, which was enough to turn a working
mechanism into a 17.6% loss overall.

So the rule this class enforces is: **decide on the turn that costs nothing, or do not decide.**
A session whose first call went out unpruned is locked to passthrough for good. That is a
strictly smaller routed arm and a strictly better one, and it also saves the embedding call
those sessions used to spend on a decision they could no longer afford to apply.

The lock is per (app, user, session, agent) scope, the same key the session decision is filed
under, because the prefix it protects is per agent too. It is in-process and not in ADK session
state, for the reason in DESIGN.md D8: a replica that did not send the first turn does not know
what prefix that turn established, and guessing is exactly the mistake being prevented.
"""

from __future__ import annotations

from collections import OrderedDict


class PrefixLock:
    """The scopes that have already sent an unpruned tool list.

    Not thread-safe, for the same reason `SessionCalls` and `CacheWatch` are not: the
    operation is one insertion, and a lost update costs one session its prune rather than
    corrupting anything.
    """

    def __init__(self, *, capacity: int = 4096) -> None:
        """
        Args:
            capacity: how many scopes to remember, least recently locked evicted first.
                An evicted scope reads as unlocked, so the bound trades a rare wrong
                answer for a fixed memory ceiling. Size it above the sessions in flight;
                the default matches the plugin's decision cache.
        """
        self._capacity = capacity
        self._locked: OrderedDict[str, None] = OrderedDict()

    def locked(self, scope: str) -> bool:
        """Has this session already sent a tool list it would now have to change?"""
        return scope in self._locked

    def sent_unpruned(self, scope: str) -> None:
        """Record that a full tool list went out for this scope. Idempotent."""
        self._locked[scope] = None
        self._locked.move_to_end(scope)
        while len(self._locked) > self._capacity:
            self._locked.popitem(last=False)
