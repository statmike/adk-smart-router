"""How large an agent's prompts settle at, so the floor guard can ask about a session
instead of about its first turn.

`pruner.breaks_cache` withholds a prune when it would carry a cacheable prompt below the
model's cache floor. Asked on turn one, it is being asked at the smallest a session will
ever be: history only grows, so a prompt that sits just under the floor after pruning
clears it a turn or two later, and the guard changes its mind. Changing its mind is the
expensive part. The kept set never moved, but whether it was applied did, and to the
cached prefix those are the same event.

Measured on a 592-session A/B (DESIGN.md 18.19). Of 61 routed sessions, the 22 the guard
interfered with cached their second turn at 14.3% and billed 8.8% *more* than the control.
The 39 it left alone cached their second turn at 99.0%, matching the control's 99.2%, and
billed 3.8% less. The guard was doing more damage than the trap it was built to avoid.

So the guard needs the size a session settles at, not the size it starts at. That is not
predictable from one request, but it is observable across many: every response reports its
prompt size, and the mean over an agent's turns is a fair statement of what that agent's
prompts cost. It runs larger than a first turn, which biases the guard toward pruning, and
that is the direction the measurement says to bias it in.

This is a prior, like the floor itself. `cache_watch.CacheWatch` remains the half of the
guard that learns what an agent was actually billed and can stop it pruning for good.
"""

from __future__ import annotations

from collections import OrderedDict

__all__ = ["PromptSizes"]


class PromptSizes:
    """A running mean of each agent's prompt size, in tokens.

    Not thread-safe, for the same reason `CacheWatch` and `TokenScale` are not: the
    arithmetic is two increments and a lost update costs one sample.
    """

    def __init__(self, *, min_samples: int = 25, capacity: int = 256) -> None:
        """
        Args:
            min_samples: how many turns an agent needs before `typical` will answer.
                Below it the mean is dominated by whichever turns happened to arrive
                first, which for a cold process is first turns, which is the bias this
                module exists to remove.
            capacity: how many agents to remember. An LRU rather than a plain dict so a
                process that sees a long tail of agent names cannot grow without bound.
        """
        self._min_samples = min_samples
        self._capacity = capacity
        self._totals: OrderedDict[str, tuple[int, int]] = OrderedDict()  # agent -> (turns, sum)

    def observe(self, agent: str, prompt_tokens: int) -> None:
        """Record one turn's prompt size. Zero and negative counts are dropped as unreported."""
        if prompt_tokens <= 0:
            return
        turns, total = self._totals.get(agent, (0, 0))
        self._totals[agent] = (turns + 1, total + prompt_tokens)
        self._totals.move_to_end(agent)
        while len(self._totals) > self._capacity:
            self._totals.popitem(last=False)

    def typical(self, agent: str) -> int | None:
        """This agent's mean prompt size, or None until `min_samples` turns have arrived.

        None means "no opinion", and the caller must fall back to measuring the request in
        front of it. It never means "small".
        """
        turns, total = self._totals.get(agent, (0, 0))
        if turns < self._min_samples:
            return None
        return round(total / turns)

    def samples(self, agent: str) -> int:
        """How many turns this agent's mean is drawn from. For logs and tests."""
        return self._totals.get(agent, (0, 0))[0]
