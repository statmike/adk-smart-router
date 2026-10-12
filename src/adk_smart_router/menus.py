"""Canonical tool menus: choosing a small fixed set of them, and snapping a session to one.

Pure, total, deterministic. Zero I/O, zero logging, zero clock, zero randomness, zero ADK,
for the same reason `pruner` is: this is a rule, and a rule should be provable in unit tests.

**Why menus exist.** Pruning to the best seven tools for each session gave every session its
own tool declaration list, and a declaration list is part of Gemini's implicit cache key.
DESIGN.md 18.16 measured the result: 40 distinct menus across 65 sessions, each prefix cold
most of the time, 13.1 points of cache hit rate lost, and a prune that removed 28% of the
prompt costing 7.5% more than not pruning at all. The control arm won by sending one
unchanging list to everybody.

The fix is to stop being clever per session. Choose a handful of menus up front, and give
each session the closest one. The cache then sees a handful of prefixes instead of hundreds,
each reused often enough to stay warm, and the token saving is untouched because a menu is
still `max_tools_kept` tools rather than the whole catalogue.

DESIGN.md 18.17 measured what that costs: the coverage ceiling is 0.9878 at 8 menus and
0.9949 at 12, against 0.8938 for one static menu and 1.0 for the bespoke sets. Menus are
close to free, and a single static menu is not.

DESIGN.md 19.10 re-measured the same ceiling in set recall, which is what the gate now asks
for, and reached the same shape: 0.9841 at 8 menus, 0.9930 at 12, at the same predicted
saving. The count is cheap in tokens and not cheap in prefix warmth, so it stays at 8 until
turn-one residency is understood.

The rejected alternative is recorded in `ROADMAP.md` because it is the one a reader will
think of: keeping the same K popular tools first and varying only the tail. It scores better
on paper and does not work, because changing any declaration invalidates the whole cached
prefix including the system instruction ahead of it. There is no partial credit.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Sequence
from math import lcm
from typing import Protocol


class PredictionLike(Protocol):
    """Anything with a label and a probability.

    Declared structurally rather than imported from `predictor` so the two rule modules,
    this one and `pruner`, depend on nothing but each other and `config`. It lives here
    rather than in `pruner` only because `pruner` imports this module and not the reverse.

    The name is deliberately not `Prediction`: that name belongs to the concrete dataclass
    in `predictor`, and a Protocol cannot be instantiated, so sharing the name would hand
    anyone who imported the wrong one a `TypeError`.
    """

    @property
    def label(self) -> str: ...

    @property
    def probability(self) -> float: ...


def choose(
    sessions: Sequence[Sequence[str]],
    size: int,
    count: int,
    candidates: Iterable[frozenset[str]] | None = None,
) -> tuple[frozenset[str], ...]:
    """Pick `count` menus of `size` tools with the best set recall across the corpus.

    Training time only, from historical sessions. `sessions` is one sequence of tool names
    per session; order and repeats are ignored, because the router declares one menu for a
    whole session and a session either has the tool it needs or it does not. Calling a
    missing tool twice is one prune that went wrong, not two.

    The score is the sum over sessions of `|T ∩ M| / |T|`, where `M` is whichever menu suits
    that session best. That is DESIGN.md 19.4's set recall, the quantity the gate measures,
    so the pool is chosen to maximise the number it will be judged on. Dividing by `|T|`
    gives every session one vote: a session that wants five tools cannot outvote five
    sessions that want one each, which is what counting raw calls used to do.

    Greedy: repeatedly add whichever candidate menu improves the total most. Choosing the
    optimal M menus is a set-cover problem and greedy is within a known factor of optimal on
    it, which is a better trade here than an exact solver nobody can read.

    Candidates default to the sessions' own tool sets, padded out to `size` with popular
    tools, so every menu is one that some real session actually wanted. Pass `candidates` to
    supply a different pool to choose from; `bucket` builds the one alternative that has been
    measured. Whatever comes in is trimmed to `size` and deduplicated, because the greedy
    loop below assumes both.

    The loop below is written for speed rather than for the shortest reading, because the
    obvious version does not finish on a large corpus. `_choose_greedily` explains what it
    does instead, and `tests/test_menus.py` holds the plain version and asserts the two
    agree on every corpus it is given.
    """
    if size < 1 or count < 1:
        return ()
    ranked = popularity(sessions)
    if not ranked:
        return ()

    wanted = [frozenset(calls) for calls in sessions if calls]
    if candidates is None:
        offered = {
            frozenset(pad(dict.fromkeys(calls), ranked, size)) for calls in sessions if calls
        }
    else:
        offered = {frozenset(sorted(menu)[:size]) for menu in candidates if menu}
    return _choose_greedily(wanted, offered, count)


def _choose_greedily(
    wanted: Sequence[frozenset[str]], candidates: set[frozenset[str]], count: int
) -> tuple[frozenset[str], ...]:
    """Add whichever candidate raises the total most, until `count` or nothing improves.

    Four departures from the obvious loop. The first is a correctness fix and the rest are
    speed. None of them changes which menus come out on a corpus where the obvious loop was
    well defined, and `tests/test_menus.py` pins that by running a plain version beside this
    one.

    **The score is an integer.** Set recall is `|T & M| / |T|`, and summing those fractions
    in floating point makes the total depend on the order the sessions arrive in. Two menus
    that tie exactly can end up an ulp apart, and then the tie-break this module promises,
    alphabetical on `_order`, never runs: rounding picks the winner instead, and the same
    corpus read back in a different row order picks a different pool. So every session's
    term is scaled by `L / |T|`, where `L` is the least common multiple of the session sizes
    present. That makes each term whole, the total exact, and equal menus equal. Whole terms
    are also what licences the two shortcuts below, because integers may be regrouped and
    reordered freely and floats may not.

    **Sessions that wanted the same tools are counted once.** They score identically against
    every menu, so they collapse to one row carrying how many there were. Most corpora repeat
    themselves heavily: the xLAM corpus has 11,761 distinct tool sets across 60,000 rows.

    **The score is carried, not recomputed.** A session takes whichever menu suits it best,
    so the score of `chosen + [m]` is the sum over rows of `max(best_so_far, score(m))`.
    Holding `best_so_far` per row turns each candidate's trial from a pass over every menu
    chosen so far into a pass over the rows alone.

    **Candidates that cannot win are skipped.** Adding a menu can only lower what another
    menu is still worth, never raise it, so a gain measured in an earlier round is an upper
    bound on the gain now. Candidates sit in a heap on their last measured gain, and a round
    re-measures from the top only until the best gain on hand beats every bound left. The
    heap key carries `_order`, so ties still settle alphabetically.

    Together the three turn a fit that would not have finished in a week on 60,000 sessions
    into one that finishes in minutes. `DESIGN.md` §19.41.
    """
    common = 1
    for tools in wanted:
        common = lcm(common, len(tools))
    repeats = Counter(wanted)
    rows = list(repeats)
    weights = [common // len(tools) * repeats[tools] for tools in rows]
    best_so_far = [0] * len(rows)
    chosen: list[frozenset[str]] = []

    def gain(menu: frozenset[str]) -> int:
        """What adding `menu` would add to the total."""
        return sum(
            max(covered, len(tools & menu) * weight) - covered
            for covered, tools, weight in zip(best_so_far, rows, weights, strict=True)
        )

    # (-gain, order, menu), sorted the way the plain version's `min` sorts, so the first
    # entry out of the heap is the one that version would have picked.
    heap = [(-gain(menu), _order(menu), menu) for menu in candidates]
    heapq.heapify(heap)

    while len(chosen) < count and heap:
        _, order, menu = heapq.heappop(heap)
        fresh = -gain(menu)
        while heap and (fresh, order) > heap[0][:2]:
            # Something still on the heap claims a better gain than this candidate's
            # re-measured one. Put this back and take that instead.
            heapq.heappush(heap, (fresh, order, menu))
            _, order, menu = heapq.heappop(heap)
            fresh = -gain(menu)
        if chosen and fresh == 0:
            break  # nothing left to add improves anything, so stop rather than pad the list
        chosen.append(menu)
        for index, tools in enumerate(rows):
            score = len(tools & menu) * weights[index]
            if score > best_so_far[index]:
                best_so_far[index] = score
    return tuple(chosen)


def rank(preds: Sequence[PredictionLike], menus: Sequence[frozenset[str]]) -> frozenset[str] | None:
    """The menu this prediction vector wants, or None if there is not one to give.

    Scored by the probability mass the menu contains, which is the same quantity
    `pruner.top_k_by_mass` maximises when it builds a bespoke set. Ties go to the
    alphabetically first menu, so every replica snaps the same way and the prefix stays
    shared, which is the entire point of menus.

    There was a `require` parameter, naming a tool the chosen menu had to contain. It
    existed so a session could move to a wider menu mid-conversation rather than mint a
    bespoke prefix, and it went when the growth path it served was deleted from `pruner`
    as unreachable. A menu is now chosen once per session and never revisited.
    """
    if not menus:
        return None
    mass = {pred.label: pred.probability for pred in preds}
    return min(
        menus,
        key=lambda menu: (-sum(mass.get(tool, 0.0) for tool in menu), _order(menu)),
    )


def popularity(sessions: Sequence[Sequence[str]]) -> list[str]:
    """Tools most-called first, ties by name so the order is stable across runs."""
    counts: Counter[str] = Counter()
    for calls in sessions:
        counts.update(calls)
    return sorted(counts, key=lambda tool: (-counts[tool], tool))


def bucket(sessions: Sequence[Sequence[str]], size: int) -> tuple[frozenset[str], ...]:
    """Candidate menus built by merging co-occurring tool sets, rather than padding each one.

    `pad` builds one candidate per session by filling that session's spare slots from the
    global popularity ranking. On our own agent that is free: a session wants 2.6 tools out
    of 25, so the filler lands on tools somebody was going to want anyway. On a wide flat
    surface it is most of the cost. An xLAM query wants 1.4 tools out of 3,179, so every
    candidate is one or two real tools followed by the same nine popular ones, and the pool
    becomes near-copies of the head: 24 menus of 11 held 91 distinct tools between their 264
    slots, and the oracle ceiling fell to 0.2390 (DESIGN.md 19.42).

    This builds candidates the other way round. Each distinct tool set joins the bucket it
    shares the most tools with, if the union still fits in `size`, and starts a new bucket
    otherwise. Spare slots are left empty rather than filled, because the whole failure being
    repaired is filler crowding out real tools; `choose` is free to prefer a fuller menu, and
    on a corpus where filler helps it still has the padded candidates to choose from.

    Most-frequent set first, so the buckets that form early are the ones most traffic lands
    on, and ties by name so the same corpus in a different row order gives the same pool.

    Sets larger than `size` are truncated to their most popular tools. A session wanting more
    tools than a menu holds cannot be covered by any menu, and dropping it entirely would
    lose the tools it shares with sessions that can be.
    """
    if size < 1:
        return ()
    ranked = popularity(sessions)
    if not ranked:
        return ()
    order = {tool: index for index, tool in enumerate(ranked)}

    counts = Counter(frozenset(calls) for calls in sessions if calls)
    arriving = sorted(counts, key=lambda tools: (-counts[tools], _order(tools)))

    buckets: list[set[str]] = []
    # Which buckets hold a given tool, so a set only weighs itself against buckets it could
    # actually overlap. Without it this is every set against every bucket, which on the full
    # xLAM surface is 11,761 by 1,757 and slower than the fit it feeds.
    holding: dict[str, set[int]] = {}

    for tools in arriving:
        wanted = set(sorted(tools, key=lambda tool: order.get(tool, len(order)))[:size])
        nearby = {index for tool in wanted for index in holding.get(tool, ())}
        best, shared = None, 0
        for index in sorted(nearby):
            overlap = len(buckets[index] & wanted)
            if overlap > shared and len(buckets[index] | wanted) <= size:
                best, shared = index, overlap
        if best is None:
            buckets.append(wanted)
            best = len(buckets) - 1
        else:
            buckets[best] |= wanted
        for tool in buckets[best]:
            holding.setdefault(tool, set()).add(best)

    return tuple(frozenset(found) for found in buckets)


def pad(tools: Iterable[str], ranked: Sequence[str], size: int) -> list[str]:
    """Trim to `size`, then fill any spare slots with the most popular tools not already in.

    A menu shorter than `size` wastes slots that cost nothing to fill, and a session that
    used one tool would otherwise produce a one-tool menu that covers almost nothing.
    """
    kept = list(dict.fromkeys(tools))[:size]
    for tool in ranked:
        if len(kept) >= size:
            break
        if tool not in kept:
            kept.append(tool)
    return kept


def _order(menu: frozenset[str]) -> tuple[str, ...]:
    """A total order on menus, used to settle ties the same way everywhere."""
    return tuple(sorted(menu))
