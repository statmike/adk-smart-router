"""Fuzzing for `cache_key`, the one place in the plugin where a bug is a cross-user leak.

`test_tool_cache.py` covers the key by example. Examples are the wrong shape for this
particular function, because the failure it has to rule out is not "the obvious case is
wrong" but "some pair of inputs nobody thought of collides". That is a search, so this file
searches.

The property under test is **injectivity**: two different call identities must never produce
the same key. `cache_key` builds its digest by joining six fields, so the classic way to break
one is to shift a boundary, moving characters from one field into the next so a different
tuple joins to the same string. The length prefix is what stops that, and the alphabet below
is chosen to attack it: separators, digits, and colons are exactly the characters a length
prefix is made of.

One thing deliberately not asserted: that unicode-equivalent principals share a key. `"café"`
composed and decomposed are different strings and get different keys. That direction is a
wasted cache slot rather than a leak, and collapsing them would mean deciding two principals
are the same principal, which is not this function's call to make.
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from adk_smart_router.tool_cache import cache_key, canonical_json

# Fragments chosen to attack the length prefix. `_FIELD_SEP` is the separator itself, the
# digits and colons imitate the `len:value` framing, and the empty string is the degenerate
# field that makes a boundary easiest to shift.
FRAGMENTS = (
    "",
    "a",
    "ab",
    "1",
    "12",
    ":",
    "1:",
    "2:a",
    "\x1f",
    "\x1fa",
    "a\x1f",
    "3:abc\x1f",
    "café",
    unicodedata.normalize("NFD", "café"),
    "üser",
    "𝐀",
)

ARG_POOL: tuple[dict[str, object], ...] = (
    {},
    {"a": 1},
    {"a": "1"},
    {"a": 1.0},
    {"a": True},
    {"a": "true"},
    {"a": None},
    {"a": 1, "b": 2},
    {"b": 2, "a": 1},  # same call as the line above; must share a key, checked separately.
    {"a": [1, 2]},
    {"a": [1, "2"]},
    {"a": {"b": 1}},
    {"a": "\x1f"},
    {"a": "}", "b": "{"},
    {"ab": 1},
    {"a": "b:1"},
)


def field(rng: random.Random) -> str:
    """A field value assembled from one or two adversarial fragments."""
    return rng.choice(FRAGMENTS) + (rng.choice(FRAGMENTS) if rng.random() < 0.5 else "")


def test_fuzz_no_two_distinct_call_identities_share_a_key():
    """~40000 randomised identities, hashed, checked for a collision.

    A hit here is either a sha256 collision or, far more likely, a boundary that can be
    shifted. The assertion prints both colliding tuples because the identity that produced it
    is the whole bug report.

    `user_id` is drawn from the same adversarial alphabet as everything else, so the search
    covers the specific attack this function exists to stop: one principal writing a value
    that makes their key equal another principal's.
    """
    rng = random.Random(20260906)
    seen: dict[str, tuple[object, ...]] = {}

    for _ in range(40000):
        # A non-empty user_id, because the empty one is a documented refusal, not a key.
        user = field(rng) or "u"
        args = rng.choice(ARG_POOL)
        identity = (field(rng), user, field(rng), field(rng), field(rng), canonical_json(args))
        key = cache_key(identity[0], user, identity[2], identity[3], identity[4], args)

        previous = seen.setdefault(key, identity)
        assert previous == identity, f"collision: {previous!r} and {identity!r} share {key}"


def test_fuzz_changing_only_the_principal_always_changes_the_key():
    """The security property in its most direct form, stated as a search rather than a case.

    Everything else about the call is held fixed and only `user_id` moves. If any pair of
    distinct principals maps to one key, the tool's result crosses between them.
    """
    rng = random.Random(20260907)
    principals = {field(rng) or "u" for _ in range(400)}

    for _ in range(200):
        rest = (field(rng), field(rng), field(rng), field(rng))
        args = rng.choice(ARG_POOL)
        keys = {
            cache_key(rest[0], user, rest[1], rest[2], rest[3], args): user for user in principals
        }
        assert len(keys) == len(principals), f"two principals shared a key under {rest!r}"


def test_argument_order_is_not_part_of_the_identity():
    """`{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the same call and must share a key.

    The opposite of the property above, and the reason `canonical_json` sorts. Without it the
    cache would miss whenever a caller built its kwargs in a different order.
    """
    common = ("app", "user", "agent", "tool", "v1")
    assert cache_key(*common, {"a": 1, "b": 2}) == cache_key(*common, {"b": 2, "a": 1})


def test_every_adjacent_pair_of_fields_holds_its_boundary():
    """The boundary attack, aimed one seam at a time instead of scattered over all six.

    Randomising all six fields at once almost never produces a collision even when the framing
    is broken, because the other four fields have to match by luck. So this fixes everything
    except one adjacent pair and sweeps that pair exhaustively over an alphabet made of
    separators and length-prefix lookalikes. If characters can migrate across a seam, two
    different pairs join to one string and this finds it in a few hundred tries.
    """
    seam = (
        "",
        "a",
        "b",
        "ab",
        "\x1f",
        "\x1fa",
        "a\x1f",
        "\x1fb",
        "b\x1f",
        "1",
        "1:",
        "1:a",
        "2:ab",
    )
    base = ["app", "user", "agent", "tool", "v1"]

    for left in range(4):
        seen: dict[str, tuple[str, str]] = {}
        for first in seam:
            for second in seam:
                fields = list(base)
                fields[left] = first
                fields[left + 1] = second
                # Position 1 is `user_id`, where the empty string is a refusal, not a key.
                if not fields[1]:
                    continue
                key = cache_key(fields[0], fields[1], fields[2], fields[3], fields[4], {})
                pair = (first, second)
                previous = seen.setdefault(key, pair)
                assert previous == pair, f"seam {left}: {previous!r} and {pair!r} share {key}"


# Characters that mean something in `canonical_json`'s own output. An argument name containing
# these is the way to try to forge a different argument structure out of one name.
STRUCTURAL = ("", "a", "b", ",", ":", '"', "{", "}", "a:1", "a:1,b", '"a"', "1")


def test_argument_structure_cannot_be_forged_out_of_an_argument_name():
    """`{"a:1,b": 2}` must not serialize to what `{"a": 1, "b": 2}` serializes to.

    `canonical_json` writes a dict as `{name:value,name:value}`, so an argument *name* that
    contains a comma and a colon can imitate a second argument. Quoting the name is what stops
    it. Same attack as the seam test above, one level down and inside the arguments, where the
    separators are `,` `:` `{` `}` rather than the field separator.
    """
    seen: dict[str, dict[str, object]] = {}

    for first in STRUCTURAL:
        for second in STRUCTURAL:
            for value in (1, "1", True, None, "a", [1]):
                for args in ({first: value}, {first: value, second: 2}):
                    encoded = canonical_json(args)
                    previous = seen.setdefault(encoded, args)
                    assert previous == args, (
                        f"collision: {previous!r} and {args!r} both encode to {encoded}"
                    )


def test_an_empty_principal_is_refused_rather_than_shared():
    """Fails closed. A key for the anonymous scope would be one bucket for every caller."""
    with pytest.raises(ValueError, match="non-empty user_id"):
        cache_key("app", "", "agent", "tool", "v1", {})
