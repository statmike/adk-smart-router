"""An opt-in variant of the tool surface whose tools return documents and take time.

The reference agent's stubs return a median of 95 characters and come back in 2 ms. That
makes two questions unmeasurable. Tool results are fed back into the next prompt, so the
tools a session calls are part of what it is billed for; with 95-character results there
is no payload bill to move, and `report_ledger.py --response-chars` has to price a
counterfactual instead of reading a measurement. And with a 2 ms tool against a 2,949 ms
model call there is no latency for a saved call to save. ROADMAP calls this E2.

This module does not rewrite the hundred tools. It wraps them, so the *declarations* are
byte-identical: same names, same signatures, same docstrings, therefore the same schema
the model reads and the same tokens the router prunes. Only what comes back changes. That
is what makes the comparison honest, because the published router artifact and its menus
apply unchanged and nothing has to be retrained to run the experiment.

Two knobs per tool, both fixed by the tool's name so a rerun reproduces itself:

* **A document.** Read-shaped tools get a `details` field holding a knowledge-base
  article of one to six kilobytes. Write-shaped tools do not, because a real write
  returns a receipt. Errors are never padded: a failed call is cheap in real systems and
  padding it would invent a saving the router does not earn.
* **A delay.** Every call sleeps. Reads are faster than writes, which is the usual shape
  when the write is doing something on the other side.

The wrappers are async and sleep with `asyncio.sleep`. A blocking sleep would stall the
event loop, and since the A/B driver runs several sessions at once that would charge one
session's tool latency to every other session's wall clock.

Off by default. `REFERENCE_AGENT_HEAVY_TOOLS=1` turns it on; see `agent.py`.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
from collections.abc import Callable, Sequence
from typing import Any

# The field the document arrives in. One name for every tool, so a query can measure the
# payload channel without knowing which domain it came from.
DOCUMENT_FIELD = "details"

# Verbs that start a tool that changes something. Everything else counts as a read.
WRITE_VERBS = (
    "add",
    "apply",
    "assign",
    "cancel",
    "clear",
    "create",
    "delete",
    "enroll",
    "escalate",
    "extend",
    "issue",
    "open",
    "pause",
    "redeem",
    "refund",
    "register",
    "remove",
    "reschedule",
    "resend",
    "reset",
    "resume",
    "revoke",
    "schedule",
    "send",
    "set",
    "start",
    "submit",
    "transfer",
    "update",
    "upgrade",
)

# Document sizes in characters, and how often each is drawn. Most reads return a record
# with a paragraph attached; a minority return something closer to a policy page. The
# weights are the shape of a support surface rather than a measurement of one, and they
# are stated here so a reader can disagree with them in one place.
DOCUMENT_SIZES = ((0, 3), (1_200, 4), (6_000, 2))

# Latency in milliseconds, low and high of a uniform draw. Reads hit a cache or an index;
# writes go through something slower and sometimes a queue.
READ_LATENCY_MS = (120, 900)
WRITE_LATENCY_MS = (250, 1_400)

_SENTENCES = (
    "Requests are served from the {subject} record as it stood at the start of the "
    "current billing period, so a change made in the last few minutes may not appear here yet.",
    "Where the {subject} record and the customer's own statement disagree, support staff "
    "should take the customer's statement as correct and note the discrepancy on the case.",
    "A {subject} entry becomes eligible for review thirty days after it is created and "
    "stays eligible for a further eleven months, after which it is archived and read-only.",
    "Customers on the standard plan may raise two {subject} requests per calendar month; "
    "customers on the business plan are not limited, but bulk requests are queued.",
    "If the {subject} lookup returns nothing, confirm the identifier with the customer "
    "before escalating, because a mistyped identifier is by far the most common cause.",
    "Amounts shown against a {subject} are inclusive of tax and exclusive of any shipping "
    "adjustment applied later in fulfilment.",
    "Changes to a {subject} propagate to the customer's notification preferences within "
    "one hour, and to partner systems overnight.",
    "This article covers the {subject} surface only. Neighbouring surfaces have their own "
    "eligibility rules and should not be assumed to behave the same way.",
    "Historical {subject} data before the migration is available on request but is not "
    "returned by this endpoint, which reads the current store only.",
    "An agent may override the {subject} outcome once per case. A second override requires "
    "a supervisor and is recorded against both accounts.",
    "The {subject} identifier is stable for the life of the account and is safe to quote "
    "back to the customer in writing.",
    "Rounding on any {subject} total follows the account's currency, and a half-unit "
    "always rounds toward the customer.",
)


def _digest(name: str) -> int:
    """A stable number per tool name. `hash()` is salted per process and would not do."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def is_write(name: str) -> bool:
    """True if the tool's name starts with a verb that changes something."""
    return name.split("_", 1)[0] in WRITE_VERBS


def subject_of(name: str) -> str:
    """The noun a tool is about, for the document to talk about.

    `get_purchase_order_status` is about a purchase order, so drop the leading verb and
    the trailing field, and read what is left. It only has to be plausible; nothing
    depends on it being right.
    """
    words = [word for word in name.split("_")[1:] if word not in ("status", "details", "info")]
    return " ".join(words) or "account"


def document_chars(name: str) -> int:
    """How many characters of article this tool returns, fixed by its name."""
    if is_write(name):
        return 0
    total = sum(weight for _, weight in DOCUMENT_SIZES)
    draw = _digest(name) % total
    for size, weight in DOCUMENT_SIZES:
        if draw < weight:
            return size
        draw -= weight
    return 0  # pragma: no cover - the loop always returns


def latency_seconds(name: str) -> float:
    """How long this tool takes, fixed by its name."""
    low, high = WRITE_LATENCY_MS if is_write(name) else READ_LATENCY_MS
    # A second digest, so a tool's size and its speed are not the same coin flip.
    return (low + _digest(name + ":latency") % (high - low)) / 1000.0


def document(name: str, chars: int) -> str:
    """A knowledge-base article about this tool's subject, at roughly `chars` characters.

    Filler, and it should read as filler: the point is the token count going back into
    the next prompt, not the prose. It is written as coherent policy sentences rather
    than as noise because the model reads it, and a wall of gibberish would change how
    the session behaves for a reason that has nothing to do with payload size.
    """
    if chars <= 0:
        return ""
    subject = subject_of(name)
    seed = _digest(name)
    parts: list[str] = []
    length = 0
    index = 0
    while length < chars:
        sentence = _SENTENCES[(seed + index) % len(_SENTENCES)].format(subject=subject)
        parts.append(sentence)
        length += len(sentence) + 1
        index += 1
    return " ".join(parts)


def heavier(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap one tool so it takes time and, if it is a read, returns a document.

    The signature is copied onto the wrapper explicitly rather than left to
    `functools.wraps`. ADK builds the function declaration from the signature, and the
    whole design of this module depends on that declaration coming out unchanged, so it
    should not rest on whether the framework happens to follow `__wrapped__`.
    """
    name = func.__name__
    chars = document_chars(name)
    delay = latency_seconds(name)
    article = document(name, chars)

    # ANN401: a pass-through over a hundred different signatures has nothing narrower to
    # say. The declared signature is copied from the wrapped function two lines below,
    # which is the type information that actually reaches the model.
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        await asyncio.sleep(delay)
        result = func(*args, **kwargs)
        if article and isinstance(result, dict) and "error" not in result:
            return {**result, DOCUMENT_FIELD: article}
        return result

    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    return wrapper


def heavy(tools: Sequence[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """The whole surface, wrapped. Order is preserved, because declaration order is."""
    return [heavier(func) for func in tools]
