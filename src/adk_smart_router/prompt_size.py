"""How large a request is, in the tokens the model will bill for.

The router asks this for one reason: would dropping these tool declarations push the
prompt under the model's context-cache floor? Below the floor nothing is cached and
pruning is free. Above it, every kept token bills at about a tenth, and a prune that
crosses the line downward turns that discount off for the rest of the session. Getting
the answer wrong in the permissive direction is expensive, and it is the mistake v1
shipped (DESIGN.md 18.12).

The quantity is the **whole prompt**, not the tool prefix. That is the correction.
Conversation history is billed and cached alongside the system instruction and the
declarations, and it grows every turn, so a request comfortably under the floor at turn
one can be well over it by turn three. No number derived from a tool count can see
that, however well calibrated it is.

Prose and schema are counted separately
---------------------------------------
Tool declarations are JSON schema and tokenize about twice as densely as conversation.
Measured on the reference agent by sweeping the declared tool count from 2 to 25 and
reading `prompt_token_count` back:

    declarations   8.33 characters per token
    conversation   4.24 characters per token

A single blended ratio cannot fit both, and which one dominates is exactly what a prune
changes, so the two terms are kept apart.

Both ratios are calibrated at runtime, and the schema one had to be. It was fixed at
8.33 on the ground that it is a property of how BPE handles JSON and therefore close to
uniform across agents, and that is false: the reference agent's original 25 declarations
measure 8.30 and the twelve domains added later measure between 5.75 and 6.03, a 44%
spread inside one repository (DEV_NOTES.md, 2026-09-04). Held fixed at 8.33 on the
100-tool surface, the estimate is 26% low on declarations, which understates what a
prune saves, overstates the pruned prompt against the cache floor, and hands the missing
tokens to prose, corrupting the one ratio that was being learned.

The constants below are still the starting point, because a process has to answer before
it has seen anything. What changed is that they stop being the answer.

That two-term split is the difference between this and the constant it replaces.
`tokens_per_declaration` was 200 against a measured 37 because declarations vary
enormously in size, and a per-declaration count cannot see that. Characters per token
is normalised by content: a verbose declaration has more characters and correctly costs
more tokens.

Characters stand in for a tokenizer we deliberately do not ship. Vertex has a
`count_tokens` endpoint, but putting a second network round trip on the hot path to
protect against a cost problem is self-defeating. Instead the first request of a process
is estimated and every one after it is measured, because the model reports the true
count on every response and the router was throwing that away.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest

# The two names ADK reads for the same question, old and new. Either being true means
# Vertex, and Vertex means every tool declaration carries a `response_json_schema` that the
# Gemini Developer API has nowhere to put: 74 characters per tool, about 1,200 tokens on a
# hundred of them (DESIGN.md 19.46). Nothing else between the two backends differs.
_VERTEX_ENV = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")

# What that difference is worth, for anything that has to explain itself to a reader.
VERTEX_SCHEMA_CHARS_PER_TOOL = 74


def counting_vertex_declarations() -> bool:
    """Whether ADK will build Vertex-shaped declarations in this process.

    Read from the environment rather than from ADK, whose own accessor for this is private
    and currently mid-rename. Read on every call rather than at import, because the answer
    is a fact about the shell and not about the installation.
    """
    return any(os.environ.get(name, "").strip().lower() in ("true", "1") for name in _VERTEX_ENV)


# Where a process starts, before it has billed anything. Measured on the reference
# agent's original 25 declarations, r-squared better than 0.999. Both are now learned,
# so these are the first guess rather than the answer (DEV_NOTES.md, 2026-09-04).
PROSE_CHARS_PER_TOKEN = 4.24
SCHEMA_CHARS_PER_TOKEN = 8.33

# Observed ratios outside these ranges are rejected rather than learned from. A truncated
# response, a streaming partial or a provider bug must not be able to move the calibration
# somewhere it cannot come back from. One token per character is a hard lower bound for any
# tokenizer worth the name; the schema band is wider because the measured spread across one
# repository's own declarations is already 5.75 to 8.30.
_PLAUSIBLE_PROSE = (1.0, 12.0)
_PLAUSIBLE_SCHEMA = (2.0, 20.0)

# A whole request billing outside this many characters per token is not a calibration
# sample, it is a broken response. Coarse on purpose: it has to be judged without either
# ratio, since it is what protects both of them.
_PLAUSIBLE_OVERALL = (1.0, 20.0)

# Below this many responses the joint fit is fitting noise. Thirty is a few minutes of
# traffic on any agent worth routing, and the fallback is what shipped before.
_MIN_JOINT_SAMPLES = 30

# How nearly parallel the base and schema columns may be before the fit stops being able
# to tell them apart. 1.0 is exact collinearity, where any split of the bill between the
# two terms fits equally well and the solution is whatever the rounding says.
#
# A process that sends both a full tool list and a pruned one sits near 0.86, because the
# schema term takes two very different values. A process that only ever sends one list,
# with history growing 20% across a session, sits at about 0.997: its schema term never
# moves and its base term barely does. Between those is a third case that looks similar
# and is not, a single-list agent whose history grows severalfold, which reaches about
# 0.88 and is genuinely separable, because a base term that spans a factor of five pins
# the constant one by itself. The threshold is set to admit that case and reject the
# other, and the two are far enough apart that it is not delicate.
_MAX_COLLINEARITY = 0.95

# Images and audio bill at a flat per-part rate, nothing like their base64 length.
# Counting the encoded bytes would swamp every other term. This router is built for text
# agents, so a coarse constant is the honest placeholder rather than a wrong measurement.
_BINARY_PART_CHARS = 1024


@dataclass(frozen=True)
class Chars:
    """One request's character counts, split by what a prune can and cannot move."""

    base: int  # system instruction plus conversation history: never prunable
    kept: int  # declarations that would survive the prune
    dropped: int  # declarations the prune would remove

    @property
    def full(self) -> int:
        """The request as it stands, with every tool still declared."""
        return self.base + self.kept + self.dropped

    @property
    def pruned(self) -> int:
        """The request as it would be sent after the prune."""
        return self.base + self.kept


def count_chars(llm_request: LlmRequest, kept: frozenset[str] | None) -> Chars:
    """Count the request, attributing each declaration to the kept or dropped side.

    `kept=None` means nothing would be dropped, which is what a caller passes when it
    wants the size of the request exactly as it stands.
    """
    base = _instruction_chars(llm_request) + sum(
        _content_chars(content) for content in (llm_request.contents or [])
    )

    survived = 0
    removed = 0
    for declaration in _declarations(llm_request):
        size = _json_chars(declaration)
        name = getattr(declaration, "name", None)
        # An unnamed declaration cannot be pruned by name, so it always survives.
        if kept is None or name is None or name in kept:
            survived += size
        else:
            removed += size
    return Chars(base=base, kept=survived, dropped=removed)


class TokenScale:
    """Characters to tokens, calibrated against what the model actually billed.

    One instance per model, held by the plugin. Every response gives one equation,

        billed tokens = base characters / prose ratio + schema characters / schema ratio

    and the two ratios are what this solves for. It keeps a running least squares rather
    than a decaying average, so a single odd turn cannot swing it and no half-life has to
    be chosen. What it measures is a property of the agent's text and its declarations,
    neither of which drifts the way traffic volume does.

    Two ratios need two independent equations, and traffic does not always supply them.
    A process that always sends the same tool list has a constant schema term and a base
    term that barely moves, so the two columns are nearly parallel and the split between
    them is arbitrary. `_MAX_COLLINEARITY` is where the fit admits that. Below it the
    joint solution is used; above it, or before `_MIN_JOINT_SAMPLES` responses, the
    fallback is what shipped before: schema at its fixed rate, everything left over
    attributed to prose. A router that prunes generates the second equation by itself,
    because a pruned request and a full one differ in exactly the term at issue.

    Not thread-safe and does not need to be. The arithmetic is five additions, and a lost
    update costs one sample out of thousands.
    """

    def __init__(
        self,
        prose: float = PROSE_CHARS_PER_TOKEN,
        schema: float = SCHEMA_CHARS_PER_TOKEN,
    ) -> None:
        self._prose = prose
        self._schema = schema
        self._prose_chars = 0.0
        self._prose_tokens = 0.0
        self._samples = 0
        # Normal equations for the joint fit, in characters and tokens as observed.
        self._bb = 0.0
        self._bs = 0.0
        self._ss = 0.0
        self._bt = 0.0
        self._st = 0.0

    @property
    def prose_ratio(self) -> float:
        """Characters per token for conversation. Moves as responses come back."""
        return self._ratios()[0]

    @property
    def schema_ratio(self) -> float:
        """Characters per token for declarations. Moves once traffic can separate it."""
        return self._ratios()[1]

    @property
    def samples(self) -> int:
        """Responses folded in. Zero means every answer so far is the starting guess."""
        return self._samples

    @property
    def fitted(self) -> bool:
        """True once the joint fit is being used rather than the fixed schema rate."""
        return self._fit() is not None

    def full_tokens(self, chars: Chars) -> int:
        """What the request bills with every tool still declared."""
        return self._estimate(chars.base, chars.kept + chars.dropped)

    def pruned_tokens(self, chars: Chars) -> int:
        """What the request would bill once the dropped declarations are gone."""
        return self._estimate(chars.base, chars.kept)

    def observe(self, base_chars: int, schema_chars: int, billed_tokens: int) -> None:
        """Fold in one response: the characters we sent, and what the model charged.

        The sample feeds both the joint fit and the fallback, because which of them is
        answering can change as traffic arrives and neither should have a gap in its
        history when it does.
        """
        if billed_tokens <= 0 or base_chars <= 0 or schema_chars < 0:
            return
        # Judged without either ratio, because this is what protects both of them.
        low, high = _PLAUSIBLE_OVERALL
        if not low <= (base_chars + schema_chars) / billed_tokens <= high:
            return

        self._bb += float(base_chars) * base_chars
        self._bs += float(base_chars) * schema_chars
        self._ss += float(schema_chars) * schema_chars
        self._bt += float(base_chars) * billed_tokens
        self._st += float(schema_chars) * billed_tokens
        self._samples += 1

        # The fallback, unchanged: subtract schema at the fixed rate, call the rest prose.
        prose_tokens = billed_tokens - schema_chars / self._schema
        if prose_tokens <= 0:
            return  # the schema estimate already exceeds the bill; nothing to attribute
        prose_low, prose_high = _PLAUSIBLE_PROSE
        if not prose_low <= base_chars / prose_tokens <= prose_high:
            return
        self._prose_chars += base_chars
        self._prose_tokens += prose_tokens

    def _fit(self) -> tuple[float, float] | None:
        """Solve the two-by-two least squares, or None if the traffic cannot support it.

        Returns characters per token for prose and for schema. The normal equations are
        solved for tokens per character and inverted at the end, because that is the form
        the model is linear in.
        """
        if self._samples < _MIN_JOINT_SAMPLES:
            return None
        scale = self._bb * self._ss
        if scale <= 0:
            return None
        # Squared cosine between the two columns. At 1.0 they point the same way and any
        # split of the bill fits equally well.
        if (self._bs * self._bs) / scale > _MAX_COLLINEARITY:
            return None
        determinant = self._bb * self._ss - self._bs * self._bs
        prose_per_char = (self._ss * self._bt - self._bs * self._st) / determinant
        schema_per_char = (self._bb * self._st - self._bs * self._bt) / determinant
        if prose_per_char <= 0 or schema_per_char <= 0:
            return None  # a negative rate is a fit, not a measurement
        prose = 1.0 / prose_per_char
        schema = 1.0 / schema_per_char
        prose_low, prose_high = _PLAUSIBLE_PROSE
        schema_low, schema_high = _PLAUSIBLE_SCHEMA
        if not (prose_low <= prose <= prose_high and schema_low <= schema <= schema_high):
            return None
        return prose, schema

    def _estimate(self, base_chars: int, schema_chars: int) -> int:
        prose, schema = self._ratios()
        return int(base_chars / prose + schema_chars / schema)

    def _ratios(self) -> tuple[float, float]:
        """Both ratios from one solve, so an estimate cannot mix a fitted with a fallback."""
        fit = self._fit()
        if fit is not None:
            return fit
        fallback = self._prose_chars / self._prose_tokens if self._prose_tokens > 0 else self._prose
        return fallback, self._schema


# --- Walking the request ---------------------------------------------------------


def _declarations(llm_request: LlmRequest) -> list[Any]:  # noqa: ANN401 - genai part union
    """Every function declaration across every Tool on the request."""
    found: list[Any] = []
    for tool in getattr(llm_request.config, "tools", None) or []:
        found.extend(getattr(tool, "function_declarations", None) or [])
    return found


def _instruction_chars(llm_request: LlmRequest) -> int:
    instruction = getattr(llm_request.config, "system_instruction", None)
    if instruction is None:
        return 0
    if isinstance(instruction, str):
        return len(instruction)
    if isinstance(instruction, (list, tuple)):
        return sum(_content_chars(item) for item in instruction)
    return _content_chars(instruction)


def _content_chars(content: Any) -> int:  # noqa: ANN401 - genai Content or str
    if isinstance(content, str):
        return len(content)
    parts = getattr(content, "parts", None)
    if parts is None:
        return _json_chars(content)
    return sum(_part_chars(part) for part in parts)


def _part_chars(part: Any) -> int:  # noqa: ANN401 - genai Part union
    text = getattr(part, "text", None)
    if text:
        return len(text)
    if getattr(part, "inline_data", None) is not None or getattr(part, "file_data", None):
        return _BINARY_PART_CHARS
    # Function calls and their results bill as the JSON they serialise to.
    return _json_chars(part)


def _json_chars(value: Any) -> int:  # noqa: ANN401 - any pydantic model
    """Serialised length of a pydantic object, falling back to `str` for anything else."""
    dump = getattr(value, "model_dump_json", None)
    if dump is None:
        return len(str(value))
    try:
        return len(dump(exclude_none=True))
    except Exception:  # a model that will not serialise must not break the request
        return len(str(value))
