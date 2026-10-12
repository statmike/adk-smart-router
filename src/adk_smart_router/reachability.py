"""Which of an agent's tools a turn-one router can ever predict, read from schemas alone.

The commonest disappointment with a router is "it never predicts half my tools", and it is
usually discovered a month into logging. It is knowable on day one, for free, because it is a
property of the tool schemas and not of the traffic.

DESIGN.md 19.40 has the rule and the evidence. A tool is reachable as a turn-one label only if
a first user message can supply what the tool requires. On the 100-tool reference agent, split
by whether the customer holds the argument, 41 of the 55 answerable tools were somebody's first
call and 7 of the 45 that need an identifier only the shop holds were. Everything else is the
second hop: the agent has to call `search_products` before it has a SKU to ask about. A
turn-one router will never have evidence about those, which is why `pruner` keeps unknown tools
unconditionally and why menus carry them by co-occurrence instead.

What this module refuses to do
------------------------------
Guess. The determining question is whether the *user* holds a value, and no amount of reading
the schema answers it. `order_id` and `subscription_id` are the same shape, the same type and
the same suffix, and on the reference agent they fall on opposite sides: a customer knows their
order number and does not know the internal id of their subscription. A name-pattern rule that
called both of them identifiers would be wrong about thirteen tools and would be wrong
confidently, which is worse than being silent.

So there are exactly two things it decides by itself, both of them structural:

    supplied    the schema lists the argument's legal values, or it is a yes-or-no. A first
                message can carry it because there is nothing to know.
    held        another declared tool returns this name. Only that tool can produce it, so
                reaching this one means calling that one first. Needs response schemas, which
                most agents do not declare, so it usually finds nothing and says so.

Everything else comes back unmarked, and the answer is a span rather than a number: at least
this many tools are reachable, at most this many, and here is the list that closes the gap.
That is the same idiom `cli.menu_range` uses for menu size, and for the same reason. A range
with its ends named is worth more than a point estimate with a guess inside it.

Marking is by argument name across the whole surface, because that is what makes it a ten
minute job instead of an afternoon. The reference agent has 76 distinct required-argument names
across 100 tools, `sku` alone gates 19 of them, and the ten commonest names resolve a third of
the surface. `template` emits them in that order, so an adopter who stops early has resolved
the most. A `tool.argument` line overrides the bare name where one tool is different.

Once the agent has traffic, `audit` checks the markings against it, and it is careful about
what traffic can prove. A session that opened with a tool proves the tool is reachable at turn
one, which is the question the span is actually asking: the label exists, so a router can learn
it. It does not prove a user held the argument. The model normalises what it was told, so
"near Toronto" arrives as `country_code: "CA"`, and it sometimes invents what it was not told,
so "I've got a gift card from Christmas" arrives as `card_number: "4567-8901-2345-6789"`. Both
look identical from here (DESIGN.md 19.56). So `audit` reports the disagreement and leaves the
marking to the person, and nothing in this module ever writes a verdict nobody chose.
`openings.py` is where the first calls come from.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from google.genai import types

Standing = Literal["supplied", "held"]
Verdict = Literal["reachable", "second-hop", "unmarked"]

_COMMENT = "#"


# --- What one argument does to its tool -------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One required argument, and whether a first user message can carry it."""

    tool: str
    argument: str
    standing: Standing | None  # None means nobody has said, and nothing structural decided it
    why: str

    @property
    def key(self) -> str:
        return f"{self.tool}.{self.argument}"


@dataclass(frozen=True)
class Tool:
    """One declared tool and every required argument standing between it and turn one."""

    name: str
    gates: tuple[Gate, ...]

    @property
    def verdict(self) -> Verdict:
        # Order matters. One held argument settles it however many others are supplied, and
        # only a tool with nothing left to decide can be called reachable.
        if any(gate.standing == "held" for gate in self.gates):
            return "second-hop"
        if all(gate.standing == "supplied" for gate in self.gates):
            return "reachable"
        return "unmarked"

    @property
    def why(self) -> str:
        held = [gate for gate in self.gates if gate.standing == "held"]
        if held:
            return f"{held[0].argument}: {held[0].why}"
        if not self.gates:
            return "requires nothing, so a first message always carries it"
        undecided = [gate.argument for gate in self.gates if gate.standing is None]
        if undecided:
            return "waiting on " + ", ".join(undecided)
        return "every required argument can be carried by a first message"


# --- The surface ------------------------------------------------------------------------


@dataclass(frozen=True)
class Survey:
    """Every tool, classified, plus the span the unmarked ones leave open."""

    tools: tuple[Tool, ...]
    responses_declared: int  # how many tools declared a response schema, usually zero

    @property
    def reachable(self) -> tuple[Tool, ...]:
        return tuple(t for t in self.tools if t.verdict == "reachable")

    @property
    def second_hop(self) -> tuple[Tool, ...]:
        return tuple(t for t in self.tools if t.verdict == "second-hop")

    @property
    def unmarked(self) -> tuple[Tool, ...]:
        return tuple(t for t in self.tools if t.verdict == "unmarked")

    @property
    def span(self) -> tuple[int, int]:
        """Reachable tools if every unmarked argument is held, and if every one is supplied."""
        return len(self.reachable), len(self.reachable) + len(self.unmarked)

    def open_names(self) -> list[tuple[str, int]]:
        """Unmarked argument names, commonest first, with how many tools each one gates.

        Commonest first is the whole point: this is the order to work down, and the count is
        what the next decision is worth.
        """
        counted: Counter[str] = Counter()
        for tool in self.unmarked:
            for gate in tool.gates:
                if gate.standing is None:
                    counted[gate.argument] += 1
        return counted.most_common()


def survey(
    declarations: list[types.FunctionDeclaration],
    marks: dict[str, Standing] | None = None,
) -> Survey:
    """Classify every declared tool. Reads schemas and `marks`, and calls nothing."""
    marks = marks or {}
    produced = _produced_names(declarations)
    tools = []
    for declaration in declarations:
        name = declaration.name or ""
        gates = tuple(
            _gate(name, argument, schema, marks, produced)
            for argument, schema in _required(declaration)
        )
        tools.append(Tool(name=name, gates=gates))
    return Survey(
        tools=tuple(tools),
        responses_declared=sum(1 for d in declarations if _returns(d)),
    )


def _schema(typed: types.Schema | None, raw: dict[str, Any] | None) -> dict[str, Any]:
    """One JSON-schema dict, whichever of the two fields a declaration used.

    `FunctionDeclaration` can carry its shape as a typed `Schema` or as a plain
    `*_json_schema` dict, and which one appears depends on how the tool was built. ADK's
    `FunctionTool` writes the dict for a plain Python function, and hand-written declarations
    usually use the typed field. Both have to work, so both are normalised to a dict here and
    nothing below this function knows there were ever two.
    """
    if raw:
        return raw
    if typed is None:
        return {}
    return typed.model_dump(exclude_none=True)


def _parameters(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    return _schema(declaration.parameters, declaration.parameters_json_schema)


def _returns(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    """What the tool says it returns, or empty. `{"type": "object"}` with no properties is
    the shape ADK writes for an unannotated return, and it names nothing, so it counts as
    empty here."""
    return _schema(declaration.response, declaration.response_json_schema).get("properties") or {}


def _required(declaration: types.FunctionDeclaration) -> list[tuple[str, dict[str, Any]]]:
    """The declaration's required arguments, in declared order.

    Optional arguments are ignored on purpose. A first message that omits one still reaches the
    tool, so an optional argument cannot be what stops prediction.
    """
    parameters = _parameters(declaration)
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    return [(name, properties[name]) for name in required if name in properties]


def _produced_names(declarations: list[types.FunctionDeclaration]) -> dict[str, str]:
    """Argument names some other tool returns, mapped to the first tool that returns them."""
    produced: dict[str, str] = {}
    for declaration in declarations:
        for name in _returns(declaration):
            produced.setdefault(name, declaration.name or "")
    return produced


def _gate(
    tool: str,
    argument: str,
    schema: dict[str, Any],
    marks: dict[str, Standing],
    produced: dict[str, str],
) -> Gate:
    """Decide one argument. A mark beats everything, then structure, then nothing."""
    # `tool.argument` before the bare name, so one exception does not force marking by hand
    # everywhere the name appears.
    for key in (f"{tool}.{argument}", argument):
        if key in marks:
            return Gate(tool, argument, marks[key], f"marked {marks[key]} as {key}")

    source = produced.get(argument)
    if source and source != tool:
        return Gate(tool, argument, "held", f"{source} returns it, so that has to be called first")
    enum = schema.get("enum")
    if enum:
        return Gate(tool, argument, "supplied", f"one of {len(enum)} values the schema lists")
    # `type` is lower case in a JSON schema and upper case in a dumped `types.Schema`.
    if str(schema.get("type", "")).lower() == "boolean":
        return Gate(tool, argument, "supplied", "a yes or no")
    return Gate(tool, argument, None, "nobody has said whether a user holds this")


# --- What the traffic demonstrates ----------------------------------------------------------


@dataclass(frozen=True)
class Witness:
    """One tool the traffic opened a session with, and what the survey had said about it."""

    tool: str
    sessions: int  # sessions whose first tool call was this one
    was: Verdict  # what the survey said before the traffic was asked
    arguments: tuple[str, ...]  # required arguments the demonstration settles as supplied
    why: str  # the survey's reason, so a contradiction can be read against its cause


@dataclass(frozen=True)
class Audit:
    """A survey held up against the agent's own traffic.

    Three findings, in descending order of how much they are worth. A contradiction is a
    marking that says no tool can reach this at turn one, next to sessions where one did, and
    it is worth a person's attention every time. A witnessed tool is an open question the
    traffic has evidence about. An unwitnessed tool is the weak one: nobody opened with it,
    which is evidence of nothing much, because a tool can be perfectly reachable and simply
    never be what anyone wants first.

    None of it is a verdict. How strong a demonstration is depends on whether the arguments
    were the user's or the model's, which nothing here can see, so all three are counts put in
    front of somebody rather than decisions taken on their behalf.
    """

    contradicted: tuple[Witness, ...]  # the survey said second hop; a session opened with it
    settled: tuple[Witness, ...]  # the survey had no answer; the traffic has evidence
    unwitnessed: tuple[str, ...]  # called reachable, and no session ever opened with it
    undeclared: tuple[str, ...]  # the traffic opened with these and the agent no longer has them
    confirmed: int  # called reachable, and the traffic agrees
    sessions: int  # sessions the finding rests on
    floor: int  # sessions a tool needed before it counted


def audit(result: Survey, openings: dict[str, int], floor: int = 1) -> Audit:
    """Check a survey against which tool each session called first. Calls nothing.

    `openings` maps a tool name to the number of sessions whose first tool call it was, which
    is what `openings.first_calls` returns. `floor` is how many sessions a tool needs before
    its demonstration counts, and raising it is how to keep one stray call from rewriting a
    marking on an agent whose logs are noisy.
    """
    witnessed = {name: count for name, count in openings.items() if count >= floor}
    declared = {tool.name for tool in result.tools}

    contradicted: list[Witness] = []
    settled: list[Witness] = []
    unwitnessed: list[str] = []
    confirmed = 0
    for tool in result.tools:
        sessions = witnessed.get(tool.name, 0)
        if not sessions:
            if tool.verdict == "reachable":
                unwitnessed.append(tool.name)
            continue
        if tool.verdict == "reachable":
            confirmed += 1
            continue
        witness = Witness(
            tool=tool.name,
            sessions=sessions,
            was=tool.verdict,
            # Everything not already settled as supplied, which for a second hop is the held
            # argument that made it one and for an unmarked tool is whatever nobody decided.
            arguments=tuple(gate.argument for gate in tool.gates if gate.standing != "supplied"),
            why=tool.why,
        )
        (contradicted if tool.verdict == "second-hop" else settled).append(witness)

    return Audit(
        contradicted=tuple(sorted(contradicted, key=_loudest)),
        settled=tuple(sorted(settled, key=_loudest)),
        unwitnessed=tuple(sorted(unwitnessed)),
        undeclared=tuple(sorted(name for name in witnessed if name not in declared)),
        confirmed=confirmed,
        sessions=sum(openings.values()),
        floor=floor,
    )


def _loudest(witness: Witness) -> tuple[int, str]:
    """Most demonstrated first. The top of the list is the marking worth arguing about."""
    return -witness.sessions, witness.tool


def demonstrated(found: Audit) -> dict[str, int]:
    """How many sessions opened with a tool that required each argument.

    Evidence to put beside a question, and deliberately not an answer to it. Filling the
    marking file in from this would be writing `supplied` next to `card_number` on the
    strength of a model inventing one, and the marking file is what every reach number this
    project prints rests on. So the count goes in the margin and the person still decides.
    """
    counts: dict[str, int] = {}
    for witness in found.contradicted + found.settled:
        for argument in witness.arguments:
            counts[argument] = counts.get(argument, 0) + witness.sessions
    return counts


# --- The marking file ---------------------------------------------------------------------


def template(result: Survey, witnessed: dict[str, int] | None = None) -> str:
    """The file to hand an adopter, ordered by how many tools each decision resolves.

    `witnessed` is `demonstrated(audit(...))`, and it annotates rather than answers: the line
    still comes back blank, with how many sessions opened with a tool needing that argument
    written beside it. That is the shape of the evidence, and it is the shape the question
    should keep.
    """
    witnessed = witnessed or {}
    lines = [
        f"{_COMMENT} Which required arguments your users hold, and which only your systems do.",
        f"{_COMMENT}",
        f"{_COMMENT} One argument name per line, then `held` or `supplied`. A name applies to",
        f"{_COMMENT} every tool that takes it; write `tool.argument` to override one tool.",
        f"{_COMMENT}",
        f"{_COMMENT}   supplied  a first message can carry it. An order number, a postcode, an",
        f"{_COMMENT}             email address, a promo code, anything the person already knows.",
        f"{_COMMENT}   held      only your systems have it. An internal record id, a SKU. A tool",
        f"{_COMMENT}             needing one is a second hop and no turn-one router reaches it.",
        f"{_COMMENT}",
        f"{_COMMENT} Commonest first, so the top of the list is worth the most. The number is how",
        f"{_COMMENT} many tools that one decision settles. Stopping early is fine and the span",
        f"{_COMMENT} just stays wider.",
    ]
    if witnessed:
        lines += [
            f"{_COMMENT}",
            f"{_COMMENT} `opened N` is your own traffic: N sessions called a tool needing that",
            f"{_COMMENT} argument before any tool had returned anything. It is evidence and not an",
            f"{_COMMENT} answer, because the model normalises what it was told and invents what it",
            f"{_COMMENT} was not, and both look the same from here. A big number on a value your",
            f"{_COMMENT} users really do quote is `supplied`. A small one is worth a look first.",
        ]
    lines.append("")
    open_names = result.open_names()
    for name, gated in open_names:
        gates = f"gates {gated} tool{'s' if gated != 1 else ''}"
        seen = f", opened {witnessed[name]}" if name in witnessed else ""
        lines.append(f"{name:<32}{'':<10}{_COMMENT} {gates}{seen}")

    # An argument the traffic opened with that is not on the list above is one already marked,
    # and marked the other way. It belongs in this file even though there is nothing to fill
    # in, because the line to revisit is in the adopter's copy and this is what names it.
    # Commented out, so a file whose only content is a disagreement still parses to nothing.
    disputed = _disputed(result, witnessed, {name for name, _ in open_names})
    if disputed:
        lines += [
            "",
            f"{_COMMENT} Already marked, and your traffic went the other way. Sessions opened with",
            f"{_COMMENT} a tool needing these, so either the marking is wrong or the model is",
            f"{_COMMENT} filling the argument in by itself. Worth knowing which.",
        ]
        lines += [
            f"{_COMMENT} {name:<30}{_COMMENT} marked {was}, opened {n}" for name, was, n in disputed
        ]
    return "\n".join(lines) + "\n"


def _disputed(
    result: Survey, witnessed: dict[str, int], open_names: set[str]
) -> list[tuple[str, str, int]]:
    """Witnessed arguments that somebody already marked, loudest first, with what they said."""
    standing = {
        gate.argument: gate.standing
        for tool in result.tools
        for gate in tool.gates
        if gate.standing is not None
    }
    found = [
        (name, standing[name], count)
        for name, count in witnessed.items()
        if name not in open_names and name in standing
    ]
    return sorted(found, key=lambda row: (-row[2], row[0]))


def read_marks(text: str) -> dict[str, Standing]:
    """Parse a marking file into the decisions it actually contains.

    Blank lines and `#` comments are skipped. A bare name with no verdict is skipped too,
    because that is exactly what an untouched line of `template` looks like and stopping part
    way down the list is a supported way to use this: the decisions made so far count and the
    span stays open on the rest. Anything else is a typo and says so, with its line number,
    rather than being silently dropped.
    """
    marks: dict[str, Standing] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split(_COMMENT, 1)[0].split()
        if len(parts) < 2:
            continue
        name, standing, *rest = parts
        if rest or standing not in ("held", "supplied"):
            raise ValueError(
                f"line {number}: expected `<argument> held` or `<argument> supplied`, got {raw!r}"
            )
        marks[name] = "held" if standing == "held" else "supplied"
    return marks
