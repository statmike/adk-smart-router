"""The front door: the questions a user needs answered around installing this and living with it.

    smart-router qualify         --app myapp:app    can pruning pay for this agent?
    smart-router readiness       --agent support    could the traffic already logged train it?
    smart-router doctor          --app myapp:app    is it wired up so it will work?
    smart-router status          --agent support    is it working, and is it paying?
    smart-router check-arguments --app myapp:app    does the traffic agree with what you
                                                    said your users hold?

`qualify` and `doctor` run against the user's own code with no cloud, no credentials and no
model call, which is the point of both. They are the day-zero commands, and `status` is the
day-two one: it reads the published artifact, the last week of traffic and the retrain ledger,
and answers the question an adopter otherwise has to build a dashboard to ask.
`check-arguments` is the narrow one, reading which tool each logged session called first.

`readiness` sits between the two ends and is the only one that needs credentials but no
provisioning. An agent already running the analytics plugin may own its training corpus
already, and this is how that gets answered before anything is created or paid for.

There is deliberately no `provision` subcommand. `scripts/provision.sh` already is one, and
it is idempotent; wrapping a shell script that talks to four APIs in a Python command that
talks to none would only move the credentials question somewhere less obvious.

`qualify` is arithmetic over the agent's declarations — see `qualify.py` for the bands and
what they rest on. `doctor` is the three wiring mistakes that no script can catch and that
each fail silently: the router registered after the analytics plugin, `smart_router` missing
from `custom_metadata_allowlist`, and no `ContextCacheConfig`. Every one of them leaves an
agent that runs, logs and looks healthy while doing the wrong thing, which is why they are
worth a command of their own rather than a paragraph in a README.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import os
import sys
import textwrap
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.apps import App
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from adk_smart_router import openings, reachability, readiness, status
from adk_smart_router.config import FRAMEWORK_TOOLS, RouterConfig, load_config
from adk_smart_router.prompt_size import (
    SCHEMA_CHARS_PER_TOKEN,
    Chars,
    TokenScale,
    count_chars,
    counting_vertex_declarations,
)
from adk_smart_router.provenance import PROVENANCE_KEY
from adk_smart_router.qualify import PAYS_ABOVE, Verdict, qualify

# Private on purpose over there, and imported anyway: duck-typing "is this the analytics
# plugin" a second time is how `doctor` and the runtime warning end up disagreeing about
# which plugin they are looking at. One fingerprint, one place.
from adk_smart_router.startup_checks import _is_bq_analytics, judge_wiring_issues

__all__ = ["main"]

# Wide enough for the reason strings, narrow enough to read in a terminal.
_WRAP = 88


class CliError(Exception):
    """A problem with what the user asked for, not with their agent. Exit code 2."""


@contextlib.contextmanager
def _quiet_experimental() -> Iterator[None]:
    """Silence ADK's own `[EXPERIMENTAL]` notices while importing and reading the user's app.

    They are facts about the user's ADK version rather than about this command, and three of
    them stacked on top of a verdict is how a front door starts looking broken. Matched on
    the message, so anything the user's own module warns about still prints.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")
        yield


# --- Loading the user's app ------------------------------------------------------


@dataclass(frozen=True)
class Loaded:
    """What `MODULE:ATTRIBUTE` resolved to.

    `app` is None when the target was a bare agent. `qualify` is happy either way; `doctor`
    needs the App, because plugins and the cache config live on it and not on the agent.
    """

    app: App | None
    agent: BaseAgent


def _load(target: str) -> Loaded:
    """Import `MODULE:ATTRIBUTE` and resolve it to an App or an agent.

    A zero-argument factory is accepted too, since an App is commonly built by a function
    rather than bound at module scope.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise CliError(f"--app must look like MODULE:ATTRIBUTE, got {target!r}")

    # A console script puts its own bin directory on sys.path, not the working directory, so
    # without this `myapp.agents:app` fails to import from the project the user is standing
    # in. uvicorn and gunicorn insert the same entry for the same reason.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        with _quiet_experimental():
            module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CliError(f"could not import {module_name!r} from {cwd}: {exc}") from exc
    except Exception as exc:
        raise CliError(f"importing {module_name!r} raised {type(exc).__name__}: {exc}") from exc
    if not hasattr(module, attribute):
        raise CliError(f"{module_name!r} has no attribute {attribute!r}")

    found: Any = getattr(module, attribute)
    if callable(found) and not isinstance(found, (App, BaseAgent)):
        try:
            found = found()
        except TypeError as exc:
            raise CliError(
                f"{target} is a callable that needs arguments ({exc}). Point --app at an "
                f"App or an agent, or at a factory that takes none."
            ) from exc
        except Exception as exc:
            raise CliError(f"calling {target} raised {type(exc).__name__}: {exc}") from exc

    if isinstance(found, App):
        root: Any = found.root_agent
        if not isinstance(root, BaseAgent):
            raise CliError(f"{target} is an App whose root_agent is {type(root).__name__}")
        return Loaded(app=found, agent=root)
    if isinstance(found, BaseAgent):
        return Loaded(app=None, agent=found)
    raise CliError(f"{target} is a {type(found).__name__}, not an App or an agent")


def _model_name(agent: BaseAgent) -> str:
    """The model string, whether the agent holds a name or a constructed BaseLlm."""
    model: Any = getattr(agent, "model", "")
    if isinstance(model, str):
        return model
    return str(getattr(model, "model", "") or "")


def _instruction(agent: BaseAgent) -> str:
    """The system instruction, or empty if it is a provider.

    A provider is resolved per session against session state, so there is nothing to read
    here without inventing a session. The caller says so in the output rather than guessing
    at a length, because the instruction is usually the largest term a prune cannot touch.
    """
    text = getattr(agent, "instruction", "")
    return text if isinstance(text, str) else ""


def _declarations(agent: BaseAgent) -> list[types.FunctionDeclaration]:
    """Every function declaration this agent would send, resolved the way ADK resolves it.

    `canonical_tools` is what the flow itself calls, so plain functions, `BaseTool`s and
    toolsets all come back already normalised. It is async and takes no context here, which
    is fine for everything that does not need session state to decide its tool list.
    """
    if not isinstance(agent, LlmAgent):
        raise CliError(
            f"agent {agent.name!r} is a {type(agent).__name__} and declares no tools; "
            f"point --app at the LlmAgent whose menu you want costed"
        )
    with _quiet_experimental():
        tools = asyncio.run(agent.canonical_tools(None))
        # `_get_declaration` is ADK's only synchronous accessor for a tool's schema and is
        # what `BaseTool.process_llm_request` uses to build the request this imitates.
        found = [tool._get_declaration() for tool in tools]
    return [declaration for declaration in found if declaration is not None]


def _request(
    model: str,
    instruction: str,
    declarations: Sequence[types.FunctionDeclaration],
    prompts: Sequence[str],
) -> LlmRequest:
    """The request this agent would send, built for counting and never sent anywhere."""
    return LlmRequest(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=text)]) for text in prompts],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            tools=[types.Tool(function_declarations=list(declarations))],
        ),
    )


# --- qualify ---------------------------------------------------------------------


@dataclass(frozen=True)
class Costing:
    """One agent's request, counted once, so a menu of any size can be priced from it.

    Held apart from the printing because the interesting question is not what one menu
    size costs but over what range of them the answer stays the same.
    """

    declared: int  # declarations on the request
    prunable: int  # of those, the ones a menu may drop
    chars: Chars  # kept = protected declarations, dropped = prunable ones
    history_tokens: int  # conversation the user says a steady-state turn carries
    scale: TokenScale

    def sizes(self, keep: int) -> tuple[int, int]:
        """(whole prompt, tokens removed) if a menu keeps `keep` of the prunable tools.

        Which tools it keeps takes a trained model, so this drops an average-sized subset:
        the expectation over menus, and the only unbiased answer available before a model
        exists. An agent whose declarations differ wildly in size will land either side.
        """
        dropped = 0
        if self.prunable > keep:
            dropped = round(self.chars.dropped * (self.prunable - keep) / self.prunable)
        # The same characters, re-attributed: what survives this menu against what it drops.
        after = Chars(
            base=self.chars.base, kept=self.chars.full - self.chars.base - dropped, dropped=dropped
        )
        full = self.scale.full_tokens(after)
        return full + self.history_tokens, full - self.scale.pruned_tokens(after)


def cost(
    request: LlmRequest,
    *,
    protected: frozenset[str],
    history_tokens: int,
    schema_chars_per_token: float,
) -> Costing:
    """Count the request once, splitting protected declarations from prunable ones."""
    names = _declaration_names(request)
    return Costing(
        declared=len(names),
        prunable=len([name for name in names if name not in protected]),
        chars=count_chars(request, protected),
        history_tokens=history_tokens,
        scale=TokenScale(schema=schema_chars_per_token),
    )


def menu_range(costing: Costing, model: str, cfg: RouterConfig) -> tuple[int | None, int | None]:
    """(smallest menu that stays out of the `trap`, largest menu that still `pays`).

    Either can be None: an agent whose instruction alone sits under the cache floor has no
    safe small menu, and an agent with little to prune never reaches `pays` at any size.
    The pair is what turns a single verdict into a decision — it says how much of the
    answer depends on the guess at menu size, which is the shakiest input here. The router
    keeps more tools than `max_tools_kept` in practice, because it also keeps every tool it
    has never seen called: the run in DESIGN.md 19.17 published menus of 11 and realised 43
    of 100.
    """
    safe: int | None = None
    pays: int | None = None
    for keep in range(costing.prunable + 1):
        prompt, removed = costing.sizes(keep)
        band = qualify(model, prompt, removed, cfg, costing.history_tokens).band
        if band != "trap" and safe is None:
            safe = keep
        if band == "pays":
            pays = keep
    return safe, pays


def _declaration_names(request: LlmRequest) -> list[str]:
    names: list[str] = []
    for tool in request.config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            names.append(declaration.name or "")
    return names


def _qualify(args: argparse.Namespace, cfg: RouterConfig) -> int:
    loaded = _load(args.app)
    agent = loaded.agent
    # Tools first: an agent that declares none is the more useful thing to be told about,
    # and the agents that declare none are also the ones that tend to name no model.
    declarations = _declarations(agent)
    if not declarations:
        raise CliError(f"agent {agent.name!r} declares no tools, so there is nothing to prune")

    reach = reachability.survey(declarations, _marks(args.arguments))
    if args.write_arguments:
        # Before the model check, not after: an adopter who has not named a model yet can
        # still get the list to fill in, and this question does not need one.
        Path(args.write_arguments).write_text(reachability.template(reach))
        print(
            f"\nwrote {len(reach.open_names())} argument names to {args.write_arguments},"
            f" commonest first.\nFill it in and pass it back with --arguments.\n"
        )
        return 0

    model = args.model or _model_name(agent)
    if not model:
        raise CliError(f"agent {agent.name!r} names no model; pass --model")
    instruction = _instruction(agent)

    costing = cost(
        _request(model, instruction, declarations, args.prompt or []),
        protected=FRAMEWORK_TOOLS | cfg.never_prune,
        history_tokens=args.history_tokens,
        schema_chars_per_token=args.schema_chars_per_token,
    )
    keep = min(args.keep if args.keep is not None else cfg.max_tools_kept, costing.prunable)
    prompt_tokens, removed_tokens = costing.sizes(keep)
    verdict = qualify(model, prompt_tokens, removed_tokens, cfg, costing.history_tokens)
    _print_qualify(agent.name, model, costing, keep, verdict, cfg, args.schema_chars_per_token)
    _print_reach(reach, marked=bool(args.arguments))
    return 0 if verdict.proceeds else 1


def _marks(path: str | None) -> dict[str, reachability.Standing]:
    if not path:
        return {}
    try:
        return reachability.read_marks(Path(path).read_text())
    except OSError as exc:
        raise CliError(f"could not read {path}: {exc}") from exc
    except ValueError as exc:
        raise CliError(f"{path}: {exc}") from exc


def _print_qualify(
    agent: str,
    model: str,
    costing: Costing,
    keep: int,
    verdict: Verdict,
    cfg: RouterConfig,
    schema_ratio: float,
) -> None:
    prompt_tokens, removed_tokens = costing.sizes(keep)
    print(f"\nsmart-router qualify — {agent} on {model}\n")
    _row("declared tools", f"{costing.declared:,}")
    _row("prunable", f"{costing.prunable:,} (this menu keeps {keep:,})")
    _row("instruction + prompts", f"{costing.chars.base:,} chars")
    _row("declarations", f"{costing.chars.kept + costing.chars.dropped:,} chars")
    if costing.history_tokens:
        _row("history added by hand", f"{costing.history_tokens:,} tokens")
    print()
    _row("steady-state prompt", f"{prompt_tokens:,} tokens")
    _row("a menu would remove", f"{removed_tokens:,} tokens")
    _row("prompt after the prune", f"{prompt_tokens - removed_tokens:,} tokens")
    _row("context-cache floor", f"{verdict.floor:,} tokens (published)")
    # The floor is where a prefix becomes eligible. The margin is where caching starts
    # happening, and it is what every band below is judged against. See DESIGN.md 19.50.
    _row("a prefix is kept above", f"{verdict.retained:,} tokens (measured)")
    print()
    if verdict.band == "trap":
        # The same number as a percentage is "-570%", which reads as a bug rather than as
        # the several-fold increase it is. Turning caching off really is that expensive.
        print(f"  TRAP  a steady-state turn's input bill goes up {1 - verdict.steady_saving:.1f}x")
        print(f"  {'':<8}{-verdict.saved_tokens:+,} billed tokens per turn\n")
    else:
        print(f"  {verdict.band.upper()}  {verdict.steady_saving:+.1%} of a steady-state turn")
        print(f"  {'':<8}{verdict.saved_tokens:+,} billed tokens saved per turn\n")
    print(textwrap.fill(verdict.reason, width=_WRAP, initial_indent="  ", subsequent_indent="  "))
    print("\n  what this rests on:")
    _note(_menu_range_note(costing, model, cfg))
    _note(
        f"declarations counted at {schema_ratio:.2f} characters per token. Measured between "
        f"5.75 and 8.30 across one repository's own declarations, so a prune can be a quarter "
        f"larger than this says. scripts/measure_menus.py measures yours."
    )
    if not costing.history_tokens:
        _note(
            "no conversation history counted. A real steady-state prompt carries several "
            "turns of it, which raises the denominator and lowers the saving — pass "
            "--history-tokens with what your own turns average."
        )
    _note(
        "turn-one menu residency, which was worth 22 points of G1 on the one agent measured "
        "and which no arithmetic over one prompt can reach (docs/evidence.md). It is a "
        "question about arrival times: a prefix idle for ten minutes retains nothing, and "
        "one sent seconds ago opens warm (scripts/probe_residency.py). Divide your turns per "
        "minute by your menu count; if a menu goes out less than about once a minute, expect "
        "every session to open cold and read the saving above as an upper bound."
    )
    if not counting_vertex_declarations():
        _note(
            "declarations were counted for the Gemini Developer API, because that is what this "
            "shell names. ADK gives every declaration a response schema on Agent Platform and "
            "not here, which was 1,200 tokens across 100 tools on the reference agent. If you "
            "deploy there, export GOOGLE_GENAI_USE_ENTERPRISE=true and run this again."
        )
    _note(
        "scripts/predict_saving.py replaces this the moment the agent has logged traffic, "
        "because it reads those arrival times instead of assuming them."
    )
    print()


def _print_reach(reach: reachability.Survey, marked: bool) -> None:
    """The second question: which of these tools can a turn-one router ever predict?

    Separate from the verdict above and printed under it, because it is a different question
    with a different answer shape. `qualify` says whether pruning pays. This says how much of
    the surface the prediction can reach at all, and the part it cannot reach is not a defect
    to fix — it is the second hop, and menus carry it by co-occurrence (docs/install.md).
    """
    low, high = reach.span
    total = len(reach.tools)
    print("  what a turn-one router can predict:")
    if low == high:
        _note(
            f"{low} of {total} tools can be a first call. The other {total - low} need "
            f"something only your systems hold, so no turn-one prediction ever reaches them "
            f"and menus carry them by co-occurrence instead."
        )
    else:
        _note(
            f"between {low} and {high} of {total} tools can be a first call. {low} are certain "
            f"because they require nothing a user has to know; the span is {high - low} tools "
            f"whose required arguments nobody has classified yet."
        )
        _note(
            "narrow it with --write-arguments FILE, which lists every unclassified argument "
            "name commonest first. Mark each `held` or `supplied`, pass it back with "
            "--arguments FILE, and the span becomes a number. It is a ten minute job and it "
            "costs nothing."
        )
    if reach.second_hop:
        shown = ", ".join(t.name for t in reach.second_hop[:6])
        more = f" and {len(reach.second_hop) - 6} more" if len(reach.second_hop) > 6 else ""
        _note(f"second hop: {shown}{more}.")
    if not marked and not reach.responses_declared:
        _note(
            "no tool declares a response schema, so nothing could be classified structurally. "
            "Nothing is guessed from argument names: `order_id` and `subscription_id` are the "
            "same shape and fall on opposite sides, so a name rule would be confidently wrong "
            "(docs/install.md)."
        )
    print()


def _menu_range_note(costing: Costing, model: str, cfg: RouterConfig) -> str:
    """How much of the verdict is riding on the guess at menu size.

    The guess is the weakest input, and it is optimistic by default: the router keeps every
    tool it has never seen called as well as the menu, which took a published menu of 11 to
    43 realised of 100 on the one agent measured (docs/evidence.md).
    """
    safe, pays = menu_range(costing, model, cfg)
    if pays is None:
        span = f"no menu size reaches the {PAYS_ABOVE:.0%} bar on this agent"
    elif safe is None:
        span = f"pruning to {pays} or fewer clears the bar, but every size trips the cache floor"
    elif safe == 0:
        span = f"any menu of {pays} tools or fewer clears the {PAYS_ABOVE:.0%} bar"
    else:
        span = f"menus between {safe} and {pays} tools clear the bar and stay above the floor"
    return (
        f"{span}. The router keeps more than the menu — every tool it has never seen called "
        f"is kept too, which took menus of 11 to 43 realised of 100 on the reference agent, so "
        f"the default keep of {cfg.max_tools_kept} is the optimistic end of that range."
    )


# --- check-arguments -------------------------------------------------------------


def _check_arguments(args: argparse.Namespace, cfg: RouterConfig) -> int:
    """Hold the reach markings up against which tool each logged session called first.

    Exits 1 when the traffic contradicts a marking, because a marking that says no first call
    can reach a tool, sitting next to sessions where one did, is worth a human's attention
    either way: either the marking is wrong, or the model is calling that tool on an argument
    it made up. Both are things an adopter wants to know and neither is decided here.
    """
    loaded = _load(args.app)
    declarations = _declarations(loaded.agent)
    if not declarations:
        raise CliError(f"agent {loaded.agent.name!r} declares no tools, so there is nothing to ask")

    project = args.project or cfg.bq_project
    if not project:
        raise CliError("no BigQuery project; pass --project or set SMART_ROUTER_BQ_PROJECT")
    agent = args.agent or loaded.agent.name

    reach = reachability.survey(declarations, _marks(args.arguments))
    counted = _first_calls(project, args.dataset or cfg.bq_dataset, agent, args.since)
    if not counted:
        raise CliError(
            f"no successful tool call is logged for agent {agent!r} since {args.since} in "
            f"{project}.{args.dataset or cfg.bq_dataset}. Check --agent against the name the "
            f"analytics plugin writes, and --since against when it was installed."
        )

    found = reachability.audit(reach, counted, floor=args.min_sessions)
    _print_audit(agent, args, reach, found)
    if args.write_arguments:
        Path(args.write_arguments).write_text(
            reachability.template(reach, reachability.demonstrated(found))
        )
        print(f"  wrote {args.write_arguments}, with the traffic's evidence in the margin.\n")
    return 1 if found.contradicted else 0


def _first_calls(project: str, dataset: str, agent: str, since: str) -> dict[str, int]:
    """The query, with its failures turned into something a user can act on."""
    from google.api_core import exceptions as api
    from google.auth import exceptions as auth

    try:
        return openings.first_calls(project, dataset, agent, since)
    except auth.GoogleAuthError as exc:
        raise CliError(
            f"no credentials: {exc}. Run `gcloud auth application-default login`."
        ) from exc
    except api.GoogleAPIError as exc:
        raise CliError(f"BigQuery said no: {exc}") from exc


def _print_audit(
    agent: str,
    args: argparse.Namespace,
    reach: reachability.Survey,
    found: reachability.Audit,
) -> None:
    """What the traffic disagreed with, what it answered, and what it never got to say."""
    low, high = reach.span
    print(f"\nsmart-router check-arguments — {agent}\n")
    _row("sessions with a call", f"{found.sessions:,} since {args.since}")
    _row(
        "distinct first calls", f"{len(found.contradicted) + len(found.settled) + found.confirmed}"
    )
    _row("counted from", f"{found.floor} session{'s' if found.floor != 1 else ''} up")
    _row("reach before asking", f"{low} to {high} of {len(reach.tools)} tools")
    print()

    if found.contradicted:
        print(f"  {len(found.contradicted)} marking(s) say no first call can reach a tool that")
        print("  first calls reached:")
        for witness in found.contradicted:
            _note(
                f"{witness.tool} opened {witness.sessions:,} session(s), and is marked a second "
                f"hop ({witness.why}). Check whether {', '.join(witness.arguments)} was really "
                f"the user's."
            )
        print()

    if found.settled:
        print(f"  {len(found.settled)} tool(s) nobody classified were a first call:")
        for witness in found.settled:
            _note(
                f"{witness.tool} opened {witness.sessions:,} session(s), which took "
                f"{', '.join(witness.arguments)}."
            )
        print()

    print("  what this rests on:")
    _note(
        f"{found.confirmed} tool(s) marked reachable were a first call. "
        f"{len(found.unwitnessed)} were not, which settles nothing: a tool can be perfectly "
        f"reachable and simply never be what anyone wants first."
    )
    _note(
        "the first call of a session, not of the first turn. Until some tool has returned a "
        "value, every argument in the request came out of the conversation."
    )
    _note(
        "a first call proves the tool is reachable at turn one, which is what the span above "
        "is asking. It does not prove your user held the argument. The model normalises what "
        "it was told, so `near Toronto` arrives as country_code=CA, and it invents what it was "
        "not, so `a gift card from Christmas` arrives as card_number=4567-8901-2345-6789. Both "
        "look the same from here, which is why none of this is written into a marking file for "
        "you (docs/install.md)."
    )
    _note(
        f"fabrication concentrates in the tail. On the reference agent every contradiction "
        f"under three sessions was an invented identifier, so --min-sessions is the control "
        f"that matters here and it is currently {found.floor}."
    )
    if found.undeclared:
        shown = ", ".join(found.undeclared[:6])
        more = f" and {len(found.undeclared) - 6} more" if len(found.undeclared) > 6 else ""
        _note(
            f"the log opens with {len(found.undeclared)} tool(s) this agent no longer declares "
            f"({shown}{more}), so some of it predates the tool list being asked about."
        )
    if not args.write_arguments:
        _note(
            "--write-arguments FILE writes the marking file back with these counts in the "
            "margin, so the questions and the evidence are in one place."
        )
    print()


# --- doctor ----------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One wiring question, its answer, and what going wrong actually costs."""

    ok: bool
    title: str
    detail: str


def inspect_app(app: App) -> list[Check]:
    """The three silent misconfigurations, checked against an assembled App.

    Silent is the operative word. None of these stops an agent running, none raises, and
    two of them corrupt the training data rather than the serving path, so the damage shows
    up a retrain later as a model that has learned from its own output.
    """
    # Imported here rather than at module scope so `qualify` does not pay for the predictor
    # stack — xgboost and the GCS client — to answer a question that is pure arithmetic.
    from adk_smart_router.plugin import SmartRouterPlugin

    plugins = list(app.plugins or [])
    routers = [i for i, plugin in enumerate(plugins) if isinstance(plugin, SmartRouterPlugin)]
    analytics = [i for i, plugin in enumerate(plugins) if _is_bq_analytics(plugin)]

    return [
        _check_order(plugins, routers, analytics),
        _check_allowlist(plugins, analytics),
        _check_cache(app),
    ]


def _check_order(plugins: list[Any], routers: list[int], analytics: list[int]) -> Check:
    if not routers:
        return Check(
            ok=False,
            title="SmartRouterPlugin is registered",
            detail=(
                f"App(plugins=[...]) holds {len(plugins)} plugin(s) and none of them is the "
                f"router. Nothing will prune anything."
            ),
        )
    if analytics and min(analytics) < routers[0]:
        return Check(
            ok=False,
            title="SmartRouterPlugin comes before the analytics plugin",
            detail=(
                f"{type(plugins[min(analytics)]).__name__} is at position {min(analytics)} and "
                f"the router at {routers[0]}. Analytics-first logs every cache hit as a real "
                f"tool call with total_ms=0, so measured TTLs inflate without bound and the "
                f"next model trains on the router's own echo (docs/operations.md#plugin-order). "
                f"Move SmartRouterPlugin first."
            ),
        )
    router_plugin = plugins[routers[0]]
    router_cfg = getattr(router_plugin, "_cfg", None)
    if isinstance(router_cfg, RouterConfig):
        has_custom = bool(getattr(router_plugin, "_custom_judges", ()))
        issues = judge_wiring_issues(router_cfg, has_custom_judges=has_custom)
        if issues:
            return Check(
                ok=False,
                title="SmartRouterPlugin judge dependencies are ready",
                detail=" ".join(issues),
            )
    where = (
        f"router at position {routers[0]}, analytics at {min(analytics)}"
        if analytics
        else f"router at position {routers[0]}; no analytics plugin to be ahead of"
    )
    return Check(True, "SmartRouterPlugin comes before the analytics plugin", where)


def _check_allowlist(plugins: list[Any], analytics: list[int]) -> Check:
    title = f"analytics allowlists {PROVENANCE_KEY!r}"
    if not analytics:
        return Check(
            ok=False,
            title=title,
            detail=(
                "no BigQueryAgentAnalyticsPlugin is registered, so nothing writes the traffic "
                "the router is trained from. Install it with "
                f"custom_metadata_allowlist=[{PROVENANCE_KEY!r}]."
            ),
        )
    for index in analytics:
        entries = list(
            getattr(getattr(plugins[index], "config", None), "custom_metadata_allowlist", None)
            or []
        )
        if not any(_allows(entry, PROVENANCE_KEY) for entry in entries):
            return Check(
                ok=False,
                title=title,
                detail=(
                    f"{type(plugins[index]).__name__} allowlists {entries or 'nothing'}. Without "
                    f"{PROVENANCE_KEY!r} the router's provenance never reaches BigQuery, and "
                    f"train_router.sql excludes routed sessions by reading exactly that field "
                    f"— so a retrain silently trains on the router's own pruned menus."
                ),
            )
    return Check(True, title, "routed sessions will be excluded from the next training set")


def _allows(entry: str, key: str) -> bool:
    """BQAA treats a trailing `*` as a prefix pattern and everything else as an exact key."""
    return key.startswith(entry[:-1]) if entry.endswith("*") else entry == key


def _check_cache(app: App) -> Check:
    title = "a ContextCacheConfig is set"
    config = app.context_cache_config
    if config is None:
        return Check(
            ok=False,
            title=title,
            detail=(
                "App(context_cache_config=...) is unset, so the prefix is never cached. "
                "Pruning then saves only the declaration tokens at full rate and none of the "
                "cache discount the router is designed around (docs/install.md), and every "
                "number `smart-router qualify` printed is wrong for this app."
            ),
        )
    return Check(True, title, f"min_tokens={config.min_tokens}, ttl_seconds={config.ttl_seconds}")


def _doctor(args: argparse.Namespace, cfg: RouterConfig) -> int:
    loaded = _load(args.app)
    if loaded.app is None:
        raise CliError(
            f"{args.app} is an agent, not an App. doctor reads plugins and the cache config, "
            f"which live on App(...)."
        )
    checks = inspect_app(loaded.app)
    print(f"\nsmart-router doctor — {loaded.app.name}\n")
    for check in checks:
        print(f"  [{'ok' if check.ok else 'FAIL'}] {check.title}")
        print(
            textwrap.fill(
                check.detail, width=_WRAP, initial_indent=" " * 9, subsequent_indent=" " * 9
            )
        )
    failed = [check for check in checks if not check.ok]
    print(f"\n  {len(checks) - len(failed)} of {len(checks)} ok\n")
    _note(
        "these are the checks that need the assembled App. What needs the cloud — dataset, "
        "connection, bucket, the published artifact — is scripts/provision.sh and "
        "scripts/check_router.py."
    )
    print()
    return 1 if failed else 0


# --- status ----------------------------------------------------------------------


def _status(args: argparse.Namespace, cfg: RouterConfig) -> int:
    """Print what the router is serving, what it cost, and whether it is due a retrain.

    Always exits 0. Everything this prints is a state rather than a verdict, and "nothing is
    published yet" is as legitimate an answer as a saving. `doctor` is the command that fails.
    """
    cfg = replace(
        cfg,
        bq_project=args.project or cfg.bq_project,
        bq_dataset=args.dataset or cfg.bq_dataset,
        gcs_bucket=args.bucket or cfg.gcs_bucket,
    )
    if not cfg.bq_project:
        raise CliError("no BigQuery project; pass --project or set SMART_ROUTER_BQ_PROJECT")
    if not cfg.gcs_bucket:
        raise CliError(
            "no bucket to read the artifact from; pass --bucket or set SMART_ROUTER_GCS_BUCKET"
        )

    for line in status.lines(_read_status(cfg, args)):
        print(line)
    _note(
        "scripts/check_drift.py refreshes the drift line, and scripts/retrain.sh runs a cycle. "
        "docs/observability.md says what each number rests on."
    )
    print()
    return 0


def _read_status(cfg: RouterConfig, args: argparse.Namespace) -> status.Status:
    """The reads, with their failures turned into something a user can act on."""
    from google.api_core import exceptions as api
    from google.auth import exceptions as auth

    try:
        return status.read(cfg, args.agent, days=args.days)
    except auth.GoogleAuthError as exc:
        raise CliError(
            f"no credentials: {exc}. Run `gcloud auth application-default login`."
        ) from exc
    except api.GoogleAPIError as exc:
        raise CliError(f"BigQuery said no: {exc}") from exc


# --- readiness -------------------------------------------------------------------


def _readiness(args: argparse.Namespace, cfg: RouterConfig) -> int:
    """Ask whether the traffic already in the warehouse could train a router today.

    Exits 1 on any stop, because every stop here means the next step would waste money: either
    provisioning for a corpus that cannot train, or training a model that a frequency table
    already beats. A thin corpus exits 0. It is a reason to wait, not a reason to stop.
    """
    project = args.project or cfg.bq_project
    if not project:
        raise CliError("no BigQuery project; pass --project or set SMART_ROUTER_BQ_PROJECT")
    dataset = args.dataset or cfg.bq_dataset

    found = _read_readiness(project, dataset, args, cfg)
    findings = readiness.verdict(found)

    print(f"\nsmart-router readiness — {args.agent}, last {args.days} days of {dataset}\n")
    for finding in findings:
        mark = {"go": "ok", "thin": "thin", "stop": "STOP"}[finding.standing]
        print(f"  [{mark}] {finding.title}")
        print(
            textwrap.fill(
                finding.detail, width=_WRAP, initial_indent=" " * 9, subsequent_indent=" " * 9
            )
        )

    stops = [f for f in findings if f.standing == "stop"]
    thin = [f for f in findings if f.standing == "thin"]
    print()
    if stops:
        _row("verdict", "not yet — see the stops above")
    elif thin:
        _row("verdict", "trainable today, on thin evidence")
    else:
        _row("verdict", "trainable today")
    _row("scanned", f"{found.scanned / 1e6:.1f} MB, nothing written")
    print()
    _note(
        "this reads the analytics log and nothing else, so it needs no provisioning and no "
        "artifact. What it cannot answer is P3, whether a model beats the frequency table, "
        "because that needs embeddings and therefore the connection scripts/provision.sh "
        "creates. docs/install.md picks up from here."
    )
    print()
    return 1 if stops else 0


def _read_readiness(
    project: str, dataset: str, args: argparse.Namespace, cfg: RouterConfig
) -> readiness.Readiness:
    """The reads, with their failures turned into something a user can act on."""
    from google.api_core import exceptions as api
    from google.auth import exceptions as auth

    try:
        return readiness.read(
            project,
            dataset,
            args.agent,
            days=args.days,
            top_n=cfg.max_tools_kept,
            floor=args.min_rows,
        )
    except auth.GoogleAuthError as exc:
        raise CliError(
            f"no credentials: {exc}. Run `gcloud auth application-default login`."
        ) from exc
    except api.GoogleAPIError as exc:
        raise CliError(f"BigQuery said no: {exc}") from exc


# --- Plumbing --------------------------------------------------------------------


def _row(label: str, value: str) -> None:
    print(f"  {label:<24}{value}")


def _note(text: str) -> None:
    print(textwrap.fill(text, width=_WRAP, initial_indent="   - ", subsequent_indent="     "))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-router",
        description="Ask whether pruning can pay for an agent, and whether it is wired up.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser(
        "qualify",
        help="cost this agent's tool menu. No cloud, no credentials, no model call.",
        description=(
            "Size the agent's prompt against the model's context-cache floor and say whether "
            "pruning it can pay. Exits 1 only when pruning would cost money."
        ),
    )
    ask.add_argument(
        "--app",
        required=True,
        metavar="MODULE:ATTRIBUTE",
        help="the App, agent, or zero-argument factory to import, e.g. myapp.agents:app",
    )
    ask.add_argument(
        "--prompt",
        action="append",
        metavar="TEXT",
        help="an example user turn, counted as conversation. Repeatable.",
    )
    ask.add_argument(
        "--history-tokens",
        type=int,
        default=0,
        metavar="N",
        help="conversation a steady-state turn carries beyond --prompt. Raising it lowers "
        "the saving, so leaving it at 0 is the optimistic end.",
    )
    ask.add_argument(
        "--keep",
        type=int,
        metavar="N",
        help="tools a menu keeps (default: max_tools_kept)",
    )
    ask.add_argument("--model", metavar="NAME", help="override the model the agent names")
    ask.add_argument(
        "--arguments",
        metavar="FILE",
        help="which required arguments your users hold, one per line. Narrows the "
        "turn-one reach span from a range to a number. See --write-arguments.",
    )
    ask.add_argument(
        "--write-arguments",
        metavar="FILE",
        help="write the marking file to fill in and exit, commonest argument first",
    )
    ask.add_argument(
        "--schema-chars-per-token",
        type=float,
        default=SCHEMA_CHARS_PER_TOKEN,
        metavar="R",
        help=f"how densely declarations tokenize (default {SCHEMA_CHARS_PER_TOKEN}; measured "
        f"between 5.75 and 8.30, and scripts/measure_menus.py measures yours)",
    )
    ask.set_defaults(run=_qualify)

    ready = commands.add_parser(
        "readiness",
        help="ask whether the traffic already logged could train a router today",
        description=(
            "The router trains on history, so an agent that has been logging to BigQuery Agent "
            "Analytics may already own its corpus. This reads that log and reports whether it "
            "could train today. It provisions nothing and writes nothing. Exits 1 on any stop."
        ),
    )
    ready.add_argument("--agent", required=True, metavar="NAME", help="the agent name in the log")
    ready.add_argument("--project", metavar="ID", help="BigQuery project (default: configured)")
    ready.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset the analytics plugin writes to (default: smart_router)",
    )
    ready.add_argument(
        "--days",
        type=int,
        default=readiness.TRAIN_LOOKBACK_DAYS,
        metavar="N",
        help=f"window to read, matching the training run's lookback "
        f"(default {readiness.TRAIN_LOOKBACK_DAYS})",
    )
    ready.add_argument(
        "--min-rows",
        type=int,
        default=readiness.MIN_CLASS_ROWS,
        metavar="N",
        help=f"rows a tool needs before the training run can learn it, matching "
        f"train_router.sql's min_class_rows (default {readiness.MIN_CLASS_ROWS})",
    )
    ready.set_defaults(run=_readiness)

    traffic = commands.add_parser(
        "check-arguments",
        help="check the reach markings against which tool each logged session called first",
        description=(
            "A marking file is somebody's belief about what their users hold. This holds it up "
            "against which tool each logged session actually called first, and reports where "
            "the two disagree. Exits 1 if any marking is contradicted."
        ),
    )
    traffic.add_argument(
        "--app",
        required=True,
        metavar="MODULE:ATTRIBUTE",
        help="the App, agent, or zero-argument factory to import, e.g. myapp.agents:app",
    )
    traffic.add_argument(
        "--arguments",
        metavar="FILE",
        help="the marking file to check. Omit it to see what the traffic answers on its own.",
    )
    traffic.add_argument(
        "--write-arguments",
        metavar="FILE",
        help="write the marking file back with the traffic's counts in the margin",
    )
    traffic.add_argument("--project", metavar="ID", help="BigQuery project (default: configured)")
    traffic.add_argument(
        "--dataset", metavar="NAME", help="BigQuery dataset (default: smart_router)"
    )
    traffic.add_argument(
        "--agent",
        metavar="NAME",
        help="the agent name in the log, if it differs from the imported agent's",
    )
    traffic.add_argument(
        "--since",
        default="1970-01-01",
        metavar="DATE",
        help="earliest day to read (default: everything logged, which scans every partition)",
    )
    traffic.add_argument(
        "--min-sessions",
        type=int,
        default=1,
        metavar="N",
        help="sessions a tool must have opened before its demonstration counts (default 1). "
        "Raise it if one stray call should not rewrite a marking.",
    )
    traffic.set_defaults(run=_check_arguments)

    check = commands.add_parser(
        "doctor",
        help="check the App wiring that fails silently",
        description=(
            "Check the three things no script can: plugin order, the analytics allowlist, and "
            "the context cache config. Exits 1 if any of them is wrong."
        ),
    )
    check.add_argument(
        "--app",
        required=True,
        metavar="MODULE:ATTRIBUTE",
        help="the App or zero-argument factory to import, e.g. myapp.agents:app",
    )
    check.set_defaults(run=_doctor)

    watch = commands.add_parser(
        "status",
        help="what the router is serving, what it cost, and whether it is due a retrain",
        description=(
            "Read the live artifact, the last week of traffic and the retrain ledger, and print "
            "them on one screen. Read-only, and always exits 0."
        ),
    )
    watch.add_argument("--agent", required=True, metavar="NAME", help="the agent name in the log")
    watch.add_argument("--project", metavar="ID", help="BigQuery project (default: configured)")
    watch.add_argument("--dataset", metavar="NAME", help="BigQuery dataset (default: smart_router)")
    watch.add_argument(
        "--bucket", metavar="NAME", help="the artifact bucket (default: SMART_ROUTER_GCS_BUCKET)"
    )
    watch.add_argument(
        "--days",
        type=int,
        default=status.DEFAULT_DAYS,
        metavar="N",
        help=f"how much traffic to summarise (default {status.DEFAULT_DAYS})",
    )
    watch.set_defaults(run=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `smart-router` console script."""
    args = _parser().parse_args(argv)
    try:
        exit_code: int = args.run(args, load_config())
    except CliError as exc:
        print(f"smart-router: {exc}", file=sys.stderr)
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
