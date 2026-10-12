"""The front door: what `smart-router qualify` costs and what `smart-router doctor` catches.

Two things are worth testing here and they are different in kind. The arithmetic — counting
an agent's declarations and turning them into the two integers `qualify` wants — is checked
against the reference agent, where the answer is known from a measured A/B and not from this
code. The wiring checks are checked against Apps assembled wrong on purpose, because every
one of those mistakes leaves an agent that runs and logs and looks healthy, so a test is the
only place they are ever visible.

The commands are also driven end to end through `main`, exit codes included: a CLI whose
verdict is right and whose exit code is not is a CLI that no one can put in a script.
"""

from __future__ import annotations

import sys
import types as pytypes

import pytest
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from adk_smart_router.cli import (
    Check,
    CliError,
    _load,
    cost,
    inspect_app,
    main,
    menu_range,
)
from adk_smart_router.config import RouterConfig
from adk_smart_router.plugin import SmartRouterPlugin
from adk_smart_router.prompt_size import Chars, TokenScale
from adk_smart_router.qualify import CACHE_FLOOR_HEADROOM

CFG = RouterConfig()
FLASH = "gemini-2.5-flash"
MODULE = "smart_router_cli_fixture"


# --- Fixtures: an agent, an App, and a module to import them from -----------------


def a_tool(order_id: str) -> str:
    """Look up one order. The docstring and the argument are what make the declaration."""
    return f"order {order_id} is fine"


def another_tool(email: str) -> str:
    """Change the customer's email address, which is a different declaration again."""
    return f"changed to {email}"


class BigQueryAgentAnalyticsPluginDouble(BasePlugin):
    """Named to match what `startup_checks` duck-types on, so `doctor` sees it as BQAA."""

    def __init__(self, allowlist: list[str] | None) -> None:
        super().__init__(name="bigquery_agent_analytics")
        self.config = pytypes.SimpleNamespace(custom_metadata_allowlist=allowlist)


def an_agent(instruction: str = "you are a support agent", tools: int = 2) -> LlmAgent:
    return LlmAgent(
        name="support",
        model=FLASH,
        instruction=instruction,
        tools=[a_tool, another_tool][:tools],
    )


CACHE_ON = ContextCacheConfig(min_tokens=2048, ttl_seconds=1800)


def an_app(
    *,
    plugins: list[BasePlugin] | None = None,
    cache: ContextCacheConfig | None = CACHE_ON,
) -> App:
    return App(
        name="support",
        root_agent=an_agent(),
        plugins=plugins if plugins is not None else [],
        context_cache_config=cache,
    )


@pytest.fixture
def published():
    """Publish objects under an importable module name, the way `--app` expects to find them.

    A real module rather than a monkeypatched import, because resolving `MODULE:ATTRIBUTE`
    through `importlib` is part of what is being tested.
    """
    module = pytypes.ModuleType(MODULE)
    sys.modules[MODULE] = module
    yield module
    del sys.modules[MODULE]


def request_of(agent: LlmAgent, prompts: tuple[str, ...] = ()):
    from adk_smart_router.cli import _declarations, _request

    return _request(FLASH, str(agent.instruction), _declarations(agent), prompts)


# --- Resolving --app --------------------------------------------------------------


def test_an_app_target_yields_both_the_app_and_its_root_agent(published):
    published.app = an_app()
    loaded = _load(f"{MODULE}:app")
    assert loaded.app is published.app
    assert loaded.agent.name == "support"


def test_a_bare_agent_target_yields_an_agent_and_no_app(published):
    published.root_agent = an_agent()
    loaded = _load(f"{MODULE}:root_agent")
    assert loaded.app is None
    assert loaded.agent.name == "support"


def test_a_zero_argument_factory_is_called_because_apps_are_usually_built_not_bound(published):
    published.build_app = an_app
    assert _load(f"{MODULE}:build_app").app is not None


def test_a_factory_that_needs_arguments_says_so_instead_of_raising_a_type_error(published):
    def build_app(*, analytics: bool) -> App:
        return an_app()

    published.build_app = build_app
    with pytest.raises(CliError, match="needs arguments"):
        _load(f"{MODULE}:build_app")


def test_a_target_without_a_colon_is_rejected_before_anything_is_imported():
    with pytest.raises(CliError, match="MODULE:ATTRIBUTE"):
        _load("myapp.agents")


def test_a_missing_attribute_names_the_attribute_rather_than_the_import(published):
    with pytest.raises(CliError, match="no attribute 'app'"):
        _load(f"{MODULE}:app")


def test_a_module_that_does_not_import_reports_the_working_directory_it_looked_in():
    with pytest.raises(CliError, match="could not import"):
        _load("no_such_module_anywhere:app")


def test_a_target_that_is_neither_an_app_nor_an_agent_is_rejected(published):
    published.app = {"name": "support"}
    with pytest.raises(CliError, match="not an App or an agent"):
        _load(f"{MODULE}:app")


def test_a_factory_that_raises_at_runtime_reports_a_clean_cli_error(published):
    def broken_factory() -> App:
        raise RuntimeError("set SMART_ROUTER_BQ_PROJECT")

    published.build_app = broken_factory
    with pytest.raises(CliError, match="RuntimeError: set SMART_ROUTER_BQ_PROJECT"):
        _load(f"{MODULE}:build_app")


def test_an_agent_that_declares_no_tools_of_its_own_is_a_clear_error_not_a_crash(published, capsys):
    published.root_agent = SequentialAgent(name="pipeline", sub_agents=[an_agent()])
    assert main(["qualify", "--app", f"{MODULE}:root_agent"]) == 2
    assert "declares no tools" in capsys.readouterr().err


# --- Counting the agent -----------------------------------------------------------


def test_the_instruction_and_the_declarations_are_counted_separately(published):
    agent = an_agent(instruction="x" * 1000)
    costing = cost(
        request_of(agent), protected=frozenset(), history_tokens=0, schema_chars_per_token=8.33
    )
    assert costing.declared == 2
    assert costing.prunable == 2
    assert costing.chars.base == 1000
    assert costing.chars.dropped > 0  # the declarations, which a menu may drop
    assert costing.chars.kept == 0  # nothing is protected on this agent


def test_a_protected_tool_is_counted_as_kept_and_never_offered_to_the_menu(published):
    agent = an_agent()
    costing = cost(
        request_of(agent),
        protected=frozenset({"a_tool"}),
        history_tokens=0,
        schema_chars_per_token=8.33,
    )
    assert costing.prunable == 1
    assert costing.chars.kept > 0


def test_an_example_prompt_is_counted_as_conversation_and_not_as_schema(published):
    agent = an_agent(instruction="x" * 1000)
    costing = cost(
        request_of(agent, ("y" * 500,)),
        protected=frozenset(),
        history_tokens=0,
        schema_chars_per_token=8.33,
    )
    assert costing.chars.base == 1500


def test_a_menu_drops_an_average_sized_share_of_the_prunable_declarations():
    # 42,400 characters of prose is 10,000 tokens at 4.24, and 83,300 of schema is 10,000
    # more at 8.33: a hundred equal declarations worth 100 tokens each.
    costing = _costing(base_chars=42_400, schema_chars=83_300, prunable=100)
    prompt, removed = costing.sizes(11)
    assert prompt == 20_000
    assert removed == pytest.approx(8_900, abs=5)  # 89 tools of 100 tokens


def test_keeping_every_tool_removes_nothing_and_keeping_none_removes_the_whole_menu():
    costing = _costing(base_chars=42_400, schema_chars=83_300, prunable=100)
    assert costing.sizes(100)[1] == 0
    assert costing.sizes(0)[1] == 10_000


def test_hand_entered_history_raises_the_prompt_without_changing_what_a_menu_removes():
    costing = _costing(base_chars=42_400, schema_chars=83_300, prunable=100, history=5_000)
    prompt, removed = costing.sizes(11)
    assert prompt == 25_000
    assert removed == pytest.approx(8_900, abs=5)


def test_a_denser_schema_ratio_makes_the_same_declarations_worth_more_tokens():
    loose = _costing(base_chars=42_400, schema_chars=83_300, prunable=100)
    dense = _costing(base_chars=42_400, schema_chars=83_300, prunable=100, schema_ratio=6.17)
    assert dense.sizes(11)[1] > loose.sizes(11)[1]


def _costing(*, base_chars, schema_chars, prunable, history=0, schema_ratio=8.33):
    from adk_smart_router.cli import Costing

    return Costing(
        declared=prunable,
        prunable=prunable,
        chars=Chars(base=base_chars, kept=0, dropped=schema_chars),
        history_tokens=history,
        scale=TokenScale(schema=schema_ratio),
    )


# --- How much of the verdict depends on the guess at menu size --------------------


def test_the_menu_range_reports_the_smallest_safe_menu_and_the_largest_paying_one():
    costing = _costing(base_chars=42_400, schema_chars=83_300, prunable=100)
    safe, pays = menu_range(costing, FLASH, CFG)
    assert safe == 0  # 10,000 tokens of prose clears the 2,560 a prefix needs by itself
    assert 0 < pays < 100  # and somewhere below 100 tools the prune stops being worth it


def test_an_agent_whose_prose_is_under_the_cache_floor_has_no_safe_empty_menu():
    # 4,240 characters of prose is 1,000 tokens, under the 2,560 a prefix needs before it is
    # kept, so a menu small enough to drop the prompt under that is the trap band.
    costing = _costing(base_chars=4_240, schema_chars=83_300, prunable=100)
    safe, _ = menu_range(costing, FLASH, CFG)
    assert safe is not None and safe > 0
    prompt, removed = costing.sizes(safe - 1)
    assert prompt - removed < 2_048 + CACHE_FLOOR_HEADROOM


def test_an_agent_with_almost_nothing_to_prune_never_reaches_the_bar_at_any_menu_size():
    costing = _costing(base_chars=424_000, schema_chars=8_330, prunable=10)
    assert menu_range(costing, FLASH, CFG)[1] is None


# --- The reference agent, where the answer is known from a measured A/B -----------


@pytest.fixture
def on_vertex(monkeypatch):
    """Pin the backend, because ADK declares a different tool schema for each one.

    On Vertex, ADK gives every declaration a `response_json_schema`. On the Gemini Developer
    API it does not, because that API has nowhere to put one. It is a fixed 74 characters per
    tool and nothing else differs, so the same agent's declared surface is 93,208 characters
    on one backend and 85,808 on the other.

    Every number this section asserts was measured on Vertex, so the tests have to say so. A
    developer with `GOOGLE_GENAI_USE_VERTEXAI` exported reads a different answer from one
    without it, and until this fixture existed the suite passed on the first machine and
    failed in CI, which sets neither.
    """
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")


def test_the_declared_surface_depends_on_which_backend_the_environment_names(
    published, monkeypatch
):
    """The cause of the above, pinned, because it is a real cost difference and not a quirk.

    A hundred tools carry about 1,200 tokens of response schema on Vertex that the same
    hundred do not carry on the Gemini Developer API, which is 7% of this agent's prompt.
    `qualify` counts whichever the environment names, which is correct: it should report the
    prompt the caller will actually be billed for. It does mean the answer is only as right
    as the shell it was run in.
    """
    from reference_agent.agent import root_agent

    from adk_smart_router.cli import _declarations

    def declared_chars() -> int:
        return sum(len(str(d.model_dump(exclude_none=True))) for d in _declarations(root_agent))

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    on_vertex_chars = declared_chars()
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI")
    elsewhere = declared_chars()

    assert on_vertex_chars > elsewhere
    # 100 tools, one response schema each, and nothing else between the two.
    assert (on_vertex_chars - elsewhere) / 100 == pytest.approx(74, abs=10)


def test_the_reference_agent_reproduces_the_a_b_that_measured_23_percent(published, on_vertex):
    """DESIGN.md 19.17 ran this agent's 100 tools at menus that realised 43 kept, and
    measured a steady prompt of 16,681 tokens, a prune of 8,295 and 34.0% of the bill.

    Counted here at the schema ratio 19.17 measured for these declarations (6.17, against
    the 8.33 default that understates them by 26%) and the ~400 tokens of history a session
    carries, the prompt comes out at 16,687 and the prune at 8,170. That the arithmetic
    lands on a measurement it never saw is the only real evidence that `qualify` is
    answering the question it claims to.
    """
    from reference_agent.agent import root_agent

    costing = cost(
        request_of(root_agent),
        protected=frozenset(),
        history_tokens=400,
        schema_chars_per_token=6.17,
    )
    assert costing.declared == 100
    prompt, removed = costing.sizes(43)
    assert prompt == pytest.approx(16_681, abs=100)
    assert removed == pytest.approx(8_295, abs=200)

    from adk_smart_router.qualify import qualify

    verdict = qualify(FLASH, prompt, removed, CFG)
    assert verdict.band == "pays"
    assert verdict.steady_saving == pytest.approx(0.340, abs=0.01)


def test_the_reference_agent_qualifies_and_the_command_exits_zero(published, capsys):
    published.root_agent = __import__("reference_agent.agent", fromlist=["root_agent"]).root_agent
    assert main(["qualify", "--app", f"{MODULE}:root_agent"]) == 0
    printed = capsys.readouterr().out
    assert "PAYS" in printed
    assert "predict_saving.py" in printed  # the better answer, named every time


def test_qualify_says_so_when_it_counted_the_smaller_of_the_two_backends(
    published, capsys, monkeypatch
):
    """Underreporting is the direction worth a caveat, and the only one that gets one.

    Someone deciding whether the router is worth installing runs `qualify` on a laptop, where
    neither variable is exported, and reads a prompt 7% smaller than the one Agent Platform
    will bill them for. Counting the other way round costs them nothing, so a caller who has
    already named Agent Platform gets no line about it.

    The advice names `GOOGLE_GENAI_USE_ENTERPRISE`, because ADK warns on the older
    `GOOGLE_GENAI_USE_VERTEXAI`. Both are still honoured on the reading side, and this asserts
    that too: whichever one the caller exports, the note goes away.
    """
    published.root_agent = __import__("reference_agent.agent", fromlist=["root_agent"]).root_agent
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
        monkeypatch.delenv(name, raising=False)
    assert main(["qualify", "--app", f"{MODULE}:root_agent"]) == 0
    out = capsys.readouterr().out
    assert "GOOGLE_GENAI_USE_ENTERPRISE=true" in out
    assert "GOOGLE_GENAI_USE_VERTEXAI" not in out

    for name in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.setenv(name, "true")
        assert main(["qualify", "--app", f"{MODULE}:root_agent"]) == 0
        assert "GOOGLE_GENAI_USE_ENTERPRISE=true" not in capsys.readouterr().out
        monkeypatch.delenv(name)


# --- The commands, end to end -----------------------------------------------------


def test_a_trapped_agent_exits_nonzero_so_a_script_can_refuse_to_continue(published, capsys):
    # 10,685 characters of instruction and two small tools puts the whole prompt just over the
    # 2,560 a prefix needs to be kept and the pruned one just under it, which is the trap.
    published.root_agent = an_agent(instruction="x" * 10_685, tools=2)
    assert main(["qualify", "--app", f"{MODULE}:root_agent", "--keep", "0"]) == 1
    printed = capsys.readouterr().out
    assert "TRAP" in printed
    # Reported as a multiple, because the same ratio as a percentage reads as a bug.
    assert "input bill goes up" in printed
    assert "-570" not in printed


def test_an_unusable_target_exits_two_and_says_why_on_stderr(published, capsys):
    assert main(["qualify", "--app", "not_a_module:app"]) == 2
    assert "smart-router: could not import" in capsys.readouterr().err


def test_qualify_never_needs_the_model_the_agent_names_to_exist(published, capsys):
    published.root_agent = an_agent(instruction="x" * 20_000)
    assert main(["qualify", "--app", f"{MODULE}:root_agent", "--model", "gemini-3-pro"]) == 0
    assert "4,096 tokens" in capsys.readouterr().out  # the Gemini 3 floor, not Flash's


def test_an_agent_naming_no_model_is_asked_for_one_rather_than_guessed_at(published, capsys):
    published.root_agent = LlmAgent(name="support", instruction="hi", tools=[a_tool])
    assert main(["qualify", "--app", f"{MODULE}:root_agent"]) == 2
    assert "names no model" in capsys.readouterr().err


def test_doctor_on_an_agent_says_to_point_at_the_app_where_the_plugins_live(published):
    published.root_agent = an_agent()
    assert main(["doctor", "--app", f"{MODULE}:root_agent"]) == 2


def test_there_are_exactly_five_subcommands():
    """`qualify` and `doctor` are day zero, `readiness` sits between them and the install,
    `status` is day two, `check-arguments` is the narrow one. The fifth has its reason in
    DESIGN.md 20.11; a sixth wants one too."""
    from adk_smart_router.cli import _parser

    actions = [a for a in _parser()._actions if a.choices and "qualify" in a.choices]
    assert sorted(actions[0].choices) == [
        "check-arguments",
        "doctor",
        "qualify",
        "readiness",
        "status",
    ]


# --- status: the day-two command --------------------------------------------------


@pytest.fixture
def readout(monkeypatch):
    """Stand in for the warehouse. Returns a setter taking the lines to print.

    The two environment variables are cleared because two of these tests are about what
    happens when neither is set, and a developer who has them exported would otherwise see
    those tests pass on their machine for the wrong reason.
    """
    from adk_smart_router import cli

    monkeypatch.delenv("SMART_ROUTER_BQ_PROJECT", raising=False)
    monkeypatch.delenv("SMART_ROUTER_GCS_BUCKET", raising=False)

    def use(lines):
        monkeypatch.setattr(cli.status, "read", lambda *a, **k: object())
        monkeypatch.setattr(cli.status, "lines", lambda report: lines)

    return use


def test_status_needs_a_project_it_can_read_and_says_which_variable_sets_one(readout, capsys):
    readout(["nothing"])
    assert main(["status", "--agent", "support"]) == 2
    assert "SMART_ROUTER_BQ_PROJECT" in capsys.readouterr().err


def test_status_needs_a_bucket_because_the_live_artifact_is_in_one(readout, capsys):
    readout(["nothing"])
    assert main(["status", "--agent", "support", "--project", "p"]) == 2
    assert "SMART_ROUTER_GCS_BUCKET" in capsys.readouterr().err


def test_status_reports_rather_than_judges_so_it_exits_zero(readout, capsys):
    readout(["live artifact      none"])
    argv = ["status", "--agent", "support", "--project", "p", "--bucket", "b"]
    assert main(argv) == 0
    assert "live artifact      none" in capsys.readouterr().out


def test_status_says_what_to_run_next(readout, capsys):
    readout([""])
    main(["status", "--agent", "support", "--project", "p", "--bucket", "b"])
    out = capsys.readouterr().out
    assert "check_drift.py" in out and "retrain.sh" in out


def test_status_turns_a_credentials_failure_into_the_command_that_fixes_it(monkeypatch, capsys):
    from google.auth import exceptions as auth

    from adk_smart_router import cli

    def refuse(*args, **kwargs):
        raise auth.DefaultCredentialsError("none found")

    monkeypatch.setattr(cli.status, "read", refuse)
    argv = ["status", "--agent", "support", "--project", "p", "--bucket", "b"]
    assert main(argv) == 2
    assert "gcloud auth application-default login" in capsys.readouterr().err


# --- check-arguments: the markings against the traffic ----------------------------


@pytest.fixture
def logged(monkeypatch):
    """Stand in for the query. Returns a setter taking `{tool: sessions}`."""
    from adk_smart_router import cli

    def use(counts):
        monkeypatch.setattr(cli.openings, "first_calls", lambda *a, **k: dict(counts))

    monkeypatch.setenv("SMART_ROUTER_BQ_PROJECT", "a-project")
    use({})
    return use


def marked(tmp_path, text):
    path = tmp_path / "arguments.txt"
    path.write_text(text)
    return str(path)


def test_a_marking_the_traffic_disagrees_with_exits_one_so_a_script_can_stop(
    published, capsys, logged, tmp_path
):
    published.root_agent = an_agent()
    logged({"a_tool": 31})
    arguments = marked(tmp_path, "order_id held\nemail held\n")

    code = main(["check-arguments", "--app", f"{MODULE}:root_agent", "--arguments", arguments])

    assert code == 1
    printed = capsys.readouterr().out
    assert "1 marking(s) say no first call can reach a tool" in printed
    assert "a_tool opened 31 session(s)" in printed
    assert "Check whether order_id was really the user's" in printed


def test_an_unclassified_tool_the_traffic_opened_with_is_reported_and_exits_zero(
    published, capsys, logged
):
    published.root_agent = an_agent()
    logged({"a_tool": 31})

    assert main(["check-arguments", "--app", f"{MODULE}:root_agent"]) == 0
    printed = capsys.readouterr().out
    assert "1 tool(s) nobody classified were a first call" in printed
    assert "a_tool opened 31 session(s), which took order_id" in printed


def test_the_command_never_answers_the_marking_file_on_the_user_s_behalf(
    published, capsys, logged, tmp_path
):
    """The one thing it must not do. A count in the margin is evidence; a verdict written into
    this file would be a guess that every reach number downstream then rests on."""
    published.root_agent = an_agent()
    logged({"a_tool": 31})
    out = tmp_path / "learned.txt"

    code = main(["check-arguments", "--app", f"{MODULE}:root_agent", "--write-arguments", str(out)])

    assert code == 0
    from adk_smart_router import reachability

    assert reachability.read_marks(out.read_text()) == {}
    assert "opened 31" in out.read_text()
    assert "email" in out.read_text()  # the question the traffic said nothing about, still asked


def test_the_fabrication_caveat_is_printed_whether_or_not_anything_disagreed(
    published, capsys, logged
):
    published.root_agent = an_agent()
    logged({"a_tool": 31})
    main(["check-arguments", "--app", f"{MODULE}:root_agent"])

    printed = capsys.readouterr().out
    assert "does not prove your user held the argument" in printed
    assert "card_number=4567-8901-2345-6789" in printed


def test_the_floor_is_a_flag_because_a_noisy_log_should_not_rewrite_a_marking(
    published, capsys, logged, tmp_path
):
    published.root_agent = an_agent()
    logged({"a_tool": 1})
    arguments = marked(tmp_path, "order_id held\n")
    argv = ["check-arguments", "--app", f"{MODULE}:root_agent", "--arguments", arguments]

    assert main(argv) == 1
    assert main([*argv, "--min-sessions", "2"]) == 0


def test_an_agent_with_nothing_logged_says_where_to_look_rather_than_reporting_zero(
    published, capsys, logged
):
    published.root_agent = an_agent()
    logged({})

    assert main(["check-arguments", "--app", f"{MODULE}:root_agent"]) == 2
    assert "no successful tool call is logged" in capsys.readouterr().err


def test_no_configured_project_is_asked_for_rather_than_guessed_at(
    published, capsys, monkeypatch, logged
):
    published.root_agent = an_agent()
    monkeypatch.delenv("SMART_ROUTER_BQ_PROJECT", raising=False)

    assert main(["check-arguments", "--app", f"{MODULE}:root_agent"]) == 2
    assert "no BigQuery project" in capsys.readouterr().err


def test_the_agent_queried_for_is_the_imported_one_unless_the_log_calls_it_something_else(
    published, capsys, monkeypatch
):
    from adk_smart_router import cli

    asked = []

    def record(project, dataset, agent, since):
        asked.append((project, dataset, agent, since))
        return {"a_tool": 4}

    monkeypatch.setattr(cli.openings, "first_calls", record)
    published.root_agent = an_agent()

    main(["check-arguments", "--app", f"{MODULE}:root_agent", "--project", "p"])
    main(["check-arguments", "--app", f"{MODULE}:root_agent", "--project", "p", "--agent", "old"])

    assert [call[2] for call in asked] == ["support", "old"]
    assert asked[0][1] == "smart_router"  # the configured default dataset


# --- doctor: the three mistakes that fail silently --------------------------------


def failures(checks: list[Check]) -> list[str]:
    return [check.title for check in checks if not check.ok]


def test_a_correctly_wired_app_passes_all_three_checks():
    app = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig()),
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
        ]
    )
    assert failures(inspect_app(app)) == []


def test_analytics_registered_before_the_router_fails_the_order_check():
    app = an_app(
        plugins=[
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
            SmartRouterPlugin(RouterConfig()),
        ]
    )
    checks = inspect_app(app)
    assert failures(checks) == ["SmartRouterPlugin comes before the analytics plugin"]
    assert "total_ms=0" in checks[0].detail  # what going wrong actually costs


def test_an_app_with_no_router_at_all_is_reported_rather_than_passing_vacuously():
    app = an_app(plugins=[BigQueryAgentAnalyticsPluginDouble(["smart_router"])])
    assert "SmartRouterPlugin is registered" in failures(inspect_app(app))


def test_an_analytics_plugin_without_the_provenance_key_fails_the_allowlist_check():
    app = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig()),
            BigQueryAgentAnalyticsPluginDouble(["something_else"]),
        ]
    )
    checks = inspect_app(app)
    assert failures(checks) == ["analytics allowlists 'smart_router'"]
    assert "trains on the router's own pruned menus" in checks[1].detail


def test_an_empty_allowlist_fails_the_same_way_a_wrong_one_does():
    app = an_app(
        plugins=[SmartRouterPlugin(RouterConfig()), BigQueryAgentAnalyticsPluginDouble(None)]
    )
    assert "analytics allowlists 'smart_router'" in failures(inspect_app(app))


def test_a_star_suffixed_allowlist_entry_covers_the_key_because_bqaa_treats_it_as_a_prefix():
    app = an_app(
        plugins=[SmartRouterPlugin(RouterConfig()), BigQueryAgentAnalyticsPluginDouble(["smart*"])]
    )
    assert failures(inspect_app(app)) == []


def test_a_plain_entry_is_not_treated_as_a_prefix():
    app = an_app(
        plugins=[SmartRouterPlugin(RouterConfig()), BigQueryAgentAnalyticsPluginDouble(["smart"])]
    )
    assert "analytics allowlists 'smart_router'" in failures(inspect_app(app))


def test_no_analytics_plugin_fails_the_allowlist_check_because_nothing_logs_the_traffic():
    app = an_app(plugins=[SmartRouterPlugin(RouterConfig())])
    checks = inspect_app(app)
    assert "analytics allowlists 'smart_router'" in failures(checks)
    assert "nothing writes the traffic" in checks[1].detail


def test_an_app_without_a_context_cache_config_fails_the_cache_check():
    app = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig()),
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
        ],
        cache=None,
    )
    checks = inspect_app(app)
    assert failures(checks) == ["a ContextCacheConfig is set"]
    assert "every number `smart-router qualify` printed is wrong" in checks[2].detail


def test_doctor_exits_one_when_a_check_fails_and_zero_when_none_do(published):
    published.broken = an_app(plugins=[], cache=None)
    published.fixed = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig()),
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
        ]
    )
    assert main(["doctor", "--app", f"{MODULE}:broken"]) == 1
    assert main(["doctor", "--app", f"{MODULE}:fixed"]) == 0


def test_doctor_prints_every_check_not_just_the_first_failure(published, capsys):
    published.broken = an_app(plugins=[], cache=None)
    main(["doctor", "--app", f"{MODULE}:broken"])
    printed = capsys.readouterr().out
    assert printed.count("[FAIL]") == 3
    assert "0 of 3 ok" in printed


def test_the_base_agent_import_is_the_one_doctor_type_checks_against():
    # Guards the isinstance in `_load`: an App's root_agent must be a BaseAgent for the
    # agent half of the CLI to be reachable at all.
    assert isinstance(an_agent(), BaseAgent)


def test_a_declaration_free_request_still_counts_its_prose():
    from adk_smart_router.cli import _request

    costing = cost(
        _request(FLASH, "x" * 424, (), ("y" * 424,)),
        protected=frozenset(),
        history_tokens=0,
        schema_chars_per_token=8.33,
    )
    assert costing.declared == 0
    assert costing.sizes(0) == (200, 0)


def test_the_request_built_for_counting_carries_the_declarations_it_was_given():
    from adk_smart_router.cli import _declarations, _request

    request = _request(FLASH, "hi", _declarations(an_agent()), ("hello",))
    tool = request.config.tools[0]
    assert isinstance(tool, types.Tool)
    assert {d.name for d in tool.function_declarations or []} == {"a_tool", "another_tool"}


def test_doctor_fails_when_active_xgboost_lacks_bucket_and_passes_for_cold_start_prototype():
    broken_xgb = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig(mode="active", judges=("xgboost",), gcs_bucket="")),
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
        ]
    )
    checks_broken = inspect_app(broken_xgb)
    assert failures(checks_broken) == ["SmartRouterPlugin judge dependencies are ready"]
    assert "no gcs_bucket configured" in checks_broken[0].detail

    healthy_proto = an_app(
        plugins=[
            SmartRouterPlugin(RouterConfig(mode="active", judges=("prototype",), gcs_bucket="")),
            BigQueryAgentAnalyticsPluginDouble(["smart_router"]),
        ]
    )
    assert failures(inspect_app(healthy_proto)) == []
