"""Tests for the xLAM agent generator.

The generated module is committed, so the thing that has to be right is not that it runs
but that what it writes is faithful. Two claims carry E1's whole argument. The declarations
the model reads have to be xLAM's words, not ours, or the fixture is the reference agent
wearing a hat. And the corpus has to hold only queries the agent can actually answer, or
the recall number grades the sampling rule instead of the router.

The rest is type coercion, which has no interesting judgement in it and every opportunity
to produce a module that will not import.
"""

from __future__ import annotations

import json

from build_xlam_agent import (
    Param,
    Tool,
    called,
    corpus,
    literal,
    load,
    module,
    parameter,
    render,
    schemas,
    surface,
    tool_of,
)

SCHEMA = {
    "name": "get_ip_zipcode",
    "description": "Retrieves the ZIP code of a given IP address.",
    "parameters": {"ip": {"description": "The IP address to locate.", "type": "str"}},
}


def row(row_id: int, query: str, calls: list[str], tools: list[dict] | None = None) -> dict:
    return {
        "id": row_id,
        "query": query,
        "answers": json.dumps([{"name": name, "arguments": {}} for name in calls]),
        "tools": json.dumps(tools if tools is not None else [SCHEMA]),
    }


def test_the_surface_is_the_most_called_tools_in_order():
    rows = [row(0, "a", ["x"]), row(1, "b", ["x"]), row(2, "c", ["y"]), row(3, "d", ["z"])]

    assert surface(rows, 2) == ["x", "y"]


def test_popularity_counts_a_tool_once_per_row_that_calls_it():
    """Two calls to the same tool in one query is one query wanting that tool, which is
    what a menu has to cover. Counting calls instead would let a tool that gets invoked in
    a loop outrank one that half the traffic needs."""
    rows = [row(0, "a", ["x", "x", "x"]), row(1, "b", ["y"]), row(2, "c", ["y"])]

    assert surface(rows, 1) == ["y"]


def test_a_query_is_kept_only_when_every_tool_it_calls_is_on_the_surface():
    """The sampling rule. A row calling something the agent does not have is not a hard
    case for the router, it is a different agent, and grading it would measure this rule."""
    rows = [row(0, "keep", ["x"]), row(1, "drop", ["x", "off_surface"]), row(2, "keep2", ["y"])]

    kept = corpus(rows, ["x", "y"])

    assert [q["query"] for q in kept] == ["keep", "keep2"]


def test_a_query_that_calls_nothing_is_not_in_the_corpus():
    """Not because it is uninteresting, but because it cannot drive a live session: there
    is no tool call to observe and nothing for the router to have been right or wrong
    about. `score_xlam.py` keeps these rows; a traffic generator cannot."""
    assert corpus([row(0, "nothing", [])], ["x"]) == []


def test_the_modal_schema_wins_when_a_tool_is_described_two_ways():
    """Eleven of the top hundred are worded more than one way. The agent needs exactly one
    declaration, and the version most of the corpus was written against is the honest pick."""
    common = {"name": "x", "description": "the common one", "parameters": {}}
    rare = {"name": "x", "description": "the rare one", "parameters": {}}
    rows = [row(0, "a", ["x"], [common]), row(1, "b", ["x"], [common]), row(2, "c", ["x"], [rare])]

    assert schemas(rows, ["x"])["x"]["description"] == "the common one"


def test_the_declaration_carries_xlams_words_and_not_ours():
    """E1's load-bearing claim. If the description or the parameter name were rewritten
    here, the surface the model reads would be ours again and the experiment would be
    measuring the reference agent with extra steps."""
    source = render(tool_of(SCHEMA))

    assert "def get_ip_zipcode(ip: str)" in source
    assert "Retrieves the ZIP code of a given IP address." in source
    assert "ip: The IP address to locate." in source


def test_a_tool_with_no_parameters_still_renders():
    source = render(tool_of({"name": "ping", "description": "Ping.", "parameters": {}}))

    assert "def ping() -> dict[str, str]:" in source
    assert "Args:" not in source


def test_a_tool_with_no_description_gets_one_rather_than_an_empty_docstring():
    """An empty description is a declaration Gemini will reject, and one blank tool would
    take the whole agent down at construction."""
    assert "Call bare." in render(tool_of({"name": "bare", "parameters": {}}))


def test_required_parameters_come_before_optional_ones():
    """Forced by Python, not chosen. xLAM's dictionaries are in neither order, and the
    order is not part of the declaration, so nothing about the surface moves."""
    schema = {
        "name": "mixed",
        "description": "Mixed.",
        "parameters": {
            "opt": {"type": "int, optional", "default": "3", "description": "o"},
            "req": {"type": "str", "description": "r"},
        },
    }

    assert [p.name for p in tool_of(schema).params] == ["req", "opt"]


def test_optional_parameters_carry_their_xlam_default():
    spec = {"type": "int, optional", "default": "20", "description": "d"}

    assert parameter("doubling_time", spec) == Param("doubling_time", "int", "d", "20")


def test_an_optional_parameter_whose_default_will_not_coerce_gets_an_empty_one():
    """xLAM's defaults are free text and some of them are prose. A parameter that cannot
    take its stated default still has to be declared, so it takes the type's empty value."""
    spec = {"type": "int, optional", "default": "whatever the caller wants", "description": "d"}

    assert parameter("n", spec).default == "0"


def test_a_required_parameter_has_no_default():
    assert parameter("ip", {"type": "str", "description": "d"}).default is None


def test_an_unrecognised_type_is_declared_as_a_string_rather_than_dropped():
    """Lossy on purpose. A nested generic Gemini rejects would take the tool out of the
    surface entirely, and a wrong-but-simple annotation still declares the parameter."""
    assert parameter("f", {"type": "Callable[[float], float]"}).annotation == "str"


def test_a_parameter_that_is_not_a_python_name_is_dropped():
    assert parameter("class", {"type": "str"}) is None
    assert parameter("not a name", {"type": "str"}) is None


def test_literal_coercion_covers_the_types_that_have_defaults():
    assert literal("hello", "str") == "'hello'"
    assert literal("true", "bool") == "True"
    assert literal("7", "int") == "7"
    assert literal("1.5", "float") == "1.5"
    assert literal("", "int") is None


def test_the_generated_module_imports_and_declares_every_tool():
    """The end-to-end check, run against a two-tool surface so it stays readable. If this
    passes, the hundred-tool version is the same code path with more of it."""
    source = module(
        [tool_of(SCHEMA), tool_of({"name": "ping", "description": "P.", "parameters": {}})]
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102 - that is the test

    assert [f.__name__ for f in namespace["ALL_TOOLS"]] == ["get_ip_zipcode", "ping"]
    assert namespace["get_ip_zipcode"]("1.2.3.4") == {
        "tool": "get_ip_zipcode",
        "status": "ok",
        "arguments": "ip='1.2.3.4'",
    }


def test_a_stub_with_no_arguments_says_so_rather_than_returning_a_blank():
    source = module([Tool("ping", "P.", ())])
    namespace: dict[str, object] = {}
    exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102

    assert namespace["ping"]()["arguments"] == "none"


def test_called_reads_the_json_string_inside_the_json():
    """xLAM nests JSON in JSON, which is the dataset's own shape and easy to half-parse."""
    assert called(row(0, "a", ["x", "y"])) == frozenset({"x", "y"})


def test_load_auto_fetch_downloads_when_missing(tmp_path, monkeypatch):
    """`--fetch` downloads the corpus on demand so derived CC-BY-4.0 files need not be committed."""
    import urllib.request

    target = tmp_path / ".xlam" / "xlam.json"
    sample = [row(0, "q", ["get_ip_zipcode"])]

    def fake_retrieve(url, dest):
        dest.write_text(json.dumps(sample))

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    loaded = load(target, auto_fetch=True)
    assert len(loaded) == 1
    assert loaded[0]["query"] == "q"
