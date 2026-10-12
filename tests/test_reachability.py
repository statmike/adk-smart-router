"""What the surface says about itself, and what it refuses to say.

The load-bearing test in here is the last one: nothing is inferred from an argument's name.
Everything else is a boundary. The module's whole value is that its answers are either
structural or somebody's stated judgement, so a rule that quietly started guessing would make
every number it prints untrustworthy while every other test stayed green.
"""

from __future__ import annotations

import pytest
from google.genai import types

from adk_smart_router import reachability


def declare(
    name: str,
    required: dict[str, dict[str, object]] | None = None,
    optional: dict[str, dict[str, object]] | None = None,
    returns: list[str] | None = None,
) -> types.FunctionDeclaration:
    """A declaration in the shape ADK writes for a plain Python function."""
    properties = {**(required or {}), **(optional or {})}
    declaration = types.FunctionDeclaration(name=name)
    if properties:
        declaration.parameters_json_schema = {
            "type": "object",
            "properties": properties,
            "required": list(required or {}),
        }
    if returns:
        declaration.response_json_schema = {
            "type": "object",
            "properties": {key: {"type": "string"} for key in returns},
        }
    return declaration


STRING: dict[str, object] = {"type": "string"}


# --- What it decides by itself ------------------------------------------------------


def test_a_tool_that_requires_nothing_is_reachable_with_no_help():
    result = reachability.survey([declare("list_orders")])
    assert result.span == (1, 1)
    assert result.tools[0].verdict == "reachable"
    assert "requires nothing" in result.tools[0].why


def test_an_optional_argument_cannot_stop_a_tool_being_reachable():
    """Omitting it still reaches the tool, so it is not what prediction is blocked on."""
    result = reachability.survey([declare("search", optional={"page": STRING})])
    assert result.tools[0].verdict == "reachable"


def test_an_enum_argument_is_supplied_because_the_schema_lists_the_answers():
    tool = declare("filter", required={"status": {"type": "string", "enum": ["open", "shut"]}})
    result = reachability.survey([tool])
    assert result.tools[0].verdict == "reachable"
    assert "one of 2 values" in result.tools[0].gates[0].why


def test_a_boolean_argument_is_supplied_because_there_is_nothing_to_know():
    result = reachability.survey([declare("toggle", required={"on": {"type": "boolean"}})])
    assert result.tools[0].verdict == "reachable"


def test_an_argument_another_tool_returns_makes_this_one_a_second_hop():
    tools = [
        declare("search_products", returns=["sku"]),
        declare("get_price", required={"sku": STRING}),
    ]
    result = reachability.survey(tools)
    assert result.tools[1].verdict == "second-hop"
    assert "search_products returns it" in result.tools[1].why
    assert result.responses_declared == 1


def test_a_tool_that_returns_its_own_required_argument_is_not_a_second_hop_on_itself():
    """`refresh(token) -> token` is a loop, not a dependency. It says nothing about turn one."""
    tool = declare("refresh", required={"token": STRING}, returns=["token"])
    assert reachability.survey([tool]).tools[0].verdict == "unmarked"


# --- The span -----------------------------------------------------------------------


def test_an_unmarked_argument_leaves_a_span_rather_than_a_number():
    tools = [declare("a"), declare("b", required={"sku": STRING})]
    result = reachability.survey(tools)
    assert result.span == (1, 2)
    assert [t.name for t in result.unmarked] == ["b"]
    assert result.open_names() == [("sku", 1)]


def test_marking_the_argument_closes_the_span():
    tools = [declare("a"), declare("b", required={"sku": STRING})]
    assert reachability.survey(tools, {"sku": "held"}).span == (1, 1)
    assert reachability.survey(tools, {"sku": "supplied"}).span == (2, 2)


def test_one_held_argument_settles_a_tool_however_many_others_are_supplied():
    tool = declare("start_return", required={"order_id": STRING, "sku": STRING})
    result = reachability.survey([tool], {"order_id": "supplied", "sku": "held"})
    assert result.tools[0].verdict == "second-hop"
    assert result.tools[0].why.startswith("sku:")


def test_a_tool_is_only_reachable_once_nothing_is_left_undecided():
    tool = declare("thing", required={"a": STRING, "b": STRING})
    assert reachability.survey([tool], {"a": "supplied"}).tools[0].verdict == "unmarked"
    assert "waiting on b" in reachability.survey([tool], {"a": "supplied"}).tools[0].why


def test_a_tool_specific_mark_overrides_the_bare_name():
    """One exception must not force marking every other use of the name by hand."""
    tools = [declare("x", required={"id": STRING}), declare("y", required={"id": STRING})]
    result = reachability.survey(tools, {"id": "held", "y.id": "supplied"})
    assert [t.verdict for t in result.tools] == ["second-hop", "reachable"]


def test_open_names_are_ordered_by_how_many_tools_the_decision_settles():
    tools = [
        declare("a", required={"sku": STRING}),
        declare("b", required={"sku": STRING}),
        declare("c", required={"zip": STRING}),
    ]
    assert reachability.survey(tools).open_names() == [("sku", 2), ("zip", 1)]


# --- Both schema shapes -------------------------------------------------------------


def test_a_typed_schema_declaration_is_read_the_same_as_a_json_one():
    """ADK writes the dict for a plain function and hand-written tools use the typed field."""
    typed = types.FunctionDeclaration(
        name="get",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"sku": types.Schema(type=types.Type.STRING)},
            required=["sku"],
        ),
    )
    assert reachability.survey([typed]).open_names() == [("sku", 1)]
    assert reachability.survey([typed], {"sku": "held"}).tools[0].verdict == "second-hop"


def test_a_response_schema_with_no_properties_names_nothing_and_is_not_counted():
    """ADK writes `{"type": "object"}` for an unannotated return. It is not evidence."""
    tool = types.FunctionDeclaration(name="x", response_json_schema={"type": "object"})
    assert reachability.survey([tool]).responses_declared == 0


# --- What the traffic demonstrates ----------------------------------------------------

# A surface with one of each: settled by the schema, blocked by a marking, and open.
OPEN = declare("get_order", required={"order_id": STRING})
BLOCKED = declare("get_subscription", required={"subscription_id": STRING})
FREE = declare("list_orders")


def surveyed(*tools, marks=None):
    return reachability.survey(list(tools), marks or {})


def test_a_marking_the_traffic_disagrees_with_is_a_contradiction_and_names_the_fix():
    """The whole point of the command. Somebody said a user cannot hold this, and one did."""
    result = surveyed(BLOCKED, marks={"subscription_id": "held"})
    found = reachability.audit(result, {"get_subscription": 12})

    assert [w.tool for w in found.contradicted] == ["get_subscription"]
    assert found.contradicted[0].sessions == 12
    assert found.contradicted[0].was == "second-hop"
    assert found.contradicted[0].arguments == ("subscription_id",)
    assert "marked held" in found.contradicted[0].why


def test_an_open_question_the_traffic_answers_is_settled_and_is_not_a_contradiction():
    found = reachability.audit(surveyed(OPEN), {"get_order": 7})

    assert not found.contradicted
    assert [w.tool for w in found.settled] == ["get_order"]
    assert found.settled[0].arguments == ("order_id",)
    assert found.settled[0].was == "unmarked"


def test_a_tool_the_schema_already_settled_is_confirmed_and_reported_nowhere_else():
    found = reachability.audit(surveyed(FREE), {"list_orders": 40})

    assert found.confirmed == 1
    assert not found.contradicted and not found.settled and not found.unwitnessed


def test_a_reachable_tool_nobody_opened_with_is_only_ever_the_weak_finding():
    """Never a contradiction. A tool can be perfectly reachable and never be wanted first."""
    found = reachability.audit(surveyed(FREE), {"get_order": 3})

    assert found.unwitnessed == ("list_orders",)
    assert found.confirmed == 0
    assert not found.contradicted


def test_the_floor_keeps_one_stray_call_from_rewriting_a_marking():
    result = surveyed(BLOCKED, marks={"subscription_id": "held"})

    assert reachability.audit(result, {"get_subscription": 1}).contradicted
    assert not reachability.audit(result, {"get_subscription": 1}, floor=2).contradicted
    assert reachability.audit(result, {"get_subscription": 2}, floor=2).contradicted


def test_a_tool_the_agent_no_longer_declares_is_reported_rather_than_ignored():
    """It is why an adopter's counts do not add up, and the log predating the tool list is
    the likeliest reason a run finds nothing."""
    found = reachability.audit(surveyed(OPEN), {"get_order": 4, "cancel_order": 9})

    assert found.undeclared == ("cancel_order",)
    assert [w.tool for w in found.settled] == ["get_order"]


def test_the_denominator_is_every_session_that_called_anything():
    found = reachability.audit(surveyed(OPEN), {"get_order": 4, "gone": 9}, floor=100)

    assert found.sessions == 13
    assert found.floor == 100
    assert not found.settled


def test_the_loudest_demonstration_comes_first_because_it_is_worth_arguing_about():
    tools = [declare(f"t{i}", required={f"a{i}": STRING}) for i in range(3)]
    found = reachability.audit(reachability.survey(tools), {"t0": 5, "t1": 50, "t2": 50})

    assert [w.tool for w in found.settled] == ["t1", "t2", "t0"]


def test_a_second_hop_argument_the_schema_proved_is_still_contradicted_by_a_first_call():
    """The held standing came from a response schema rather than from a person, and a
    demonstration outranks it either way: something reached the tool without the first hop."""
    finder = declare("search", returns=["sku"])
    user = declare("get_product", required={"sku": STRING})
    found = reachability.audit(surveyed(finder, user), {"get_product": 6})

    assert [w.tool for w in found.contradicted] == ["get_product"]
    assert found.contradicted[0].arguments == ("sku",)


def test_the_traffic_is_counted_per_argument_and_never_turned_into_a_verdict():
    """A count is what the evidence is. `supplied` would be a decision, and it is not ours:
    the model normalises what it was told and invents what it was not, and from here the two
    are the same row (DESIGN.md 19.56)."""
    result = surveyed(OPEN, BLOCKED, FREE, marks={"subscription_id": "held"})
    found = reachability.audit(result, {"get_order": 3, "get_subscription": 5})

    assert reachability.demonstrated(found) == {"order_id": 3, "subscription_id": 5}


def test_an_argument_several_tools_take_adds_its_demonstrations_up():
    tools = [declare("a", required={"sku": STRING}), declare("b", required={"sku": STRING})]
    found = reachability.audit(reachability.survey(tools), {"a": 4, "b": 6})

    assert reachability.demonstrated(found) == {"sku": 10}


def test_demonstrating_nothing_leaves_the_marking_file_alone():
    found = reachability.audit(surveyed(OPEN, FREE), {"list_orders": 9})
    assert reachability.demonstrated(found) == {}


# --- The marking file ---------------------------------------------------------------


def test_the_template_round_trips_through_the_parser_once_it_is_filled_in():
    tools = [declare("a", required={"sku": STRING}), declare("b", required={"zip": STRING})]
    result = reachability.survey(tools)
    filled = reachability.template(result).replace("sku ", "sku held ").replace("zip ", "zip x ")
    filled = filled.replace("zip x ", "zip supplied ")
    marks = reachability.read_marks(filled)
    assert marks == {"sku": "held", "zip": "supplied"}


def test_an_unfilled_template_parses_to_nothing_rather_than_failing():
    """Stopping early is allowed and the docstring says so, so it has to actually work."""
    result = reachability.survey([declare("a", required={"sku": STRING})])
    assert reachability.read_marks(reachability.template(result)) == {}


def test_comments_and_blank_lines_are_skipped_and_trailing_comments_ignored():
    text = "# a note\n\nsku held   # gates 19 tools\n  \norder_id supplied\n"
    assert reachability.read_marks(text) == {"sku": "held", "order_id": "supplied"}


@pytest.mark.parametrize("line", ["sku maybe", "sku held supplied", "sku Held"])
def test_a_line_that_does_not_parse_is_an_error_and_names_its_number(line):
    with pytest.raises(ValueError, match="line 2"):
        reachability.read_marks(f"# header\n{line}\n")


def test_a_bare_name_is_an_undecided_line_and_not_a_typo():
    """Every untouched line of the template looks like this, and stopping early is supported."""
    assert reachability.read_marks("sku\norder_id supplied\n") == {"order_id": "supplied"}


def test_the_traffic_annotates_the_template_and_answers_none_of_it():
    """The loop that makes this worth having: survey, ask the traffic, hand back the evidence.

    Every line still parses to nothing, because a file that came back pre-answered is a file
    nobody rereads, and this is the file every reach number in the project rests on.
    """
    result = surveyed(OPEN, declare("get_invoice", required={"invoice_no": STRING}))
    found = reachability.audit(result, {"get_order": 30})
    drafted = reachability.template(result, reachability.demonstrated(found))

    assert reachability.read_marks(drafted) == {}
    assert "order_id" in drafted and "opened 30" in drafted
    assert "invoice_no" in drafted and "opened" not in drafted.split("invoice_no")[1]


def test_a_template_with_no_traffic_is_the_same_file_as_before():
    result = surveyed(OPEN)
    assert reachability.template(result, {}) == reachability.template(result)


def test_a_finished_marking_file_still_comes_back_with_the_disagreement_on_it():
    """The case that makes the write-back worth running twice.

    Once every argument is marked there is nothing left to fill in, so without this the file
    would be a header and no body, and the one line worth revisiting would be the one line
    missing from it.
    """
    result = surveyed(BLOCKED, marks={"subscription_id": "held"})
    found = reachability.audit(result, {"get_subscription": 12})
    drafted = reachability.template(result, reachability.demonstrated(found))

    assert "subscription_id" in drafted
    assert "marked held, opened 12" in drafted


def test_the_disagreement_is_written_as_a_comment_so_it_decides_nothing():
    """It is the adopter's own file that holds the marking. This one only points at it."""
    result = surveyed(BLOCKED, marks={"subscription_id": "held"})
    found = reachability.audit(result, {"get_subscription": 12})

    drafted = reachability.template(result, reachability.demonstrated(found))
    assert reachability.read_marks(drafted) == {}


def test_an_argument_still_open_is_a_question_and_not_a_disagreement():
    """It belongs in the body with `opened N` beside it, and must not be listed twice."""
    result = surveyed(OPEN)
    found = reachability.audit(result, {"get_order": 9})
    drafted = reachability.template(result, reachability.demonstrated(found))

    assert "opened 9" in drafted
    assert "marked" not in drafted.split("order_id")[-1]


# --- The refusal ---------------------------------------------------------------------


def test_nothing_is_inferred_from_what_an_argument_is_called():
    """The one rule that must never be added, pinned so it cannot arrive by accident.

    `order_id` and `subscription_id` are the same type, the same shape and the same suffix,
    and on the reference agent a customer holds the first and not the second. Any name rule
    that could tell them apart would be a coincidence, and one that treats them alike is
    wrong about one of them. So the module treats both as undecided and asks.
    """
    tools = [
        declare("a", required={"order_id": STRING}),
        declare("b", required={"subscription_id": STRING}),
        declare("c", required={"internal_uuid": STRING}),
        declare("d", required={"postcode": STRING}),
    ]
    result = reachability.survey(tools)
    assert result.span == (0, 4), "every one of these is a judgement, not a schema fact"
    assert all(gate.standing is None for tool in result.tools for gate in tool.gates)
