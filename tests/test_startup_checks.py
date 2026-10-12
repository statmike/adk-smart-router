"""Tests for the boot-time diagnostics that describe `prunable_by_default`.

Both checks exist because the failure they describe is silent. A tool named in both lists
still ships, so the config says one thing and the router does another and nothing breaks.
A tool named while the holdout is tiny is worse: nothing at all goes wrong until a user
asks for a tool that is no longer declared, and no counter anywhere records that.

The plugin-order, context-cache and before_tool_callback checks live in
tests/test_cache_safety.py, next to the control they protect.
"""

from __future__ import annotations

import logging

import pytest
from test_plugin import FakeAgent, FakeContext, FakeInvocationContext

from adk_smart_router.config import RouterConfig
from adk_smart_router.startup_checks import MIN_HOLDOUT_FOR_PRUNABLE, StartupChecks


def run_checks(cfg: RouterConfig) -> StartupChecks:
    """Boot one plugin's worth of checks against a healthy App, so only cfg can warn."""
    checks = StartupChecks(owner=object(), cfg=cfg)
    ictx = FakeInvocationContext(FakeAgent("support"), [])
    checks.run(FakeContext(invocation_context=ictx))  # type: ignore[arg-type]
    return checks


def messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# --- The two lists contradicting each other ---------------------------------------


def test_a_tool_named_in_both_lists_is_reported_rather_than_resolved_silently(
    caplog: pytest.LogCaptureFixture,
):
    cfg = RouterConfig(
        holdout_fraction=0.5,
        never_prune=frozenset({"audit_log"}),
        prunable_by_default=frozenset({"audit_log", "book_callback"}),
    )
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    out = messages(caplog)
    assert "audit_log" in out
    assert "keep shipping" in out
    # The other name is not a contradiction, so it must not be swept into the same line.
    assert "book_callback" not in out


def test_declaring_the_framework_tool_prunable_counts_as_a_contradiction(
    caplog: pytest.LogCaptureFixture,
):
    """`transfer_to_agent` is protected without anybody naming it, so the operator gets no
    other signal that they asked for something the router will refuse."""
    cfg = RouterConfig(holdout_fraction=0.5, prunable_by_default=frozenset({"transfer_to_agent"}))
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    assert "transfer_to_agent" in messages(caplog)


def test_two_lists_that_do_not_overlap_produce_no_contradiction_line(
    caplog: pytest.LogCaptureFixture,
):
    cfg = RouterConfig(
        holdout_fraction=0.5,
        never_prune=frozenset({"audit_log"}),
        prunable_by_default=frozenset({"book_callback"}),
    )
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    assert "keep shipping" not in messages(caplog)


# --- A holdout too small to notice a wrong name -----------------------------------


def test_naming_a_tool_prunable_at_the_shipped_holdout_warns(caplog: pytest.LogCaptureFixture):
    """0.05 is the shipped default and is below the bar, so the common mistake warns."""
    assert RouterConfig().holdout_fraction < MIN_HOLDOUT_FOR_PRUNABLE
    cfg = RouterConfig(prunable_by_default=frozenset({"book_callback"}))
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    assert "holdout_fraction is 0.05" in messages(caplog)


def test_a_holdout_large_enough_to_find_a_wrong_name_is_not_warned_about(
    caplog: pytest.LogCaptureFixture,
):
    cfg = RouterConfig(
        holdout_fraction=MIN_HOLDOUT_FOR_PRUNABLE,
        prunable_by_default=frozenset({"book_callback"}),
    )
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    assert "holdout_fraction" not in messages(caplog)


def test_the_shipped_empty_default_says_nothing_at_all(caplog: pytest.LogCaptureFixture):
    """Neither warning may fire on a config nobody has touched, or both become noise."""
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(RouterConfig())
    assert messages(caplog) == ""


# --- The classless inventory ------------------------------------------------------


def test_the_classless_tools_are_named_with_what_they_cost(caplog: pytest.LogCaptureFixture):
    """An operator cannot set the knob without the names, and nothing else prints them."""
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    with caplog.at_level(logging.INFO, logger="adk_smart_router.startup_checks"):
        checks.classless_tools(
            "support",
            frozenset({"a", "b", "dead_one", "dead_two"}),
            frozenset({"a", "b"}),
            lambda tools: 250 * len(tools),
        )
    out = messages(caplog)
    assert "2 of 4 tools" in out
    assert "500 tokens" in out
    assert "dead_one, dead_two" in out


def test_a_tool_already_declared_prunable_is_not_counted_as_still_costing(
    caplog: pytest.LogCaptureFixture,
):
    cfg = RouterConfig(prunable_by_default=frozenset({"dead_one"}))
    checks = StartupChecks(owner=object(), cfg=cfg)
    with caplog.at_level(logging.INFO, logger="adk_smart_router.startup_checks"):
        checks.classless_tools(
            "support",
            frozenset({"a", "b", "dead_one", "dead_two"}),
            frozenset({"a", "b"}),
            lambda tools: 250 * len(tools),
        )
    out = messages(caplog)
    assert "1 of 4 tools" in out
    assert "dead_one" not in out


def test_an_agent_whose_tools_all_have_classes_produces_no_line(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    with caplog.at_level(logging.INFO, logger="adk_smart_router.startup_checks"):
        checks.classless_tools("support", frozenset({"a"}), frozenset({"a"}), lambda _: 0)
    assert messages(caplog) == ""


def test_the_inventory_is_reported_once_per_agent_and_not_once_per_turn(
    caplog: pytest.LogCaptureFixture,
):
    """It runs on the hot path, so a repeat costs a walk of the request as well as a line."""
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    measured = 0

    def cost(tools: frozenset[str]) -> int:
        nonlocal measured
        measured += 1
        return 100

    with caplog.at_level(logging.INFO, logger="adk_smart_router.startup_checks"):
        for _ in range(3):
            checks.classless_tools("support", frozenset({"a", "dead"}), frozenset({"a"}), cost)
        checks.classless_tools("billing", frozenset({"a", "dead"}), frozenset({"a"}), cost)

    assert measured == 2  # one per agent, not one per call
    assert len(caplog.records) == 2


# --- One raising check must not take the other two with it --------------------------


def test_a_check_that_raises_does_not_suppress_the_ones_after_it(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """Three independent statements about three independent misconfigurations.

    They used to run as one statement under one `try`, after `_done` was already set, so
    the first one to raise silenced the other two for the life of the process. The one
    most likely to raise is the one that walks ADK's plugin list, which is exactly the
    structure most likely to move under a version bump.
    """

    def boom(self: StartupChecks, ictx: object) -> None:
        raise RuntimeError("ADK moved the plugin list")

    monkeypatch.setattr(StartupChecks, "_check_plugin_order", boom)
    cfg = RouterConfig(holdout_fraction=0.5, prunable_by_default=frozenset({"transfer_to_agent"}))

    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        checks = run_checks(cfg)

    out = messages(caplog)
    assert "a startup check failed" in out, "the failure is still reported"
    assert "transfer_to_agent" in out, "and the checks behind it still ran"
    assert checks is not None


def test_the_agent_scan_still_runs_when_a_first_turn_check_raises(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """`_scan_agents` is not a diagnostic. It finds the agents the cache must not serve for."""

    def boom(self: StartupChecks, ictx: object) -> None:
        raise RuntimeError("ADK moved the plugin list")

    monkeypatch.setattr(StartupChecks, "_check_plugin_order", boom)
    checks = StartupChecks(owner=object(), cfg=RouterConfig())
    # An agent that gates its own tools, which is the thing the cache must not serve for.
    ictx = FakeInvocationContext(FakeAgent("support", before_tool_callback=lambda **_: None), [])

    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        checks.run(FakeContext(invocation_context=ictx))  # type: ignore[arg-type]

    assert checks.gates_own_tools("support") is True


# --- Proactive judge dependency and bucket wiring checks --------------------------


def test_judge_wiring_issues_detects_missing_xgboost_and_gcs_packages() -> None:
    from adk_smart_router.startup_checks import judge_wiring_issues

    cfg = RouterConfig(judges=("xgboost", "prototype"), gcs_bucket="my-bucket")
    issues = judge_wiring_issues(cfg, find_spec=lambda _: None)
    assert len(issues) == 1
    assert "xgboost, google-cloud-storage (not installed)" in issues[0]

    # Zero-data prototype mode requires neither xgboost nor google-cloud-storage.
    proto_cfg = RouterConfig(mode="active", judges=("prototype",), gcs_bucket="")
    assert judge_wiring_issues(proto_cfg, find_spec=lambda _: None) == []


def test_active_xgboost_only_without_gcs_bucket_warns_at_startup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = RouterConfig(mode="active", judges=("xgboost",), gcs_bucket="")
    with caplog.at_level(logging.WARNING, logger="adk_smart_router.startup_checks"):
        run_checks(cfg)
    out = messages(caplog)
    assert "no gcs_bucket configured" in out
    assert "include 'prototype' in SMART_ROUTER_JUDGES" in out
