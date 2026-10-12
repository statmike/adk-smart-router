"""Tests for the one kind of artifact mismatch that is allowed to load anyway.

`model_store._check_compatible` refuses an artifact whose environment, embedding model,
embedding width or backend disagrees with the config, and `max_model_age_days` demotes one
that is simply old. All of those make the model wrong to use. The two settings that decide the
served set size are different: serving a size other than the one the gate measured leaves the
model correct and the published numbers stale, so refusing would be an overreaction and
silence would be a lie.

Those two are `max_tools_kept`, which caps the set, and `keep_probability_mass`, which without
menus usually stops it before the cap. Either can drift on its own, so each is checked on its
own (DESIGN.md 19.72, 19.74).

The last test states that split in one place, because the interesting thing about this check
is not what it does but why it is the only one of its kind.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from adk_smart_router.config import RouterConfig
from adk_smart_router.model_store import ArtifactMetadata, _Refused
from adk_smart_router.startup_checks import StartupChecks

TOOLS = ("lookup_order", "issue_refund", "check_stock", "escalate")


def make_meta(
    gate_metrics: dict[str, float] | None = None,
    menus: tuple[frozenset[str], ...] = (),
) -> ArtifactMetadata:
    return ArtifactMetadata(
        version="20260907T000000Z",
        sha256="a" * 64,
        size_bytes=10,
        trained_at="2026-09-07",
        rows=100,
        classes=TOOLS,
        backend="vertex",
        environment="prod",
        embedding_model="gemini-embedding-2",
        embedding_dims=768,
        gate_metrics={"recall": 0.99, "top_n": 11} if gate_metrics is None else gate_metrics,
        menus=menus,
    )


def messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# --- The number matching, and not matching ----------------------------------------


def test_an_artifact_gated_at_the_size_being_served_says_nothing(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=11))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta())
    assert messages(caplog) == ""


def test_a_different_size_is_reported_with_both_numbers_in_the_line(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=6))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta())

    text = messages(caplog)
    # Both numbers, because "your config disagrees with the artifact" is useless without them.
    assert "max_tools_kept=6" in text
    assert "top_n=11" in text
    # And the reason it is a warning and not a refusal.
    assert "still predicts correctly" in text


def test_the_size_read_back_as_a_float_still_compares_as_a_number(
    caplog: pytest.LogCaptureFixture,
):
    """`gate_metrics` is a map of floats, so a `top_n` of 11 arrives as 11.0 after JSON.

    Comparing it to an int without saying so would warn on every correctly configured
    router, which is the fastest way to teach an operator to ignore the log.
    """
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=11))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics={"top_n": 11.0}))
    assert messages(caplog) == ""


def test_the_warning_names_the_file_to_edit_rather_than_only_the_problem(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=6))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta())

    text = messages(caplog)
    assert "sql/train_router.sql" in text
    assert "docs/training.md" in text


def test_the_skew_is_reported_once_per_agent_and_separately_per_agent(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=6))
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            checks.artifact_skew("support", make_meta())
            checks.artifact_skew("billing", make_meta())

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2  # once each, not once per turn
    assert sum("agent support" in line for line in lines) == 1
    assert sum("agent billing" in line for line in lines) == 1


# --- Artifacts that predate the check ---------------------------------------------


def test_an_artifact_published_before_top_n_existed_is_not_second_guessed(
    caplog: pytest.LogCaptureFixture,
):
    """Absent `top_n` means an older publisher, not a mismatch. Silence is the honest answer."""
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=6))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics={"recall": 0.99}))
    assert messages(caplog) == ""


def test_a_metric_the_router_has_never_heard_of_changes_nothing(
    caplog: pytest.LogCaptureFixture,
):
    """A newer publisher adding fields must not make an older router noisy or refuse."""
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=11))
    metrics = {"recall": 0.99, "top_n": 11, "some_metric_from_2027": 0.5}
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics=metrics))
    assert messages(caplog) == ""


# --- Menus, where the setting is not read at all ----------------------------------


def test_a_menu_artifact_is_told_that_the_setting_does_nothing_at_all(
    caplog: pytest.LogCaptureFixture,
):
    """`pruner.decide` takes a menu whole and never reads `max_tools_kept`.

    Someone who lowers it to shorten their declarations gets no shorter declarations and no
    error, which is the worst combination available. The moment they have clearly tried is
    the moment to say so.
    """
    menus = (frozenset(TOOLS[:2]), frozenset(TOOLS[:3]))
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=6))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(menus=menus))

    text = messages(caplog)
    assert "2 menus" in text
    assert "of 2, 3 tools" in text  # the sizes actually served, deduplicated and sorted
    assert "has no effect" in text


def test_a_menu_artifact_that_agrees_on_the_size_stays_quiet_anyway(
    caplog: pytest.LogCaptureFixture,
):
    """The menu line is a footnote on the skew line, not a standing complaint about menus.

    An operator whose config matches the gate has nothing to fix, and telling them their
    unused setting is unused would be noise on every boot of a correctly configured router.
    """
    menus = (frozenset(TOOLS[:2]),)
    checks = StartupChecks(owner=object(), cfg=RouterConfig(max_tools_kept=11))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(menus=menus))
    assert messages(caplog) == ""


# --- The other half of the set size ------------------------------------------------


def test_an_artifact_measured_at_the_mass_being_served_says_nothing(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(keep_probability_mass=0.90))
    metrics = {"top_n": 11, "keep_probability_mass": 0.90}
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics=metrics))
    assert messages(caplog) == ""


def test_a_different_mass_is_reported_with_both_numbers_and_its_own_section(
    caplog: pytest.LogCaptureFixture,
):
    """The cap can match while the mass does not, and that is the common case.

    Without menus the mass rule is usually what binds, so an operator who matched
    `max_tools_kept` to `top_n` and stopped there is serving a smaller set than the one the
    published coverage describes, with nothing on screen saying so.
    """
    cfg = RouterConfig(max_tools_kept=11, keep_probability_mass=0.75)
    metrics = {"top_n": 11, "keep_probability_mass": 0.90}
    checks = StartupChecks(owner=object(), cfg=cfg)
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics=metrics))

    text = messages(caplog)
    assert "keep_probability_mass=0.75" in text
    assert "measured at 0.9" in text
    assert "docs/training.md#criterion-1-when-there-are-no-menus" in text
    # The cap agreed, so its line must not appear. Two warnings for one mistake trains
    # an operator to skim.
    assert "max_tools_kept=11" not in text


def test_floating_point_noise_in_the_published_mass_is_not_a_mismatch(
    caplog: pytest.LogCaptureFixture,
):
    """The number makes a round trip through JSON and BigQuery before it gets here.

    An exact `!=` on a float would warn on a correctly configured router often enough to
    make the whole check ignorable.
    """
    checks = StartupChecks(owner=object(), cfg=RouterConfig(keep_probability_mass=0.90))
    metrics = {"keep_probability_mass": 0.90 + 1e-12}
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics=metrics))
    assert messages(caplog) == ""


def test_an_artifact_published_before_the_mass_was_recorded_is_not_second_guessed(
    caplog: pytest.LogCaptureFixture,
):
    checks = StartupChecks(owner=object(), cfg=RouterConfig(keep_probability_mass=0.50))
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics={"top_n": 11}))
    assert messages(caplog) == ""


def test_both_settings_wrong_are_two_lines_and_one_footnote(
    caplog: pytest.LogCaptureFixture,
):
    """Two independent mistakes, said independently, and the menu footnote said once.

    The footnote is about the path, not about either setting, so repeating it under each
    would be the same sentence twice.
    """
    cfg = RouterConfig(max_tools_kept=6, keep_probability_mass=0.75)
    metrics = {"top_n": 11, "keep_probability_mass": 0.90}
    checks = StartupChecks(owner=object(), cfg=cfg)
    with caplog.at_level(logging.WARNING):
        checks.artifact_skew("support", make_meta(gate_metrics=metrics, menus=(frozenset(TOOLS),)))

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 3
    assert sum("Neither max_tools_kept nor keep_probability_mass is read" in x for x in lines) == 1


# --- The split this check sits on the far side of ---------------------------------


def test_the_skews_that_refuse_still_refuse_and_this_one_still_does_not():
    """One statement of which mismatches stop a load and which only annotate it.

    Written as a table rather than four separate tests because the point is the boundary,
    not any single field. A field moving from one column to the other is a design change and
    should fail here, loudly, next to the reasoning.
    """
    from adk_smart_router.model_store import ModelStore

    cfg = RouterConfig(
        environment="prod",
        embedding_model="gemini-embedding-2",
        embedding_dims=768,
        max_tools_kept=6,  # disagrees with the artifact's top_n of 11, deliberately
    )
    store = ModelStore(cfg)

    refuse = {
        "environment": "staging",
        "embedding_model": "text-embedding-004",
        "embedding_dims": 1536,
        "backend": "local",
    }
    for field, wrong in refuse.items():
        with pytest.raises(_Refused, match=field.replace("_", " ")):
            store._check_compatible(replace(make_meta(), **{field: wrong}))  # noqa: SLF001

    # And the one that does not: same store, same config, an artifact whose only disagreement
    # is the set size. It loads, and `artifact_skew` is the entire consequence.
    store._check_compatible(make_meta())  # noqa: SLF001
