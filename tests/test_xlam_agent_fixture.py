"""One guard on the xLAM fixture: it has to stay out of the trap band.

`smart-router qualify` is the thing that actually decides this, and it decides it properly,
by importing the agent and counting the declarations. It is not run here because that pulls
in ADK and a model name for a question that has one moving part.

The moving part is the instruction. Everything else in the pruned prompt comes from xLAM's
schemas and is regenerated rather than edited. If somebody tightens the instruction, the
pruned prompt drops toward the 2,560 tokens a prefix needs before Gemini keeps it, and this
fixture stops measuring prediction and starts measuring a cache cliff. That failure is
silent: the A/B still runs, still reports, and reports a loss.
"""

from __future__ import annotations

from xlam_agent.instruction import INSTRUCTION

# What the instruction was at when `qualify` read PAYS with 605 tokens of headroom, on
# 2026-09-06: a pruned prompt of 3,165 tokens against the reference agent's 3,127. Losing a
# fifth of it would spend that headroom, so the floor sits there rather than at the exact
# measured length, which would fail on any honest edit.
MIN_CHARS = 8_200


def test_the_instruction_is_long_enough_that_a_prune_still_caches():
    assert len(INSTRUCTION) >= MIN_CHARS


def test_the_instruction_names_no_tool_on_the_surface():
    """The training labels are the model's own tool choices, so a hint here would be a hint
    in the labels, and the fixture would be grading a prompt this project wrote."""
    import pytest

    tools_mod = pytest.importorskip("xlam_agent.tools")

    named = [tool.__name__ for tool in tools_mod.ALL_TOOLS if tool.__name__ in INSTRUCTION]

    assert named == []
