"""Reference agent for `adk-smart-router` (DESIGN.md 3.2).

Three jobs: generate the telemetry corpus the preflight queries read, be the integration
test target for the full plugin loop, and be the worked example in the docs.

Re-exporting `agent` here is the layout `adk run` / `adk web` expect, so the agent is
usable from the ADK CLI as well as from `run_agent.py`.
"""

from __future__ import annotations

from . import agent

__all__ = ["agent"]
