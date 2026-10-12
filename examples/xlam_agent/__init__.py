"""The external-validity fixture for `adk-smart-router` (ROADMAP E1).

An agent whose tool surface and whose traffic both come from
`xlam-function-calling-60k` (Salesforce, CC-BY-4.0, arXiv:2406.18518), so that a live
measurement can be taken on data this project did not write. The reference agent answers
"does it work"; this one answers "does it work on somebody else's data".

Re-exporting `agent` here is the layout `adk run` and `adk web` expect.
"""

from __future__ import annotations

from . import agent

__all__ = ["agent"]
