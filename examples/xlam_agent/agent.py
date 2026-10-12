"""The xLAM agent: one generic API assistant over 100 tools nobody here designed.

ROADMAP's E1. This is the reference agent with the data swapped out and nothing else
changed: same model, same surface width, same router settings, same everything the plugin
sees. The tools come from `xlam-function-calling-60k` and so do the user messages, so a
number measured here is a number measured on somebody else's intent distribution.

`tools.py` is generated. Regenerate it with `scripts/build_xlam_agent.py`, which also
writes the `queries.jsonl` that drives the traffic.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from .instruction import INSTRUCTION

try:
    from .tools import ALL_TOOLS
except ImportError:
    ALL_TOOLS = []

MODEL = os.environ.get("XLAM_AGENT_MODEL", "gemini-2.5-flash")

# Fixed for the same reason the reference agent's is: the name is the agent identity in
# every telemetry row, and a per-developer name would split the corpus in two.
AGENT_NAME = "xlam"

root_agent = LlmAgent(
    name=AGENT_NAME,
    model=MODEL,
    description="A general API assistant over a hundred third-party endpoints: lookups, "
    "conversions, calculations, and account actions across unrelated domains.",
    instruction=INSTRUCTION,
    tools=ALL_TOOLS,
)
