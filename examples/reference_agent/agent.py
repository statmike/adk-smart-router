"""The reference agent: one e-commerce support agent over 25 deterministic stub tools.

This is the corpus source (DESIGN.md 3.2). Its tool choices become the training labels,
so the only thing that must be right here is that the model picks tools the way a real
support agent would: from the user's words, with no hints from us about which tool exists.

Single root agent, deliberately. Multi-agent transfer is a separate fixture (DESIGN.md 7.6);
mixing it in would confound Preflight P1's distinct-tools-per-session measurement, because
a transfer splits one user intent across two tool menus.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from .heavy_tools import heavy
from .instruction import INSTRUCTION
from .tools import ALL_TOOLS

# 25 tools sits just above `min_tools_to_route = 20` and inside the Flash qualification
# floor's grey zone (DESIGN.md 2.3), so the floor gets exercised rather than assumed.
#
# The instruction is long on purpose and lives in its own module; see `instruction.py` for
# why, and DESIGN.md 18.15 for the measurement that forced it. In short, a short system
# prompt puts every pruned request under the context-cache floor, so the router correctly
# declines to prune and there is nothing to measure.
MODEL = os.environ.get("REFERENCE_AGENT_MODEL", "gemini-2.5-flash")

# Fixed, not env-overridable: the name lands in every BQAA row as the agent identity, and
# a per-developer name would fragment the corpus the preflight queries read.
AGENT_NAME = "support"

# E2 in ROADMAP.md. Off by default, and when it is on the declarations are unchanged, so
# the agent the model sees is the same agent and the published router artifact still
# applies. Only the tool results grow and slow down. See `heavy_tools.py`.
HEAVY = os.environ.get("REFERENCE_AGENT_HEAVY_TOOLS") == "1"

TOOLS = heavy(ALL_TOOLS) if HEAVY else ALL_TOOLS


root_agent = LlmAgent(
    name=AGENT_NAME,
    model=MODEL,
    description="Customer support for an online retailer: orders, returns, shipping, "
    "payments, catalog, account, and help requests.",
    instruction=INSTRUCTION,
    tools=TOOLS,
)
