"""The reference agent packaged as a deployable ADK agent folder (DESIGN.md 3.2, 11.1).

`adk deploy agent_engine` and `adk deploy cloud_run` both take a folder, import the module
inside it, and look for `app`. That is the only reason this folder is separate from
`reference_agent`: the agent is the same 100-tool support agent, but the wiring is the
wiring a deployment needs and the folder name is what the deployed resource is called.

Nothing in `src/` imports this. See `agent.py` for the wiring and `README.md` for how to
put it in a cloud.
"""

from __future__ import annotations

from . import agent

__all__ = ["agent"]
