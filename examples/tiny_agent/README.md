# tiny_agent: the target of the install check

Two files, four tools' worth of nothing, and no reason to exist except that
`scripts/check_install.sh` needs something to point at.

`agent.py` is an [ADK](https://adk.dev/) agent with two tools and a one-line
instruction, written the way somebody writes their first one. It does not import
the router. `wired.py` takes that same agent and installs the router on it the
way `README.md` says to.

The install check builds the wheel, installs it into an empty environment, and
runs both commands against both files:

    ./scripts/check_install.sh

`smart-router qualify --app agent:app` has to answer, and has to say that pruning
does not pay for a two-tool agent, because it does not. `smart-router doctor` has
to fail all three of its checks on `agent.py` and pass all three on `wired.py`.

The plugin order in `wired.py` is the mistake `doctor` exists to catch, and it is
easy to make: the router has to come before `BigQueryAgentAnalyticsPlugin`, or
analytics records the router's own cache hits as real tool calls and the next
model trains on its own echo.

For an agent worth actually routing, see `examples/reference_agent/`.
