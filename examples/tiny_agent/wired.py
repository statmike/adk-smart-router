"""The same agent with the router wired the way `README.md` says to wire it.

`agent.py` is the before and this is the after, so `smart-router doctor` has one target that
must fail all three checks and one that must pass all three. Both are asserted by
`scripts/check_install.sh`.

The plugin order below is the check people get wrong, and it is the reason `doctor` exists.
The router has to run before `BigQueryAgentAnalyticsPlugin`, or analytics logs the router's
own cache hits as real tool calls and the next model trains on its own echo.
"""

from agent import root_agent
from google.adk.apps import App
from google.adk.apps.app import ContextCacheConfig
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin

from adk_smart_router import SmartRouterPlugin

app = App(
    name="tiny",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(min_tokens=2048),
    plugins=[
        SmartRouterPlugin(),
        BigQueryAgentAnalyticsPlugin(
            project_id="example-project",
            dataset_id="smart_router",
            custom_metadata_allowlist=["smart_router"],
        ),
    ],
)
