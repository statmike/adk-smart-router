"""The smallest ADK agent there is, written the way a first-time user would write one.

This exists to be the target of `scripts/check_install.sh`, which installs the built wheel
into an empty environment and runs `smart-router qualify` and `smart-router doctor` against
this file. Nothing here imports the router, and nothing here knows it is being measured.

Two tools and a one-line instruction is on purpose. It is far below the prompt size where
pruning pays, so `qualify` has to say so, and that is the answer most people asking the
question for the first time should get.
"""

from google.adk.agents import LlmAgent
from google.adk.apps import App


def get_weather(city: str) -> dict:
    """Look up the current weather in a city."""
    return {"city": city, "sky": "clear"}


def get_time(city: str) -> dict:
    """Look up the current local time in a city."""
    return {"city": city, "time": "12:00"}


root_agent = LlmAgent(
    name="tiny",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Answer the user's question.",
    tools=[get_weather, get_time],
)

app = App(name="tiny", root_agent=root_agent)
