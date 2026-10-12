# Reference agent

A single-root [ADK](https://adk.dev/) support agent for a fictional online retailer, over 100
deterministic offline stub tools. It has three jobs:

1. Generate the telemetry corpus the preflight queries P1-P4 read.
2. Be the integration test target: the only place the full plugin loop runs against a real
   Runner, a real Gemini, and a real [BigQuery Agent Analytics
   plugin](https://adk.dev/integrations/bigquery-agent-analytics/), BQAA below.
3. Be the worked example in the project docs.

The tools are stubs with no external calls, so a corpus run is reproducible and free apart
from the Gemini turns themselves. The toolset is chosen to exercise specific design features,
not for realism: zero-arg tools like `get_loyalty_points` test ambient identity in the cache
key, `get_order_status` is the poller trap that must get `ttl=0`, and `escalate_to_human` must
be blocklisted from caching automatically.

## Layout

| Path | What |
| --- | --- |
| `agent.py` | The root `LlmAgent`, named `support`, on `gemini-2.5-flash`. |
| `tools/` | The 100 stubs, one module per domain, each exporting `TOOLS`. |
| `scenarios/` | Personas and scenarios for the corpus generator, committed so bias is reviewable in the diff. |
| `corpus/` | The generated `corpus.jsonl.gz` plus `corpus.meta.json`. Gzipped and without `LLM_REQUEST` rows, or it would not be committable. |
| `run_agent.py` | Local runner: builds the `App`, wires BQAA, drives one turn or a REPL. |

## Running it

Authentication is [Application Default
Credentials](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc). Run
`gcloud auth application-default login` once, then from the repo root:

```bash
uv run python -m examples.reference_agent.run_agent --prompt "where is order A-1042?"
uv run python -m examples.reference_agent.run_agent --interactive
uv run python -m examples.reference_agent.run_agent --prompt "hi" --no-analytics
```

`--interactive` keeps one session across turns, which is what you want when looking at
caching: the explicit context cache starts on turn 2 at the earliest.

Every default is an environment variable. The project has none: the runner asks for it by
name rather than falling back to whichever project the last person used.

| Variable | Default |
| --- | --- |
| `SMART_ROUTER_BQ_PROJECT` | none, required |
| `SMART_ROUTER_BQ_DATASET` | `smart_router` |
| `SMART_ROUTER_BQ_LOCATION` | `US` |
| `REFERENCE_AGENT_MODEL` | `gemini-2.5-flash` |
| `REFERENCE_AGENT_USER_ID` | `local-dev` |

Without `--no-analytics`, every turn writes rows to `smart_router.agent_events`. Use
`--no-analytics` when you are debugging the agent itself and do not want the corpus polluted
with your test prompts.

## The one thing not to do

**Do not install `SmartRouterPlugin` in `run_agent.py` until the corpus exists.**

The router prunes the tool menu Gemini chooses from. Run it during corpus generation and the
recorded labels are the router's guesses rather than Gemini's independent choices, so the
model trains on its own output and every preflight number becomes a measurement of the router
agreeing with itself. `run_agent.py` marks where the plugin goes, commented out, along with
the ordering rule: the router registers *before* the BQAA plugin, never after.
