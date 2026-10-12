# deployed_support

The reference agent, packaged as a deployable ADK agent folder. It is the same 100-tool
support agent as `../reference_agent`, wired the way a deployment has to be wired, and it
exists so that the managed-runtime claims in [docs/deploying.md](../../docs/deploying.md)
come from a deployment that ran rather than from reading ADK's source.

## What is here

| File | What it is |
| --- | --- |
| `agent.py` | The `App`. Router first, analytics second, context caching on. |
| `requirements.txt` | What the container installs. Kept in step with `pyproject.toml` by a test. |
| `__init__.py` | Re-exports `agent`, which is the layout the ADK CLI expects. |

There is no `.env` here and there never should be, because it would name a project. The
deploy script writes one into a staging copy under `/tmp` and throws it away afterwards.

## Deploying it

You need a provisioned project first: an analytics dataset, a bucket, and a published
artifact for the agent named `support`. [docs/install.md](../../docs/install.md) is the path
to that. Then:

```bash
export SMART_ROUTER_BQ_PROJECT=your-gcp-project
export SMART_ROUTER_GCS_BUCKET=your-bucket

./scripts/deploy_agent_engine.sh                    # print the plan
./scripts/deploy_agent_engine.sh --confirm          # do it
uv run --group deploy python scripts/verify_agent_engine.py
./scripts/deploy_agent_engine.sh --delete --confirm # when you are done
```

The deployment is a billable resource that runs until it is deleted, which is why `--confirm`
is required and `--delete` exists.

## What the deploy stages, and why

ADK's `--extra_packages` puts a local directory at `/app/<basename>` and adds `/app` to
`PYTHONPATH`, so the deploy ships three things: this folder as the agent, a trimmed copy of
`reference_agent` for the tools and the instruction, and `src/adk_smart_router` as source.

The third one lets this example run against your local checkout of `src/adk_smart_router`
without publishing a release first. For a standalone agent deployment, add one line to
`requirements.txt` instead of staging source:

```
adk-smart-router
# or from GitHub:
# adk-smart-router @ git+https://github.com/statmike/adk-smart-router
```

and the five dependencies already listed there arrive as its own.

## Why this is a separate folder from `reference_agent`

`adk deploy` names the deployed resource after the folder, and `reference_agent` is a corpus
generator and an experiment target as well as an agent: 7 MB of synthetic conversations, a
scenario library, and `run_agent.py`, which is a command-line runner with flags for turning
routing off. None of that belongs in a serving image, and a deployment should not have a
switch that quietly disables the thing being tested.

What did not change is the agent itself. `agent.py` imports `root_agent` unmodified, so the
deployed agent is still named `support` and still declares the same 100 tools, which is what
lets it load the artifact that was already trained and published rather than needing a new
one.
