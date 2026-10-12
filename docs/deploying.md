# Deploying an agent that has the router installed

The router is a plugin on your `App`, so it deploys wherever your agent deploys and there is no
separate service to run. What changes between hosting shapes is small and mostly about identity:
the serving path reaches outside the process for exactly two things, and both of them are IAM.

This page covers the three shapes people use and what each needs. The managed-runtime section ends
with a deployment you can run out of this repository, because the questions that section raises are
not ones you can settle by reading code.

## What the serving path needs

Two calls, and nothing else.

| It reads | For what | Permission |
| --- | --- | --- |
| One Cloud Storage object | The published model artifact, on first use and then whenever a new one appears | `roles/storage.objectViewer` on the bucket |
| One embedding call | Turning the session's first user message into a vector | `roles/aiplatform.user` in the project |

Both calls fail open. If the bucket is unset, unreadable, or empty, the plugin serves every request
untouched. If the embedding call fails or times out, that request is untouched. The agent does not
stop, and an identity you forgot to grant shows up as a saving that never arrives rather than as an
outage.

BigQuery is not on that list, and the distinction is worth being exact about. The router reads and
writes no warehouse at request time; the training pipeline and the `readiness` and `status`
commands are the only things that touch it, and they run from a workstation or a scheduler.

**But the deployment as a whole does need BigQuery write access**, because the analytics plugin
sits beside the router on the same `App` and it writes from inside the runtime. That is where the
routing provenance goes, and it is what `status`, the holdout arm and the retrain trigger all read
later. Without it the writes are dropped with a 403 on `TABLES_UPDATE_DATA`, the agent answers
normally, and the warehouse stays empty. So a third row, on the same identity as the two above:

| It writes | For what | Permission |
| --- | --- | --- |
| Rows into the analytics dataset | Every event, carrying the router's provenance | `WRITER` on the dataset, which is `roles/bigquery.dataEditor` |

This one does not belong to the router and it is not optional in practice, because a router you
cannot measure is a router you cannot retrain. [The analytics plugin needs its own
grant](#the-analytics-plugin-needs-its-own-grant) has the commands and the one optional extra.

Configuration is environment variables, read once at startup:

| | |
| --- | --- |
| `SMART_ROUTER_GCS_BUCKET` | Where artifacts are published, as a bare bucket name and not a `gs://` URL. Unset means the plugin passes everything through. |
| `SMART_ROUTER_BQ_PROJECT` | The project the embedding client bills to. |
| `SMART_ROUTER_VERTEX_LOCATION` | Defaults to `global`. |
| `SMART_ROUTER_MODE` | `off` until you turn it on. |
| `SMART_ROUTER_ROUTE_FRACTION` | `0.0` until you turn it on. `mode` alone does not route. |
| `SMART_ROUTER_LOCAL_CACHE_DIR` | Defaults to `/tmp/adk_routers`. |

`smart-router doctor --app myapp.agents:app` checks the wiring before you ship it, and it is worth
running in CI against the same module the deployment loads.

## Sizing

The router adds about 104 MB of resident memory to a process that has already imported ADK, and
about 280 MB to the image. Measured by importing the package and its dependencies in a fresh
interpreter and reading peak RSS, and by reading the installed packages off disk:

| | |
| --- | --- |
| Resident memory added at import | ~104 MB |
| xgboost on disk | ~228 MB |
| numpy on disk | ~35 MB |
| The rest, `google-genai` and `google-cloud-storage` | under 10 MB |
| A published artifact, on disk and in the model | 0.5 to 0.8 MB |

Almost all of it is xgboost. On a 512 MB replica that is a real fraction of the budget and worth
knowing before you find out. On the 1 GB and up sizes people usually run agents at, it is noise.

The artifact itself is small and stays small. It is cached under `local_cache_dir`, which is
`/tmp`, which on a container runtime is memory rather than disk, which is why the model store
checks the object's size at GCS before downloading a byte rather than after. For the same
reason it collects a version directory once a newer one has been in place for an hour. A
replica that lives for weeks would otherwise be holding every artifact published in that time,
and the way that ends is an OOM kill rather than a log line.

## Running locally

Nothing special. Application Default Credentials from `gcloud auth application-default login`
already carry both permissions if your own account has them, and `/tmp` is a real filesystem, so
the artifact cache survives restarts.

```bash
gcloud auth application-default login
export SMART_ROUTER_GCS_BUCKET=my-project-smart-router
export SMART_ROUTER_BQ_PROJECT=my-project
uv run adk web
```

This is the shape everything in this repository was measured on, including the live A/B.

## Cloud Run

`adk deploy cloud_run` builds a container from your agent folder and runs `adk api_server` in it.
The router needs three things from that:

1. **The package in `requirements.txt`.** ADK installs the agent folder's `requirements.txt` inside
   the image. Add `adk-smart-router` (or `adk-smart-router @ git+https://github.com/statmike/adk-smart-router`) to it.
2. **The environment variables**, set on the service.
3. **The service account.** The runtime identity is whatever the service runs as, which is the
   Compute Engine default service account unless you passed `--service-account`. Grant it
   `roles/storage.objectViewer` on the artifact bucket and `roles/aiplatform.user`, plus `WRITER`
   on the analytics dataset so the plugin beside the router can log. The Compute Engine default
   account often already has the third by virtue of being broad; a service account you created for
   this will not.

Each revision starts cold and fetches the artifact once per instance, which is one GCS read of
under a megabyte on the first request that instance serves. Instances do not share the cache, and
they do not need to.

## Agent Platform, the managed runtime

This is the one worth reading carefully, because the shape of the deploy changed and the older
advice about it is wrong.

`adk deploy agent_engine` in ADK 2.8 does not serialise your agent and ship the pickle. It writes a
Dockerfile, copies your agent folder into it, installs your `requirements.txt`, and runs
`adk api_server` exactly as the Cloud Run path does. The consequences are all good ones for a
plugin:

- **A plugin-bearing `App` deploys unchanged.** The agent loader looks for a module attribute named
  `app` that is an `App` instance before it looks for `root_agent`, so exporting
  `app = App(name=..., root_agent=..., plugins=[SmartRouterPlugin(...)])` from your `agent.py` is
  the supported shape and it is the same shape `adk web` loads locally. Nothing about the plugin
  has to be picklable, because nothing gets pickled.
- **Dependencies come from `requirements.txt`**, the same line as Cloud Run.
- **Environment variables come from a `.env` file in the agent folder**, or from `env_vars` in
  `.agent_engine_config.json`. The deploy reads the `.env`, strips `GOOGLE_CLOUD_PROJECT` and
  `GOOGLE_CLOUD_LOCATION` into the deployment's own settings, and passes the rest through to the
  runtime. That is where the `SMART_ROUTER_*` variables go.
- **`/tmp` is writable** and the artifact cache works within one instance, the same as Cloud Run.

The base image is `python:3.11-slim`, which satisfies the package's `requires-python >= 3.11`.

### A deployment you can run yourself

Everything above is read out of ADK 2.8's own deploy code and the agent loader, so it is true
without anyone deploying anything. Three things are not readable that way, and rather than leave
them as caveats the repository ships the deployment that answers them.

`examples/deployed_support` is the reference agent as an agent folder: the same 100 tools, the same
`support` artifact, an `App` that exports `app` with the router registered before the analytics
plugin. `scripts/deploy_agent_engine.sh` puts it on Agent Platform and
`scripts/verify_agent_engine.py` drives it and reads the routing provenance back out of the event
stream.

```bash
export SMART_ROUTER_BQ_PROJECT=your-gcp-project
export SMART_ROUTER_GCS_BUCKET=your-bucket

./scripts/deploy_agent_engine.sh                    # print the plan, create nothing
./scripts/deploy_agent_engine.sh --confirm
uv run --group deploy python scripts/verify_agent_engine.py
./scripts/deploy_agent_engine.sh --delete --confirm
```

The deploy creates a billable resource that runs until it is deleted, which is why `--confirm` is
required and `--delete` is part of the same script.

The three facts, and what the run said about each. Measured 2026-09-08, twelve sessions against a
100-tool agent in `us-central1`.

1. **Which identity the runtime holds.** Agent Platform runs your agent as a service agent rather
   than as a service account you created, and the two permissions above have to land on whichever
   identity that is. Here that was
   `service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com`, and it already held both:
   twelve of twelve sessions came back `fresh` on model version `20260905T171738Z`, which is only
   reachable if the artifact was read from the bucket *and* the embedding call answered. The
   verifier tells the two apart by `decision_source`, because the router fails open on both and a
   missing grant otherwise reads as an agent that simply never routes.
2. **Whether the build can reach a git URL.** Still open, and now deliberately so. The deploy stages
   `src/adk_smart_router` into the image as source through `--extra_packages`, which is what makes
   this work from a private repository, so what the run proved is that the staged package imports
   and runs in the container. The `git+https` install line is the one thing still waiting on the
   repository going public.
3. **How long an instance lives.** Twelve sessions produced twelve records carrying one distinct
   `replica_id`, so a single process served all of them and the artifact download was paid once.
   The cache amortises. This is a small sample and a quiet deployment, so read it as "instances are
   reused" rather than as a number.

None of the three can break an agent. The failure mode for all of them is the router doing nothing,
which is the state it ships in.

### Two settings that are not about where the agent runs

Both of these were wrong on the first two attempts at the deploy above, and both times the agent
answered every question and routed nothing. They are worth checking first when a deployment looks
healthy and saves nothing.

`SMART_ROUTER_ENVIRONMENT` is matched against the environment the artifact was published under. It
is a gate, not a label for the runtime. A mismatch declines every session with a perfectly good
artifact sitting in the bucket, and says so once:

```
smart_router: no model for agent support: artifact environment 'prod' != configured 'agent-platform'
```

`SMART_ROUTER_VERTEX_LOCATION` is where the embedding model is served, not where the deployment
lives. `gemini-embedding-2` is published to `global` and not to `us-central1`, so setting it to the
deployment's own region 404s every call, the breaker latches after five, and the router passes
everything through. Leave both at their defaults unless you have a reason.

The router now records `disabled` when it has no artifact in hand, so the first of these shows up
in `decision_source` rather than as an absence of rows. That distinction matters more than it
sounds: an agent with no artifact and an agent with no plugin installed used to produce the same
warehouse rows, which is none.

### The analytics plugin needs its own grant

The third row from the top of this page, with the commands. It applies to every hosting shape and
not only this one; it is written out here because Agent Platform is where the identity is least
obvious, being a service agent Google creates rather than an account you chose.

Without it every batch is dropped with a 403 on `TABLES_UPDATE_DATA`, the agent keeps answering,
and nothing reaches `status` or the next training run.

Two grants, and they are different scopes on purpose. The rows only need write access to the one
dataset, which is a `WRITER` entry on the dataset's access list:

```bash
RUNTIME="service-$PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
DS="$PROJECT:$SMART_ROUTER_BQ_DATASET"

bq show --format=prettyjson --dataset "$DS" > /tmp/ds.json
python3 - <<PY
import json
d = json.load(open("/tmp/ds.json"))
d["access"].append({"role": "WRITER", "userByEmail": "$RUNTIME"})
json.dump(d, open("/tmp/ds.json", "w"))
PY
bq update --source /tmp/ds.json "$DS"
```

The access list rather than `bq add-iam-policy-binding -d`, which is the tidier command and
answers `This feature requires allowlisting` on a project that has not been enrolled in
dataset-level IAM. The two are equivalent: `WRITER` is `roles/bigquery.dataEditor`.

The plugin also creates a handful of analytics views at startup, and a view is a query job, which
is a project-level permission no dataset binding can grant:

```bash
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$RUNTIME" --role="roles/bigquery.jobUser"
```

The second is optional. Without it the views fail to create and log a 403 on
`bigquery.tables.create`, and every row still lands, which is what `status` and the retrain
trigger actually read. Skip it if a project-level binding is more than you want to give a runtime.

With both in place the loop closes end to end. Twelve more sessions through the same deployment,
read back out of `agent_events` rather than out of the response stream:

| decision_source | n | sessions | replicas | model_version | avg kept |
| --- | --- | --- | --- | --- | --- |
| fresh | 12 | 12 | 1 | 20260905T171738Z | 38.0 |
| reused | 6 | 4 | 1 | | 38.0 |

38 of 100 tools declared, and the six `reused` rows are later turns of four of those sessions
holding the decision their first turn made. That is a routing decision taken inside a managed
runtime, stamped onto the event, written by the analytics plugin, and queryable, which is every
hop the retrain trigger and `smart-router status` depend on.

The router itself needs nothing from BigQuery at request time. Both grants are for the plugin that
carries its provenance to the warehouse.

## Keeping the measurement running

Wherever you deploy, the BigQuery Agent Analytics plugin has to keep logging for `status`, the
holdout arm and the retrain trigger to work. It is a second plugin on the same `App`, and
`doctor` checks that it is registered *after* the router, because the other order throws every
routing decision away before it is written. That check matters more after a deploy than before
one, because a plugin list is the kind of thing that gets reordered by a merge.

The retrain and withdrawal jobs are a separate deployment and a separate identity. They run as
Cloud Run jobs on a timer out of a clone of this repository, they need BigQuery and write access to
the bucket, and they have nothing to do with the agent's runtime.

[install.md](install.md) covers getting to this point. [operations.md](operations.md) covers the
jobs above and what to watch once the router is serving.
