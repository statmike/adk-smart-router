# Installing, in the order the questions come up

There are two very different things people mean by installing this. One is adding a package to an
[ADK](https://adk.dev/) agent and finding out whether it can help. The other is standing up the
[BigQuery](https://cloud.google.com/bigquery) pipeline that trains the model it serves. The first
takes a minute and needs no cloud account. The second needs a Google Cloud project and a corpus of
logged turns.

How much work that second part is depends entirely on one thing: whether your agent is already
logging to BigQuery through ADK's BigQuery Agent Analytics plugin ([ADK integration
guide](https://adk.dev/integrations/bigquery-agent-analytics/), [BigQuery
documentation](https://docs.cloud.google.com/bigquery/docs/bigquery-agent-analytics)). If it is,
the corpus already exists and you may be able to train today. Step 2 is the command that tells you
which of those you are.

Do step 1 first either way. Most of the value of `qualify` is telling you not to bother with any of
the rest.

## 1. Add the package and ask whether pruning can pay

```bash
uv add adk-smart-router
# or directly from GitHub:
uv add git+https://github.com/statmike/adk-smart-router
uv run smart-router qualify --app myapp.agents:root_agent
```

`--app` takes `MODULE:ATTRIBUTE` and accepts an ADK agent, an `App`, or a zero-argument factory
that returns either. Nothing here touches the network. It reads your agent's declarations and
instruction, counts them, and does the arithmetic in `qualify.py`.

The output ends in one of four verdicts.

| Verdict | Meaning | What to do |
| --- | --- | --- |
| `PAYS` | The steady-state saving is at or above 25% of a turn and the pruned prompt still clears the model's cache floor. | Go on to step 2. |
| `THIN` | Above the floor, but either the saving is small or pruning would carry the prefix your sessions share under the floor. | You can proceed. Expect a modest result, and read the reason: the two causes have different remedies. |
| `UNCACHED` | The prompt is under the cache floor already, so nothing caches. Every removed token saves at full rate. | Proceeds, but no agent in this band has been measured end to end. Treat the number as untested. |
| `TRAP` | Pruning would carry a cacheable prompt *under* the floor, turning a prompt that caches into one that does not. | Do not install. This is the one case with measured evidence of loss: +25.5%. |

Exit codes: `0` when pruning is worth trying, `1` for `TRAP`, `2` when the command could not do its
job, usually a bad `--app` path.

Two things it cannot see, both of which it says so in its own output. It counts no conversation
history, so pass `--history-tokens` with what your turns actually average; and it cannot see how
warm a menu's prefix gets under your traffic, which was worth 22 points of G1 on the one agent
measured end to end. Once you have logged traffic, `scripts/predict_saving.py` supersedes it.

On the second one, there is a rule of thumb you can apply before you have any traffic at all.
Prefix caching is a short timer and nothing else: a prefix sent seconds ago opens warm, one that
has idled a minute is about a coin flip, and one that has idled ten minutes retains nothing. No
amount of repetition changes that. So the number that matters is sends per menu per minute, which
is your turns per minute divided by your menu count. Below about one, expect every session to
open cold and read `qualify`'s saving as an upper bound. A quiet agent wants fewer and wider
menus; a busy one can afford more and narrower ones.

`--history-tokens` does more than fix the denominator, which is why it is worth a real number
rather than a guess of zero. Your agent has two cache floors, not one. Inside a session each turn
begins with the last one, so the whole prompt is what caches and the whole prompt is what the
floor is judged against. Across sessions, all that two of them share is your system instruction
and your declarations, because no other session has this session's history. If pruning carries
*that* under the floor, every session opens cold no matter how much the rest of the prompt
caches, and turn one is the expensive one. `qualify` can only separate the two if you tell it how
much of the prompt is history.

### How much of your surface a turn-one router can reach

`qualify` answers a second question in the same run, and it is the one adopters usually hit a
month in: not every tool can be predicted from a first message. A tool is reachable only if that
first message can supply what the tool requires, so anything gated on an identifier your systems
assign is a second hop. The agent has to call `search_products` before it has a SKU to ask about,
and no prediction made at turn one ever reaches the tools behind that. Those tools are not lost.
The menus carry them by co-occurrence, which is why the pruner keeps unknown tools
unconditionally.

Nothing about this is guessed from argument names. `order_id` and `subscription_id` are the same
type, the same shape and the same suffix, and on the reference agent a customer holds the first
and not the second, so any rule that told them apart would be a coincidence. What the schemas
settle by themselves is settled, and the rest is reported as a span: at least this many tools
reachable, at most this many, and here is the list that closes the gap.

To close it, mark the arguments once:

```bash
uv run smart-router qualify --app myapp.agents:root_agent --write-arguments args.txt
# edit args.txt, one line per argument: `held` or `supplied`
uv run smart-router qualify --app myapp.agents:root_agent --arguments args.txt
```

`supplied` means a first message can carry it, like an order number or a postcode. `held` means
only your systems have it. The file is keyed by argument name across the whole surface rather
than by tool, which is what makes it a ten minute job: the reference agent needs 74 decisions for
100 tools, and `sku` alone settles 19 of them. Lines come out commonest first, so stopping part
way down still resolves the most, and the span just stays wider. Write `tool.argument` on a line
to override one tool. `examples/reference_agent/arguments.txt` is a filled-in example.

### Checking the markings against your own traffic

Once the analytics plugin has logged a few thousand sessions, you can ask them:

```bash
uv run smart-router check-arguments --app myapp.agents:root_agent \
  --arguments args.txt --project my-gcp-project --agent support --min-sessions 3
```

It reports which tools your sessions actually opened with, which of your markings that
contradicts, and exits 1 if there are any. A contradiction is worth reading either way: either the
marking is wrong, or the model is inventing that argument to satisfy a required field. On the
reference agent it was the second one every time, so raise `--min-sessions` before believing a
contradiction with only one or two sessions behind it. Adding `--write-arguments args2.txt` writes
a fresh template with the session counts in the margin, and every line still blank, because
nothing here is confident enough to answer for you.

## 2. Ask whether you already have the corpus

The router trains on history. Nothing in the training run requires the router to have been
installed while that history was being written, so an agent that has been logging through the
analytics plugin for a month already owns a month of training data. This command reads that log
and says whether it could train today.

```bash
uv run smart-router readiness --agent support --project my-gcp-project --dataset my_bqaa_dataset
```

`--dataset` is the dataset the analytics plugin already writes to, not one this project creates.
The command needs read access and nothing else: no bucket, no connection, no clone, no artifact,
and it writes nothing.

```
smart-router readiness — support, last 35 days of smart_router

  [ok] first user messages carry text
         99.6% of 27,129 prompts carry text at $.text_summary, which is where the labels
         come from.
  [ok] enough labelled sessions, over enough tools
         8,496 trainable sessions, 7,097 of them opening with a tool over 59 distinct
         tools, 32 of which clear the 10-row floor. The rest fold into ABSTAIN rather
         than blocking the run.
  [ok] sessions stay inside a menu of 11
         a session touches 1 distinct tool at the median, 2 at p90 and 3 at p99, widest
         5, over 7,151 sessions that called anything.
  [ok] a frequency table is not already enough
         the 11 most-called tools would have covered 86.5% of tool-opening sessions,
         which leaves room for a model to beat.

  verdict                 trainable today
  scanned                 552.5 MB, nothing written
```

The four questions are in the order that kills the idea fastest. Missing prompt text is somebody
else's plugin configuration and no amount of traffic fixes it. Sessions wider than a menu cannot
be fixed at all, because the router chooses once and never widens. A frequency table that already
covers 95% of sessions means the correct product is a YAML file and none of this machinery. Only
a thin corpus fixes itself, by waiting.

Each finding reads `ok`, `thin` or `STOP`. A `thin` means the answer is yes on evidence too small
to lean on, which is a real state and a different one from no. The command exits 1 if anything
stopped and 0 otherwise, so `thin` does not fail a script.

It stops one question short on purpose. The last gate, whether a learned model actually beats that
frequency table, needs embeddings, and embeddings need the connection `provision.sh` creates. So
that one waits for step 4, where `sql/preflight.sql` asks it.

## 3. Wire the plugin in, then check the wiring

```python
from adk_smart_router import SmartRouterPlugin

app = App(
    name="myapp",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800),
    plugins=[
        SmartRouterPlugin(config=RouterConfig(...)),
        BigQueryAgentAnalyticsPlugin(
            project_id=...,
            dataset_id=...,
            custom_metadata_allowlist=["smart_router"],
        ),
    ],
)
```

```bash
uv run smart-router doctor --app myapp.agents:app
```

`doctor` needs an `App` rather than a bare agent, because plugins and the cache config live there.
It checks the wiring issues that fail *silently*:

1. **The router is registered before the analytics plugin.** Reversed, the analytics plugin
   persists the event before the router has stamped its provenance onto it, and every routing
   decision is thrown away. The agent runs fine and the warehouse is empty of routing records.
2. **`smart_router` is in `custom_metadata_allowlist`.** Without it the analytics plugin drops the
   key. Same outcome, different cause.
3. **A `ContextCacheConfig` is set.** Without caching, session-stable pruning buys almost nothing,
   because the whole economic argument is about tokens that would otherwise bill at 0.10.
4. **Active judge dependencies (`xgboost`, `google-cloud-storage`) and `gcs_bucket`.** When
   `"xgboost"` is in `judges`, `doctor` verifies that `xgboost` and `google-cloud-storage` are
   installed (`importlib.util.find_spec`) and, if `mode == "active"` with `"xgboost"` as the only
   judge, that `gcs_bucket` is set.

Each of these leaves an agent that starts, serves and looks healthy while doing the wrong thing,
which is why they get a command instead of a paragraph.

[deploying.md](deploying.md) picks up from here if you are shipping this to Cloud Run or Agent
Platform rather than running it locally. The short version is that the plugin adds two permissions
and about 104 MB of memory, and nothing else changes. Check one thing that is not the router's
while you are there: the runtime needs write access to the analytics dataset, or the plugin beside
the router logs nothing and every measurement below is empty.

## 4. The training pipeline needs a clone

The wheel ships the library, its SQL, and the `smart-router` command. It does not ship
`scripts/`, and that is deliberate: those are analysis and operations tools that read a specific
warehouse layout, and versioning them as part of a runtime dependency would tie an agent's
deployment to the schedule of a data pipeline.

So the GCP side is a clone, not an install:

```bash
git clone https://github.com/statmike/adk-smart-router
cd adk-smart-router
uv sync
gcloud auth application-default login
export SMART_ROUTER_BQ_DATASET=my_bqaa_dataset   # the one the analytics plugin already writes to
./scripts/provision.sh
```

`provision.sh` is idempotent and enables the APIs, creates the dataset and the bucket, and deploys
the SQL. There is no `smart-router provision` on purpose: wrapping a shell script that talks to
four APIs in a Python command that talks to none would only move the credentials question somewhere
less obvious.

**It does not build you a second warehouse.** If step 2 said trainable, point
`SMART_ROUTER_BQ_DATASET` at the dataset you already have and the training run reads the analytics
tables in place, writing its own feature and model tables beside them. What provisioning adds on
top of an existing analytics setup is three things: a BigQuery connection with permission to call
the embedding model, a Cloud Storage bucket to publish artifacts into, and the SQL. Your logged
turns are not copied, exported or reshaped, and nothing in this project reads any other project.

Authentication is [Application Default
Credentials](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc),
everywhere, always. Nothing in this project reads a service account JSON key, and `.gitignore`
blocks committing one.

From there, [training.md](training.md) covers building a corpus, training and publishing an
artifact, and what each gate means. [operations.md](operations.md) covers what to watch once it is
serving.

## 5. Day two: is it working, and is it paying?

The four commands above are all things you run before the first deploy. This is the one you run
after, and it is the same command every day from then on:

```bash
uv run smart-router status --agent support
```

It prints what artifact is being served, what the last seven days cost against the holdout arm,
where the routing decisions went, whether the model has drifted, and when the next retrain is
due. It reads and never writes, sends no model traffic, and prints how many bytes it scanned.

Run it the day you deploy, before there is anything to see. Almost every line will be an absence:
no artifact published, no control arm, drift never checked. That is the useful part. Each absence
names the thing that would fill it, so the output doubles as the list of what is left to set up.

Two of those absences have commands attached:

```bash
./scripts/deploy_sql.sh src/adk_smart_router/sql/dashboard.sql   # the views status reads
uv run scripts/check_drift.py --agent support                    # fills the drift line
```

[observability.md](observability.md) covers what each line rests on, the ledger the training
scripts write their history into, and which drift signals are allowed to trigger a retrain.

## What you get without a clone

An installed package with no artifact published does nothing at all. The plugin fails open and
passes every request through untouched. That is the intended state while you are still deciding.
`qualify`, `readiness` and `doctor` all work in it, which is the point: you can answer "should we
do this", "do we already have what it needs" and "did we wire it right" before spending anything.
Only the middle one needs credentials, and it needs read access and no more.
