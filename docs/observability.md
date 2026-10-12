# Watching it, and reading back how it got here

A router that prunes tools is only worth running if you can tell whether it is helping. This page
is about the second half of that: how the plugin records what it did, and how you read it back.

There are two questions and they have different answers.

**Is it working, and is it paying?** That is one command, and it takes about ten seconds.

```bash
uv run smart-router status --agent support
```

**How did it get here?** That is the retrain ledger, a small BigQuery table the training scripts
write one row into per step. [docs/dashboard.md](dashboard.md) covers the two views that read it.

Everything on this page is offline. Nothing here runs on the request path, nothing adds a serving
dependency, and if BigQuery is unreachable the router routes exactly as it does today.

## `smart-router status`

The day-two command. `qualify` asks whether pruning can pay on your agent and `doctor` asks
whether the wiring is right; both are things you run before the first deploy.
[docs/install.md](install.md) covers those. This one is for every day after.

```
$ uv run smart-router status --agent support

live artifact      20260905T171738Z    trained 2026-09-05, 3 days ago
  from run         f8897a5a-0419-498b-a873-25ab978d4b53  ·  9,043 rows / 6,789 sessions over 35 days
  gates            miss reduction 0.81  ·  menu recall 0.9659
  settings         24 menus of 11   (no tuning recorded)

last 7 days        1,479 sessions routed  ·  1,547 holdout  ·  1,851 in neither arm
  cost             G1 -11.0%   over a 1,547-session control arm
  decisions        holdout 32.7%  ·  reused 25.7%  ·  lowconf 17.6%  ·  late 8.3%  ·  floor 5.1%  ·  fail-open 4.4%
  tool cache       nothing served from cache, over 12,552 tool calls

drift              unseen share 27.4%  against a 26.1% baseline   (+1.3)
                   confidence p50 0.79  against 0.81
  verdict          not due, checked 2026-09-06

next retrain       2026-09-19, on age (max_model_age_days 14)
last cycle         2026-09-04   train ok  ·  tune declined  ·  publish ok

  read 0.25 GB
```

Four blocks, and each one rests on something different.

**The artifact** comes from GCS, not from BigQuery. It is the file a serving process would load
right now, loaded the same way the plugin loads it, so what you are reading is what is being
served rather than what the last training run produced. The gate numbers under it are the ones
recorded on the artifact at publish time.

**The last seven days** come from `v_router_cost_daily` and `v_router_decisions`. G1 is
turn-weighted over the whole window, `SUM(billed_per_turn × turns) / SUM(turns)`, rather than an
average of the daily percentages, because averaging those weights a quiet Sunday like a Monday.

**Drift** is read back from the last drift row in the ledger, not measured again. Measuring it
means scanning `LLM_REQUEST` rows that carry the whole serialized tool list at about 94 KB each,
which is the expensive read in this project, and it should be paid for once by the scheduled
check rather than every time somebody is curious. The line says when it was checked so you can
see it going stale.

**Next retrain** is the age rule and the drift verdict together, whichever comes first.

The last line is what the command scanned. Asking has a price, and it should be visible to
whoever is asking.

### Every absence is stated

The first person to run this will have almost none of these numbers, and that is not an error
condition. Nothing published yet reads as `live artifact none`. No holdout traffic reads as `no
control arm, so there is no saving to report`, with no percentage at all. Never having run the
drift check reads as `never checked`, followed by the command that would change that. The views
not being deployed reads as a hint to deploy them. `status` always exits 0, because everything it
prints is a state rather than a verdict.

`doctor` is the command that fails.

## The retrain ledger

`router_training_runs` records the training step and nothing else. The tuner's decision lands in
a local JSON file the next cycle overwrites. The publisher's verdict lands in a GCS object and on
stdout. So *why did this version ship with these settings, and what did the tuner reject to get
here* was not answerable from the warehouse a week later, and the runner that knew had gone home.

`router_lifecycle` is the missing record. One row per step of one cycle.

| Column | Holds |
| --- | --- |
| `cycle_id` | Twelve hex characters, minted by `retrain.sh` and passed down to every step. |
| `step` | `train`, `tune`, `publish`, `backtest`, `drift`, `deploy` or `withdraw`. |
| `recorded_at` | When the row was written. The partitioning column. |
| `agent`, `environment` | Which agent, and which deployment of it. |
| `outcome` | `ok`, `declined`, `rejected`, `failed`, `skipped` or `due`. |
| `reason` | A sentence written to be printed. This is the "why" the table exists for. |
| `run_id`, `version` | The training run and the published artifact, for joining outward. |
| `settings_before`, `settings_after` | The tuner's shape either side of its decision, as JSON. |
| `metrics` | Whatever that step measured, as JSON. |
| `source` | Which script wrote the row. |

`declined` and `rejected` are kept apart on purpose. The tuner **declines**, which is the normal
answer meaning no recommendation beat the one already standing, and it leaves the previous
settings in place. A gate **rejects**, which means a model was measured and found wanting.
Collapsing the two would make a healthy cycle and a failed promotion read the same in a chart.

Each step writes its own row rather than the runner collecting them at the end, because all of
these scripts are usable standalone and frequently are. A publish run outside a cycle still
leaves a trace: it mints its own `cycle_id` and becomes a single-step cycle, which is honest
about what happened.

**The ledger is authoritative for nothing.** `router_training_runs` still gates promotion,
`metadata.json` is still what the loader verifies its checksum against, and the settings file is
still what the publisher reads. It follows that a failed ledger write must never fail the step it
describes, so every write catches everything and carries on. An observability system that can
take a retrain down is a worse trade than a gap in a chart.

Two steps are not written by anything this project owns. `train` runs as a BigQuery stored
procedure and `deploy` is whatever you use to put a new artifact in front of traffic, so both say
what they did through a script:

```bash
uv run scripts/record_step.py --step deploy --outcome ok \
    --version 20260905T171738Z --reason 'rolled out to prod'
```

Recording a deploy is optional and worth doing. Without it, `v_router_generations` has to assume
a version went live when it was published, and the hours between publishing and rolling out get
credited to the wrong version.

On an agent that has never retrained the table does not exist yet, which is fine everywhere
except when deploying the dashboard, because BigQuery resolves a view's references when it
creates the view. `uv run scripts/record_step.py --ensure-table` creates the empty table without
inventing a row to put in it.

## Drift, and what is allowed to trigger a retrain

The router already had two ways of noticing it had gone stale, and neither is a detector.
`cache_watch` reads the bill, so it fires after the money is spent. `max_model_age_days` reads
the clock, so it fires whether or not anything has changed. Drift is the third, and it is
readable from the requests themselves.

```bash
uv run scripts/check_drift.py --agent support
```

Three signals are measured. **Only one of them gates.**

**Unseen share** is the fraction of the agent's currently declared tools that the model has no
class for. It gates. A real drift event on 2026-09-04, a surface change from 25 tools to 100,
cost 36 points of recall and moved unseen share from 26.1% to 77.3% at the same instant. The
signal tracks the harm.

**Top-1 confidence**, at p50 and p10, is reported and never gates. A constructed intent shift
moved confidence by 0.25 and cost 1.3 points of recall. That is sensitive to change rather than
to harm, which makes it worth printing and wrong to trigger on.

**Age** gates, and always did. It is the fallback for everything the first two cannot see.

The threshold is 20 points of unseen share above the model's own baseline. Two things about that
are worth knowing before you rely on it.

**The baseline is not zero.** A healthy model had no class for 26.1% of its declared surface,
because the rare-class floor drops tools that never appear often enough to learn. A detector
comparing against a constant would read a healthy model as a quarter drifted on its first day.
So the baseline is measured at publish time and written into the artifact's own metadata, and
drift is always measured against the model's own starting point. A model published before this
existed has no baseline, and the check says so rather than guessing one.

**Twenty points is a two-point fit and should be read as one.** There is one real drift event and
one real non-event behind it. `--backfill 30` replays your agent's own surface history against a
range of candidate margins, so you can argue with the default using your numbers instead of ours.

## What runs by itself

The goal is an agent you install this on once. The worst outcome should be no improvement, never
degradation, and any saving that becomes available should arrive without anyone doing anything.

**The no-degradation half runs by itself already, and it is all on the serving side where the
blast radius is one session.** Every callback fails open to passthrough. `cache_watch` demotes an
agent on its own when the bill says pruning is costing money. A model that fails to load leaves
the router in passthrough rather than raising. A failed load expires and is retried. The permanent
holdout arm is never routed, so there is always a live measurement of what not routing costs on
the same traffic.

**Delivery runs by itself now too.** A running process re-reads the bucket every
`refresh_interval_s`, 900 seconds by default, and swaps in a newer artifact without a redeploy.
Publishing is therefore enough: a better model reaches the fleet in about a quarter of an hour. A
poll that cannot reach the bucket, an artifact that fails its digest, and a version built against a
different embedding model all leave the process serving what it already had.

**Rollback runs by itself too.** `scripts/watch_generations.py` compares the generation that is
serving against its own concurrent holdout arm and withdraws it if routing is costing more per turn
than not routing at all. Withdrawing writes one small marker into the version's directory, the
loader stops counting that version as published, and the fleet falls back to the generation before
it on its next re-poll. It is reversible with `--restore`, rate-limited to one withdrawal per agent
per day, and it records a ledger row every time it looks including the times it does nothing.
[docs/operations.md](operations.md) has the operating detail.

**And the trigger runs by itself, last of the four and last on purpose.**
`scripts/run_cycle.sh` reads the drift verdict, stops when a retrain is not due, and runs
`retrain.sh` when it is, under the same cycle id so the verdict and the retrain it caused are one
row. A minimum of seven days between cycles is read out of the ledger rather than off the runner,
so two triggers in one day retrain once. It went last because an automatic deploy without an
automatic rollback is the one ordering that can make an agent worse.

What stays with a person is turning the router on for the first time, the arguments file, and
`holdout_fraction` (covered in [docs/install.md](install.md) and [docs/operations.md](operations.md)). Everything between a
published model and the next one is now on a timer; [docs/operations.md](operations.md) has the
Cloud Scheduler recipe.

The two commands on this page are still how you find out what happened, and `status` tells you
what each agent is serving.

## What it costs

The ledger is a few hundred rows a year and the two views over it are the cheapest in the
project.

`status` is five BigQuery reads and one GCS download, and it prints what it scanned. Two of the
five read views that expose only a date, so an outer filter cannot prune partitions and those
reads cover the view's 90-day floor regardless of `--days`.

`check_drift.py` is the expensive one, because reading the declared tool surface means reading
`LLM_REQUEST` rows that carry the whole serialized tool list. It prunes partitions, samples, and
defaults to one day of surface history. `--estimate` prices the reads without running them.

## Where to look next

- [docs/dashboard.md](dashboard.md) — the ten views, including `v_router_generations` and
  `v_router_cycles`, and what each panel can lie about.
- [docs/operations.md](operations.md) — the operating loop, the kill switch, and the counters a
  running process exposes.
- [docs/training.md](training.md) — what a retrain actually does.
