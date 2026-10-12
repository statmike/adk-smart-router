# The dashboard

Everything a running router is worth is already in [BigQuery](https://cloud.google.com/bigquery).
Getting a number out of it means an
`UNNEST`, a JSON path and a join, which is fine in a script and impossible in a chart. So the
project ships ten views that have already done the join. Point a BI tool at them and the
panels are columns.

```bash
./scripts/deploy_sql.sh src/adk_smart_router/sql/dashboard.sql
```

Views cost nothing to create and nothing to keep, so redeploying after a change is always safe.
There is no state here and nothing to migrate.

If the deploy fails, it is almost always a missing table, because BigQuery resolves a view's
references when it creates the view. `Not found: router_backtest` means the project has trained
a model and never backtested one, so run `backtest.sql` once and redeploy. `Not found:
router_lifecycle` means it has never retrained, and this creates the empty ledger without
inventing a row to put in it:

```bash
uv run scripts/record_step.py --ensure-table
```

## The ten views

| View | Grain | Answers |
| --- | --- | --- |
| `v_router_decisions` | one decision | Every routing decision, flattened. The base for the rest. |
| `v_router_sessions` | one session | Which arm it was in, how many turns, what it cost. |
| `v_router_cost_daily` | day, agent, arm | Billed-equivalent tokens per turn. |
| `v_router_g1_daily` | day, agent | The cost gate: routed against holdout, as a percentage. |
| `v_router_decision_sources_daily` | day, agent, source | Why sessions landed where they did. |
| `v_router_coverage_weekly` | week, agent, arm | Coverage against the frequency baseline. |
| `v_router_tool_cache_daily` | day, agent, tool | Tool cache hit rate. |
| `v_router_stale_serves_daily` | day, agent, tool | How often the cached answer had already moved. |
| `v_router_generations` | one published version | What it was trained on, and what it was worth. |
| `v_router_cycles` | one retrain cycle | Every step of it, pivoted into one row. |

The first eight read what the router logged while it was serving. The last two read the retrain
ledger, which is the training side of the project rather than the serving side, and they are
empty until an agent has retrained once. [docs/observability.md](observability.md) covers the
ledger itself and the `smart-router status` command that reads the same rows from a terminal.

None of them takes a query parameter, because a view cannot. `agent` and `day` are columns and
the BI tool filters on them. That is also why none of them mentions a session prefix:
`scripts/report_ab.py` filters by prefix because it reports on one run, and a dashboard reports
on production, where every session counts.

Every view that reads `agent_events` carries a 90-day floor on `timestamp`, the partitioning
column, so the worst a careless chart can do is scan a quarter. Widen it if you keep more history
than that, but widen it on purpose. `agent_events` is the largest table in the dataset and a
dashboard queries it on a refresh timer.

## Connecting Looker Studio

Create a data source per view rather than one over the dataset. In Looker Studio, add a BigQuery
source, pick **Custom query** only if you need something the views do not have; otherwise pick the
view by name, which lets Looker Studio push your date filter down into the partition.

Set the date range control on `day` (or `week_start` for coverage). Set the default aggregation
on every ratio column to **None** and compute the ratio in the chart if you need to re-aggregate,
because averaging a rate across days weights a quiet day the same as a busy one. `billed_per_turn`,
`hit_rate`, `stale_rate`, `share` and `g1` are all rates.

The same views work unchanged in Looker, in a notebook, or in `bq query`. Nothing here is
specific to a BI tool.

## The four panels worth having

**Is it paying?** `v_router_g1_daily`, a line on `g1` with `meets_target` as the colour. Negative
is cheaper and the gate is a 10% reduction. Put `holdout_sessions` on the same page as a number:
a percentage computed against a nearly empty control arm is noise wearing a decimal point.

**Is it still routing?** `v_router_decision_sources_daily`, a stacked area on `decisions` grouped
by `decision_class`. A healthy routed agent is mostly `reused` with one `fresh` per session.
Anything in `fail-open` is an embedding endpoint problem rather than a router one. `withheld`
appearing means a guard stopped a prune the router had already decided on, and if the source
under it is `watch` rather than `floor`, routing has stopped for every session on that agent
until someone re-promotes it.

**Is the model still right?** `v_router_coverage_weekly`, two lines: `coverage_model` against
`coverage_static`. The gap is what the model is worth over counting how often each tool gets
called. Filter to `corpus_source = 'production'` and `overlaps_training = false` before reading
either number, or you are reading an in-sample score on synthetic traffic.

**Is the tool cache safe?** `v_router_stale_serves_daily`, a bar on `stale_rate` by tool. This is
the readout from the refresh probes and it is the only thing in the system that can see a tool
whose answer moved while a cached copy of the old one was being served. A rate near zero means
the TTL is conservative enough. Anything materially above zero means that tool should not be
cached at that TTL, and possibly should not be cached at all.

## The retrain history

The two ledger views answer a different question from the eight above. Those tell you how the
router is doing today. These tell you how it got here.

**What has shipped, and what each one was worth.** `v_router_generations`, one row per published
version. The training numbers and the serving numbers sit on the same row, so the gap between
`predicted_g1` at publish time and `g1` from real traffic is a subtraction rather than a project.
`is_current` marks the version live now, and `live_hours` is how long each one ran. Judge a
generation on `g1_later` rather than `g1`, for the reason below, and that is the column the
automatic withdrawal in [operations.md#the-generation-watch](operations.md#the-generation-watch) reads.

**What happened on each retrain.** `v_router_cycles`, one row per cycle with the seven steps
pivoted into columns. `result` is the cycle's own verdict: `published`, `rejected`, `failed`, a
standalone `drift check`, a `withdrawn`, a `held off` where `run_cycle.sh` found a retrain due and
the last one too recent, or `incomplete` when a runner died partway. `shape_changed` is the one to
watch over time, because it says the retrain actually moved the router's configuration, which is
much rarer than a retrain.

On an agent with the loop on a timer, most rows are `drift check` and that is what healthy looks
like. The one to alert on is the absence of rows, not the presence of a quiet verdict.

**What the watch decided.** `withdraw` is the one step here nobody asked for.
`scripts/watch_generations.py` runs on a schedule and records a row every time it looks: `ok` when
it took a generation out of service, `declined` when the generation is fine, `skipped` when there
was not enough traffic in both arms to judge. `withdraw_g1_later` is the number it acted on and
`withdraw_routed_billed_per_turn` and `withdraw_holdout_billed_per_turn` are where that number came
from. Charting the three outcomes together is how you tell a quiet watch from one that stopped
running.

Both are small. The ledger is a few hundred rows a year, so these two are the cheapest views in
the file by a wide margin and there is no reason to sample them.

## What each panel can lie about, and the column that tells you

Each of these is a way a chart can be confident and wrong, so the check sits in the view
rather than in a comment on the dashboard.

**A cost saving with no control arm.** `v_router_g1_daily` returns `NULL` for `g1` rather than a
number when the holdout arm is empty, which is a gap in the line instead of a flat zero.

**A cost comparison that is really a comparison of turn positions.** A session's first model call
has no prefix of its own to hit and bills about three times a continuation. So any average over
both arms is partly a statement about how much of each arm is first calls, and the router's own
rules are what decide that. `v_router_generations` reports the comparison both ways. `g1` pools
every call and is a true statement about the invoice. `g1_later` drops each session's first call
from both arms and is the statement about pruning. On this project's own warehouse the two came
out at -10.6% and -29.7%, because pruning made turn one 22% dearer and later turns 30% cheaper at
the same time. Neither number is wrong. They answer different questions, and a chart showing only
the first will mis-attribute the difference. `routed_later_turns` and `holdout_later_turns` are
the denominators, and `v_router_sessions` carries `later_turns` and `later_billed_input_tokens` if
you want to slice it yourself.

**Sessions leaving both arms.** `unassigned_sessions` is the count of sessions the router neither
routed nor held out: an embedding failure, a demoted agent, or a decline. A rising one is what a
`cache_watch` demotion looks like on the way past. The treatment arm quietly becomes a second
control, the ratio stays flat, and nothing else on the page changes.

**A coverage trend stitched from several backtests.** `v_router_coverage_weekly` is the newest
run for each agent, all of it, and not the newest row for each week. One run reports every week
in its window under one model and one set of arguments, so its weeks are a trend. Taking each
week from whichever run touched it last produces a line whose points came from different models,
and that reads as a trend too. Older runs are still in `router_backtest` when you want to compare
two of them.

**A tool cache hit rate computed from a path that is dropping records.**
`v_router_tool_cache_daily` counts serves twice over, once from the router's own provenance and
once from the marker the cache writes into the tool result. `serve_marker_gap` is the difference
and should be zero. Anything else means one of the two logging paths is losing rows, and a hit
rate computed from either alone would look perfectly reasonable while it happened.

**A generation credited with the wrong window's traffic.** A routed session records the model
version it ran under, so `routed_sessions` in `v_router_generations` is exact. A holdout session
records no version, because the point of holding it out is that no model touched it, so
`holdout_sessions` is matched by time instead: the window from this version going live to the
next one going live. Two columns say whether that window is trustworthy. `deploy_recorded` is
false when nothing logged a deploy, in which case the window opens at publish time and the hours
between publishing and rolling out are credited to the wrong version. `first_served` before
`live_from` is the same fault showing through from the other side.

## Two caches, two hit rates

`cached_share` in `v_router_cost_daily` and `hit_rate` in `v_router_tool_cache_daily` are not the
same measurement and are not related.

`cached_share` is Agent Platform's [context
cache](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/context-cache/context-cache-overview):
the share of the prompt the model served from a prefix it had already seen. It is the number pruning moves, and the one the cost gate is really about.

`hit_rate` is the router's own tool result cache: how much of a tool's demand was answered from a
stored payload instead of running the tool. It saves latency and whatever the tool costs to call,
and it has nothing to do with tokens.

An agent can have an excellent one and a terrible other.

## Empty views are usually correct

`v_router_tool_cache_daily` and `v_router_stale_serves_daily` return nothing until some tool is
declared cacheable, because until then there are no cache entries to serve or to go stale. That
is the default and it is a reasonable place to stay. `v_router_coverage_weekly` returns nothing
until a backtest has run. `v_router_g1_daily` returns rows with a `NULL` `g1` until the holdout
arm has traffic. `v_router_generations` and `v_router_cycles` return nothing until an agent has
retrained once, because until then there is no ledger to read.

None of these is a deployment problem, and telling them apart from one is why every view exposes
its denominator as a column next to the rate computed from it.
