# The evidence

One page: what was measured, on what, at what it cost, and what it does not show. Everything here
is reproducible from this repository, and the last section says which script produces which number.

## The headline

A live A/B over 600 sessions and 1,894 turns of a 100-tool customer support agent, half routed and
half with the router declining to act. Total spend $12.72.

| | routed against control |
| --- | --- |
| billed input tokens per turn | **−23.4%** |
| the whole bill per session, input and output, unnormalised | **−22.5%** |
| latency | −10.5% |
| turns per session | +0.7%, p = 0.74 |

Billed-equivalent means `uncached + 0.10 × cached`, because [Agent
Platform](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing) charges
about a tenth for a cached input token. Counting raw prompt tokens would credit the router for
cache hits it did not cause. Raw prompt tokens fell 49.7%, and the gap between that and 23.4% is
entirely the cache.

Both arms ran in the same process, in the same hour, against the same model version. Which arm a
session landed in was decided by the shipped `rollout` module hashing
`(app_name, user_id, session_id)`, and read back out of the router's own provenance rather than
recomputed by the reporting script. The harness never assigns an arm, so it cannot assign one
wrong.

## What kind of number this is

Every model call, every billed token and every cache hit was really incurred. The conversations
were not. There is no production traffic for this agent, so the user turns were written by this
project. Read it as a measurement on authored traffic, not as a production lift claim.

Two things narrow that gap without closing it. The prediction rule has been run on a third-party
corpus with no retuning, below. And the plugin ships with the holdout arm enabled, so the first
adopter measures their own lift on their own traffic with the same `report_ab.py` that produced
this table.

## Why −22.5% is a floor

The declarations the router removes were already nearly free, because they were cached. You save a
tenth of what you remove, and only while the cache keeps hitting.

The experiment's shape works against the routed arm here. The control arm sends one tool list, so
all 289 of its sessions keep one prefix warm. The routed arm splits its sessions over 24 menus, so
each menu mostly waits. Repricing every routed opening at the warmth the routed arm itself reaches
when a menu is busy reads 32.1%, and closing the rest of the gap would read 42.4% as an upper
bound. Denser traffic per menu is what moves the number up that range, which is why the router
sends one of a few fixed menus rather than a bespoke set per session.

## What it costs

Added latency is 223.3 ms p95 per session against a 250 ms bar, and 214 ms of that is the one
embedding call. The plugin's own work is 9.3 ms: 3.86 ms p50 to decide, once per session, and
0.42 ms p50 to recall that decision on each later turn. The three terms are added at their own p95
rather than convolved, which overstates the total, in the safe direction for a bar.

Quality did not move. Both arms called a tool in the same share of sessions, made the same number
of calls per session, reached about the same number of distinct tools, and sent the same share of
calls into the long tail. A router hiding tools people needed would show up as a shorter tail. It
does not. Turn count is the direct version of the same question, and it is flat at p = 0.74, which
matters because the two savings above differ only by that turn count.

## Somebody else's data

[`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
is a public function-calling benchmark over real, executable third-party APIs, built with the
[APIGen](https://arxiv.org/abs/2406.18518) pipeline and shipping an answer key. Nothing about this
project shaped it.

The slice routed here holds 3,179 distinct APIs. With no retuning, the prediction rule reaches
0.9960 set recall at a mean menu width of 2.0 tools on a 100-tool slice, and 0.9666 at 5.7 tools
across all 3,179, a 99.8% prune. A live 599-session A/B on the 100-tool surface, driven by the
corpus's own queries, took 13.4% off the whole bill.

**That run is also the one that found a cost.** Because xLAM has an answer key, it is the first
run that could ask whether the saving bought worse answers, and it did: the routed arm called
every tool the key names on 75.6% of sessions against the control's 87.2%, p = 0.003. The cause is
not mysterious. On a wide surface the fixed menu pool cannot cover the traffic, and the share it
misses at publish time is close to the share of requests the agent then fails. Pruning per session
instead removes the cost entirely, at p = 0.71, and removes eleven of the thirteen points of
saving with it, because a prefix no other session shares is a prefix that never caches. Both sides
of that trade are in [scenarios.md](scenarios.md).

## Cold-start and hybrid judges (`PrototypeJudge` + `XGBoost`)

`PrototypeJudge` (`judges=("prototype",)` or `judges=("xgboost", "prototype")`) embeds the live
tool docstrings and JSON schemas once per process, subtracts the catalog mean vector, and
propagates top direct semantic matches one hop upstream across the `Returns:` $\to$ `Args:`
prerequisite graph. Measured across all 3,397 logged `support` sessions and all 2,000 logged
Salesforce `xlam` sessions (`scripts/score_sets.py --compare-tiers`):

| Surface & Arm | Set Recall | F8 | Token Efficiency | Complete Sessions | G3 Total Latency |
| --- | --- | --- | --- | --- | --- |
| **`support` (3,397 sessions, 100 tools, 24 menus)** | | | | | |
| `full list (no router)` | `1.0000` | `0.4623` | `0.0000` | `100.0%` | — |
| `cold-start (proto k=15)` *(0 training rows)* | `0.9264` | `0.7904` | `0.8896` | `91.0%` | `222.0 ms` |
| `cold-start (proto 24 menus)` *(0 training rows)* | `0.8675` | `0.7400` | `0.8874` | `85.4%` | `222.0 ms` |
| `trained (xgb @1600)` | `0.9754` | `0.6735` | `0.6317` | `96.7%` | `222.4 ms` |
| `hybrid (xgb+proto @1600)` | `0.9755` | `0.8690` | `0.9338` | `96.7%` | `225.7 ms` |
| **Salesforce `xlam` (2,000 sessions, 100 tools)** | | | | | |
| `full list (no router)` | `1.0000` | `0.4468` | `0.0000` | `100.0%` | — |
| `cold-start (proto k=15)` *(0 training rows)* | `0.9894` | `0.8376` | `0.8802` | `97.8%` | `222.0 ms` |
| `cold-start (proto 24 menus)` *(0 training rows)* | `0.8269` | `0.6947` | `0.8810` | `74.4%` | `222.0 ms` |
| `trained (xgb @1600, 24 menus)` | `0.9895` | `0.8441` | `0.8126` | `98.2%` | `222.4 ms` |
| `hybrid (xgb+proto @1600, 24 menus)` | `0.9927` | `0.8420` | `0.8027` | `98.7%` | `225.7 ms` |
| `trained (xgb @1600, bespoke --no-menus)` | `0.9961` | `0.9650` | `0.9316` | `99.4%` | `222.4 ms` |
| `hybrid (xgb+proto @1600, bespoke --no-menus)` | `0.9972` | `0.9655` | `0.9527` | `99.6%` | `225.7 ms` |

On the **Newly Deployed / Unseen Tool Stress Test** (`--unseen-holdout 15`, holding 15 active tools
out of `xgboost`), pruning unseen tools under `xgboost` alone drops `support` recall to **0.6617**
(`62.9%` complete sessions), whereas `hybrid (xgb+proto)` recovers **0.9004 recall** (`88.4%`
complete sessions, **0.8308 F8**, **0.9483 token efficiency**), and on Salesforce `xlam` lifts
pruned-session recall from **0.9512** (`91.8%` complete) to **0.9784** (`96.1%` complete; **0.9815
overall**).

## The worst case

An agent small enough that pruning carries its shared prefix under the model's cache floor can
never open a session warm. Twelve sessions of one confirm it: turn one cached exactly zero tokens
every time, against an unpruned control that opened warm seven times in eleven. It still billed
43.4% less per turn, which is why `smart-router qualify` warns about this case rather than
refusing it.

## What is not proven

The saving is measured on one agent, and that agent was built by this project to have a realistic
tool surface. The number most likely to move on yours is the cache behaviour, which depends on how
densely your sessions arrive.

The known limit is the menu pool. The fixed set of menus that keeps the prompt cache warm stops
covering the traffic somewhere past a hundred tools, and on the widest third-party surface it
recalls 0.23 where the bespoke prediction recalls 0.97. If your catalogue is that large you would
run the router without the pool, which is the 2.0% case rather than the 13.4% one.

## Re-running it

The A/B is two scripts and needs a published artifact, a BigQuery dataset and real money. See
[training.md](training.md) for how the artifact is produced and [operations.md](operations.md) for
the environment.

```bash
# Price it without calling anything.
uv run scripts/run_ab.py --sessions 600 --dry-run

# Drive the traffic. The arm split is the plugin's, not the script's.
uv run scripts/run_ab.py --sessions 600 --confirm

# Read the arms back out of the router's own provenance.
uv run scripts/report_ab.py --prefix ab-20260901-
```

For a fair prefix-cache comparison, run two sequential periods rather than one concurrent split.
`run_ab.py --help` gives the flags, and the reason is in its module docstring: two arms in one
process share a prompt cache, and the control's single prefix is warmed by traffic the routed arm
paid for.

The parts that need no cloud at all:

```bash
uv run pytest -q                      # 1,900 tests, no network, no credentials
uv run scripts/bench_overhead.py      # the latency budget above
uv run scripts/predict_saving.py      # what a menu is worth, priced from schemas
```

To compare Baseline vs. Cold-Start (`PrototypeJudge`) vs. Trained (`XGBoost`) vs. Hybrid
(`XGBoost + PrototypeJudge`) across the logged corpora:

```bash
uv run scripts/score_sets.py --compare-tiers
uv run scripts/score_sets.py --agent xlam --session-prefix xlam-c1- --compare-tiers
```

## Where each number comes from

| Claim | Produced by | Documented in |
| --- | --- | --- |
| −23.4% billed input per turn, −10.5% latency, +0.7% turns | `scripts/report_ab.py` over `scripts/run_ab.py` traffic | [scenarios.md#broad-support-agent](scenarios.md#broad-support-agent) |
| −22.5% of the whole bill per session | `scripts/report_ledger.py` over the same run, ledgered end to end | [scenarios.md#broad-support-agent](scenarios.md#broad-support-agent) |
| 32.1% measured and 42.4% bounded, repriced at menu warmth | `scripts/study_prefix_arrivals.py`, repriced against prefix residency | [README.md#status](../README.md#status) |
| 222.0–225.7 ms p95 added latency across `xgboost`, `prototype`, and `xgboost+prototype` | `scripts/bench_overhead.py` | [evidence.md#added-latency-22202257-ms-p95](#added-latency-22202257-ms-p95) |
| 0.9264 cold-start recall and 0.9755 / 0.8690 F8 hybrid recall on `support`, plus 0.9004 unseen-tool rescue | `scripts/score_sets.py --compare-tiers` | [README.md#status](../README.md#status) |
| 0.9894 cold-start recall and 0.9927 / 0.9972 hybrid recall on Salesforce `xlam` | `scripts/score_sets.py --agent xlam --session-prefix xlam-c1- --compare-tiers` | [scenarios.md#very-large-surface](scenarios.md#very-large-surface) |
| 214 ms p95 embedding call | `scripts/check_embedding_parity.py` | [training.md#checking-the-artifact-end-to-end](training.md#checking-the-artifact-end-to-end) |
| 0.9960 and 0.9666 set recall on 3,179 APIs, and the pool's 0.2298 against them | `scripts/score_xlam.py` | [scenarios.md#very-large-surface](scenarios.md#very-large-surface) |
| −13.4% and the answer-key cost | `scripts/run_xlam_traffic.py`, `scripts/score_xlam_answers.py` | [scenarios.md#very-large-surface](scenarios.md#very-large-surface) |
| −2.0% at p = 0.714, pruning per session instead | the same pair, second arm | [scenarios.md#very-large-surface](scenarios.md#very-large-surface) |
| 43.4% on an agent under the cache floor | `scripts/probe_cold_start.py` | [README.md#status](../README.md#status) |

The headline A/B was not the first run. The first live measurement failed in the wrong direction
(+25.5%) because pruning pushed the prompt below the model's context-cache floor
([scenarios.md#the-trap](scenarios.md#the-trap)). The second failed on prefix fragmentation when
every session built a bespoke tool subset ([architecture.md#menus-and-why-the-kept-set-is-not-bespoke](architecture.md#menus-and-why-the-kept-set-is-not-bespoke)).
The third uncovered a harness confound where two arms sharing one process shared a warm prompt
cache, and the sequential canonical-menu run with the floor guard, prefix lock, and cache watch in
place is what produced the −23.4% / −22.5% result above.
