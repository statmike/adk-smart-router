# `scripts/`

Operational and analysis scripts. Nothing here is imported by `src/adk_smart_router/`, and
nothing here ships in the wheel.

Two files here are libraries rather than scripts. `deployment.py` and `deployment.sh` hold the
answer to "which cloud deployment is this", read from the environment, so that no script carries a
project id and none of them has a fallback. Everything else in this directory is meant to be run,
and every one of them is in one of the four tables below.

## Standing the router up

The build order, once through. [Order to run things in](#order-to-run-things-in) below is the
same sequence as commands, and [`docs/training.md`](../docs/training.md) gives the reasoning for the order.

| Script | What it does | Where it sits in the build order |
| --- | --- | --- |
| `probe_cache_regime.py` | Preflight P5, the cache-regime probe. Two `generateContent` calls. | Task 0, before any infrastructure exists |
| `provision.sh` | Creates the dataset, the BigQuery connection and its IAM grant, the GCS bucket, and optionally loads the committed corpus. | Task 1 |
| `generate_corpus.py` | Generates the tool-list-blind corpus against the reference agent. | Task 3 |
| `deploy_sql.sh` | Substitutes the deploy-time placeholders and creates the stored procedures, or the dashboard views. `--dry-run` validates without creating, `--check` reports drift against the deployed copy. | Before task 6, and after every SQL edit |
| `check_embedding_parity.py` | Embeds prompts training already embedded and compares the vectors. The train/serve parity contract, checked against live endpoints. | Task 7, before any model is trusted |
| `score_candidate.py` | Scores a model that has been trained but not published, by predicting through BigQuery instead of an artifact. The same metrics as `score_sets.py`, so the two are comparable. | After task 9's training run, before publishing |
| `publish_router.py` | Turns an export into a loadable artifact: converts the booster, reconciles the label order against the exported assets, gates on menu recall and predicted saving, writes `metadata.json`. Takes `--settings` from `tune_settings.py`; explicit flags always win over the file. `--declaration-tokens` from `measure_menus.py --dump-tokens` is what makes the saving gate measurable. | Task 9, after a run passes its gate |
| `convert_booster.py` | Rewrites BQML's legacy `model.bst` as `model.ubj`. Runs under an old XGBoost in a throwaway environment; only `publish_router.py` calls it. | Inside task 9 |
| `check_router.py` | Scores the published artifact on held-out rows, through the real serving path, and compares it to the run record. | Task 10, before routing any traffic |
| `run_ab.py` | Drives live A/B traffic through the reference agent with the router installed. Assigns no arms and computes no metrics. | Task 11 |
| `report_ab.py` | Reads the arms back out of BigQuery provenance and measures G1. | Task 11, after `run_ab.py` |
| `report_ledger.py` | Prices the same run as a whole system: input and output per session, whether the router lengthened conversations, and what the tool-response channel is worth. Reproduces G1 first, so the two scripts cannot drift apart unnoticed. | Task 11, after `report_ab.py` |
| `deploy_agent_engine.sh` | Puts `examples/deployed_support` onto Agent Platform: the reference agent, the router staged as source, and a `.env` generated into a temp directory. `--list` and `--delete` manage what it creates. | Task 12, once the artifact is trusted |
| `agent_engines.py` | Finds, lists and deletes Agent Platform deployments by display name, which is the handle `adk deploy` gives them. The two scripts either side of this row both need the lookup, so it lives in one file rather than in both. | Called by the two scripts around it |
| `verify_agent_engine.py` | Drives that deployment and reads the routing provenance back out of the event stream, to answer whether the runtime's identity reaches Cloud Storage and the embedding endpoint, whether the build resolved a router that is not on PyPI, and whether one process serves more than one session. | Task 12, right after the deploy |
| `teardown.sh` | Destroys exactly what `provision.sh` created. | Last |

## Keeping it running

The loop that runs without a person in it. Two of these go on timers and the rest are called by
them. `docs/operations.md` has the scheduler recipe.

| Script | What it does | When it runs |
| --- | --- | --- |
| `check_drift.py` | Asks whether the published model still fits the surface it is serving, by counting how many declared tools it has no class for. Writes a `drift` row saying `due` or `ok` and retrains nothing. `--exit-code` is how `run_cycle.sh` reads the verdict, and `--backfill` re-fits the margin on the agent's own history. | Daily, inside `run_cycle.sh` |
| `run_cycle.sh` | Asks `check_drift.py` whether a retrain is due and runs `retrain.sh` when it is, under one cycle id, no closer together than seven days. The piece between a verdict and a retrain that used to be a person. | Daily, on a timer |
| `retrain.sh` | One retrain cycle, in order: train, tune, publish. Stops if the gate rejects the model, and keeps the previous settings if the tuner declines. | Whenever `run_cycle.sh` fires, or by hand |
| `tune_settings.py` | Chooses the menu size, menu count and miss price by pricing each one against held-out sessions, and prints the recall/saving frontier so the bar can be moved with its cost in view. Unlike `tune_menus.py` it assigns menus with the real published model and prices with the cache-aware cost model. | Inside `retrain.sh`, and any time the tool surface moves |
| `watch_generations.py` | Compares the live generation against its own concurrent holdout arm on billed tokens per later turn, and withdraws it if routing is costing more than not routing. Writes a ledger row every time it looks, including the times it does nothing. `--restore` undoes a withdrawal. | Daily, on its own timer beside `run_cycle.sh` |
| `record_step.py` | Writes one row into the retrain ledger from a shell, for the two steps no Python script here owns: BigQuery training, and whatever an adopter uses to put a new artifact in front of traffic. Always exits 0, because a ledger that can fail a retrain is a worse trade than a gap in a chart. `--ensure-table` creates the ledger. | Called by the shell scripts above |
| `fleet_cache_watch.py` | The in-process cache guard, asked of the whole fleet. `cache_watch` counts in memory, so a thinly routed agent never reaches a verdict in any single replica and a demotion earned in one does not reach the others. This imports the same rule and asks it of the warehouse rows every replica already writes, standardised by replica so the traffic mix cannot fake the verdict. | Any time an agent looks thinly routed |

## Measuring and tuning

Read-only, offline, and none of them changes what is served. Run them before quoting a number or
moving a setting.

| Script | What it does | When to run it |
| --- | --- | --- |
| `predict_saving.py` | Predicts the router's G1 from logged traffic, before an A/B is paid for. Sweeps the route fraction and reports the rollout and the A/B separately, because they are not the same number. | Before deciding to run an A/B |
| `study_prefix_arrivals.py` | Groups every turn-one call by how long its prefix had been idle, then reprices the routed arm at the control's arrival rate. The number behind the production-shape claim, from data already paid for. | After any A/B, before quoting a saving as if it were production |
| `bench_overhead.py` | Times the plugin's own code on the hot path, over the real reference agent and the published artifact, and reports G3 against its 250 ms bar. The embedding call is stubbed and added back from a flag. | After any change to the decide path, and before quoting G3 |
| `measure_menus.py` | Scores canonical menus and a stable core against the corpus for coverage, offline. Imports the shipped chooser, so it measures published code. | Before changing `menu_count` |
| `score_sets.py` | Scores the published artifact as a *set* predictor, on set recall and menu width, against two oracles. `--compare-tiers` compares `full list`, `cold-start (PrototypeJudge)`, `trained (XGBoost)`, and `hybrid (XGBoost + PrototypeJudge)` on `support` or `--agent xlam`, including an unseen-tool holdout stress test (`--unseen-holdout 15`). | After every retrain, or to benchmark cold-start and hybrid tiers |
| `tune_menus.py` | Sweeps menu size and count and reports each pool's ceiling: the best set recall any router could reach with it, and what that ceiling costs in saving. | Before changing `menu_count` or `top_n` |
| `sweep_menu_size.py` | Sweeps how wide a menu has to be, on third-party traffic, and reports two frontiers against each other: a fixed cap of N tools, and an adaptive width that keeps tools until their support covers `p`. The evidence for not fixing in advance how many tools a prediction is allowed to name. | Before defending `max_tools_kept` |
| `measure_mass_cut.py` | Measures the distance between the cut the gate scores and the cut the pruner makes. Four arms on held-out sessions with the shipped booster: the strict top-N the SQL reports, the mass rule that actually serves, that rule with ABSTAIN stripped first, and the whole served set with protected tools unioned back. | After a publish, when gate coverage and served coverage need reconciling |
| `check_calibration.py` | Asks whether `1 - contained_mass` is the miss probability the decline rule reads it as. Reports a reliability table, the calibration error, whether the score discriminates at all, and how many prunes the miscalibration would needlessly decline at miss prices above the shipped one. | After a publish |
| `measure_residency.py` | Measures how much of a prefix is still cached against how long it idled, and fits the residency curve `predict_saving.py` ships. Needs *routed* traffic: an unrouted agent has one prefix and it is always hot. | After an A/B has run |
| `score_unseen_intent.py` | Re-scores the corpus with the split grouped by intent, persona or opening text instead of by session, so the recall lift is measured against something the model has not already seen in another costume. | Before quoting any lift number |
| `score_menu_candidates.py` | Scores how the menu pool is built, padding each session out against merging co-occurring ones, on coverage and on cached-prefix reuse together. Prices both as billed turn-one prefix tokens across four arrival rates. | Before changing how menus are built |
| `study_session_length.py` | Asks whether a session property is worth predicting, by pricing what perfect knowledge of it would be worth to the decline rule. The answer for session length is no. | Before proposing any new per-session prediction |
| `price_prefetch.py` | Prices speculative execution without building it, so the v1.2 conversation can start from a number. Weighs top-1 accuracy against the latency a parallel call would hide, under the ceiling that only 25.2% of first calls have their arguments present in the message the router sees. | Before reopening prefetch |
| `red_team.py` | Attacks the threat model instead of reasoning about it. Adversarial first messages through the real artifact, the real embedder and the real `pruner.decide`, on five goals: keep a named tool, drop one, force a decline, evict a `never_prune` entry, forge a tool the host never declared. The last two are assertions rather than measurements, and the benign control is what makes the first two mean anything. | After any change to the decide path or to `never_prune` |
| `cost_model.py` | Emits the project's published cost tables. Local arithmetic, no cloud calls. | Any time |
| `check_install.sh` | Builds the wheel, installs it into an empty environment, and runs `smart-router qualify` and `doctor` against `examples/tiny_agent`. The only gate that does not point at the source tree. | CI runs it on every push |

## Probes and experiments

These already ran, and their conclusions are summarized in [`docs/evidence.md`](../docs/evidence.md)
and [`docs/scenarios.md`](../docs/scenarios.md). They are committed so the answers can be
re-derived rather than trusted, and so the next surprise has a harness to reuse.

| Script | What it does | What it settled |
| --- | --- | --- |
| `probe_shared_prefix.py` | Asks whether the implicit cache matches part of a tool list or all of it. Twelve calls. | The tool list is atomic: changing two declarations at the end invalidated the whole prefix. Stable core died here and canonical menus survived |
| `probe_directive.py` | Asks what the steering directive costs turn two of a session. Twelve calls at the default. | It roughly halves turn-two caching, and moving it to the other side of the user message recovers nothing |
| `probe_padding.py` | Asks whether padding a prompt over the model's cache floor makes it cache like content that earns its place. | It does, at the prefix level, and `probe_cold_start.py` shows it does not survive a session |
| `probe_prefix_headroom.py` | Sweeps shared-prefix sizes from under the model's published cache floor to four thousand tokens over it, to find how much margin a prefix needs before it is actually kept. | `qualify.CACHE_FLOOR_HEADROOM` |
| `probe_residency.py` | Measures what governs how long a pruned prefix stays in the implicit cache, when the warehouse says idle time does not. Separates prefix size, send volume, send cadence and a shared head, which move together on live traffic. | Size is partial and the other three are dead, which took three queued interventions off the table. What is left is time |
| `probe_cold_start.py` | Runs three arms of the same agent as real multi-turn sessions to measure what a prune under the cross-session floor costs end to end. | `qualify`'s cross-session warning |
| `score_xlam.py` | Scores the shipped rule on the 60,000 third-party queries of [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), over 3,179 real APIs, on four nested tool surfaces. | The rule generalises off our corpus, offline |
| `build_xlam_agent.py` | Builds an agent whose tools are somebody else's words: xLAM's own schemas for the 100 most-called tools, regenerated as stubs whose signatures and docstrings are xLAM's text, plus the 15,786 queries that fall entirely on that surface (`--fetch` downloads the dataset on demand from Hugging Face). | The fixture E1 needed to send a real request |
| `run_xlam_traffic.py` | Sends those queries through that agent, twice. `--route-fraction 0` writes the training corpus with the router off, because routing while generating labels would train the next model on its own guesses. `--route-fraction 1` runs the A/B. Computes no metric; `report_ab.py --agent xlam` does that. | E1: −13.4% of the whole bill on somebody else's tools and traffic |
| `score_xlam_answers.py` | Asks what that saving cost, against the answer key xLAM ships. Hit rate, recall and silence per arm, on the arms the router's own provenance assigns, so it cannot disagree with `report_ab.py` about who was where. | The only answer-quality measurement in the project, and it is why `min_menu_recall` stayed at 0.90 |
| `score_local_embedder.py` | Asks whether a CPU model on the box could replace the remote embedding call, the only network hop on the critical path. Swaps the vectors under `score_xlam.py`'s fixture and reads the recall, three ONNX candidates against the remote arm on identical cases. | The recall half of the answer, and it passed: `all-MiniLM-L6-v2` is under a point behind at every tool surface |
| `measure_embedder_footprint.py` | Asks what that model would cost a replica, which a corpus study cannot: cold-start load time, resident memory, and image growth. Every arm runs in a fresh subprocess, five per arm, because a warm process cannot report a cold start. | The cost half, and it decided it: 146 MB a replica and a 195 MB image layer, against 193 ms a session saved. The swap is declined and the API embeds on both sides |
| `simulate_drift.py` | Drives a real drift event, the reference agent's move from 25 tools to 100, and scores three label-free signals against it under three conditions: the old model on old traffic, on new traffic, and on new traffic restricted to the old surface. | Unseen share gates and confidence does not, which is `check_drift.py`'s threshold |
| `replay_cache_watch.py` | Streams twelve logged A/B windows back through the shipped `cache_watch` class in timestamp order, so the guard can be calibrated against a real 17.6% loss without paying for a second one. Sweeps `cooloff_calls` and prices a demotion as the counterfactual it is. | It fires five times and is right five times. The cool-off moves flapping and not detection, so 500 stays |
| `replay_promotion.py` | Replays a candidate automatic promoter over twelve windows with known outcomes and reports what it would have done. The decision comes from a bootstrapped interval over *sessions*, because sessions are the unit of randomisation and resampling turns would shrink the interval to a third of its honest width. | 200 sessions split evenly is enough to tell a real win from noise, and a promoter reading the router's own provenance is structurally blind to a sequential A/B |
| `simulate_censoring.py` | Asks whether the router poisons its own training data, and whether ε-greedy exploration would cost more than the bug. Prices the ratchet on, where labels stay uncensored and the corpus shrinks as the rollout ramps, against the ratchet off, where every session teaches and the routed ones teach the model to agree with itself. | The loop does not turn, so ε-greedy is declined rather than deferred. The ratchet is the expensive side, and `route_fraction` is the lever |

## Prerequisites

- The Google Cloud CLI, providing both `gcloud` and `bq`.
- `jq`. The connection's service agent is only available as a field inside `bq show` JSON.
- An active credential: `gcloud auth login` and `gcloud auth application-default login`.

Authentication is [Application Default
Credentials](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc) only. No
script here creates, downloads or
reads a service-account key, and none accepts `--key-file`. The scripts use whatever credentials
`gcloud` already has.

## Order to run things in

```bash
# 0. P5 first. It needs no corpus and no infrastructure, and for the price of two API
#    calls it tells you whether this model caches implicitly at all.
uv run python scripts/probe_cache_regime.py

# 1. Provision. Idempotent, so run it as often as you like.
./scripts/provision.sh

# 2. Generate the corpus (long wall clock, see the cost table below).
uv run python scripts/generate_corpus.py

# 3. Load the corpus by re-running provision. It picks up
#    examples/reference_agent/corpus/corpus.jsonl.gz if it exists.
./scripts/provision.sh

# 4. Deploy the stored procedures. --dry-run first; it validates and creates nothing.
./scripts/deploy_sql.sh --dry-run src/adk_smart_router/sql/train_router.sql
./scripts/deploy_sql.sh src/adk_smart_router/sql/train_router.sql

# ... preflight queries, then CALL train_router(...) ...

# 5. Check that serving embeds the way training did. Do this before trusting a model,
#    not after. Fractions of a cent.
uv run python scripts/check_embedding_parity.py

# 6. Score the candidate before promoting it. Publishing a model to find out whether
#    it is any good is how a bad gate promotes a bad model.
uv run python scripts/score_candidate.py

# 7. Publish. An export is not loadable as it stands: the booster is in a format
#    XGBoost no longer reads, and metadata.json does not exist yet.
uv run python scripts/publish_router.py --dry-run
uv run python scripts/publish_router.py --confirm

# 8. Score the published artifact end to end before routing anything to it.
#    --skip-embedding makes it free.
uv run python scripts/check_router.py

# 9. Run the live A/B and measure G1. Pilot a handful of sessions first and check
#    that provenance reached BigQuery before paying for the full run.
uv run python scripts/run_ab.py --dry-run
uv run python scripts/run_ab.py --confirm --limit 6
uv run python scripts/report_ab.py --prefix ab-20260901-
uv run python scripts/run_ab.py --confirm

# 10. Deploy the dashboard views once the router is live and logging. Free to create,
#     free to keep, and safe to redeploy. See docs/dashboard.md.
./scripts/deploy_sql.sh src/adk_smart_router/sql/dashboard.sql

# Last. Prints the plan, deletes nothing, and tells you what --confirm would do.
./scripts/teardown.sh
./scripts/teardown.sh --confirm
```

`provision.sh` checks before it creates at every step, so re-running is safe and reports what
already exists. The corpus load is the one step with a conditional: if `agent_events` already
holds rows it is skipped, because a second load would append duplicates. Set
`SMART_ROUTER_CORPUS_RELOAD=true` to replace the table contents instead.

## What `provision.sh` creates

| Resource | Default name | Notes |
| --- | --- | --- |
| [BigQuery](https://cloud.google.com/bigquery) dataset | `smart_router`, location `US` | Co-located with the connection and the embedding endpoint. ADK's [BigQuery Agent Analytics plugin](https://adk.dev/integrations/bigquery-agent-analytics/), BQAA below, creates `agent_events` and the `v_*` views on first write. |
| BigQuery connection | `smart_router_conn`, location `us` | `CLOUD_RESOURCE`. `AI.EMBED` requires it for the multimodal embedding endpoint. |
| IAM binding | `roles/aiplatform.user` on the connection's service agent | The step people forget. The role keeps its old id on [Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes), the platform formerly called Vertex AI. Without the binding, `AI.EMBED` fails at query time with a permission error that does not mention the connection. |
| [Cloud Storage](https://cloud.google.com/storage) bucket | `$SMART_ROUTER_GCS_BUCKET`, location `US` | Object versioning on, uniform bucket-level access on, public access prevention enforced. Router artifacts live under `routers/{agent}/{version}/`. |
| `agent_events` rows | from `examples/reference_agent/corpus/corpus.jsonl.gz` | Only if the file exists. Created day-partitioned on `timestamp` and clustered on `(event_type, agent, user_id)`, matching what BQAA writes, so the preflight and training queries prune partitions. |

Versioning is not decoration. It is what makes a bad router artifact roll back by generation, and
the model loader's `sha256` check assumes the previous generation is still there to fall back to.

If the corpus file is missing, `provision.sh` says so and provisions everything else. The corpus
is generated separately and committed, so a fresh clone has it and a first-ever run does not.

It is committed gzipped, at about 7 MB. `bq load` will not decompress a local `.gz`, so
`provision.sh` unpacks it to a temp file, loads that, and deletes it.

The export leaves out one event type. An `LLM_REQUEST` row carries the whole tool list serialized
into its attributes, which on the 100-tool surface is 94 KB per row and 89% of the export's bytes.
Nothing in this project reads them: the SQL reads `USER_MESSAGE_RECEIVED`, `TOOL_STARTING` and
`TOOL_COMPLETED`, and the A/B report reads BQAA's `v_llm_response` view. Keeping them would make
the artifact 197 MB, which GitHub refuses outright. What a fresh clone reloads is every row the
project actually queries; the full set stays in BigQuery.

## Environment variables

Every resource name has one. The project, dataset, connection and bucket variables are the
same ones the runtime package reads, so provisioning and serving cannot drift apart.

The project and the bucket have no default, and every script here refuses to run without them.
That is deliberate: a defaulted teardown deletes somebody else's dataset the first morning
somebody forgets an export. Set them once.

```bash
export SMART_ROUTER_BQ_PROJECT=your-gcp-project
export SMART_ROUTER_GCS_BUCKET=your-bucket
```

| Variable | Default | Used by |
| --- | --- | --- |
| `SMART_ROUTER_BQ_PROJECT` | none, required | both |
| `SMART_ROUTER_BQ_DATASET` | `smart_router` | both |
| `SMART_ROUTER_BQ_LOCATION` | `US` | provision |
| `SMART_ROUTER_BQ_CONNECTION_ID` | `smart_router_conn` | both |
| `SMART_ROUTER_BQ_CONNECTION_LOCATION` | `us` | both |
| `SMART_ROUTER_CONNECTION_ROLE` | `roles/aiplatform.user` | both |
| `SMART_ROUTER_GCS_BUCKET` | none, required | both |
| `SMART_ROUTER_GCS_LOCATION` | `US` | provision |
| `SMART_ROUTER_GCS_STORAGE_CLASS` | `STANDARD` | provision |
| `SMART_ROUTER_EVENTS_TABLE` | `agent_events` | provision |
| `SMART_ROUTER_CORPUS_FILE` | `examples/reference_agent/corpus/corpus.jsonl.gz` | provision |
| `SMART_ROUTER_CORPUS_RELOAD` | `false` | provision |

`provision.sh` prints the `export` lines for the runtime package at the end of every run,
including the fully qualified `SMART_ROUTER_BQ_CONNECTION` in `project.location.connection` form.

## Cost

| Script | Cost to run | Cost it leaves behind |
| --- | --- | --- |
| `probe_cache_regime.py` | Two `generateContent` calls on Flash. Fractions of a cent. | None. |
| `provision.sh` | Creating a dataset, a connection and a bucket is free. Batch loading the corpus is free. | Storage only. The corpus is about 110 MB in BigQuery, which is under a cent a month of active storage. An empty bucket costs nothing. |
| `generate_corpus.py` | About 5,400 Flash turns on the 100-tool surface, roughly **$45**. This is the one script with a real bill. | The committed `corpus.jsonl.gz`, about 7 MB, which is why it is committed rather than regenerated. |
| Preflight P1 to P4 | `AI.EMBED` over about 5,000 prompts plus a leave-one-out `VECTOR_SEARCH`. The embedding calls dominate; the bytes scanned are a couple of MB. | Nothing beyond the embedding table inside the dataset. |
| Training (`sql/train_router.sql`) | One BQML boosted-tree job per agent per cycle, plus the exported booster in GCS. About 8 minutes of wall clock on the reference corpus. | Artifacts in the bucket, every version of them, since versioning is on. |
| `check_embedding_parity.py` | Fifty embedding calls on a text endpoint. Fractions of a cent. | None. |
| `deploy_sql.sh` | Creating stored procedures and views is free, and so are `--dry-run`, `--print` and `--check`. | The views. A view stores nothing; it costs whatever the queries against it scan. |
| `publish_router.py` | A BigQuery metadata read and one booster download, about 200 KB. Fractions of a cent. | `model.ubj` (about 280 KB) and `metadata.json` per version. |
| `check_router.py` | One BigQuery read, plus one embedding call per sampled prompt. Fractions of a cent, or nothing at all with `--skip-embedding`. | None. |
| `run_ab.py` | About 600 Flash turns at the default 200 sessions, roughly **$1**. Half of them run routed, so they cost less than corpus turns. | `agent_events` rows under the `ab-` session prefix. Small, and worth keeping: they are the evidence behind G1. |
| `report_ab.py` | Two BigQuery reads over the `ab-` prefix. Fractions of a cent. | None. |
| `report_ledger.py` | Two BigQuery reads over the same prefix. No model traffic at all. Fractions of a cent. | None. |
| `study_prefix_arrivals.py` | Three BigQuery reads over the same prefix. No model traffic. Fractions of a cent. | None. |
| `bench_overhead.py` | One GCS read for the artifact. No BigQuery, no model traffic, no embedding calls. Free. | None. |
| `measure_menus.py` | One BigQuery read plus the declaration sizes. Fractions of a cent. | None. |
| `score_sets.py` | One BigQuery read plus the declaration sizes. No embedding calls: the vectors are the ones training already computed. Fractions of a cent. | None. |
| `score_candidate.py` | Two BigQuery reads, one of them an `ML.PREDICT` over a few thousand held-out rows, plus the declaration sizes. Fractions of a cent. | None. It writes nothing. |
| `tune_menus.py` | Two BigQuery reads plus the declaration sizes. No model and no embedding calls. Fractions of a cent. | None. |
| `tune_settings.py` | Three BigQuery reads and one `count_tokens` call, then arithmetic. No training and no model traffic. Fractions of a cent. | Nothing, unless `--out` is given, which writes one local JSON file. |
| `retrain.sh` | Whatever its three steps cost: one training run over the lookback window, then `tune_settings.py` and `publish_router.py`. A dry run trains nothing and is fractions of a cent. | A training run row, and on success one new artifact version. |
| `check_drift.py` | One BigQuery scan over the lookback window, which `--estimate` prices before it runs. Fractions of a cent on a corpus this size. | One `drift` ledger row per run. |
| `run_cycle.sh` | On most days, whatever `check_drift.py` costs plus one small ledger read, and nothing else. On the days it fires, a full `retrain.sh`. The minimum interval is what bounds how often that happens. | One ledger row when it holds off, and whatever the cycle it started leaves. |
| `watch_generations.py` | Two BigQuery reads over the ledger views, then arithmetic. No model traffic. Fractions of a cent, and it is meant to run on a timer. | One ledger row per run, and on a withdrawal one small marker object in the bucket. |
| `record_step.py` | One streaming insert. Fractions of a cent. | One ledger row, which is the whole point of it. |
| `red_team.py` | One embedding call per adversarial message, about eighty of them. Fractions of a cent. No BigQuery read. | None. |
| `score_xlam.py` | One embedding pass over 60,000 queries through the same `AI.EMBED` call training uses, about 1.6M tokens and roughly **$0.25**. | A cached vector table and a local `.npy`, which is why every script that reuses those vectors is free. |
| `run_xlam_traffic.py` | The two passes together cost **$17.69** at E1's sizes: $14.37 for 2,000 corpus sessions and $3.32 for the 599-session A/B, at roughly 2.5 model calls a session. It refuses to start without `--confirm`, prints a running estimate, and journals each finished session so a restart skips what was already paid for. | `agent_events` rows under the `xlam` agent, which are the evidence behind `docs/scenarios.md#very-large-surface`. |
| `probe_shared_prefix.py` | Twelve Flash calls of about 2.3k tokens. A few cents. | None. |
| `probe_directive.py` | Twelve Flash calls of about 2.5k tokens at the default, sixty at `--repeats 20`. A few cents either way. | None. |
| `probe_padding.py` | Twelve Flash calls with padded prefixes. A few cents, and the sixty-second gap is a free second reading. | None. |
| `probe_cold_start.py`, `probe_prefix_headroom.py`, `probe_residency.py` | Each one plans and prices its own run and calls nothing without `--confirm`. About $0.22 and three minutes, about $0.30 and four minutes, and $2.25 for `probe_residency.py`'s full 634 calls, which `--blocks` can cut down. | None. |
| `predict_saving.py`, `measure_residency.py`, `check_calibration.py`, `study_session_length.py`, `score_unseen_intent.py`, `measure_mass_cut.py`, `price_prefetch.py`, `score_xlam_answers.py`, `fleet_cache_watch.py`, `simulate_drift.py`, `simulate_censoring.py`, `replay_cache_watch.py`, `replay_promotion.py` | One or two BigQuery reads each, and for three of them a `count_tokens` call, then local arithmetic. No training, no model traffic, and no embedding calls: the vectors are the ones training already computed. Fractions of a cent. | None. They write nothing. |
| `cost_model.py`, `convert_booster.py`, `build_xlam_agent.py`, `sweep_menu_size.py`, `score_menu_candidates.py`, `score_local_embedder.py`, `check_install.sh` | No embedding spend and no model traffic. The three xLAM scorers read the vectors `score_xlam.py` cached: `sweep_menu_size.py` and `score_local_embedder.py` stop and say so if the local `.npy` is missing, and `score_menu_candidates.py` re-reads them from BigQuery instead. `score_local_embedder.py` downloads three ONNX models on first run, a few hundred MB of disk. `check_install.sh` builds a wheel into a temporary environment. | Nothing in the cloud. |
| `measure_embedder_footprint.py` | Free. No cloud calls and no credentials. It fetches one 87 MB model, into its own cache directory rather than the one `score_local_embedder.py` fills, and deletes that directory on every run to time the download. `--skip-download` leaves it alone. | Nothing, once the cache directory is removed. |
| `deploy_agent_engine.sh` | The deploy itself is a Cloud Build run, a few minutes and a few cents. Without `--confirm` it prints the plan and calls nothing. | **A running Agent Platform deployment, which bills until it is deleted.** This is the only script here that leaves compute running. `--delete --confirm` stops it. |
| `agent_engines.py` | One control-plane list call. Free. `delete` is the one subcommand that changes anything, and it is what stops a deployment billing. | Nothing. |
| `verify_agent_engine.py` | One Flash turn per session plus one embedding call per routed session, six sessions by default. A few cents. | `agent_events` rows under the `verify-` user prefix, which `train_router.sql` censors anyway because they are routed. |
| `teardown.sh` | Free. | Nothing. It is what stops the storage meters. It does not touch an Agent Platform deployment; that is `deploy_agent_engine.sh --delete`. |

Storage after a full pipeline run is still small: a few MB of BigQuery and a booster measured in
MB per version. The `max_artifact_bytes` ceiling in `RouterConfig` is 50 MB, so bucket growth is
bounded by versions retained, not by artifact size.

## What `teardown.sh` removes, and what it does not

It requires `--confirm`. Without it, it prints the exact list of resources it would destroy and
exits non-zero, having touched nothing. With `--confirm` it re-checks every resource afterwards
and exits non-zero if any survived, so a partial teardown cannot be mistaken for a clean one.

It refuses to run at all unless the names match the patterns this project provisions
(`^smart_router[a-z0-9_]*$` for the dataset and the connection,
`-smart-router` for the bucket). Those patterns are deliberately not configurable. Their job is
to be the thing a mistyped `SMART_ROUTER_BQ_DATASET` cannot get past.

**Removed:**

- The BigQuery dataset and everything in it: `agent_events`, the BQAA `v_*` views, the embedding
  tables, and any BQML model, because BQML models live inside the dataset.
- The `roles/aiplatform.user` binding on the connection's service agent. It is removed before the
  connection, while the member name can still be resolved. Delete the connection first and the
  binding is stranded pointing at a principal that no longer exists.
- The BigQuery connection.
- The GCS bucket, every live object in it, and **every noncurrent version**. Versioning is on, so
  the noncurrent generations have to be deleted explicitly (`gcloud storage rm --all-versions`)
  or the bucket delete fails on a bucket that lists as empty.
- **Trained router artifacts are inside that bucket, so they are removed with it.** Every version
  of every booster under `routers/{agent}/{version}/`, plus its metadata. There is no undo, no
  soft delete, and no copy anywhere else. If you want to keep a model, copy it out first.

**Not removed:**

- Enabled APIs. BigQuery, BigQuery Connection, Agent Platform and Cloud Storage stay enabled on
  the project. Disabling them would reach outside what `provision.sh` created.
- BigQuery job history and the metering records for jobs already run. Those are project-level and
  not deletable.
- Anything local: the on-disk artifact cache under `local_cache_dir` (default `/tmp/adk_routers`),
  and the committed corpus in the repository.
- Any other dataset, bucket, connection or IAM binding in the project, including ones whose names
  merely look similar. The name patterns and the explicit resource list are the only things it
  will act on.
- Your credentials. Nothing here touches `gcloud` config or ADC.

Nothing created on Agent Platform needs teardown: embedding calls are stateless, and the trained
model is a BQML model inside the dataset plus a booster inside the bucket, both covered above.
