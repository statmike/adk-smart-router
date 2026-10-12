-- dashboard.sql — the ten views a dashboard reads, and nothing else.
--
-- Everything a running router is worth is already in `agent_events`, in a JSON array under
-- `attributes.custom_metadata.smart_router`, alongside BQAA's own token counts. Getting a number
-- out of it takes an UNNEST, a JSON path and a join, and every script in `scripts/` writes that
-- join again from scratch. A dashboard cannot. Looker Studio, Looker and every other BI tool
-- want a table with columns, so this file is that table: ten views, each answering one
-- question, each safe to point a chart at directly.
--
-- WHAT EACH ONE ANSWERS
--   v_router_decisions              every routing decision, flattened. The base for the rest.
--   v_router_sessions               one row per session: which arm, how many turns, what it cost
--   v_router_cost_daily             billed-equivalent tokens per turn, by day and arm
--   v_router_g1_daily               the cost gate itself: routed against holdout, as a percentage
--   v_router_decision_sources_daily why sessions landed where they did, fail-opens included
--   v_router_coverage_weekly        coverage against the frequency baseline, from the backtest
--   v_router_tool_cache_daily       tool cache hit rate, per tool
--   v_router_stale_serves_daily     how often the cached answer had already moved
--   v_router_generations            one row per published version, and what it was worth
--   v_router_cycles                 one row per retrain cycle, with the steps pivoted
--
-- The first eight read what the router logged while serving. The last two read
-- `router_lifecycle`, the retrain ledger, and are empty until an agent has retrained once.
--
-- NO QUERY PARAMETERS. A view cannot take one, and a dashboard should not want it to: `agent`
-- and `day` are columns here, and the BI tool filters on them. That is also why nothing in this
-- file mentions a session prefix. `scripts/report_ab.py` filters by prefix because it reports on
-- one run; a dashboard reports on production, where every session counts.
--
-- COST. Every view that touches `agent_events` carries a 90-day floor on `timestamp`, which is
-- the partitioning column, so the worst a careless chart can do is scan a quarter. Widen it if
-- you keep more history than that and want to see it, but widen it on purpose: this table is the
-- largest thing in the dataset and a dashboard queries it on a refresh timer.
--
-- Deploy-time text substitution: `@project.@dataset` only. Everything else is a column.
--
--   ./scripts/deploy_sql.sh src/adk_smart_router/sql/dashboard.sql
--   ./scripts/deploy_sql.sh --check src/adk_smart_router/sql/dashboard.sql
--
-- BigQuery stores a view body verbatim, comments included, so `--check` compares each of the
-- ten against its deployed copy line by line. An edited view that was never redeployed leaves
-- the dashboard drawing numbers computed by SQL that is no longer in the repository, which is
-- the same fault a stale stored procedure causes and considerably harder to notice.
--
-- Views cost nothing to create and nothing to keep, so redeploying is always safe. Everything
-- runs under ADC.


-- ==================================================================================
-- 1. Every routing decision, flattened.
-- ==================================================================================
-- One row per decision record. `tool_name IS NULL` is the discriminator: cache-serve records
-- share this array with routing records and carry no `decision_source`, so without the filter
-- every count below would be inflated by cache traffic (provenance.py NOTE(contract)).
--
-- `decision_class` is the column most charts should group by. Thirteen sources is too many for a
-- pie chart and the distinctions that matter operationally are coarser: the router routed, the
-- router chose not to, a guard stopped it, or the embedder failed. `decision_source` is still
-- here for when the coarse answer is "declined" and the next question is which kind.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_decisions` AS
SELECT
  DATE(e.timestamp) AS day,
  e.timestamp,
  e.agent,
  e.session_id,
  e.user_id,
  JSON_VALUE(record, '$.decision_id') AS decision_id,
  JSON_VALUE(record, '$.decision_source') AS decision_source,
  JSON_VALUE(record, '$.mode') AS mode,
  JSON_VALUE(record, '$.model_version') AS model_version,
  JSON_VALUE(record, '$.model_digest') AS model_digest,
  JSON_VALUE(record, '$.top_label') AS top_label,
  SAFE_CAST(JSON_VALUE(record, '$.confidence') AS FLOAT64) AS confidence,
  ARRAY_LENGTH(JSON_QUERY_ARRAY(record, '$.kept_tools')) AS kept_tools,
  SAFE_CAST(JSON_VALUE(record, '$.cache_break') AS BOOL) AS cache_break,
  CASE JSON_VALUE(record, '$.decision_source')
    WHEN 'fresh' THEN 'routed'
    WHEN 'reused' THEN 'routed'
    WHEN 'holdout' THEN 'holdout'
    -- The embedder did not answer, so the session runs unrouted. This is the fail-open path and
    -- the three causes have different remedies: a slow endpoint, a raising one, and one the
    -- breaker has already given up on.
    WHEN 'error' THEN 'fail-open'
    WHEN 'timeout' THEN 'fail-open'
    WHEN 'breaker' THEN 'fail-open'
    -- A guard withheld a prune the router had already decided on. `floor` is per session and
    -- ordinary; `watch` is per agent and means routing has stopped for everyone.
    WHEN 'floor' THEN 'withheld'
    WHEN 'watch' THEN 'withheld'
    -- cheap, late, small, lowconf, noop, disabled. The router worked and said no.
    ELSE 'declined'
  END AS decision_class
FROM `@project.@dataset.agent_events` AS e,
  UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
WHERE e.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  AND JSON_VALUE(record, '$.tool_name') IS NULL;


-- ==================================================================================
-- 2. One row per session: which arm it was in, and what it cost.
-- ==================================================================================
-- The arm is decided from the router's own records rather than from a hash of the session id,
-- because the recorded answer is the one that was acted on and a recomputed one is a guess about
-- a config that may since have changed.
--
-- `neither` is not a rounding error and should be charted. A session lands there when the router
-- never routed it and never held it out: an embedding failure, a demoted agent, or a decline. It
-- belongs to no arm and including it in either one would bias the comparison.
--
-- `billed_input_tokens` is the quantity the cost gate is written in. Vertex bills a cached input
-- token at a tenth of an uncached one, so a raw prompt-token count credits pruning with savings
-- the invoice never shows. The cached count is clamped to the prompt count first: BQAA has been
-- seen to report the two out of order on a truncated response, and unclamped that turns into a
-- negative bill.
--
-- The `later_` columns are the same quantities with each session's FIRST model call dropped, and
-- they exist because a pooled average across the two arms is not a measurement of pruning. Turn
-- position dominates the price: a session's first call has no prefix of its own to hit and bills
-- about three times a continuation. The two arms do not have the same share of first calls,
-- because the router's own rules decide which sessions prune and therefore decide the mix. Over
-- the 2554 priced calls of DESIGN.md 18.22 the pooled figure came out 2.4% cheaper for pruning
-- while pruning was 25.9% worse on first calls and 18.4% worse on later ones. That is Simpson's
-- paradox, not noise, and no amount of extra sampling fixes it.
--
-- So: read the plain columns for what the two arms cost, which is a true statement about the
-- invoice, and the `later_` ones for whether pruning is what made the difference. Nothing
-- automatic is allowed to act on the first pair (DESIGN.md 21.5).
CREATE OR REPLACE VIEW `@project.@dataset.v_router_sessions` AS
WITH arms AS (
  SELECT
    agent,
    session_id,
    MIN(day) AS day,
    ANY_VALUE(user_id) AS user_id,
    CASE
      WHEN LOGICAL_OR(decision_source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(decision_source IN ('fresh', 'reused')) THEN 'routed'
      ELSE 'neither'
    END AS arm,
    ANY_VALUE(model_version) AS model_version,
    APPROX_TOP_COUNT(decision_source, 1)[SAFE_OFFSET(0)].value AS commonest_source,
    COUNTIF(decision_class = 'fail-open') AS fail_open_decisions,
    AVG(kept_tools) AS avg_kept_tools
  FROM `@project.@dataset.v_router_decisions`
  GROUP BY agent, session_id
),
calls AS (
  -- One row per priced model call, numbered within its session. `event_id` breaks a tie on
  -- `timestamp`, so the numbering is deterministic and two runs of this view agree.
  SELECT
    agent,
    session_id,
    COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
    LEAST(COALESCE(usage_cached_tokens, 0), COALESCE(usage_prompt_tokens, 0)) AS cached_tokens,
    COALESCE(usage_completion_tokens, 0) AS completion_tokens,
    COALESCE(total_ms, 0) AS total_ms,
    ROW_NUMBER() OVER (PARTITION BY agent, session_id ORDER BY timestamp, event_id) AS call_index
  FROM `@project.@dataset.v_llm_response`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND usage_prompt_tokens IS NOT NULL
),
turns AS (
  SELECT
    agent,
    session_id,
    COUNT(*) AS turns,
    SUM(prompt_tokens) AS prompt_tokens,
    SUM(cached_tokens) AS cached_tokens,
    SUM(completion_tokens) AS completion_tokens,
    SUM(total_ms) AS total_ms,
    -- A one-call session contributes 0 here, to the count and to the tokens alike, so it drops
    -- out of a later-turn ratio rather than dividing it by zero.
    COUNTIF(call_index > 1) AS later_turns,
    SUM(IF(call_index > 1, prompt_tokens, 0)) AS later_prompt_tokens,
    SUM(IF(call_index > 1, cached_tokens, 0)) AS later_cached_tokens
  FROM calls
  GROUP BY agent, session_id
)
SELECT
  arms.day,
  arms.agent,
  arms.session_id,
  arms.user_id,
  arms.arm,
  arms.model_version,
  arms.commonest_source,
  arms.fail_open_decisions,
  arms.avg_kept_tools,
  turns.turns,
  turns.prompt_tokens,
  turns.cached_tokens,
  turns.completion_tokens,
  turns.total_ms,
  (turns.prompt_tokens - turns.cached_tokens) + 0.10 * turns.cached_tokens AS billed_input_tokens,
  turns.later_turns,
  (turns.later_prompt_tokens - turns.later_cached_tokens) + 0.10 * turns.later_cached_tokens
    AS later_billed_input_tokens
FROM arms
-- INNER on purpose. A session with decisions and no LLM responses has no cost to report, and a
-- session with responses and no decisions had no router in it.
JOIN turns
  USING (agent, session_id);


-- ==================================================================================
-- 3. What a turn cost, by day and arm.
-- ==================================================================================
-- Per turn rather than per session throughout, because sessions differ in length between arms
-- for reasons that have nothing to do with routing, and a per-session number moves when they do.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_cost_daily` AS
SELECT
  day,
  agent,
  arm,
  COUNT(*) AS sessions,
  SUM(turns) AS turns,
  SAFE_DIVIDE(SUM(billed_input_tokens), SUM(turns)) AS billed_per_turn,
  SAFE_DIVIDE(SUM(prompt_tokens), SUM(turns)) AS prompt_per_turn,
  SAFE_DIVIDE(SUM(completion_tokens), SUM(turns)) AS output_per_turn,
  -- The prefix cache hit rate, which is a different cache from the tool cache in view 7. This is
  -- the share of the prompt Vertex served from its own context cache.
  SAFE_DIVIDE(SUM(cached_tokens), SUM(prompt_tokens)) AS cached_share,
  SAFE_DIVIDE(SUM(total_ms), SUM(turns)) AS ms_per_turn,
  AVG(avg_kept_tools) AS avg_kept_tools
FROM `@project.@dataset.v_router_sessions`
GROUP BY day, agent, arm;


-- ==================================================================================
-- 4. The cost gate: routed against holdout.
-- ==================================================================================
-- G1, the one number the project is judged on. Negative is cheaper.
--
-- Two ways this reads as a result when it is not, and both are columns here rather than
-- footnotes. `holdout_sessions` near zero means the percentage is dividing by noise. A rising
-- `unassigned_sessions` means sessions are leaving both arms, which is what a `cache_watch`
-- demotion looks like on the way past: the treatment arm quietly becomes a second control and
-- the ratio stays flat while the router does nothing at all.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_g1_daily` AS
SELECT
  day,
  agent,
  SUM(IF(arm = 'routed', sessions, 0)) AS routed_sessions,
  SUM(IF(arm = 'holdout', sessions, 0)) AS holdout_sessions,
  SUM(IF(arm = 'neither', sessions, 0)) AS unassigned_sessions,
  MAX(IF(arm = 'routed', billed_per_turn, NULL)) AS routed_billed_per_turn,
  MAX(IF(arm = 'holdout', billed_per_turn, NULL)) AS holdout_billed_per_turn,
  SAFE_DIVIDE(
    MAX(IF(arm = 'routed', billed_per_turn, NULL))
      - MAX(IF(arm = 'holdout', billed_per_turn, NULL)),
    MAX(IF(arm = 'holdout', billed_per_turn, NULL))
  ) AS g1,
  -- The gate is a 10% reduction. One day is far too small a sample to promote on, so read this
  -- as a daily reading of a number that is judged over a run, not as a daily verdict.
  SAFE_DIVIDE(
    MAX(IF(arm = 'routed', billed_per_turn, NULL))
      - MAX(IF(arm = 'holdout', billed_per_turn, NULL)),
    MAX(IF(arm = 'holdout', billed_per_turn, NULL))
  ) <= -0.10 AS meets_target
FROM `@project.@dataset.v_router_cost_daily`
GROUP BY day, agent;


-- ==================================================================================
-- 5. Why sessions landed where they did.
-- ==================================================================================
-- The health panel. A healthy routed agent is mostly `reused` with one `fresh` per session.
-- Anything in the `fail-open` class is an embedding endpoint problem rather than a router one,
-- and `watch` appearing at all means routing has stopped for every session on that agent.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_decision_sources_daily` AS
SELECT
  day,
  agent,
  decision_source,
  ANY_VALUE(decision_class) AS decision_class,
  COUNT(*) AS decisions,
  COUNT(DISTINCT session_id) AS sessions,
  AVG(kept_tools) AS avg_kept_tools,
  AVG(confidence) AS avg_confidence,
  SAFE_DIVIDE(COUNT(*), SUM(COUNT(*)) OVER (PARTITION BY day, agent)) AS share
FROM `@project.@dataset.v_router_decisions`
GROUP BY day, agent, decision_source;


-- ==================================================================================
-- 6. Coverage against the frequency baseline.
-- ==================================================================================
-- Straight from `router_backtest`, which `backtest.sql` writes. Backtests are re-run as the model
-- changes and every run appends, so a chart over the raw table draws one line per run on top of
-- itself. This view is the newest run for each agent, all of it and nothing else.
--
-- The newest RUN, not the newest row for each week, and the difference is not cosmetic. One run
-- reports every week in its window under one model and one set of arguments, so its weeks are a
-- trend. Stitching each week from whichever run touched it last produces a line whose points
-- were computed by different models against different corpus definitions, and it reads as a
-- trend too. Older runs are still in `router_backtest` if you want to compare two of them, but
-- comparing them is a deliberate act rather than the default a chart falls into.
--
-- Three columns decide whether the coverage number beside them means anything.
-- `corpus_source` separates synthetic rows from production ones and a gate met only on synthetic
-- traffic is not met. `overlaps_training` marks weeks the model was fitted on, where coverage is
-- an in-sample number and reads high. `unscored_share` is rows that had no persisted vector and
-- could not be scored at all; a large one means the chart is drawn on a fraction of the traffic.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_coverage_weekly` AS
WITH ranked AS (
  SELECT
    *,
    -- DENSE_RANK over run_at, so every row of the newest run ties at 1 and the run arrives
    -- whole. ROW_NUMBER here would keep exactly one row of it.
    DENSE_RANK() OVER (PARTITION BY agent ORDER BY run_at DESC) AS recency
  FROM `@project.@dataset.router_backtest`
)
SELECT
  week_start,
  agent,
  corpus_source,
  arm,
  run_id,
  run_at,
  top_n,
  n_labeled,
  n_scored,
  n_unscored,
  n_abstain,
  coverage_model,
  coverage_static,
  miss_reduction,
  overlaps_training,
  SAFE_DIVIDE(n_unscored, n_labeled) AS unscored_share
FROM ranked
WHERE recency = 1;


-- ==================================================================================
-- 7. Tool cache hit rate, per tool.
-- ==================================================================================
-- Two independent counts of the same thing, kept side by side on purpose. `serves` comes from the
-- router's own provenance and `from_cache` comes from the marker the cache writes into the tool
-- result that BQAA then logs. They should agree once probes are subtracted, because a probe is a
-- serve that ran the tool for real. When they do not, one of the two paths is dropping records,
-- and a hit rate computed from either alone would look fine while it happened.
--
-- `hit_rate` divides by every completion of that tool, cached and real, so it answers "how much
-- of this tool's demand did the cache absorb" rather than "how often did a lookup succeed".
CREATE OR REPLACE VIEW `@project.@dataset.v_router_tool_cache_daily` AS
WITH serves AS (
  SELECT
    DATE(e.timestamp) AS day,
    e.agent,
    JSON_VALUE(record, '$.tool_name') AS tool_name,
    COUNT(*) AS serves,
    COUNTIF(SAFE_CAST(JSON_VALUE(record, '$.probe') AS BOOL)) AS probes,
    AVG(SAFE_CAST(JSON_VALUE(record, '$.entry_age_s') AS FLOAT64)) AS avg_entry_age_s,
    MAX(SAFE_CAST(JSON_VALUE(record, '$.ttl_used') AS FLOAT64)) AS ttl_used_s,
    APPROX_TOP_COUNT(JSON_VALUE(record, '$.ttl_source'), 1)[SAFE_OFFSET(0)].value AS ttl_source
  FROM `@project.@dataset.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND JSON_VALUE(record, '$.tool_name') IS NOT NULL
  GROUP BY day, agent, tool_name
),
completions AS (
  -- The cache-hit marker is COUNTED here rather than filtered out, which is the whole point of
  -- this view and the one place in the project where admitting a cached row is correct. The
  -- spelling is the view's, `tool_result`, because the source is the view.
  SELECT
    DATE(timestamp) AS day,
    agent,
    tool_name,
    COUNT(*) AS completions,
    COUNTIF(JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL) AS from_cache
  FROM `@project.@dataset.v_tool_completed`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND status = 'OK'
    AND tool_name != 'transfer_to_agent'
  GROUP BY day, agent, tool_name
)
SELECT
  COALESCE(c.day, s.day) AS day,
  COALESCE(c.agent, s.agent) AS agent,
  COALESCE(c.tool_name, s.tool_name) AS tool_name,
  COALESCE(c.completions, 0) AS completions,
  COALESCE(c.from_cache, 0) AS from_cache,
  COALESCE(c.completions, 0) - COALESCE(c.from_cache, 0) AS ran_for_real,
  SAFE_DIVIDE(c.from_cache, c.completions) AS hit_rate,
  COALESCE(s.serves, 0) AS serves,
  COALESCE(s.probes, 0) AS probes,
  SAFE_DIVIDE(s.probes, s.serves) AS probe_rate,
  -- Zero when the two paths agree. Anything else is a logging fault, not a cache result.
  (COALESCE(s.serves, 0) - COALESCE(s.probes, 0)) - COALESCE(c.from_cache, 0) AS serve_marker_gap,
  s.avg_entry_age_s,
  s.ttl_used_s,
  s.ttl_source
FROM completions AS c
-- FULL, so a tool with demand and no cache entries and a tool with entries and no logged
-- completions both appear. Either one on its own is a finding.
FULL OUTER JOIN serves AS s
  ON c.day = s.day AND c.agent = s.agent AND c.tool_name = s.tool_name;


-- ==================================================================================
-- 8. How often the cached answer had already moved.
-- ==================================================================================
-- The refresh probe exists to catch a tool whose answer changes faster than its TTL, and this is
-- the readout. Nothing else in the system can see it: a cache hit returns the stored payload and
-- looks identical whether or not the world moved underneath it.
--
-- The measurement is indirect, because provenance never records tool arguments and so a probe
-- record cannot be joined to the call it caused. It does not need to be. A probe is a real call
-- made while an entry was still live, so the signature is visible in the tool rows alone: two
-- consecutive real calls to the same tool, by the same user, with the same arguments, less than
-- one TTL apart. If their results differ, the entry between them was stale and every hit it
-- served in that window returned the older answer.
--
-- Read `pairs_within_ttl` before `stale_rate`. A tool with no probes produces no pairs and a
-- silent zero would read as a clean bill of health.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_stale_serves_daily` AS
WITH completed AS (
  -- Numbered before anything is filtered. Dropping a cached row first would renumber the
  -- survivors on this side only, and the args side cannot drop the matching row because a
  -- TOOL_STARTING event carries no result and so no marker. The second real call would then be
  -- paired with the first call's arguments, which is the one error that manufactures a
  -- difference where there was none.
  SELECT
    timestamp,
    event_id,
    agent,
    user_id,
    invocation_id,
    status,
    JSON_VALUE(content, '$.tool') AS tool_name,
    IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), '<no-result-key>') AS result_json,
    JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL AS is_cache_hit,
    ROW_NUMBER() OVER (
      PARTITION BY invocation_id, JSON_VALUE(content, '$.tool') ORDER BY timestamp, event_id
    ) AS call_seq
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_COMPLETED'
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
),
started AS (
  -- The arguments half. No cache-hit predicate belongs here: a start row has no result for the
  -- marker to be in, so one would be a no-op wearing the look of a safeguard.
  SELECT
    invocation_id,
    JSON_VALUE(content, '$.tool') AS tool_name,
    JSON_QUERY(content, '$.args') AS args,
    ROW_NUMBER() OVER (
      PARTITION BY invocation_id, JSON_VALUE(content, '$.tool') ORDER BY timestamp, event_id
    ) AS call_seq
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_STARTING'
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
),
real_calls AS (
  SELECT
    c.timestamp,
    c.event_id,
    c.agent,
    c.user_id,
    c.tool_name,
    c.result_json,
    TO_HEX(SHA256(IFNULL(TO_JSON_STRING(s.args), '<no-args-key>'))) AS args_hash
  FROM completed AS c
  -- LEFT. A completion whose start row is missing keeps its result and lands in the no-args
  -- bucket with the other unknowns, which understates nothing. An INNER join would drop it and
  -- shrink the denominator without saying so.
  LEFT JOIN started AS s
    ON s.invocation_id = c.invocation_id
    AND s.tool_name = c.tool_name
    AND s.call_seq = c.call_seq
  WHERE c.status = 'OK'
    AND NOT c.is_cache_hit
),
-- One TTL per tool. `ttl_used` is recorded per entry and can move as the estimator learns, so
-- the largest one seen is the widest window in which a hit could have been served, and the
-- pessimistic choice.
ttl AS (
  SELECT
    e.agent,
    JSON_VALUE(record, '$.tool_name') AS tool_name,
    MAX(SAFE_CAST(JSON_VALUE(record, '$.ttl_used') AS FLOAT64)) AS ttl_s
  FROM `@project.@dataset.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND JSON_VALUE(record, '$.tool_name') IS NOT NULL
  GROUP BY agent, tool_name
),
pairs AS (
  SELECT
    DATE(r.timestamp) AS day,
    r.agent,
    r.tool_name,
    TIMESTAMP_DIFF(r.timestamp, LAG(r.timestamp) OVER w, MILLISECOND) / 1000.0 AS gap_s,
    r.result_json != LAG(r.result_json) OVER w AS changed,
    ttl.ttl_s
  FROM real_calls AS r
  JOIN ttl
    USING (agent, tool_name)
  -- The partition is the cache key: tool, user and arguments. user_id is unconditionally part of
  -- the key, so two users polling the same tool are two series rather than one interleaved one.
  -- event_id breaks timestamp ties, without which LAG picks arbitrarily between two calls in the
  -- same millisecond and the answer changes run to run on unchanged data.
  WINDOW w AS (
    PARTITION BY r.agent, r.tool_name, r.user_id, r.args_hash ORDER BY r.timestamp, r.event_id
  )
)
SELECT
  day,
  agent,
  tool_name,
  ANY_VALUE(ttl_s) AS ttl_s,
  COUNT(*) AS pairs_within_ttl,
  COUNTIF(changed) AS changed_within_ttl,
  SAFE_DIVIDE(COUNTIF(changed), COUNT(*)) AS stale_rate,
  AVG(gap_s) AS avg_gap_s
FROM pairs
WHERE gap_s IS NOT NULL
  AND gap_s <= ttl_s
GROUP BY day, agent, tool_name;


-- ==================================================================================
-- 9. One row per published version, and what it was worth.
-- ==================================================================================
-- Views 1 to 8 read the router's serving records. This one and the next read
-- `router_lifecycle`, the retrain ledger, which is what the training side of the project leaves
-- behind. A generation is one published artifact: what it was trained on, what the gates said
-- about it before it shipped, and what the traffic said about it afterwards, side by side.
--
-- THE JOIN TO TRAFFIC IS ASYMMETRIC, AND HAS TO BE. A routed session records the
-- `model_version` it ran under, so the routed arm is matched to a version exactly. A holdout
-- session records no version at all, because the point of holding it out is that no model
-- touched it. So the control arm is matched by *when*: the window from this version going live
-- to the next one going live. `routed_sessions` and `holdout_sessions` sit next to `g1` so the
-- difference in how each was gathered stays visible.
--
-- `live_from` is the `deploy` step when a deploy hook recorded one, and the `publish` step
-- otherwise. Publishing is not deploying, so on an agent that records no deploy the window opens
-- early and the first hours of it credit this version with the previous version's traffic.
-- `deploy_recorded` says which of the two you are reading. The visible symptom of a window that
-- does not line up is `first_served` before `live_from`, which means the version was already
-- serving when the window opened and the holdout sessions before that point went to the row above.
--
-- A version republished after a rollback collapses into one row starting at its first publish,
-- because serving records carry the version and not the window, and there is no honest way to
-- split one version's traffic across two windows.
CREATE OR REPLACE VIEW `@project.@dataset.v_router_generations` AS
WITH published AS (
  SELECT
    agent,
    cycle_id,
    version,
    run_id,
    environment,
    recorded_at AS published_at,
    reason AS publish_reason,
    settings_after AS settings
  FROM `@project.@dataset.router_lifecycle`
  WHERE step = 'publish'
    AND outcome = 'ok'
    AND version IS NOT NULL
    AND recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  -- The earliest publish row wins, and `cycle_id` breaks a tie on the timestamp. Two publishes
  -- of one version in the same instant is degenerate data, but without the tiebreaker the row
  -- this picks up, and so the `cycle_id` and `run_id` the whole generation is joined out on,
  -- could differ between two refreshes of the same chart.
  QUALIFY ROW_NUMBER() OVER (PARTITION BY agent, version ORDER BY recorded_at, cycle_id) = 1
),
deployed AS (
  SELECT agent, cycle_id, MIN(recorded_at) AS deployed_at
  FROM `@project.@dataset.router_lifecycle`
  WHERE step = 'deploy'
    AND outcome = 'ok'
    AND recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  GROUP BY agent, cycle_id
),
windows AS (
  SELECT
    p.*,
    d.deployed_at,
    d.deployed_at IS NOT NULL AS deploy_recorded,
    COALESCE(d.deployed_at, p.published_at) AS live_from,
    -- NULL on the newest generation, which is what `is_current` reads off below.
    LEAD(COALESCE(d.deployed_at, p.published_at)) OVER (
      PARTITION BY p.agent ORDER BY COALESCE(d.deployed_at, p.published_at)
    ) AS live_until
  FROM published AS p
  LEFT JOIN deployed AS d
    USING (agent, cycle_id)
),
-- One row per session, carrying both the version it recorded and the moment it started. The
-- decisions view supplies the timestamp, which `v_router_sessions` aggregates away into a date,
-- and a window that opens mid-afternoon needs better than a date.
sessions AS (
  SELECT
    agent,
    session_id,
    ANY_VALUE(model_version) AS model_version,
    MIN(timestamp) AS started_at
  FROM `@project.@dataset.v_router_decisions`
  GROUP BY agent, session_id
),
scored AS (
  SELECT
    s.agent,
    s.session_id,
    s.model_version,
    s.started_at,
    v.arm,
    v.turns,
    v.billed_input_tokens,
    v.later_turns,
    v.later_billed_input_tokens
  FROM sessions AS s
  JOIN `@project.@dataset.v_router_sessions` AS v
    USING (agent, session_id)
),
routed AS (
  SELECT
    agent,
    model_version AS version,
    COUNT(*) AS routed_sessions,
    SUM(turns) AS routed_turns,
    SAFE_DIVIDE(SUM(billed_input_tokens), SUM(turns)) AS routed_billed_per_turn,
    SUM(later_turns) AS routed_later_turns,
    SAFE_DIVIDE(SUM(later_billed_input_tokens), SUM(later_turns))
      AS routed_later_billed_per_turn,
    MIN(started_at) AS first_served,
    MAX(started_at) AS last_served
  FROM scored
  WHERE arm = 'routed'
    AND model_version IS NOT NULL
  GROUP BY agent, version
),
control AS (
  SELECT
    w.agent,
    w.version,
    COUNT(*) AS holdout_sessions,
    SUM(h.turns) AS holdout_turns,
    SAFE_DIVIDE(SUM(h.billed_input_tokens), SUM(h.turns)) AS holdout_billed_per_turn,
    SUM(h.later_turns) AS holdout_later_turns,
    SAFE_DIVIDE(SUM(h.later_billed_input_tokens), SUM(h.later_turns))
      AS holdout_later_billed_per_turn
  FROM windows AS w
  JOIN scored AS h
    ON h.agent = w.agent
    AND h.arm = 'holdout'
    AND h.started_at >= w.live_from
    AND (w.live_until IS NULL OR h.started_at < w.live_until)
  GROUP BY w.agent, w.version
)
SELECT
  w.agent,
  w.version,
  w.cycle_id,
  w.run_id,
  w.environment,
  w.published_at,
  w.deployed_at,
  w.deploy_recorded,
  w.live_from,
  w.live_until,
  w.live_until IS NULL AS is_current,
  TIMESTAMP_DIFF(COALESCE(w.live_until, CURRENT_TIMESTAMP()), w.live_from, HOUR) AS live_hours,
  w.publish_reason,
  w.settings,
  -- The training side. NULL here means the run aged out of `router_training_runs` or the publish
  -- row recorded no run id, not that the model was never trained.
  t.trained_at,
  t.corpus_since,
  t.n_rows,
  t.n_sessions,
  t.n_tool_classes,
  t.top_n,
  t.coverage,
  t.baseline_coverage,
  t.miss_reduction,
  t.macro_recall,
  t.abstain_precision,
  t.gate_passed,
  t.export_uri,
  -- The serving side. Zero sessions is the normal reading for a version published minutes ago.
  COALESCE(r.routed_sessions, 0) AS routed_sessions,
  COALESCE(r.routed_turns, 0) AS routed_turns,
  r.routed_billed_per_turn,
  r.first_served,
  r.last_served,
  COALESCE(c.holdout_sessions, 0) AS holdout_sessions,
  COALESCE(c.holdout_turns, 0) AS holdout_turns,
  c.holdout_billed_per_turn,
  -- NULL rather than zero when there was no holdout in the window, for the same reason
  -- `v_router_g1_daily` returns NULL: an A/B run sequentially has no control arm to divide by.
  SAFE_DIVIDE(
    r.routed_billed_per_turn - c.holdout_billed_per_turn,
    c.holdout_billed_per_turn
  ) AS g1,
  -- The same comparison with each session's first model call removed from both arms. `g1` above
  -- is a true statement about the invoice and a poor one about pruning, because the arms differ
  -- in how much of themselves is turn one and turn one costs about three times a continuation
  -- (DESIGN.md 18.22). This pair is the one to judge a generation on, and the one
  -- `watch_generations.py` reads. Comparing them is itself informative: a large gap between `g1`
  -- and `g1_later` means the arms' turn-one mix is doing the talking.
  COALESCE(r.routed_later_turns, 0) AS routed_later_turns,
  r.routed_later_billed_per_turn,
  COALESCE(c.holdout_later_turns, 0) AS holdout_later_turns,
  c.holdout_later_billed_per_turn,
  SAFE_DIVIDE(
    r.routed_later_billed_per_turn - c.holdout_later_billed_per_turn,
    c.holdout_later_billed_per_turn
  ) AS g1_later
FROM windows AS w
LEFT JOIN `@project.@dataset.router_training_runs` AS t
  ON t.agent = w.agent AND t.run_id = w.run_id
LEFT JOIN routed AS r
  ON r.agent = w.agent AND r.version = w.version
LEFT JOIN control AS c
  ON c.agent = w.agent AND c.version = w.version;


-- ==================================================================================
-- 10. One row per retrain cycle, with the steps pivoted.
-- ==================================================================================
-- The ledger is one row per step, which is right for writing and wrong for reading: a retrain
-- history filtered six ways is not a history. This pivots it, so a cycle is a row and the
-- question "what happened on the fourth" is a lookup.
--
-- `result` is the cycle's own verdict and distinguishes the ways a cycle can end without
-- publishing. A `drift check` is a single `check_drift.py` run that minted its own cycle id, and
-- is a complete and healthy thing. `held off` is `run_cycle.sh` finding a retrain due and the
-- last one too recent, which is the minimum interval working. `incomplete` means a cycle started
-- and the publish step never recorded an outcome, which usually means the runner died.
--
-- `tune` deserves its own reading. `declined` is the normal answer and means the sweep found
-- nothing that beat the settings already standing, so `settings_after` is NULL and the previous
-- shape carries. `shape_changed` is the column that says a retrain actually moved the router's
-- configuration, which is rarer than a retrain.
--
-- So does `withdraw`, which is the only step here nobody asked for. `watch_generations.py` runs on
-- a schedule and records one row every time it looks: `skipped` when there was not enough traffic
-- to judge, `declined` when the generation is fine, `ok` when it took one out of service. Charting
-- the three together is how you tell a quiet watch from one that is not running (DESIGN.md 21.5).
CREATE OR REPLACE VIEW `@project.@dataset.v_router_cycles` AS
SELECT
  cycle_id,
  agent,
  ANY_VALUE(environment) AS environment,
  MIN(recorded_at) AS started_at,
  MAX(recorded_at) AS ended_at,
  TIMESTAMP_DIFF(MAX(recorded_at), MIN(recorded_at), MINUTE) AS minutes,
  COUNT(*) AS steps_recorded,
  MAX(IF(step = 'train', run_id, NULL)) AS run_id,
  MAX(IF(step = 'publish', version, NULL)) AS published_version,
  MAX(IF(step = 'withdraw', version, NULL)) AS withdrawn_version,
  CASE
    WHEN COUNTIF(outcome = 'failed') > 0 THEN 'failed'
    WHEN COUNTIF(step = 'publish' AND outcome = 'ok') > 0 THEN 'published'
    WHEN COUNTIF(outcome = 'rejected') > 0 THEN 'rejected'
    -- Before `drift check`, because a withdrawal is the one verdict that undoes a deploy and it
    -- should never be filed under anything vaguer. Like a drift check it mints its own cycle id
    -- and is a complete thing on its own, so the same shape of test.
    WHEN COUNTIF(step = 'withdraw' AND outcome = 'ok') > 0 THEN 'withdrawn'
    WHEN COUNTIF(step = 'drift') = COUNT(*) THEN 'drift check'
    -- `run_cycle.sh` decided a retrain was due and then that it was too soon. A drift row and a
    -- skipped train and nothing else, which without this arm reads as `incomplete` and is the
    -- opposite: the runner did not die, it declined on purpose (DESIGN.md 21.6).
    WHEN COUNTIF(step = 'train' AND outcome = 'skipped') > 0
      AND COUNTIF(step IN ('drift', 'train')) = COUNT(*) THEN 'held off'
    ELSE 'incomplete'
  END AS result,
  -- The seven steps, in the order `retrain.sh` runs them, with the one it does not run last.
  -- NULL means the step did not run at all, which is different from every outcome it could
  -- have recorded.
  MAX(IF(step = 'train', outcome, NULL)) AS train,
  MAX(IF(step = 'tune', outcome, NULL)) AS tune,
  MAX(IF(step = 'publish', outcome, NULL)) AS publish,
  MAX(IF(step = 'backtest', outcome, NULL)) AS backtest,
  MAX(IF(step = 'drift', outcome, NULL)) AS drift,
  MAX(IF(step = 'deploy', outcome, NULL)) AS deploy,
  MAX(IF(step = 'withdraw', outcome, NULL)) AS withdraw,
  MAX(IF(step = 'train', reason, NULL)) AS train_reason,
  MAX(IF(step = 'tune', reason, NULL)) AS tune_reason,
  MAX(IF(step = 'publish', reason, NULL)) AS publish_reason,
  MAX(IF(step = 'drift', reason, NULL)) AS drift_reason,
  MAX(IF(step = 'withdraw', reason, NULL)) AS withdraw_reason,
  -- What the tuner was working from and what it left standing.
  MAX(IF(step = 'tune', settings_before, NULL)) AS settings_before,
  MAX(IF(step = 'tune', settings_after, NULL)) AS settings_after,
  MAX(IF(step = 'tune', settings_after, NULL)) IS NOT NULL
    AND MAX(IF(step = 'tune', settings_after, NULL))
      != IFNULL(MAX(IF(step = 'tune', settings_before, NULL)), '') AS shape_changed,
  -- The numbers each step measured, lifted out of the JSON so a chart can plot them. Everything
  -- else stays in `metrics` on the underlying table for whoever wants it.
  MAX(IF(step = 'publish',
    SAFE_CAST(JSON_VALUE(metrics, '$.menu_recall') AS FLOAT64), NULL)) AS menu_recall,
  MAX(IF(step = 'publish',
    SAFE_CAST(JSON_VALUE(metrics, '$.miss_reduction') AS FLOAT64), NULL)) AS miss_reduction,
  MAX(IF(step = 'publish',
    SAFE_CAST(JSON_VALUE(metrics, '$.predicted_g1') AS FLOAT64), NULL)) AS predicted_g1,
  MAX(IF(step = 'drift', JSON_VALUE(metrics, '$.signal'), NULL)) AS drift_signal,
  MAX(IF(step = 'drift',
    SAFE_CAST(JSON_VALUE(metrics, '$.unseen_share') AS FLOAT64), NULL)) AS unseen_share,
  MAX(IF(step = 'drift',
    SAFE_CAST(JSON_VALUE(metrics, '$.unseen_share_baseline') AS FLOAT64),
    NULL)) AS unseen_share_baseline,
  MAX(IF(step = 'drift',
    SAFE_CAST(JSON_VALUE(metrics, '$.confidence_p50') AS FLOAT64), NULL)) AS confidence_p50,
  -- What the watch saw. `withdraw_g1_later` is the number it acted on and the other two are the
  -- arms it came from, so a rollback can be argued with rather than only observed. A `skipped`
  -- withdraw row carries them too, which is how "it looked and left the model alone" is visible.
  MAX(IF(step = 'withdraw',
    SAFE_CAST(JSON_VALUE(metrics, '$.g1_later') AS FLOAT64), NULL)) AS withdraw_g1_later,
  MAX(IF(step = 'withdraw',
    SAFE_CAST(JSON_VALUE(metrics, '$.routed_later_billed_per_turn') AS FLOAT64),
    NULL)) AS withdraw_routed_billed_per_turn,
  MAX(IF(step = 'withdraw',
    SAFE_CAST(JSON_VALUE(metrics, '$.holdout_later_billed_per_turn') AS FLOAT64),
    NULL)) AS withdraw_holdout_billed_per_turn,
  -- Which script wrote which step. A cycle whose steps came from different sources is one
  -- somebody drove by hand, which is worth being able to see.
  STRING_AGG(DISTINCT source, ', ' ORDER BY source) AS sources
FROM `@project.@dataset.router_lifecycle`
WHERE recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY cycle_id, agent;
