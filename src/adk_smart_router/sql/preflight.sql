-- preflight.sql: P1, P2, P3, P4 (DESIGN 3.4). Four separate, individually runnable queries.
--
-- Any of these can end the project cheaply, which is the point: no plugin code, no training,
-- no artifacts exist yet. P5 is not here: it is a Python probe against generateContent
-- (scripts/probe_cache_regime.py) and it runs FIRST, because it needs no corpus at all.
--
-- Run order: P5 (elsewhere) -> P1 -> P2 -> P3 -> P4.
-- P2 must run before P3: P2 is P3's comparator, and P3 recomputes it internally so that the
-- lift is a difference over one shared denominator rather than two.
--
-- WHAT GATES WHAT
--   P1  gate         session-stable pruning (DESIGN 7)
--   P2  gate input   the static frequency baseline; DESIGN 3.5 turns a high one into a STOP
--   P3  gate         the entire ML stack
--   P4  NOT A GATE   pipeline test only (DESIGN 3.3). Its header says why, at length.
--
-- CORPUS PROVENANCE
-- Every query emits `corpus_source` and groups by it, so a synthetic run can never satisfy a
-- gate silently (DESIGN 3.3, R12). BQAA has no provenance column, so the discriminator is the
-- user id the corpus generator writes: scripts/generate_corpus.py sets
-- `user_id = f"corpus-{persona_id}"`, hence the default prefix `corpus-`. Pass the SAME value
-- to backtest.sql's `synthetic_user_prefix` argument, or the two reports will disagree about
-- which rows are synthetic.
--
-- DEPLOY-TIME TEXT SUBSTITUTION (BigQuery cannot parameterise an identifier)
--   @project      SMART_ROUTER_BQ_PROJECT      required, no default
--   @dataset      SMART_ROUTER_BQ_DATASET      default smart_router          (location US)
--   @location     SMART_ROUTER_BQ_CONNECTION_LOCATION  default us
--   @connection   SMART_ROUTER_BQ_CONNECTION_ID        default smart_router_conn
-- `@project.@location.@connection` is the AI.EMBED connection and matches the form
-- train_router.sql uses (DESIGN 9.3). The substituted project must equal @project.
--
-- RUNTIME QUERY PARAMETERS (real @-parameters, listed per query in each header)
--   @agent STRING                   the ADK agent name, verbatim
--   @since TIMESTAMP                corpus floor; also the partition-pruning predicate
--   @top_n INT64                    N. Pass RouterConfig.max_tools_kept (7), or these numbers
--                                   do not compare to the gate or to backtest.sql. The recorded
--                                   preflight run used 5, which is what max_tools_kept was then.
--   @k INT64                        P3 only: k-NN neighbours, 25 per DESIGN 3.4
--   @knn_overfetch INT64            P3 only: VECTOR_SEARCH top_k before same-session filtering
--   @synthetic_user_prefix STRING   '' calls everything production
--
-- Pipe the query in on stdin. Passing it as an argument does not work: each extract starts
-- with `-- Pn BEGIN`, and bq hands anything starting with `-` to its flag parser.
--
--   sed -n '/-- P1 BEGIN/,/-- P1 END/p' preflight.sql \
--     | sed -e 's/@project/your-project/g' -e 's/@dataset/smart_router/g' \
--     | bq --project_id=your-project --location=US query --use_legacy_sql=false \
--         --parameter='agent:STRING:support' \
--         --parameter='since:TIMESTAMP:2026-09-01 00:00:00' \
--         --parameter='top_n:INT64:7' \
--         --parameter='synthetic_user_prefix:STRING:corpus-'
--
-- P3 is two statements separated by a `;`. Run them one at a time so the embedding table can
-- be inspected before the k-NN reads it.
--
-- The `-- Pn BEGIN` / `-- Pn END` markers exist so one query can be extracted mechanically.
-- Everything runs under ADC. No service-account key is referenced anywhere.
--
-- SCHEMA FACTS THESE QUERIES DEPEND ON (DESIGN 18.2, verified against live datasets)
--   1. `v_user_message_received` has NO prompt text. The prompt is
--      JSON_VALUE(content, '$.text_summary') on the BASE `agent_events` table.
--   2. The base table stores the tool name at `$.tool` and the payload at `$.result`;
--      `v_tool_completed` RENAMES them to `tool_name` and `tool_result`. The cache-hit
--      exclusion predicate therefore differs by source, and the wrong spelling matches
--      nothing and fails OPEN (DESIGN 9.1, 11.5). Every use below names its source.
--   3. `agent_events` is DAY-partitioned on `timestamp` and clustered on
--      (event_type, agent, user_id), so every query filters on timestamp and agent, and the
--      base-table reads also filter on event_type to hit the leading cluster column.


-- =====================================================================================
-- P1 BEGIN: distinct tools per session
-- =====================================================================================
-- GATES: session-stable pruning (DESIGN 7, 3.4). If a session touches many distinct tools,
-- no menu small enough to be worth sending contains them all, and the session reaches for a
-- tool that is no longer declared. This gate used to be justified by growth events firing
-- constantly at two turns of cache apiece (DESIGN 18.1). Growth is gone: the router chooses
-- once and never widens, so a miss is not repaired, it is just a miss, and pruning saves
-- nothing on an agent whose sessions are this wide.
--
-- GO CONDITION: p90_distinct_tools <= @top_n (7) against the reference agent's 25 tools.
-- A p90 of 15 of 25 is a NO-GO.
--
-- TRUST: PARTIAL (DESIGN 3.3). Session length and topic drift are properties of our own
-- scenario design, so this is directionally useful rather than conclusive.
--
-- PARAMETERS: @agent, @since, @top_n, @synthetic_user_prefix
--
-- Cache hits are COUNTED, NOT FILTERED here, and that is deliberate. A cache hit is still a
-- tool the model chose and therefore still a tool that must be inside the kept set; removing
-- those rows would shrink the measured aperture and flatter the gate. The conservative number
-- (`p90_distinct_tools`) is the one the go condition reads. `p90_distinct_tools_excl_cache_hits`
-- is reported next to it so the gap is visible, and `n_cache_hit_calls` should be 0 at
-- preflight time because no router is running yet.
SELECT
  corpus_source,
  COUNT(*) AS n_sessions,
  SUM(n_calls) AS n_tool_calls,
  SUM(cache_hit_calls) AS n_cache_hit_calls,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(50)] AS p50_distinct_tools,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(90)] AS p90_distinct_tools,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(99)] AS p99_distinct_tools,
  MAX(distinct_tools) AS max_distinct_tools,
  APPROX_QUANTILES(distinct_tools_excl_cache_hits, 100)[SAFE_OFFSET(90)]
    AS p90_distinct_tools_excl_cache_hits,
  @top_n AS top_n,
  CASE
    WHEN APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(90)] > @top_n THEN 'NO-GO'
    WHEN corpus_source = 'production' THEN 'GO'
    ELSE 'GO (provisional: synthetic corpus, DESIGN 3.3/R12)'
  END AS gate_status
FROM (
  SELECT
    session_id,
    IF(
      @synthetic_user_prefix != ''
        AND STARTS_WITH(IFNULL(ANY_VALUE(user_id), ''), @synthetic_user_prefix),
      'synthetic',
      'production'
    ) AS corpus_source,
    COUNT(*) AS n_calls,
    COUNTIF(is_cache_hit) AS cache_hit_calls,
    COUNT(DISTINCT tool_name) AS distinct_tools,
    COUNT(DISTINCT IF(is_cache_hit, NULL, tool_name)) AS distinct_tools_excl_cache_hits
  FROM (
    -- SOURCE: v_tool_completed (a VIEW). The cache-hit marker is read from `tool_result`,
    -- the view's spelling (DESIGN 9.1). The base-table spelling
    -- JSON_VALUE(content, '$.result...') would match nothing here.
    SELECT
      session_id,
      user_id,
      tool_name,
      JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL AS is_cache_hit
    FROM `@project.@dataset.v_tool_completed`
    WHERE agent = @agent
      AND timestamp >= @since
      AND status = 'OK'
      -- ADK control plane, never a routed tool and never prunable in practice (DESIGN 7.2).
      AND tool_name != 'transfer_to_agent'
  )
  GROUP BY session_id
)
GROUP BY corpus_source
ORDER BY corpus_source;
-- P1 END


-- =====================================================================================
-- P2 BEGIN: static frequency baseline, coverage@N, global and per-user
-- =====================================================================================
-- GATES: nothing on its own. It is P3's comparator and it must run first, because the whole
-- ML stack is justified only by the DIFFERENCE between P3 and this number.
--
-- READ IT AS A STOP SIGN TOO: DESIGN 3.5: if coverage_static_global at N >= 0.95, the
-- correct product is a YAML file. Delete embedding.py, model_store.py, predictor.py, all the
-- training SQL and the parity contract, and ship static pruning plus the memo cache.
--
-- TRUST: GOOD, with a caveat (DESIGN 3.3). The labels come from what Gemini actually chose, so
-- the prompt->tool mapping is not ours to fake. The head-heaviness of the traffic, which is
-- exactly what this query measures, IS ours.
--
-- PARAMETERS: @agent, @since, @top_n, @synthetic_user_prefix
--
-- The per-user baseline is fitted IN SAMPLE: each user's top-N is chosen using the same rows
-- it is then scored on, so it is an upper bound on what a per-user YAML could really do. That
-- bias flatters the comparator and therefore biases P3's lift DOWNWARD, which is the safe
-- direction: the model still has to beat an inflated baseline.
WITH prompts AS (
  -- SOURCE: BASE agent_events. v_user_message_received has no prompt text (DESIGN 18.2), and
  -- the length filter has to see the text, so the base table is the only option.
  -- event_type + agent + timestamp keep this inside partition and cluster pruning.
  SELECT
    invocation_id,
    ANY_VALUE(session_id) AS session_id,
    ANY_VALUE(user_id) AS user_id
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'
    AND agent = @agent
    AND timestamp >= @since
    AND is_truncated = FALSE
    AND JSON_VALUE(content, '$.text_summary') IS NOT NULL
    AND LENGTH(JSON_VALUE(content, '$.text_summary')) BETWEEN 3 AND 4000
  GROUP BY invocation_id
),
first_tool AS (
  -- SOURCE: v_tool_completed (a VIEW). Cache-hit exclusion uses `tool_result` (DESIGN 9.1).
  SELECT
    invocation_id,
    ARRAY_AGG(tool_name ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS tool_name
  FROM `@project.@dataset.v_tool_completed`
  WHERE agent = @agent
    AND status = 'OK'
    AND timestamp >= @since
    AND tool_name != 'transfer_to_agent'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  GROUP BY invocation_id
),
dropped AS (
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.v_tool_error`
  WHERE agent = @agent AND timestamp >= @since
  UNION DISTINCT
  -- SOURCE: v_tool_completed (a VIEW), `tool_result` spelling. Cache-hit invocations are
  -- dropped WHOLE, not row by row: an invocation whose only tool came from cache would fall
  -- through the join and be labelled ABSTAIN, which would understate the baseline it is
  -- supposed to measure.
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.v_tool_completed`
  WHERE agent = @agent
    AND timestamp >= @since
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
  UNION DISTINCT
  -- SOURCE: BASE agent_events, `content.$.result` spelling. Not a duplicate of the branch
  -- above: under router-first registration a cache hit lands as an orphaned TOOL_COMPLETED
  -- that the views do not pair up (DESIGN 11.5, 18.1), and the base table is the only source
  -- that still carries it.
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_COMPLETED'
    AND agent = @agent
    AND timestamp >= @since
    AND JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL
),
labels AS (
  -- ABSTAIN rows are excluded, not counted as trivially covered. Coverage@N asks "is the true
  -- tool inside the kept set", which is undefined when there is no true tool; counting them as
  -- covered would lift every arm and shrink the difference P3 exists to measure.
  SELECT
    p.invocation_id,
    p.user_id,
    IF(
      @synthetic_user_prefix != ''
        AND STARTS_WITH(IFNULL(p.user_id, ''), @synthetic_user_prefix),
      'synthetic',
      'production'
    ) AS corpus_source,
    t.tool_name
  FROM prompts p
  JOIN first_tool t USING (invocation_id)
  WHERE p.invocation_id NOT IN (SELECT invocation_id FROM dropped)
),
grid AS (
  SELECT at_n FROM UNNEST(GENERATE_ARRAY(1, @top_n)) AS at_n
),
global_rank AS (
  SELECT
    corpus_source,
    tool_name,
    COUNT(*) AS n,
    ROW_NUMBER() OVER (
      PARTITION BY corpus_source ORDER BY COUNT(*) DESC, tool_name
    ) AS rk
  FROM labels
  GROUP BY corpus_source, tool_name
),
user_rank AS (
  SELECT
    corpus_source,
    user_id,
    tool_name,
    COUNT(*) AS n,
    ROW_NUMBER() OVER (
      PARTITION BY corpus_source, user_id ORDER BY COUNT(*) DESC, tool_name
    ) AS rk
  FROM labels
  GROUP BY corpus_source, user_id, tool_name
),
totals AS (
  SELECT
    corpus_source,
    COUNT(*) AS n_labeled,
    COUNT(DISTINCT user_id) AS n_users,
    COUNT(DISTINCT tool_name) AS n_tools_observed
  FROM labels
  GROUP BY corpus_source
),
cov_global AS (
  SELECT g.corpus_source, k.at_n, SUM(IF(g.rk <= k.at_n, g.n, 0)) AS covered
  FROM global_rank g CROSS JOIN grid k
  GROUP BY g.corpus_source, k.at_n
),
cov_user AS (
  SELECT u.corpus_source, k.at_n, SUM(IF(u.rk <= k.at_n, u.n, 0)) AS covered
  FROM user_rank u CROSS JOIN grid k
  GROUP BY u.corpus_source, k.at_n
)
SELECT
  cg.corpus_source,
  cg.at_n,
  t.n_labeled,
  t.n_users,
  t.n_tools_observed,
  SAFE_DIVIDE(cg.covered, t.n_labeled) AS coverage_static_global,
  SAFE_DIVIDE(cu.covered, t.n_labeled) AS coverage_static_per_user_insample,
  CASE
    WHEN cg.at_n != @top_n THEN NULL
    WHEN SAFE_DIVIDE(cg.covered, t.n_labeled) >= 0.95
      THEN 'STOP: static baseline >= 0.95 at N. DESIGN 3.5 says ship a YAML file.'
    ELSE 'Continue to P3. This row is P3 comparator.'
  END AS p2_signal
FROM cov_global cg
JOIN cov_user cu USING (corpus_source, at_n)
JOIN totals t USING (corpus_source)
ORDER BY cg.corpus_source, cg.at_n;
-- P2 END


-- =====================================================================================
-- P3 BEGIN: leave-one-out VECTOR_SEARCH k-NN over AI.EMBED'd prompts
-- =====================================================================================
-- GATES: the entire ML stack. This is the classifier's ceiling WITHOUT training a classifier.
-- If k-NN over the embeddings cannot keep the right tool in a top-N set, no boosted tree
-- trained on the same embeddings will either, and DESIGN 7/9 should not be written.
--
-- GO CONDITION (both, at at_n = @top_n):
--     coverage_knn >= 0.95   AND   coverage_knn - coverage_static >= 0.10
--
-- It also discharges build risk R3b as a side effect: a green P3 proves the AI.EMBED endpoint
-- string, dimension => 768, the BQ connection and its IAM all work.
--
-- TRUST: GOOD, with a caveat (DESIGN 3.3, R12). A green run authorises building. It does not
-- authorise the lift claim in README.md, which must come from production traffic or from the
-- permanent holdout arm (DESIGN 11.4).
--
-- PARAMETERS: @agent, @since, @top_n, @k, @knn_overfetch, @synthetic_user_prefix
--
-- TWO STATEMENTS, run in order. The embeddings are materialised because VECTOR_SEARCH needs a
-- base table, and because embedding the corpus once and searching it twice is the difference
-- between one AI.EMBED bill and two.
--
-- AI.EMBED facts (DESIGN 18.4, 9.3): it returns STRUCT<result ARRAY<FLOAT64>, status STRING>.
-- The field is `.result`, not `.embedding`, and status = '' (the EMPTY STRING) means success.
-- gemini-embedding-2 is a MULTIMODAL endpoint, so connection_id is REQUIRED and the width key
-- in model_params is `outputDimensionality`, the same name google-genai sends from
-- embedding.py. AI.EMBED IGNORES a model_params key it does not recognise instead of rejecting
-- it, so a misspelling returns the endpoint's native 3072-wide vector, status is still '', and
-- nothing anywhere says so. The width filter in statement 1 then drops every row and the
-- symptom is an empty table. Statement 2's guard turns that back into a message.
--
-- The model name is the train half of the parity contract (DESIGN 9.1, D1). It is written as a
-- literal here on purpose: it must match RouterConfig.embedding_model exactly, and a parameter
-- would let the two drift silently.

CREATE OR REPLACE TABLE `@project.@dataset.preflight_p3_embeddings`
CLUSTER BY corpus_source AS
WITH prompts AS (
  -- SOURCE: BASE agent_events. The prompt text lives at $.text_summary here and nowhere in
  -- v_user_message_received (DESIGN 18.2, 9.2).
  SELECT
    invocation_id,
    ANY_VALUE(session_id) AS session_id,
    ANY_VALUE(user_id) AS user_id,
    ANY_VALUE(JSON_VALUE(content, '$.text_summary')) AS prompt_text
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'
    AND agent = @agent
    AND timestamp >= @since
    AND is_truncated = FALSE
    AND JSON_VALUE(content, '$.text_summary') IS NOT NULL
    AND LENGTH(JSON_VALUE(content, '$.text_summary')) BETWEEN 3 AND 4000
  GROUP BY invocation_id
),
first_tool AS (
  -- SOURCE: v_tool_completed (a VIEW). Cache-hit exclusion uses `tool_result` (DESIGN 9.1).
  SELECT
    invocation_id,
    ARRAY_AGG(tool_name ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS tool_name
  FROM `@project.@dataset.v_tool_completed`
  WHERE agent = @agent
    AND status = 'OK'
    AND timestamp >= @since
    AND tool_name != 'transfer_to_agent'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
  GROUP BY invocation_id
),
dropped AS (
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.v_tool_error`
  WHERE agent = @agent AND timestamp >= @since
  UNION DISTINCT
  -- SOURCE: v_tool_completed (a VIEW), `tool_result` spelling.
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.v_tool_completed`
  WHERE agent = @agent
    AND timestamp >= @since
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
  UNION DISTINCT
  -- SOURCE: BASE agent_events, `content.$.result` spelling: the orphaned-event case
  -- (DESIGN 11.5, 18.1).
  SELECT DISTINCT invocation_id
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_COMPLETED'
    AND agent = @agent
    AND timestamp >= @since
    AND JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL
),
labeled AS (
  SELECT
    p.invocation_id,
    p.session_id,
    p.user_id,
    IF(
      @synthetic_user_prefix != ''
        AND STARTS_WITH(IFNULL(p.user_id, ''), @synthetic_user_prefix),
      'synthetic',
      'production'
    ) AS corpus_source,
    t.tool_name,
    p.prompt_text
  FROM prompts p
  JOIN first_tool t USING (invocation_id)
  WHERE p.invocation_id NOT IN (SELECT invocation_id FROM dropped)
),
embedded AS (
  SELECT
    invocation_id,
    session_id,
    user_id,
    corpus_source,
    tool_name,
    AI.EMBED(
      content => prompt_text,
      endpoint => 'gemini-embedding-2',
      connection_id => '@project.@location.@connection',
      model_params => JSON '{"outputDimensionality": 768}'
    ) AS e
  FROM labeled
)
SELECT
  @agent AS agent,
  invocation_id,
  session_id,
  user_id,
  corpus_source,
  tool_name,
  e.result AS v
FROM embedded
-- Empty string is success. A row that failed to embed is dropped here rather than carried as
-- a NULL vector, and the count difference against P2's n_labeled is the failure rate.
--
-- The WIDTH is deliberately not filtered here. A wrong width means every row is wrong, so
-- filtering would empty the table, and an empty table is the one shape statement 2's guard
-- cannot report on. Keeping the rows lets the guard read the width and name the problem.
WHERE e.status = '';

-- Statement 2: leave-one-out k-NN, and the static comparator recomputed over the SAME rows.
--
-- LEAVE-ONE-OUT IS AT SESSION GRANULARITY, NOT ROW GRANULARITY. Dropping only the self-match
-- would leave near-duplicate prompts from the same conversation in the neighbour set, and
-- session-stable routing produces many correlated rows per session, so the reported coverage
-- would be inflated exactly the way a RANDOM data split inflates training accuracy
-- (DESIGN 9.4). Excluding the whole session is the same correction applied earlier.
--
-- That filter runs AFTER VECTOR_SEARCH, so top_k must over-fetch: @knn_overfetch neighbours
-- are retrieved and the first @k survivors are kept. `n_rows_short_of_k` reports how often the
-- over-fetch was not enough; if it is not ~0, raise @knn_overfetch and re-run. If your
-- BigQuery release rejects a query parameter for top_k, inline the integer.
WITH emb AS (
  SELECT invocation_id, session_id, corpus_source, tool_name, v
  FROM `@project.@dataset.preflight_p3_embeddings`
  WHERE agent = @agent
),
guard AS (
  -- Two ways this table can be wrong, both of which otherwise produce a plausible-looking
  -- result rather than an error. Reading it is free: statement 1 already materialised it.
  --
  -- WRONG WIDTH: AI.EMBED ignored the model_params width key and returned the endpoint's
  -- native size. Coverage computed on those vectors would look fine and would not be the
  -- number serving reproduces, so this has to stop the query, not warn.
  --
  -- MIXED AGENTS: VECTOR_SEARCH's base argument must be a bare TABLE and cannot be filtered,
  -- so the query-side `agent` filter above cannot protect the BASE side. A table left over
  -- from another agent's run would silently supply neighbours from the wrong corpus.
  --
  -- Statement 1 uses CREATE OR REPLACE, so the fix for either is to re-run it.
  SELECT IF(
    COUNTIF(ARRAY_LENGTH(v) != 768) > 0,
    ERROR(FORMAT(
      'preflight_p3_embeddings holds %d vectors that are not 768 wide (widest %d). AI.EMBED '
      || 'ignores an unrecognised model_params key rather than rejecting it, so check that the '
      || 'width key reads outputDimensionality.',
      COUNTIF(ARRAY_LENGTH(v) != 768), MAX(ARRAY_LENGTH(v))
    )),
    IF(
      COUNT(DISTINCT agent) > 1,
      ERROR('preflight_p3_embeddings holds more than one agent. Re-run P3 statement 1.'),
      TRUE
    )
  ) AS one_agent_only
  FROM `@project.@dataset.preflight_p3_embeddings`
),
neighbors AS (
  SELECT
    query.invocation_id AS q_invocation_id,
    base.tool_name AS neighbor_tool,
    distance,
    ROW_NUMBER() OVER (PARTITION BY query.invocation_id ORDER BY distance) AS nn_rank
  FROM VECTOR_SEARCH(
    TABLE `@project.@dataset.preflight_p3_embeddings`,
    'v',
    (SELECT invocation_id, session_id, v FROM emb),
    query_column_to_search => 'v',
    top_k => @knn_overfetch,
    distance_type => 'COSINE'
  )
  WHERE base.session_id != query.session_id
),
knn AS (
  SELECT * FROM neighbors WHERE nn_rank <= @k
),
neighbor_counts AS (
  SELECT q_invocation_id, COUNT(*) AS n_neighbors FROM knn GROUP BY q_invocation_id
),
votes AS (
  -- Similarity-weighted vote. distance_type is COSINE, so 1 - distance is cosine similarity
  -- and a nearer neighbour counts for more. Ties break on raw vote count, then on name, so
  -- the ranking is deterministic and the query is re-runnable.
  SELECT
    q_invocation_id,
    neighbor_tool,
    SUM(1 - distance) AS weight,
    COUNT(*) AS n_votes
  FROM knn
  GROUP BY q_invocation_id, neighbor_tool
),
ranked AS (
  SELECT
    q_invocation_id,
    neighbor_tool,
    ROW_NUMBER() OVER (
      PARTITION BY q_invocation_id ORDER BY weight DESC, n_votes DESC, neighbor_tool
    ) AS rk
  FROM votes
),
static_rank AS (
  -- The comparator, recomputed here rather than read from P2. Both coverages must come from
  -- one denominator or their difference is not a lift, and only the rows that survived
  -- AI.EMBED are in this table.
  SELECT
    corpus_source,
    tool_name,
    ROW_NUMBER() OVER (
      PARTITION BY corpus_source ORDER BY COUNT(*) DESC, tool_name
    ) AS rk
  FROM emb
  GROUP BY corpus_source, tool_name
),
grid AS (
  SELECT at_n FROM UNNEST(GENERATE_ARRAY(1, @top_n)) AS at_n
),
per_row AS (
  -- One row per (invocation, at_n). Rows whose neighbours were all same-session survive with
  -- covered_knn = FALSE rather than vanishing: dropping them would remove the hardest cases
  -- from the denominator.
  SELECT
    e.corpus_source,
    g.at_n,
    e.invocation_id,
    ANY_VALUE(IFNULL(nc.n_neighbors, 0)) AS n_neighbors,
    IFNULL(ANY_VALUE(sr.rk) <= g.at_n, FALSE) AS covered_static,
    LOGICAL_OR(IFNULL(r.neighbor_tool = e.tool_name AND r.rk <= g.at_n, FALSE)) AS covered_knn
  FROM emb e
  CROSS JOIN grid g
  CROSS JOIN guard
  LEFT JOIN neighbor_counts nc ON nc.q_invocation_id = e.invocation_id
  LEFT JOIN ranked r ON r.q_invocation_id = e.invocation_id
  LEFT JOIN static_rank sr
    ON sr.corpus_source = e.corpus_source AND sr.tool_name = e.tool_name
  -- Referenced, not merely joined, so the assertion cannot be optimised away.
  WHERE guard.one_agent_only
  GROUP BY e.corpus_source, g.at_n, e.invocation_id
)
SELECT
  corpus_source,
  at_n,
  COUNT(*) AS n_rows,
  @k AS k,
  COUNTIF(n_neighbors < @k) AS n_rows_short_of_k,
  AVG(n_neighbors) AS avg_neighbors,
  AVG(IF(covered_knn, 1.0, 0.0)) AS coverage_knn,
  AVG(IF(covered_static, 1.0, 0.0)) AS coverage_static,
  AVG(IF(covered_knn, 1.0, 0.0)) - AVG(IF(covered_static, 1.0, 0.0)) AS lift,
  CASE
    WHEN at_n != @top_n THEN NULL
    WHEN AVG(IF(covered_knn, 1.0, 0.0)) < 0.95
      THEN 'NO-GO: coverage_knn < 0.95. Top-N pruning is too lossy by any method (DESIGN 3.5).'
    WHEN AVG(IF(covered_knn, 1.0, 0.0)) - AVG(IF(covered_static, 1.0, 0.0)) < 0.02
      THEN 'NO-GO: lift < 0.02. Ship static pruning; delete the ML stack (DESIGN 3.5).'
    WHEN AVG(IF(covered_knn, 1.0, 0.0)) - AVG(IF(covered_static, 1.0, 0.0)) < 0.10
      THEN 'AMBER: lift in [0.02, 0.10). Static pruning in v1.0, ML in v1.1 (DESIGN 3.5).'
    WHEN corpus_source = 'production' THEN 'GO'
    ELSE 'GO (provisional: synthetic corpus, DESIGN 3.3/R12)'
  END AS gate_status
FROM per_row
GROUP BY corpus_source, at_n
ORDER BY corpus_source, at_n;
-- P3 END


-- =====================================================================================
-- P4 BEGIN: per-tool repeat rate, identical rate, inter-call interval, stability window
-- =====================================================================================
-- THIS IS NOT A GATE. It is a pipeline test (DESIGN 3.3, 3.4).
--
-- WHY NOT: repetition is a property of the traffic WE author. Unlike the labels in P2/P3, which
-- come from what Gemini independently chose and so cannot be faked by writing scenarios, a
-- repeat rate passes through no independent decision at all. A generator that repeats gives
-- DESIGN 8 a green light we wrote ourselves. So this query proves that tool_ttl_report.sql runs
-- and that its columns mean what they say. It does not justify the cache, and no number here
-- may be written into `cacheable_tools`.
--
-- DESIGN 8 is justified on first principles instead: it is opt-in, human-declared, and removes
-- a measured tool round-trip. MEASURED TTLs STAY BLOCKED ON REAL TRAFFIC, permanently.
--
-- PARAMETERS: @agent, @since, @synthetic_user_prefix
--
-- SOURCE: BASE agent_events, and TWO event types, mirroring tool_ttl_report.sql exactly. BQAA
-- splits one tool call across them: TOOL_STARTING carries `content.$.args` and TOOL_COMPLETED
-- carries `content.$.result`. Neither row has both, and grouping repeats needs both, so they
-- are paired below. v_tool_completed is no help: its SELECT list lifts `$.tool` and `$.result`
-- out of `content` and drops the rest, so `$.args` is unreachable through the view.
-- Consequences of reading the base table:
--   * the cache-hit exclusion must be the BASE spelling,
--     JSON_VALUE(content, '$.result._smart_router.cache_hit'), annotated at its use below,
--   * latency is CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64). The bare name `total_ms`
--     is the view's, not the table's, and there is no base column by that name,
--   * `event_type` is also the leading cluster column, so this is cheaper than the equivalent
--     read through the view.
--
-- THE PAIRING IS POSITIONAL. There is no `attributes.adk.function_call_id` on either row, so
-- within one invocation and one tool name the nth start is taken as the nth completion. Exact
-- for sequential calls; two concurrent calls to the same tool in one invocation could swap two
-- args hashes. That moves a call between buckets, it does not invent a repeat.
-- `args_present_rate` reports whether the pairing found args at all. 0.0 means every call for
-- a tool collapsed into one bucket, which fires the poller detector more often and depresses
-- identical_rate, both pushing toward "do not cache" and so failing in the safe direction.
WITH completed AS (
  -- The result half, plus the sequence number the args half is matched on.
  --
  -- NUMBER FIRST, FILTER LATER. Both the cache-hit marker and `status` are read here and acted
  -- on in `calls` below, not in this WHERE clause. Dropping a row before ROW_NUMBER renumbers
  -- the survivors, and only this side would renumber: a cache hit is invisible on a
  -- TOOL_STARTING row, so `started` cannot drop the matching one. An invocation whose first
  -- call was served from cache would then pair its second, real completion with its FIRST
  -- call's arguments. Wrong args on a real result is the one error that inflates identical_rate
  -- rather than depressing it, so it is the one that had to be designed out.
  SELECT
    invocation_id,
    event_id,
    user_id,
    timestamp,
    status,
    CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms,
    JSON_VALUE(content, '$.tool') AS tool_name,
    -- Never NULL, so the run-boundary and identity comparisons below stay total.
    IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), '<no-result-key>') AS result_json,
    -- SOURCE: BASE agent_events. This is the base spelling of the cache-hit marker (DESIGN 9.1).
    -- Written as the view's `tool_result` spelling it would match nothing and fail OPEN, and a
    -- cached payload is CONSTANT BY CONSTRUCTION, so admitting one inflates identical_rate and
    -- the stability window without bound (DESIGN 11.5).
    JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL AS is_cache_hit,
    ROW_NUMBER() OVER (
      PARTITION BY invocation_id, JSON_VALUE(content, '$.tool') ORDER BY timestamp, event_id
    ) AS call_seq
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_COMPLETED'
    AND agent = @agent
    AND timestamp >= @since
    -- Safe in the WHERE because `started` applies the identical predicate, so both sides
    -- number the same set of calls.
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
),
started AS (
  -- The args half. Same partition and same ORDER BY as `completed`, which is what makes the two
  -- sequence numbers line up. event_id breaks timestamp ties so both sides agree on order.
  --
  -- NO CACHE-HIT EXCLUSION HERE, and it is not an omission. A start row carries `$.args` and no
  -- `$.result`, so the marker cannot be present on it and the predicate would be a no-op that
  -- reads as a safeguard. The exclusion lives in `calls`, on the result side, where the marker
  -- actually is.
  SELECT
    invocation_id,
    JSON_VALUE(content, '$.tool') AS tool_name,
    JSON_QUERY(content, '$.args') AS args,
    ROW_NUMBER() OVER (
      PARTITION BY invocation_id, JSON_VALUE(content, '$.tool') ORDER BY timestamp, event_id
    ) AS call_seq
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_STARTING'
    AND agent = @agent
    AND timestamp >= @since
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
),
calls AS (
  SELECT
    c.user_id,
    c.timestamp,
    -- Carried only to break timestamp ties in `seq`. See the WINDOW clause there.
    c.event_id,
    c.total_ms,
    c.tool_name,
    c.result_json,
    TO_HEX(SHA256(IFNULL(TO_JSON_STRING(s.args), '<no-args-key>'))) AS args_hash,
    s.args IS NOT NULL AS args_present,
    IF(
      @synthetic_user_prefix != ''
        AND STARTS_WITH(IFNULL(c.user_id, ''), @synthetic_user_prefix),
      'synthetic',
      'production'
    ) AS corpus_source
  FROM completed c
  -- LEFT, not INNER. A completion whose start is missing keeps its result and falls into the
  -- '<no-args-key>' bucket, the pessimistic side. An INNER JOIN would drop the row and quietly
  -- shrink the denominator instead.
  LEFT JOIN started s
    ON s.invocation_id = c.invocation_id
    AND s.tool_name = c.tool_name
    AND s.call_seq = c.call_seq
  -- The two filters deferred out of `completed` so that call_seq could be assigned first.
  WHERE c.status = 'OK'
    AND NOT c.is_cache_hit
),
seq AS (
  -- The partition IS the cache key minus the constants: (tool, user, args). user_id is in it
  -- because it is unconditionally in the key (DESIGN 8.1), so two users polling the same tool
  -- are two series, not one interleaved one.
  SELECT
    corpus_source,
    tool_name,
    user_id,
    args_hash,
    args_present,
    timestamp,
    event_id,
    total_ms,
    result_json,
    LAG(timestamp) OVER w AS prev_ts,
    LAG(result_json) OVER w AS prev_result
  FROM calls
  -- ORDER BY timestamp ALONE IS NOT ENOUGH. Two calls in one series can share a millisecond, and
  -- when they do, LAG picks between them arbitrarily and identical_rate changes run to run for
  -- the same data. This corpus happens to contain no such tie, which is exactly why the tiebreak
  -- has to be written down now rather than after a run disagrees with itself.
  WINDOW w AS (
    PARTITION BY corpus_source, tool_name, user_id, args_hash ORDER BY timestamp, event_id
  )
),
pairs AS (
  SELECT
    *,
    prev_ts IS NOT NULL AS has_prev,
    TIMESTAMP_DIFF(timestamp, prev_ts, MILLISECOND) / 1000.0 AS gap_s,
    prev_ts IS NOT NULL AND result_json = prev_result AS identical_to_prev,
    IF(prev_result IS NULL OR result_json != prev_result, 1, 0) AS is_run_start
  FROM seq
),
runs AS (
  SELECT
    *,
    -- Same order as `seq`'s window, tiebreak included. A run boundary computed under one order
    -- and accumulated under another would split runs that never ended.
    SUM(is_run_start) OVER (
      PARTITION BY corpus_source, tool_name, user_id, args_hash
      ORDER BY timestamp, event_id
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS run_id
  FROM pairs
),
run_spans AS (
  -- A run is a maximal streak of byte-identical consecutive results. Its span is the time we
  -- OBSERVED the value hold. Single-observation runs contribute 0 seconds, which is correct:
  -- one observation demonstrates no stability at all. Keeping them shortens the reported
  -- window, which is the conservative direction.
  SELECT
    corpus_source,
    tool_name,
    TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), MILLISECOND) / 1000.0 AS stable_s
  FROM runs
  GROUP BY corpus_source, tool_name, user_id, args_hash, run_id
),
poller_pairs AS (
  -- STRUCTURAL poller detection (DESIGN 8.6). The regex is deliberately depth-blind: it finds
  -- `"key": value` for the reserved key set anywhere in the serialised result, so a status
  -- buried under `$.result.job.state` is caught too.
  SELECT
    corpus_source,
    tool_name,
    user_id,
    args_hash,
    LOWER(REGEXP_EXTRACT(kv, r'^"([A-Za-z_]+)"')) AS poll_key,
    kv
  FROM calls,
    UNNEST(REGEXP_EXTRACT_ALL(
      result_json,
      r'(?i)("(?:state|status|progress|pct|percent|eta|remaining|phase)"\s*:\s*(?:"(?:[^"\\]|\\.)*"|true|false|null|-?[0-9][0-9.eE+\-]*))'
    )) AS kv
),
pollers AS (
  -- Cardinality > 1 for a fixed args hash, per key, per user. A PENDING->PENDING->RUNNING->DONE
  -- run produces a perfectly plausible stability window, which is exactly why a statistical
  -- detector gives the wrong answer here and this one does not look at timing at all.
  SELECT corpus_source, tool_name, STRING_AGG(DISTINCT poll_key ORDER BY poll_key) AS poller_keys
  FROM (
    SELECT corpus_source, tool_name, user_id, args_hash, poll_key
    FROM poller_pairs
    GROUP BY corpus_source, tool_name, user_id, args_hash, poll_key
    HAVING COUNT(DISTINCT kv) > 1
  )
  GROUP BY corpus_source, tool_name
),
per_tool AS (
  SELECT
    corpus_source,
    tool_name,
    COUNT(*) AS n_calls,
    COUNT(DISTINCT user_id) AS n_users,
    COUNT(DISTINCT args_hash) AS n_arg_signatures,
    AVG(IF(args_present, 1.0, 0.0)) AS args_present_rate,
    APPROX_QUANTILES(total_ms, 100)[SAFE_OFFSET(50)] AS p50_latency_ms,
    COUNTIF(has_prev) AS n_repeat_pairs,
    -- Denominator is all calls: "what share of calls to this tool were a repeat of the same
    -- (user, args) within five minutes", which is the quantity DESIGN 8 would exploit.
    SAFE_DIVIDE(COUNTIF(has_prev AND gap_s <= 300), COUNT(*)) AS repeat_within_5m,
    SAFE_DIVIDE(COUNTIF(identical_to_prev), COUNTIF(has_prev)) AS identical_rate,
    APPROX_QUANTILES(IF(has_prev, gap_s, NULL), 100)[SAFE_OFFSET(50)]
      AS median_inter_call_interval_s,
    -- change_gap: the elapsed time between two consecutive observations THAT DIFFERED. It is
    -- an upper bound on the true change interval, never the interval itself.
    COUNTIF(has_prev AND NOT identical_to_prev) AS n_observed_changes,
    APPROX_QUANTILES(IF(has_prev AND NOT identical_to_prev, gap_s, NULL), 100)[SAFE_OFFSET(5)]
      AS p05_change_gap_s
  FROM pairs
  GROUP BY corpus_source, tool_name
),
stability AS (
  SELECT
    corpus_source,
    tool_name,
    APPROX_QUANTILES(stable_s, 100)[SAFE_OFFSET(90)] AS p90_stable_seconds
  FROM run_spans
  GROUP BY corpus_source, tool_name
)
SELECT
  t.corpus_source,
  t.tool_name,
  t.n_calls,
  t.n_users,
  t.n_arg_signatures,
  t.args_present_rate,
  t.p50_latency_ms,
  t.n_repeat_pairs,
  t.repeat_within_5m,
  t.identical_rate,
  t.median_inter_call_interval_s,
  t.n_observed_changes,
  t.p05_change_gap_s,
  s.p90_stable_seconds,
  p.poller_keys IS NOT NULL AS poller_detected,
  p.poller_keys,
  'NOT A GATE: pipeline test only (DESIGN 3.3, 3.4). Measured TTLs stay blocked on real traffic.'
    AS gate_status
FROM per_tool t
LEFT JOIN stability s USING (corpus_source, tool_name)
LEFT JOIN pollers p USING (corpus_source, tool_name)
ORDER BY t.corpus_source, t.n_calls DESC, t.tool_name;
-- P4 END
