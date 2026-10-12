-- tool_ttl_report.sql: the operational form of Preflight P4 (DESIGN 8.6).
--
-- A HUMAN READS THIS AND WRITES NUMBERS INTO `cacheable_tools`. It is not an automated
-- pipeline, deliberately: automating it means another artifact, another GCS path and another
-- parity contract for maybe 10% more value, and it removes the human from between a
-- measurement and a correctness-affecting default (DESIGN 8.6).
--
-- One row per (corpus_source, tool). The `recommendation` column is written to be pasted, or
-- refused, by that human.
--
-- ------------------------------------------------------------------------------------------
-- THE THREE CAVEATS FROM DESIGN 8.6, IMPLEMENTED AS SQL RATHER THAN LEFT AS PROSE
-- ------------------------------------------------------------------------------------------
-- 1. THE ESTIMATOR IS BIASED LONG. `change_gap` is the time between two OBSERVATIONS that
--    differed, which equals the true change interval only when sampling outpaces change. The
--    bias is worst for rarely-called tools, which are exactly the ones with the least
--    telemetry and the least review attention. So `ttl_source` is emitted in
--    {measured, ceiling_undersampled, declared}, and a MEASURED TTL IS REFUSED whenever
--    median_inter_call_interval > p05_change_gap. Refused means NULL, not "use the ceiling".
--
-- 2. POLLERS ARE DETECTED STRUCTURALLY, NEVER STATISTICALLY. Any tool whose result contains a
--    key in {state, status, progress, pct, percent, eta, remaining, phase} whose observed
--    value set has cardinality > 1 for a fixed args hash gets a HARD ttl OF 0, permanently,
--    with no override. A PENDING -> PENDING -> RUNNING -> DONE run produces an entirely
--    plausible stability window, so a statistical detector hands `get_order_status` a ~150s
--    TTL: precisely the tool for which any TTL is wrong. This detector never looks at timing.
--
-- 3. THE MEASUREMENT MAY BE UNDEFINABLE. Many results embed a nonce (a timestamp, a request
--    id, a signed URL, a pagination cursor), so repeats are never byte-identical and every
--    stability window collapses to zero for reasons that have nothing to do with staleness.
--    `identical_rate` is therefore reported in this same query, next to the numbers it
--    invalidates, and below @min_identical_rate the measured TTL is refused.
--
-- WHAT THIS REPORT CANNOT SEE: `tool_context.actions`. A tool that sets state_delta,
-- artifact_delta, transfer_to_agent, escalate, skip_summarization or an auth/confirmation
-- request is not cacheable at any TTL, and no SQL over BQAA telemetry can tell. That check is
-- the write-time blocklist in DESIGN 8.2. This report is necessary, not sufficient.
--
-- ------------------------------------------------------------------------------------------
-- MEASUREMENT HYGIENE
-- ------------------------------------------------------------------------------------------
-- Cache hits are excluded (DESIGN 11.5). A cached payload is CONSTANT BY CONSTRUCTION, so one
-- admitted hit inflates identical_rate and stretches every stability window; the cache would
-- teach the estimator that the cache is always fresh.
--
-- Exclusion alone is not enough, because the poisoning is by ABSENCE as well as presence: a
-- served hit means no tool execution and no row at all, so the observed inter-call interval
-- stretches toward the TTL and the next measurement ratchets it up (DESIGN 11.4). Hence
-- @holdout_only: restrict the corpus to the permanent holdout arm, the only traffic the cache
-- never suppressed. Refresh probes (DESIGN 8.6) restore unbiased data too, but they are
-- identified by `probe: true` in provenance custom_metadata, which only exists if the host
-- application added the router's keys to BQAA's allowlist (DESIGN 11.3); the holdout arm is
-- reconstructible from the session id alone and so is what this query uses.
--
-- At PREFLIGHT time there is no router, no cache and no suppression, so pass
-- @holdout_only = FALSE and use the whole corpus. Once the router is live, pass TRUE.
--
-- ------------------------------------------------------------------------------------------
-- SOURCE, AND WHY
-- ------------------------------------------------------------------------------------------
-- BASE `agent_events` throughout, and TWO event types, because BQAA splits one tool call across
-- them: TOOL_STARTING carries `content.$.args` and TOOL_COMPLETED carries `content.$.result`.
-- Neither row has both. The report needs both, so it pairs them. v_tool_completed is no help:
-- its SELECT list lifts `$.tool` and `$.result` out of `content` and drops everything else, so
-- the args are unreachable through the view. Consequences:
--   * the cache-hit exclusion must be the BASE spelling,
--     JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NULL. Written as the view's
--     `tool_result` spelling it matches nothing and fails OPEN. Annotated at its use below.
--   * latency is `CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64)`. The bare name `total_ms`
--     is the view's, not the table's, and there is no base column by that name.
--   * `event_type` is the leading cluster column and `timestamp`/`agent` are the partition and
--     cluster keys (DESIGN 18.2), so all three are in both WHERE clauses.
--
-- HOW THE PAIRING WORKS, and where it can be wrong. This corpus has no
-- `attributes.adk.function_call_id` on either row, so there is no id to join on and the pairing
-- is positional: within one invocation and one tool name, the nth start is the nth completion.
-- That is exact for calls issued in sequence. Two CONCURRENT calls to the SAME tool in ONE
-- invocation could pair start A with completion B, which swaps two args hashes. It cannot
-- invent a repeat that did not happen, so it perturbs which bucket a call lands in rather than
-- inflating the cache case. `args_present_rate` still reports whether args were found at all;
-- 0.0 means the pairing produced nothing, every call for a tool collapsed into one bucket, and
-- no TTL from that run should be adopted. That degenerate case fires the poller detector more
-- often and depresses identical_rate, so it errs toward "do not cache".
--
-- ------------------------------------------------------------------------------------------
-- DEPLOY-TIME TEXT SUBSTITUTION (BigQuery cannot parameterise an identifier)
--   @project   SMART_ROUTER_BQ_PROJECT   required, no default
--   @dataset   SMART_ROUTER_BQ_DATASET   default smart_router   (location US)
--
-- RUNTIME QUERY PARAMETERS
--   @agent STRING                   the ADK agent name, verbatim
--   @since TIMESTAMP                measurement floor; also the partition-pruning predicate
--   @synthetic_user_prefix STRING   corpus_source discriminator; '' calls everything
--                                   production. scripts/generate_corpus.py writes
--                                   user_id = f"corpus-{persona_id}", so the default is
--                                   'corpus-'. Pass the same value to backtest.sql.
--   @holdout_only BOOL              TRUE once the router is live (see above)
--   @holdout_fraction FLOAT64       RouterConfig.holdout_fraction, default 0.05
--   @min_repeat_pairs INT64         evidence floor, suggested 30
--   @min_identical_rate FLOAT64     nonce threshold, suggested 0.90
--   @ttl_safety_divisor FLOAT64     suggested 2.0; see the comment on suggested_ttl_seconds
--   @cache_min_ttl_s INT64          RouterConfig.cache_min_ttl_s, default 30
--   @cache_max_ttl_s INT64          RouterConfig.cache_max_ttl_s, default 3600
--
--   bq query --project_id=your-project --location=US --use_legacy_sql=false \
--     --parameter='agent:STRING:support' \
--     --parameter='since:TIMESTAMP:2026-08-01 00:00:00' \
--     --parameter='synthetic_user_prefix:STRING:corpus-' \
--     --parameter='holdout_only:BOOL:false' \
--     --parameter='holdout_fraction:FLOAT64:0.05' \
--     --parameter='min_repeat_pairs:INT64:30' \
--     --parameter='min_identical_rate:FLOAT64:0.90' \
--     --parameter='ttl_safety_divisor:FLOAT64:2.0' \
--     --parameter='cache_min_ttl_s:INT64:30' \
--     --parameter='cache_max_ttl_s:INT64:3600' \
--     < tool_ttl_report.sql
--
-- Everything runs under ADC. No service-account key is referenced anywhere.

WITH completed AS (
  -- The result half of a tool call, plus the sequence number the args half is matched on.
  --
  -- NUMBER FIRST, FILTER LATER. Both the cache-hit marker and `status` are read here and acted
  -- on in `raw` below, not in this WHERE clause. Dropping a row before ROW_NUMBER renumbers the
  -- survivors, and only this side would renumber: a cache hit is invisible on a TOOL_STARTING
  -- row, so `started` cannot drop the matching one. An invocation whose first call was served
  -- from cache would then pair its second, real completion with its FIRST call's arguments.
  -- Wrong args on a real result is the one error that inflates identical_rate rather than
  -- depressing it, so it is the one that had to be designed out.
  SELECT
    invocation_id,
    event_id,
    user_id,
    session_id,
    timestamp,
    status,
    CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms,
    JSON_VALUE(content, '$.tool') AS tool_name,
    -- Never NULL, so the run-boundary and identity comparisons below stay total. A missing
    -- `$.result` becomes a constant sentinel, which is honest: every call returned the same
    -- nothing, and identical_rate should say so.
    IFNULL(TO_JSON_STRING(JSON_QUERY(content, '$.result')), '<no-result-key>') AS result_json,
    -- SOURCE: BASE agent_events, so this is the base spelling of the cache-hit marker
    -- (DESIGN 9.1, 11.5). The view's `tool_result` spelling would match nothing here.
    JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL AS is_cache_hit,
    ROW_NUMBER() OVER (
      PARTITION BY invocation_id, JSON_VALUE(content, '$.tool') ORDER BY timestamp, event_id
    ) AS call_seq
  FROM `@project.@dataset.agent_events`
  WHERE event_type = 'TOOL_COMPLETED'
    AND agent = @agent
    AND timestamp >= @since
    -- ADK control plane, never a cacheable tool (DESIGN 7.2). Safe in the WHERE because
    -- `started` applies the identical predicate, so both sides number the same set of calls.
    AND JSON_VALUE(content, '$.tool') != 'transfer_to_agent'
),
started AS (
  -- The args half. Same partition and the same ORDER BY as `completed`, which is what makes the
  -- two sequence numbers line up. event_id breaks timestamp ties so both sides agree on order.
  --
  -- NO CACHE-HIT EXCLUSION HERE, and it is not an omission. A start row carries `$.args` and no
  -- `$.result`, so the marker cannot be present on it and the predicate would be a no-op that
  -- reads as a safeguard. The exclusion lives in `raw`, on the result side, where the marker is.
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
raw AS (
  SELECT
    c.user_id,
    c.timestamp,
    -- Carried only to break timestamp ties in `seq`. See the WINDOW clause there.
    c.event_id,
    c.total_ms,
    c.tool_name,
    c.result_json,
    -- The grouping key is the DESIGN 8.1 cache key minus its constants. Key order inside the
    -- serialised args can differ between rows and would split a bucket; a split bucket loses
    -- repeats, which lowers identical_rate and shortens the window, so that error also fails
    -- toward "do not cache".
    TO_HEX(SHA256(IFNULL(TO_JSON_STRING(s.args), '<no-args-key>'))) AS args_hash,
    s.args IS NOT NULL AS args_present,
    IF(
      @synthetic_user_prefix != ''
        AND STARTS_WITH(IFNULL(c.user_id, ''), @synthetic_user_prefix),
      'synthetic',
      'production'
    ) AS corpus_source,
    -- Arm, reconstructed rather than read back, so this works over history that predates the
    -- router and needs no provenance column.
    --
    -- CONTRACT with plugin.py and with backtest.sql: the same session lands in the same arm on
    -- all three sides only if the hash matches exactly. The Python equivalent is
    --   int(hashlib.sha256(f"{agent}|{user_id}|{session_id}".encode())
    --       .hexdigest()[:8], 16) % 10000 < holdout_fraction * 10000
    -- Session-hashed, never turn-hashed: within a session the routing decision is already
    -- fixed, so a per-turn holdout is contaminated by construction (DESIGN 11.4). DESIGN 11.4
    -- writes hash(app, user, session); `agent` stands in for the app because the verified
    -- schema (DESIGN 18.2) has no app column.
    MOD(
      CAST(CONCAT('0x', SUBSTR(TO_HEX(SHA256(
        CONCAT(@agent, '|', IFNULL(c.user_id, ''), '|', IFNULL(c.session_id, ''))
      )), 1, 8)) AS INT64),
      10000
    ) < CAST(@holdout_fraction * 10000 AS INT64) AS is_holdout
  FROM completed c
  -- LEFT, not INNER. A completion whose start is missing keeps its result and falls into the
  -- '<no-args-key>' bucket, which is the pessimistic side. An INNER JOIN would drop the row and
  -- quietly shrink the denominator instead.
  LEFT JOIN started s
    ON s.invocation_id = c.invocation_id
    AND s.tool_name = c.tool_name
    AND s.call_seq = c.call_seq
  -- The two filters deferred out of `completed` so that call_seq could be assigned first.
  WHERE c.status = 'OK'
    AND NOT c.is_cache_hit
),
calls AS (
  SELECT * FROM raw WHERE NOT @holdout_only OR is_holdout
),
seq AS (
  -- One series per (tool, user, args). user_id is in the partition because it is
  -- unconditionally in the cache key (DESIGN 8.1), so two users polling the same tool are two
  -- series rather than one interleaved one whose results appear to flap.
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
  -- the same data. Since this number sets a TTL that ships, the order it is computed under has
  -- to be total.
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
  -- A run is a maximal streak of byte-identical consecutive results, and its span is how long
  -- the value was OBSERVED to hold. Single-observation runs contribute 0 seconds, which is
  -- correct: one observation demonstrates no stability whatsoever. Keeping them shortens the
  -- reported window, which is the conservative direction.
  SELECT
    corpus_source,
    tool_name,
    TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), MILLISECOND) / 1000.0 AS stable_s
  FROM runs
  GROUP BY corpus_source, tool_name, user_id, args_hash, run_id
),
poller_pairs AS (
  -- CAVEAT 2. Depth-blind on purpose: the regex finds `"key": value` for the reserved key set
  -- anywhere in the serialised result, so a status buried at `$.result.job.state` is caught
  -- too. Case-insensitive for the same reason: missing `"Status"` would be the unsafe
  -- direction. The captured group is the whole key/value pair, because REGEXP_EXTRACT_ALL
  -- allows exactly one capturing group; the key is recovered from the front of it.
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
  -- Cardinality > 1 for a fixed args hash, per key, per user. No timing input at all, which is
  -- the entire point: the timing of a poller looks like the timing of a slowly changing value.
  SELECT
    corpus_source,
    tool_name,
    STRING_AGG(DISTINCT poll_key ORDER BY poll_key) AS poller_keys
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
    -- Denominator is all calls to the tool: "what share of calls repeated the same
    -- (user, args) within five minutes", which is the share DESIGN 8 could ever serve.
    SAFE_DIVIDE(COUNTIF(has_prev AND gap_s <= 300), COUNT(*)) AS repeat_within_5m,
    -- CAVEAT 3, reported next to the numbers it invalidates.
    SAFE_DIVIDE(COUNTIF(identical_to_prev), COUNTIF(has_prev)) AS identical_rate,
    APPROX_QUANTILES(IF(has_prev, gap_s, NULL), 100)[SAFE_OFFSET(50)]
      AS median_inter_call_interval_s,
    COUNTIF(has_prev AND NOT identical_to_prev) AS n_observed_changes,
    -- CAVEAT 1. change_gap is the elapsed time between two consecutive observations THAT
    -- DIFFERED, so it is an upper bound on the true change interval, never the interval. p05
    -- is the earliest change we caught, which is the only defensible input to a TTL.
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
),
classified AS (
  SELECT
    t.*,
    s.p90_stable_seconds,
    p.poller_keys,
    p.poller_keys IS NOT NULL AS poller_detected,
    -- Halving p05_change_gap is the safety margin for caveat 1: the gap is an upper bound on
    -- the change interval, so serving for the whole of it is serving past the change roughly
    -- half the time. The divisor is a parameter because it is a judgement, not a measurement.
    CAST(FLOOR(SAFE_DIVIDE(t.p05_change_gap_s, @ttl_safety_divisor)) AS INT64) AS raw_ttl_s,
    -- The refusal ladder. Order matters, and the first two rungs are refusals of the
    -- measurement itself rather than of its value.
    CASE
      -- A poller is never measurable and never cacheable, whatever the numbers say.
      WHEN p.poller_keys IS NOT NULL THEN 'declared'
      -- Nothing repeated often enough for any of the pair statistics to mean anything.
      WHEN t.n_repeat_pairs < @min_repeat_pairs THEN 'declared'
      -- Caveat 3: a nonce in the payload makes byte-identity, and therefore change detection,
      -- undefinable.
      WHEN IFNULL(t.identical_rate, 0.0) < @min_identical_rate THEN 'declared'
      -- Never observed to change. That is not evidence of stability, it is absence of
      -- evidence, and it is the shape rarely-called tools take.
      WHEN t.n_observed_changes = 0 OR t.p05_change_gap_s IS NULL THEN 'ceiling_undersampled'
      -- Caveat 1: sampling did not outpace change, so change_gap is biased long. Refuse.
      WHEN t.median_inter_call_interval_s > t.p05_change_gap_s THEN 'ceiling_undersampled'
      ELSE 'measured'
    END AS ttl_source
  FROM per_tool t
  LEFT JOIN stability s USING (corpus_source, tool_name)
  LEFT JOIN pollers p USING (corpus_source, tool_name)
)
SELECT
  corpus_source,
  IF(@holdout_only, 'holdout', 'all') AS arm,
  tool_name,
  n_calls,
  n_users,
  n_arg_signatures,
  args_present_rate,
  p50_latency_ms,
  n_repeat_pairs,
  repeat_within_5m,
  identical_rate,
  median_inter_call_interval_s,
  n_observed_changes,
  p05_change_gap_s,
  p90_stable_seconds,
  poller_detected,
  poller_keys,
  ttl_source,
  -- An UPPER BOUND, never a value to adopt. Emitted only where the measured TTL was refused
  -- for undersampling, so that a human can see how much room there might be and go collect
  -- better data rather than guess from this row.
  IF(
    ttl_source = 'ceiling_undersampled',
    CAST(FLOOR(IFNULL(p05_change_gap_s, p90_stable_seconds)) AS INT64),
    NULL
  ) AS ttl_ceiling_seconds,
  CASE
    -- Hard zero, permanent, no override (DESIGN 8.6). Not NULL: NULL invites a human to fill
    -- in a number, and for a poller there is no number.
    WHEN poller_detected THEN 0
    WHEN ttl_source != 'measured' THEN NULL
    -- Below the floor, DESIGN 8.6 says mark the tool non-cacheable rather than cache it
    -- briefly. A 4-second TTL buys nothing and costs a staleness class.
    WHEN raw_ttl_s < @cache_min_ttl_s THEN NULL
    -- cache_max_ttl_s is the hard stop on the ratchet in DESIGN 11.4.
    ELSE LEAST(raw_ttl_s, @cache_max_ttl_s)
  END AS suggested_ttl_seconds,
  CASE
    WHEN poller_detected
      THEN CONCAT(
        'DO NOT CACHE. Structural poller on key(s): ', poller_keys,
        '. Hard ttl=0, permanent, no override (DESIGN 8.6).')
    WHEN n_repeat_pairs < @min_repeat_pairs
      THEN CONCAT(
        'Insufficient evidence: ', CAST(n_repeat_pairs AS STRING),
        ' repeat pairs. Hand-declare a TTL or leave the tool out of cacheable_tools.')
    WHEN IFNULL(identical_rate, 0.0) < @min_identical_rate
      THEN CONCAT(
        'Measured TTL UNDEFINABLE: identical_rate ',
        CAST(ROUND(IFNULL(identical_rate, 0.0), 3) AS STRING),
        ' is below the threshold, so repeats are not byte-identical. Suspect a nonce ',
        '(timestamp, request id, signed URL, cursor). Hand-declare or do not cache.')
    WHEN ttl_source = 'ceiling_undersampled' AND n_observed_changes = 0
      THEN 'No change ever observed, so there is no evidence of WHEN it changes. Ceiling only.'
    WHEN ttl_source = 'ceiling_undersampled'
      THEN CONCAT(
        'Sampled slower than it changes (median_inter_call_interval > p05_change_gap), ',
        'so change_gap is biased long. Ceiling only, and no measured TTL.')
    WHEN raw_ttl_s < @cache_min_ttl_s
      THEN CONCAT(
        'Measured TTL ', CAST(raw_ttl_s AS STRING),
        's is below cache_min_ttl_s. Mark NON-CACHEABLE rather than caching it (DESIGN 8.6).')
    ELSE CONCAT(
      'cacheable_tools: {"', tool_name, '": ',
      CAST(LEAST(raw_ttl_s, @cache_max_ttl_s) AS STRING), '}')
  END AS recommendation,
  -- Carried on every row so a screenshot of this report cannot lose it (DESIGN 3.3, R12).
  IF(
    corpus_source = 'production',
    'measured on production traffic',
    'SYNTHETIC CORPUS: no TTL from this row may be adopted (DESIGN 3.3, 8.6)'
  ) AS provenance_warning
FROM classified
ORDER BY corpus_source, n_calls DESC, tool_name;
