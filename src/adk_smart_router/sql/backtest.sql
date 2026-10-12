-- backtest.sql — what shadow mode would have measured, computed over months of history
-- at zero hot-path cost (D7).
--
-- Coverage@N of the trained model against the static frequency baseline, split by
-- corpus_source (DESIGN 3.3) and by arm, reported per ISO week so a trend is visible.
-- The holdout arm is reported separately: it is the only unbiased corpus and the only
-- valid A/B baseline for every claim in DESIGN 2 (11.4).
--
-- This procedure never calls AI.EMBED. It reuses the vectors train_router.sql already
-- persisted, which is what makes a months-long window cost a table scan instead of $45.
-- Rows with no persisted vector cannot be scored; they are counted, never dropped
-- silently, because a silent drop would bias the coverage it is reporting.
--
--   CALL `your-project.smart_router.backtest_router`(
--     'your-project',            -- project      SMART_ROUTER_BQ_PROJECT
--     'smart_router',            -- dataset      SMART_ROUTER_BQ_DATASET   (location US)
--     'support',                 -- agent        the ADK agent name, verbatim
--     NULL,                      -- since        explicit floor, or NULL to use lookback
--     180,                       -- lookback     days
--     7,                         -- top_n        pass RouterConfig.max_tools_kept, or these
--                                --              numbers do not compare to the gate
--     0.05,                      -- holdout_fraction  SMART_ROUTER_HOLDOUT_FRACTION
--     'corpus-',                 -- synthetic_user_prefix, '' to call everything production
--                                --              MUST match generate_corpus.py's user id
--                                --              prefix and preflight.sql's. Get it wrong
--                                --              and every row reports as production,
--                                --              which is a silently wrong answer rather
--                                --              than an error.
--     TIMESTAMP '2026-07-28 16:08:59');
--                                -- trained_since  the model's corpus floor:
--                                --              router_training_runs.corpus_since. NULL
--                                --              leaves overlaps_training false on every
--                                --              row, which reads as "none of this is
--                                --              in-sample" and is usually a lie.
--
-- Deploy-time text substitution: `@project.@dataset` only, for the procedure's own name.
-- Everything else is a runtime argument. Everything runs under ADC.

CREATE OR REPLACE PROCEDURE `@project.@dataset.backtest_router`(
  project STRING,
  dataset STRING,
  agent STRING,
  since TIMESTAMP,
  lookback INT64,
  top_n INT64,
  holdout_fraction FLOAT64,
  synthetic_user_prefix STRING,
  trained_since TIMESTAMP
)
BEGIN
  DECLARE embedding_dims INT64 DEFAULT 768;

  DECLARE run_id STRING;
  DECLARE run_since TIMESTAMP;
  DECLARE agent_key STRING;
  DECLARE features_table STRING;
  DECLARE model_name STRING;
  DECLARE results_table STRING;
  DECLARE flatten_cols STRING;

  SET run_id = GENERATE_UUID();
  SET run_since = IFNULL(since, TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL lookback DAY));
  SET agent_key = REGEXP_REPLACE(agent, r'[^a-zA-Z0-9_]', '_');
  SET features_table = FORMAT('%s.%s.router_features_%s', project, dataset, agent_key);
  SET model_name = FORMAT('%s.%s.router_%s', project, dataset, agent_key);
  SET results_table = FORMAT('%s.%s.router_backtest', project, dataset);

  -- ==================================================================================
  -- 1. Label the window, then attach the persisted vectors.
  -- ==================================================================================
  -- Label extraction is identical to train_router.sql step 1 (DESIGN 9.2). It is
  -- repeated rather than shared because a backtest that used a different corpus
  -- definition from training would be measuring a different thing, and because the
  -- cache-hit exclusion has to be visible in every query that reads this corpus (11.5).
  EXECUTE IMMEDIATE FORMAT("""
    CREATE OR REPLACE TEMP TABLE bt_labeled AS
    WITH prompts AS (
      -- v_user_message_received has no prompt text (18.2): read the BASE table. The
      -- timestamp and agent predicates keep this inside the DAY partitioning and the
      -- (event_type, agent, user_id) clustering, which is what makes a 180-day window
      -- affordable.
      SELECT
        invocation_id,
        ANY_VALUE(session_id) AS session_id,
        ANY_VALUE(user_id) AS user_id,
        ANY_VALUE(JSON_VALUE(content, '$.text_summary')) AS prompt_text,
        MIN(timestamp) AS prompt_ts
      FROM `%s.%s.agent_events`
      WHERE event_type = 'USER_MESSAGE_RECEIVED'
        AND agent = @agent_name
        AND timestamp >= @since
        AND is_truncated = FALSE
      GROUP BY invocation_id
    ),
    first_tool AS (
      SELECT
        invocation_id,
        ARRAY_AGG(tool_name ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS tool_name
      FROM `%s.%s.v_tool_completed`
      WHERE agent = @agent_name
        AND status = 'OK'
        AND timestamp >= @since
        AND tool_name != 'transfer_to_agent'          -- ADK control plane, never a label
        -- Cache-hit exclusion written for V_TOOL_COMPLETED (9.1). The view renames
        -- content.tool -> tool_name and content.result -> tool_result, so the base-table
        -- spelling would match nothing here and fail OPEN.
        AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
      GROUP BY invocation_id
    ),
    dropped AS (
      SELECT DISTINCT invocation_id
      FROM `%s.%s.v_tool_error`
      WHERE agent = @agent_name AND timestamp >= @since
      UNION DISTINCT
      -- Cache-hit invocations are dropped whole, not merely filtered row by row. An
      -- invocation whose only tool came from cache would fall through the LEFT JOIN and
      -- be labelled ABSTAIN, teaching the router to prune everything for exactly the
      -- prompts the cache serves best.
      SELECT DISTINCT invocation_id
      FROM `%s.%s.v_tool_completed`
      WHERE agent = @agent_name
        AND timestamp >= @since
        AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
      UNION DISTINCT
      -- The same exclusion written for the BASE agent_events table (9.1). Not a
      -- duplicate of the branch above: under router-first registration a cache hit can
      -- land as an orphaned event the views do not pair up (11.5, 18.1), and the base
      -- table is the only source that still carries it.
      SELECT DISTINCT invocation_id
      FROM `%s.%s.agent_events`
      WHERE agent = @agent_name
        AND timestamp >= @since
        AND JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL
    )
    SELECT
      p.invocation_id,
      p.session_id,
      DATE_TRUNC(DATE(p.prompt_ts), WEEK(MONDAY)) AS week_start,
      -- BQAA has no corpus-provenance column, so the discriminator is the user id the
      -- corpus generator writes (DESIGN 3.3 requires the split; provision.sh owns the
      -- prefix). An empty prefix calls everything production.
      IF(@synthetic_prefix != '' AND STARTS_WITH(IFNULL(p.user_id, ''), @synthetic_prefix),
         'synthetic', 'production') AS corpus_source,
      -- Arm, reconstructed rather than read back. Reconstruction works over history that
      -- predates the router, which is the point of a backtest, and it needs no
      -- provenance column.
      --
      -- CONTRACT with plugin.py: the same session lands in the same arm on both sides
      -- only if the hash matches exactly. The Python equivalent of this expression is
      --   int(hashlib.sha256(f"{agent}|{user_id}|{session_id}".encode())
      --       .hexdigest()[:8], 16) %% 10000 < holdout_fraction * 10000
      -- Session-hashed, never turn-hashed: within a session the decision is already
      -- fixed, so a per-turn holdout is contaminated by construction (11.4).
      -- DESIGN 11.4 writes hash(app, user, session); `agent` stands in for the app
      -- because the verified schema (18.2) does not include an app column.
      IF(MOD(CAST(CONCAT('0x', SUBSTR(TO_HEX(SHA256(
             CONCAT(@agent_name, '|', IFNULL(p.user_id, ''), '|', IFNULL(p.session_id, ''))
           )), 1, 8)) AS INT64), 10000) < CAST(@holdout_fraction * 10000 AS INT64),
         'holdout', 'routed') AS arm,
      IFNULL(t.tool_name, 'ABSTAIN') AS tool_name,
      f.v AS v
    FROM prompts p
    LEFT JOIN first_tool t USING (invocation_id)
    LEFT JOIN `%s` f USING (invocation_id)
    -- Byte-for-byte the admission rule train_router.sql uses, applied at the same point
    -- in the query. A backtest over a different corpus than the model was trained on is
    -- measuring something else.
    WHERE p.invocation_id NOT IN (SELECT invocation_id FROM dropped)
      AND p.prompt_text IS NOT NULL
      AND LENGTH(p.prompt_text) BETWEEN 3 AND 4000
  """, project, dataset, project, dataset, project, dataset, project, dataset,
       project, dataset, features_table)
  USING agent AS agent_name,
        run_since AS since,
        IFNULL(synthetic_user_prefix, '') AS synthetic_prefix,
        holdout_fraction AS holdout_fraction;

  -- ==================================================================================
  -- 2. Flatten and score.
  -- ==================================================================================
  -- Same f0..f767 layout as training (D5). If the two ever diverge, the booster reads
  -- the wrong feature at every index and the backtest reports confident nonsense.
  SET flatten_cols = (
    SELECT STRING_AGG(FORMAT('v[OFFSET(%d)] AS f%d', i, i), ', ' ORDER BY i)
    FROM UNNEST(GENERATE_ARRAY(0, embedding_dims - 1)) AS i
  );

  EXECUTE IMMEDIATE FORMAT("""
    CREATE OR REPLACE TEMP TABLE bt_flat AS
    SELECT week_start, corpus_source, arm, tool_name, %s
    FROM bt_labeled
    WHERE ARRAY_LENGTH(v) = @dims
  """, flatten_cols)
  USING embedding_dims AS dims;

  -- The static comparator: the N most-called tools over the WHOLE window, one fixed set
  -- for every week. Fitting it per week would make it an oracle on the week it scores;
  -- fitting it on the whole window still gives it a look at the future, which biases the
  -- lift DOWNWARD. That is the safe direction: the baseline is being flattered, and
  -- criterion 2 (DESIGN 10) still has to beat it.
  EXECUTE IMMEDIATE """
    CREATE OR REPLACE TEMP TABLE bt_baseline AS
    SELECT tool_name
    FROM bt_labeled
    WHERE tool_name != 'ABSTAIN'
    GROUP BY tool_name
    ORDER BY COUNT(*) DESC
    LIMIT @top_n
  """
  USING top_n AS top_n;

  EXECUTE IMMEDIATE FORMAT("""
    CREATE OR REPLACE TEMP TABLE bt_scored AS
    SELECT
      week_start,
      corpus_source,
      arm,
      tool_name AS actual,
      -- Coverage@N: is the true tool inside the set the router would have kept?
      -- ABSTAIN is stripped from the ranking first, because ABSTAIN is never an edge
      -- action (D10) and so never occupies one of the N slots.
      EXISTS(
        SELECT 1
        FROM UNNEST(predicted_tool_name_probs) hit
        WHERE hit.label = tool_name
          AND (SELECT COUNT(*) FROM UNNEST(predicted_tool_name_probs) other
               WHERE other.label != 'ABSTAIN' AND other.prob > hit.prob) < @top_n
      ) AS covered_model,
      tool_name IN (SELECT tool_name FROM bt_baseline) AS covered_static
    FROM ML.PREDICT(MODEL `%s`, (SELECT * FROM bt_flat))
  """, model_name)
  USING top_n AS top_n;

  -- ==================================================================================
  -- 3. Per week, per corpus_source, per arm.
  -- ==================================================================================
  -- Both coverages are computed over the SCORED rows only, so they share a denominator
  -- and the difference means something. n_unscored says how much of the week that
  -- leaves out.
  CREATE OR REPLACE TEMP TABLE bt_summary AS
  WITH labeled_counts AS (
    SELECT
      week_start,
      corpus_source,
      arm,
      COUNT(*) AS n_labeled
    FROM bt_labeled
    GROUP BY week_start, corpus_source, arm
  ),
  scored_counts AS (
    SELECT
      week_start,
      corpus_source,
      arm,
      COUNT(*) AS n_scored,
      COUNTIF(actual = 'ABSTAIN') AS n_abstain,
      COUNTIF(actual != 'ABSTAIN') AS n_tool_rows,
      -- An ABSTAIN row has no true tool, so "the true tool is in the kept set" is
      -- undefined for it. Counting it as trivially covered would lift both arms and
      -- shrink the very difference this report exists to measure.
      SAFE_DIVIDE(COUNTIF(covered_model AND actual != 'ABSTAIN'),
                  COUNTIF(actual != 'ABSTAIN')) AS coverage_model,
      SAFE_DIVIDE(COUNTIF(covered_static AND actual != 'ABSTAIN'),
                  COUNTIF(actual != 'ABSTAIN')) AS coverage_static
    FROM bt_scored
    GROUP BY week_start, corpus_source, arm
  )
  SELECT
    l.week_start,
    l.corpus_source,
    l.arm,
    l.n_labeled,
    -- By subtraction, so it stays exact whatever kept a row out of bt_flat: a missing
    -- vector, or one that is not 768-d.
    l.n_labeled - IFNULL(s.n_scored, 0) AS n_unscored,
    IFNULL(s.n_scored, 0) AS n_scored,
    IFNULL(s.n_abstain, 0) AS n_abstain,
    IFNULL(s.n_tool_rows, 0) AS n_tool_rows,
    s.coverage_model,
    s.coverage_static,
    s.coverage_model - s.coverage_static AS lift,
    -- G2b, and the same quantity the promotion gate's criterion 2 uses: the share of the
    -- static ranking's misses that the model removes. The absolute gap beside it is kept
    -- for continuity, but it moves with top_n even when the model does not (DESIGN 10).
    -- SAFE_DIVIDE returns NULL for a week where the static ranking missed nothing, which
    -- is the honest answer: there was nothing there for the model to win.
    1 - SAFE_DIVIDE(1 - s.coverage_model, 1 - s.coverage_static) AS miss_reduction,
    -- A week the model was trained on is not a backtest of that week. Flagged rather
    -- than dropped, so the contrast between memorised and unseen weeks stays visible.
    trained_since IS NOT NULL
      AND l.week_start >= DATE(trained_since) AS overlaps_training
  FROM labeled_counts l
  LEFT JOIN scored_counts s
    USING (week_start, corpus_source, arm);

  -- ==================================================================================
  -- 4. Persist and return.
  -- ==================================================================================
  EXECUTE IMMEDIATE FORMAT("""
    CREATE TABLE IF NOT EXISTS `%s` (
      run_id STRING,
      run_at TIMESTAMP,
      agent STRING,
      corpus_since TIMESTAMP,
      top_n INT64,
      holdout_fraction FLOAT64,
      week_start DATE,
      corpus_source STRING,
      arm STRING,
      n_labeled INT64,
      n_unscored INT64,
      n_scored INT64,
      n_abstain INT64,
      n_tool_rows INT64,
      coverage_model FLOAT64,
      coverage_static FLOAT64,
      lift FLOAT64,           -- absolute gap. Reported for continuity, G2b no longer reads it.
      miss_reduction FLOAT64, -- G2b (DESIGN 2.4, 10).
      overlaps_training BOOL
    )
    PARTITION BY DATE(run_at)
    CLUSTER BY agent, arm
  """, results_table);

  -- The results table is CREATE IF NOT EXISTS, so one written by an earlier version of this
  -- procedure keeps its old schema and the INSERT below would fail on the unknown column.
  -- Existing rows read NULL, which is honest: those runs did not measure it.
  EXECUTE IMMEDIATE FORMAT("""
    ALTER TABLE `%s`
      ADD COLUMN IF NOT EXISTS miss_reduction FLOAT64
  """, results_table);

  EXECUTE IMMEDIATE FORMAT("""
    INSERT INTO `%s` (
      run_id, run_at, agent, corpus_since, top_n, holdout_fraction, week_start,
      corpus_source, arm, n_labeled, n_unscored, n_scored, n_abstain, n_tool_rows,
      coverage_model, coverage_static, lift, miss_reduction, overlaps_training)
    SELECT
      @run_id, CURRENT_TIMESTAMP(), @agent, @since, @top_n, @holdout_fraction, week_start,
      corpus_source, arm, n_labeled, n_unscored, n_scored, n_abstain, n_tool_rows,
      coverage_model, coverage_static, lift, miss_reduction, overlaps_training
    FROM bt_summary
  """, results_table)
  USING run_id AS run_id,
        agent AS agent,
        run_since AS since,
        top_n AS top_n,
        holdout_fraction AS holdout_fraction;

  SELECT *
  FROM bt_summary
  ORDER BY arm, corpus_source, week_start;
END;
