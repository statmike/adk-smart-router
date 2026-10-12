-- train_router.sql — the whole training pipeline for ONE agent, as one procedure.
--
-- labels (DESIGN 9.2) -> incremental embed (9.3) -> flatten to f0..f767 (D5)
--   -> train (9.4) -> promotion gate (10) -> EXPORT MODEL (9.4, 11.6)
--
-- Runnable from a scheduled query:
--
--   CALL `your-project.smart_router.train_router`(
--     'your-project',            -- project      SMART_ROUTER_BQ_PROJECT
--     'smart_router',            -- dataset      SMART_ROUTER_BQ_DATASET   (location US)
--     'support',                 -- agent        the ADK agent name, verbatim
--     'your-bucket',             -- bucket       SMART_ROUTER_GCS_BUCKET
--     'prod',                    -- environment  SMART_ROUTER_ENVIRONMENT
--     NULL,                      -- since        explicit floor, or NULL to use lookback
--     35);                       -- lookback     days
--
-- Deploy-time text substitution, two placeholders only. Everything else is a runtime
-- argument, because a scheduled query cannot template its own SQL.
--
--   @project.@dataset               where the procedure itself lives
--   @project.@location.@connection  the AI.EMBED connection (DESIGN 9.3 uses this same
--                                   form), e.g. your-project.us.smart_router_conn.
--                                   Set via SMART_ROUTER_BQ_CONNECTION. The
--                                   substituted project must be the same one passed as the
--                                   `project` argument.
--
-- Everything runs under ADC. No service-account key is referenced anywhere.

CREATE OR REPLACE PROCEDURE `@project.@dataset.train_router`(
  project STRING,
  dataset STRING,
  agent STRING,
  bucket STRING,
  environment STRING,
  since TIMESTAMP,
  lookback INT64
)
BEGIN
  -- --- Constants -------------------------------------------------------------------
  -- The embedding triple is the train half of the parity contract (DESIGN 9.1). One
  -- model name on both sides; a train/serve split here is a silent-vector bug, so these
  -- values are echoed into the run record and end up in metadata.json for the loader.
  DECLARE embedding_model STRING DEFAULT 'gemini-embedding-2';   -- GA, verified 18.3
  DECLARE embedding_dims INT64 DEFAULT 768;
  DECLARE backend STRING DEFAULT 'vertex';                       -- AI.EMBED resolves to Vertex
  DECLARE bq_connection STRING DEFAULT '@project.@location.@connection';

  DECLARE top_n INT64 DEFAULT 11;         -- must equal RouterConfig.max_tools_kept, or the
                                          -- gate is calibrated for a set size we never serve.
                                          -- Was 5, then 7 on 25 tools. Raised to 11 for the
                                          -- 100-tool surface: `tune_menus.py` measured realised
                                          -- recall 0.8958 at menu size 7 against 0.9187 at 11,
                                          -- for 1.8 points of saving (DESIGN 19.16).
  DECLARE keep_probability_mass FLOAT64 DEFAULT 0.90;
                                          -- must equal RouterConfig.keep_probability_mass, and
                                          -- it is the OTHER half of the served set size. Without
                                          -- menus the pruner walks the ranking keeping labels
                                          -- until their cumulative probability reaches this, so
                                          -- top_n is a cap and this is usually what binds. On
                                          -- the reference corpus a strict top-11 covers 0.9889
                                          -- and this rule covers 0.9263, which is why
                                          -- `coverage_mass` below exists (DESIGN 19.74).
  DECLARE eval_pct INT64 DEFAULT 25;      -- share of sessions held out for the gate. Was 20,
                                          -- which left 391 scorable eval rows: about +/-0.7
                                          -- points of noise on a 0.98 threshold (9.4).
  DECLARE min_class_rows INT64 DEFAULT 10;-- per-class floor (9.4). NOT a drop threshold.
                                          -- Was 30, which held 7% of tool-calling rows out of
                                          -- eval and handed the static baseline 6.3 free points.
  DECLARE max_embed_rows INT64 DEFAULT 50000;  -- per-run AI.EMBED cap (11.7 quota)

  -- --- Derived names and run state ------------------------------------------------
  DECLARE run_since TIMESTAMP;
  DECLARE version STRING;
  DECLARE agent_key STRING;               -- agent name, sanitised for use in an identifier
  DECLARE features_table STRING;
  DECLARE flat_table STRING;
  DECLARE model_name STRING;
  DECLARE runs_table STRING;
  DECLARE export_uri STRING;
  DECLARE flatten_cols STRING;

  DECLARE total_rows INT64 DEFAULT 0;
  DECLARE total_sessions INT64 DEFAULT 0;  -- what criterion 4 counts. See step 4.
  DECLARE train_rows INT64 DEFAULT 0;
  DECLARE eval_rows INT64 DEFAULT 0;
  DECLARE tool_classes INT64 DEFAULT 0;
  DECLARE pending_rows INT64 DEFAULT 0;
  DECLARE bad_width_rows INT64 DEFAULT 0;  -- successful embeddings of the wrong width (18.4)
  DECLARE widest_vector INT64 DEFAULT 0;
  DECLARE class_labels ARRAY<STRING>;
  DECLARE coverage FLOAT64;
  DECLARE coverage_mass FLOAT64;          -- the same question asked of the set the pruner
                                          -- actually builds without menus. Reported, never
                                          -- gates: see the gate section for why (DESIGN 19.74).
  DECLARE baseline FLOAT64;               -- static top-N coverage on the EVAL split.
  DECLARE baseline_corpus FLOAT64;        -- same top-N set over EVERY row. Logged, never gates.
  DECLARE miss_reduction FLOAT64;         -- share of the baseline's misses the model removes.
                                          -- This is criterion 2 (DESIGN 10).
  DECLARE abstain_precision FLOAT64;      -- logged, never gates (DESIGN 10)
  DECLARE macro_recall FLOAT64;           -- logged, never gates (DESIGN 10)
  DECLARE gate_passed BOOL DEFAULT FALSE;
  DECLARE gate_reason STRING DEFAULT '';

  SET run_since = IFNULL(since, TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL lookback DAY));
  -- UTC, YYYYMMDDTHHMMSSZ, so lexical order is chronological. model_store picks the
  -- highest complete version directory by string comparison and nothing else (D11).
  SET version = FORMAT_TIMESTAMP('%Y%m%dT%H%M%SZ', CURRENT_TIMESTAMP(), 'UTC');

  -- Two different spellings of the agent, deliberately. BigQuery identifiers get the
  -- sanitised one; the GCS path gets the raw ADK agent name, because model_store looks
  -- up `routers/{agent_name}/` using the name the plugin sees at runtime.
  SET agent_key = REGEXP_REPLACE(agent, r'[^a-zA-Z0-9_]', '_');
  SET features_table = FORMAT('%s.%s.router_features_%s', project, dataset, agent_key);
  SET flat_table = FORMAT('%s.%s.router_flat_%s', project, dataset, agent_key);
  SET model_name = FORMAT('%s.%s.router_%s', project, dataset, agent_key);
  SET runs_table = FORMAT('%s.%s.router_training_runs', project, dataset);
  SET export_uri = FORMAT(
    '%s/routers/%s/%s',
    IF(STARTS_WITH(bucket, 'gs://'), RTRIM(bucket, '/'), 'gs://' || RTRIM(bucket, '/')),
    agent,
    version);

  -- ==================================================================================
  -- 1. Labels (DESIGN 9.2) merged into the persisted feature table.
  -- ==================================================================================
  -- Persisting is a requirement, not an optimisation (9.3): a nightly full re-embed of
  -- 100k rows is ~$45/month for zero information gain, the same order as the entire
  -- token saving the product delivers at 20 tools.
  --
  -- `v ARRAY<FLOAT64>` starts empty, not NULL. BigQuery does not store NULL arrays, it
  -- stores them as empty, so the "re-embed where the vector IS NULL" rule must be
  -- written as ARRAY_LENGTH(v) = 0. Spelled `v IS NULL` it matches nothing and no row is
  -- ever embedded.
  EXECUTE IMMEDIATE FORMAT("""
    CREATE TABLE IF NOT EXISTS `%s` (
      invocation_id STRING NOT NULL,
      session_id STRING,
      user_id STRING,
      prompt_text STRING,
      prompt_ts TIMESTAMP,
      tool_name STRING,
      v ARRAY<FLOAT64>,
      embedded_at TIMESTAMP
    )
    PARTITION BY DATE(prompt_ts)
    CLUSTER BY tool_name
  """, features_table);

  EXECUTE IMMEDIATE FORMAT("""
    MERGE `%s` AS t
    USING (
      WITH prompts AS (
        -- v_user_message_received projects only the common columns and has NO prompt
        -- text (18.2), so the prompt is read off the BASE table. Filtering on timestamp
        -- and agent keeps us inside the DAY partition and the (event_type, agent,
        -- user_id) clustering.
        --
        -- Grouped by invocation_id alone: MERGE requires a unique source key, and a
        -- second grouping column would silently admit duplicate source rows.
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
        -- The first SUCCESSFUL tool of the invocation. This was the training label
        -- through v1 and is no longer one: step 3 builds the label from the session's
        -- whole tool set (19.2). It stays because the feature table is also the
        -- warehouse's record of what each invocation did, and `check_router.py` and the
        -- backtest read it. Nothing downstream of step 3 looks at it.
        SELECT
          invocation_id,
          ARRAY_AGG(tool_name ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS tool_name
        FROM `%s.%s.v_tool_completed`
        WHERE agent = @agent_name
          AND status = 'OK'
          AND timestamp >= @since
          -- transfer_to_agent is ADK control plane, never a label (7.2).
          AND tool_name != 'transfer_to_agent'
          -- Cache-hit exclusion, written for v_tool_completed (9.1, 11.5). The view
          -- renames content.tool -> tool_name and content.result -> tool_result, so the
          -- base-table spelling
          --   JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NULL
          -- matches NOTHING here and fails OPEN, admitting every cache hit into the
          -- corpus and training the model on its own echo.
          AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
        GROUP BY invocation_id
      ),
      dropped AS (
        -- Invocations that must not produce a training row at all.
        --
        -- Errors: the tool the model picked did not work, so the pairing is not
        -- evidence of a correct route.
        SELECT DISTINCT invocation_id
        FROM `%s.%s.v_tool_error`
        WHERE agent = @agent_name AND timestamp >= @since
        UNION DISTINCT
        -- Cache hits: filtering the ROW alone is not enough. An invocation whose only
        -- tool was served from cache would fall through the LEFT JOIN and be labelled
        -- ABSTAIN, which teaches the router to prune everything for exactly the prompts
        -- the cache serves best. Drop the invocation, the way an error does.
        SELECT DISTINCT invocation_id
        FROM `%s.%s.v_tool_completed`
        WHERE agent = @agent_name
          AND timestamp >= @since
          AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
        UNION DISTINCT
        -- The same exclusion written for the BASE agent_events table (9.1). Not a
        -- duplicate of the branch above: under router-first registration a cache hit can
        -- land as an orphaned event the views do not pair up (11.5, 18.1), and the base
        -- table is the only source that still carries it. backtest.sql drops the same
        -- set, so the two corpora cannot disagree about what a row is.
        SELECT DISTINCT invocation_id
        FROM `%s.%s.agent_events`
        WHERE agent = @agent_name
          AND timestamp >= @since
          AND JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL
      )
      SELECT
        p.invocation_id,
        p.session_id,        -- carried through: the train/eval split is session-hashed (9.4)
        p.user_id,           -- carried through: backtest.sql splits by corpus_source
        p.prompt_text,
        p.prompt_ts,
        -- A prompt with no tool is the ABSTAIN class (D10: a label, never an action).
        IFNULL(t.tool_name, 'ABSTAIN') AS tool_name
      FROM prompts p
      LEFT JOIN first_tool t USING (invocation_id)
      WHERE p.invocation_id NOT IN (SELECT invocation_id FROM dropped)
        AND p.prompt_text IS NOT NULL
        AND LENGTH(p.prompt_text) BETWEEN 3 AND 4000
    ) AS s
    -- The prompt_ts predicate prunes the target partitions. It cannot cause a duplicate
    -- insert: every source row has prompt_ts >= @since, and an invocation's prompt_ts
    -- never changes.
    ON t.invocation_id = s.invocation_id AND t.prompt_ts >= @since
    WHEN MATCHED AND t.tool_name != s.tool_name THEN
      -- Self-healing: an invocation still in flight on the previous run was labelled
      -- ABSTAIN and now has its tool. The vector depends only on the prompt, so it is
      -- kept and not re-embedded.
      UPDATE SET tool_name = s.tool_name
    WHEN NOT MATCHED THEN
      INSERT (invocation_id, session_id, user_id, prompt_text, prompt_ts, tool_name)
      VALUES (s.invocation_id, s.session_id, s.user_id, s.prompt_text, s.prompt_ts, s.tool_name)
  """, features_table, project, dataset, project, dataset, project, dataset,
       project, dataset, project, dataset)
  USING agent AS agent_name, run_since AS since;

  -- ==================================================================================
  -- 2. Embed the rows that have no vector yet (DESIGN 9.3).
  -- ==================================================================================
  -- AI.EMBED returns STRUCT<result ARRAY<FLOAT64>, status STRING> (18.4). The field is
  -- `result`, not `embedding`, and status = '' (empty string) means success.
  -- gemini-embedding-2 is a multimodal endpoint: connection_id is required and the width
  -- key is `outputDimensionality`, the same name google-genai sends from embedding.py.
  -- AI.EMBED IGNORES a model_params key it does not recognise instead of rejecting it, so a
  -- misspelling returns the endpoint's native 3072-wide vector with status = '' and nothing
  -- anywhere says so. Step 3 checks the width for that reason.
  --
  -- Oldest first, so a backlog drains deterministically instead of starving behind new
  -- traffic. Rows whose embedding failed keep an empty vector and are retried next run.
  -- endpoint, connection_id and model_params are injected as literals rather than passed
  -- as query parameters: AI.* named arguments want constants, and a literal is the one
  -- form that is certainly accepted.
  EXECUTE IMMEDIATE FORMAT("""
    CREATE OR REPLACE TEMP TABLE fresh_vectors AS
    SELECT
      invocation_id,
      AI.EMBED(
        content       => prompt_text,
        endpoint      => '%s',
        connection_id => '%s',
        model_params  => JSON '{"outputDimensionality": %d}'
      ) AS e
    FROM (
      SELECT invocation_id, prompt_text
      FROM `%s`
      WHERE IFNULL(ARRAY_LENGTH(v), 0) = 0
      ORDER BY prompt_ts
      LIMIT @max_embed_rows
    )
  """, embedding_model, bq_connection, embedding_dims, features_table)
  USING max_embed_rows AS max_embed_rows;

  EXECUTE IMMEDIATE FORMAT("""
    MERGE `%s` AS t
    USING (
      SELECT invocation_id, e.result AS v
      FROM fresh_vectors
      WHERE e.status = '' AND ARRAY_LENGTH(e.result) = @dims
    ) AS s
    ON t.invocation_id = s.invocation_id
    WHEN MATCHED THEN UPDATE SET v = s.v, embedded_at = CURRENT_TIMESTAMP()
  """, features_table)
  USING embedding_dims AS dims;

  -- The width check step 2 promises. It is written as a check rather than left to the MERGE's
  -- width filter because that filter fails SILENTLY: a misspelt width key returns 3072-wide
  -- vectors with status = '', the MERGE matches nothing, and training proceeds on whatever was
  -- already embedded or on an empty corpus. Every downstream step reports success. This is the
  -- only failure in the pipeline that ships a model instead of an error, so it stops the run.
  EXECUTE IMMEDIATE """
    SELECT
      COUNTIF(ARRAY_LENGTH(e.result) != @dims),
      IFNULL(MAX(ARRAY_LENGTH(e.result)), 0)
    FROM fresh_vectors
    WHERE e.status = ''
  """
  INTO bad_width_rows, widest_vector
  USING embedding_dims AS dims;

  IF bad_width_rows > 0 THEN
    RAISE USING MESSAGE = FORMAT(
      'AI.EMBED returned %d vectors that are not %d wide (widest %d). Check that model_params '
      || 'spells the width key outputDimensionality. An unrecognised key is ignored rather than '
      || 'rejected, and the endpoint then returns its native width (DESIGN 18.4).',
      bad_width_rows, embedding_dims, widest_vector);
  END IF;

  -- Surfaced in the run record rather than swallowed: max_embed_rows makes a partial
  -- corpus possible, and a partial corpus that nobody can see is a silent one.
  EXECUTE IMMEDIATE FORMAT(
    "SELECT COUNT(*) FROM `%s` WHERE IFNULL(ARRAY_LENGTH(v), 0) = 0", features_table)
  INTO pending_rows;

  -- ==================================================================================
  -- 3. Flatten to f0..f767 (D5), and build the v2 label (DESIGN 19.2).
  -- ==================================================================================
  -- This step, not step 1, decides what the model is fitted on. Step 1 maintains the
  -- feature table, which is an embedding cache keyed by invocation and carries a
  -- per-invocation `tool_name` that v2 no longer trains on. Reading that column as the
  -- label is the mistake this comment exists to prevent.
  --
  -- What changed and why (19.1, 19.2): the router declares one tool list per SESSION,
  -- chosen from the first message, and it is wrong exactly when the session goes on to
  -- want something the list does not hold. A missing tool costs about 2,278 billed
  -- tokens and a spare one about 6.5, a ratio near 75:1, so the quantity worth predicting
  -- is the session's whole set and not the first tool of each turn.
  -- EXPORT MODEL ... ML_XGBOOST_BOOSTER renames features positionally. Naming them
  -- explicitly makes the index mapping deterministic instead of resting on undocumented
  -- behaviour, and it is what embedding.py reproduces on the hot path.
  SET flatten_cols = (
    SELECT STRING_AGG(FORMAT('v[OFFSET(%d)] AS f%d', i, i), ', ' ORDER BY i)
    FROM UNNEST(GENERATE_ARRAY(0, embedding_dims - 1)) AS i
  );

  EXECUTE IMMEDIATE FORMAT("""
    CREATE OR REPLACE TABLE `%s`
    CLUSTER BY tool_name AS
    WITH opening AS (
      -- One row per session: the id of its FIRST message. That message is the only
      -- prompt the router ever decides on, because a session that has sent a full tool
      -- list can no longer prune (18.22), so it is the only prompt worth training on.
      --
      -- The id rather than the row: BigQuery cannot ARRAY_AGG a STRUCT that holds an
      -- ARRAY, and `v` is one. invocation_id breaks the timestamp tie so the pick is
      -- deterministic across runs.
      SELECT
        session_id,
        ARRAY_AGG(invocation_id ORDER BY prompt_ts, invocation_id LIMIT 1)[OFFSET(0)]
          AS invocation_id
      FROM `%s`
      WHERE prompt_ts >= @since AND ARRAY_LENGTH(v) = @dims
      GROUP BY session_id
    ),
    first_message AS (
      SELECT
        f.session_id,
        f.v,
        -- The train/eval split key. Normalisation is deliberately the three edits nobody
        -- has to think about — lowercase, collapse runs of whitespace, trim the ends —
        -- because a split key that is clever is a split key nobody can reproduce by hand
        -- when a run looks wrong. Stemming or punctuation-stripping would fold openings
        -- that a human would not call the same question.
        --
        -- IFNULL to session_id: prompt_text is NOT NULL by the time it reaches here (step 1
        -- filters it), and a NULL key would make split_col NULL rather than raise, which
        -- BQML's CUSTOM split has no defined answer for. One IFNULL is cheaper than
        -- finding that out from a training run.
        IFNULL(TRIM(REGEXP_REPLACE(LOWER(f.prompt_text), r'\\s+', ' ')), f.session_id)
          AS opening_key
      FROM `%s` f
      JOIN opening o USING (invocation_id)
    ),
    attempted AS (
      -- The label set (19.2): every tool the session TRIED. Errors count, unlike the
      -- per-invocation label in step 1, and the inversion is deliberate. A tool the
      -- model called and got an error from still had to be DECLARED for the call to
      -- happen, and what this model predicts is a declaration list.
      SELECT session_id, tool_name
      FROM `%s.%s.v_tool_completed`
      WHERE agent = @agent_name AND timestamp >= @since AND status = 'OK'
        AND tool_name != 'transfer_to_agent'
        AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
      UNION DISTINCT
      SELECT session_id, tool_name
      FROM `%s.%s.v_tool_error`
      WHERE agent = @agent_name AND timestamp >= @since
        AND tool_name != 'transfer_to_agent'
    ),
    dropped_sessions AS (
      -- Whole sessions, because the label is now a whole session's set and one bad row
      -- corrupts the set rather than itself.
      --
      -- Cache hits: a tool served from memory leaves no call in `attempted`, so the
      -- session looks like it wanted less than it did. The old per-invocation version of
      -- this filter is at step 1 and the reasoning is the same (11.5).
      SELECT DISTINCT session_id
      FROM `%s.%s.v_tool_completed`
      WHERE agent = @agent_name AND timestamp >= @since
        AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NOT NULL
      UNION DISTINCT
      SELECT DISTINCT session_id
      FROM `%s.%s.agent_events`
      WHERE agent = @agent_name AND timestamp >= @since
        AND JSON_VALUE(content, '$.result._smart_router.cache_hit') IS NOT NULL
      UNION DISTINCT
      -- Routed sessions: the ratchet (11.4). A session that was pruned could only call
      -- the tools it was left with, so its set is censored by the router's own decision
      -- and training on it teaches the next model to agree with this one. The holdout
      -- arm exists so there is always uncensored traffic to learn from; this is the
      -- filter that spends it. 'fresh' and 'reused' are the sources that mean a prune
      -- actually happened; they are provenance.ROUTED_SOURCES, and the copy below is
      -- pinned to it by tests/test_decision_sources.py. A third, 'grown', went with the
      -- pruner's growth path; it was never written to the warehouse, so this censors the
      -- same sessions it always did. Do not re-add it from an old schema.
      SELECT DISTINCT e.session_id
      FROM `%s.%s.agent_events` AS e,
        UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
      WHERE e.agent = @agent_name AND e.timestamp >= @since
        AND JSON_VALUE(record, '$.tool_name') IS NULL
        AND JSON_VALUE(record, '$.decision_source') IN ('fresh', 'reused')
    ),
    corpus AS (
      -- One row per (session, tool the session wanted), the vector repeating across a
      -- session's tools. That repetition is the whole trick: a softmax fitted on it
      -- learns a distribution over the set rather than a single winner, and `menus.rank`
      -- scores a menu by the probability mass it contains, so the two agree by
      -- construction (19.3).
      --
      -- A session that called nothing is one ABSTAIN row and is kept, not dropped.
      -- Those are the sessions where every removal is free (19.5).
      --
      -- Sessions are unweighted, so one that wanted four tools casts four votes where
      -- one that wanted a single tool casts one. BQML boosted trees take no per-row
      -- weight, and on this corpus a session wants 1.1 tools on average (19.10), so the
      -- distortion is small. It will not stay small on an agent with real tool chains.
      SELECT f.session_id, IFNULL(a.tool_name, 'ABSTAIN') AS tool_name, f.v, f.opening_key
      FROM first_message f
      LEFT JOIN attempted a USING (session_id)
      WHERE f.session_id NOT IN (SELECT session_id FROM dropped_sessions)
    ),
    class_counts AS (
      SELECT tool_name, COUNT(*) AS n FROM corpus GROUP BY tool_name
    ),
    rare_sessions AS (
      -- Sessions touching a class below the per-class floor. Whole SESSIONS, not rows:
      -- pinning individual rows would put near-duplicate prompts from one session on
      -- both sides of the split, which is the exact leak the custom split exists to
      -- prevent.
      SELECT DISTINCT session_id
      FROM corpus JOIN class_counts USING (tool_name)
      WHERE n < @min_class_rows
    )
    SELECT
      session_id,
      tool_name,
      -- CUSTOM split column. BQML contract: TRUE means evaluation, and the column is
      -- excluded from the features automatically. `opening_key` is NOT projected below,
      -- so it never becomes one either.
      --
      -- NOT RANDOM (9.4). A random split over conversational telemetry puts
      -- near-duplicate prompts from one session on both sides, and session-stable
      -- routing produces many correlated rows per session, so reported accuracy is
      -- inflated and the gate is then calibrated on the inflated number.
      --
      -- HASHED ON THE OPENING TEXT, NOT session_id. Hashing the session id makes every
      -- session its own unit, which is the wrong unit: two sessions that open with the
      -- same words are the same question asked twice, and a session hash puts one on each
      -- side. Measured on corpus-20260904, 88 of 409 eval sessions, 21.5 per cent, had a
      -- byte-identical opening sitting in train, so the gate was partly scoring a lookup
      -- of prompts the model had already been shown. Hashing the normalised text pins all
      -- of them to one side. It is the text and not an intent id because the warehouse has
      -- no intent id and an adopter's traffic never will; identical wording is the only
      -- duplicate signal that exists in telemetry.
      --
      -- THE LEAK WAS WORTH NOTHING, MEASURED (19.35). This change was expected to drop
      -- baseline_coverage and lift, because the old numbers were measured with part of
      -- the answer key in the training half. Run on the same 9,043 rows with only the
      -- split key changed, coverage went 0.9730 to 0.9753 and lift 0.1059 to 0.1057. The
      -- model reads an embedding, and a duplicate opening's neighbours already carry its
      -- label, so memorising it and generalising to it give the same answer. Keep the
      -- opening split anyway: on real traffic two identical questions need not want the
      -- same tools, and this costs nothing.
      --
      -- MOD before ABS: ABS(FARM_FINGERPRINT(x)) overflows on INT64 min.
      (session_id NOT IN (SELECT session_id FROM rare_sessions)
        AND ABS(MOD(FARM_FINGERPRINT(opening_key), 100)) < @eval_pct) AS split_col,
      %s
    FROM corpus
  """, flat_table, features_table, features_table,
       project, dataset, project, dataset,
       project, dataset, project, dataset, project, dataset,
       flatten_cols)
  USING run_since AS since,
        agent AS agent_name,
        embedding_dims AS dims,
        min_class_rows AS min_class_rows,
        eval_pct AS eval_pct;

  -- Rare classes are never DROPPED, only floored (9.4, and the first ratchet in 11.4).
  -- A tool that pruning has starved of traffic must be able to recover, and a 200-row
  -- floor beside a 1000-row training floor lets at most five classes survive, which can
  -- make the model worse than the static top-5 baseline by construction.
  --
  -- The floor is enforced two ways, and deliberately NOT by upsampling:
  --   1. no class is filtered out, at any support;
  --   2. sessions carrying a sub-floor class go entirely to the training side, so the
  --      class is learnable at all rather than stranded in the eval split by the hash.
  -- Duplicating rare rows would be a no-op here: auto_class_weights weights a class by
  -- 1/count, so total gradient mass is invariant to duplication. It would cost money and
  -- change nothing.
  --
  -- The cost of (2) is that sub-floor classes are not measured by the gate. This comment
  -- used to say that made coverage "slightly optimistic, bounded by their share of
  -- traffic", which named the mechanism and got the damage backwards. Both arms are
  -- optimistic, but the STATIC BASELINE gains far more than the model does, because
  -- sub-floor classes are exactly the rows a top-N frequency ranking never covers. At a
  -- floor of 30 that was 157 rows, 7% of tool-calling rows, and it moved the baseline
  -- from 0.8297 over the corpus to 0.8926 on eval. Criterion 2 measures a difference, so
  -- the whole 6.3 points came straight off the lift and the first run failed at 0.0895
  -- with the model sitting on the k-NN ceiling. Hence the floor of 10 (9.4), and hence
  -- baseline_corpus below, so the next occurrence is visible in the run record.

  EXECUTE IMMEDIATE FORMAT("""
    SELECT
      COUNT(*),
      COUNT(DISTINCT session_id),
      COUNTIF(NOT split_col),
      COUNTIF(split_col),
      COUNT(DISTINCT IF(tool_name = 'ABSTAIN', NULL, tool_name)),
      ARRAY_AGG(DISTINCT tool_name ORDER BY tool_name)
    FROM `%s`
  """, flat_table)
  INTO total_rows, total_sessions, train_rows, eval_rows, tool_classes, class_labels;

  -- ==================================================================================
  -- 4. Gate criterion 3 first — it needs no model (DESIGN 10).
  -- ==================================================================================
  -- Checking data sufficiency before training saves a whole Vertex training job on a
  -- corpus that could never have been published. ABSTAIN is excluded from the class
  -- count: one real tool plus ABSTAIN is two model classes but a degenerate router.
  --
  -- SESSIONS, not rows. The floor is 1,000 sessions (19.5 criterion 4); "1,000 rows" is
  -- v1 wording from when a row WAS a session. Under the v2 label a session contributes one
  -- row per tool it wanted, so counting rows scales the floor by the agent's tools-per-
  -- session and lets a thin corpus through: on corpus-20260904 that ratio is 1.37, making
  -- the row floor a ~730-session floor, and on an agent whose sessions want three tools it
  -- would be a ~333-session floor. The row count is still the run record's `n_rows`,
  -- because that is what the model was fitted on, and `n_sessions` beside it is the number
  -- this floor actually tested, so a rejected run says what it was rejected on.
  IF total_sessions < 1000 THEN
    SET gate_reason = 'insufficient_sessions';
  END IF;
  IF tool_classes < 2 THEN
    SET gate_reason = IF(gate_reason = '', '', gate_reason || ',') || 'insufficient_classes';
  END IF;

  IF gate_reason = '' THEN
    -- ================================================================================
    -- 5. Train (DESIGN 9.4).
    -- ================================================================================
    EXECUTE IMMEDIATE FORMAT("""
      CREATE OR REPLACE MODEL `%s`
      OPTIONS (
        model_type = 'BOOSTED_TREE_CLASSIFIER',
        input_label_cols = ['tool_name'],
        max_tree_depth = 6,
        auto_class_weights = TRUE,
        early_stop = TRUE,
        data_split_method = 'CUSTOM',
        data_split_col = 'split_col'
      ) AS
      SELECT * EXCEPT (session_id) FROM `%s`
    """, model_name, flat_table);

    -- ================================================================================
    -- 6. Score the held-out sessions.
    -- ================================================================================
    -- The static comparator is fitted on the TRAINING side only. Fitting it on the eval
    -- rows would let the baseline peek and would understate the lift.
    EXECUTE IMMEDIATE FORMAT("""
      CREATE OR REPLACE TEMP TABLE gate_baseline AS
      SELECT tool_name
      FROM `%s`
      WHERE NOT split_col AND tool_name != 'ABSTAIN'
      GROUP BY tool_name
      ORDER BY COUNT(*) DESC
      LIMIT @top_n
    """, flat_table)
    USING top_n AS top_n;

    EXECUTE IMMEDIATE FORMAT("""
      CREATE OR REPLACE TEMP TABLE gate_scored AS
      SELECT
        tool_name AS actual,
        (SELECT p.label FROM UNNEST(predicted_tool_name_probs) p
         ORDER BY p.prob DESC, p.label LIMIT 1) AS argmax_label,
        -- Coverage@N: is the true tool inside the set we would have kept?
        -- ABSTAIN is stripped from the ranking before the cut, because ABSTAIN is never
        -- an edge action (D10) and so never occupies one of the N slots.
        -- Ties are counted as covered; at N=5 this is a rounding-level effect.
        EXISTS(
          SELECT 1
          FROM UNNEST(predicted_tool_name_probs) hit
          WHERE hit.label = tool_name
            AND (SELECT COUNT(*) FROM UNNEST(predicted_tool_name_probs) other
                 WHERE other.label != 'ABSTAIN' AND other.prob > hit.prob) < @top_n
        ) AS covered_model,
        -- Coverage under the rule `pruner.top_k_by_mass` applies when the artifact ships no
        -- menus. Three differences from the line above, and all three are deliberate copies
        -- of the serving code rather than improvements on it:
        --
        --   * ABSTAIN is NOT stripped. It occupies a slot and contributes mass like any other
        --     label, because that is what the pruner does. Measured, this costs 0.4 points.
        --   * the tie-break is by label ascending, matching `top_k_by_mass`, so a tie is
        --     resolved the same way here and there rather than counted as covered.
        --   * the cut stops once the labels ranked BEFORE this one already hold
        --     `keep_probability_mass`. The label that crosses the threshold is kept, which is
        --     why the condition is on the mass before it and not including it.
        --   * that last comparison carries `pruner.MASS_TOLERANCE`, 1e-9 of slack. A vector
        --     whose mass before a label is exactly the threshold is common, and the two sides
        --     add it up differently: the pruner accumulates left to right in Python, BigQuery
        --     sums in whatever order it likes. Without the slack the two disagree in the last
        --     bit and this column scores a set the router did not serve, which is the one
        --     thing it exists not to do.
        --
        -- This never gates. It is here so the run record says what the menu-less path would
        -- serve, which until DESIGN 19.74 no number anywhere did.
        EXISTS(
          SELECT 1
          FROM UNNEST(predicted_tool_name_probs) hit
          WHERE hit.label = tool_name
            AND (SELECT COUNT(*) FROM UNNEST(predicted_tool_name_probs) other
                 WHERE other.prob > hit.prob
                    OR (other.prob = hit.prob AND other.label < hit.label)) < @top_n
            AND (SELECT IFNULL(SUM(other.prob), 0)
                 FROM UNNEST(predicted_tool_name_probs) other
                 WHERE other.prob > hit.prob
                    OR (other.prob = hit.prob AND other.label < hit.label)) < @mass - 1e-9
        ) AS covered_mass,
        tool_name IN (SELECT tool_name FROM gate_baseline) AS covered_static
      FROM ML.PREDICT(MODEL `%s`, (SELECT * FROM `%s` WHERE split_col))
    """, model_name, flat_table)
    USING top_n AS top_n, keep_probability_mass AS mass;

    -- Coverage is measured over eval rows that actually called a tool. An ABSTAIN row
    -- has no true tool, so "the true tool is in the kept set" is not defined for it, and
    -- scoring it as trivially covered would inflate both arms and shrink the lift.
    --
    -- Under the v2 label a row is a (session, tool it wanted) pair, so this quantity is
    -- set recall (19.4) weighted by set size: a session wanting four tools contributes
    -- four rows. It is NOT gate criterion 1, for the reason spelled out at the gate below
    -- (19.11): the criterion is defined on the MENU the router ships, menus are chosen
    -- after training, and the raw ranking measured here is harsher than the menu. It
    -- gates only through criterion 2, and it is reported because a coverage that moves
    -- between retrains is the first sign that the corpus did.
    SET (coverage, coverage_mass, baseline, abstain_precision, macro_recall) = (
      SELECT AS STRUCT
        SAFE_DIVIDE(COUNTIF(covered_model AND actual != 'ABSTAIN'), COUNTIF(actual != 'ABSTAIN')),
        SAFE_DIVIDE(COUNTIF(covered_mass AND actual != 'ABSTAIN'), COUNTIF(actual != 'ABSTAIN')),
        SAFE_DIVIDE(COUNTIF(covered_static AND actual != 'ABSTAIN'), COUNTIF(actual != 'ABSTAIN')),
        SAFE_DIVIDE(COUNTIF(argmax_label = 'ABSTAIN' AND actual = 'ABSTAIN'),
                    COUNTIF(argmax_label = 'ABSTAIN')),
        (SELECT AVG(recall) FROM (
           SELECT SAFE_DIVIDE(COUNTIF(argmax_label = actual), COUNT(*)) AS recall
           FROM gate_scored GROUP BY actual))
      FROM gate_scored
    );

    -- The same top-N set, scored over EVERY row instead of the eval split. It never gates:
    -- the model has seen the training rows, so a lift computed against it would not be out
    -- of sample. Its job is diagnostic. When this figure and `baseline` disagree, the split
    -- is handing the baseline an easier distribution than the corpus, and criterion 2 is
    -- measuring the split rather than the model (DESIGN 9.4, 10). The first run's failure
    -- was 0.8297 here against 0.8926 there, and reading it required a hand-written query.
    EXECUTE IMMEDIATE FORMAT("""
      SELECT SAFE_DIVIDE(
        COUNTIF(tool_name IN (SELECT tool_name FROM gate_baseline)),
        COUNT(*))
      FROM `%s`
      WHERE tool_name != 'ABSTAIN'
    """, flat_table)
    INTO baseline_corpus;

    -- Criterion 2, as a share of the baseline's misses rather than an absolute gap
    -- (DESIGN 10). SAFE_DIVIDE guards the degenerate case where the static ranking
    -- covers everything: there are then no misses to remove, and a NULL here reads as
    -- a failure two lines below, which is the correct answer. A model cannot earn its
    -- keep against a baseline that never misses.
    SET miss_reduction = 1 - SAFE_DIVIDE(1 - coverage, 1 - baseline);

    -- ================================================================================
    -- 7. The promotion gate (DESIGN 10).
    -- ================================================================================
    -- Criterion 1 is NOT here, and that is the change of 19.11. It is defined on the MENU
    -- the router ships, and BigQuery has no menus: they are chosen after training, in
    -- publish_router.py. What this procedure can measure is the model's raw top-N ranking,
    -- which is strictly harsher than what ships — measured, the same model covers 0.9785
    -- ranking and 0.9895 by menu, because a menu holds tools that travel together and
    -- catches a session's second tool when the ranker put it eighth. Holding the raw
    -- number to criterion 1's threshold rejects models that pass criterion 1.
    --
    -- Coverage still gates, through criterion 2, and adaptively: miss_reduction >= 0.50
    -- means coverage cannot fall below halfway from the static baseline to 1.0, a floor
    -- that tightens on its own whenever the baseline gets easier. A model that cannot rank
    -- still cannot export. Only the hardcoded 0.98, which was criterion 1 living in the
    -- wrong place, is gone.
    --
    -- NULL coverage means the eval split was empty. IFNULL(...,0) makes that a failure
    -- rather than a NULL comparison that quietly evaluates to non-TRUE somewhere else,
    -- and it lands on criterion 2 below because miss_reduction is NULL with it.
    IF IFNULL(coverage, 0) <= 0 THEN
      SET gate_reason = IF(gate_reason = '', '', gate_reason || ',') || 'no_eval_coverage';
    END IF;
    -- Criterion 2 is the one that justifies the project existing: if a frequency ranking
    -- makes nearly the same mistakes, the entire ML stack is dead weight and the right
    -- product is a YAML file. Re-proven on every retrain, not once.
    --
    -- RELATIVE, not the absolute gap it used to be. Both criteria depend on top_n, and
    -- in opposite directions, so an absolute threshold made them jointly unsatisfiable:
    -- coverage crossed 0.98 between N=6 and N=7 and the 0.10 gap crossed in the same
    -- place from the other side. Miss reduction sits at 0.78 to 0.84 across N=3..12 on
    -- the same model, so it measures the model rather than the baseline's difficulty.
    IF IFNULL(miss_reduction, 0) < 0.50 THEN
      SET gate_reason = IF(gate_reason = '', '', gate_reason || ',') || 'miss_reduction_below_0.50';
    END IF;
  END IF;

  SET gate_passed = (gate_reason = '');

  -- ==================================================================================
  -- 8. Export (DESIGN 9.4, 11.6) — only a model that passed.
  -- ==================================================================================
  -- URI is the only option EXPORT MODEL takes. `destination_format` was here and BigQuery
  -- rejects it outright: `unsupported option destination_format`. There is nothing to
  -- select anyway, because the format follows from the model type — a boosted tree always
  -- exports as an XGBoost booster. The failure lands at the very end of a full training
  -- run, after the money is spent, and it aborts the procedure before the run record is
  -- written, so the run leaves no trace in `router_training_runs` (DESIGN 18.8).
  IF gate_passed THEN
    EXECUTE IMMEDIATE FORMAT("""
      EXPORT MODEL `%s`
      OPTIONS (URI = '%s')
    """, model_name, export_uri);
  END IF;

  -- ==================================================================================
  -- 9. Run record — the metadata source for the artifact.
  -- ==================================================================================
  -- metadata.json (11.6) needs: version, sha256, size_bytes, trained_at, rows, classes,
  -- backend, environment, embedding_model, embedding_dims, gate_metrics.
  --
  -- BigQuery produces every one of those EXCEPT sha256 and size_bytes, which are
  -- properties of the exported object rather than of the query. And the object this
  -- procedure exports is not the one that gets served: model.bst is XGBoost's pre-1.0
  -- binary format, which XGBoost 2.1 removed the ability to read (DESIGN 18.9).
  -- scripts/publish_router.py converts it to model.ubj, digests THAT, and writes
  -- metadata.json LAST: model_store treats a version directory as publishable only once
  -- both metadata.json and model.ubj exist, so completeness is the atomicity mechanism
  -- (D11). Neither file is written here, so a raw export is inert until published.
  --
  -- `classes` is emitted sorted, and the wrapper MUST NOT copy that order into
  -- metadata.json. The booster's probability vector is indexed by the order in the
  -- exported `assets/model_metadata.json`, which BigQuery writes in DESCENDING
  -- alphabetical order. Measured, not assumed (DESIGN 18.8). predictor.py zips classes
  -- against that vector positionally, so copying the sorted array here would mislabel
  -- nearly every prediction and raise nothing. `classes` in this table is therefore a
  -- SET to reconcile against, never an order to publish. scripts/publish_router.py does
  -- the reconciliation and refuses on a mismatch.
  EXECUTE IMMEDIATE FORMAT("""
    CREATE TABLE IF NOT EXISTS `%s` (
      run_id STRING,
      agent STRING,
      version STRING,
      trained_at TIMESTAMP,
      corpus_since TIMESTAMP,
      environment STRING,
      backend STRING,
      embedding_model STRING,
      embedding_dims INT64,
      n_rows INT64,             -- metadata.json calls this `rows`; ROWS is reserved in SQL
      n_sessions INT64,         -- what criterion 4 tests. `n_rows` is what the model was fitted on
      train_rows INT64,
      eval_rows INT64,
      pending_embed_rows INT64,
      n_tool_classes INT64,
      classes ARRAY<STRING>,
      top_n INT64,
      keep_probability_mass FLOAT64,     -- the other half of the served set size (DESIGN 19.74)
      coverage FLOAT64,                  -- strict top-N. What gates, through criterion 2.
      coverage_mass FLOAT64,             -- the menu-less served rule. Reported, never gates.
      baseline_coverage FLOAT64,         -- static top-N on the eval split. This is what gates.
      baseline_coverage_corpus FLOAT64,  -- same set over every row. Diagnostic only (DESIGN 10).
      lift FLOAT64,                      -- absolute gap. Logged for continuity, no longer gates.
      miss_reduction FLOAT64,            -- criterion 2 (DESIGN 10). This is what gates.
      abstain_precision FLOAT64,
      macro_recall FLOAT64,
      gate_passed BOOL,
      gate_reason STRING,
      export_uri STRING
    )
    PARTITION BY DATE(trained_at)
    CLUSTER BY agent
  """, runs_table);

  -- The runs table is CREATE IF NOT EXISTS, so a table created by an earlier version of this
  -- procedure keeps its old schema and the INSERT below would fail on the unknown column.
  -- ADD COLUMN IF NOT EXISTS makes the upgrade in place, and existing rows read NULL, which
  -- is honest: those runs did not measure it.
  EXECUTE IMMEDIATE FORMAT("""
    ALTER TABLE `%s`
      ADD COLUMN IF NOT EXISTS baseline_coverage_corpus FLOAT64,
      ADD COLUMN IF NOT EXISTS miss_reduction FLOAT64,
      ADD COLUMN IF NOT EXISTS n_sessions INT64,
      ADD COLUMN IF NOT EXISTS keep_probability_mass FLOAT64,
      ADD COLUMN IF NOT EXISTS coverage_mass FLOAT64
  """, runs_table);

  EXECUTE IMMEDIATE FORMAT("""
    INSERT INTO `%s` (
      run_id, agent, version, trained_at, corpus_since, environment, backend,
      embedding_model, embedding_dims, n_rows, n_sessions, train_rows, eval_rows,
      pending_embed_rows,
      n_tool_classes, classes, top_n, keep_probability_mass, coverage, coverage_mass,
      baseline_coverage, baseline_coverage_corpus, lift, miss_reduction,
      abstain_precision, macro_recall, gate_passed, gate_reason, export_uri)
    VALUES (
      GENERATE_UUID(), @agent, @version, CURRENT_TIMESTAMP(), @since, @environment,
      @backend, @embedding_model, @embedding_dims, @n_rows, @n_sessions, @train_rows,
      @eval_rows, @pending_rows, @n_tool_classes, @classes, @top_n, @mass, @coverage,
      @coverage_mass, @baseline, @baseline_corpus, @coverage - @baseline, @miss_reduction,
      @abstain_precision, @macro_recall, @gate_passed,
      IF(@gate_reason = '', 'ok', @gate_reason),
      IF(@gate_passed, @export_uri, CAST(NULL AS STRING)))
  """, runs_table)
  USING agent AS agent,
        version AS version,
        run_since AS since,
        environment AS environment,
        backend AS backend,
        embedding_model AS embedding_model,
        embedding_dims AS embedding_dims,
        total_rows AS n_rows,
        total_sessions AS n_sessions,
        train_rows AS train_rows,
        eval_rows AS eval_rows,
        pending_rows AS pending_rows,
        tool_classes AS n_tool_classes,
        class_labels AS classes,
        top_n AS top_n,
        keep_probability_mass AS mass,
        coverage AS coverage,
        coverage_mass AS coverage_mass,
        baseline AS baseline,
        baseline_corpus AS baseline_corpus,
        miss_reduction AS miss_reduction,
        abstain_precision AS abstain_precision,
        macro_recall AS macro_recall,
        gate_passed AS gate_passed,
        gate_reason AS gate_reason,
        export_uri AS export_uri;
END;
