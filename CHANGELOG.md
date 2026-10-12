# Changelog

All notable changes to `adk-smart-router` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-12

### Added
- **Runner-level `SmartRouterPlugin` (`adk_smart_router`):**
  - Turn-one prompt embedding (`gemini-embedding-2` on Vertex AI `global`) and session-pinned tool declaration pruning (`PrefixLock`) that preserves Gemini implicit context caching across multi-turn conversations.
  - Three composable judge tiers (`RouterConfig.judges` / `SMART_ROUTER_JUDGES`):
    - `"prototype"` (`PrototypeJudge`): Zero-training-data cold-start semantic router using tool docstring + JSON schema embeddings, catalog mean-centering, 1-hop upstream `Returns:` $\to$ `Args:` prerequisite graph propagation, and optional zero-data canonical menu synthesis (`cold_start_menu_count`).
    - `"xgboost"` (`Predictor`): In-memory XGBoost booster exported from BigQuery ML (`model.ubj` + `metadata.json`) with greedy max-coverage canonical menu snapping.
    - `("xgboost", "prototype")`: Hybrid combiner sharing a single Turn-1 prompt embedding across both judges to score trained classes via XGBoost and rescue/prune newly deployed (`classless`) tools via `PrototypeJudge`.
  - Context-cache floor protection (`MIN_CACHE_TOKENS` + `CACHE_FLOOR_HEADROOM`) and live `CacheWatch` automatic circuit breaker comparing routed vs. holdout billed input tokens.
  - Principal-scoped in-memory `ToolResultCache` (`cacheable_tools`) keyed by SHA-256 over `(app_name, user_id, agent_name, tool_name, tool_version, canonical_args)`, with automatic read-your-own-writes invalidation and runtime `EventActions` mutation blocklisting.
  - Startup pre-warming (`await plugin.warmup(app)`) for embedder connection pooling and tool-catalog prototype embeddings.
- **`smart-router` CLI (`adk_smart_router.cli`):**
  - `smart-router qualify`: Pre-provisioning offline prompt-floor (`PAYS`, `THIN`, `UNCACHED`, `TRAP`) and `Returns:` $\to$ `Args:` argument-reachability audit.
  - `smart-router readiness`: BigQuery Agent Analytics corpus readiness check before training.
  - `smart-router doctor`: Static wiring verification for plugin registration order, `custom_metadata_allowlist`, `ContextCacheConfig`, and active judge configuration.
  - `smart-router check-arguments`: Empirical audit of `--arguments` markings against logged first-turn user messages.
  - `smart-router status`: Live model artifact, 7-day billed-token delta vs. holdout, feature drift, and retrain ledger inspection.
- **BigQuery ML Training & Operations Suite (`sql/` and `scripts/`):**
  - 5-stage BigQuery ML training stored procedure (`sql/train_router.sql`), dashboard views (`sql/dashboard.sql`), and autonomous daily retraining/withdrawal cycle (`scripts/run_cycle.sh`, `scripts/publish_router.py`, `scripts/tune_settings.py`, `scripts/watch_generations.py`).
- **Agent Plugin & Skill (`plugin.json` and `skills/adk-smart-router/SKILL.md`):**
  - Google Agent Plugins v1.0.0 manifest and 5-step agentic integration playbook for AI coding assistants.
