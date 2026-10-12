# Security Policy

## Threat Model

See [`docs/threat-model.md`](docs/threat-model.md) for the complete security architecture and operator checklist covering:
- Subset-only tool menu pruning (`kept <= live`) and `never_prune` invariants
- Cold-start, hybrid, and external judge bounds (`PrototypeJudge` menu-cache / docstring-churn caps and `LiveLlmJudge` system-instruction separation)
- Principal-scoped tool result caching (`user_id` inside SHA-256 cache keys, `tool_context.actions` side-effect blocklisting, and automatic bypass for agents declaring `before_tool_callback`)
- Cloud Storage artifact supply-chain verification (environment matching, pre-download byte cap, SHA-256 verification, and UBJSON format validation)

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in `adk-smart-router`, please report it privately via [GitHub Security Advisories](https://github.com/statmike/adk-smart-router/security/advisories/new) rather than opening a public issue.

Please include:
- A description of the issue and the affected module (`src/adk_smart_router/...`)
- Steps or a minimal test case to reproduce the behavior
- Any relevant configuration (`RouterConfig` settings)
