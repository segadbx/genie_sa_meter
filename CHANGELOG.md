# Changelog

All notable changes to the adapter contract and the report schema are recorded here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-09-14

### Added

- Initial benchmark harness: `preflight`, `run`, `report`, and `redact` CLI commands.
- Typed contracts (`models.py`): `BenchmarkQuestion`, `RunContext`, `InvocationResult`,
  `TokenUsage` (with an explicit `source` provenance), and `RunStatus`.
- Adapter contract (`adapters/base.py`) with three adapters:
  - `direct_genie` — implemented against the documented Genie Conversation API.
  - `supervisor` and `supervisor_mcp` — typed customer boundaries that report a
    prerequisite until the customer-approved invocation is wired in.
- Strict config loader with secret refusal, env overrides, serial-by-default concurrency,
  and a non-secret preflight report.
- Question validation (required fields, duplicate ids, follow-up references) with a stable
  SHA-256 of the input file recorded in run metadata.
- Defensive token normalization: trace-level value **and** child-span sum, with conflict
  and partial-usage warnings.
- Deterministic reporting: `benchmark_results.json`, `benchmark_summary.csv`,
  `benchmark_report.md`, `trace_links.txt`, sorted by `question_id, variant, repetition`.
- Billing reconciliation SQL template (`sql/billing_reconciliation.sql`) — kept separate
  from trace telemetry.
- Offline test suite (unit + fixture) and an opt-in live integration test.
- Databricks notebook (`notebooks/run_benchmark.py`) that runs the harness in a workspace
  using ambient notebook credentials (preflight → calibration → benchmark → report).

### Report schema (v0.1.0)

`benchmark_summary.csv` columns:
`question_id, variant, repetition, status, latency_ms, trace_level_total_tokens,
child_span_total_tokens, token_source, llm_call_count, tool_call_count, mcp_call_count,
response_chars, quality_score, trace_id, notes`
