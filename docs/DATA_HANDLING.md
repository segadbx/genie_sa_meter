# Data handling

This document explains what the harness collects, where it is stored, and how to remove
it. The harness is designed to be **safe to share**: by default it stores only metadata,
hashes, and lengths — not raw content — and it never sends benchmark content to an
external observability or evaluation service.

## What is collected

For each question × variant × repetition the harness records:

- **Identifiers**: `run_id`, `question_id`, `variant`, `repetition`, `conversation_id`,
  `trace_id`, and provider message/response ids.
- **Timings**: UTC start/end timestamps and client-measured latency.
- **Status**: `COMPLETED` / `FAILED` / `TIMEOUT` / `NOT_CONFIGURED`, plus warnings.
- **Token usage**: trace-level and child-span-summed input/output/total tokens, each with
  its provenance (`provider_reported` / `trace_attribute` / `child_span_sum` /
  `unavailable`). Token counts are telemetry, **never** billed cost.
- **Content metadata**: SHA-256 hash and character length of the response and generated
  SQL.
- **Call counts**: LLM, tool, MCP, and SQL counts where visible in traces.

## Content storage (off by default)

Full prompts, generated SQL, MCP arguments, tool results, and responses are **not stored**
unless you explicitly set `store_response_text: true` in the config. When disabled, only
the SHA-256 hash and character length are kept, so results can be correlated and compared
without retaining sensitive text.

## Redaction

Logs and persisted notes are redacted by default. The harness never prints access tokens,
authorization headers, cookies, signed URLs, or secret values, and it scrubs likely
secrets, emails, phone numbers, account identifiers, and bearer tokens. The
`benchmark redact` command produces a copy with content dropped and strings scrubbed for
sharing.

## Where it is stored

- Results are written to the configured `output_dir` (default `outputs/`) on the machine
  running the harness:
  `benchmark_results.json`, `benchmark_summary.csv`, `benchmark_report.md`,
  `trace_links.txt`, and a `preflight.ok` marker.
- MLflow traces (when a tracking backend is configured) are stored in the customer's
  MLflow experiment / trace destination — governed by the customer's workspace, not by
  this tool.
- Nothing is transmitted to any third-party service by the harness.

## How to remove it

```bash
rm -rf outputs/*            # removes all local results, reports, and the preflight marker
```

Traces stored in the customer's MLflow experiment or Unity Catalog trace destination are
managed and deleted through Databricks according to the customer's retention policy; this
tool does not delete them.

## Credentials

Credentials are **never** read from the config file (the loader refuses to start if it
detects them). They are resolved only from the environment, a Databricks authentication
profile, or a secret manager, and are never written to any output.

## Shareable vs. do-not-share

- **Share**: `README.md`, `config.example.yaml`, `questions.example.json`,
  `pyproject.toml`, the lock file, `src/`, `tests/`, `notebooks/`,
  `sql/billing_reconciliation.sql`, `docs/DATA_HANDLING.md`, `CHANGELOG.md`.
- **Do not share**: `.env`, your `config.yaml`, `questions.json`, live results under
  `outputs/`, raw traces, credentials, local virtual environments, and the internal
  reference docs under `docs/ref/` (the PRD and implementation plan).
