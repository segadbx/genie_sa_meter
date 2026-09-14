# Genie + Supervisor Benchmark Harness

A small, reproducible, **read-only** Python harness that compares three invocation paths
for the same benchmark questions — **direct Genie**, **Supervisor**, and
**Supervisor + MCP** — and preserves run-level evidence (traces, token usage, latency) so
another engineer can regenerate the report from the same inputs.

The harness is **configuration-driven, serial by default, and safe to share**. It does not
depend on undocumented internal endpoints, hard-coded workspace names, embedded
credentials, or hidden state. It answers: *why does a simple question appear expensive, and
where does the overhead come from?*

> This tool is read-only. It does not edit the Genie Agent, Supervisor Agent, MCP server,
> source data, benchmarks, or permissions. It cannot guarantee that an underlying agent is
> read-only — use read-only questions and a test identity.

## What it measures

For each question × variant it records latency, completion status, trace id, observed
token usage (trace-level **and** child-span sum), LLM/tool/MCP call counts, generated-SQL
metadata, and warnings — then produces a deterministic JSON/CSV/Markdown report.

## Requirements

- Python **3.10+**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- For a live run: a Databricks workspace, a Genie space id, and appropriate read access.
  Credentials are resolved from the environment or a Databricks profile — never from the
  config file.

## Install

```bash
uv sync                      # installs pinned runtime + dev dependencies from uv.lock
# or, without uv:
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

## One-command local test (offline, no network)

```bash
uv run pytest                # unit + fixture tests; makes no live calls
```

## One-command dry run (validate config + questions, invoke nothing)

```bash
cp config.example.yaml config.yaml       # then fill in placeholders
cp questions.example.json questions.json
uv run benchmark run --config config.yaml --dry-run
```

## Customer run procedure

```bash
# 1. Configure (never put credentials in config.yaml)
cp config.example.yaml config.yaml       # edit workspace_host, genie_space_id, variants, ...
cp questions.example.json questions.json # edit approved, read-only questions

# 2. Authenticate via a Databricks profile or environment (see .env.example)
#    e.g. a profile in ~/.databrickscfg referenced by `databricks_profile:` in config.yaml

# 3. Preflight (prints resolved, non-secret identifiers; writes a preflight marker)
uv run benchmark preflight --config config.yaml

# 4. Run (fails closed unless preflight passed for this exact config)
uv run benchmark run --config config.yaml

# 5. (Re)generate the report from persisted results at any time
uv run benchmark report --results outputs/benchmark_results.json

# 6. Produce a shareable, redacted results file
uv run benchmark redact --input outputs/benchmark_results.json --output outputs/redacted_results.json
```

## Run from a Databricks notebook

`notebooks/run_benchmark.py` is a Databricks notebook (source format) that runs the same
harness inside a workspace using the notebook's ambient credentials — no profile or token
needed. Add the repo as a Git folder (or sync it to a Workspace folder), open the notebook,
fill in the widgets (Genie space id, variants, …), and run the cells top to bottom:
install → preflight → calibration → benchmark → report. Stop conditions are listed at the
top of the notebook, and outputs are written next to it under `outputs/`.

## Outputs (written to `output_dir`, default `outputs/`)

| File | Contents |
|---|---|
| `benchmark_results.json` | Complete normalized records, warnings, and trace references. |
| `benchmark_summary.csv` | One deterministic row per question/variant/repetition. |
| `benchmark_report.md` | Customer-readable findings, measurement categories, limitations. |
| `trace_links.txt` | Trace ids only — never credentials or signed URLs. |

## Variants and the customer boundaries

- **`direct_genie`** is implemented against the documented Genie Conversation API
  (`start-conversation` → poll `get-message` → collect SQL/response).
- **`supervisor`** and **`supervisor_mcp`** are **customer boundaries**. The harness does
  not invent an API shape; these adapters report a clear prerequisite until you wire the
  customer-approved invocation in `src/genie_benchmark/adapters/supervisor.py` and
  `supervisor_mcp.py`. See those files for exactly what to implement.

## Trace and billing

The harness reads trace-level token usage **and** sums child LLM/Chat spans, recording the
provenance of each value (the trace UI can understate totals when child spans omit usage).
It never converts token counts into billed cost. Billing reconciliation is a separate step
using `sql/billing_reconciliation.sql` against `system.billing.usage`.

## Data handling

See [`docs/DATA_HANDLING.md`](docs/DATA_HANDLING.md) for what is collected, where it is stored, and
how to remove it. By default, full prompts/SQL/responses are **not** stored — only hashes
and length metadata. Set `store_response_text: true` to retain content.

## Opt-in integration test

Live tests are skipped unless you opt in:

```bash
GENIE_BENCHMARK_INTEGRATION=1 GENIE_BENCHMARK_INTEGRATION_CONFIG=config.yaml \
  uv run pytest -m integration
```

## Project layout

The harness lives in `src/genie_benchmark/` (adapters under `adapters/`), with tests in
`tests/`, the notebook in `notebooks/`, and billing SQL in `sql/`. Changes to adapter or
report schemas are recorded in [`CHANGELOG.md`](CHANGELOG.md). The full design and original
runbook are kept as internal reference under `docs/ref/` (not part of the shared bundle).
