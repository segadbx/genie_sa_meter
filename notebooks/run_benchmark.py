# Databricks notebook source
# MAGIC %md
# MAGIC # Genie + Supervisor Benchmark — session runbook
# MAGIC
# MAGIC Runs the **read-only** benchmark harness inside a Databricks notebook and produces a
# MAGIC portable JSON/CSV/Markdown report. Compares `direct_genie`, `supervisor`, and
# MAGIC `supervisor_mcp` for the same questions to see where the overhead comes from.
# MAGIC
# MAGIC > **Read-only.** This notebook does not edit the Genie Agent, Supervisor Agent, MCP
# MAGIC > server, source data, benchmarks, or permissions. Use a test identity and read-only
# MAGIC > questions.
# MAGIC
# MAGIC ### Stop conditions (visible up front — halt the session if any occur)
# MAGIC - A request is routed to an **unexpected data source**.
# MAGIC - A trace is **absent** or cannot be associated with the request.
# MAGIC - The response indicates an **authorization** problem.
# MAGIC - A **budget / usage guardrail** is reached.
# MAGIC - A request **loops, fans out unexpectedly, or exceeds the timeout**.
# MAGIC
# MAGIC The harness stops the run automatically on authorization / budget / loop signals and on
# MAGIC the configured `max_runtime_seconds`; the calibration cell below is where you eyeball
# MAGIC routing and traces before running the full set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies
# MAGIC `mlflow` and `databricks-sdk` are usually present on Databricks Runtime; install
# MAGIC anyway to pin the versions this harness was tested with, then restart Python.

# COMMAND ----------

# MAGIC %pip install -q pyyaml "databricks-sdk>=0.30.0" "mlflow>=2.15.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Put the harness on the path
# MAGIC This notebook lives in `notebooks/` inside the repo (a Databricks Git folder or a
# MAGIC synced Workspace folder). The package source is in `../src`.

# COMMAND ----------

import os
import sys

# In a Databricks Git folder / Workspace folder, the notebook's directory is the CWD.
_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
_src = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import genie_benchmark  # noqa: E402

print("genie_benchmark version:", genie_benchmark.__version__)
print("repo root:", _repo_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parameters
# MAGIC Fill these in with widgets. **Never** put a token or secret here — the notebook uses
# MAGIC the notebook's ambient Databricks credentials (a test identity).

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "", "Genie space / agent id")
dbutils.widgets.text("experiment_id", "", "MLflow experiment id (optional)")
dbutils.widgets.text("supervisor_target", "", "Supervisor target (if using supervisor)")
dbutils.widgets.text("mcp_target", "", "MCP target (if using supervisor_mcp)")
dbutils.widgets.multiselect(
    "variants", "direct_genie", ["direct_genie", "supervisor", "supervisor_mcp"], "Variants"
)
dbutils.widgets.text("questions_file", "../questions.json", "Questions file (repo-relative)")
dbutils.widgets.text("max_questions", "8", "Max questions")
dbutils.widgets.text("environment", "notebook-session", "Environment label")
dbutils.widgets.dropdown("store_response_text", "false", ["false", "true"], "Store response text")

# COMMAND ----------

# Resolve the workspace host from the notebook context (no scheme prefix from Spark conf).
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", None)  # type: ignore[name-defined]  # noqa: F821
workspace_host = f"https://{_ws_url}" if _ws_url else ""

questions_file = os.path.abspath(
    os.path.join(os.getcwd(), dbutils.widgets.get("questions_file"))
)

config_dict = {
    "workspace_host": workspace_host,
    "use_default_auth": True,  # use the notebook's ambient credentials (no profile)
    "environment": dbutils.widgets.get("environment"),
    "genie_space_id": dbutils.widgets.get("genie_space_id"),
    "experiment_id": dbutils.widgets.get("experiment_id") or None,
    "supervisor_target": dbutils.widgets.get("supervisor_target") or None,
    "mcp_target": dbutils.widgets.get("mcp_target") or None,
    "variants": [v for v in dbutils.widgets.get("variants").split(",") if v],
    "questions_file": questions_file,
    "max_questions": int(dbutils.widgets.get("max_questions")),
    "repetitions": 1,
    "concurrency": 1,
    "store_response_text": dbutils.widgets.get("store_response_text") == "true",
    "output_dir": os.path.join(os.getcwd(), "outputs"),
}
config_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preflight — validate config + questions (no agent invoked)
# MAGIC Prints resolved, non-secret identifiers. Fails closed on missing config, refuses to
# MAGIC start if credentials are present, and rejects invalid questions.

# COMMAND ----------

import json

from genie_benchmark.config import build_preflight_report, parse_config
from genie_benchmark.questions import load_questions

config = parse_config(config_dict, source_path="<notebook>")
questions, question_hash = load_questions(config.questions_file)

report = build_preflight_report(config)
report["question_count"] = len(questions)
report["question_file_hash"] = question_hash
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set the MLflow experiment (optional)
# MAGIC Traces from the harness boundary land in this experiment. If left blank, the notebook's
# MAGIC default experiment is used.

# COMMAND ----------

import mlflow

if config.experiment_id:
    mlflow.set_experiment(experiment_id=config.experiment_id)
    print("MLflow experiment set:", config.experiment_id)
else:
    print("Using the notebook's default MLflow experiment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build adapters
# MAGIC `direct_genie` calls the documented Genie Conversation API using the notebook's
# MAGIC credentials. `supervisor` / `supervisor_mcp` are customer boundaries: they report a
# MAGIC clear prerequisite until the approved invocation is wired into
# MAGIC `src/genie_benchmark/adapters/`.

# COMMAND ----------

from genie_benchmark.adapters.direct_genie import DirectGenieAdapter, build_default_client
from genie_benchmark.adapters.supervisor import SupervisorAdapter
from genie_benchmark.adapters.supervisor_mcp import SupervisorMcpAdapter

adapters = {}
for variant in config.variants:
    if variant == "direct_genie":
        adapters[variant] = DirectGenieAdapter(build_default_client(config), config)
    elif variant == "supervisor":
        adapters[variant] = SupervisorAdapter(config)
    elif variant == "supervisor_mcp":
        adapters[variant] = SupervisorMcpAdapter(config)
list(adapters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Calibration — run the first question once through each variant
# MAGIC Inspect routing, traces, and status **before** the full run. Stop the session if any
# MAGIC stop condition above is hit.

# COMMAND ----------

from genie_benchmark.models import RunContext

calib_ctx = RunContext(
    run_id="calibration",
    environment=config.environment,
    request_timeout_seconds=config.request_timeout_seconds,
    poll_interval_seconds=config.poll_interval_seconds,
    store_response_text=config.store_response_text,
)
for variant, adapter in adapters.items():
    result = adapter.invoke(questions[0], calib_ctx)
    print(f"[{variant}] status={result.status.value} latency_ms={result.client_metrics.get('latency_ms')} "
          f"trace_id={result.trace_id} warnings={result.warnings}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Benchmark — run all questions serially
# MAGIC Each result is persisted immediately, so an interrupted run still yields evidence.

# COMMAND ----------

import uuid

from genie_benchmark.runner import run_benchmark

run_id = str(uuid.uuid4())
results_path = os.path.join(config.output_dir, "benchmark_results.json")
store = run_benchmark(
    config, questions, question_hash, adapters, run_id=run_id, results_path=results_path
)
print(f"run {run_id}: status={store.metadata.status} stop_reason={store.metadata.stop_reason}")
print("counts:", store.metadata.counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Report — generate and display the outputs
# MAGIC Writes `benchmark_results.json`, `benchmark_summary.csv`, `benchmark_report.md`, and
# MAGIC `trace_links.txt` to the output directory.

# COMMAND ----------

from genie_benchmark.reporting import generate_reports

written = generate_reports(results_path, config.output_dir)
written

# COMMAND ----------

# Render the Markdown report.
with open(written["report"], encoding="utf-8") as f:
    report_md = f.read()
displayHTML(f"<pre style='white-space:pre-wrap'>{report_md}</pre>")  # noqa: F821

# COMMAND ----------

# Show the summary as a DataFrame for sorting/filtering.
import pandas as pd

display(pd.read_csv(written["summary"]))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Review checklist
# MAGIC 1. **Direct Genie vs Supervisor** — orchestration overhead.
# MAGIC 2. **Supervisor vs Supervisor + MCP** — MCP connection / tool overhead.
# MAGIC 3. **Token usage by LLM span** — planning, routing, Genie, synthesis.
# MAGIC 4. **Tool/MCP call count and payload sizes** — context / fan-out overhead.
# MAGIC 5. **Quality vs overhead** — whether the extra work is justified (score correctness
# MAGIC    separately; a trace status of `OK` is not proof the answer is correct).
# MAGIC
# MAGIC Reconcile observed telemetry with billing separately using
# MAGIC `sql/billing_reconciliation.sql` against `system.billing.usage`. Token counts here are
# MAGIC telemetry, **not** billed cost.
