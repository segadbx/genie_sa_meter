"""Side-effecting provisioner for the demo bootstrap.

Creates (create-or-reuse, idempotent) the Databricks resources a live three-variant
benchmark needs, then emits ``config.yaml`` + ``questions.json``. Everything Databricks is
imported lazily so importing this module never requires a workspace or the SDK.

Auth resolves from the ``--profile`` at call time; no credentials are read from or written
to any file. Where a documented public API is uncertain in the target workspace (Genie
space CRUD on an older SDK, the ``supervisor-agents`` CLI), the provisioner probes and
degrades gracefully — it prints an explicit prerequisite rather than guessing an API shape.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any

from . import genie_spaces, oil_gas
from .emit import build_config_dict, write_config, write_questions
from .settings import BootstrapSettings

Log = Callable[[str], None]


class BootstrapError(RuntimeError):
    """Raised when provisioning cannot proceed and the user must act."""


def _https(host: str) -> str:
    return host if host.startswith("http") else f"https://{host}"


class Provisioner:
    def __init__(self, settings: BootstrapSettings, profile: str | None, *, log: Log) -> None:
        from databricks.sdk import WorkspaceClient  # lazy

        self.settings = settings
        self.profile = profile
        self.log = log
        self.w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        raw_host = settings.workspace_host or self.w.config.host or ""
        if not raw_host:
            raise BootstrapError(
                "Could not resolve the workspace host from the profile. Set workspace_host in "
                "the bootstrap settings file."
            )
        self.host = _https(raw_host)

    # --- SQL over the Statement Execution API (no cluster) --------------------------------

    def run_sql(self, warehouse_id: str, statement: str) -> Any:
        """Execute one SQL statement and block until a terminal state; return the response."""
        from databricks.sdk.service.sql import (  # lazy
            ExecuteStatementRequestOnWaitTimeout,
            StatementState,
        )

        resp = self.w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=statement,
            wait_timeout="30s",
            # DDL / large inserts exceed the 50s sync cap → continue asynchronously and poll.
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
        )
        terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED}
        while resp.status is not None and resp.status.state not in terminal:
            time.sleep(1.0)
            resp = self.w.statement_execution.get_statement(resp.statement_id)
        state = resp.status.state if resp.status else None
        if state != StatementState.SUCCEEDED:
            detail = ""
            if resp.status and resp.status.error:
                detail = f": {resp.status.error.message}"
            raise BootstrapError(f"SQL failed ({state}){detail}\n  statement: {statement[:200]}")
        return resp

    def scalar(self, warehouse_id: str, statement: str) -> Any:
        resp = self.run_sql(warehouse_id, statement)
        result = getattr(resp, "result", None)
        rows = getattr(result, "data_array", None) if result else None
        if rows:
            return rows[0][0]
        return None

    # --- Warehouse ------------------------------------------------------------------------

    def resolve_warehouse(self) -> str:
        wh = self.settings.warehouse
        if wh.reuse_id:
            self.log(f"warehouse: reusing {wh.reuse_id}")
            return wh.reuse_id
        # Prefer an existing serverless warehouse.
        for ep in self.w.warehouses.list():
            if getattr(ep, "enable_serverless_compute", False):
                self.log(f"warehouse: discovered serverless '{ep.name}' ({ep.id})")
                return ep.id  # type: ignore[return-value]
        if not wh.create_if_missing:
            raise BootstrapError(
                "No serverless SQL warehouse found and warehouse.create_if_missing is false. "
                "Set warehouse.reuse_id or allow creation."
            )
        from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType  # lazy

        self.log(f"warehouse: creating serverless '{wh.name}' ({wh.size}) ...")
        created = self.w.warehouses.create(
            name=wh.name,
            cluster_size=wh.size,
            max_num_clusters=1,
            auto_stop_mins=10,
            enable_serverless_compute=True,
            warehouse_type=CreateWarehouseRequestWarehouseType.PRO,
        ).result()
        self.log(f"warehouse: created {created.id}")
        return created.id  # type: ignore[return-value]

    # --- Unity Catalog: schema + tables + data --------------------------------------------

    def ensure_catalog_schema(self, warehouse_id: str) -> None:
        s = self.settings
        if s.catalog.create:
            self.run_sql(warehouse_id, f"CREATE CATALOG IF NOT EXISTS {s.catalog.name}")
            self.log(f"catalog: ensured {s.catalog.name}")
        self.run_sql(warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {s.fq_schema}")
        self.log(f"schema: ensured {s.fq_schema}")

    def ensure_tables_and_data(self, warehouse_id: str) -> None:
        fq = self.settings.fq_schema
        ddl = oil_gas.table_ddl(fq)
        inserts = oil_gas.data_inserts(fq, self.settings.data)
        for table in oil_gas.ORDERED_TABLES:
            self.run_sql(warehouse_id, ddl[table])
            count = self.scalar(warehouse_id, f"SELECT COUNT(*) FROM {fq}.{table}")
            if count and int(count) > 0:
                self.log(f"table {table}: already has {count} rows, skipping load")
                continue
            self.log(f"table {table}: loading synthetic data ...")
            self.run_sql(warehouse_id, inserts[table])
            loaded = self.scalar(warehouse_id, f"SELECT COUNT(*) FROM {fq}.{table}")
            self.log(f"table {table}: {loaded} rows")

    # --- MLflow experiment ----------------------------------------------------------------

    def ensure_experiment(self) -> str | None:
        s = self.settings
        path = s.experiment.path or self._default_experiment_path()
        if not path:
            return None
        try:
            import mlflow  # lazy

            mlflow.set_tracking_uri(f"databricks://{self.profile}" if self.profile else "databricks")
            existing = mlflow.get_experiment_by_name(path)
            if existing is not None:
                self.log(f"experiment: reusing {path} ({existing.experiment_id})")
                return existing.experiment_id
            exp_id = mlflow.create_experiment(path)
            self.log(f"experiment: created {path} ({exp_id})")
            return exp_id
        except Exception as exc:  # optional resource — never fail the whole bootstrap
            self.log(f"experiment: skipped (could not create '{path}': {exc})")
            return None

    def _default_experiment_path(self) -> str | None:
        try:
            user = self.w.current_user.me().user_name
        except Exception:
            return None
        return f"/Users/{user}/genie-bench-demo" if user else None

    # --- Genie spaces ---------------------------------------------------------------------

    def _genie_parent_path(self) -> str:
        if self.settings.genie.parent_path:
            return self.settings.genie.parent_path
        user = self.w.current_user.me().user_name
        return f"/Workspace/Users/{user}/genie_spaces"

    def create_genie_space(
        self, *, warehouse_id: str, title: str, description: str, serialized_space: str, parent_path: str
    ) -> str:
        payload = {
            "warehouse_id": warehouse_id,
            "title": title,
            "description": description,
            "parent_path": parent_path,
            "serialized_space": serialized_space,
        }
        # Primary path: the documented CLI. Falls back to REST if the subcommand is absent.
        try:
            out = self._cli_json(["genie", "create-space", "--json", json.dumps(payload)])
        except FileNotFoundError as exc:  # `databricks` not on PATH
            raise BootstrapError("The `databricks` CLI is required for Genie space creation.") from exc
        except BootstrapError:
            self.log("genie: CLI create-space unavailable, trying REST /api/2.0/genie/spaces ...")
            out = self.w.api_client.do("POST", "/api/2.0/genie/spaces", body=payload)  # type: ignore[assignment]
        space_id = out.get("space_id") or out.get("id") or (out.get("space") or {}).get("id")
        if not space_id:
            raise BootstrapError(f"Genie create-space returned no space id: {out}")
        self.log(f"genie space '{title}': {space_id}")
        return str(space_id)

    def ensure_genie_spaces(self, warehouse_id: str) -> tuple[str, str]:
        fq = self.settings.fq_schema
        parent = self._genie_parent_path()
        try:
            self.w.workspace.mkdirs(parent)
        except Exception:  # already exists / permission handled by the create call
            pass
        prod = self.create_genie_space(
            warehouse_id=warehouse_id,
            title=self.settings.genie.production_space_name,
            description=genie_spaces.PRODUCTION_DESCRIPTION,
            serialized_space=genie_spaces.production_serialized_space(fq),
            parent_path=parent,
        )
        maint = self.create_genie_space(
            warehouse_id=warehouse_id,
            title=self.settings.genie.maintenance_space_name,
            description=genie_spaces.MAINTENANCE_DESCRIPTION,
            serialized_space=genie_spaces.maintenance_serialized_space(fq),
            parent_path=parent,
        )
        return prod, maint

    # --- Supervisor (Agent Bricks Multi-Agent Supervisor) ---------------------------------

    def build_supervisor(self, production_space_id: str, maintenance_space_id: str) -> str | None:
        """Create a MAS over the two Genie spaces; return its serving-endpoint URL, or None.

        Returns None (with an explanatory log) when the `supervisor-agents` CLI is not
        available — the caller then drops the `supervisor` variant rather than guessing an
        API shape. The `supervisor_mcp` variant does not depend on this.
        """
        if not self._supervisor_cli_available():
            self.log(
                "supervisor: `databricks supervisor-agents` not available on this CLI. "
                "Skipping MAS. Create it in the Agent Bricks UI over the two Genie spaces and "
                "set `supervisor_target` in config.yaml to enable the 'supervisor' variant."
            )
            return None
        name = self.settings.supervisor.name
        instructions = (
            "Route production, oil/gas/water volume, rate, decline, and revenue questions to the "
            "production agent. Route equipment, maintenance, work-order, failure, and downtime "
            "questions to the maintenance agent."
        )
        out = self._cli_json(
            [
                "supervisor-agents", "create-supervisor-agent", name,
                "--description", "Upstream O&G benchmark supervisor over production + maintenance Genie agents.",
                "--instructions", instructions,
            ]
        )
        agent_name = out.get("name")
        endpoint_name = out.get("endpoint_name")
        if not agent_name or not endpoint_name:
            self.log(f"supervisor: unexpected create response, skipping ({out}). ")
            return None
        for tool, space_id, desc in (
            ("production", production_space_id, "Wells and daily production volumes, rates, and revenue"),
            ("maintenance", maintenance_space_id, "Equipment and maintenance events / downtime"),
        ):
            self._cli_json(
                [
                    "supervisor-agents", "create-tool", agent_name, tool, "--json",
                    json.dumps({"tool_type": "genie_space", "description": desc, "genie_space": {"id": space_id}}),
                ]
            )
        self._wait_endpoint_ready(endpoint_name)
        return f"{self.host}/serving-endpoints/{endpoint_name}/invocations"

    def _supervisor_cli_available(self) -> bool:
        try:
            proc = subprocess.run(
                ["databricks", "supervisor-agents", "-h"], capture_output=True, text=True, timeout=30
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _wait_endpoint_ready(self, endpoint_name: str, *, timeout_s: int = 900) -> None:
        self.log(f"supervisor: waiting for endpoint '{endpoint_name}' to be ready ...")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                ep = self.w.serving_endpoints.get(endpoint_name)
                state = getattr(ep, "state", None)
                ready = getattr(getattr(state, "ready", None), "value", None) or getattr(state, "ready", None)
                if str(ready).upper().endswith("READY"):
                    self.log(f"supervisor: endpoint '{endpoint_name}' ready")
                    return
            except Exception:
                pass
            time.sleep(15)
        self.log(f"supervisor: endpoint '{endpoint_name}' not confirmed ready within {timeout_s}s (continuing)")

    # --- CLI helper -----------------------------------------------------------------------

    def _cli_json(self, args: list[str]) -> dict[str, Any]:
        cmd = ["databricks", *args]
        if self.profile:
            cmd += ["--profile", self.profile]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BootstrapError(f"CLI failed: {' '.join(args[:2])}\n{proc.stderr.strip()[:400]}")
        text = proc.stdout.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BootstrapError(f"CLI returned non-JSON for {' '.join(args[:2])}: {text[:200]}") from exc
        return data if isinstance(data, dict) else {"_": data}


def run_bootstrap(
    settings: BootstrapSettings,
    profile: str | None,
    *,
    teardown: bool = False,
    log: Log = print,
) -> dict[str, Any]:
    """Provision (or tear down) the demo and emit config + questions.

    Returns a summary dict of the resolved identifiers.
    """
    prov = Provisioner(settings, profile, log=log)
    if teardown:
        return _teardown(prov)

    warehouse_id = prov.resolve_warehouse()
    prov.ensure_catalog_schema(warehouse_id)
    prov.ensure_tables_and_data(warehouse_id)
    experiment_id = prov.ensure_experiment()
    production_space_id, maintenance_space_id = prov.ensure_genie_spaces(warehouse_id)

    mcp_target = f"{prov.host}/api/2.0/mcp/genie/{production_space_id}"
    supervisor_target: str | None = None
    if settings.supervisor.build:
        supervisor_target = prov.build_supervisor(production_space_id, maintenance_space_id)

    # Emit only variants whose targets we actually produced.
    variants = list(settings.emit.variants)
    if "supervisor" in variants and not supervisor_target:
        variants = [v for v in variants if v != "supervisor"]
        log("emit: dropping 'supervisor' variant (no supervisor endpoint was created)")
    settings.emit.variants = variants

    config = build_config_dict(
        settings,
        workspace_host=prov.host,
        profile=profile,
        genie_space_id=production_space_id,
        experiment_id=experiment_id,
        supervisor_target=supervisor_target,
        mcp_target=mcp_target,
    )
    config_path = write_config(settings.emit.config_path, config)
    questions_path = write_questions(settings.emit.questions_path, oil_gas.benchmark_questions())
    log(f"wrote {config_path}")
    log(f"wrote {questions_path}")
    log("")
    log("Bootstrap complete. Next:")
    log(f"  uv run benchmark preflight --config {settings.emit.config_path}")
    log(f"  uv run benchmark run --config {settings.emit.config_path}")

    return {
        "workspace_host": prov.host,
        "warehouse_id": warehouse_id,
        "schema": settings.fq_schema,
        "production_space_id": production_space_id,
        "maintenance_space_id": maintenance_space_id,
        "experiment_id": experiment_id,
        "supervisor_target": supervisor_target,
        "mcp_target": mcp_target,
        "variants": variants,
        "config_path": str(config_path),
        "questions_path": str(questions_path),
    }


def _teardown(prov: Provisioner) -> dict[str, Any]:
    """Best-effort teardown of resources matching the configured names/titles."""
    s = prov.settings
    warehouse_id = s.warehouse.reuse_id or prov.resolve_warehouse()

    # Genie spaces (trash by matching title).
    try:
        listed = prov._cli_json(["genie", "list-spaces"])
        spaces = listed.get("spaces") or listed.get("_") or []
        wanted = {s.genie.production_space_name, s.genie.maintenance_space_name}
        for space in spaces if isinstance(spaces, list) else []:
            if isinstance(space, dict) and space.get("title") in wanted and space.get("space_id"):
                prov._cli_json(["genie", "trash-space", str(space["space_id"])])
                prov.log(f"trashed genie space: {space.get('title')}")
    except Exception as exc:
        prov.log(f"teardown: genie spaces skipped ({exc})")

    # Schema (drop cascade), then catalog if we created it.
    try:
        prov.run_sql(warehouse_id, f"DROP SCHEMA IF EXISTS {s.fq_schema} CASCADE")
        prov.log(f"dropped schema {s.fq_schema}")
        if s.catalog.create:
            prov.run_sql(warehouse_id, f"DROP CATALOG IF EXISTS {s.catalog.name} CASCADE")
            prov.log(f"dropped catalog {s.catalog.name}")
    except Exception as exc:
        prov.log(f"teardown: schema/catalog skipped ({exc})")

    prov.log(
        "teardown: the Multi-Agent Supervisor and any created SQL warehouse are left in place; "
        "delete them from the workspace UI if desired."
    )
    return {"teardown": True, "schema": s.fq_schema}
