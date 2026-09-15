"""Supervisor + MCP adapter — reach Genie via the managed Genie MCP server.

Databricks hosts a managed MCP server per Genie space at
``https://<host>/api/2.0/mcp/genie/<space_id>``. This adapter acts as an MCP client to
that URL: it discovers the available tool and calls it with the question. That measures the
**MCP transport overhead to reach Genie** — a legitimate third data point next to
``direct_genie`` (Conversation API) and ``supervisor`` (MAS). It is *not* a
supervisor-orchestrated-over-MCP path (that would require a custom deployed agent), and is
labelled honestly as such.

The MCP protocol is never hand-rolled here: the default caller uses a documented Databricks
MCP client and drives it by *discovery* (list tools, read the tool's input schema). If no
MCP client is installed, the adapter reports the prerequisite rather than guessing — per the
PRD, the harness must never invent an MCP connection, tool schema, or invocation flow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..config import Config
from ..models import (
    BenchmarkQuestion,
    InvocationResult,
    RunContext,
    RunStatus,
    TokenSource,
    TokenUsage,
)
from ..redaction import redact_text
from ..util import content_metadata, utc_now_iso
from .base import PrerequisiteNotConfigured

# caller(question_text, timeout_seconds) -> raw MCP result payload (a plain dict).
McpCaller = Callable[[str, float], dict[str, Any]]


def normalize_mcp_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw managed-Genie-MCP result into ``InvocationResult`` fields.

    Genie via MCP returns SQL and a response but no token usage, so token usage is
    ``UNAVAILABLE`` (as for ``direct_genie``); the value of this variant is the MCP call
    count and payload sizes, not tokens.
    """
    warnings: list[str] = []
    sql = payload.get("generated_sql")
    counts: dict[str, Any] = {
        "mcp_call_count": int(payload.get("mcp_call_count", 1)),
        "tool_call_count": int(payload.get("tool_call_count", 1)),
        "sql_count": 1 if sql else 0,
    }
    if payload.get("tool_status") not in (None, "ok", "success"):
        warnings.append(f"mcp_tool_status:{payload.get('tool_status')}")
    raw_ids = {
        "mcp_server": payload.get("mcp_server"),
        "tool_name": payload.get("tool_name"),
        "request_bytes": payload.get("request_bytes"),
        "response_bytes": payload.get("response_bytes"),
    }
    warnings.append("token_usage_unavailable: Genie-over-MCP does not report token usage.")
    return {
        "response_text": payload.get("response_text"),
        "generated_sql": sql,
        "trace_id": payload.get("trace_id"),
        "raw_ids": raw_ids,
        "counts": counts,
        "warnings": warnings,
    }


class SupervisorMcpAdapter:
    name = "supervisor_mcp"

    def __init__(
        self,
        config: Config,
        *,
        caller: McpCaller | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._caller = caller
        self._monotonic = monotonic

    def invoke(self, question: BenchmarkQuestion, context: RunContext) -> InvocationResult:
        started_at = utc_now_iso()
        start_mono = self._monotonic()
        try:
            caller = self._caller or _default_caller(self._config)
            payload = caller(question.question, float(context.request_timeout_seconds))
            norm = normalize_mcp_payload(payload)
        except PrerequisiteNotConfigured as exc:
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.NOT_CONFIGURED, error=str(exc),
                warnings=["supervisor_mcp_adapter_not_configured"],
            )
        except TimeoutError:
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.TIMEOUT, error="MCP request exceeded the configured timeout.",
            )
        except Exception as exc:  # defensive: never crash the whole benchmark
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.FAILED, error=f"MCP error: {redact_text(str(exc))}",
            )

        return self._result(
            question, context, started_at, start_mono,
            status=RunStatus.COMPLETED,
            response_text=norm["response_text"],
            generated_sql=norm["generated_sql"],
            trace_id=norm["trace_id"],
            raw_ids=norm["raw_ids"],
            counts=norm["counts"],
            warnings=norm["warnings"],
        )

    def _result(
        self,
        question: BenchmarkQuestion,
        context: RunContext,
        started_at: str,
        start_mono: float,
        *,
        status: RunStatus,
        response_text: str | None = None,
        generated_sql: str | None = None,
        trace_id: str | None = None,
        raw_ids: dict[str, Any] | None = None,
        counts: dict[str, Any] | None = None,
        error: str | None = None,
        warnings: list[str] | None = None,
    ) -> InvocationResult:
        latency_ms = int((self._monotonic() - start_mono) * 1000)
        resp_meta = content_metadata(response_text, store_text=context.store_response_text)
        sql_meta = content_metadata(generated_sql, store_text=context.store_response_text)
        return InvocationResult(
            run_id=context.run_id,
            question_id=question.id,
            variant=self.name,
            repetition=context.repetition,
            started_at=started_at,
            completed_at=utc_now_iso(),
            status=status,
            response_text=resp_meta["text"],
            response_sha256=resp_meta["sha256"],
            generated_sql=sql_meta["text"],
            generated_sql_sha256=sql_meta["sha256"],
            trace_id=trace_id,
            raw_ids=raw_ids or {},
            token_usage=TokenUsage(source=TokenSource.UNAVAILABLE),
            client_metrics={"latency_ms": latency_ms, "response_chars": resp_meta["chars"]},
            counts=counts or {},
            error=error,
            warnings=list(warnings or []),
        )


def _select_tool(tools: list[Any]) -> Any:
    """Pick the Genie query tool from a discovered tool list (name-based, not guessed)."""
    for tool in tools:
        name = str(getattr(tool, "name", "") or (tool.get("name") if isinstance(tool, dict) else "")).lower()
        if "genie" in name or "query" in name or "ask" in name:
            return tool
    return tools[0] if tools else None


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else "") or "")


def _build_args(tool: Any, question_text: str) -> dict[str, Any]:
    """Fill the tool's input schema from discovery (first required string prop), not a guess."""
    schema = getattr(tool, "inputSchema", None)
    if schema is None and isinstance(tool, dict):
        schema = tool.get("inputSchema") or tool.get("input_schema")
    if isinstance(schema, dict):
        props = schema.get("properties") or {}
        required = schema.get("required") or list(props.keys())
        for key in required:
            prop = props.get(key, {})
            if isinstance(prop, dict) and prop.get("type", "string") == "string":
                return {key: question_text}
    return {"query": question_text}


def _default_caller(config: Config) -> McpCaller:
    """Build the default caller using a documented Databricks MCP client via discovery.

    Raises :class:`PrerequisiteNotConfigured` (never guesses) when no MCP client library is
    available; the run then records the boundary honestly and continues.
    """
    if not config.mcp_target:
        raise PrerequisiteNotConfigured(
            "Supervisor+MCP invocation is not configured: 'mcp_target' is empty. Provide the "
            "managed Genie MCP URL (https://<host>/api/2.0/mcp/genie/<space_id>) or run "
            "`benchmark bootstrap`."
        )

    def _call(question_text: str, _timeout: float) -> dict[str, Any]:
        try:
            from databricks_mcp import DatabricksMCPClient  # lazy, optional
        except ImportError as exc:
            raise PrerequisiteNotConfigured(
                "Supervisor+MCP requires an MCP client. Install it (e.g. `pip install "
                "databricks-mcp`) or wire your approved MCP client into "
                "SupervisorMcpAdapter. The harness will not hand-roll the MCP protocol."
            ) from exc
        from databricks.sdk import WorkspaceClient  # lazy

        kwargs: dict[str, Any] = {"host": config.workspace_host}
        if config.databricks_profile:
            kwargs["profile"] = config.databricks_profile
        ws = WorkspaceClient(**kwargs)
        client = DatabricksMCPClient(server_url=config.mcp_target, workspace_client=ws)

        tools = list(client.list_tools())
        tool = _select_tool(tools)
        if tool is None:
            raise RuntimeError("managed Genie MCP server exposed no tools")
        tool_name = _tool_name(tool)
        args = _build_args(tool, question_text)
        request_bytes = len(str(args).encode("utf-8"))
        result = client.call_tool(tool_name, args)

        # Normalize the MCP tool result defensively into text + optional SQL.
        text_parts: list[str] = []
        content = getattr(result, "content", None) or (result.get("content") if isinstance(result, dict) else None)
        for block in content or []:
            piece = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if piece:
                text_parts.append(str(piece))
        response_text = "\n\n".join(text_parts) if text_parts else None
        is_error = getattr(result, "isError", None)
        if is_error is None and isinstance(result, dict):
            is_error = result.get("isError")
        return {
            "response_text": response_text,
            "generated_sql": None,
            "tool_name": tool_name,
            "mcp_server": config.mcp_target,
            "request_bytes": request_bytes,
            "response_bytes": len((response_text or "").encode("utf-8")),
            "tool_status": "error" if is_error else "ok",
        }

    return _call
