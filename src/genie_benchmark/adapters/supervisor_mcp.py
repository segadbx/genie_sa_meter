"""Supervisor + MCP adapter — a customer-supplied invocation boundary.

Uses the customer's actual production-like MCP connection. As with the Supervisor
adapter, the harness must not invent an MCP connection, tool schema, or invocation flow;
until the customer wires their approved MCP path here, this adapter reports the
prerequisite.

When implemented, record the MCP server name, selected tool name, request size, response
size, and tool-call status. Do not log full tool arguments by default — store a redacted
preview and a hash for correlation (see :mod:`genie_benchmark.redaction`).
"""

from __future__ import annotations

from ..config import Config
from ..models import BenchmarkQuestion, InvocationResult, RunContext, RunStatus
from ..util import utc_now_iso
from .base import PrerequisiteNotConfigured


class SupervisorMcpAdapter:
    name = "supervisor_mcp"

    def __init__(self, config: Config) -> None:
        self._config = config

    def invoke(self, question: BenchmarkQuestion, context: RunContext) -> InvocationResult:
        started_at = utc_now_iso()
        try:
            return self._call_supervisor_mcp(question, context)
        except PrerequisiteNotConfigured as exc:
            return InvocationResult(
                run_id=context.run_id,
                question_id=question.id,
                variant=self.name,
                repetition=context.repetition,
                started_at=started_at,
                completed_at=utc_now_iso(),
                status=RunStatus.NOT_CONFIGURED,
                error=str(exc),
                warnings=["supervisor_mcp_adapter_not_configured"],
            )

    def _call_supervisor_mcp(
        self, question: BenchmarkQuestion, context: RunContext
    ) -> InvocationResult:
        raise PrerequisiteNotConfigured(
            "Supervisor+MCP invocation is not configured. Provide the customer-approved "
            f"MCP connection for target '{self._config.mcp_target}' and implement "
            "SupervisorMcpAdapter._call_supervisor_mcp. The harness will not guess the MCP "
            "connection, tool schema, or invocation flow."
        )
