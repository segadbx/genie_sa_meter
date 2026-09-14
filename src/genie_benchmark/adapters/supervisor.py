"""Supervisor adapter — a customer-supplied invocation boundary.

The PRD is explicit: the Supervisor adapter must be a thin wrapper around the customer's
*already-approved* invocation method. It must not silently substitute a different
endpoint or model, and the harness must not invent an API shape. Until the customer wires
their approved invocation here, this adapter reports the prerequisite rather than
guessing.

To implement it, replace the body of :meth:`SupervisorAdapter._call_supervisor` with the
approved call and normalize the response into an ``InvocationResult`` (record the
invocation URL/resource name, response id, and any returned trace id as metadata).
"""

from __future__ import annotations

from ..config import Config
from ..models import BenchmarkQuestion, InvocationResult, RunContext, RunStatus
from ..util import utc_now_iso
from .base import PrerequisiteNotConfigured


class SupervisorAdapter:
    name = "supervisor"

    def __init__(self, config: Config) -> None:
        self._config = config

    def invoke(self, question: BenchmarkQuestion, context: RunContext) -> InvocationResult:
        started_at = utc_now_iso()
        try:
            return self._call_supervisor(question, context)
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
                warnings=["supervisor_adapter_not_configured"],
            )

    def _call_supervisor(self, question: BenchmarkQuestion, context: RunContext) -> InvocationResult:
        raise PrerequisiteNotConfigured(
            "Supervisor invocation is not configured. Provide the customer-approved "
            f"Supervisor invocation method for target '{self._config.supervisor_target}' "
            "and implement SupervisorAdapter._call_supervisor. The harness will not guess "
            "the Supervisor endpoint, request schema, or authentication flow."
        )
