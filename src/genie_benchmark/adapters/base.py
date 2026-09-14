"""Adapter interface and shared adapter errors.

Every invocation path implements :class:`AgentAdapter`: it accepts one normalized
``BenchmarkQuestion`` plus a ``RunContext`` and returns one normalized
``InvocationResult``. Customer-specific calls are isolated behind adapters so the
harness never hard-codes an endpoint, schema, or authentication flow.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import BenchmarkQuestion, InvocationResult, RunContext


class AdapterError(Exception):
    """Base class for adapter errors."""


class PrerequisiteNotConfigured(AdapterError):
    """Raised when an adapter cannot run because a customer-supplied prerequisite is missing.

    The message must name exactly what the customer needs to provide (for example, the
    Supervisor invocation URL/resource or the MCP connection). Adapters must never guess
    an API shape to work around a missing prerequisite.
    """


class NonRetryableError(AdapterError):
    """An error that must not be retried (e.g. authorization failure, bad request)."""


class TransientError(AdapterError):
    """A documented transient error safe to retry with bounded backoff (e.g. poll timeout)."""


@runtime_checkable
class AgentAdapter(Protocol):
    """Contract implemented by every invocation path."""

    name: str

    def invoke(
        self,
        question: BenchmarkQuestion,
        context: RunContext,
    ) -> InvocationResult:
        """Invoke the agent for one question and return a normalized result.

        Implementations must: capture UTC start/end timestamps, enforce the configured
        timeout, preserve request/response identifiers where available, return a
        structured error instead of terminating the benchmark, never retry
        non-idempotent operations automatically, and populate ``warnings`` when fields
        are unavailable.
        """
        ...
