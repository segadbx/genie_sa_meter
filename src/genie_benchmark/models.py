"""Typed contracts shared across the harness.

These are plain :mod:`dataclasses` (no third-party dependency) so the contracts are
easy to construct in tests and serialize deterministically. Every adapter accepts one
:class:`BenchmarkQuestion` and returns one :class:`InvocationResult`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Terminal status of a single invocation."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    # Reserved for adapters that are configured only as customer boundaries.
    NOT_CONFIGURED = "NOT_CONFIGURED"


class TokenSource(str, Enum):
    """Provenance of a normalized token count.

    Used so the report never labels an inferred value as billed cost.
    """

    PROVIDER_REPORTED = "provider_reported"
    TRACE_ATTRIBUTE = "trace_attribute"
    CHILD_SPAN_SUM = "child_span_sum"
    UNAVAILABLE = "unavailable"


# Recognized benchmark variants. Kept as a frozenset so config validation can reject
# unknown variants with an actionable message.
KNOWN_VARIANTS = frozenset({"direct_genie", "supervisor", "supervisor_mcp"})


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One normalized benchmark question.

    ``id``, ``question``, ``category`` and ``expected_behavior`` are required by input
    validation. ``follow_up_to`` marks a question that intentionally reuses conversation
    state from a prior question id (used only for the follow-up test).
    """

    id: str
    question: str
    category: str
    expected_behavior: str
    expected_route: str | None = None
    ground_truth_notes: str | None = None
    follow_up_to: str | None = None

    @property
    def is_follow_up(self) -> bool:
        return self.follow_up_to is not None


@dataclass
class TokenUsage:
    """Normalized token usage with explicit provenance.

    ``None`` values mean the count was not available from the recorded source. The raw
    provider fields are preserved separately on the :class:`InvocationResult`.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: TokenSource = TokenSource.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source.value,
        }


@dataclass
class RunContext:
    """Per-run context passed to every adapter invocation."""

    run_id: str
    environment: str
    repetition: int = 1
    # When set, the adapter should reuse this conversation id instead of starting a new
    # conversation (used for explicit follow-up questions only).
    reuse_conversation_id: str | None = None
    # When False, adapters/persistence store only hashes and lengths, never raw content.
    store_response_text: bool = False
    request_timeout_seconds: int = 300
    poll_interval_seconds: int = 2
    redaction_policy: str = "default"


@dataclass
class InvocationResult:
    """Normalized record returned by every adapter (PRD "Adapter interface").

    The raw provider identifiers are preserved in ``raw_ids`` and ``raw_token_usage``;
    the normalized view is in ``token_usage``. ``response_text`` / ``generated_sql`` are
    populated only when content storage is enabled; otherwise the ``*_sha256`` and
    ``*_chars`` fields carry length/hash metadata only.
    """

    run_id: str
    question_id: str
    variant: str
    repetition: int
    started_at: str  # UTC ISO-8601
    completed_at: str  # UTC ISO-8601
    status: RunStatus

    response_text: str | None = None
    response_sha256: str | None = None
    generated_sql: str | None = None
    generated_sql_sha256: str | None = None

    conversation_id: str | None = None
    trace_id: str | None = None
    raw_ids: dict[str, Any] = field(default_factory=dict)

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    raw_token_usage: dict[str, Any] = field(default_factory=dict)

    client_metrics: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)  # llm/tool/mcp/sql call counts
    error: str | None = None  # redacted, structured error string
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        data["token_usage"] = self.token_usage.to_dict()
        return data
