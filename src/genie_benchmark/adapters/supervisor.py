"""Supervisor adapter — query an Agent Bricks Multi-Agent Supervisor serving endpoint.

The MAS endpoint speaks the Responses API: an ``input`` array of messages, with
``databricks_options.return_trace`` requesting the trace inline. This adapter sends that
request (via the injectable ``caller`` so tests use fixtures and never make live calls),
then normalizes the response into the harness ``InvocationResult`` contract.

Databricks does not publicly pin every response field name for a MAS endpoint, so the
normalizer reads defensively: it prefers a provider-reported ``usage`` block, and falls
back to summing child-span tokens from the returned trace (reusing
:mod:`genie_benchmark.tracing`). It never labels an inferred value as billed cost. If
``supervisor_target`` is missing it reports the prerequisite rather than guessing an
endpoint — the harness must never invent an API shape.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

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
from ..tracing import _usage_from_mapping, count_calls, normalize_token_usage
from ..util import content_metadata, utc_now_iso
from .base import PrerequisiteNotConfigured

# caller(question_text, timeout_seconds) -> raw response payload (a plain dict).
SupervisorCaller = Callable[[str, float], dict[str, Any]]


def _extract_text(payload: dict[str, Any]) -> str | None:
    """Pull assistant text out of a Responses-API (or chat) payload, defensively."""
    if isinstance(payload.get("output_text"), str) and payload["output_text"]:
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
    if parts:
        return "\n\n".join(parts)
    # Fallback: chat-completions shape.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _extract_trace(payload: dict[str, Any]) -> dict[str, Any] | None:
    db_out = payload.get("databricks_output")
    trace = None
    if isinstance(db_out, dict):
        trace = db_out.get("trace")
    if trace is None:
        trace = payload.get("trace")
    return trace if isinstance(trace, dict) else None


def _trace_id(payload: dict[str, Any], trace: dict[str, Any] | None) -> str | None:
    for candidate in (
        payload.get("trace_id"),
        (trace or {}).get("trace_id"),
        ((trace or {}).get("info") or {}).get("trace_id"),
        ((trace or {}).get("info") or {}).get("request_id"),
        (trace or {}).get("request_id"),
    ):
        if candidate:
            return str(candidate)
    return None


def normalize_supervisor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw MAS response into the fields an ``InvocationResult`` needs."""
    trace = _extract_trace(payload)
    warnings: list[str] = []

    # Tokens: prefer provider-reported usage; else derive from the trace's child spans.
    usage_map = _usage_from_mapping(payload.get("usage"))
    raw_token_usage: dict[str, Any] = {"provider_usage": payload.get("usage")}
    if usage_map is not None:
        token_usage = TokenUsage(
            input_tokens=usage_map["input_tokens"],
            output_tokens=usage_map["output_tokens"],
            total_tokens=usage_map["total_tokens"],
            source=TokenSource.PROVIDER_REPORTED,
        )
    elif trace is not None:
        token_usage, trace_raw, trace_warnings = normalize_token_usage(trace)
        raw_token_usage.update(trace_raw)
        warnings.extend(trace_warnings)
    else:
        token_usage = TokenUsage(source=TokenSource.UNAVAILABLE)
        warnings.append("token_usage_unavailable: supervisor response carried no usage or trace.")

    counts = count_calls(trace) if trace is not None else {}
    if trace is None:
        warnings.append("trace_unavailable: supervisor response did not return a trace.")

    return {
        "response_text": _extract_text(payload),
        "trace_id": _trace_id(payload, trace),
        "token_usage": token_usage,
        "raw_token_usage": raw_token_usage,
        "counts": counts,
        "warnings": warnings,
    }


class SupervisorAdapter:
    name = "supervisor"

    def __init__(
        self,
        config: Config,
        *,
        caller: SupervisorCaller | None = None,
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
            norm = normalize_supervisor_payload(payload)
        except PrerequisiteNotConfigured as exc:
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.NOT_CONFIGURED, error=str(exc),
                warnings=["supervisor_adapter_not_configured"],
            )
        except TimeoutError:
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.TIMEOUT, error="Supervisor request exceeded the configured timeout.",
            )
        except Exception as exc:  # defensive: never crash the whole benchmark
            return self._result(
                question, context, started_at, start_mono,
                status=RunStatus.FAILED, error=f"Supervisor error: {redact_text(str(exc))}",
            )

        return self._result(
            question, context, started_at, start_mono,
            status=RunStatus.COMPLETED,
            response_text=norm["response_text"],
            trace_id=norm["trace_id"],
            token_usage=norm["token_usage"],
            raw_token_usage=norm["raw_token_usage"],
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
        trace_id: str | None = None,
        token_usage: TokenUsage | None = None,
        raw_token_usage: dict[str, Any] | None = None,
        counts: dict[str, Any] | None = None,
        error: str | None = None,
        warnings: list[str] | None = None,
    ) -> InvocationResult:
        latency_ms = int((self._monotonic() - start_mono) * 1000)
        resp_meta = content_metadata(response_text, store_text=context.store_response_text)
        warns = list(warnings or [])
        if status is RunStatus.COMPLETED and trace_id is None:
            warns.append("trace_id_unavailable: supervisor did not return a trace id.")
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
            trace_id=trace_id,
            token_usage=token_usage or TokenUsage(source=TokenSource.UNAVAILABLE),
            raw_token_usage=raw_token_usage or {},
            client_metrics={"latency_ms": latency_ms, "response_chars": resp_meta["chars"]},
            counts=counts or {},
            error=error,
            warnings=warns,
        )


def _endpoint_path(target: str) -> str:
    """Derive the ``/serving-endpoints/<name>/invocations`` request path from the target.

    Accepts a full invocations URL or a bare endpoint name.
    """
    if target.startswith("http"):
        path = urlsplit(target).path
        return path if path else target
    if target.startswith("/"):
        return target
    return f"/serving-endpoints/{target}/invocations"


def _default_caller(config: Config) -> SupervisorCaller:
    """Build the default caller that posts to the MAS serving endpoint via the SDK.

    Uses ``WorkspaceClient.api_client.do`` so no extra dependency is needed. Auth resolves
    from the environment or the named profile — never from the config file.
    """
    if not config.supervisor_target:
        raise PrerequisiteNotConfigured(
            "Supervisor invocation is not configured: 'supervisor_target' is empty. Provide "
            "the MAS serving-endpoint URL or run `benchmark bootstrap` to create one."
        )
    from databricks.sdk import WorkspaceClient  # lazy

    kwargs: dict[str, Any] = {"host": config.workspace_host}
    if config.databricks_profile:
        kwargs["profile"] = config.databricks_profile
    ws = WorkspaceClient(**kwargs)
    path = _endpoint_path(config.supervisor_target)

    def _call(question_text: str, _timeout: float) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": [{"type": "message", "role": "user", "content": question_text}],
            "databricks_options": {"return_trace": True},
        }
        if config.trace_destination is not None:
            body["trace_destination"] = {
                "catalog_name": config.trace_destination.catalog_name,
                "schema_name": config.trace_destination.schema_name,
                "table_prefix": config.trace_destination.table_prefix,
            }
        result = ws.api_client.do("POST", path, body=body)
        return result if isinstance(result, dict) else {"output_text": str(result)}

    return _call
