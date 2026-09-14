"""MLflow tracing helpers and defensive token normalization.

Two concerns live here:

1. **Span helpers** wrap the harness boundary in MLflow spans so each benchmark request
   has a root span carrying the run attributes. MLflow is imported lazily so the pure
   normalization functions (and the offline test suite) do not require a tracking
   backend.
2. **Token normalization** reads the trace-level ``token_usage`` when present and *also*
   sums child LLM/Chat-model span usage, recording which source produced each value.
   The trace UI can understate totals when child spans do not report usage, so both the
   trace-level value and the child-span sum are preserved. An inferred value is never
   labeled as billed cost.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from .models import TokenSource, TokenUsage

# Span-attribute keys that MLflow / common providers use for per-span token usage.
_SPAN_USAGE_KEYS = (
    "mlflow.chat.tokenUsage",
    "llm.token_usage",
    "token_usage",
    "usage",
)
# Span types considered LLM/Chat model calls for child-span summation.
_LLM_SPAN_TYPES = {"CHAT_MODEL", "LLM", "CHAT", "LLM_MODEL"}

_INPUT_KEYS = ("input_tokens", "prompt_tokens", "input")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens", "output")
_TOTAL_KEYS = ("total_tokens", "total")


def _first_int(d: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except (TypeError, ValueError):
                return None
    return None


def _usage_from_mapping(usage: Any) -> dict[str, int | None] | None:
    if not isinstance(usage, dict):
        return None
    inp = _first_int(usage, _INPUT_KEYS)
    out = _first_int(usage, _OUTPUT_KEYS)
    tot = _first_int(usage, _TOTAL_KEYS)
    if inp is None and out is None and tot is None:
        return None
    if tot is None and (inp is not None or out is not None):
        tot = (inp or 0) + (out or 0)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def extract_trace_level_tokens(trace: dict[str, Any]) -> dict[str, int | None] | None:
    """Return trace-level token usage if the trace reports it, else ``None``."""
    info = trace.get("info", trace)
    for container in (info, trace):
        if isinstance(container, dict):
            usage = container.get("token_usage")
            parsed = _usage_from_mapping(usage)
            if parsed is not None:
                return parsed
    return None


def _iter_spans(trace: dict[str, Any]) -> Iterator[dict[str, Any]]:
    data = trace.get("data", {})
    spans = data.get("spans") if isinstance(data, dict) else None
    if spans is None:
        spans = trace.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict):
                yield span


def _span_usage(span: dict[str, Any]) -> dict[str, int | None] | None:
    attributes = span.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    for key in _SPAN_USAGE_KEYS:
        if key in attributes:
            parsed = _usage_from_mapping(attributes[key])
            if parsed is not None:
                return parsed
    # Some traces attach usage directly on the span.
    return _usage_from_mapping(span.get("token_usage"))


def sum_child_span_tokens(trace: dict[str, Any]) -> dict[str, Any]:
    """Sum token usage across LLM/Chat-model child spans.

    Returns the summed input/output/total plus how many LLM spans reported usage and how
    many LLM spans were seen in total (so missing usage is visible).
    """
    inp_total = 0
    out_total = 0
    tot_total = 0
    spans_with_usage = 0
    llm_span_count = 0
    for span in _iter_spans(trace):
        span_type = str(span.get("span_type") or span.get("type") or "").upper()
        is_llm = span_type in _LLM_SPAN_TYPES
        if is_llm:
            llm_span_count += 1
        usage = _span_usage(span)
        if usage is None:
            continue
        # Count usage only from LLM-type spans to avoid double counting wrapper spans.
        if not is_llm and span_type:
            continue
        spans_with_usage += 1
        inp_total += usage["input_tokens"] or 0
        out_total += usage["output_tokens"] or 0
        tot_total += usage["total_tokens"] or ((usage["input_tokens"] or 0) + (usage["output_tokens"] or 0))
    if spans_with_usage == 0:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "spans_with_usage": 0,
            "llm_span_count": llm_span_count,
        }
    return {
        "input_tokens": inp_total,
        "output_tokens": out_total,
        "total_tokens": tot_total,
        "spans_with_usage": spans_with_usage,
        "llm_span_count": llm_span_count,
    }


def normalize_token_usage(trace: dict[str, Any]) -> tuple[TokenUsage, dict[str, Any], list[str]]:
    """Normalize token usage from a trace.

    Returns ``(chosen_usage, raw, warnings)`` where ``raw`` preserves both the
    trace-level value and the child-span sum, and ``warnings`` flags conflicts or
    missing data. The trace-level value is preferred when present; otherwise the
    child-span sum is used. Neither is ever interpreted as billed cost.
    """
    warnings: list[str] = []
    trace_level = extract_trace_level_tokens(trace)
    child_sum = sum_child_span_tokens(trace)
    raw = {"trace_level": trace_level, "child_span_sum": child_sum}

    child_total = child_sum.get("total_tokens")

    if trace_level is not None:
        usage = TokenUsage(
            input_tokens=trace_level["input_tokens"],
            output_tokens=trace_level["output_tokens"],
            total_tokens=trace_level["total_tokens"],
            source=TokenSource.TRACE_ATTRIBUTE,
        )
        if (
            child_total is not None
            and trace_level["total_tokens"] is not None
            and child_total != trace_level["total_tokens"]
        ):
            warnings.append(
                "token_usage_conflict: trace-level total "
                f"({trace_level['total_tokens']}) differs from child-span sum "
                f"({child_total}); both are preserved in raw_token_usage."
            )
        return usage, raw, warnings

    if child_total is not None:
        usage = TokenUsage(
            input_tokens=child_sum["input_tokens"],
            output_tokens=child_sum["output_tokens"],
            total_tokens=child_total,
            source=TokenSource.CHILD_SPAN_SUM,
        )
        if child_sum.get("llm_span_count", 0) > child_sum.get("spans_with_usage", 0):
            warnings.append(
                "token_usage_partial: some LLM spans did not report usage; "
                "child-span sum may understate the true total."
            )
        return usage, raw, warnings

    warnings.append("token_usage_unavailable: no trace-level or child-span token usage found.")
    return TokenUsage(source=TokenSource.UNAVAILABLE), raw, warnings


def count_calls(trace: dict[str, Any]) -> dict[str, int]:
    """Count LLM, tool, and MCP calls visible in the trace spans."""
    llm = tool = mcp = 0
    for span in _iter_spans(trace):
        span_type = str(span.get("span_type") or span.get("type") or "").upper()
        name = str(span.get("name") or "").lower()
        if span_type in _LLM_SPAN_TYPES:
            llm += 1
        if span_type == "TOOL" or "tool" in name:
            tool += 1
        if "mcp" in name or span_type == "MCP":
            mcp += 1
    return {"llm_call_count": llm, "tool_call_count": tool, "mcp_call_count": mcp}


@contextlib.contextmanager
def benchmark_span(name: str, attributes: dict[str, Any]) -> Iterator[Any]:
    """Open an MLflow span for the harness boundary, degrading gracefully if unavailable.

    MLflow is imported lazily; if it is not installed or no tracking backend is
    configured, the context still runs so a benchmark can proceed (with a warning left to
    the caller). Only non-sensitive attributes should be passed here.
    """
    try:
        import mlflow  # noqa: PLC0415  (lazy import by design)

        with mlflow.start_span(name=name) as span:
            try:
                span.set_attributes(attributes)
            except Exception:  # pragma: no cover - attribute API variance
                pass
            yield span
    except Exception:
        # No MLflow backend available; run without a span rather than failing the request.
        yield None
