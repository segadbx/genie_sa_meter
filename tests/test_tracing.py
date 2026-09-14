"""Trace normalization tests: trace-level usage, child-span sums, missing usage, conflicts."""

from __future__ import annotations

from tests.conftest import load_fixture

from genie_benchmark.models import TokenSource
from genie_benchmark.tracing import (
    count_calls,
    normalize_token_usage,
    sum_child_span_tokens,
)


def test_trace_level_preferred_when_present() -> None:
    trace = load_fixture("trace_response.json")
    usage, raw, warnings = normalize_token_usage(trace)
    assert usage.total_tokens == 1500
    assert usage.source == TokenSource.TRACE_ATTRIBUTE
    # Both views preserved.
    assert raw["trace_level"]["total_tokens"] == 1500
    assert raw["child_span_sum"]["total_tokens"] == 1500


def test_child_span_sum() -> None:
    trace = load_fixture("trace_response.json")
    summed = sum_child_span_tokens(trace)
    assert summed["total_tokens"] == 1500  # 950 + 550
    assert summed["input_tokens"] == 1200
    assert summed["spans_with_usage"] == 2


def test_child_span_used_when_no_trace_level() -> None:
    trace = {
        "data": {
            "spans": [
                {
                    "span_type": "LLM",
                    "attributes": {
                        "mlflow.chat.tokenUsage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    },
                },
                {"span_type": "LLM", "attributes": {"token_usage": {"input_tokens": 20, "output_tokens": 10}}},
            ]
        }
    }
    usage, _, _ = normalize_token_usage(trace)
    assert usage.source == TokenSource.CHILD_SPAN_SUM
    assert usage.total_tokens == 45  # 15 + (20+10)


def test_conflicting_values_warn() -> None:
    trace = {
        "info": {"token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 200}},
        "data": {"spans": [{"span_type": "LLM", "attributes": {"mlflow.chat.tokenUsage": {"total_tokens": 150}}}]},
    }
    usage, raw, warnings = normalize_token_usage(trace)
    assert usage.total_tokens == 200  # trace-level preferred
    assert any("conflict" in w for w in warnings)
    assert raw["child_span_sum"]["total_tokens"] == 150


def test_missing_usage_is_unavailable() -> None:
    trace = {"data": {"spans": [{"span_type": "TOOL", "attributes": {}}]}}
    usage, _, warnings = normalize_token_usage(trace)
    assert usage.source == TokenSource.UNAVAILABLE
    assert usage.total_tokens is None
    assert any("unavailable" in w for w in warnings)


def test_partial_child_usage_warns() -> None:
    trace = {
        "data": {
            "spans": [
                {"span_type": "LLM", "attributes": {"mlflow.chat.tokenUsage": {"total_tokens": 30}}},
                {"span_type": "LLM", "attributes": {}},  # no usage reported
            ]
        }
    }
    usage, _, warnings = normalize_token_usage(trace)
    assert usage.source == TokenSource.CHILD_SPAN_SUM
    assert any("partial" in w for w in warnings)


def test_count_calls() -> None:
    trace = load_fixture("trace_response.json")
    counts = count_calls(trace)
    assert counts["llm_call_count"] == 2
    assert counts["tool_call_count"] == 2
    assert counts["mcp_call_count"] == 1  # "mcp_lookup" span name
