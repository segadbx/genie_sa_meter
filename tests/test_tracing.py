"""Trace normalization tests: trace-level usage, child-span sums, missing usage, conflicts.

Also covers the runtime tracing setup (``configure_mlflow_tracing``) and the graceful
degradation reporting in ``benchmark_span``.
"""

from __future__ import annotations

import sys
import types

import pytest
from tests.conftest import load_fixture, valid_config_dict

from genie_benchmark.config import parse_config
from genie_benchmark.models import TokenSource
from genie_benchmark.tracing import (
    benchmark_span,
    configure_mlflow_tracing,
    count_calls,
    normalize_token_usage,
    sum_child_span_tokens,
)


class _FakeMlflow(types.ModuleType):
    """A stand-in for the ``mlflow`` module recording tracking-uri / experiment calls."""

    def __init__(self, tracking_uri: str = "file:///tmp/mlruns") -> None:
        super().__init__("mlflow")
        self._tracking_uri = tracking_uri
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []

    def get_tracking_uri(self) -> str:
        return self._tracking_uri

    def set_tracking_uri(self, uri: str) -> None:
        self.set_tracking_uri_calls.append(uri)
        self._tracking_uri = uri

    def set_experiment(self, experiment_name=None, experiment_id=None):  # noqa: ANN001
        self.set_experiment_calls.append(experiment_id or experiment_name)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    def _install(uri: str = "file:///tmp/mlruns") -> _FakeMlflow:
        fake = _FakeMlflow(uri)
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        return fake

    return _install


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


# --- configure_mlflow_tracing -----------------------------------------------------------


def test_configure_points_local_uri_at_workspace_and_experiment(fake_mlflow) -> None:
    fake = fake_mlflow("file:///tmp/mlruns")  # off-platform default
    config = parse_config(
        valid_config_dict(databricks_profile="FEVM-SANDBOX-AZURE", experiment_id="719529908095423")
    )
    warnings = configure_mlflow_tracing(config)
    assert warnings == []
    assert fake.set_tracking_uri_calls == ["databricks://FEVM-SANDBOX-AZURE"]
    assert fake.set_experiment_calls == ["719529908095423"]


def test_configure_does_not_override_databricks_uri(fake_mlflow) -> None:
    # On a Databricks notebook the tracking URI is already 'databricks'; a laptop-only
    # profile must not clobber it.
    fake = fake_mlflow("databricks")
    config = parse_config(
        valid_config_dict(databricks_profile="FEVM-SANDBOX-AZURE", experiment_id="42")
    )
    warnings = configure_mlflow_tracing(config)
    assert warnings == []
    assert fake.set_tracking_uri_calls == []  # left untouched
    assert fake.set_experiment_calls == ["42"]


def test_configure_warns_when_experiment_unset(fake_mlflow) -> None:
    fake_mlflow("file:///tmp/mlruns")
    config = parse_config(valid_config_dict(databricks_profile="FEVM-SANDBOX-AZURE"))
    warnings = configure_mlflow_tracing(config)
    assert any("experiment_unset" in w for w in warnings)


def test_configure_warns_when_mlflow_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "mlflow":
            raise ImportError("no mlflow here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    config = parse_config(valid_config_dict(experiment_id="1"))
    warnings = configure_mlflow_tracing(config)
    assert len(warnings) == 1
    assert "tracing_disabled" in warnings[0]


# --- benchmark_span degradation reporting ----------------------------------------------


def test_benchmark_span_reports_degradation_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "mlflow":
            raise ImportError("no mlflow here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    reported: list[str] = []
    with benchmark_span("benchmark_run", {"k": "v"}, on_degraded=reported.append) as span:
        assert span is None
    assert len(reported) == 1
    assert "span_skipped" in reported[0]


def test_benchmark_span_reports_when_start_span_fails(fake_mlflow) -> None:
    fake = fake_mlflow("databricks")

    def _boom(name: str):  # noqa: ANN202
        raise RuntimeError("tracing backend unavailable")

    fake.start_span = _boom  # type: ignore[attr-defined]

    reported: list[str] = []
    with benchmark_span("benchmark_run", {"k": "v"}, on_degraded=reported.append) as span:
        assert span is None
    assert len(reported) == 1
    assert "span_skipped" in reported[0]
