"""Adapter contract tests using checked-in fixtures. No live customer calls are made."""

from __future__ import annotations

from tests.conftest import load_fixture, valid_config_dict

from genie_benchmark.adapters.base import AgentAdapter
from genie_benchmark.adapters.direct_genie import DirectGenieAdapter
from genie_benchmark.adapters.supervisor import SupervisorAdapter
from genie_benchmark.adapters.supervisor_mcp import SupervisorMcpAdapter
from genie_benchmark.config import parse_config
from genie_benchmark.models import BenchmarkQuestion, InvocationResult, RunContext, RunStatus

_QUESTION = BenchmarkQuestion(
    id="q001", question="revenue?", category="simple_lookup", expected_behavior="metric"
)


def _ctx():
    return RunContext(run_id="run1", environment="test", poll_interval_seconds=0)


class _FixtureGenieClient:
    def start_conversation(self, space_id, content):
        return {"conversation_id": "conv_xyz789", "message_id": "msg_abc123"}

    def get_message(self, *a, **k):
        return load_fixture("direct_genie_response.json")


def _assert_normalized(result: InvocationResult, variant: str) -> None:
    assert isinstance(result, InvocationResult)
    assert result.variant == variant
    assert result.question_id == "q001"
    assert result.run_id == "run1"
    assert result.started_at and result.completed_at
    assert isinstance(result.warnings, list)


def test_direct_genie_adapter_conforms() -> None:
    config = parse_config(valid_config_dict())
    adapter = DirectGenieAdapter(_FixtureGenieClient(), config, sleep=lambda s: None)
    assert isinstance(adapter, AgentAdapter)
    result = adapter.invoke(_QUESTION, _ctx())
    _assert_normalized(result, "direct_genie")
    assert result.status == RunStatus.COMPLETED


def test_supervisor_adapter_reports_prerequisite() -> None:
    config = parse_config(valid_config_dict(variants=["supervisor"], supervisor_target="res/1"))
    adapter = SupervisorAdapter(config)
    assert isinstance(adapter, AgentAdapter)
    result = adapter.invoke(_QUESTION, _ctx())
    _assert_normalized(result, "supervisor")
    assert result.status == RunStatus.NOT_CONFIGURED
    assert "not configured" in (result.error or "").lower()


def test_supervisor_mcp_adapter_reports_prerequisite() -> None:
    config = parse_config(valid_config_dict(variants=["supervisor_mcp"], mcp_target="mcp/1"))
    adapter = SupervisorMcpAdapter(config)
    assert isinstance(adapter, AgentAdapter)
    result = adapter.invoke(_QUESTION, _ctx())
    _assert_normalized(result, "supervisor_mcp")
    assert result.status == RunStatus.NOT_CONFIGURED


def test_result_to_dict_is_json_safe() -> None:
    config = parse_config(valid_config_dict())
    adapter = DirectGenieAdapter(_FixtureGenieClient(), config, sleep=lambda s: None)
    result = adapter.invoke(_QUESTION, _ctx())
    d = result.to_dict()
    assert d["status"] == "COMPLETED"
    assert d["token_usage"]["source"] == "unavailable"  # Genie response carries no token usage
