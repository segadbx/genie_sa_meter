"""Retry tests: transient polling failure vs non-retryable authorization failure."""

from __future__ import annotations

from tests.conftest import load_fixture, valid_config_dict

from genie_benchmark.adapters.base import NonRetryableError, TransientError
from genie_benchmark.adapters.direct_genie import DirectGenieAdapter
from genie_benchmark.config import parse_config
from genie_benchmark.models import BenchmarkQuestion, RunContext, RunStatus

_QUESTION = BenchmarkQuestion(
    id="q001", question="revenue?", category="simple_lookup", expected_behavior="metric"
)


def _ctx():
    return RunContext(
        run_id="run1", environment="test", request_timeout_seconds=300, poll_interval_seconds=0
    )


def _clock():
    t = {"v": 0.0}

    def c():
        t["v"] += 0.001
        return t["v"]

    return c


def _adapter(client, **kw):
    config = parse_config(valid_config_dict())
    return DirectGenieAdapter(client, config, sleep=lambda s: None, monotonic=_clock(), **kw)


class TransientThenOk:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def start_conversation(self, space_id, content):
        return {"conversation_id": "c1", "message_id": "m1"}

    def get_message(self, *a, **k):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError("temporary poll glitch")
        return load_fixture("direct_genie_response.json")


class AuthFailure:
    def start_conversation(self, space_id, content):
        return {"conversation_id": "c1", "message_id": "m1"}

    def get_message(self, *a, **k):
        raise NonRetryableError("authorization failed: permission denied")


def test_transient_failures_below_threshold_recover() -> None:
    client = TransientThenOk(fail_times=2)
    adapter = _adapter(client, max_transient_retries=3)
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.COMPLETED
    assert any("transient_poll_retry" in w for w in result.warnings)


def test_transient_failures_above_threshold_fail() -> None:
    client = TransientThenOk(fail_times=10)
    adapter = _adapter(client, max_transient_retries=3)
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.FAILED


def test_non_retryable_auth_error_not_retried() -> None:
    client = AuthFailure()
    adapter = _adapter(client)
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.FAILED
    assert "authorization" in (result.error or "").lower()
