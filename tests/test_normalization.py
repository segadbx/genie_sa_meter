"""Response normalization tests: successful responses, polling states, malformed, timeout."""

from __future__ import annotations

from tests.conftest import load_fixture, valid_config_dict

from genie_benchmark.adapters.direct_genie import (
    DirectGenieAdapter,
    extract_response,
    is_terminal,
)
from genie_benchmark.config import parse_config
from genie_benchmark.models import BenchmarkQuestion, RunContext, RunStatus


class FakeGenieClient:
    """Fake Genie client driven by a sequence of get_message states."""

    def __init__(self, states: list[dict], *, conversation_id="conv1", message_id="msg1"):
        self._states = states
        self._i = 0
        self._conversation_id = conversation_id
        self._message_id = message_id
        self.create_message_calls = 0

    def start_conversation(self, space_id, content):
        return {"conversation_id": self._conversation_id, "message_id": self._message_id}

    def create_message(self, space_id, conversation_id, content):
        self.create_message_calls += 1
        return {"conversation_id": conversation_id, "message_id": self._message_id}

    def get_message(self, space_id, conversation_id, message_id):
        state = self._states[min(self._i, len(self._states) - 1)]
        self._i += 1
        return state


def _adapter(client, **cfg):
    config = parse_config(valid_config_dict(**cfg))
    return DirectGenieAdapter(client, config, sleep=lambda s: None, monotonic=_fake_clock())


def _fake_clock():
    t = {"v": 0.0}

    def clock():
        t["v"] += 0.001
        return t["v"]

    return clock


_QUESTION = BenchmarkQuestion(
    id="q001", question="revenue?", category="simple_lookup", expected_behavior="metric"
)


def _ctx(**kw):
    base = dict(run_id="run1", environment="test", request_timeout_seconds=300, poll_interval_seconds=0)
    base.update(kw)
    return RunContext(**base)


def test_is_terminal() -> None:
    assert is_terminal("COMPLETED")
    assert is_terminal("FAILED")
    assert not is_terminal("ASKING_AI")


def test_extract_response_from_fixture() -> None:
    msg = load_fixture("direct_genie_response.json")
    extracted = extract_response(msg)
    assert "Total revenue" in extracted["response_text"]
    assert extracted["generated_sql"].startswith("SELECT SUM(amount)")
    assert extracted["sql_count"] == 1
    assert "att_query_1" in extracted["attachment_ids"]


def test_polling_reaches_completed() -> None:
    states = [
        {"status": "SUBMITTED"},
        {"status": "ASKING_AI"},
        {"status": "EXECUTING_QUERY"},
        load_fixture("direct_genie_response.json"),
    ]
    adapter = _adapter(FakeGenieClient(states))
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.COMPLETED
    assert result.conversation_id == "conv_xyz789"
    assert result.client_metrics["latency_ms"] >= 0


def test_failed_terminal_status() -> None:
    adapter = _adapter(FakeGenieClient([{"status": "FAILED", "error": "boom", "id": "m"}]))
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.FAILED
    assert any("genie_terminal_status" in w for w in result.warnings)


def test_no_response_storage_keeps_only_hash() -> None:
    states = [load_fixture("direct_genie_response.json")]
    adapter = _adapter(FakeGenieClient(states))
    result = adapter.invoke(_QUESTION, _ctx(store_response_text=False))
    assert result.response_text is None
    assert result.response_sha256 is not None
    assert result.client_metrics["response_chars"] > 0


def test_response_storage_when_enabled() -> None:
    states = [load_fixture("direct_genie_response.json")]
    adapter = _adapter(FakeGenieClient(states))
    result = adapter.invoke(_QUESTION, _ctx(store_response_text=True))
    assert result.response_text is not None
    assert "Total revenue" in result.response_text


def test_timeout_returns_timeout_status() -> None:
    # Never terminal; clock advances past the timeout.
    class NeverDone(FakeGenieClient):
        def get_message(self, *a, **k):
            return {"status": "ASKING_AI"}

    config = parse_config(valid_config_dict())

    # A clock that jumps beyond the timeout on the second read.
    ticks = iter([0.0, 0.0, 1000.0, 2000.0, 3000.0])

    def clock():
        try:
            return next(ticks)
        except StopIteration:
            return 9999.0

    adapter = DirectGenieAdapter(NeverDone([]), config, sleep=lambda s: None, monotonic=clock)
    result = adapter.invoke(_QUESTION, _ctx(request_timeout_seconds=1))
    assert result.status == RunStatus.TIMEOUT


def test_missing_ids_is_failure() -> None:
    class NoIds(FakeGenieClient):
        def start_conversation(self, space_id, content):
            return {}

    adapter = _adapter(NoIds([]))
    result = adapter.invoke(_QUESTION, _ctx())
    assert result.status == RunStatus.FAILED


def test_follow_up_reuses_conversation() -> None:
    client = FakeGenieClient([load_fixture("direct_genie_response.json")])
    adapter = _adapter(client)
    result = adapter.invoke(_QUESTION, _ctx(reuse_conversation_id="conv_existing"))
    assert client.create_message_calls == 1
    assert result.status == RunStatus.COMPLETED
