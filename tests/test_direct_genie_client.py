"""Regression tests for the SDK-backed Genie client wrapper.

The offline adapter contract tests inject a fake client that returns plain dicts, so they
never exercise ``build_default_client`` / ``_SdkGenieClient``. That gap let a real bug
ship: ``genie.start_conversation`` / ``genie.create_message`` return a ``Wait[GenieMessage]``
long-running-operation wrapper (not the message), and calling ``as_dict()`` on the ``Wait``
tripped its ``__getattr__`` and raised ``KeyError('as_dict')`` — surfacing in the benchmark
report as ``Unexpected Genie error: 'as_dict'`` for every direct_genie question.

These tests drive the wrapper through the *real* SDK ``Wait`` type so the regression can't
recur, without making any live workspace call.
"""

from __future__ import annotations

import databricks.sdk
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import GenieMessage, GenieStartConversationResponse
from tests.conftest import valid_config_dict

from genie_benchmark.adapters.direct_genie import build_default_client
from genie_benchmark.config import parse_config


class _FakeGenieApi:
    """Mimics ``WorkspaceClient.genie`` return types for the three used methods."""

    def start_conversation(self, space_id: str, content: str) -> Wait[GenieMessage]:
        op = {
            "conversation_id": "conv_xyz789",
            "message_id": "msg_abc123",
            "message": {"id": "msg_abc123", "status": "SUBMITTED"},
        }
        return Wait(
            lambda **k: None,
            response=GenieStartConversationResponse.from_dict(op),
            conversation_id=op["conversation_id"],
            message_id=op["message_id"],
            space_id=space_id,
        )

    def create_message(
        self, space_id: str, conversation_id: str, content: str
    ) -> Wait[GenieMessage]:
        op = {"id": "msg_def456", "message_id": "msg_def456", "conversation_id": conversation_id}
        return Wait(
            lambda **k: None,
            response=GenieMessage.from_dict(op),
            conversation_id=conversation_id,
            message_id=op["message_id"],
            space_id=space_id,
        )

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> GenieMessage:
        return GenieMessage.from_dict(
            {
                "id": message_id,
                "message_id": message_id,
                "conversation_id": conversation_id,
                "status": "COMPLETED",
            }
        )


class _FakeWorkspaceClient:
    def __init__(self, **_kwargs) -> None:  # ignores host/profile
        self.genie = _FakeGenieApi()


def _client(monkeypatch):
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", _FakeWorkspaceClient)
    config = parse_config(valid_config_dict())
    return build_default_client(config)


def test_start_conversation_unwraps_wait(monkeypatch) -> None:
    client = _client(monkeypatch)
    # Must not raise KeyError('as_dict'); must expose the ids the adapter reads.
    result = client.start_conversation("sp1", "revenue?")
    assert result["conversation_id"] == "conv_xyz789"
    assert result["message_id"] == "msg_abc123"


def test_create_message_unwraps_wait(monkeypatch) -> None:
    client = _client(monkeypatch)
    result = client.create_message("sp1", "conv_xyz789", "and by region?")
    assert result["message_id"] == "msg_def456"
    assert result["conversation_id"] == "conv_xyz789"


def test_get_message_returns_dict(monkeypatch) -> None:
    client = _client(monkeypatch)
    result = client.get_message("sp1", "conv_xyz789", "msg_abc123")
    assert result["status"] == "COMPLETED"
