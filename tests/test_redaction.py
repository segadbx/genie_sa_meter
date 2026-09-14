"""Redaction tests: tokens, headers, signed URLs, emails, nested tool arguments."""

from __future__ import annotations

from tests.conftest import FAKE_DATABRICKS_TOKEN

from genie_benchmark.redaction import (
    REDACTED,
    preview,
    redact_text,
    redact_value,
    sha256_hex,
)


def test_bearer_token_redacted() -> None:
    out = redact_text("Authorization: Bearer abcDEF123.token-value_here==")
    assert "abcDEF123" not in out
    assert REDACTED in out


def test_databricks_pat_redacted() -> None:
    out = redact_text(f"token is {FAKE_DATABRICKS_TOKEN}")
    assert FAKE_DATABRICKS_TOKEN not in out
    assert REDACTED in out


def test_email_redacted() -> None:
    out = redact_text("contact alice.smith@example.com for access")
    assert "alice.smith@example.com" not in out


def test_signed_url_redacted() -> None:
    url = "https://host/file?X-Amz-Signature=deadbeefdeadbeefdeadbeef&other=1"
    out = redact_text(url)
    assert "X-Amz-Signature" not in out or REDACTED in out
    assert "deadbeef" not in out


def test_sensitive_keys_dropped_in_nested_structure() -> None:
    payload = {
        "authorization": "Bearer secretvalue",
        "tool_args": {
            "api_key": FAKE_DATABRICKS_TOKEN,
            "note": "email me at bob@example.com",
        },
        "safe": "hello world",
    }
    out = redact_value(payload)
    assert out["authorization"] == REDACTED
    assert out["tool_args"]["api_key"] == REDACTED
    assert "bob@example.com" not in out["tool_args"]["note"]
    assert out["safe"] == "hello world"


def test_list_of_args_redacted() -> None:
    out = redact_value(["ok", "reach me: carol@example.com"])
    assert "carol@example.com" not in out[1]


def test_preview_is_bounded_and_redacted() -> None:
    text = "x" * 500 + " dave@example.com"
    p = preview(text, limit=50)
    assert len(p) <= 51  # 50 + ellipsis
    assert "dave@example.com" not in p


def test_sha256_is_stable() -> None:
    assert sha256_hex("abc") == sha256_hex("abc")
    assert sha256_hex("abc") != sha256_hex("abd")
