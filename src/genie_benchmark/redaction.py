"""Redaction and hashing helpers.

Redaction is applied by default to anything that may reach logs or persisted output.
The harness never prints access tokens, authorization headers, cookies, signed URLs,
or secret values, and redacts likely secrets, emails, phone numbers, account
identifiers, and bearer tokens.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

REDACTED = "[REDACTED]"

# Keys whose values are always sensitive and must never be emitted.
_SENSITIVE_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "cookie",
    "api_key",
    "apikey",
    "client_secret",
    "private_key",
    "credential",
    "bearer",
    "signature",
    "sig",
    "x-amz-",
)

# Ordered (pattern, replacement) rules applied to free text. Order matters: more
# specific patterns (bearer tokens, signed URLs) run before generic ones (emails).
_TEXT_RULES: list[tuple[re.Pattern[str], str]] = [
    # Authorization: Bearer <token>
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), r"\1 " + REDACTED),
    # Databricks personal access tokens (dapi...) and OAuth-style long tokens.
    (re.compile(r"\bdapi[a-f0-9]{16,}\b", re.IGNORECASE), REDACTED),
    (re.compile(r"\bdkea[a-z0-9]{8,}\b", re.IGNORECASE), REDACTED),
    # Signed URLs / presigned query strings (drop the whole query string).
    (
        re.compile(
            r"https?://[^\s\"']+[?&](?:x-amz-signature|sig|signature|token|"
            r"x-goog-signature|se|sv|sig)=[^\s\"'&]+[^\s\"']*",
            re.IGNORECASE,
        ),
        REDACTED,
    ),
    # Emails.
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), REDACTED),
    # Phone numbers (loose international / US formats).
    (
        re.compile(r"(?<!\w)(?:\+?\d[\d\-\s().]{7,}\d)(?!\w)"),
        REDACTED,
    ),
    # Long opaque high-entropy tokens (>=32 base64/hex-ish chars).
    (re.compile(r"\b[A-Za-z0-9\-_]{32,}\b"), REDACTED),
]


def _key_is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(tok in lowered for tok in _SENSITIVE_KEY_TOKENS)


def redact_text(value: str) -> str:
    """Redact likely secrets and PII from a free-text string."""
    redacted = value
    for pattern, replacement in _TEXT_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact a JSON-like structure.

    A value under a sensitive key (e.g. ``authorization``) is dropped entirely; other
    strings are scrubbed with :func:`redact_text`.
    """
    if key is not None and _key_is_sensitive(key):
        return REDACTED
    if isinstance(value, dict):
        return {k: redact_value(v, key=k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def sha256_hex(value: str) -> str:
    """Stable SHA-256 hex digest of a UTF-8 string (used for content hashing)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def preview(value: str, *, limit: int = 120) -> str:
    """Redacted, length-bounded preview suitable for logs and correlation."""
    scrubbed = redact_text(value)
    if len(scrubbed) <= limit:
        return scrubbed
    return scrubbed[:limit] + "…"
