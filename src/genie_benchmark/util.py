"""Small shared utilities (time, content-metadata capture)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .redaction import sha256_hex


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def content_metadata(
    text: str | None,
    *,
    store_text: bool,
) -> dict[str, Any]:
    """Return the storable view of a piece of content.

    When ``store_text`` is False (default privacy posture), only the character length and
    a SHA-256 hash are returned; the raw text is dropped. When True, the text is included
    as well so content is retained only under an explicit configuration flag.
    """
    if text is None:
        return {"text": None, "sha256": None, "chars": 0}
    meta: dict[str, Any] = {"sha256": sha256_hex(text), "chars": len(text)}
    meta["text"] = text if store_text else None
    return meta
