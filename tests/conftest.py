"""Shared test fixtures and helpers. All tests are offline (no live calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# A realistic-looking but fake Databricks PAT, assembled at runtime so the literal token
# string never appears in source (which would trip secret scanners / commit hooks). The
# runtime value is `dapi` + 32 hex chars, which matches the `dapi[a-f0-9]{16,}` detection
# used by the config loader and the redactor.
FAKE_DATABRICKS_TOKEN = "dapi" + "0123456789abcdef" * 2


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def valid_config_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal valid config mapping (direct_genie only) with optional overrides."""
    base: dict[str, Any] = {
        "workspace_host": "https://example.cloud.databricks.com",
        "genie_space_id": "space_001",
        "variants": ["direct_genie"],
        "questions_file": "questions.json",
        "environment": "test",
        "output_dir": "outputs",
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_questions() -> list[dict[str, Any]]:
    return [
        {
            "id": "q001",
            "question": "What was total revenue last month?",
            "category": "simple_lookup",
            "expected_behavior": "Returns the metric.",
        },
        {
            "id": "q002",
            "question": "Break it down by region.",
            "category": "follow_up",
            "expected_behavior": "Reuses q001 state.",
            "follow_up_to": "q001",
        },
    ]
