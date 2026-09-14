"""Configuration validation tests: missing fields, secret detection, env overrides, invalid combos."""

from __future__ import annotations

import json

import pytest
from tests.conftest import FAKE_DATABRICKS_TOKEN, valid_config_dict

from genie_benchmark.config import ConfigError, load_config, parse_config


def test_valid_config_parses() -> None:
    config = parse_config(valid_config_dict())
    assert config.genie_space_id == "space_001"
    assert config.variants == ["direct_genie"]
    assert config.concurrency == 1
    assert config.config_hash  # deterministic hash computed


def test_missing_required_field_is_actionable() -> None:
    raw = valid_config_dict()
    del raw["genie_space_id"]
    with pytest.raises(ConfigError, match="genie_space_id"):
        parse_config(raw)


def test_credentials_in_config_are_refused_by_key() -> None:
    raw = valid_config_dict(databricks_token=FAKE_DATABRICKS_TOKEN)
    with pytest.raises(ConfigError, match="Credentials must not be placed"):
        parse_config(raw)


def test_credentials_in_config_are_refused_by_value_pattern() -> None:
    # Key name is innocuous, but the value looks like a PAT.
    raw = valid_config_dict(note=FAKE_DATABRICKS_TOKEN)
    with pytest.raises(ConfigError, match="Credentials must not be placed"):
        parse_config(raw)


def test_env_override_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENIE_BENCHMARK_GENIE_SPACE_ID", "space_from_env")
    config = parse_config(valid_config_dict())
    assert config.genie_space_id == "space_from_env"


def test_concurrency_must_be_one_unless_overridden() -> None:
    with pytest.raises(ConfigError, match="Concurrency must be 1"):
        parse_config(valid_config_dict(concurrency=4))
    config = parse_config(valid_config_dict(concurrency=4, allow_concurrency=True))
    assert config.concurrency == 4


def test_ambiguous_auth_rejected() -> None:
    raw = valid_config_dict(databricks_profile="DEFAULT", use_default_auth=True)
    with pytest.raises(ConfigError, match="Ambiguous authentication"):
        parse_config(raw)


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown variant"):
        parse_config(valid_config_dict(variants=["direct_genie", "made_up"]))


def test_supervisor_variant_requires_target() -> None:
    with pytest.raises(ConfigError, match="supervisor_target"):
        parse_config(valid_config_dict(variants=["supervisor"]))
    config = parse_config(
        valid_config_dict(variants=["supervisor"], supervisor_target="endpoints/sup1")
    )
    assert config.supervisor_target == "endpoints/sup1"


def test_mcp_variant_requires_target() -> None:
    with pytest.raises(ConfigError, match="mcp_target"):
        parse_config(valid_config_dict(variants=["supervisor_mcp"]))


def test_resolved_targets_contains_no_secrets() -> None:
    config = parse_config(valid_config_dict())
    targets = config.resolved_targets()
    serialized = json.dumps(targets).lower()
    for token in ("token", "secret", "password", "bearer"):
        assert token not in serialized


def test_load_config_from_file(tmp_path) -> None:
    import yaml

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(valid_config_dict()), encoding="utf-8")
    config = load_config(cfg)
    assert config.source_path == str(cfg)
    assert config.config_hash


def test_same_config_produces_same_hash() -> None:
    a = parse_config(valid_config_dict())
    b = parse_config(valid_config_dict())
    assert a.config_hash == b.config_hash
