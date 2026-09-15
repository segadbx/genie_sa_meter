"""Offline tests for the demo bootstrap's pure modules (no workspace calls)."""

from __future__ import annotations

import json
import re

import pytest
import yaml

from genie_benchmark.bootstrap import genie_spaces, oil_gas
from genie_benchmark.bootstrap.emit import build_config_dict, write_config, write_questions
from genie_benchmark.bootstrap.settings import (
    BootstrapSettingsError,
    load_bootstrap_settings,
    parse_bootstrap_settings,
)
from genie_benchmark.config import parse_config
from genie_benchmark.questions import parse_questions

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


# --- settings -------------------------------------------------------------------------

def test_settings_defaults_from_empty_mapping() -> None:
    s = parse_bootstrap_settings({})
    assert s.catalog.name == "genie_bench_demo"
    assert s.schema.name == "oil_gas"
    assert s.fq_schema == "genie_bench_demo.oil_gas"
    assert s.emit.variants == ["direct_genie", "supervisor", "supervisor_mcp"]


def test_settings_reject_bad_identifier() -> None:
    with pytest.raises(BootstrapSettingsError, match="catalog.name"):
        parse_bootstrap_settings({"catalog": {"name": "bad-name!"}})


def test_settings_reject_credentials() -> None:
    with pytest.raises(BootstrapSettingsError, match="Credentials must not"):
        parse_bootstrap_settings({"api_token": "x"})


def test_settings_reject_no_warehouse() -> None:
    with pytest.raises(BootstrapSettingsError, match="no warehouse"):
        parse_bootstrap_settings({"warehouse": {"reuse_id": "", "create_if_missing": False}})


def test_settings_reject_supervisor_variant_without_build() -> None:
    with pytest.raises(BootstrapSettingsError, match="supervisor.build"):
        parse_bootstrap_settings(
            {"supervisor": {"build": False}, "emit": {"variants": ["direct_genie", "supervisor"]}}
        )


def test_settings_load_from_file(tmp_path) -> None:
    p = tmp_path / "bootstrap.yaml"
    p.write_text(yaml.safe_dump({"catalog": {"name": "demo_cat"}, "data": {"wells": 5}}), encoding="utf-8")
    s = load_bootstrap_settings(p)
    assert s.catalog.name == "demo_cat"
    assert s.data.wells == 5


# --- oil_gas --------------------------------------------------------------------------

def test_ddl_and_inserts_cover_all_tables() -> None:
    fq = "cat.sch"
    ddl = oil_gas.table_ddl(fq)
    inserts = oil_gas.data_inserts(fq, oil_gas.DataSettings(wells=3, days=10, seed=1))
    assert set(ddl) == set(oil_gas.ORDERED_TABLES) == set(inserts)
    for table in oil_gas.ORDERED_TABLES:
        assert "COMMENT" in ddl[table]
        assert f"{fq}.{table}" in ddl[table]
        assert f"{fq}.{table}" in inserts[table]


def test_benchmark_questions_are_valid_and_linked() -> None:
    questions = oil_gas.benchmark_questions()
    parsed = parse_questions(questions)  # raises if invalid (dup ids, missing fields, bad follow-up)
    assert len(parsed) == 8
    ids = {q.id for q in parsed}
    assert len(ids) == len(parsed)
    follow = next(q for q in parsed if q.id == "q003")
    assert follow.follow_up_to == "q002"


# --- genie_spaces ---------------------------------------------------------------------

def _assert_serialized_space_valid(raw: str) -> dict:
    payload = json.loads(raw)
    assert payload["version"] == 2
    tables = payload["data_sources"]["tables"]
    assert tables == sorted(tables, key=lambda t: t["identifier"])
    sq = payload["config"]["sample_questions"]
    assert all(_HEX32.match(q["id"]) for q in sq)
    assert sq == sorted(sq, key=lambda x: x["id"])
    assert all(isinstance(q["question"], list) for q in sq)
    ti = payload["instructions"]["text_instructions"]
    assert len(ti) == 1 and isinstance(ti[0]["content"], list)
    eq = payload["instructions"]["example_question_sqls"]
    assert eq == sorted(eq, key=lambda x: x["id"])
    return payload


def test_production_space_serialized_shape() -> None:
    payload = _assert_serialized_space_valid(genie_spaces.production_serialized_space("cat.sch"))
    identifiers = {t["identifier"] for t in payload["data_sources"]["tables"]}
    assert identifiers == {"cat.sch.wells", "cat.sch.production_daily"}


def test_maintenance_space_serialized_shape() -> None:
    payload = _assert_serialized_space_valid(genie_spaces.maintenance_serialized_space("cat.sch"))
    identifiers = {t["identifier"] for t in payload["data_sources"]["tables"]}
    assert identifiers == {"cat.sch.equipment", "cat.sch.maintenance_events", "cat.sch.wells"}


def test_serialized_space_is_deterministic() -> None:
    assert genie_spaces.production_serialized_space("cat.sch") == genie_spaces.production_serialized_space("cat.sch")


# --- emit -----------------------------------------------------------------------------

def test_build_config_dict_passes_validation_and_has_no_secrets() -> None:
    settings = parse_bootstrap_settings({"catalog": {"name": "c"}, "schema": {"name": "s"}})
    config = build_config_dict(
        settings,
        workspace_host="https://example.cloud.databricks.com",
        profile="MY_PROFILE",
        genie_space_id="space_prod",
        experiment_id="exp1",
        supervisor_target="https://example.cloud.databricks.com/serving-endpoints/mas/invocations",
        mcp_target="https://example.cloud.databricks.com/api/2.0/mcp/genie/space_prod",
    )
    parse_config(dict(config))  # must not raise
    assert config["genie_space_id"] == "space_prod"
    assert config["supervisor_target"].endswith("/invocations")
    serialized = json.dumps(config).lower()
    for tok in ("token", "secret", "password", "bearer"):
        assert tok not in serialized


def test_build_config_dict_omits_targets_for_absent_variants() -> None:
    settings = parse_bootstrap_settings({"emit": {"variants": ["direct_genie"]}})
    config = build_config_dict(
        settings,
        workspace_host="https://example.cloud.databricks.com",
        profile=None,
        genie_space_id="space_prod",
        experiment_id=None,
        supervisor_target="https://x/serving-endpoints/mas/invocations",
        mcp_target="https://x/api/2.0/mcp/genie/space_prod",
    )
    assert "supervisor_target" not in config
    assert "mcp_target" not in config


def test_write_config_patches_existing(tmp_path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"store_response_text": True, "max_questions": 3}), encoding="utf-8")
    write_config(p, {"workspace_host": "https://x", "genie_space_id": "g1"})
    merged = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert merged["store_response_text"] is True  # preserved
    assert merged["genie_space_id"] == "g1"  # added


def test_write_questions_round_trips(tmp_path) -> None:
    p = tmp_path / "questions.json"
    write_questions(p, oil_gas.benchmark_questions())
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    assert parse_questions(reloaded)  # valid
