"""CLI tests: preflight, dry-run, fail-closed run, redact. All offline."""

from __future__ import annotations

import json

import yaml
from tests.conftest import valid_config_dict

from genie_benchmark.cli import main


def _write_config_and_questions(tmp_path) -> str:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [{"id": "q001", "question": "revenue?", "category": "c", "expected_behavior": "b"}]
        ),
        encoding="utf-8",
    )
    cfg = valid_config_dict(
        questions_file=str(questions),
        output_dir=str(tmp_path / "out"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(config_path)


def test_preflight_succeeds_and_hides_secrets(tmp_path, capsys) -> None:
    config_path = _write_config_and_questions(tmp_path)
    rc = main(["preflight", "--config", config_path])
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["question_count"] == 1
    assert "token" not in json.dumps(report["resolved_targets"]).lower()
    # Marker written for the run fail-closed check.
    assert (tmp_path / "out" / "preflight.ok").exists()


def test_dry_run_validates_without_invoking(tmp_path, capsys) -> None:
    config_path = _write_config_and_questions(tmp_path)
    rc = main(["run", "--config", config_path, "--dry-run"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["planned_invocations"] == 1


def test_run_fails_closed_without_preflight(tmp_path, capsys) -> None:
    config_path = _write_config_and_questions(tmp_path)
    rc = main(["run", "--config", config_path])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Preflight has not passed" in err


def test_invalid_config_reports_error(tmp_path, capsys) -> None:
    cfg = valid_config_dict()
    del cfg["genie_space_id"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = main(["preflight", "--config", str(config_path)])
    assert rc == 2
    assert "genie_space_id" in capsys.readouterr().err


def test_redact_command(tmp_path, capsys) -> None:
    payload = {
        "run_metadata": {"run_id": "r1"},
        "results": [
            {
                "question_id": "q001",
                "variant": "direct_genie",
                "repetition": 1,
                "status": "COMPLETED",
                "response_text": "answer for alice@example.com",
            }
        ],
    }
    src = tmp_path / "res.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    dst = tmp_path / "redacted.json"
    rc = main(["redact", "--input", str(src), "--output", str(dst)])
    assert rc == 0
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["results"][0]["response_text"] is None
    assert "alice@example.com" not in json.dumps(data)
