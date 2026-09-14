"""Reporting tests: deterministic CSV ordering, partial-failure report, redaction, no-storage."""

from __future__ import annotations

import json

from genie_benchmark.reporting import (
    generate_reports,
    maybe_estimated_cost,
    redact_results_file,
    render_summary_csv,
)


def _results_payload() -> dict:
    return {
        "run_metadata": {
            "run_id": "run1",
            "runner_version": "0.1.0",
            "environment": "test",
            "config_hash": "cfg",
            "question_file_hash": "qhash",
            "variants": ["direct_genie", "supervisor"],
            "started_at": "2026-09-14T00:00:00Z",
            "completed_at": "2026-09-14T00:05:00Z",
            "status": "completed",
        },
        "results": [
            # Deliberately out of order to test deterministic sorting.
            {
                "question_id": "q002",
                "variant": "direct_genie",
                "repetition": 1,
                "status": "COMPLETED",
                "client_metrics": {"latency_ms": 900, "response_chars": 42},
                "token_usage": {"total_tokens": 100, "source": "trace_attribute"},
                "raw_token_usage": {"trace_level": {"total_tokens": 100}, "child_span_sum": {"total_tokens": 120}},
                "counts": {"llm_call_count": 2, "tool_call_count": 1, "mcp_call_count": 0},
                "trace_id": "tr2",
                "warnings": [],
            },
            {
                "question_id": "q001",
                "variant": "supervisor",
                "repetition": 1,
                "status": "NOT_CONFIGURED",
                "error": "Supervisor invocation is not configured.",
                "warnings": ["supervisor_adapter_not_configured"],
            },
            {
                "question_id": "q001",
                "variant": "direct_genie",
                "repetition": 1,
                "status": "FAILED",
                "error": "reach me at leak@example.com",
                "client_metrics": {"latency_ms": 500, "response_chars": 0},
                "token_usage": {"source": "unavailable"},
                "warnings": ["trace_id_unavailable"],
            },
        ],
    }


def _write(tmp_path):
    p = tmp_path / "benchmark_results.json"
    p.write_text(json.dumps(_results_payload()), encoding="utf-8")
    return p


def test_csv_is_deterministically_sorted(tmp_path) -> None:
    csv_text = render_summary_csv(_results_payload())
    lines = csv_text.strip().splitlines()
    # header + 3 rows, sorted by (question_id, variant, repetition)
    assert lines[1].startswith("q001,direct_genie")
    assert lines[2].startswith("q001,supervisor")
    assert lines[3].startswith("q002,direct_genie")


def test_report_is_byte_stable(tmp_path) -> None:
    results = _write(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    generate_reports(results, out1)
    generate_reports(results, out2)
    a = (out1 / "benchmark_summary.csv").read_bytes()
    b = (out2 / "benchmark_summary.csv").read_bytes()
    assert a == b


def test_report_handles_partial_failures(tmp_path) -> None:
    results = _write(tmp_path)
    out = tmp_path / "out"
    written = generate_reports(results, out)
    report = (out / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Data quality" in report
    assert "NOT_CONFIGURED" in report
    assert "failed or timed out" in report
    assert set(written) == {"results", "summary", "report", "trace_links"}


def test_summary_notes_are_redacted(tmp_path) -> None:
    csv_text = render_summary_csv(_results_payload())
    assert "leak@example.com" not in csv_text


def test_trace_links_never_contain_credentials(tmp_path) -> None:
    results = _write(tmp_path)
    out = tmp_path / "out"
    generate_reports(results, out)
    links = (out / "trace_links.txt").read_text(encoding="utf-8")
    assert "tr2" in links
    assert "no trace id available" in links  # the failed row


def test_redact_results_file_drops_content(tmp_path) -> None:
    payload = _results_payload()
    payload["results"][0]["response_text"] = "sensitive answer with bob@example.com"
    src = tmp_path / "r.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    dst = tmp_path / "redacted.json"
    redact_results_file(src, dst)
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["results"][0]["response_text"] is None
    text = json.dumps(data)
    assert "bob@example.com" not in text


def test_estimated_cost_only_with_price_sheet() -> None:
    assert maybe_estimated_cost(1000, None) is None
    assert maybe_estimated_cost(None, {"total_per_1k": 1.0}) is None
    assert maybe_estimated_cost(2000, {"total_per_1k": 1.5}) == 3.0
