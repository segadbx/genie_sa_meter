"""Deterministic report generation from persisted results.

All outputs are generated from the persisted normalized results file (not from in-memory
summaries), sorted deterministically by ``question_id, variant, repetition`` so re-running
the report on the same inputs produces byte-stable output. The report distinguishes
observed token usage, child-span totals, pending billing, and unavailable measurements,
and never labels an inferred value as billed cost.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .redaction import redact_text

SUMMARY_COLUMNS = [
    "question_id",
    "variant",
    "repetition",
    "status",
    "latency_ms",
    "trace_level_total_tokens",
    "child_span_total_tokens",
    "token_source",
    "llm_call_count",
    "tool_call_count",
    "mcp_call_count",
    "response_chars",
    "quality_score",
    "trace_id",
    "notes",
]


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a persisted results file (``{run_metadata, results}``)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"{p} is not a valid benchmark results file.")
    return data


def _sort_key(result: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(result.get("question_id", "")),
        str(result.get("variant", "")),
        int(result.get("repetition", 0) or 0),
    )


def _row(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("raw_token_usage", {}) or {}
    trace_level = (raw.get("trace_level") or {}) if isinstance(raw.get("trace_level"), dict) else {}
    child = (raw.get("child_span_sum") or {}) if isinstance(raw.get("child_span_sum"), dict) else {}
    token_usage = result.get("token_usage", {}) or {}
    counts = result.get("counts", {}) or {}
    client_metrics = result.get("client_metrics", {}) or {}
    warnings = result.get("warnings", []) or []
    notes_parts = list(warnings)
    if result.get("error"):
        notes_parts.append(f"error={result['error']}")
    notes = redact_text("; ".join(str(n) for n in notes_parts))
    return {
        "question_id": result.get("question_id", ""),
        "variant": result.get("variant", ""),
        "repetition": result.get("repetition", ""),
        "status": result.get("status", ""),
        "latency_ms": client_metrics.get("latency_ms", ""),
        "trace_level_total_tokens": trace_level.get("total_tokens", ""),
        "child_span_total_tokens": child.get("total_tokens", ""),
        "token_source": token_usage.get("source", ""),
        "llm_call_count": counts.get("llm_call_count", ""),
        "tool_call_count": counts.get("tool_call_count", ""),
        "mcp_call_count": counts.get("mcp_call_count", ""),
        "response_chars": client_metrics.get("response_chars", ""),
        "quality_score": result.get("quality_score", ""),
        "trace_id": result.get("trace_id") or "",
        "notes": notes.replace("\n", " ").replace("\r", " "),
    }


def render_summary_csv(data: dict[str, Any]) -> str:
    """Render the deterministic summary CSV as a string."""
    results = sorted(data.get("results", []), key=_sort_key)
    buffer = io.StringIO()
    # Fixed newline so output is byte-stable across platforms.
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow(_row(result))
    return buffer.getvalue()


def render_trace_links(data: dict[str, Any]) -> str:
    """Render trace references only — never credentials or signed URLs."""
    results = sorted(data.get("results", []), key=_sort_key)
    lines = ["# trace references (trace ids only; no credentials or signed URLs)"]
    for result in results:
        trace_id = result.get("trace_id")
        label = f"{result.get('question_id')}/{result.get('variant')}/rep{result.get('repetition')}"
        if trace_id:
            lines.append(f"{label}\t{trace_id}")
        else:
            lines.append(f"{label}\t<no trace id available>")
    return "\n".join(lines) + "\n"


def _data_quality(data: dict[str, Any]) -> list[str]:
    results = data.get("results", [])
    issues: list[str] = []
    missing_trace = [r for r in results if not r.get("trace_id")]
    missing_tokens = [
        r
        for r in results
        if (r.get("token_usage", {}) or {}).get("source") in (None, "unavailable")
    ]
    not_configured = [r for r in results if r.get("status") == "NOT_CONFIGURED"]
    failed = [r for r in results if r.get("status") in ("FAILED", "TIMEOUT")]
    if missing_trace:
        issues.append(f"- {len(missing_trace)} result(s) have no trace id (span-level analysis limited).")
    if missing_tokens:
        issues.append(f"- {len(missing_tokens)} result(s) have no token usage (trace-level or child-span).")
    if not_configured:
        variants = sorted({r.get("variant") for r in not_configured})
        issues.append(
            f"- {len(not_configured)} result(s) are NOT_CONFIGURED "
            f"(customer boundary not wired): {', '.join(str(v) for v in variants)}. "
            "These variants are not comparable until implemented."
        )
    if failed:
        issues.append(f"- {len(failed)} result(s) failed or timed out; see the notes column.")
    issues.append("- Billing reconciliation is pending; token counts are observed telemetry, not billed cost.")
    return issues


def render_report_md(data: dict[str, Any]) -> str:
    """Render the customer-readable Markdown report."""
    meta = data.get("run_metadata", {})
    results = sorted(data.get("results", []), key=_sort_key)
    lines: list[str] = []
    lines.append("# Genie + Supervisor Benchmark Report")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- Runner version: `{meta.get('runner_version', 'unknown')}`")
    lines.append(f"- Run id: `{meta.get('run_id', 'unknown')}`")
    lines.append(f"- Environment: `{meta.get('environment', 'unspecified')}`")
    lines.append(f"- Config hash: `{meta.get('config_hash', 'unknown')}`")
    lines.append(f"- Question file hash: `{meta.get('question_file_hash', 'unknown')}`")
    lines.append(f"- Variants: {', '.join(meta.get('variants', []))}")
    lines.append(f"- Started at: {meta.get('started_at', 'unknown')}")
    lines.append(f"- Completed at: {meta.get('completed_at', 'unknown')}")
    lines.append(f"- Status: **{meta.get('status', 'unknown')}**")
    if meta.get("stop_reason"):
        lines.append(f"- Stop reason: {redact_text(str(meta['stop_reason']))}")
    lines.append(f"- Result count: {len(results)}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| question | variant | rep | status | latency (ms) | trace tokens | "
        "child-span tokens | llm | tool | mcp |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        row = _row(r)
        lines.append(
            f"| {row['question_id']} | {row['variant']} | {row['repetition']} | "
            f"{row['status']} | {row['latency_ms']} | {row['trace_level_total_tokens']} | "
            f"{row['child_span_total_tokens']} | {row['llm_call_count']} | "
            f"{row['tool_call_count']} | {row['mcp_call_count']} |"
        )
    lines.append("")

    lines.append("## Measurement categories")
    lines.append("")
    lines.append("- **Observed tokens**: read from trace-level usage or summed from child spans.")
    lines.append("- **Child-span totals**: summed LLM/Chat-model span usage (may exceed trace-level UI totals).")
    lines.append("- **Billing pending**: reconcile against `system.billing.usage` separately (see `sql/`).")
    lines.append("- **Estimated cost**: only computed when a customer-approved price sheet is supplied.")
    lines.append("- **Unavailable / non-comparable**: fields that could not be measured or variants not wired.")
    lines.append("")

    lines.append("## Data quality")
    lines.append("")
    lines.extend(_data_quality(data))
    lines.append("")

    lines.append("## Limitations and next actions")
    lines.append("")
    lines.append("- A trace status of `OK` is not proof the answer is correct; score correctness separately.")
    lines.append("- Supervisor / MCP variants require the customer-approved invocation to be wired in.")
    lines.append("- Reconcile observed telemetry with billing by workspace, identity, endpoint, SKU, and time window.")
    lines.append("")
    return "\n".join(lines)


def generate_reports(
    results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Generate all report files from a persisted results file.

    Returns a mapping of logical name -> written path. ``benchmark_results.json`` is
    written as a normalized, deterministically sorted copy of the input.
    """
    data = load_results(results_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Normalized results copy (sorted) so the canonical results file is deterministic too.
    sorted_data = {
        "run_metadata": data.get("run_metadata", {}),
        "results": sorted(data.get("results", []), key=_sort_key),
    }
    written: dict[str, str] = {}

    results_out = out / "benchmark_results.json"
    results_out.write_text(json.dumps(sorted_data, indent=2), encoding="utf-8")
    written["results"] = str(results_out)

    summary_out = out / "benchmark_summary.csv"
    summary_out.write_text(render_summary_csv(data), encoding="utf-8")
    written["summary"] = str(summary_out)

    report_out = out / "benchmark_report.md"
    report_out.write_text(render_report_md(data), encoding="utf-8")
    written["report"] = str(report_out)

    links_out = out / "trace_links.txt"
    links_out.write_text(render_trace_links(data), encoding="utf-8")
    written["trace_links"] = str(links_out)

    return written


def redact_results_file(input_path: str | Path, output_path: str | Path) -> str:
    """Write a redacted copy of a results file (drops content, scrubs strings)."""
    from .redaction import redact_value

    data = load_results(input_path)
    redacted_results = []
    for r in data.get("results", []):
        copy = dict(r)
        # Drop any stored raw content; keep hashes and lengths.
        copy["response_text"] = None
        copy["generated_sql"] = None
        redacted_results.append(redact_value(copy))
    payload = {
        "run_metadata": redact_value(data.get("run_metadata", {})),
        "results": redacted_results,
    }
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(outp)


def maybe_estimated_cost(
    total_tokens: int | None, price_sheet: dict[str, Any] | None
) -> float | None:
    """Compute an estimated cost only when a customer-approved price sheet is supplied.

    Returns ``None`` (never zero, never an inferred cost) when no price is available, so
    the report never presents an unapproved dollar figure.
    """
    if total_tokens is None or not price_sheet:
        return None
    per_1k = price_sheet.get("total_per_1k")
    if per_1k is None:
        return None
    return round((total_tokens / 1000.0) * float(per_1k), 6)
