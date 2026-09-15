"""Serial benchmark orchestration with per-request persistence and guardrails.

The runner runs only the configured variants, serially by default, using a fresh
conversation for independent questions and reusing conversation state only for explicit
follow-ups. Each result is persisted immediately so an interrupted run still produces
useful evidence. A single request failure is recorded and the run continues; a guardrail
trip (authorization, budget, loop/fan-out, unexpected routing, or the max-runtime limit)
stops the run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .models import BenchmarkQuestion, InvocationResult, RunContext
from .tracing import benchmark_span, configure_mlflow_tracing

# Adapter factory: variant name -> callable building an AgentAdapter for that variant.
AdapterFactory = Callable[[Config], Any]

# Substrings that mark an authorization guardrail (case-insensitive).
_AUTH_SIGNALS = ("authorization", "unauthorized", "permission denied", "forbidden", "403", "401")
_LOOP_SIGNALS = ("loop detected", "max iterations", "fan-out", "fanout")
_BUDGET_SIGNALS = ("budget", "quota exceeded", "usage limit")


class GuardrailStop(Exception):
    """Raised to stop the whole run when a guardrail trips."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class RunMetadata:
    run_id: str
    runner_version: str
    environment: str
    config_hash: str | None
    question_file_hash: str
    variants: list[str]
    started_at: str
    completed_at: str | None = None
    status: str = "running"
    stop_reason: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runner_version": self.runner_version,
            "environment": self.environment,
            "config_hash": self.config_hash,
            "question_file_hash": self.question_file_hash,
            "variants": self.variants,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "counts": self.counts,
            "warnings": self.warnings,
        }


class ResultStore:
    """Persists run metadata and results, rewriting the file after each result.

    Rewriting the full file (rather than appending) keeps a single valid JSON document at
    all times, so an interrupted run leaves a readable, report-able file behind.
    """

    def __init__(self, path: str | Path, metadata: RunMetadata) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.results: list[InvocationResult] = []
        self._flush()

    def add(self, result: InvocationResult) -> None:
        self.results.append(result)
        self._flush()

    def finalize(self, status: str, stop_reason: str | None, completed_at: str) -> None:
        self.metadata.status = status
        self.metadata.stop_reason = stop_reason
        self.metadata.completed_at = completed_at
        self.metadata.counts = self._status_counts()
        self._flush()

    def _status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts

    def _flush(self) -> None:
        payload = {
            "run_metadata": self.metadata.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)


def classify_guardrail(result: InvocationResult) -> str | None:
    """Return a guardrail reason if this result should stop the run, else ``None``.

    A plain FAILED/TIMEOUT row does not stop the run (the row is marked failed and the run
    continues). Authorization, budget, and loop/fan-out signals do stop it.
    """
    text = " ".join(
        str(x) for x in ([result.error] + list(result.warnings)) if x
    ).lower()
    if any(sig in text for sig in _AUTH_SIGNALS):
        return f"authorization guardrail on {result.question_id}/{result.variant}"
    if any(sig in text for sig in _BUDGET_SIGNALS):
        return f"budget guardrail on {result.question_id}/{result.variant}"
    if any(sig in text for sig in _LOOP_SIGNALS):
        return f"loop/fan-out guardrail on {result.question_id}/{result.variant}"
    return None


def _plan(
    questions: list[BenchmarkQuestion], variants: list[str], repetitions: int, max_questions: int
) -> list[tuple[BenchmarkQuestion, str, int]]:
    """Build the deterministic (question, variant, repetition) execution plan."""
    selected = questions[:max_questions]
    plan: list[tuple[BenchmarkQuestion, str, int]] = []
    for question in selected:
        for variant in variants:
            for rep in range(1, repetitions + 1):
                plan.append((question, variant, rep))
    return plan


def run_benchmark(
    config: Config,
    questions: list[BenchmarkQuestion],
    question_file_hash: str,
    adapters: dict[str, Any],
    *,
    run_id: str,
    results_path: str | Path,
    monotonic: Callable[[], float] = time.monotonic,
) -> ResultStore:
    """Execute the benchmark and return the populated :class:`ResultStore`.

    ``adapters`` maps variant name to a ready adapter instance (injected so tests can use
    fakes and the CLI can build real ones).
    """
    from .util import utc_now_iso

    metadata = RunMetadata(
        run_id=run_id,
        runner_version=__version__,
        environment=config.environment,
        config_hash=config.config_hash,
        question_file_hash=question_file_hash,
        variants=config.variants,
        started_at=utc_now_iso(),
    )
    # Point MLflow at the workspace experiment before any span opens; record any
    # degradation as a run-level warning instead of silently dropping traces.
    metadata.warnings.extend(configure_mlflow_tracing(config))
    store = ResultStore(results_path, metadata)

    # Deduplicate per-request span degradations into one warning apiece.
    seen_span_warnings: set[str] = set()

    def _record_span_degraded(reason: str) -> None:
        if reason not in seen_span_warnings:
            seen_span_warnings.add(reason)
            metadata.warnings.append(reason)

    plan = _plan(questions, config.variants, config.repetitions, config.max_questions)
    # conversation ids keyed by (question_id, variant) for follow-up reuse.
    conversations: dict[tuple[str, str], str] = {}
    deadline = (
        monotonic() + config.max_runtime_seconds
        if config.max_runtime_seconds is not None
        else None
    )
    stop_reason: str | None = None
    status = "completed"

    try:
        for question, variant, rep in plan:
            if deadline is not None and monotonic() >= deadline:
                stop_reason = "max_runtime_seconds reached"
                status = "stopped"
                break

            adapter = adapters.get(variant)
            if adapter is None:
                continue

            reuse_id = None
            if question.is_follow_up:
                reuse_id = conversations.get((question.follow_up_to or "", variant))

            context = RunContext(
                run_id=run_id,
                environment=config.environment,
                repetition=rep,
                reuse_conversation_id=reuse_id,
                store_response_text=config.store_response_text,
                request_timeout_seconds=config.request_timeout_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
                redaction_policy="default",
            )

            span_attrs = {
                "run_id": run_id,
                "question_id": question.id,
                "variant": variant,
                "environment": config.environment,
                "target_name": _target_name(config, variant),
                "conversation_mode": "follow_up" if question.is_follow_up else "fresh",
                "redaction_policy": context.redaction_policy,
            }
            with benchmark_span(
                "benchmark_run", span_attrs, on_degraded=_record_span_degraded
            ):
                result = adapter.invoke(question, context)

            store.add(result)

            if result.conversation_id:
                conversations[(question.id, variant)] = result.conversation_id

            reason = classify_guardrail(result)
            if reason is not None:
                stop_reason = reason
                status = "stopped"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "interrupted by user; completed results were preserved"
    except GuardrailStop as exc:
        status = "stopped"
        stop_reason = exc.reason

    store.finalize(status=status, stop_reason=stop_reason, completed_at=utc_now_iso())
    return store


def _target_name(config: Config, variant: str) -> str | None:
    return {
        "direct_genie": config.genie_space_id,
        "supervisor": config.supervisor_target,
        "supervisor_mcp": config.mcp_target,
    }.get(variant)
