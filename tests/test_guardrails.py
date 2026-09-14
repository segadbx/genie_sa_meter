"""Guardrail and orchestration tests: max questions, runtime, stop conditions, persistence."""

from __future__ import annotations

import json

from tests.conftest import valid_config_dict

from genie_benchmark.config import parse_config
from genie_benchmark.models import BenchmarkQuestion, InvocationResult, RunStatus
from genie_benchmark.questions import parse_questions
from genie_benchmark.runner import classify_guardrail, run_benchmark


def _result(question, context, *, status=RunStatus.COMPLETED, error=None, warnings=None, conversation_id="c1"):
    return InvocationResult(
        run_id=context.run_id,
        question_id=question.id,
        variant="direct_genie",
        repetition=context.repetition,
        started_at="2026-09-14T00:00:00Z",
        completed_at="2026-09-14T00:00:01Z",
        status=status,
        conversation_id=conversation_id,
        error=error,
        warnings=warnings or [],
    )


class FakeAdapter:
    name = "direct_genie"

    def __init__(self, *, error_on=None, raise_kbi_on=None):
        self.error_on = error_on or {}
        self.raise_kbi_on = raise_kbi_on
        self.calls = 0
        self.contexts = []

    def invoke(self, question, context):
        self.calls += 1
        self.contexts.append(context)
        if self.raise_kbi_on is not None and self.calls == self.raise_kbi_on:
            raise KeyboardInterrupt()
        err = self.error_on.get(question.id)
        if err:
            return _result(question, context, status=RunStatus.FAILED, error=err)
        return _result(question, context)


def _questions(n):
    return [
        BenchmarkQuestion(id=f"q{i:03d}", question="?", category="c", expected_behavior="b")
        for i in range(1, n + 1)
    ]


def test_max_questions_limits_plan(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=2))
    adapter = FakeAdapter()
    store = run_benchmark(
        config, _questions(5), "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=tmp_path / "res.json",
    )
    assert adapter.calls == 2
    assert len(store.results) == 2
    assert store.metadata.status == "completed"


def test_authorization_guardrail_stops_run(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=5))
    adapter = FakeAdapter(error_on={"q002": "authorization failed: permission denied"})
    store = run_benchmark(
        config, _questions(5), "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=tmp_path / "res.json",
    )
    assert store.metadata.status == "stopped"
    assert "authorization" in (store.metadata.stop_reason or "")
    # Stopped after the offending question; did not process all 5.
    assert adapter.calls == 2


def test_plain_failure_does_not_stop_run(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=3))
    adapter = FakeAdapter(error_on={"q002": "Genie could not answer the question."})
    store = run_benchmark(
        config, _questions(3), "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=tmp_path / "res.json",
    )
    assert adapter.calls == 3
    assert store.metadata.status == "completed"


def test_max_runtime_stops_run(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=10, max_runtime_seconds=5))
    adapter = FakeAdapter()
    ticks = iter([0.0, 1.0, 2.0, 100.0, 200.0])

    def clock():
        try:
            return next(ticks)
        except StopIteration:
            return 999.0

    store = run_benchmark(
        config, _questions(10), "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=tmp_path / "res.json", monotonic=clock,
    )
    assert store.metadata.status == "stopped"
    assert "max_runtime" in (store.metadata.stop_reason or "")


def test_interrupt_preserves_completed_results(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=5))
    adapter = FakeAdapter(raise_kbi_on=3)
    path = tmp_path / "res.json"
    store = run_benchmark(
        config, _questions(5), "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=path,
    )
    assert store.metadata.status == "interrupted"
    # Two results completed and were persisted before the interrupt on the third call.
    assert len(store.results) == 2
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk["results"]) == 2
    assert on_disk["run_metadata"]["status"] == "interrupted"


def test_results_persisted_incrementally(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=3))
    path = tmp_path / "res.json"

    class WatcherAdapter(FakeAdapter):
        def invoke(self, question, context):
            result = super().invoke(question, context)
            # File should already exist (metadata flushed at construction).
            assert path.exists()
            return result

    run_benchmark(
        config, _questions(3), "qhash", {"direct_genie": WatcherAdapter()},
        run_id="r1", results_path=path,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["run_metadata"]["status"] == "completed"


def test_follow_up_reuses_prior_conversation_id(tmp_path) -> None:
    config = parse_config(valid_config_dict(max_questions=5))
    questions = parse_questions(
        [
            {"id": "q001", "question": "a", "category": "c", "expected_behavior": "b"},
            {"id": "q002", "question": "b", "category": "c", "expected_behavior": "b", "follow_up_to": "q001"},
        ]
    )
    adapter = FakeAdapter()
    run_benchmark(
        config, questions, "qhash", {"direct_genie": adapter},
        run_id="r1", results_path=tmp_path / "res.json",
    )
    # Second invocation (q002) should carry the reuse id from q001's conversation.
    assert adapter.contexts[0].reuse_conversation_id is None
    assert adapter.contexts[1].reuse_conversation_id == "c1"


def test_classify_guardrail_variants() -> None:
    q = BenchmarkQuestion(id="q1", question="?", category="c", expected_behavior="b")
    from genie_benchmark.models import RunContext

    ctx = RunContext(run_id="r", environment="t")
    assert classify_guardrail(_result(q, ctx, status=RunStatus.FAILED, error="boom")) is None
    assert classify_guardrail(_result(q, ctx, error="403 Forbidden")) is not None
    assert classify_guardrail(_result(q, ctx, error="usage limit reached: budget")) is not None
    assert classify_guardrail(_result(q, ctx, warnings=["loop detected"])) is not None
