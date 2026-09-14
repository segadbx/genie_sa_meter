"""Opt-in live integration test.

Skipped unless the customer explicitly opts in by setting both:

    GENIE_BENCHMARK_INTEGRATION=1
    GENIE_BENCHMARK_INTEGRATION_CONFIG=<path to a real config.yaml>

This is the only test that may make live calls. The offline suite never reaches here.
"""

from __future__ import annotations

import os

import pytest

_ENABLED = os.environ.get("GENIE_BENCHMARK_INTEGRATION") == "1"
_CONFIG = os.environ.get("GENIE_BENCHMARK_INTEGRATION_CONFIG")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not (_ENABLED and _CONFIG),
    reason="Set GENIE_BENCHMARK_INTEGRATION=1 and GENIE_BENCHMARK_INTEGRATION_CONFIG to opt in.",
)
def test_direct_genie_calibration_single_question() -> None:
    """Run a single calibration question through the direct Genie path against a live space."""
    from genie_benchmark.adapters.direct_genie import DirectGenieAdapter, build_default_client
    from genie_benchmark.config import load_config
    from genie_benchmark.models import RunContext, RunStatus
    from genie_benchmark.questions import load_questions

    config = load_config(_CONFIG)
    questions, _ = load_questions(config.questions_file)
    assert questions, "integration config must reference at least one question"

    adapter = DirectGenieAdapter(build_default_client(config), config)
    ctx = RunContext(
        run_id="integration",
        environment=config.environment,
        request_timeout_seconds=config.request_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    result = adapter.invoke(questions[0], ctx)
    assert result.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.TIMEOUT)
    assert result.conversation_id is not None or result.status != RunStatus.COMPLETED
