"""Command-line interface for the benchmark harness.

Commands:

    benchmark bootstrap --config bootstrap.yaml --profile <name> [--teardown]
    benchmark preflight --config config.yaml
    benchmark run       --config config.yaml [--dry-run] [--force]
    benchmark report    --results outputs/benchmark_results.json [--output outputs]
    benchmark redact    --input outputs/benchmark_results.json --output outputs/redacted_results.json

``bootstrap`` creates the Databricks resources a live run needs (a Unity Catalog schema of
synthetic oil & gas data, two Genie spaces, an optional Multi-Agent Supervisor, and an
MLflow experiment), then writes a ready-to-run ``config.yaml`` and ``questions.json``.

``run`` fails closed if preflight has not passed for the current configuration, unless
``--force`` is supplied. ``--force`` requires an interactive confirmation and is disabled
in non-interactive (CI) execution.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, ConfigError, build_preflight_report, load_config
from .questions import QuestionValidationError, load_questions


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _preflight_marker(config: Config) -> Path:
    return Path(config.output_dir) / "preflight.ok"


def _write_preflight_marker(config: Config) -> None:
    marker = _preflight_marker(config)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"config_hash": config.config_hash, "runner_version": __version__}),
        encoding="utf-8",
    )


def _preflight_passed(config: Config) -> bool:
    marker = _preflight_marker(config)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("config_hash") == config.config_hash


def build_adapters(config: Config, *, dry_run: bool) -> dict[str, Any]:
    """Construct adapters for the configured variants.

    In dry-run mode the real Genie client (which needs a workspace) is not built.
    """
    from .adapters.supervisor import SupervisorAdapter
    from .adapters.supervisor_mcp import SupervisorMcpAdapter

    adapters: dict[str, Any] = {}
    for variant in config.variants:
        if variant == "direct_genie":
            if dry_run:
                continue
            from .adapters.direct_genie import DirectGenieAdapter, build_default_client

            adapters[variant] = DirectGenieAdapter(build_default_client(config), config)
        elif variant == "supervisor":
            adapters[variant] = SupervisorAdapter(config)
        elif variant == "supervisor_mcp":
            adapters[variant] = SupervisorMcpAdapter(config)
    return adapters


def cmd_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    questions, question_hash = load_questions(config.questions_file)
    report = build_preflight_report(config)
    report["question_count"] = len(questions)
    report["question_file_hash"] = question_hash
    report["questions_within_max"] = min(len(questions), config.max_questions)
    print(json.dumps(report, indent=2))
    _write_preflight_marker(config)
    print("\nPreflight passed. Resolved identifiers above contain no secrets.", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_benchmark

    config = load_config(args.config)
    questions, question_hash = load_questions(config.questions_file)

    if args.dry_run:
        report = build_preflight_report(config)
        planned = min(len(questions), config.max_questions) * len(config.variants) * config.repetitions
        report["dry_run"] = True
        report["question_count"] = len(questions)
        report["planned_invocations"] = planned
        report["note"] = "Dry run: configuration and questions validated; no agent was invoked."
        print(json.dumps(report, indent=2))
        _write_preflight_marker(config)
        return 0

    if not _preflight_passed(config):
        if not args.force:
            _eprint(
                "Preflight has not passed for this configuration. Run "
                "`benchmark preflight --config <file>` first, or pass --force."
            )
            return 2
        if not sys.stdin.isatty():
            _eprint("--force requires an interactive terminal and is disabled in CI.")
            return 2
        answer = input("Preflight not passed. Type 'yes' to run anyway: ").strip().lower()
        if answer != "yes":
            _eprint("Aborted.")
            return 2

    run_id = str(uuid.uuid4())
    adapters = build_adapters(config, dry_run=False)
    results_path = Path(config.output_dir) / "benchmark_results.json"
    store = run_benchmark(
        config,
        questions,
        question_hash,
        adapters,
        run_id=run_id,
        results_path=results_path,
    )
    from .reporting import generate_reports

    written = generate_reports(results_path, config.output_dir)
    print(f"Run {run_id} {store.metadata.status}. Wrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting import generate_reports

    output_dir = args.output or str(Path(args.results).parent)
    written = generate_reports(args.results, output_dir)
    print("Wrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    from .reporting import redact_results_file

    out = redact_results_file(args.input, args.output)
    print(f"Wrote redacted results: {out}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from .bootstrap import load_bootstrap_settings, run_bootstrap

    settings = load_bootstrap_settings(args.config)
    run_bootstrap(settings, args.profile, teardown=args.teardown, log=lambda m: print(m))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap", help="Provision the demo Databricks resources and emit config + questions.")
    p_boot.add_argument("--config", required=True, help="Bootstrap settings file (bootstrap.yaml).")
    p_boot.add_argument("--profile", default=None, help="Databricks profile to provision into.")
    p_boot.add_argument("--teardown", action="store_true", help="Best-effort removal of the demo resources.")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_pre = sub.add_parser("preflight", help="Validate configuration and questions.")
    p_pre.add_argument("--config", required=True)
    p_pre.set_defaults(func=cmd_preflight)

    p_run = sub.add_parser("run", help="Run the benchmark.")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--dry-run", action="store_true", help="Validate without invoking agents.")
    p_run.add_argument("--force", action="store_true", help="Run even if preflight has not passed.")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Generate reports from a results file.")
    p_rep.add_argument("--results", required=True)
    p_rep.add_argument("--output", default=None, help="Output directory (default: results dir).")
    p_rep.set_defaults(func=cmd_report)

    p_red = sub.add_parser("redact", help="Write a redacted copy of a results file.")
    p_red.add_argument("--input", required=True)
    p_red.add_argument("--output", required=True)
    p_red.set_defaults(func=cmd_redact)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, QuestionValidationError) as exc:
        _eprint(f"error: {exc}")
        return 2
    except Exception as exc:  # bootstrap settings/provisioning errors carry actionable messages
        from .bootstrap.provision import BootstrapError
        from .bootstrap.settings import BootstrapSettingsError

        if isinstance(exc, (BootstrapSettingsError, BootstrapError)):
            _eprint(f"error: {exc}")
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
