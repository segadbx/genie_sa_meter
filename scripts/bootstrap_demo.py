#!/usr/bin/env python3
"""Thin wrapper around ``benchmark bootstrap`` for running the demo bootstrap directly.

Equivalent to ``uv run benchmark bootstrap --config <file> --profile <name>``; provided so
the provisioner can be launched without the console script installed.

    python scripts/bootstrap_demo.py --config bootstrap.yaml --profile <name> [--teardown]
"""

from __future__ import annotations

import sys

from genie_benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bootstrap", *sys.argv[1:]]))
