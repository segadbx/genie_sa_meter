"""Genie + Supervisor benchmark and trace harness.

A read-only, configuration-driven harness that compares the direct Genie, Supervisor,
and Supervisor+MCP invocation paths for the same benchmark questions, preserving
run-level evidence (traces, token usage, latency) and producing a regenerable report.
"""

# Single source of truth for the runner version. Embedded in every persisted result so
# that a report can always be traced back to the code that produced it.
__version__ = "0.1.0"

__all__ = ["__version__"]
