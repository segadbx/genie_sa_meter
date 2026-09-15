"""Demo bootstrap for the benchmark harness.

This package is **additive and isolated** from the read-only core (config, questions,
runner, reporting, redaction, models). It provisions the Databricks resources a live
benchmark needs — a Unity Catalog schema of synthetic oil & gas data, two Genie spaces,
an optional Agent Bricks Multi-Agent Supervisor, and an MLflow experiment — then emits a
ready-to-run ``config.yaml`` and ``questions.json`` so testing becomes
``benchmark preflight`` -> ``benchmark run``.

The oil & gas use case exists only to land realistic test data in Unity Catalog; every
harness resource then points at that data.

The pure pieces (settings parsing, SQL/questions generation, ``serialized_space``
builders, config emission) are import-safe and unit-tested offline. The side-effecting
provisioner in :mod:`genie_benchmark.bootstrap.provision` imports ``databricks-sdk``
lazily so importing this package never requires a workspace.
"""

from __future__ import annotations

from .settings import BootstrapSettings, BootstrapSettingsError, load_bootstrap_settings

__all__ = [
    "BootstrapSettings",
    "BootstrapSettingsError",
    "load_bootstrap_settings",
    "run_bootstrap",
]


def run_bootstrap(*args: object, **kwargs: object) -> object:
    """Lazy entry point to the side-effecting provisioner.

    Imported lazily so ``import genie_benchmark.bootstrap`` (and the offline test suite)
    never pulls in ``databricks-sdk``.
    """
    from .provision import run_bootstrap as _run_bootstrap

    return _run_bootstrap(*args, **kwargs)
