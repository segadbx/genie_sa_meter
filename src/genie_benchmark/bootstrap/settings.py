"""Bootstrap settings loader.

The bootstrap is *config-driven*: catalog/schema names, warehouse reuse-vs-create, data
scale, Genie space titles, supervisor choice, experiment path, and what to emit all come
from a ``bootstrap.yaml`` file — nothing is hard-coded. This module is pure (no workspace
calls) so it can be unit-tested offline.

Like the runtime config loader, it refuses to start if a credential-looking key or value
is present: the bootstrap resolves authentication from the ``--profile`` at call time,
never from the settings file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Hold the bootstrap file to the same "no secrets on disk" bar as config.yaml. We reuse the
# runtime loader's value patterns (leaked-token shapes) and its key flattener, but use a
# local key-token set: the runtime set includes bare "pat", which false-positives on the
# many legitimate "*_path" keys in bootstrap settings.
from ..config import _CREDENTIAL_VALUE_PATTERNS, _flatten_keys  # noqa: PLC2701

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_CREDENTIAL_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "passwd",
    "client_secret",
    "bearer",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
    "credential",
)


def _detect_credentials(raw: dict[str, Any]) -> list[str]:
    """Return dotted key paths that look like leaked credentials (bootstrap-scoped)."""
    offending: list[str] = []
    for dotted, value in _flatten_keys(raw):
        leaf = dotted.split(".")[-1].split("[")[0].lower()
        if any(tok in leaf for tok in _CREDENTIAL_KEY_TOKENS):
            offending.append(dotted)
            continue
        if isinstance(value, str) and any(p.search(value) for p in _CREDENTIAL_VALUE_PATTERNS):
            offending.append(dotted)
    return sorted(set(offending))


class BootstrapSettingsError(ValueError):
    """Raised for invalid bootstrap settings, with an actionable message."""


@dataclass
class CatalogSettings:
    name: str = "genie_bench_demo"
    create: bool = True


@dataclass
class SchemaSettings:
    name: str = "oil_gas"


@dataclass
class WarehouseSettings:
    reuse_id: str | None = None
    create_if_missing: bool = True
    name: str = "genie-bench-demo"
    size: str = "2X-Small"


@dataclass
class DataSettings:
    wells: int = 40
    days: int = 540
    seed: int = 42


@dataclass
class GenieSettings:
    parent_path: str | None = None  # default resolved to /Workspace/Users/<me>/genie_spaces
    production_space_name: str = "Upstream Production"
    maintenance_space_name: str = "Upstream Maintenance"


@dataclass
class SupervisorSettings:
    build: bool = True
    name: str = "Upstream Benchmark MAS"


@dataclass
class ExperimentSettings:
    path: str | None = None  # default resolved to /Users/<me>/genie-bench-demo


@dataclass
class EmitSettings:
    config_path: str = "config.yaml"
    questions_path: str = "questions.json"
    variants: list[str] = field(
        default_factory=lambda: ["direct_genie", "supervisor", "supervisor_mcp"]
    )
    environment: str = "oil-gas-demo"


@dataclass
class BootstrapSettings:
    """Validated bootstrap inputs. Only non-secret names/toggles live here."""

    workspace_host: str | None = None
    catalog: CatalogSettings = field(default_factory=CatalogSettings)
    schema: SchemaSettings = field(default_factory=SchemaSettings)
    warehouse: WarehouseSettings = field(default_factory=WarehouseSettings)
    data: DataSettings = field(default_factory=DataSettings)
    genie: GenieSettings = field(default_factory=GenieSettings)
    supervisor: SupervisorSettings = field(default_factory=SupervisorSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    emit: EmitSettings = field(default_factory=EmitSettings)

    source_path: str | None = None

    @property
    def catalog_name(self) -> str:
        return self.catalog.name

    @property
    def schema_name(self) -> str:
        return self.schema.name

    @property
    def fq_schema(self) -> str:
        return f"{self.catalog.name}.{self.schema.name}"


def _as_dict(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BootstrapSettingsError(f"'{key}' must be a mapping when provided.")
    return value


def _as_bool(value: Any, key: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise BootstrapSettingsError(f"'{key}' must be true or false, got {value!r}.")


def _as_int(value: Any, key: str, default: int, *, minimum: int = 1) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BootstrapSettingsError(f"'{key}' must be an integer, got {value!r}.") from exc
    if parsed < minimum:
        raise BootstrapSettingsError(f"'{key}' must be >= {minimum}, got {parsed}.")
    return parsed


def _as_str(value: Any, key: str, default: str) -> str:
    if value in (None, ""):
        return default
    return str(value)


def _opt_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _require_identifier(value: str, key: str) -> str:
    # Catalog/schema names go straight into SQL DDL, so constrain them to a safe shape.
    if not _IDENTIFIER_RE.match(value):
        raise BootstrapSettingsError(
            f"'{key}' must be a simple SQL identifier (letters, digits, underscore; not "
            f"starting with a digit); got {value!r}."
        )
    return value


def parse_bootstrap_settings(raw: dict[str, Any], *, source_path: str | None = None) -> BootstrapSettings:
    """Validate a raw mapping into :class:`BootstrapSettings`."""
    if not isinstance(raw, dict):
        raise BootstrapSettingsError("Bootstrap settings must be a mapping at the top level.")

    offending = _detect_credentials(raw)
    if offending:
        raise BootstrapSettingsError(
            "Credentials must not be placed in the bootstrap settings file. Remove these "
            "fields and authenticate with --profile instead: " + ", ".join(offending)
        )

    cat_raw = _as_dict(raw.get("catalog"), "catalog")
    catalog = CatalogSettings(
        name=_require_identifier(_as_str(cat_raw.get("name"), "catalog.name", "genie_bench_demo"), "catalog.name"),
        create=_as_bool(cat_raw.get("create"), "catalog.create", True),
    )

    sch_raw = _as_dict(raw.get("schema"), "schema")
    schema = SchemaSettings(
        name=_require_identifier(_as_str(sch_raw.get("name"), "schema.name", "oil_gas"), "schema.name"),
    )

    wh_raw = _as_dict(raw.get("warehouse"), "warehouse")
    warehouse = WarehouseSettings(
        reuse_id=_opt_str(wh_raw.get("reuse_id")),
        create_if_missing=_as_bool(wh_raw.get("create_if_missing"), "warehouse.create_if_missing", True),
        name=_as_str(wh_raw.get("name"), "warehouse.name", "genie-bench-demo"),
        size=_as_str(wh_raw.get("size"), "warehouse.size", "2X-Small"),
    )
    if warehouse.reuse_id is None and not warehouse.create_if_missing:
        raise BootstrapSettingsError(
            "warehouse.reuse_id is empty and warehouse.create_if_missing is false: the "
            "bootstrap has no warehouse to use. Set one of them."
        )

    data_raw = _as_dict(raw.get("data"), "data")
    data = DataSettings(
        wells=_as_int(data_raw.get("wells"), "data.wells", 40),
        days=_as_int(data_raw.get("days"), "data.days", 540),
        seed=_as_int(data_raw.get("seed"), "data.seed", 42, minimum=0),
    )

    genie_raw = _as_dict(raw.get("genie"), "genie")
    genie = GenieSettings(
        parent_path=_opt_str(genie_raw.get("parent_path")),
        production_space_name=_as_str(
            genie_raw.get("production_space_name"), "genie.production_space_name", "Upstream Production"
        ),
        maintenance_space_name=_as_str(
            genie_raw.get("maintenance_space_name"), "genie.maintenance_space_name", "Upstream Maintenance"
        ),
    )

    sup_raw = _as_dict(raw.get("supervisor"), "supervisor")
    supervisor = SupervisorSettings(
        build=_as_bool(sup_raw.get("build"), "supervisor.build", True),
        name=_as_str(sup_raw.get("name"), "supervisor.name", "Upstream Benchmark MAS"),
    )

    exp_raw = _as_dict(raw.get("experiment"), "experiment")
    experiment = ExperimentSettings(path=_opt_str(exp_raw.get("path")))

    emit_raw = _as_dict(raw.get("emit"), "emit")
    variants_raw = emit_raw.get("variants")
    if variants_raw is None:
        variants = ["direct_genie", "supervisor", "supervisor_mcp"]
    elif isinstance(variants_raw, list) and variants_raw:
        variants = [str(v) for v in variants_raw]
    else:
        raise BootstrapSettingsError("'emit.variants' must be a non-empty list when provided.")
    emit = EmitSettings(
        config_path=_as_str(emit_raw.get("config_path"), "emit.config_path", "config.yaml"),
        questions_path=_as_str(emit_raw.get("questions_path"), "emit.questions_path", "questions.json"),
        variants=variants,
        environment=_as_str(emit_raw.get("environment"), "emit.environment", "oil-gas-demo"),
    )

    # If a variant is emitted, the bootstrap must be able to satisfy its target. The
    # supervisor/supervisor_mcp targets are produced only when supervisor.build is true.
    if not supervisor.build and ("supervisor" in variants or "supervisor_mcp" in variants):
        raise BootstrapSettingsError(
            "emit.variants includes 'supervisor' or 'supervisor_mcp' but supervisor.build is "
            "false, so no supervisor/MCP target would be produced. Either set "
            "supervisor.build: true or drop those variants from emit.variants."
        )

    return BootstrapSettings(
        workspace_host=_opt_str(raw.get("workspace_host")),
        catalog=catalog,
        schema=schema,
        warehouse=warehouse,
        data=data,
        genie=genie,
        supervisor=supervisor,
        experiment=experiment,
        emit=emit,
        source_path=source_path,
    )


def load_bootstrap_settings(path: str | Path) -> BootstrapSettings:
    """Load and validate bootstrap settings from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise BootstrapSettingsError(f"Bootstrap settings file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise BootstrapSettingsError(f"Could not parse bootstrap settings file {p}: {exc}") from exc
    if data is None:
        data = {}
    return parse_bootstrap_settings(data, source_path=str(p))
