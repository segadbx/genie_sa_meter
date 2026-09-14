"""Strict configuration loader, validation, and preflight report.

The loader fails closed: required identifiers must be present, authentication options
must be unambiguous, and the harness refuses to start if credentials appear in the
configuration file. Credentials come only from the environment, a Databricks profile,
or a secret manager — never from the config file itself.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import KNOWN_VARIANTS
from .redaction import redact_text, sha256_hex

# Prefix for environment-variable overrides, e.g. GENIE_BENCHMARK_GENIE_SPACE_ID.
ENV_PREFIX = "GENIE_BENCHMARK_"

# Scalar fields that can be overridden by an environment variable.
_ENV_OVERRIDABLE = {
    "workspace_host",
    "databricks_profile",
    "experiment_id",
    "genie_space_id",
    "supervisor_target",
    "mcp_target",
    "questions_file",
    "output_dir",
    "environment",
}

# Keys that indicate a credential was placed in the config file. Their presence makes
# the loader refuse to start (we do not read their values, only detect the key).
_CREDENTIAL_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "passwd",
    "client_secret",
    "pat",
    "bearer",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
)

# Value patterns that look like a leaked credential regardless of the key name.
_CREDENTIAL_VALUE_PATTERNS = [
    re.compile(r"\bdapi[a-f0-9]{16,}\b", re.IGNORECASE),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
]


class ConfigError(ValueError):
    """Raised for any invalid or unsafe configuration, with an actionable message."""


@dataclass
class TraceDestination:
    catalog_name: str
    schema_name: str
    table_prefix: str


@dataclass
class Config:
    """Validated, resolved configuration.

    Only non-secret identifiers live here; authentication material is resolved at call
    time from the environment or the named Databricks profile.
    """

    workspace_host: str
    genie_space_id: str
    variants: list[str]
    questions_file: str

    experiment_id: str | None = None
    databricks_profile: str | None = None
    use_default_auth: bool = False
    supervisor_target: str | None = None
    mcp_target: str | None = None
    environment: str = "unspecified"

    repetitions: int = 1
    max_questions: int = 10
    max_runtime_seconds: int | None = None
    request_timeout_seconds: int = 300
    poll_interval_seconds: int = 2
    concurrency: int = 1
    allow_concurrency: bool = False

    # Privacy: when False, only hashes and lengths of content are stored.
    store_response_text: bool = False
    output_dir: str = "outputs"

    trace_destination: TraceDestination | None = None
    # Optional customer-approved price sheet, e.g. {"model": {"input_per_1k": ..., ...}}.
    price_sheet: dict[str, Any] | None = None

    # Provenance, filled in by the loader.
    source_path: str | None = None
    config_hash: str | None = None

    def resolved_targets(self) -> dict[str, str | None]:
        """Non-secret identifiers safe to print in a preflight report."""
        return {
            "workspace_host": self.workspace_host,
            "environment": self.environment,
            "genie_space_id": self.genie_space_id,
            "experiment_id": self.experiment_id,
            "databricks_profile": self.databricks_profile,
            "supervisor_target": self.supervisor_target,
            "mcp_target": self.mcp_target,
            "variants": ",".join(self.variants),
            "questions_file": self.questions_file,
            "output_dir": self.output_dir,
        }


def _flatten_keys(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Yield (dotted_key, value) pairs for every leaf in a nested structure."""
    pairs: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                pairs.extend(_flatten_keys(value, new_prefix))
            else:
                pairs.append((new_prefix, value))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            pairs.extend(_flatten_keys(value, f"{prefix}[{i}]"))
    return pairs


def _detect_credentials(raw: dict[str, Any]) -> list[str]:
    """Return dotted key paths that look like leaked credentials."""
    offending: list[str] = []
    for dotted, value in _flatten_keys(raw):
        leaf = dotted.split(".")[-1].split("[")[0].lower()
        if any(tok in leaf for tok in _CREDENTIAL_KEY_TOKENS):
            offending.append(dotted)
            continue
        if isinstance(value, str):
            if any(p.search(value) for p in _CREDENTIAL_VALUE_PATTERNS):
                offending.append(dotted)
    return sorted(set(offending))


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    merged = dict(raw)
    for field_name in _ENV_OVERRIDABLE:
        env_key = ENV_PREFIX + field_name.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            merged[field_name] = os.environ[env_key]
    return merged


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw or raw[key] in (None, ""):
        raise ConfigError(
            f"Missing required configuration field '{key}'. "
            f"Add it to the config file or set {ENV_PREFIX}{key.upper()}."
        )
    return raw[key]


def _load_raw(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Configuration file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:  # pragma: no cover - message path
        raise ConfigError(f"Could not parse configuration file {p}: {redact_text(str(exc))}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {p} must contain a mapping at the top level.")
    return data


def parse_config(raw: dict[str, Any], *, source_path: str | None = None) -> Config:
    """Validate a raw config mapping and return a :class:`Config`.

    Split out from file loading so it can be unit-tested directly.
    """
    # 1. Refuse credentials in the config file (fail closed, before anything else).
    offending = _detect_credentials(raw)
    if offending:
        raise ConfigError(
            "Credentials must not be placed in the configuration file. "
            "Remove these fields and use the environment, a Databricks profile, or a "
            f"secret manager instead: {', '.join(offending)}"
        )

    # 2. Environment-variable overrides.
    merged = _apply_env_overrides(raw)

    # 3. Required identifiers.
    workspace_host = str(_require(merged, "workspace_host"))
    genie_space_id = str(_require(merged, "genie_space_id"))
    questions_file = str(_require(merged, "questions_file"))

    variants_raw = _require(merged, "variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ConfigError("'variants' must be a non-empty list.")
    variants = [str(v) for v in variants_raw]
    unknown = [v for v in variants if v not in KNOWN_VARIANTS]
    if unknown:
        raise ConfigError(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Valid variants are: {', '.join(sorted(KNOWN_VARIANTS))}."
        )

    # 4. Authentication options must be unambiguous (mutually exclusive).
    databricks_profile = merged.get("databricks_profile")
    use_default_auth = bool(merged.get("use_default_auth", False))
    if databricks_profile and use_default_auth:
        raise ConfigError(
            "Ambiguous authentication: set either 'databricks_profile' or "
            "'use_default_auth', not both."
        )

    # 5. Numeric guardrails.
    def _int(key: str, default: int) -> int:
        value = merged.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"'{key}' must be an integer, got {value!r}.") from exc

    concurrency = _int("concurrency", 1)
    allow_concurrency = bool(merged.get("allow_concurrency", False))
    if concurrency != 1 and not allow_concurrency:
        raise ConfigError(
            "Concurrency must be 1 for a comparable, serial benchmark. To override, set "
            "'allow_concurrency: true' explicitly (not recommended for the first session)."
        )
    if concurrency < 1:
        raise ConfigError("'concurrency' must be >= 1.")

    repetitions = _int("repetitions", 1)
    max_questions = _int("max_questions", 10)
    request_timeout_seconds = _int("request_timeout_seconds", 300)
    poll_interval_seconds = _int("poll_interval_seconds", 2)
    max_runtime = merged.get("max_runtime_seconds")
    max_runtime_seconds = int(max_runtime) if max_runtime not in (None, "") else None
    for name, value in (
        ("repetitions", repetitions),
        ("max_questions", max_questions),
        ("request_timeout_seconds", request_timeout_seconds),
        ("poll_interval_seconds", poll_interval_seconds),
    ):
        if value < 1:
            raise ConfigError(f"'{name}' must be >= 1.")

    # 6. Adapter prerequisites: if a variant is enabled, its target must be present.
    supervisor_target = merged.get("supervisor_target")
    mcp_target = merged.get("mcp_target")
    if "supervisor" in variants and not supervisor_target:
        raise ConfigError(
            "Variant 'supervisor' is enabled but 'supervisor_target' is not set. "
            "Provide the customer-approved Supervisor invocation URL or resource name."
        )
    if "supervisor_mcp" in variants and not mcp_target:
        raise ConfigError(
            "Variant 'supervisor_mcp' is enabled but 'mcp_target' is not set. "
            "Provide the customer-approved MCP connection or URL."
        )

    # 7. Optional trace destination.
    trace_destination = None
    td_raw = merged.get("trace_destination")
    if td_raw:
        if not isinstance(td_raw, dict):
            raise ConfigError("'trace_destination' must be a mapping when provided.")
        try:
            trace_destination = TraceDestination(
                catalog_name=str(td_raw["catalog_name"]),
                schema_name=str(td_raw["schema_name"]),
                table_prefix=str(td_raw["table_prefix"]),
            )
        except KeyError as exc:
            raise ConfigError(
                f"'trace_destination' is missing required field {exc}."
            ) from exc

    config = Config(
        workspace_host=workspace_host,
        genie_space_id=genie_space_id,
        variants=variants,
        questions_file=questions_file,
        experiment_id=(str(merged["experiment_id"]) if merged.get("experiment_id") else None),
        databricks_profile=(str(databricks_profile) if databricks_profile else None),
        use_default_auth=use_default_auth,
        supervisor_target=(str(supervisor_target) if supervisor_target else None),
        mcp_target=(str(mcp_target) if mcp_target else None),
        environment=str(merged.get("environment", "unspecified")),
        repetitions=repetitions,
        max_questions=max_questions,
        max_runtime_seconds=max_runtime_seconds,
        request_timeout_seconds=request_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        concurrency=concurrency,
        allow_concurrency=allow_concurrency,
        store_response_text=bool(merged.get("store_response_text", False)),
        output_dir=str(merged.get("output_dir", "outputs")),
        trace_destination=trace_destination,
        price_sheet=merged.get("price_sheet"),
        source_path=source_path,
    )

    # Deterministic config hash over the resolved, non-secret identifiers.
    config.config_hash = sha256_hex(
        json.dumps(config.resolved_targets(), sort_keys=True, separators=(",", ":"))
    )
    return config


def load_config(path: str | Path) -> Config:
    """Load, override, and validate configuration from a YAML or JSON file."""
    raw = _load_raw(path)
    return parse_config(raw, source_path=str(path))


def build_preflight_report(config: Config) -> dict[str, Any]:
    """Assemble a preflight report of resolved, non-secret identifiers.

    The report intentionally contains no credentials or full payloads.
    """
    return {
        "runner_version": _runner_version(),
        "config_hash": config.config_hash,
        "resolved_targets": config.resolved_targets(),
        "settings": {
            "repetitions": config.repetitions,
            "max_questions": config.max_questions,
            "max_runtime_seconds": config.max_runtime_seconds,
            "request_timeout_seconds": config.request_timeout_seconds,
            "poll_interval_seconds": config.poll_interval_seconds,
            "concurrency": config.concurrency,
            "store_response_text": config.store_response_text,
        },
        "trace_destination": (
            {
                "catalog_name": config.trace_destination.catalog_name,
                "schema_name": config.trace_destination.schema_name,
                "table_prefix": config.trace_destination.table_prefix,
            }
            if config.trace_destination
            else None
        ),
    }


def _runner_version() -> str:
    from . import __version__

    return __version__
