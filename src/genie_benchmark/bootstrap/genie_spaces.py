"""Builders for the two Genie spaces' ``serialized_space`` payloads.

Follows the version-2 ``serialized_space`` schema (see the databricks-genie-agents skill,
``serialized-space.md``): ids are 32-char lowercase hex, ``question``/``sql``/``content``
are arrays of strings, id-keyed arrays are sorted by id, and ``data_sources.tables`` are
sorted by ``identifier``. Ids are derived deterministically (md5 of stable content) so the
same inputs produce the same payload — friendly to idempotent re-provisioning.

Descriptions are written to be *routing-oriented* because the Multi-Agent Supervisor uses
each space's description to decide where to send a question.
"""

from __future__ import annotations

import hashlib
import json

PRODUCTION_DESCRIPTION = (
    "Wells and daily production for upstream oil & gas. Route questions about production "
    "volumes, oil/gas/water rates, revenue, decline, and breakdowns by well, field, region, "
    "or basin here."
)
MAINTENANCE_DESCRIPTION = (
    "Wellsite equipment and maintenance work orders. Route questions about equipment, "
    "maintenance events, work orders, failures, downtime hours, and repair cost here."
)


def _hexid(kind: str, content: str) -> str:
    """Deterministic 32-char lowercase hex id, unique per (kind, content)."""
    return hashlib.md5(f"{kind}|{content}".encode()).hexdigest()


def build_serialized_space(
    tables: list[str],
    *,
    sample_questions: list[str] | None = None,
    instruction_rules: list[str] | None = None,
    example_sqls: list[tuple[str, str]] | None = None,
) -> str:
    """Return a ``serialized_space`` JSON string for the given fully-qualified tables."""
    data_tables = sorted(({"identifier": t} for t in tables), key=lambda x: x["identifier"])
    payload: dict[str, object] = {"version": 2, "data_sources": {"tables": data_tables}}

    if sample_questions:
        sq = [{"id": _hexid("sq", q), "question": [q]} for q in sample_questions]
        payload["config"] = {"sample_questions": sorted(sq, key=lambda x: x["id"])}

    instructions: dict[str, object] = {}
    if instruction_rules:
        # text_instructions must contain at most one item; content is an array of strings.
        instructions["text_instructions"] = [
            {"id": _hexid("ti", "\n".join(instruction_rules)), "content": list(instruction_rules)}
        ]
    if example_sqls:
        eq = [{"id": _hexid("eq", q), "question": [q], "sql": [s]} for q, s in example_sqls]
        instructions["example_question_sqls"] = sorted(eq, key=lambda x: x["id"])
    if instructions:
        payload["instructions"] = instructions

    return json.dumps(payload)


def production_serialized_space(fq_schema: str) -> str:
    """Genie space over ``wells`` + ``production_daily``."""
    return build_serialized_space(
        [f"{fq_schema}.wells", f"{fq_schema}.production_daily"],
        sample_questions=[
            "What was total oil production last month?",
            "Show weekly oil production for the last 8 weeks",
            "Which fields produced the most oil last month?",
        ],
        instruction_rules=[
            "Production volumes are in production_daily: oil_bbl (barrels), gas_mcf (thousand "
            "cubic feet), water_bbl (barrels); revenue_usd is estimated gross revenue.",
            "Join production_daily to wells on well_id for field_name, region, basin, and well_type.",
            "'Last month' means the previous full calendar month. 'Last 8 weeks' means "
            "prod_date >= date_sub(current_date(), 56).",
        ],
        example_sqls=[
            (
                "What was total oil production last month?",
                f"SELECT SUM(oil_bbl) AS total_oil_bbl FROM {fq_schema}.production_daily "
                "WHERE prod_date >= date_trunc('MONTH', add_months(current_date(), -1)) "
                "AND prod_date < date_trunc('MONTH', current_date())",
            ),
            (
                "Show weekly oil production for the last 8 weeks",
                f"SELECT date_trunc('WEEK', prod_date) AS week, SUM(oil_bbl) AS oil_bbl "
                f"FROM {fq_schema}.production_daily WHERE prod_date >= date_sub(current_date(), 56) "
                "GROUP BY 1 ORDER BY 1",
            ),
        ],
    )


def maintenance_serialized_space(fq_schema: str) -> str:
    """Genie space over ``equipment`` + ``maintenance_events`` (+ shared ``wells``)."""
    return build_serialized_space(
        [f"{fq_schema}.equipment", f"{fq_schema}.maintenance_events", f"{fq_schema}.wells"],
        sample_questions=[
            "How many work orders were opened last week?",
            "Which equipment types have the most downtime?",
            "Show open critical work orders",
        ],
        instruction_rules=[
            "Maintenance work orders are rows in maintenance_events; each has event_type "
            "(inspection, repair, failure, calibration), priority, status (open/closed), "
            "downtime_hours, and cost_usd.",
            "Join maintenance_events to equipment on equipment_id for equipment_type and "
            "manufacturer; join to wells on well_id.",
            "A 'work order' is a row in maintenance_events. 'Opened last week' means "
            "event_date >= date_sub(current_date(), 7).",
        ],
        example_sqls=[
            (
                "How many work orders were opened last week?",
                f"SELECT COUNT(*) AS work_orders FROM {fq_schema}.maintenance_events "
                "WHERE event_date >= date_sub(current_date(), 7)",
            ),
            (
                "What is total equipment downtime by equipment type this month?",
                f"SELECT e.equipment_type, SUM(m.downtime_hours) AS downtime_hours "
                f"FROM {fq_schema}.maintenance_events m JOIN {fq_schema}.equipment e "
                "ON m.equipment_id = e.equipment_id "
                "WHERE m.event_date >= date_trunc('MONTH', current_date()) "
                "GROUP BY 1 ORDER BY 2 DESC",
            ),
        ],
    )
