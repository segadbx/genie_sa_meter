"""Oil & gas demo data model: DDL, synthetic-data SQL, and benchmark questions.

The use case is **upstream production & equipment monitoring**, split into two domains so
two Genie spaces (and a meaningful supervisor router) are natural:

* production: ``wells`` + ``production_daily``
* maintenance: ``equipment`` + ``maintenance_events`` (+ shared ``wells``)

All SQL here is Databricks/Spark SQL, generated with ``sequence``/``explode``/``rand`` so
data is produced on a SQL warehouse via the Statement Execution API — no cluster, no
Faker/Spark dependency. Every table and column carries a ``COMMENT`` because Genie answer
quality depends on it. This module is pure: it returns SQL strings and question dicts; the
provisioner executes them.
"""

from __future__ import annotations

from .settings import DataSettings

# Table load order: parents before children (foreign keys reference earlier tables).
ORDERED_TABLES = ["wells", "production_daily", "equipment", "maintenance_events"]


def table_ddl(fq_schema: str) -> dict[str, str]:
    """``CREATE TABLE IF NOT EXISTS`` for each table, with table + column comments."""
    return {
        "wells": f"""
CREATE TABLE IF NOT EXISTS {fq_schema}.wells (
  well_id    STRING COMMENT 'Unique well identifier (primary key)',
  well_name  STRING COMMENT 'Human-readable well name',
  field_name STRING COMMENT 'Production field the well belongs to',
  region     STRING COMMENT 'Geographic region / operating area',
  basin      STRING COMMENT 'Geological basin',
  well_type  STRING COMMENT 'Primary product of the well: oil or gas',
  spud_date  DATE   COMMENT 'Date drilling began (spud date)',
  status     STRING COMMENT 'Current well status: producing or shut_in',
  lat        DOUBLE COMMENT 'Approximate surface latitude',
  lon        DOUBLE COMMENT 'Approximate surface longitude'
) COMMENT 'Upstream oil & gas wells master list'
""".strip(),
        "production_daily": f"""
CREATE TABLE IF NOT EXISTS {fq_schema}.production_daily (
  well_id            STRING COMMENT 'Well identifier (foreign key to wells.well_id)',
  prod_date          DATE   COMMENT 'Production date (one row per well per calendar day)',
  oil_bbl            DOUBLE COMMENT 'Oil produced that day, in barrels (bbl)',
  gas_mcf            DOUBLE COMMENT 'Gas produced that day, in thousand cubic feet (mcf)',
  water_bbl          DOUBLE COMMENT 'Water produced that day, in barrels (bbl)',
  revenue_usd        DOUBLE COMMENT 'Estimated gross revenue for the day, in USD',
  operating_cost_usd DOUBLE COMMENT 'Estimated operating cost for the day, in USD'
) COMMENT 'Daily production volumes and revenue per well'
""".strip(),
        "equipment": f"""
CREATE TABLE IF NOT EXISTS {fq_schema}.equipment (
  equipment_id   STRING COMMENT 'Unique equipment identifier (primary key)',
  well_id        STRING COMMENT 'Well the equipment is installed on (foreign key to wells.well_id)',
  equipment_type STRING COMMENT 'Type of equipment (ESP Pump, Separator, Compressor, Wellhead, ...)',
  manufacturer   STRING COMMENT 'Equipment manufacturer',
  install_date   DATE   COMMENT 'Date the equipment was installed'
) COMMENT 'Wellsite equipment inventory'
""".strip(),
        "maintenance_events": f"""
CREATE TABLE IF NOT EXISTS {fq_schema}.maintenance_events (
  event_id       STRING COMMENT 'Unique maintenance work-order identifier (primary key)',
  equipment_id   STRING COMMENT 'Equipment the event applies to (foreign key to equipment.equipment_id)',
  well_id        STRING COMMENT 'Well the event applies to (foreign key to wells.well_id)',
  event_date     DATE   COMMENT 'Date the maintenance work order was opened',
  event_type     STRING COMMENT 'Event type: inspection, repair, failure, calibration',
  priority       STRING COMMENT 'Work-order priority: low, medium, high, critical',
  status         STRING COMMENT 'Work-order status: open or closed',
  downtime_hours DOUBLE COMMENT 'Equipment downtime attributed to the event, in hours',
  cost_usd       DOUBLE COMMENT 'Maintenance cost for the event, in USD'
) COMMENT 'Equipment maintenance events / work orders'
""".strip(),
    }


def data_inserts(fq_schema: str, data: DataSettings) -> dict[str, str]:
    """``INSERT`` statements that generate seeded synthetic rows for each table.

    Data is anchored to end on the current date so that time-relative questions ("last
    month", "last 8 weeks") always have rows. Randomness uses ``rand(<seed>)`` for
    reproducibility across runs with the same seed.
    """
    wells = data.wells
    days = data.days
    seed = data.seed
    last_day = days - 1
    # Mild decline: keep the multiplier positive across the whole window.
    decline_oil = days * 3
    decline_gas = days * 4

    return {
        "wells": f"""
INSERT INTO {fq_schema}.wells
SELECT
  concat('WELL-', lpad(cast(id AS string), 4, '0')) AS well_id,
  concat('Well ', cast(id AS string), ' ',
         element_at(array('Alpha','Bravo','Charlie','Delta','Echo'), cast(pmod(id, 5) AS int) + 1)) AS well_name,
  element_at(array('Midland','Delaware','Bakken','Eagle Ford','DJ'), cast(pmod(id, 5) AS int) + 1) AS field_name,
  element_at(array('West Texas','West Texas','North Dakota','South Texas','Colorado'),
             cast(pmod(id, 5) AS int) + 1) AS region,
  element_at(array('Permian','Permian','Williston','Western Gulf','Denver-Julesburg'),
             cast(pmod(id, 5) AS int) + 1) AS basin,
  CASE WHEN pmod(id, 4) = 0 THEN 'gas' ELSE 'oil' END AS well_type,
  date_add(DATE'2015-01-01', cast(pmod(id * 7, 2500) AS int)) AS spud_date,
  CASE WHEN pmod(id, 13) = 0 THEN 'shut_in' ELSE 'producing' END AS status,
  round(31.0 + rand({seed}) * 6.0, 5) AS lat,
  round(-103.5 + rand({seed + 1}) * 6.0, 5) AS lon
FROM (SELECT explode(sequence(1, {wells})) AS id)
""".strip(),
        "production_daily": f"""
INSERT INTO {fq_schema}.production_daily
SELECT
  well_id, prod_date, oil_bbl, gas_mcf, water_bbl,
  round(oil_bbl * 75.0 + gas_mcf * 2.5, 2) AS revenue_usd,
  round(oil_bbl * 11.0 + gas_mcf * 0.4 + 450.0, 2) AS operating_cost_usd
FROM (
  SELECT
    w.well_id,
    date_add(date_sub(current_date(), {last_day}), d.day_no) AS prod_date,
    round(greatest(0.0, (120 + pmod(hash(w.well_id), 500))
          * (1.0 - d.day_no / {decline_oil}.0) * (0.85 + rand({seed}) * 0.3)), 1) AS oil_bbl,
    round(greatest(0.0, (400 + pmod(hash(w.well_id), 2200))
          * (1.0 - d.day_no / {decline_gas}.0) * (0.85 + rand({seed + 1}) * 0.3)
          * CASE WHEN w.well_type = 'gas' THEN 2.5 ELSE 1.0 END), 1) AS gas_mcf,
    round(greatest(0.0, (30 + pmod(hash(w.well_id), 220)) * (0.85 + rand({seed + 2}) * 0.3)), 1) AS water_bbl
  FROM {fq_schema}.wells w
  LATERAL VIEW explode(sequence(0, {last_day})) d AS day_no
  WHERE w.status = 'producing'
)
""".strip(),
        "equipment": f"""
INSERT INTO {fq_schema}.equipment
SELECT
  concat('EQ-', substr(w.well_id, 6), '-', cast(k AS string)) AS equipment_id,
  w.well_id,
  element_at(array('ESP Pump','Separator','Compressor','Wellhead','Flow Meter','Rod Pump'),
             cast(pmod(hash(w.well_id, k), 6) AS int) + 1) AS equipment_type,
  element_at(array('Baker Hughes','SLB','Halliburton','Weatherford'),
             cast(pmod(hash(w.well_id), 4) AS int) + 1) AS manufacturer,
  date_add(DATE'2016-01-01', cast(pmod(hash(w.well_id, k), 2000) AS int)) AS install_date
FROM {fq_schema}.wells w
LATERAL VIEW explode(sequence(1, 2)) e AS k
""".strip(),
        "maintenance_events": f"""
INSERT INTO {fq_schema}.maintenance_events
SELECT
  concat('WO-', substr(e.equipment_id, 4), '-', cast(m AS string)) AS event_id,
  e.equipment_id,
  e.well_id,
  date_add(date_sub(current_date(), {last_day}),
           cast(pmod(hash(e.equipment_id, m), {days}) AS int)) AS event_date,
  element_at(array('inspection','repair','failure','calibration'),
             cast(pmod(hash(e.equipment_id, m), 4) AS int) + 1) AS event_type,
  element_at(array('low','medium','high','critical'), cast(pmod(hash(e.equipment_id, m, 7), 4) AS int) + 1) AS priority,
  CASE WHEN pmod(hash(e.equipment_id, m, 3), 6) = 0 THEN 'open' ELSE 'closed' END AS status,
  round(pmod(hash(e.equipment_id, m, 5), 72) * (0.4 + rand({seed}) * 0.8), 1) AS downtime_hours,
  round(400 + pmod(hash(e.equipment_id, m, 9), 18000), 2) AS cost_usd
FROM {fq_schema}.equipment e
LATERAL VIEW explode(sequence(1, 5)) mm AS m
""".strip(),
    }


def benchmark_questions() -> list[dict[str, object]]:
    """The oil & gas benchmark set, mirroring the harness question template.

    Grounded in the demo data and safe to write to ``questions.json`` (validated by
    :mod:`genie_benchmark.questions`). ``expected_route`` names the domain a supervisor
    should route to; it is advisory metadata, not enforced by the harness.
    """
    return [
        {
            "id": "q001",
            "question": "What was total oil production last month, in barrels?",
            "category": "simple_lookup",
            "expected_behavior": "Sums oil_bbl over production_daily for the previous calendar month.",
            "expected_route": "production",
            "ground_truth_notes": "Cross-check against SUM(oil_bbl) filtered to last month.",
        },
        {
            "id": "q002",
            "question": "Show weekly oil production for the last 8 weeks.",
            "category": "aggregation_time",
            "expected_behavior": "Groups production_daily by week and returns oil_bbl for the last 8 weeks.",
            "expected_route": "production",
        },
        {
            "id": "q003",
            "question": "Break that down by field.",
            "category": "follow_up",
            "expected_behavior": "Reuses conversation state from q002 and splits weekly production by field_name.",
            "expected_route": "production",
            "follow_up_to": "q002",
        },
        {
            "id": "q004",
            "question": "How many maintenance work orders were opened last week?",
            "category": "routing",
            "expected_behavior": (
                "Routes to the maintenance domain and counts maintenance_events opened in the last week."
            ),
            "expected_route": "maintenance",
        },
        {
            "id": "q005",
            "question": "What is the capital of France?",
            "category": "out_of_scope",
            "expected_behavior": "Declines or clarifies without querying data or fanning out to tools.",
        },
        {
            "id": "q006",
            "question": "How are we doing this quarter?",
            "category": "ambiguous",
            "expected_behavior": "Asks a clarifying question (which metric?) rather than forcing an execution.",
        },
        {
            "id": "q007",
            "question": (
                "Compare oil production and equipment downtime over the last quarter and "
                "summarize the relationship."
            ),
            "category": "multi_step_synthesis",
            "expected_behavior": (
                "Agent-mode synthesis across production_daily and maintenance_events; use only "
                "if Agent Mode is in scope."
            ),
        },
        {
            "id": "q008",
            "question": "What was total oil production last month, in barrels?",
            "category": "repeat",
            "expected_behavior": "Repeat of q001 to estimate run-to-run variance; run once, not as a load test.",
            "expected_route": "production",
        },
    ]
