"""Billing reconciliation helpers.

Billing reconciliation is deliberately a *separate* step from trace collection: trace
data is used for diagnosis, and billed cost comes only from the customer's approved
billing query against ``system.billing.usage``. Nothing here converts token counts into a
dollar figure; that would misrepresent inferred telemetry as billed cost.
"""

from __future__ import annotations

from pathlib import Path

_SQL_FILE = Path(__file__).resolve().parents[2] / "sql" / "billing_reconciliation.sql"


def billing_reconciliation_sql() -> str:
    """Return the packaged billing reconciliation SQL template.

    The customer supplies the concrete workspace id, identity, endpoint, SKU, and time
    window before running it; the harness does not execute billing queries itself.
    """
    if _SQL_FILE.exists():
        return _SQL_FILE.read_text(encoding="utf-8")
    # Fallback when the package is installed without the sql/ directory alongside it.
    return (
        "-- billing_reconciliation.sql not packaged with this install.\n"
        "-- See the project's sql/ directory for the template.\n"
    )
