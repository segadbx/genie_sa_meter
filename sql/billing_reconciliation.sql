-- Billing reconciliation for a benchmark session.
--
-- Billing reconciliation is a SEPARATE step from trace collection. Trace/token telemetry
-- is used for diagnosis; billed cost comes only from system.billing.usage. Do not treat
-- token counts as billed cost.
--
-- Replace the placeholders before running. Billing data can be delayed; wait for it to
-- settle before drawing conclusions. Reconcile by workspace, identity, endpoint, SKU, and
-- time window.
--
--   :workspace_id     -- numeric workspace id the benchmark ran in
--   :start_time       -- session start (UTC), e.g. '2026-09-14T00:00:00Z'
--   :end_time         -- session end   (UTC)
--   :identity         -- the test identity's principal (email or service principal id)

SELECT
    usage_date,
    sku_name,
    usage_metadata.endpoint_name        AS endpoint_name,
    identity_metadata.run_as            AS run_as,
    usage_unit,
    SUM(usage_quantity)                 AS total_usage_quantity,
    COUNT(*)                            AS usage_records
FROM system.billing.usage
WHERE workspace_id = :workspace_id
  AND usage_start_time >= :start_time
  AND usage_end_time   <= :end_time
  -- Narrow to the test identity when available:
  -- AND identity_metadata.run_as = :identity
GROUP BY
    usage_date,
    sku_name,
    usage_metadata.endpoint_name,
    identity_metadata.run_as,
    usage_unit
ORDER BY
    usage_date,
    sku_name,
    endpoint_name;

-- To attach list prices, join system.billing.list_prices on sku_name and the matching
-- price window, using the customer-approved price sheet. Only then present an estimated
-- cost; otherwise report usage quantities and mark cost as billing_pending.
