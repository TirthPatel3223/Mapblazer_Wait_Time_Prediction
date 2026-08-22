-- Databricks bootstrap. Run ONCE in the workspace SQL editor before the first ingestion.
--
-- Why this exists: the collector uploads Parquet into a Unity Catalog volume, but the
-- volume is normally created by the bronze_load task -- which runs *after* the collector.
-- On a brand-new workspace that is a deadlock, so the namespaces are created up front.
-- Everything here is idempotent; re-running it is harmless.

CREATE CATALOG IF NOT EXISTS themepark;

CREATE SCHEMA IF NOT EXISTS themepark.bronze
  COMMENT 'Raw ingested wait times, append-only. Never edited -- this is the audit trail.';

CREATE SCHEMA IF NOT EXISTS themepark.silver
  COMMENT 'Timezone-corrected, filtered, 30-minute gridded observations. Rebuilt weekly.';

CREATE SCHEMA IF NOT EXISTS themepark.gold
  COMMENT 'Published forecasts, model metrics, production accuracy, promotion log.';

-- Landing zone the GitHub Actions collector writes Parquet batches into.
CREATE VOLUME IF NOT EXISTS themepark.bronze.landing
  COMMENT 'Ingestion landing zone, partitioned by UTC date (dt=YYYY-MM-DD).';


-- ---------------------------------------------------------------------------------
-- Verify. Expect: one catalog, three schemas, one volume.
-- ---------------------------------------------------------------------------------
SHOW SCHEMAS IN themepark;
SHOW VOLUMES IN themepark.bronze;


-- ---------------------------------------------------------------------------------
-- If CREATE CATALOG is refused (some workspaces restrict metastore-level DDL), fall
-- back to the default catalog instead of fighting it. Create the schemas under
-- `workspace`, then set DATABRICKS_CATALOG=workspace in GitHub Secrets and
-- `catalog: workspace` in databricks.yml -- every table name in the code is derived
-- from that one setting, so nothing else changes.
-- ---------------------------------------------------------------------------------
-- CREATE SCHEMA IF NOT EXISTS workspace.bronze;
-- CREATE SCHEMA IF NOT EXISTS workspace.silver;
-- CREATE SCHEMA IF NOT EXISTS workspace.gold;
-- CREATE VOLUME IF NOT EXISTS workspace.bronze.landing;
