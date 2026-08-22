"""Landing volume -> bronze Delta.

Bronze is append-only and never edited: it is the audit trail. Every correction happens
downstream in silver, so if a filter or a timezone rule turns out to be wrong, the fix is
a silver rebuild rather than a re-ingestion from someone else's production database.

Deduplication is an anti-join on `wait_time_id` rather than a MERGE, because the collector
deliberately re-reads overlapping ranges when a run is interrupted and we would rather
absorb duplicates cheaply than coordinate exactly-once delivery across two clouds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.sql.functions import current_timestamp  # noqa: E402

from _databricks import ensure_namespaces, log, spark_session, table_exists  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from themepark.config import DatabricksSettings  # noqa: E402


def main() -> int:
    cfg = DatabricksSettings()
    spark = spark_session()

    ensure_namespaces(
        spark,
        cfg.catalog,
        [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema],
        cfg.landing_volume,
    )

    try:
        incoming = spark.read.parquet(f"{cfg.landing_path}/*/*.parquet")
    except Exception as exc:
        log.warning("no parquet files in %s yet (%s)", cfg.landing_path, exc)
        return 0

    incoming = incoming.withColumn("_ingested_at", current_timestamp())
    incoming = incoming.dropDuplicates(["wait_time_id"])
    log.info("landing volume holds %d distinct rows", incoming.count())

    if table_exists(spark, cfg.bronze_table):
        existing = spark.table(cfg.bronze_table).select("wait_time_id")
        new_rows = incoming.join(existing, on="wait_time_id", how="left_anti")
        count = new_rows.count()
        if count:
            new_rows.write.format("delta").mode("append").saveAsTable(cfg.bronze_table)
        log.info("appended %d new rows to %s", count, cfg.bronze_table)
    else:
        incoming.write.format("delta").mode("overwrite").saveAsTable(cfg.bronze_table)
        log.info("created %s with %d rows", cfg.bronze_table, incoming.count())

    total = spark.table(cfg.bronze_table).count()
    high = spark.sql(f"SELECT MAX(wait_time_id) FROM {cfg.bronze_table}").collect()[0][0]
    log.info("bronze now holds %d rows, watermark %s", total, high)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
