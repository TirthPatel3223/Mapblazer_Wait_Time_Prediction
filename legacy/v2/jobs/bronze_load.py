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

# Databricks serverless runs a spark_python_task as exec(compile(source, path, "exec")),
# which leaves __file__ undefined while still recording the real path on the code object.
# Both lookups are needed: __file__ on a laptop or a runner, the frame on Databricks.
# Repeated verbatim in each entry point -- see jobs/_databricks.py for why it cannot be
# imported from there.
HERE = Path(globals().get("__file__", sys._getframe().f_code.co_filename)).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from pyspark.sql.functions import current_timestamp  # noqa: E402

from _databricks import ensure_namespaces, log, spark_session, table_exists  # noqa: E402
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
        # recursiveFileLookup disables partition discovery, so the `dt=` directories the
        # collector writes stay directories instead of becoming an inferred `dt` column
        # that would change bronze's schema. It also decouples this from the directory
        # depth, where the previous `*/*.parquet` glob silently matched nothing if the
        # layout ever gained or lost a level.
        incoming = (
            spark.read.option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.parquet")
            .parquet(cfg.landing_path)
        )
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
    # Only raise on failure. A Databricks task runs this inside an IPython kernel, where
    # SystemExit is reported as an error whatever its code -- so `raise SystemExit(0)`
    # marked bronze_load FAILED after it had already written all 842,539 rows, and the
    # retry policy then ran the successful job twice more. Returning normally on success
    # is the difference between a green task and a red one that did the work.
    _status = main()
    if _status:
        raise SystemExit(_status)
