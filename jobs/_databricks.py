"""Shared helpers for tasks that execute inside Databricks.

Keeping the Spark-specific glue here means the modelling code in `themepark` never imports
`pyspark`, so the identical logic runs on a laptop or a GitHub Actions runner when the
Databricks path is unavailable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("databricks")


def spark_session():
    """The active session inside a Databricks task, or a local one for tests."""
    from pyspark.sql import SparkSession

    session = SparkSession.getActiveSession()
    return session or SparkSession.builder.getOrCreate()


def ensure_namespaces(spark, catalog: str, schemas: list[str], volume: str | None = None) -> None:
    """Create the catalog, schemas and landing volume if they are not there yet.

    Declared in code rather than clicked into the UI so the workspace can be rebuilt from
    scratch -- which matters when the free-tier account is the only environment there is.
    """
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    for schema in schemas:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    if volume:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schemas[0]}.{volume}")


def table_exists(spark, table: str) -> bool:
    try:
        spark.sql(f"DESCRIBE TABLE {table}")
        return True
    except Exception:
        return False


def write_delta(spark, df: pd.DataFrame, table: str, mode: str = "overwrite") -> int:
    """Persist a pandas frame as a Delta table.

    The datasets here are small -- a year of 30-minute observations across ~120
    attractions is a few million rows -- so pandas is the honest tool and Spark is only
    the storage layer. Pretending otherwise would add distributed-systems complexity for
    no benefit.
    """
    if df.empty:
        log.warning("refusing to write an empty frame to %s", table)
        return 0
    sdf = spark.createDataFrame(df)
    writer = sdf.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table)
    log.info("wrote %d rows to %s (%s)", len(df), table, mode)
    return len(df)


def read_delta(spark, table: str, where: str = "") -> pd.DataFrame:
    clause = f" WHERE {where}" if where else ""
    return spark.sql(f"SELECT * FROM {table}{clause}").toPandas()
