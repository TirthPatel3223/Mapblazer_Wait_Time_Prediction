"""Shared helpers for tasks that execute inside Databricks.

Keeping the Spark-specific glue here means the modelling code in `themepark` never imports
`pyspark`, so the identical logic runs on a laptop or a GitHub Actions runner when the
Databricks path is unavailable.

Why every entry point repeats a two-line sys.path bootstrap instead of calling one here:
it is a chicken-and-egg. This module is only importable *after* `jobs/` is on sys.path, so
whatever puts it there cannot itself be imported. The idiom those files use is

    HERE = Path(globals().get("__file__", sys._getframe().f_code.co_filename)).resolve().parent

because a Databricks serverless `spark_python_task` is executed as
`exec(compile(source, path, "exec"))`. That leaves `__file__` undefined -- the plain
`Path(__file__)` form raised `NameError` and failed the first deployed run -- while the
real path survives as the code object's filename. Modules imported normally, this one
included, always have `__file__`, so only the four entry points need the fallback.
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
