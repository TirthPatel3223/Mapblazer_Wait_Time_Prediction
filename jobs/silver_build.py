"""Bronze -> silver: timezone correction, filtering, and the 30-minute grid.

The transform itself lives in `themepark.silver` so that this task, the CI tests and the
local dry run cannot drift apart. This file is only the Spark plumbing plus the data-
quality assertions that decide whether the week's training run is allowed to proceed.

Silver is rebuilt in full every week rather than appended to. It is cheap at this scale,
and it means a fix to a filter or a park's operating hours takes effect everywhere on the
next run instead of leaving a seam in the history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _databricks import log, read_delta, spark_session, write_delta  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from themepark.config import DatabricksSettings  # noqa: E402
from themepark.silver import build_silver, silver_quality_report  # noqa: E402

# Tripwires. These are not style checks -- each one corresponds to a way the upstream feed
# has actually been observed to fail.
MIN_ROWS = 10_000
MIN_ENTITIES = 50
MAX_PCT_ZERO = 75.0  # a collector returning closed-park readings pushes this up fast


def main() -> int:
    cfg = DatabricksSettings()
    spark = spark_session()

    bronze = read_delta(spark, cfg.bronze_table)
    log.info("read %d bronze rows", len(bronze))

    silver = build_silver(bronze)
    report = silver_quality_report(silver)
    log.info("silver quality: %s", json.dumps(report, indent=2))

    failures = []
    if report["rows"] < MIN_ROWS:
        failures.append(f"only {report['rows']} rows (expected >= {MIN_ROWS})")
    if report["entities"] < MIN_ENTITIES:
        failures.append(f"only {report['entities']} attractions (expected >= {MIN_ENTITIES})")
    if report["pct_zero"] > MAX_PCT_ZERO:
        failures.append(
            f"{report['pct_zero']:.1f}% of observations are zero (ceiling {MAX_PCT_ZERO}%) -- "
            "this is what a timezone or operating-hours regression looks like"
        )
    if failures:
        raise ValueError("silver quality gate failed: " + "; ".join(failures))

    write_delta(spark, silver, cfg.silver_table, mode="overwrite")

    # Handy for the dashboard and for eyeballing a run without opening a notebook.
    spark.sql(
        f"COMMENT ON TABLE {cfg.silver_table} IS "
        f"'Rebuilt weekly. {report['rows']} rows, {report['entities']} attractions, "
        f"{report['start']} to {report['end']} park-local.'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
