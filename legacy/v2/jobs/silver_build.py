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

# Databricks serverless runs a spark_python_task as exec(compile(source, path, "exec")),
# which leaves __file__ undefined while still recording the real path on the code object.
# Both lookups are needed: __file__ on a laptop or a runner, the frame on Databricks.
# Repeated verbatim in each entry point -- see jobs/_databricks.py for why it cannot be
# imported from there.
HERE = Path(globals().get("__file__", sys._getframe().f_code.co_filename)).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from _databricks import log, read_delta, spark_session, write_delta  # noqa: E402
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
    # Only raise on failure. A Databricks task runs this inside an IPython kernel, where
    # SystemExit is reported as an error whatever its code -- so `raise SystemExit(0)`
    # marked bronze_load FAILED after it had already written all 842,539 rows, and the
    # retry policy then ran the successful job twice more. Returning normally on success
    # is the difference between a green task and a red one that did the work.
    _status = main()
    if _status:
        raise SystemExit(_status)
