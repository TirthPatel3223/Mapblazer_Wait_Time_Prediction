"""Weekly verification: score last week's published forecast against what actually happened.

This is the only genuinely out-of-sample measurement in the system. The predictions being
graded here were written to `gold.predictions` days before any of these observations
existed, so there is no split to draw wrong and nothing to leak.

It is also the honesty check on the backtest. v1 reported a 3.27-minute holdout MAE while
its one live snapshot came in near 12 -- and the model ranking flipped between the two.
Publishing the backtest number alone would have overstated accuracy roughly threefold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _databricks import log, read_delta, spark_session, table_exists, write_delta  # noqa: E402
from themepark.config import DatabricksSettings  # noqa: E402
from themepark.verify import accuracy_by_ride, drift_summary, verify_week  # noqa: E402


def main() -> int:
    cfg = DatabricksSettings()
    spark = spark_session()

    if not table_exists(spark, cfg.predictions_table):
        log.info("no predictions published yet; nothing to verify on the first run")
        return 0

    silver = read_delta(spark, cfg.silver_table)
    silver["ts_local"] = pd.to_datetime(silver["ts_local"])

    predictions = read_delta(spark, cfg.predictions_table)
    predictions["ts_local"] = pd.to_datetime(predictions["ts_local"])

    # Only grade slots whose actuals have arrived. Anything at or after the silver
    # high-water mark simply has not happened yet.
    observed_through = silver["ts_local"].max()
    due = predictions[predictions["ts_local"] <= observed_through]
    if due.empty:
        log.info("no published slots have come due yet")
        return 0

    # A slot can appear more than once if a week was re-scored; grade the newest.
    due = (
        due.sort_values("generated_at")
        .drop_duplicates(subset=["park_name", "ride_key", "ts_local"], keep="last")
        .reset_index(drop=True)
    )
    log.info("grading %d published slots through %s", len(due), observed_through)

    verification = verify_week(due, silver)
    summary = verification["summary"]
    if summary.empty:
        log.warning("no forecast slot matched an actual observation")
        return 0

    write_delta(spark, summary, cfg.accuracy_table, mode="append")

    by_ride = accuracy_by_ride(verification["joined"], top_n=50)
    if not by_ride.empty:
        by_ride["computed_at"] = pd.Timestamp.utcnow().tz_localize(None)
        write_delta(spark, by_ride, cfg.table(cfg.gold_schema, "accuracy_by_ride"), mode="overwrite")

    drift = drift_summary(silver)
    write_delta(spark, drift, cfg.table(cfg.gold_schema, "data_drift"), mode="overwrite")

    overall = summary[summary.segment == "overall"].iloc[0]
    log.info(
        "PRODUCTION ACCURACY  MAE %.2f | RMSE %.2f | within 10 min %.1f%% | band coverage %.1f%%",
        overall["mae"],
        overall["rmse"],
        overall["within_10min_pct"],
        overall["interval_coverage_pct"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
