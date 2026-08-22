"""Run one complete weekly cycle offline, against the historical extract.

This is the dress rehearsal for the scheduled job and the fastest way to check a change
end to end without touching cloud infrastructure. It also produces the honest headline
numbers: the model is trained only on data up to a cut-off, then asked to forecast the
week *after* it, and scored against what actually happened -- the same thing
`themepark.verify` does in production every Sunday.

    python scripts/local_pipeline.py [--fast] [--out DIR]

`--fast` restricts training to the busiest attractions so the loop takes a minute rather
than ten; use it while iterating, never for a number you intend to quote.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from themepark.evaluate import format_scorecard  # noqa: E402
from themepark.score import generate_forecast  # noqa: E402
from themepark.silver import build_silver, silver_quality_report  # noqa: E402
from themepark.train import run_training  # noqa: E402
from themepark.verify import accuracy_by_ride, drift_summary, verify_week  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "wait_times_join_attractions_themeparks_table_data-1778551975724.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
log = logging.getLogger("local_pipeline")


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path, usecols=["wait_time_id", "wait_time_upd_dt", "tp_name", "at_name", "wait_time"]
    )
    df["wait_time_upd_dt"] = pd.to_datetime(df["wait_time_upd_dt"])
    return df.rename(
        columns={"wait_time_upd_dt": "ts_utc", "tp_name": "park_name", "at_name": "ride_name"}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "local_run")
    parser.add_argument("--fast", action="store_true", help="busiest 12 attractions only")
    parser.add_argument("--holdout-days", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("STEP 1  bronze -> silver")
    log.info("=" * 78)
    silver = build_silver(load_raw(args.csv))
    quality = silver_quality_report(silver)
    log.info("silver quality: %s", json.dumps(quality, indent=2))

    if args.fast:
        busiest = (
            silver.groupby(["park_name", "ride_key"])["wait_time"].mean().nlargest(12).index
        )
        silver = silver.set_index(["park_name", "ride_key"]).loc[busiest].reset_index()
        log.info("--fast: restricted to %d attractions, %d rows", len(busiest), len(silver))

    # Everything at or before `as_of` is what the pipeline would have had last Sunday.
    # The week after it is unseen future used only for production-accuracy verification.
    as_of = silver["ts_local"].max() - pd.Timedelta(days=args.holdout_days)
    history, future = silver[silver.ts_local <= as_of], silver[silver.ts_local > as_of]
    log.info("simulating a run as of %s (%d future rows withheld)", as_of, len(future))

    log.info("=" * 78)
    log.info("STEP 2  train candidates and decide a champion")
    log.info("=" * 78)
    run = run_training(history)
    print("\n" + run.scorecard() + "\n")
    log.info("decision: %s", run.decision)

    log.info("=" * 78)
    log.info("STEP 3  score the forecast grid")
    log.info("=" * 78)
    forecast = generate_forecast(
        run.champion, history, model_version="local", start=as_of.normalize() + pd.Timedelta(days=1)
    )
    log.info("forecast rows: %d", len(forecast))

    log.info("=" * 78)
    log.info("STEP 4  verify the forecast against what actually happened")
    log.info("=" * 78)
    verification = verify_week(forecast, future)

    summary = verification["summary"]
    if summary.empty:
        log.warning("no overlap between forecast and withheld actuals; nothing to verify")
    else:
        print("\nPRODUCTION ACCURACY (forecast made before these observations existed)")
        print(
            summary[
                ["segment", "n", "mae", "rmse", "within_10min_pct", "bias", "interval_coverage_pct"]
            ].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )

        backtest_mae = run.decision.details["winner_mae"]
        production_mae = float(summary[summary.segment == "overall"]["mae"].iloc[0])
        print(
            f"\nbacktest MAE {backtest_mae:.2f}  ->  production MAE {production_mae:.2f} "
            f"({production_mae / backtest_mae:.2f}x)"
        )
        print("The gap between those two is the number worth publishing.\n")

        worst = accuracy_by_ride(verification["joined"], top_n=10)
        print("WORST-PREDICTED ATTRACTIONS")
        print(
            worst[["park_name", "ride_name", "n", "mae", "bias", "mean_actual"]].to_string(
                index=False, float_format=lambda v: f"{v:.2f}"
            )
        )

    pd.DataFrame(run.results).to_csv(args.out / "scorecard.csv", index=False)
    forecast.to_parquet(args.out / "forecast.parquet", index=False)
    if not summary.empty:
        summary.to_csv(args.out / "production_accuracy.csv", index=False)
        accuracy_by_ride(verification["joined"], top_n=50).to_csv(
            args.out / "accuracy_by_ride.csv", index=False
        )
    drift_summary(silver).to_csv(args.out / "data_drift.csv", index=False)
    # A synthetic run record so the freshness banner renders in local previews.
    pd.DataFrame(
        [
            {
                "job": job,
                "run_at": pd.Timestamp.now(tz="UTC"),
                "status": "success",
                "detail": "local dry run",
                "rows_affected": n,
            }
            for job, n in [("collect", len(silver)), ("train", len(forecast)), ("publish", len(forecast))]
        ]
    ).to_csv(args.out / "pipeline_runs.csv", index=False)
    (args.out / "silver_quality.json").write_text(json.dumps(quality, indent=2))
    (args.out / "scorecard.txt").write_text(format_scorecard(run.results))
    (args.out / "decision.json").write_text(
        json.dumps({k: str(v) for k, v in run.decision.as_row().items()}, indent=2)
    )
    log.info("artifacts written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
