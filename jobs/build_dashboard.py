"""Render the static dashboard and write it where GitHub Pages will serve it.

Runs on a GitHub Actions runner after the publish step. Reads the gold tables over the
Databricks SQL warehouse (or from local artifacts with `--source local`, which is how you
iterate on the layout without touching the cloud), renders one self-contained HTML file,
and leaves it for the Pages deploy action to pick up.

    python jobs/build_dashboard.py --source local --dir artifacts/full_run
    python jobs/build_dashboard.py --out site/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from themepark.config import DatabricksSettings  # noqa: E402
from themepark.dashboard_page import render  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("dashboard")

TABLES = [
    "predictions",
    "model_metrics",
    "prediction_accuracy",
    "accuracy_by_ride",
    "data_drift",
    "promotion_log",
    "pipeline_runs",
]


def from_databricks() -> dict[str, pd.DataFrame]:
    from themepark.sources import Databricks, Supabase  # noqa: F401

    cfg = DatabricksSettings()
    lake = Databricks(cfg)
    data: dict[str, pd.DataFrame] = {}

    mapping = {
        "predictions": cfg.predictions_table,
        "model_metrics": cfg.metrics_table,
        "prediction_accuracy": cfg.accuracy_table,
        "accuracy_by_ride": cfg.table(cfg.gold_schema, "accuracy_by_ride"),
        "data_drift": cfg.table(cfg.gold_schema, "data_drift"),
        "promotion_log": cfg.promotion_table,
    }
    for name, table in mapping.items():
        try:
            data[name] = lake.read_table(table)
        except Exception as exc:
            log.warning("skipping %s (%s)", name, exc)
            data[name] = pd.DataFrame()

    # Run history lives in Supabase, since that is what the collector can reach.
    try:
        import requests

        supabase = Supabase()
        response = requests.get(
            f"{supabase.settings.rest_url}/pipeline_runs?select=*&order=run_at.desc&limit=200",
            headers={
                "apikey": supabase.settings.service_key,
                "Authorization": f"Bearer {supabase.settings.service_key}",
            },
            timeout=30,
        )
        data["pipeline_runs"] = pd.DataFrame(response.json()) if response.ok else pd.DataFrame()
    except Exception as exc:
        log.warning("could not read pipeline_runs (%s)", exc)
        data["pipeline_runs"] = pd.DataFrame()

    return data


def from_local(directory: Path) -> dict[str, pd.DataFrame]:
    """Reconstruct the gold tables from a `scripts/local_pipeline.py` run."""
    data = dict.fromkeys(TABLES, pd.DataFrame())

    scorecard = directory / "scorecard.csv"
    if scorecard.exists():
        data["model_metrics"] = pd.read_csv(scorecard)

    accuracy = directory / "production_accuracy.csv"
    if accuracy.exists():
        data["prediction_accuracy"] = pd.read_csv(accuracy)

    forecast = directory / "forecast.parquet"
    if forecast.exists():
        data["predictions"] = pd.read_parquet(forecast)

    by_ride = directory / "accuracy_by_ride.csv"
    if by_ride.exists():
        data["accuracy_by_ride"] = pd.read_csv(by_ride)

    drift = directory / "data_drift.csv"
    if drift.exists():
        data["data_drift"] = pd.read_csv(drift)

    decision = directory / "decision.json"
    if decision.exists():
        data["promotion_log"] = pd.DataFrame([json.loads(decision.read_text())])

    runs = directory / "pipeline_runs.csv"
    if runs.exists():
        data["pipeline_runs"] = pd.read_csv(runs)

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["databricks", "local"], default="databricks")
    parser.add_argument("--dir", type=Path, default=Path("artifacts/full_run"))
    parser.add_argument("--out", type=Path, default=Path("site/index.html"))
    args = parser.parse_args()

    data = from_local(args.dir) if args.source == "local" else from_databricks()
    for name, frame in data.items():
        log.info("%-22s %6d rows", name, len(frame))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data), encoding="utf-8")

    # Tell GitHub Pages not to run the output through Jekyll.
    (args.out.parent / ".nojekyll").write_text("")

    log.info("wrote %s (%.1f KB)", args.out, args.out.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
