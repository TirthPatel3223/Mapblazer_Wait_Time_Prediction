"""Publish gold tables from Databricks to Supabase, where they become a REST API.

Runs on a GitHub Actions runner and pulls: Databricks Free Edition cannot open outbound
connections, so it can never push to Supabase itself.

Supabase is doing something specific here that is easy to undersell. PostgREST turns any
table into a filtered, paginated, authenticated REST endpoint with no API code at all --
no FastAPI service to write, containerise, deploy, monitor or keep patched. For a
read-mostly forecast table consumed by one downstream optimiser, hand-rolling a service
would be strictly worse.

    python jobs/publish.py [--weeks 8] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from themepark.config import DatabricksSettings  # noqa: E402
from themepark.sources import Databricks, Supabase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("publish")

JOB_NAME = "publish"

# Supabase free tier allows 500 MB. A week of forecasts is ~25k rows, so eight weeks of
# history sits comfortably inside it while Delta keeps the full record upstream.
RETENTION_WEEKS = 8


def _fetch(lake: Databricks, table: str, where: str = "") -> pd.DataFrame:
    try:
        return lake.read_table(table, where)
    except Exception as exc:
        log.warning("could not read %s (%s); skipping", table, exc)
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=RETENTION_WEEKS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = DatabricksSettings()
    lake = Databricks(cfg)
    supabase = Supabase()

    cutoff = (datetime.now(timezone.utc) - timedelta(weeks=args.weeks)).strftime("%Y-%m-%d")

    payloads = {
        "predictions": _fetch(lake, cfg.predictions_table, f"ts_local >= '{cutoff}'"),
        "model_metrics": _fetch(lake, cfg.metrics_table),
        "prediction_accuracy": _fetch(lake, cfg.accuracy_table),
        "accuracy_by_ride": _fetch(lake, cfg.table(cfg.gold_schema, "accuracy_by_ride")),
        "data_drift": _fetch(lake, cfg.table(cfg.gold_schema, "data_drift")),
        "promotion_log": _fetch(lake, cfg.promotion_table),
    }

    for table, frame in payloads.items():
        log.info("%-22s %6d rows", table, len(frame))

    if args.dry_run:
        log.info("dry run: nothing written to Supabase")
        return 0

    total = 0
    try:
        for table, frame in payloads.items():
            if frame.empty:
                continue
            total += supabase.upsert(table, frame)

        # Trim, so the free tier is never the thing that breaks the deployment.
        supabase.delete_older_than("predictions", "ts_local", cutoff)

    except Exception as exc:
        log.exception("publish failed")
        try:
            supabase.heartbeat(JOB_NAME, "failed", str(exc), total)
        except Exception:
            pass
        return 1

    supabase.heartbeat(JOB_NAME, "success", f"{len(payloads)} tables", total)
    log.info("published %d rows to Supabase", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
