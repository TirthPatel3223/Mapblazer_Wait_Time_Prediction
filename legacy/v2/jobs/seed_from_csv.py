"""Warm-start the lakehouse from a local CSV extract.

This exists so the deployment can be proven end to end before any live ingestion is
wired up. It writes Parquet into the same landing volume, in the same schema and with
the same partitioning the EC2 collector produces, so `bronze_load` cannot tell the
difference between a seeded batch and a live one.

That decoupling matters: proving the pipeline works and getting access to someone else's
production VM are separate problems, and blocking the first on the second is how a
two-day build becomes a two-week one.

    python jobs/seed_from_csv.py --dry-run          # inspect and validate, upload nothing
    python jobs/seed_from_csv.py                    # upload the default extract
    python jobs/seed_from_csv.py --csv other.csv --batch-size 100000

Idempotent: bronze_load de-duplicates on `wait_time_id`, so re-running is harmless.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("seed")

DEFAULT_CSV = ROOT / "data" / "wait_times_join_attractions_themeparks_table_data-1778551975724.csv"

# The exact frame the live collector uploads. Column names and order must match, or
# bronze ends up with two incompatible schemas and the Delta append fails.
TARGET_SCHEMA = ["wait_time_id", "wait_time", "ts_utc", "at_id", "ride_name", "tp_id", "park_name"]

# Source extract -> collector schema.
COLUMN_MAP = {
    "wait_time_upd_dt": "ts_utc",
    "at_name": "ride_name",
    "tp_name": "park_name",
}


def load_csv(path: Path) -> pd.DataFrame:
    """Read the extract and reshape it into the collector's output schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Pass --csv, or check that the extract is still in data/."
        )

    df = pd.read_csv(path)
    log.info("read %s: %d rows, columns %s", path.name, len(df), list(df.columns))

    df = df.rename(columns=COLUMN_MAP)

    # The export carries a duplicated join key ("at_id-2") that nothing downstream uses.
    df = df.drop(columns=[c for c in df.columns if c not in TARGET_SCHEMA], errors="ignore")

    missing = [c for c in TARGET_SCHEMA if c not in df.columns]
    if missing:
        raise KeyError(
            f"extract is missing required columns after mapping: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    df["wait_time"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["wait_time_id"] = pd.to_numeric(df["wait_time_id"], errors="coerce").astype("int64")

    before = len(df)
    df = df.dropna(subset=["wait_time_id", "ts_utc", "wait_time"])
    if len(df) < before:
        log.warning("dropped %d rows with unparseable id, timestamp or wait time", before - len(df))

    # Sort by id so batches carry contiguous ranges, exactly as the watermarked
    # collector would emit them.
    return df[TARGET_SCHEMA].sort_values("wait_time_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    log.info("--- seed summary ---")
    log.info("rows              %s", f"{len(df):,}")
    log.info("wait_time_id      %s .. %s", f"{df.wait_time_id.min():,}", f"{df.wait_time_id.max():,}")
    log.info("timestamps (UTC)  %s .. %s", df.ts_utc.min(), df.ts_utc.max())
    log.info("parks             %d", df.park_name.nunique())
    log.info("attractions       %d", df.ride_name.nunique())
    log.info("mean wait         %.2f min", df.wait_time.mean())
    for park, count in df.park_name.value_counts().items():
        log.info("    %-36s %s rows", park, f"{count:,}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--dry-run", action="store_true", help="validate and summarise only")
    parser.add_argument("--out-dir", type=Path, help="write Parquet locally instead of uploading")
    args = parser.parse_args()

    df = load_csv(args.csv)
    summarize(df)

    batches = [df.iloc[i : i + args.batch_size] for i in range(0, len(df), args.batch_size)]
    log.info("will produce %d Parquet batch(es) of up to %s rows", len(batches), f"{args.batch_size:,}")

    if args.dry_run:
        log.info("dry run: nothing written")
        return 0

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batches):
            path = args.out_dir / f"seed-{i:03d}.parquet"
            batch.to_parquet(path, index=False, compression="snappy")
            log.info("wrote %s (%d rows)", path, len(batch))
        return 0

    from themepark.sources import Databricks

    lake = Databricks()

    # Refuse to seed on top of live data: a seed is a cold-start operation, and running
    # it against a populated bronze would look like it worked while quietly doing nothing.
    watermark = lake.watermark()
    if watermark > 0:
        log.warning(
            "bronze already holds data up to wait_time_id %s. Seeding is meant for a cold "
            "start; bronze_load will de-duplicate, so this is safe but probably redundant.",
            watermark,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for i, batch in enumerate(batches):
        low = int(batch.wait_time_id.min())
        lake.upload_parquet(batch, f"seed-{stamp}-{i:03d}-from-{low:012d}.parquet")
        log.info("uploaded batch %d/%d (%d rows)", i + 1, len(batches), len(batch))

    log.info("seeded %s rows into %s", f"{len(df):,}", lake.settings.landing_path)
    log.info("next: run the weekly job so bronze_load picks these up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
