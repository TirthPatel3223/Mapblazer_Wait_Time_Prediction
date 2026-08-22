"""Ingestion: pull new rows from the upstream Postgres into the Databricks landing volume.

Runs on a GitHub Actions runner every 30 minutes, because Databricks Free Edition cannot
make outbound connections to the upstream database.

Idempotent by construction. The watermark is the highest `wait_time_id` that actually
landed in bronze, read fresh at the start of every run, so a failed, killed or duplicated
run costs nothing: the next one simply resumes from what is really there. That is also
what makes the cold-start backfill work -- `--drain` just repeats the same batch loop
until there is nothing left.

    python jobs/collect.py                 # one batch (the scheduled path)
    python jobs/collect.py --drain         # repeat until caught up (backfill)
    python jobs/collect.py --dry-run       # report the gap, write nothing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from themepark.sources import Databricks, MapblazerPostgres, Supabase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("collect")

JOB_NAME = "collect"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--drain", action="store_true", help="loop until caught up")
    parser.add_argument("--max-batches", type=int, default=40, help="safety stop for --drain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-heartbeat", action="store_true")
    args = parser.parse_args()

    source = MapblazerPostgres()
    lake = Databricks()

    watermark = lake.watermark()
    upstream_max = source.max_id()
    gap = max(0, upstream_max - watermark)
    log.info("watermark %s | upstream max %s | %s rows behind", watermark, upstream_max, f"{gap:,}")

    if args.dry_run:
        log.info("dry run: nothing written")
        return 0

    total, batches = 0, 0
    try:
        while batches < args.max_batches:
            batch = source.read_since(watermark, args.batch_size)
            if batch.empty:
                log.info("caught up")
                break

            filename = f"part-{watermark:012d}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.parquet"
            lake.upload_parquet(batch, filename)

            watermark = int(batch["wait_time_id"].max())
            total += len(batch)
            batches += 1
            log.info("batch %d: %d rows, watermark now %s", batches, len(batch), watermark)

            if not args.drain:
                break
            time.sleep(1)  # be a polite client of someone else's production database
        else:
            log.warning("hit --max-batches=%d; rerun to continue draining", args.max_batches)

    except Exception as exc:
        log.exception("ingestion failed")
        if not args.no_heartbeat:
            _safe_heartbeat("failed", str(exc), total)
        return 1

    if not args.no_heartbeat:
        # Doubles as the keepalive that stops Supabase pausing the free project.
        _safe_heartbeat("success", f"{batches} batch(es), watermark {watermark}", total)

    log.info("ingested %s rows across %d batch(es)", f"{total:,}", batches)
    return 0


def _safe_heartbeat(status: str, detail: str, rows: int) -> None:
    """A heartbeat failure must never fail the ingestion that already succeeded."""
    try:
        Supabase().heartbeat(JOB_NAME, status, detail, rows)
    except Exception as exc:
        log.warning("heartbeat write failed (ingestion itself was fine): %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
