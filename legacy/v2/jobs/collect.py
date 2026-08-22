"""Ingestion: read new rows from the source Postgres and push them to Databricks.

Runs on the VM that hosts the database, on a 30-minute systemd timer -- see deploy/ec2/.
Running it there means the database connection is local, so Postgres never has to accept
a connection from the internet and no firewall rule has to change. The only network
access needed is outbound HTTPS, which the host already has.

Idempotent by construction. Each run resumes from the highest `wait_time_id` already
uploaded, so a failed, killed or duplicated run costs nothing, and `--drain` is just the
same batch loop repeated until there is nothing left. `bronze_load` de-duplicates on the
same key, so overlapping batches are absorbed rather than doubled.

    python jobs/collect.py                 # one batch (the scheduled path)
    python jobs/collect.py --drain         # repeat until caught up (backfill)
    python jobs/collect.py --dry-run       # report the gap, write nothing
    python jobs/collect.py --resync        # re-read the watermark from Databricks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from themepark.sources import Databricks, MapblazerPostgres, Supabase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("collect")

JOB_NAME = "collect"


STATE_FILE = Path(__file__).resolve().parents[1] / ".collector-state.json"

# How often to reconcile the local watermark against Databricks rather than trusting it.
RESYNC_AFTER = timedelta(hours=12)


def read_watermark(lake: Databricks, state_path: Path, force_resync: bool = False) -> tuple[int, str]:
    """Resume point for the next batch, preferring local state over a warehouse query.

    Querying Databricks for `max(wait_time_id)` every 30 minutes means 48 wake-ups a day
    of a serverless SQL warehouse that scales to zero. On the free tier, where exceeding
    the compute quota shuts the whole workspace down for the rest of the day, that costs
    considerably more than the weekly training run it is meant to support.

    So the collector records what it last uploaded and trusts that, reconciling with
    Databricks on a cold start, every 12 hours, or on demand. Trusting local state is
    also strictly more correct for this purpose: it tracks what reached the landing
    volume, whereas bronze only reflects what the weekly job has since loaded.
    """
    if not force_resync and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            checked = datetime.fromisoformat(state["last_resync"])
            if datetime.now(timezone.utc) - checked < RESYNC_AFTER:
                return int(state["watermark"]), "local state"
            log.info("local watermark is older than %s; reconciling with Databricks", RESYNC_AFTER)
        except (OSError, KeyError, ValueError) as exc:
            log.warning("could not read %s (%s); falling back to Databricks", state_path, exc)

    remote = lake.watermark()
    write_watermark(state_path, remote)
    return remote, "Databricks"


def write_watermark(state_path: Path, watermark: int, resynced: bool = True) -> None:
    try:
        previous = {}
        if state_path.exists():
            try:
                previous = json.loads(state_path.read_text())
            except (OSError, ValueError):
                # Recovering from a corrupt state file is the whole point of this path;
                # failing to parse it here must not take down the run that is repairing it.
                previous = {}
        state_path.write_text(
            json.dumps(
                {
                    "watermark": int(watermark),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "last_resync": (
                        datetime.now(timezone.utc).isoformat()
                        if resynced
                        else previous.get("last_resync", datetime.now(timezone.utc).isoformat())
                    ),
                },
                indent=2,
            )
        )
    except OSError as exc:
        # Losing the cache costs one extra warehouse query, never correctness.
        log.warning("could not persist watermark to %s: %s", state_path, exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--drain", action="store_true", help="loop until caught up")
    parser.add_argument("--max-batches", type=int, default=40, help="safety stop for --drain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-heartbeat", action="store_true")
    parser.add_argument(
        "--resync",
        action="store_true",
        help="ignore local state and re-read the watermark from Databricks",
    )
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    args = parser.parse_args()

    source = MapblazerPostgres()
    lake = Databricks()

    watermark, origin = read_watermark(lake, args.state_file, args.resync)
    upstream_max = source.max_id()
    gap = max(0, upstream_max - watermark)
    log.info(
        "watermark %s (from %s) | upstream max %s | %s rows behind",
        watermark,
        origin,
        upstream_max,
        f"{gap:,}",
    )

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
            # Recorded only after the upload returns, so a crash mid-batch resumes from
            # the last batch that actually landed rather than skipping it.
            write_watermark(args.state_file, watermark, resynced=False)
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
