#!/usr/bin/env python3
"""Ingestion for the theme-park pipeline: upstream Postgres -> Databricks bronze.

Runs on the EC2 instance that hosts the source database, on a systemd timer. See
deploy/ec2/README.md. Nothing else from this repository is installed there: this file,
its two systemd units and a small virtualenv are the entire footprint. It deliberately
imports nothing from the training side, so the box never needs Prophet, XGBoost or
cmdstan.

Why it runs there and pushes, rather than being pulled: the alternative is exposing
Postgres to the internet so a runner can dial in. Running here keeps the database
connection on the loopback interface, changes no firewall rule, and needs only outbound
HTTPS -- which the box already has.

One run is one pass:

    1. watermark  MAX(wait_time_id) already in themepark.bronze.wait_times_raw
    2. read       SELECT ... FROM postgres WHERE wait_time_id > watermark LIMIT n
    3. land       write the batch as Parquet into the bronze landing volume
    4. merge      MERGE the landed file into the bronze table, keyed on wait_time_id

Idempotent by construction, with no local state to corrupt. The watermark comes from the
bronze table itself, so it always describes what actually landed; a run killed at any
point re-reads the same range next time and the MERGE absorbs it. An interrupted run
leaves its Parquet file in the landing volume unloaded and harmless -- landing is raw
transport, bronze is the audit trail.

    python ingest.py                 one batch (what the timer runs)
    python ingest.py --drain         repeat until caught up (backfill)
    python ingest.py --dry-run       report the gap, write nothing
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ingest")
# The SQL connector logs an INFO line per HTTP round trip, which buries everything else
# in the journal.
logging.getLogger("databricks.sql").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent

CATALOG = "themepark"
BRONZE_SCHEMA = "bronze"
BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.wait_times_raw"
LANDING_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/landing"

BATCH_SIZE = 200_000
MAX_BATCHES = 40  # safety stop for --drain
UPLOAD_ATTEMPTS = 3

PG_ENV = ("PG_HOST", "PG_PORT", "PG_DATABASE", "PG_USER", "PG_PASSWORD")
DATABRICKS_ENV = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID")

# The upstream join. Table names are overridable from the environment because this schema
# belongs to someone else and may not match what the original extract implied.
SOURCE_QUERY = """
SELECT  w.wait_time_id,
        w.wait_time,
        w.wait_time_upd_dt AS ts_utc,
        a.at_id,
        a.at_name          AS ride_name,
        t.tp_id,
        t.tp_name          AS park_name
FROM        {wait_times}  w
INNER JOIN  {attractions} a ON a.at_id = w.at_id
INNER JOIN  {themeparks}  t ON t.tp_id = a.tp_id
WHERE   w.wait_time_id > %(watermark)s
ORDER BY w.wait_time_id
LIMIT   %(batch_size)s
"""

# Column order is the bronze table's column order; the MERGE below relies on the names.
BRONZE_COLUMNS = ["wait_time_id", "wait_time", "ts_utc", "at_id", "ride_name", "tp_id", "park_name"]

# Kept identical to the live table so a fresh workspace bootstraps itself.
CREATE_BRONZE = f"""
CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
    wait_time_id BIGINT,
    wait_time    BIGINT,
    ts_utc       TIMESTAMP_NTZ,
    at_id        BIGINT,
    ride_name    STRING,
    tp_id        BIGINT,
    park_name    STRING,
    _ingested_at TIMESTAMP
) USING DELTA
"""

# WHEN NOT MATCHED only: bronze is append-only and never edited. A row that is already
# there is already correct, and every correction happens downstream in silver.
# The extra ON predicate is a pruning hint -- the batch's own ids bound the search, so
# Delta skips every file below them instead of scanning the whole table.
MERGE_BATCH = """
MERGE INTO {table} AS t
USING (
    SELECT CAST(wait_time_id AS BIGINT)  AS wait_time_id,
           CAST(wait_time    AS BIGINT)  AS wait_time,
           CAST(ts_utc AS TIMESTAMP_NTZ) AS ts_utc,
           CAST(at_id AS BIGINT)         AS at_id,
           CAST(ride_name AS STRING)     AS ride_name,
           CAST(tp_id AS BIGINT)         AS tp_id,
           CAST(park_name AS STRING)     AS park_name,
           current_timestamp()           AS _ingested_at
    FROM read_files('{path}', format => 'parquet')
) AS s
ON t.wait_time_id = s.wait_time_id AND t.wait_time_id >= {low}
WHEN NOT MATCHED THEN INSERT *
"""


# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines without overriding real environment variables.

    systemd already loads this file via EnvironmentFile=, so this only matters when the
    script is run by hand. Values are secrets: never log them.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


def require_env(*names: str) -> dict[str, str]:
    values, missing = {}, []
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing required environment variables: {', '.join(missing)}")
    return values


def series_to_us(s: pd.Series) -> pd.Series:
    """Downcast a datetime series to microseconds across pandas versions.

    pandas holds datetimes as datetime64[ns] and writes them as Parquet TIMESTAMP(NANOS),
    which Spark refuses to read at all. Microseconds are Delta's native resolution and the
    feed is minute-granularity, so nothing is lost. .dt.as_unit exists only from pandas
    2.2; astype covers the older releases.
    """
    if hasattr(s.dt, "as_unit"):
        return s.dt.as_unit("us")
    try:
        return s.astype("datetime64[us]")
    except (TypeError, ValueError):
        return s


# ------------------------------------------------------------------------------------
# Upstream Postgres (read-only)
# ------------------------------------------------------------------------------------


class SourceDatabase:
    """Read-only reader for the upstream wait-time database.

    Reads only. This database belongs to someone else: the collector never writes, never
    locks, never runs DDL. Ask for a read-only role; it needs nothing more.
    """

    def __init__(self) -> None:
        env = require_env(*PG_ENV)
        self.dsn = (
            f"host={env['PG_HOST']} port={env['PG_PORT']} dbname={env['PG_DATABASE']} "
            f"user={env['PG_USER']} password={env['PG_PASSWORD']} "
            f"sslmode={os.environ.get('PG_SSLMODE', 'prefer')}"
        )
        self.tables = {
            "wait_times": os.environ.get("PG_WAIT_TIMES_TABLE", "wait_times"),
            "attractions": os.environ.get("PG_ATTRACTIONS_TABLE", "attractions"),
            "themeparks": os.environ.get("PG_THEMEPARKS_TABLE", "themeparks"),
        }

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, connect_timeout=30)

    def max_id(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COALESCE(MAX(wait_time_id), 0) FROM {self.tables['wait_times']}")
            return int(cur.fetchone()[0])

    def read_since(self, watermark: int, batch_size: int) -> pd.DataFrame:
        """Rows above the watermark, oldest first, capped so a cold-start backfill of
        800k+ rows drains over repeated batches instead of one enormous transfer."""
        sql = SOURCE_QUERY.format(**self.tables)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {"watermark": watermark, "batch_size": batch_size})
            rows = cur.fetchall()
            columns = [d.name for d in cur.description]

        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return pd.DataFrame(columns=BRONZE_COLUMNS)

        df = df[BRONZE_COLUMNS]
        df["ts_utc"] = series_to_us(pd.to_datetime(df["ts_utc"]))
        df["wait_time"] = pd.to_numeric(df["wait_time"], errors="coerce").astype("Int64")
        return df


# ------------------------------------------------------------------------------------
# Databricks: files in over the Files API, SQL over the warehouse
# ------------------------------------------------------------------------------------


class Lakehouse:
    """The Databricks side: one warehouse session per run, plus volume uploads.

    The warehouse is serverless and scales to zero, so every run wakes it. One session is
    opened for the whole run and every statement goes through it -- waking it once per run
    rather than once per statement is the difference that matters on a metered workspace.
    """

    def __init__(self) -> None:
        env = require_env(*DATABRICKS_ENV)
        self.host = env["DATABRICKS_HOST"].removeprefix("https://").removeprefix("http://").rstrip("/")
        self.token = env["DATABRICKS_TOKEN"]
        # Accept either the bare warehouse id or a pasted path like sql/warehouses/<id>.
        self.warehouse_id = env["DATABRICKS_WAREHOUSE_ID"].strip("/").split("/")[-1]
        self._conn = None

    def __enter__(self) -> Lakehouse:
        from databricks import sql as dbsql

        self._conn = dbsql.connect(
            server_hostname=self.host,
            http_path=f"/sql/1.0/warehouses/{self.warehouse_id}",
            access_token=self.token,
        )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, statement: str) -> list:
        with self._conn.cursor() as cur:
            cur.execute(statement)
            return cur.fetchall() if cur.description else []

    def ensure_objects(self) -> None:
        """Create the schema, landing volume and bronze table if they are missing.

        Three idempotent metadata statements on an already-open session. Running them
        every time costs a fraction of a second and means a fresh workspace needs no
        manual setup step that someone has to remember six months from now.
        """
        self.execute(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")
        self.execute(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}.landing")
        self.execute(CREATE_BRONZE)

    def watermark(self) -> int:
        """Highest wait_time_id already in bronze.

        Derived from the data rather than stored in a cursor file, so there is nothing
        that can fall out of sync with what actually landed.
        """
        rows = self.execute(f"SELECT COALESCE(MAX(wait_time_id), 0) FROM {BRONZE_TABLE}")
        return int(rows[0][0])

    def upload_parquet(self, df: pd.DataFrame, filename: str) -> str:
        """Land a batch in the ingestion volume, partitioned by UTC date.

        The Files API is a plain authenticated PUT, which keeps the databricks-sdk off
        this box -- one less dependency on a machine that is not ours.
        """
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = f"{LANDING_VOLUME}/dt={day}/{filename}"

        buffer = io.BytesIO()
        # coerce_timestamps is the backstop for a datetime column series_to_us cannot see,
        # such as one nested inside an object column.
        df.to_parquet(
            buffer,
            index=False,
            compression="snappy",
            coerce_timestamps="us",
            allow_truncated_timestamps=True,
        )
        payload = buffer.getvalue()

        url = f"https://{self.host}/api/2.0/fs/files{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        last: Exception | None = None
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                r = requests.put(
                    url, headers=headers, params={"overwrite": "true"}, data=payload, timeout=300
                )
                if r.status_code < 300:
                    log.info("landed %d rows (%d KB) at %s", len(df), len(payload) // 1024, path)
                    return path
                last = RuntimeError(f"HTTP {r.status_code} {r.text[:300]}")
            except requests.RequestException as exc:
                last = exc
            log.warning("upload attempt %d/%d failed: %s", attempt, UPLOAD_ATTEMPTS, last)
            if attempt < UPLOAD_ATTEMPTS:
                time.sleep(5 * attempt)
        raise RuntimeError(f"could not upload {path}: {last}")

    def merge_file(self, path: str, low: int, table: str = BRONZE_TABLE) -> None:
        self.execute(MERGE_BATCH.format(table=table, path=path, low=low))


# ------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    # Both ends up front, so a half-filled .env is reported in one pass rather than one
    # missing key per attempt.
    require_env(*PG_ENV, *DATABRICKS_ENV)
    source = SourceDatabase()

    with Lakehouse() as lake:
        lake.ensure_objects()
        watermark = lake.watermark()
        upstream = source.max_id()
        gap = max(0, upstream - watermark)
        log.info("bronze watermark %d | upstream max %d | %s rows behind", watermark, upstream, f"{gap:,}")

        if args.dry_run:
            log.info("dry run: nothing written")
            return 0

        total, batches = 0, 0
        while batches < args.max_batches:
            batch = source.read_since(watermark, args.batch_size)
            if batch.empty:
                log.info("caught up at wait_time_id %d", watermark)
                break

            low, high = int(batch["wait_time_id"].min()), int(batch["wait_time_id"].max())
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = lake.upload_parquet(batch, f"part-{low:012d}-{stamp}.parquet")
            lake.merge_file(path, low)

            watermark = high
            total += len(batch)
            batches += 1
            log.info("batch %d merged: %d rows, wait_time_id %d..%d", batches, len(batch), low, high)

            if not args.drain:
                break
            time.sleep(1)  # be a polite client of someone else's production database
        else:
            log.warning("stopped at --max-batches=%d; rerun to continue draining", args.max_batches)

        rows = lake.execute(f"SELECT COUNT(*), MAX(wait_time_id) FROM {BRONZE_TABLE}")
        log.info("ingested %s rows in %d batch(es); bronze now holds %s rows, watermark %s",
                 f"{total:,}", batches, f"{rows[0][0]:,}", rows[0][1])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--drain", action="store_true", help="loop until caught up")
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES, help="safety stop for --drain")
    parser.add_argument("--dry-run", action="store_true", help="report the gap, write nothing")
    parser.add_argument("--env-file", type=Path, default=HERE.parent.parent / ".env")
    args = parser.parse_args()

    load_env(args.env_file)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception:
        log.exception("ingestion failed; nothing was left half-written in bronze")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
