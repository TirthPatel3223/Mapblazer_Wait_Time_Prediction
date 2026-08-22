"""Data access: the upstream Postgres, the Databricks lakehouse, and Supabase.

Databricks Free Edition restricts outbound internet to a short allowlist of trusted
domains, so a Databricks job cannot dial the Mapblazer database or Supabase. Every network
hop therefore runs *inbound* to Databricks from a GitHub Actions runner. That is not a
workaround to apologise for -- keeping ingestion out of the compute engine is a reasonable
separation regardless -- but it does mean the readers and writers below live on the runner
side, and the Databricks notebooks only ever touch tables that are already there.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import DatabricksSettings, SourceDBSettings, SupabaseSettings

log = logging.getLogger(__name__)

# The upstream join. Table names are overridable because this schema belongs to someone
# else and may not match what the CSV export implied.
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

RAW_COLUMNS = ["wait_time_id", "wait_time", "ts_utc", "at_id", "ride_name", "tp_id", "park_name"]


class MapblazerPostgres:
    """Read-only reader for the upstream wait-time database.

    Reads only. This database belongs to someone else and the pipeline must never write,
    update or lock anything in it.
    """

    def __init__(
        self,
        settings: SourceDBSettings | None = None,
        wait_times_table: str = "wait_times",
        attractions_table: str = "attractions",
        themeparks_table: str = "themeparks",
    ) -> None:
        self.settings = settings or SourceDBSettings()
        self.tables = {
            "wait_times": wait_times_table,
            "attractions": attractions_table,
            "themeparks": themeparks_table,
        }

    def _connect(self):
        import psycopg  # imported lazily: Databricks tasks never need it

        return psycopg.connect(self.settings.dsn, connect_timeout=30)

    def read_since(self, watermark: int = 0, batch_size: int = 200_000) -> pd.DataFrame:
        """Rows with `wait_time_id` above the watermark, oldest first.

        Bounded by `batch_size` so a cold-start backfill of 800k+ rows drains over
        repeated runs instead of timing out a runner. Because the watermark is read from
        what actually landed, an interrupted run simply resumes.
        """
        sql = SOURCE_QUERY.format(**self.tables)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {"watermark": watermark, "batch_size": batch_size})
            rows = cur.fetchall()
            columns = [d.name for d in cur.description]

        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            log.info("no new rows above watermark %s", watermark)
            return pd.DataFrame(columns=RAW_COLUMNS)

        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
        df["wait_time"] = pd.to_numeric(df["wait_time"], errors="coerce")
        log.info(
            "read %d rows, wait_time_id %s..%s",
            len(df),
            df.wait_time_id.min(),
            df.wait_time_id.max(),
        )
        return df

    def max_id(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COALESCE(MAX(wait_time_id), 0) FROM {self.tables['wait_times']}")
            return int(cur.fetchone()[0])


class Databricks:
    """Workspace access from outside: SQL over the warehouse, files into a UC volume."""

    def __init__(self, settings: DatabricksSettings | None = None) -> None:
        self.settings = settings or DatabricksSettings()

    def _workspace(self):
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient(host=self.settings.host, token=self.settings.token)

    def _sql(self):
        from databricks import sql

        return sql.connect(
            server_hostname=self.settings.server_hostname,
            http_path=self.settings.http_path,
            access_token=self.settings.token,
        )

    def query(self, statement: str) -> pd.DataFrame:
        with self._sql() as conn, conn.cursor() as cur:
            cur.execute(statement)
            return cur.fetchall_arrow().to_pandas()

    def execute(self, statement: str) -> None:
        with self._sql() as conn, conn.cursor() as cur:
            cur.execute(statement)

    def scalar(self, statement: str, default=None):
        df = self.query(statement)
        if df.empty or pd.isna(df.iloc[0, 0]):
            return default
        return df.iloc[0, 0]

    def watermark(self) -> int:
        """Highest `wait_time_id` already landed in bronze.

        Derived from the data rather than stored separately, so there is no cursor to fall
        out of sync with reality.
        """
        table = self.settings.bronze_table
        exists = self.scalar(
            f"SELECT count(*) FROM information_schema.tables "
            f"WHERE table_catalog='{self.settings.catalog}' "
            f"AND table_schema='{self.settings.bronze_schema}' AND table_name='wait_times_raw'",
            0,
        )
        if not exists:
            log.info("bronze table %s does not exist yet; starting from zero", table)
            return 0
        return int(self.scalar(f"SELECT COALESCE(MAX(wait_time_id), 0) FROM {table}", 0))

    def upload_parquet(self, df: pd.DataFrame, filename: str) -> str:
        """Land a Parquet batch in the ingestion volume, partitioned by UTC date."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = f"{self.settings.landing_path}/dt={today}/{filename}"

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, compression="snappy")
        buffer.seek(0)

        self._workspace().files.upload(path, buffer, overwrite=True)
        log.info("uploaded %d rows to %s", len(df), path)
        return path

    def read_table(self, table: str, where: str = "") -> pd.DataFrame:
        clause = f" WHERE {where}" if where else ""
        return self.query(f"SELECT * FROM {table}{clause}")


class Supabase:
    """Serving store. PostgREST gives the REST API for free -- no API code to maintain."""

    def __init__(self, settings: SupabaseSettings | None = None) -> None:
        self.settings = settings or SupabaseSettings()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.settings.service_key,
            "Authorization": f"Bearer {self.settings.service_key}",
            "Content-Type": "application/json",
            # Upsert, and do not echo the inserted rows back over the wire.
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }

    def upsert(self, table: str, df: pd.DataFrame, chunk_size: int = 1000) -> int:
        """Upsert a frame in chunks. Idempotent, so a retried publish is harmless."""
        import requests

        if df.empty:
            return 0

        payload = df.copy()
        for col in payload.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
            payload[col] = payload[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        records = payload.where(pd.notna(payload), None).to_dict(orient="records")

        url = f"{self.settings.rest_url}/{table}"
        written = 0
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            response = requests.post(url, headers=self._headers, json=chunk, timeout=60)
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Supabase upsert into {table} failed ({response.status_code}): {response.text[:500]}"
                )
            written += len(chunk)
        log.info("upserted %d rows into %s", written, table)
        return written

    def delete_older_than(self, table: str, column: str, cutoff: str) -> None:
        """Trim history so the 500 MB free tier is never the thing that breaks."""
        import requests

        response = requests.delete(
            f"{self.settings.rest_url}/{table}?{column}=lt.{cutoff}",
            headers=self._headers,
            timeout=60,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"Supabase delete from {table} failed: {response.text[:500]}")

    def heartbeat(self, job: str, status: str, detail: str = "", rows: int = 0) -> None:
        """Record a run and, incidentally, keep the project alive.

        Supabase pauses a free project after 7 days without a database request. The
        collector runs every 30 minutes, so this write alone guarantees the serving API
        never goes to sleep.
        """
        self.upsert(
            "pipeline_runs",
            pd.DataFrame(
                [
                    {
                        "job": job,
                        "run_at": datetime.now(timezone.utc),
                        "status": status,
                        "detail": detail[:1000],
                        "rows_affected": rows,
                    }
                ]
            ),
        )
