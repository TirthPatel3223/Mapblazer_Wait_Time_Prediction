"""Push the serving gold tables to Supabase. Runs locally or in GitHub Actions.

Databricks Free Edition jobs have no outbound internet, so this cannot run inside the
weekly job: it reads themepark.gold.kpis_last and themepark.gold.predictions_last over
the SQL warehouse (never the _current tables) and loads Supabase atomically.

Atomicity: the two serving tables must agree with each other, so the load happens in
one Postgres transaction. The only Supabase credential available is the service key
(REST), not a database password, so the transaction is a PostgREST RPC: rows are
chunk-loaded into *_staging tables, then publish_serving() swaps both serving tables
inside a single function call -- which Postgres executes as a single transaction. If
anything fails it rolls back and Supabase keeps the previous consistent pair.

A failed push does NOT touch the Databricks _last tables: they are the source of truth
and their data is good; only the downstream copy failed. Retry, then fail loudly.

Usage:
    python publish.py                          fetch from warehouse, push to Supabase
    python publish.py --save-parquet DIR       also save the fetched tables
    python publish.py --from-parquet DIR       read tables from DIR instead of warehouse
    python publish.py --no-push                fetch/validate/save only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("publish")

ROOT = Path(__file__).resolve().parent

KPI_TABLE = "themepark.gold.kpis_last"
PRED_TABLE = "themepark.gold.predictions_last"

CHUNK_ROWS = 5000
PUSH_ATTEMPTS = 3


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables.
    Values are secrets: never log them."""
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
    values = {}
    missing = []
    for name in names:
        value = os.environ.get(name, "")
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing required environment variables: {', '.join(missing)}")
    return values


def fetch_from_warehouse() -> tuple[pd.DataFrame, pd.DataFrame]:
    from databricks import sql as dbsql

    env = require_env("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID")
    host = env["DATABRICKS_HOST"].removeprefix("https://").removeprefix("http://").rstrip("/")
    # Accept either the bare warehouse id or a pasted path like sql/warehouses/<id>.
    warehouse_id = env["DATABRICKS_WAREHOUSE_ID"].strip("/").split("/")[-1]
    # One connection, both queries back to back: every warehouse query is a cold start
    # on the free tier, so batch them.
    with dbsql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        access_token=env["DATABRICKS_TOKEN"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {KPI_TABLE}")
            kpis = cur.fetchall_arrow().to_pandas()
            cur.execute(f"SELECT * FROM {PRED_TABLE}")
            preds = cur.fetchall_arrow().to_pandas()
    log.info("fetched %d KPI rows, %d prediction rows from the warehouse", len(kpis), len(preds))
    return kpis, preds


def fetch_from_parquet(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    kpis = pd.read_parquet(directory / "themepark_gold_kpis_last.parquet")
    preds = pd.read_parquet(directory / "themepark_gold_predictions_last.parquet")
    log.info("read %d KPI rows, %d prediction rows from %s", len(kpis), len(preds), directory)
    return kpis, preds


def validate_pair(kpis: pd.DataFrame, preds: pd.DataFrame) -> None:
    """The two tables must be one consistent set before anything is published."""
    if kpis.empty or preds.empty:
        raise SystemExit("refusing to publish: a serving table is empty")
    kpi_runs = set(kpis["run_id"].astype(str).unique())
    pred_runs = set(preds["run_id"].astype(str).unique())
    if len(kpi_runs) != 1 or kpi_runs != pred_runs:
        raise SystemExit(
            f"refusing to publish: run_id mismatch between tables (kpis {kpi_runs}, "
            f"predictions {pred_runs})"
        )
    forecast = preds[preds["row_kind"] == "forecast"]
    if forecast.empty:
        raise SystemExit("refusing to publish: no forecast rows")
    log.info(
        "serving set: run_id %s, status %s, model %s, generated_at %s, forecast %s..%s",
        kpi_runs.pop(),
        preds["run_status"].iloc[0],
        forecast["model_name"].iloc[0],
        preds["generated_at"].iloc[0],
        forecast["ts_local"].min(),
        forecast["ts_local"].max(),
    )


def to_records(df: pd.DataFrame) -> list[dict]:
    # to_json handles NaN -> null and datetimes -> ISO strings in one pass.
    return json.loads(df.to_json(orient="records", date_format="iso"))


class SupabaseREST:
    def __init__(self, url: str, service_key: str):
        import requests

        self.base = url.rstrip("/") + "/rest/v1"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
            }
        )

    def _raise_readably(self, response, context: str) -> None:
        if response.status_code == 404:
            # A missing table or function is a configuration problem, not a transient
            # failure: retrying cannot fix it, so fail fast with the actual remedy.
            raise SystemExit(
                f"{context}: HTTP 404. Run supabase_schema.sql in the Supabase SQL "
                "editor first (projects created after 2026-05-30 grant no PostgREST "
                "access implicitly, so missing objects and missing grants both 404)."
            )
        if response.status_code >= 400:
            raise RuntimeError(f"{context}: HTTP {response.status_code} {response.text[:500]}")

    def rpc(self, function: str):
        r = self.session.post(f"{self.base}/rpc/{function}", json={}, timeout=120)
        self._raise_readably(r, f"rpc {function}")
        return r.json() if r.text else None

    def insert_df(self, table: str, df: pd.DataFrame) -> None:
        # Serialize chunk by chunk: the predictions table carries every model's
        # backtest rows, and one records list for all of them wastes memory.
        # resolution=ignore-duplicates makes each chunk idempotent: if the edge drops
        # a kept-alive connection after Postgres committed and the HTTP client
        # resends the chunk, the resend must not collide with its own first copy.
        for start in range(0, len(df), CHUNK_ROWS):
            chunk = to_records(df.iloc[start : start + CHUNK_ROWS])
            r = self.session.post(
                f"{self.base}/{table}",
                json=chunk,
                headers={"Prefer": "return=minimal,resolution=ignore-duplicates"},
                timeout=300,
            )
            self._raise_readably(r, f"insert into {table} (rows {start}..{start + len(chunk)})")


def push_to_supabase(kpis: pd.DataFrame, preds: pd.DataFrame) -> None:
    env = require_env("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
    api = SupabaseREST(env["SUPABASE_URL"], env["SUPABASE_SERVICE_KEY"])

    last_error: Exception | None = None
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        try:
            api.rpc("stage_reset")
            api.insert_df("kpis_staging", kpis)
            api.insert_df("predictions_staging", preds)
            counts = api.rpc("publish_serving")
            log.info("supabase publish committed atomically: %s", counts)
            return
        except Exception as exc:
            last_error = exc
            log.warning("push attempt %d/%d failed: %s", attempt, PUSH_ATTEMPTS, exc)
            if attempt < PUSH_ATTEMPTS:
                time.sleep(10 * attempt)
    # The Databricks _last tables stay exactly as they are: they are correct, only the
    # downstream copy failed. Supabase still serves the previous consistent pair.
    raise SystemExit(f"supabase push failed after {PUSH_ATTEMPTS} attempts: {last_error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-parquet", type=Path, default=None)
    parser.add_argument("--save-parquet", type=Path, default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)

    load_env(ROOT / ".env")

    if args.from_parquet:
        kpis, preds = fetch_from_parquet(args.from_parquet)
    else:
        kpis, preds = fetch_from_warehouse()

    validate_pair(kpis, preds)

    if args.save_parquet:
        args.save_parquet.mkdir(parents=True, exist_ok=True)
        kpis.to_parquet(args.save_parquet / "themepark_gold_kpis_last.parquet", index=False)
        preds.to_parquet(args.save_parquet / "themepark_gold_predictions_last.parquet", index=False)
        log.info("saved serving tables to %s", args.save_parquet)

    if args.no_push:
        log.info("push skipped (--no-push)")
        return

    push_to_supabase(kpis, preds)


if __name__ == "__main__":
    main()
