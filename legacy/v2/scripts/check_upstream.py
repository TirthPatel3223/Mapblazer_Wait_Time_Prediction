"""Prove -- or disprove -- that the upstream Postgres is reachable and usable.

Nothing else in this pipeline is worth debugging until this passes. It tests one layer at
a time and stops at the first failure, because "connection refused" and "relation does not
exist" need completely different fixes and a single stack trace rarely says which you have.

    python scripts/check_upstream.py            # read .env
    python scripts/check_upstream.py --tunnel   # assume an SSH tunnel on localhost:5432

Prints no credential values, so the output is safe to paste into a chat or a ticket.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from themepark.envfile import load as load_dotenv  # noqa: E402

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", " -- "
EXPECTED_TABLES = ["wait_times", "attractions", "themeparks"]
NEEDED_COLUMNS = {
    "wait_times": ["wait_time_id", "wait_time", "wait_time_upd_dt", "at_id"],
    "attractions": ["at_id", "at_name", "tp_id"],
    "themeparks": ["tp_id", "tp_name"],
}


def line(status: str, message: str, detail: str = "") -> None:
    print(f"  [{status:^4}] {message}")
    if detail:
        for part in detail.strip().splitlines():
            print(f"         {part}")


def load_env() -> dict[str, str]:
    """Merge `.env` into the environment and hand back the result.

    Delegates to themepark.envfile so this script and the jobs it is meant to de-risk
    cannot disagree about which credentials they are using -- a preflight that reads its
    configuration differently from the thing it checks is worse than no preflight.
    """
    load_dotenv()
    return dict(os.environ)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tunnel", action="store_true", help="test localhost:5432 instead")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    env = load_env()
    host = "localhost" if args.tunnel else env.get("PG_HOST", "")
    port = int(env.get("PG_PORT") or 5432)
    database = env.get("PG_DATABASE", "")
    user = env.get("PG_USER", "")
    password = env.get("PG_PASSWORD", "")
    sslmode = env.get("PG_SSLMODE", "require")

    print("\nUPSTREAM POSTGRES PREFLIGHT")
    print("=" * 62)

    # ---- 0. configuration present -------------------------------------------------
    print("\n0. configuration")
    missing = [
        name
        for name, value in [
            ("PG_HOST", host),
            ("PG_DATABASE", database),
            ("PG_USER", user),
            ("PG_PASSWORD", password),
        ]
        if not value
    ]
    if missing:
        line(FAIL, f"not set in .env: {', '.join(missing)}")
        line(INFO, "Fill these in before anything else can be tested.")
        return 1
    line(PASS, f"host {host}:{port}, database {database}, user {user}, sslmode {sslmode}")

    # ---- 1. DNS -------------------------------------------------------------------
    print("\n1. name resolution")
    try:
        address = socket.gethostbyname(host)
        line(PASS, f"{host} resolves to {address}")
    except socket.gaierror as exc:
        line(FAIL, f"cannot resolve {host}", str(exc))
        line(INFO, "Check the hostname, or use the EC2 public IP directly.")
        return 1

    # ---- 2. TCP -------------------------------------------------------------------
    # The decisive test. On a shared VM this is where it usually stops: Postgres bound
    # to localhost, or no security-group rule on 5432.
    print("\n2. TCP reachability")
    sock = socket.socket()
    sock.settimeout(args.timeout)
    try:
        sock.connect((address, port))
        line(PASS, f"port {port} is open and accepting connections")
    except TimeoutError:
        line(FAIL, f"port {port} timed out after {args.timeout}s")
        line(
            INFO,
            "A timeout (rather than a refusal) almost always means a firewall is\n"
            "dropping the packet -- the EC2 security group has no inbound rule for\n"
            "5432 from your address. See the options printed below.",
        )
        return remediation()
    except ConnectionRefusedError:
        line(FAIL, f"port {port} refused the connection")
        line(
            INFO,
            "Something answered and said no. Usually Postgres is listening only on\n"
            "localhost -- the normal setup when an app on the same VM is its only\n"
            "client. Check `listen_addresses` in postgresql.conf.",
        )
        return remediation()
    except OSError as exc:
        line(FAIL, f"could not reach port {port}", str(exc))
        return remediation()
    finally:
        sock.close()

    # ---- 3. Postgres handshake ----------------------------------------------------
    print("\n3. postgres authentication")
    try:
        import psycopg
    except ImportError:
        line(FAIL, "psycopg is not installed", "pip install 'psycopg[binary]'")
        return 1

    dsn = (
        f"host={host} port={port} dbname={database} user={user} "
        f"password={password} sslmode={sslmode}"
    )
    try:
        conn = psycopg.connect(dsn, connect_timeout=args.timeout)
    except Exception as exc:
        text = str(exc)
        line(FAIL, "connection failed", text[:400])
        if "SSL" in text or "ssl" in text:
            line(INFO, "Self-hosted Postgres often has no TLS. Set PG_SSLMODE=prefer.")
        elif "pg_hba" in text or "no encryption" in text:
            line(INFO, "pg_hba.conf has no rule permitting this user from your address.")
        elif "password" in text.lower() or "authentication" in text.lower():
            line(INFO, "Credentials rejected -- check PG_USER / PG_PASSWORD.")
        elif "database" in text.lower():
            line(INFO, "Database name may be wrong; list them with \\l on the host.")
        return 1

    with conn:
        line(PASS, "authenticated")
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_user, current_database()")
            version, whoami, dbname = cur.fetchone()
            line(INFO, f"{version.split(',')[0]}")
            line(INFO, f"connected as {whoami} to {dbname}")

            cur.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
            row = cur.fetchone()
            encrypted = bool(row and row[0])
            line(
                PASS if encrypted else WARN,
                f"connection is {'encrypted' if encrypted else 'NOT encrypted'}",
                "" if encrypted else "Credentials cross the internet in the clear. Prefer an SSH tunnel.",
            )

            # ---- 4. schema --------------------------------------------------------
            print("\n4. schema")
            cur.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type='BASE TABLE' AND table_schema NOT IN "
                "('pg_catalog','information_schema')"
            )
            found = cur.fetchall()
            names = {t for _, t in found}
            line(INFO, f"{len(found)} table(s) visible to this user")

            ok = True
            for table in EXPECTED_TABLES:
                if table in names:
                    line(PASS, f"{table} exists")
                else:
                    ok = False
                    close = [n for n in names if table.split("_")[0] in n][:5]
                    line(FAIL, f"{table} NOT found", f"similar: {close}" if close else "")
            if not ok:
                line(
                    INFO,
                    "Table names are constructor arguments on MapblazerPostgres --\n"
                    "edit the call in jobs/collect.py rather than the query.",
                )
                return 1

            # ---- 5. columns -------------------------------------------------------
            print("\n5. required columns")
            for table, needed in NEEDED_COLUMNS.items():
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table,),
                )
                present = {c for (c,) in cur.fetchall()}
                absent = [c for c in needed if c not in present]
                if absent:
                    line(FAIL, f"{table} missing: {', '.join(absent)}")
                    return 1
                line(PASS, f"{table}: all {len(needed)} required columns present")

            # ---- 6. the actual query ---------------------------------------------
            print("\n6. the ingestion query")
            cur.execute("SELECT count(*), max(wait_time_id) FROM wait_times")
            rows, high = cur.fetchone()
            line(PASS, f"wait_times holds {rows:,} rows, max id {high:,}")

            cur.execute(
                "SELECT max(wait_time_upd_dt), "
                "now() - max(wait_time_upd_dt) FROM wait_times"
            )
            newest, age = cur.fetchone()
            fresh = age is not None and age.total_seconds() < 7200
            line(
                PASS if fresh else WARN,
                f"newest observation {newest} ({age} ago)",
                "" if fresh else "Feed looks stale -- confirm the collector upstream is still running.",
            )

            cur.execute(
                "SELECT count(*) FROM wait_times w "
                "JOIN attractions a ON a.at_id = w.at_id "
                "JOIN themeparks t ON t.tp_id = a.tp_id "
                "WHERE w.wait_time_id > 0 LIMIT 1"
            )
            line(PASS, f"three-table join returns {cur.fetchone()[0]:,} rows")

            # ---- 7. permissions ---------------------------------------------------
            print("\n7. permissions")
            cur.execute("SELECT has_table_privilege(%s,'wait_times','INSERT')", (whoami,))
            can_write = cur.fetchone()[0]
            line(
                WARN if can_write else PASS,
                "account can write to wait_times" if can_write else "account is read-only",
                "The pipeline only reads. A read-only role would be safer -- ask for one."
                if can_write
                else "",
            )

    print("\n" + "=" * 62)
    print("  ALL CHECKS PASSED -- ingestion will work against this database.")
    print("=" * 62 + "\n")
    return 0


def remediation() -> int:
    print(
        """
  ------------------------------------------------------------------
  The database is not reachable from this machine. Four options,
  best first:

  1. SSH TUNNEL  (recommended for a shared VM)
     Postgres stays bound to localhost and 5432 never faces the
     internet. Only port 22 is exposed, restricted by key.

       ssh -N -L 5432:localhost:5432 -i key.pem ec2-user@<host>
       python scripts/check_upstream.py --tunnel

     If that passes, I can wire the same tunnel into collect.yml.

  2. RUN THE COLLECTOR ON THE VM
     It already has local database access. A cron entry pushes
     Parquet to Databricks. Nothing needs to be exposed at all --
     but it means deploying onto someone else's server.

  3. OPEN 5432 TO THE INTERNET
     Fastest, and the one I would not choose. On a VM running other
     services this widens the attack surface of all of them, and it
     is not your call to make alone.

  4. USE queue-times.com INSTEAD
     A public API covering all five of your parks, no credentials,
     nothing to negotiate. The historical extract still seeds the
     backfill. The ingestion layer was built pluggable for exactly
     this -- it is a config change, not a rewrite.
  ------------------------------------------------------------------
"""
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
