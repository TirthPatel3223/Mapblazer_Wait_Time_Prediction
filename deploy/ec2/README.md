# Ingestion agent — the EC2 instance hosting the source database

This directory is the whole other end of the pipeline: the component that runs on the
AWS EC2 instance where the upstream wait-time Postgres lives, and **pushes** new rows
into `themepark.bronze.wait_times_raw`. Everything else in the repository runs in
Databricks, in GitHub Actions, or on a laptop — none of it is installed here.

    deploy/ec2/
      ingest.py                  the agent (Postgres -> bronze), no other imports
      requirements.txt           five packages, no Prophet/XGBoost/cmdstan
      install.sh                 idempotent installer
      themepark-ingest.service   oneshot unit, hardened
      themepark-ingest.timer     hourly

## Why push instead of pull

The alternative was a GitHub Actions runner connecting inward to Postgres, which needs
the database exposed to the internet. On a VM running several other services that is a
poor trade: you widen the attack surface of everything else on the box to solve a
problem that has a better answer.

| | Pull (rejected) | Push (this) |
|---|---|---|
| Direction | runner -> VM:5432, inbound | VM -> Databricks:443, outbound |
| Security group | inbound rule for 5432 | unchanged |
| `listen_addresses` | must be `'*'` | stays `'localhost'` |
| `pg_hba.conf` | remote rule needed | unchanged |
| Exposed surface | Postgres, to rotating runner IPs | nothing |
| Credentials in transit | across the internet | over the loopback interface |

Outbound HTTPS works from essentially any VM without asking anyone for a firewall change.

## What one run does

    1. watermark   MAX(wait_time_id) already in themepark.bronze.wait_times_raw
    2. read        SELECT ... FROM wait_times JOIN attractions JOIN themeparks
                   WHERE wait_time_id > watermark ORDER BY wait_time_id LIMIT 200000
    3. land        write the batch as Parquet into /Volumes/themepark/bronze/landing/
                   dt=<utc date>/part-<low id>-<timestamp>.parquet
    4. merge       MERGE that one file into bronze on wait_time_id, insert-only

Four steps, one warehouse session, no local state.

**The watermark is read from bronze, not from a cursor file.** That is the whole
reliability story. It always describes what actually landed, so a run killed at any
point simply re-reads the same range next time; the `MERGE ... WHEN NOT MATCHED` absorbs
whatever overlaps. An interrupted run leaves its Parquet file in the landing volume
unloaded, and that is harmless — landing is raw transport, bronze is the audit trail.
There is no state on this box that can drift out of sync with reality, which means
there is nothing to repair by hand at 3am.

Bronze is append-only and never updated in place. If a filter or a timezone rule turns
out to be wrong, the fix is a silver rebuild, not a re-read of someone else's production
database.

## Install

```bash
git clone https://github.com/TirthPatel3223/Mapblazer_Wait_Time_Prediction.git
cd Mapblazer_Wait_Time_Prediction
sudo bash deploy/ec2/install.sh

sudo -e /opt/themepark/.env        # fill in PG_* and DATABRICKS_*
```

Verify both ends are reachable before trusting the timer. `--dry-run` reports the gap
between the two databases and writes nothing:

```bash
sudo -u themepark /opt/themepark/.venv/bin/python /opt/themepark/deploy/ec2/ingest.py --dry-run
sudo systemctl start themepark-ingest.service
journalctl -u themepark-ingest -n 50 --no-pager
```

A cold start with hundreds of thousands of rows to catch up on drains in one pass
instead of one batch per hour:

```bash
sudo -u themepark /opt/themepark/.venv/bin/python /opt/themepark/deploy/ec2/ingest.py --drain
```

## Credentials it needs

In `/opt/themepark/.env`, mode 600, owned by the service account. Never in the repo.

| Key | Why |
|---|---|
| `PG_HOST` `PG_PORT` `PG_DATABASE` `PG_USER` `PG_PASSWORD` `PG_SSLMODE` | the local source database — a **read-only** role is enough |
| `PG_WAIT_TIMES_TABLE` `PG_ATTRACTIONS_TABLE` `PG_THEMEPARKS_TABLE` | optional, if the upstream table names differ |
| `DATABRICKS_HOST` `DATABRICKS_TOKEN` `DATABRICKS_WAREHOUSE_ID` | the target — the token needs `CAN_USE` on the warehouse and write access to `themepark.bronze` |

These live only here. The repo-root `.env` used by `publish.py` and `dashboard.py`
carries no `PG_*` keys at all: nothing outside this machine ever touches Postgres.

## Footprint on the host

Deliberately small, because this is not our machine:

- **Disk** — about 120 MB, almost all of it pandas and pyarrow.
- **Database load** — one `SELECT` per hour returning roughly 220 rows, filtered by
  `wait_time_id > watermark` and capped by `LIMIT`. Reads only; never writes, never
  locks, never runs DDL.
- **Network** — a few MB/day of outbound HTTPS.
- **CPU/RAM** — a few seconds of one core per run; idle otherwise.
- **Privileges** — a system account with no login shell, `ProtectSystem=strict`,
  `NoNewPrivileges`, and write access to `/opt/themepark` only.

## Why hourly

Each run wakes a serverless SQL warehouse, and on a metered workspace those wake-ups
cost more than the freshness is worth. The feed has 30-minute granularity and the model
retrains weekly, so hourly is already far fresher than anything downstream reads.
`OnUnitActiveSec` in the timer is the dial: widening it to `6h` loses nothing, the next
run just carries a larger batch.

## Operating it

```bash
systemctl list-timers 'themepark-ingest*'   # next scheduled run
journalctl -u themepark-ingest -f           # follow
journalctl -u themepark-ingest --since today | grep -i error
sudo systemctl disable --now themepark-ingest.timer   # stop completely
```

Upgrading is a re-run of the installer: it fetches, resets to `origin`, rebuilds the
venv and restarts the timer, leaving `.env` alone.

## Removing it

```bash
sudo systemctl disable --now themepark-ingest.timer
sudo rm /etc/systemd/system/themepark-ingest.{service,timer}
sudo systemctl daemon-reload
sudo rm -rf /opt/themepark
sudo userdel themepark
```

Nothing is left behind, and nothing in the source database was ever modified.
