# Collector deployment — the VM hosting the source database

The collector runs on the same machine as the source Postgres, and **pushes** to
Databricks rather than being pulled from.

## Why push

The alternative was a GitHub Actions runner connecting inward to Postgres, which needs
the database exposed to the internet. On a VM running several services that is a poor
trade: you widen the attack surface of everything else on the box to solve a problem that
has a better answer.

| | Pull (rejected) | Push (this) |
|---|---|---|
| Direction | runner → VM:5432, inbound | VM → Databricks:443, outbound |
| Security group | inbound rule for 5432 | unchanged |
| `listen_addresses` | must be `'*'` | stays `'localhost'` |
| `pg_hba.conf` | remote rule needed | unchanged |
| Exposed surface | Postgres, to rotating runner IPs | nothing |
| Credentials in transit | across the internet | over the loopback interface |

Outbound HTTPS works from essentially any VM without asking anyone for a firewall change.

## What gets installed

```
/opt/themepark/            code + .venv, owned by the themepark service account
/opt/themepark/.env        credentials, mode 600
/etc/systemd/system/themepark-collect.{service,timer}
```

A `oneshot` service on a 30-minute timer. Not a daemon — nothing stays resident, and a
crashed run has no state to corrupt because the watermark is read from what actually
landed in Databricks.

## Install

```bash
git clone https://github.com/TirthPatel3223/Mapblazer_Wait_Time_Prediction.git
cd Mapblazer_Wait_Time_Prediction
sudo bash deploy/ec2/install.sh

sudo -e /opt/themepark/.env        # fill in PG_* and DATABRICKS_*
```

Then verify before trusting the timer:

```bash
sudo -u themepark /opt/themepark/.venv/bin/python /opt/themepark/scripts/check_upstream.py
sudo -u themepark /opt/themepark/.venv/bin/python /opt/themepark/jobs/collect.py --dry-run
sudo systemctl start themepark-collect.service
journalctl -u themepark-collect -n 50 --no-pager
```

One-pass backfill rather than waiting 30 minutes per batch:

```bash
sudo -u themepark /opt/themepark/.venv/bin/python /opt/themepark/jobs/collect.py --drain
```

## Footprint on the host

Deliberately small, because this is not your machine:

- **Disk** — about 120 MB (venv is pandas, pyarrow, psycopg, the Databricks SDK; no
  Prophet, XGBoost or cmdstan, since training happens in Databricks).
- **Database load** — one `SELECT` every 30 minutes returning roughly 110 rows, filtered
  by `wait_time_id > watermark` and capped by `LIMIT`. Reads only; never writes, never
  locks, never runs DDL.
- **Network** — a few MB/day of outbound HTTPS.
- **CPU/RAM** — a few seconds of one core per run; idle otherwise.
- **Privileges** — a system account with no login shell, `ProtectSystem=strict`,
  `NoNewPrivileges`, and write access to `/opt/themepark` only.

Ask for a **read-only** Postgres role. The collector never needs more, and
`check_upstream.py` reports which kind of account it is connected with.

## Operating it

```bash
systemctl list-timers 'themepark-collect*'   # next scheduled run
journalctl -u themepark-collect -f           # follow
journalctl -u themepark-collect --since today | grep -i error
sudo systemctl disable --now themepark-collect.timer   # stop completely
```

Upgrading is a re-run of the installer: it fetches, resets to origin, rebuilds the venv
and restarts the timer, leaving `.env` alone.

## Removing it

```bash
sudo systemctl disable --now themepark-collect.timer
sudo rm /etc/systemd/system/themepark-collect.{service,timer}
sudo systemctl daemon-reload
sudo rm -rf /opt/themepark
sudo userdel themepark
```

Nothing is left behind, and nothing in the source database was ever modified.

## If you cannot get access to this VM

The pipeline does not depend on it. `jobs/seed_from_csv.py` warm-starts the lakehouse
from a local CSV extract in exactly this format, which is enough to deploy and validate
the entire system. Live ingestion can be attached later — or replaced with the public
queue-times.com API, which the reader interface in `src/themepark/sources.py` was
structured to accommodate.
