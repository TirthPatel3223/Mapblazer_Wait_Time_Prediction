#!/usr/bin/env bash
#
# Install the wait-time ingestion agent on the EC2 instance that hosts the source
# Postgres.
#
# It runs here rather than in the cloud so the database connection is local. Nothing
# about Postgres has to be exposed: no inbound firewall rule, no `listen_addresses = '*'`,
# no SSH tunnel. The only network access needed is outbound HTTPS to Databricks, which
# this box already has.
#
# Safe to re-run: it upgrades the code and restarts the timer without touching .env.
#
#   sudo bash deploy/ec2/install.sh
#
set -euo pipefail

APP_DIR=/opt/themepark
SERVICE_USER=themepark
REPO_URL="${REPO_URL:-https://github.com/TirthPatel3223/Mapblazer_Wait_Time_Prediction.git}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

say "Checking prerequisites"
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "$PYTHON_BIN is required" >&2; exit 1; }
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10+ required, found {sys.version.split()[0]}")
PY
echo "ok: $("$PYTHON_BIN" --version)"

say "Creating the service account"
# A dedicated, non-login account. Ingestion never needs a shell, and this keeps the
# Databricks token out of any human user's dotfiles.
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
    echo "created $SERVICE_USER"
else
    echo "$SERVICE_USER already exists"
fi

say "Fetching the code"
if [[ -d "$APP_DIR/.git" ]]; then
    git -C "$APP_DIR" fetch --quiet origin
    git -C "$APP_DIR" reset --hard --quiet origin/HEAD
    echo "updated $APP_DIR"
else
    mkdir -p "$APP_DIR"
    git clone --quiet --depth 1 "$REPO_URL" "$APP_DIR"
    echo "cloned into $APP_DIR"
fi

say "Building the virtualenv"
"$PYTHON_BIN" -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/deploy/ec2/requirements.txt"
echo "installed the ingestion dependencies only (no Prophet, XGBoost or cmdstan)"

say "Preparing configuration"
if [[ ! -f "$APP_DIR/.env" ]]; then
    cat > "$APP_DIR/.env" <<'ENVEOF'
# Postgres is on this machine, so it is reached over the loopback interface. Nothing
# here needs to be exposed to the network. A read-only role is sufficient: ingestion
# never writes, locks or runs DDL against this database.
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=
PG_USER=
PG_PASSWORD=
# Local connections are not usually TLS-terminated; they never leave the host.
PG_SSLMODE=prefer

# Optional, only if the upstream table names differ from the defaults.
# PG_WAIT_TIMES_TABLE=wait_times
# PG_ATTRACTIONS_TABLE=attractions
# PG_THEMEPARKS_TABLE=themeparks

# Needs CAN_USE on the SQL warehouse and write access to themepark.bronze.
DATABRICKS_HOST=
DATABRICKS_TOKEN=
DATABRICKS_WAREHOUSE_ID=
ENVEOF
    echo "wrote $APP_DIR/.env  <-- FILL THIS IN"
else
    echo "$APP_DIR/.env already exists, left untouched"
fi

# The token in here is equivalent to workspace access. Nobody else on a shared box
# should be able to read it.
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
chmod 600 "$APP_DIR/.env"

say "Installing the systemd timer"
install -m 644 "$APP_DIR/deploy/ec2/themepark-ingest.service" /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/ec2/themepark-ingest.timer"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now themepark-ingest.timer
echo "timer enabled"

cat <<EOF

────────────────────────────────────────────────────────────────────
Installed. Two things left:

  1. Fill in the credentials
       sudo -e $APP_DIR/.env

  2. Check both ends are reachable before trusting the timer, then run once
       sudo -u $SERVICE_USER $APP_DIR/.venv/bin/python $APP_DIR/deploy/ec2/ingest.py --dry-run
       sudo systemctl start themepark-ingest.service
       journalctl -u themepark-ingest -n 50 --no-pager

Useful afterwards:
  systemctl list-timers themepark-ingest\*     # when it next fires
  journalctl -u themepark-ingest -f            # follow the logs
  systemctl disable --now themepark-ingest.timer   # stop it entirely

Backfill in one pass instead of one batch per hour:
  sudo -u $SERVICE_USER $APP_DIR/.venv/bin/python $APP_DIR/deploy/ec2/ingest.py --drain
────────────────────────────────────────────────────────────────────
EOF
