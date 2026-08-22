"""What is actually in the lakehouse, and why the last job run ended that way.

Answers "did bronze load?" without starting any compute. Unity Catalog metadata and job
run history are REST calls against the control plane, so this costs nothing against the
free-tier quota -- which matters, because the obvious way to check (SELECT count(*)) wakes
the SQL warehouse and the obvious way to debug (re-run the job) spends the thing you are
trying to protect.

    python scripts/check_lakehouse.py            # tables + last run
    python scripts/check_lakehouse.py --runs 5   # more history

Row counts are deliberately absent: there is no way to get them without compute. Presence,
schema and the failure reason are what you need first, and they are free.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from themepark.config import DatabricksSettings  # noqa: E402


def when(ms: int | None) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="how many job runs to report")
    args = parser.parse_args()

    from databricks.sdk import WorkspaceClient

    cfg = DatabricksSettings()
    cfg.require_api_credentials()
    w = WorkspaceClient(host=cfg.host, token=cfg.token)

    print("\nLAKEHOUSE STATE  (no compute used)")
    print("=" * 66)

    print(f"\nvolume {cfg.landing_path}")
    files = 0
    try:
        for entry in w.files.list_directory_contents(cfg.landing_path):
            if entry.is_directory:
                inner = list(w.files.list_directory_contents(entry.path))
                size = sum(e.file_size or 0 for e in inner)
                files += len(inner)
                print(f"   {entry.name + '/':<28} {len(inner):>3} files  {size:>12,} bytes")
            else:
                files += 1
                print(f"   {entry.name:<28}          {entry.file_size or 0:>12,} bytes")
    except Exception as exc:
        print(f"   cannot list: {type(exc).__name__}: {exc}")
    if not files:
        print("   (empty -- nothing has been uploaded, so bronze_load has nothing to read)")

    for schema in (cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema):
        print(f"\nschema {cfg.catalog}.{schema}")
        try:
            tables = list(w.tables.list(catalog_name=cfg.catalog, schema_name=schema))
        except Exception as exc:
            print(f"   cannot list: {type(exc).__name__}: {exc}")
            continue
        if not tables:
            print("   (no tables)")
        for t in tables:
            cols = len(t.columns) if t.columns else 0
            print(f"   {t.name:<28} {cols:>3} cols   updated {when(t.updated_at)}")

    print("\n" + "=" * 66)
    print("RECENT JOB RUNS")
    print("=" * 66)
    jobs = [j for j in w.jobs.list() if "themepark" in (j.settings.name or "").lower()]
    if not jobs:
        print("\n   no themepark job -- run `databricks bundle deploy --target prod`")
        return 0

    for job in jobs:
        print(f"\njob {job.job_id}  {job.settings.name}")
        runs = list(w.jobs.list_runs(job_id=job.job_id, limit=args.runs))
        if not runs:
            print("   never run")
        for run in runs:
            print(f"\n   run {run.run_id}  started {when(run.start_time)}")
            if run.state and run.state.state_message:
                print(f"      {run.state.state_message}")
            for task in (w.jobs.get_run(run_id=run.run_id).tasks or []):
                state = task.state.result_state if task.state else None
                print(f"      {task.task_key:<14} {state}")
                try:
                    out = w.jobs.get_run_output(run_id=task.run_id)
                    if out.error:
                        print(f"         {out.error.strip().splitlines()[0][:160]}")
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
