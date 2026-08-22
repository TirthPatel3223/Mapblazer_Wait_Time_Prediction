"""Weekly retrain: fit the candidates, score them, decide the champion, publish a forecast.

Runs as a Databricks task by default. The `--source local` / `--source sql` switches exist
because the modelling code has no Databricks dependency, so if Prophet ever misbehaves on
serverless the identical job can run on a GitHub Actions runner with MLflow still pointed
at Databricks. That fallback is deliberate: it is the largest single platform risk here and
it costs nothing to keep the escape hatch open.

    # inside Databricks (spark available)
    python jobs/train_job.py

    # from a runner, reading silver over the SQL warehouse
    python jobs/train_job.py --source sql
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _databricks import log  # noqa: E402
from themepark.config import DatabricksSettings, pipeline  # noqa: E402
from themepark.promote import NoViableModelError, versions_to_retire  # noqa: E402
from themepark.score import generate_forecast  # noqa: E402
from themepark.train import load_champion, log_to_mlflow, run_training  # noqa: E402


def load_silver(source: str, cfg: DatabricksSettings, path: str | None) -> pd.DataFrame:
    if source == "local":
        return pd.read_parquet(path)
    if source == "sql":
        from themepark.sources import Databricks

        return Databricks(cfg).read_table(cfg.silver_table)
    from _databricks import read_delta, spark_session

    return read_delta(spark_session(), cfg.silver_table)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["spark", "sql", "local"], default="spark")
    parser.add_argument("--path", help="parquet path when --source local")
    parser.add_argument("--candidates", help="comma-separated subset, for debugging")
    parser.add_argument("--no-register", action="store_true", help="train and score, register nothing")
    args = parser.parse_args()

    cfg = DatabricksSettings()
    settings = pipeline()

    silver = load_silver(args.source, cfg, args.path)
    silver["ts_local"] = pd.to_datetime(silver["ts_local"])
    log.info("loaded %d silver rows", len(silver))

    # Re-score the incumbent on this week's holdout rather than trusting its stored
    # metrics: those were measured on a different week's data and are not comparable.
    incumbent = load_champion()
    if incumbent is not None:
        log.info("incumbent champion: %s", incumbent.name)

    candidates = args.candidates.split(",") if args.candidates else None

    try:
        run = run_training(silver, incumbent=incumbent, candidates=candidates)
    except NoViableModelError:
        # Nothing beat a per-ride historical mean. Something upstream is broken and the
        # right move is to fail loudly, keep serving last week's forecast, and get paged.
        log.exception("PROMOTION BLOCKED -- no candidate beat the baseline")
        raise

    print("\n" + run.scorecard() + "\n")

    model_uri = None if args.no_register else log_to_mlflow(run)
    log.info("decision: %s", run.decision)

    _write_promotion_log(cfg, run)

    if run.decision.promoted and model_uri:
        _set_champion_alias(cfg, settings)

    # Score with whichever model actually holds champion: on a declined promotion that is
    # still the incumbent, so the published forecast never silently switches models.
    champion = run.champion if run.decision.promoted else (incumbent or run.champion)
    forecast = generate_forecast(champion, silver, model_version=str(model_uri or "incumbent"))
    _write_forecast(args, cfg, forecast)

    log.info("published %d forecast rows from %s", len(forecast), champion.name)
    return 0


def _write_promotion_log(cfg: DatabricksSettings, run) -> None:
    """Persist the decision either way. A gate nobody can audit is not a gate."""
    row = run.decision.as_row()
    row["git_sha"] = run.git_sha
    row["scorecard"] = json.dumps(run.results, default=str)[:8000]
    frame = pd.DataFrame([{k: (str(v) if isinstance(v, bool) else v) for k, v in row.items()}])
    try:
        from _databricks import spark_session, write_delta

        write_delta(spark_session(), frame, cfg.promotion_table, mode="append")
    except Exception as exc:
        log.warning("could not append to the promotion log: %s", exc)


def _set_champion_alias(cfg: DatabricksSettings, settings) -> None:
    """Point the @champion alias at the newest version and retire stale ones."""
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        name = settings.registered_model_name
        versions = [int(v.version) for v in client.search_model_versions(f"name='{name}'")]
        newest = max(versions)
        client.set_registered_model_alias(name, "champion", newest)
        log.info("@champion -> version %d", newest)

        for stale in versions_to_retire(versions):
            client.delete_model_version(name, str(stale))
            log.info("retired version %d (free-tier storage quota)", stale)
    except Exception as exc:
        log.warning("could not update the champion alias: %s", exc)


def _write_forecast(args, cfg: DatabricksSettings, forecast: pd.DataFrame) -> None:
    if args.source == "local":
        out = Path(args.path).parent / "forecast.parquet"
        forecast.to_parquet(out, index=False)
        log.info("wrote forecast to %s", out)
        return
    from _databricks import spark_session, write_delta

    write_delta(spark_session(), forecast, cfg.predictions_table, mode="append")


if __name__ == "__main__":
    raise SystemExit(main())
