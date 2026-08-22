"""Weekly retraining: train every candidate, score them on one holdout, decide a champion.

This module is deliberately plain Python with no Databricks imports, so the identical code
path runs on a laptop, in a GitHub Actions runner and in a Databricks task. Only the entry
point differs. That is the escape hatch for the largest platform risk in this build: if
Prophet misbehaves on Databricks serverless -- cmdstan is the usual culprit -- training
moves to a runner with no code change, and MLflow tracking still points at Databricks.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .config import pipeline
from .evaluate import chronological_split, evaluate_model, format_scorecard, high_wait_entities
from .models import CANDIDATE_REGISTRY
from .models.base import WaitTimeModel
from .promote import PromotionDecision, decide

log = logging.getLogger(__name__)

SILVER_COLUMNS = ["park_name", "ride_key", "ride_name", "ts_local", "wait_time"]


@dataclass
class TrainingRun:
    """Everything one weekly execution produced."""

    results: list[dict]
    decision: PromotionDecision
    models: dict[str, WaitTimeModel]
    train_rows: int
    test_rows: int
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    git_sha: str
    started_at: datetime

    @property
    def champion(self) -> WaitTimeModel:
        return self.models[self.decision.winner]

    def scorecard(self) -> str:
        return format_scorecard(self.results)


def git_sha() -> str:
    """Which commit produced this model. Cheap, and the first thing you want in an incident."""
    for env_var in ("GITHUB_SHA", "GIT_SHA"):
        if os.environ.get(env_var):
            return os.environ[env_var][:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def validate_silver(df: pd.DataFrame) -> None:
    """Fail fast on a malformed silver table rather than training on nonsense."""
    missing = [c for c in SILVER_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"silver table is missing columns: {missing}")
    if df.empty:
        raise ValueError("silver table is empty; check ingestion and the bronze->silver build")
    if pd.to_datetime(df["ts_local"]).dt.tz is not None:
        raise ValueError("ts_local must be tz-naive park-local wall time, not tz-aware")
    if df["wait_time"].isna().any():
        raise ValueError("silver contains null wait_time values")


def train_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: list[str] | None = None,
) -> tuple[list[dict], dict[str, WaitTimeModel]]:
    """Fit and score each candidate on the same split.

    The high-wait segment is computed once from the training data and shared, so every
    model is judged against an identical definition of "the rides that matter".
    """
    names = candidates or list(CANDIDATE_REGISTRY)
    segment = high_wait_entities(train)
    log.info("high-wait segment: %d entities above %.0f min", len(segment), pipeline().high_wait_threshold_min)

    results, models = [], {}
    for name in names:
        log.info("training %s ...", name)
        model = CANDIDATE_REGISTRY[name]().fit(train)
        scores = evaluate_model(model, train, test, high_wait=segment)
        results.append(scores)
        models[name] = model
        log.info(
            "  %s: MAE %.2f | high-wait MAE %.2f | coverage %.3f",
            name,
            scores["overall_mae"],
            scores["high_wait_rides_mae"],
            scores["coverage"],
        )
    return results, models


def run_training(
    silver: pd.DataFrame,
    incumbent: WaitTimeModel | None = None,
    candidates: list[str] | None = None,
    backtest_days: int | None = None,
) -> TrainingRun:
    """One full weekly cycle: split, train, score, decide.

    The incumbent champion is re-scored on this same fresh holdout rather than compared
    against its stored metrics from a previous week, so the comparison is like-for-like.
    """
    started = datetime.now(timezone.utc)
    validate_silver(silver)

    silver = silver.sort_values("ts_local").reset_index(drop=True)
    train, test = chronological_split(silver, backtest_days)
    if test.empty:
        raise ValueError(
            f"holdout is empty -- silver spans only {silver.ts_local.min()} to "
            f"{silver.ts_local.max()}, shorter than the backtest window"
        )
    log.info(
        "split: train %d rows (to %s), holdout %d rows (%s to %s)",
        len(train),
        train.ts_local.max(),
        len(test),
        test.ts_local.min(),
        test.ts_local.max(),
    )

    results, models = train_candidates(train, test, candidates)

    incumbent_result = None
    if incumbent is not None:
        log.info("re-scoring incumbent champion %s on the same holdout", incumbent.name)
        incumbent_result = evaluate_model(incumbent, train, test)
        incumbent_result["model"] = incumbent.name

    decision = decide(results, incumbent_result)
    log.info("%s", decision)

    return TrainingRun(
        results=results,
        decision=decision,
        models=models,
        train_rows=len(train),
        test_rows=len(test),
        train_end=train.ts_local.max(),
        test_end=test.ts_local.max(),
        git_sha=git_sha(),
        started_at=started,
    )


def log_to_mlflow(run: TrainingRun, experiment: str | None = None) -> str | None:
    """Record the run: one parent run for the cycle, one nested run per candidate.

    Returns the champion's model URI, or None if MLflow is unavailable (local dev).
    """
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping tracking")
        return None

    cfg = pipeline()
    mlflow.set_experiment(experiment or cfg.mlflow_experiment)

    with mlflow.start_run(run_name=f"weekly-{run.started_at:%Y-%m-%d}"):
        mlflow.log_params(
            {
                "git_sha": run.git_sha,
                "train_rows": run.train_rows,
                "test_rows": run.test_rows,
                "train_end": str(run.train_end),
                "test_end": str(run.test_end),
                "backtest_days": cfg.backtest_days,
                "candidates": ",".join(r["model"] for r in run.results),
            }
        )

        for scores in run.results:
            name = scores["model"]
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"model": name, "git_sha": run.git_sha})
                mlflow.log_metrics(
                    {k: v for k, v in scores.items() if isinstance(v, (int, float))}
                )

        mlflow.log_metrics(
            {
                "champion_mae": run.decision.details.get("winner_mae", float("nan")),
                "baseline_mae": run.decision.details.get("baseline_mae", float("nan")),
                "promoted": int(run.decision.promoted),
            }
        )
        mlflow.set_tags(
            {
                "champion": run.decision.winner,
                "promoted": str(run.decision.promoted),
                "decision_reason": run.decision.reason[:500],
            }
        )
        mlflow.log_text(run.scorecard(), "scorecard.txt")

        if not run.decision.promoted:
            log.info("promotion declined; incumbent artefact is retained unchanged")
            return None

        champion = run.champion
        sample = pd.DataFrame(
            {
                "park_name": ["Disneyland"],
                "ride_key": ["Space_Mountain"],
                "ts_local": [pd.Timestamp("2026-01-01 12:00:00")],
            }
        )
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_adapter(champion),
            input_example=sample,
            registered_model_name=cfg.registered_model_name,
        )
        log.info("registered champion %s as %s", champion.name, info.model_uri)
        return info.model_uri


def pyfunc_adapter(model: WaitTimeModel):
    """Wrap a `WaitTimeModel` in the MLflow pyfunc interface.

    Built lazily rather than declared at module scope so that `themepark.models` and this
    module stay importable -- and unit-testable -- without MLflow installed. The wrapped
    model is what gets pickled into the registry, which is why every family had to be
    picklable in the first place.
    """
    import mlflow.pyfunc

    class WaitTimeForecaster(mlflow.pyfunc.PythonModel):
        def __init__(self, wrapped: WaitTimeModel) -> None:
            self.wrapped = wrapped

        def predict(self, context, model_input, params=None):  # noqa: ARG002
            frame = pd.DataFrame(model_input)
            frame["ts_local"] = pd.to_datetime(frame["ts_local"])
            return self.wrapped.predict(frame)

    return WaitTimeForecaster(model)


def load_champion(model_name: str | None = None, alias: str = "champion") -> WaitTimeModel | None:
    """Load the registered champion, or None on the first ever run.

    Reaches through the pyfunc wrapper to the underlying `WaitTimeModel` so the gate can
    re-score it with the same code path as the challengers.
    """
    try:
        import mlflow
    except ImportError:
        return None

    uri = f"models:/{model_name or pipeline().registered_model_name}@{alias}"
    try:
        loaded = mlflow.pyfunc.load_model(uri)
        return loaded.unwrap_python_model().wrapped
    except Exception as exc:
        log.info("no champion at %s (%s); treating this as a first deployment", uri, exc)
        return None
