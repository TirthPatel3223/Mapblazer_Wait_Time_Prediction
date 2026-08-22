"""The segmented backtest scorecard.

A single overall MAE is a bad way to rank these models. Even after the timezone fix
roughly 37% of observations are exactly zero, so a model that learns nothing but "usually
there is no queue" scores respectably while being useless for routing. And the failure
mode that actually hurts a guest is the tail: a route planned around a 15-minute estimate
that turns out to be 55 is worse than one that was merely a little wrong all day.

So every candidate is scored on the same holdout across several slices, and the gate in
`themepark.promote` reads more than one of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import pipeline
from .features import build_features, is_high_traffic, is_peak_hour
from .models.base import WaitTimeModel
from .naming import entity_key

# Reported for every segment. `bias` earns its place because MAE is blind to systematic
# offset: v1's baseline sat 15.2 minutes low on live data while still looking acceptable.
METRIC_NAMES = ["n", "mae", "rmse", "p95_abs_error", "within_10min_pct", "bias", "mean_actual"]


def score_arrays(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Point metrics for one slice."""
    if len(actual) == 0:
        return dict.fromkeys(METRIC_NAMES, float("nan")) | {"n": 0}

    error = predicted - actual
    abs_error = np.abs(error)
    return {
        "n": int(len(actual)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "within_10min_pct": float((abs_error <= 10).mean() * 100),
        "bias": float(error.mean()),
        "mean_actual": float(actual.mean()),
    }


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals falling inside the predicted band.

    A model advertising an 80% interval should land near 80. Far above means the band is
    uselessly wide; far below means the optimiser is being handed false confidence.
    """
    if len(actual) == 0:
        return float("nan")
    return float(((actual >= lower) & (actual <= upper)).mean() * 100)


def high_wait_entities(train: pd.DataFrame, threshold: float | None = None) -> set[str]:
    """Entities whose training mean clears the threshold -- where accuracy has value.

    Derived from training data only. Choosing the segment from the holdout would let a
    quiet week redefine which rides count.
    """
    threshold = pipeline().high_wait_threshold_min if threshold is None else threshold
    work = train.copy()
    work["_entity"] = work.apply(lambda r: entity_key(r["park_name"], r["ride_key"]), axis=1)
    means = work.groupby("_entity")["wait_time"].mean()
    return set(means[means > threshold].index)


def build_segments(test: pd.DataFrame, high_wait: set[str]) -> dict[str, pd.Series]:
    """Boolean masks over the holdout, aligned to `test.index`."""
    feats = build_features(test)
    entities = test.apply(lambda r: entity_key(r["park_name"], r["ride_key"]), axis=1)
    return {
        "overall": pd.Series(True, index=test.index),
        "high_wait_rides": entities.isin(high_wait),
        "peak_hours": is_peak_hour(feats),
        "weekend_holiday": is_high_traffic(feats),
        "nonzero_actual": test["wait_time"] > 0,
    }


def evaluate_model(
    model: WaitTimeModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    high_wait: set[str] | None = None,
) -> dict:
    """Score one fitted model on the holdout across every segment.

    Returns a flat dict shaped for MLflow: `{segment}_{metric}`, plus `coverage`.
    """
    if high_wait is None:
        high_wait = high_wait_entities(train)

    preds = model.predict(test[["park_name", "ride_key", "ts_local"]])
    actual = test["wait_time"].to_numpy(dtype=float)

    result: dict[str, float | str] = {
        "model": model.name,
        # The v1 tripwire: share of holdout rows this model can actually serve.
        "coverage": model.coverage(test),
        "interval_coverage_pct": interval_coverage(
            actual, preds["yhat_lower"].to_numpy(), preds["yhat_upper"].to_numpy()
        ),
        "n_entities_fitted": len(model.entities_),
        "train_rows": model.train_rows_,
    }

    for segment, mask in build_segments(test, high_wait).items():
        sel = mask.to_numpy()
        scores = score_arrays(actual[sel], preds["yhat"].to_numpy()[sel])
        for metric, value in scores.items():
            result[f"{segment}_{metric}"] = value

    return result


def scorecard_frame(results: list[dict]) -> pd.DataFrame:
    """Leaderboard ordered by the headline metric, best first."""
    df = pd.DataFrame(results)
    return df.sort_values("overall_mae").reset_index(drop=True)


def format_scorecard(results: list[dict]) -> str:
    """Human-readable leaderboard for job logs and the dashboard."""
    df = scorecard_frame(results)
    header = (
        f"{'model':<18}{'MAE':>7}{'RMSE':>8}{'p95':>7}{'bias':>8}"
        f"{'<10m%':>8}{'hiMAE':>8}{'peakMAE':>9}{'cov':>7}"
    )
    lines = [header, "-" * len(header)]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']:<18}{r['overall_mae']:>7.2f}{r['overall_rmse']:>8.2f}"
            f"{r['overall_p95_abs_error']:>7.1f}{r['overall_bias']:>+8.2f}"
            f"{r['overall_within_10min_pct']:>8.1f}{r['high_wait_rides_mae']:>8.2f}"
            f"{r['peak_hours_mae']:>9.2f}{r['coverage']:>7.2f}"
        )
    return "\n".join(lines)


def chronological_split(
    df: pd.DataFrame, backtest_days: int | None = None, ts_col: str = "ts_local"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing-window holdout: everything before the cut trains, the tail is the test.

    A rolling origin, not the random 80/20 of v1. This is what a weekly-retrain system
    genuinely faces -- predict a period that has not happened yet, using only what came
    before it.
    """
    days = pipeline().backtest_days if backtest_days is None else backtest_days
    cut = df[ts_col].max() - pd.Timedelta(days=days)
    return df[df[ts_col] <= cut].copy(), df[df[ts_col] > cut].copy()
