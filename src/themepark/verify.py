"""Weekly production accuracy: what we predicted, versus what actually happened.

This is the most valuable component in the pipeline and the one an interviewer should be
pointed at first. Every other metric here is a backtest -- a model scored on data that
already existed when it was trained, however carefully the split was drawn. This job
scores predictions that were published *before the observations existed*. There is no way
to leak into it.

The v1 repo is the argument for why it matters: offline holdout MAE was 3.3-4.4 minutes,
but the single live snapshot came in at 9.8-12.1, and the ranking of the model families
flipped between the two. Quoting the offline number would have been wrong by a factor of
three. A deployment that measures and publishes that gap every week is telling the truth;
one that quotes its backtest is not.
"""

from __future__ import annotations

import logging

import pandas as pd

from .evaluate import score_arrays
from .features import build_features, is_high_traffic, is_peak_hour

log = logging.getLogger(__name__)

# Predictions sit on an exact 30-minute local grid; observations are resampled onto the
# same grid upstream, so the join is exact rather than nearest-neighbour.
JOIN_KEYS = ["park_name", "ride_key", "ts_local"]


def join_predictions_to_actuals(predictions: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a past forecast to the observations that have since landed.

    Inner, deliberately: a slot with no observation means the ride was down or the park was
    closed, and scoring against an absent actual would invent accuracy that was never
    measured.
    """
    preds = predictions.copy()
    preds["ts_local"] = pd.to_datetime(preds["ts_local"])

    actuals = silver[[*JOIN_KEYS, "wait_time"]].copy()
    actuals["ts_local"] = pd.to_datetime(actuals["ts_local"])

    joined = preds.merge(actuals, on=JOIN_KEYS, how="inner", validate="one_to_one")
    joined["error"] = joined["predicted_wait_min"] - joined["wait_time"]
    joined["abs_error"] = joined["error"].abs()
    joined["within_band"] = (joined["wait_time"] >= joined["lower_bound"]) & (
        joined["wait_time"] <= joined["upper_bound"]
    )

    matched = len(joined) / len(preds) if len(preds) else 0.0
    log.info(
        "matched %d of %d predicted slots (%.1f%%) to observed actuals",
        len(joined),
        len(preds),
        matched * 100,
    )
    return joined


def accuracy_summary(joined: pd.DataFrame, forecast_week: str | None = None) -> pd.DataFrame:
    """Segment-level production accuracy, one row per segment.

    Mirrors the backtest scorecard's segments so the dashboard can show the two side by
    side. The distance between them is the number that matters.
    """
    if joined.empty:
        return pd.DataFrame()

    feats = build_features(joined)
    segments = {
        "overall": pd.Series(True, index=joined.index),
        "peak_hours": is_peak_hour(feats),
        "weekend_holiday": is_high_traffic(feats),
        "nonzero_actual": joined["wait_time"] > 0,
        "long_queues": joined["wait_time"] >= 30,
    }

    week = forecast_week or (
        joined["forecast_week"].iloc[0] if "forecast_week" in joined else "unknown"
    )

    rows = []
    for name, mask in segments.items():
        sel = joined[mask.to_numpy()]
        if sel.empty:
            continue
        scores = score_arrays(
            sel["wait_time"].to_numpy(dtype=float),
            sel["predicted_wait_min"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "forecast_week": week,
                "segment": name,
                "model_name": sel["model_name"].iloc[0],
                "model_version": sel["model_version"].iloc[0],
                "interval_coverage_pct": float(sel["within_band"].mean() * 100),
                **scores,
            }
        )
    return pd.DataFrame(rows)


def accuracy_by_ride(joined: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Worst-performing attractions, for the dashboard and for deciding what to fix next."""
    if joined.empty:
        return pd.DataFrame()

    by_ride = (
        joined.groupby(["park_name", "ride_key", "ride_name"])
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            mean_actual=("wait_time", "mean"),
            interval_coverage_pct=("within_band", lambda s: float(s.mean() * 100)),
        )
        .reset_index()
    )
    by_ride["rmse"] = (
        joined.groupby(["park_name", "ride_key", "ride_name"])["error"]
        .apply(lambda s: float((s**2).mean() ** 0.5))
        .to_numpy()
    )
    return by_ride.sort_values("mae", ascending=False).head(top_n).reset_index(drop=True)


def drift_summary(silver: pd.DataFrame, weeks: int = 8, drop_partial: bool = True) -> pd.DataFrame:
    """Weekly shape of the incoming data.

    Catches the failure a purely accuracy-based monitor misses: a collector that quietly
    half-fails still produces a reasonable MAE on the rows it does return, but the row
    count and the share of zeros move first.

    The trailing week is dropped by default because it is almost always partial -- the
    weekly job runs mid-week relative to the period boundary, so the current week holds a
    day or two of data and renders as a 90% volume collapse. A monitor that cries wolf on
    every single run is a monitor people learn to ignore, which is worse than not having
    one.
    """
    work = silver.copy()
    work["ts_local"] = pd.to_datetime(work["ts_local"])
    cutoff = work["ts_local"].max() - pd.Timedelta(weeks=weeks + 1)
    work = work[work["ts_local"] > cutoff]
    work["_period"] = work["ts_local"].dt.to_period("W")

    summary = (
        work.groupby("_period")
        .agg(
            rows=("wait_time", "size"),
            entities=("ride_key", "nunique"),
            mean_wait=("wait_time", "mean"),
            pct_zero=("wait_time", lambda s: float((s == 0).mean() * 100)),
            p95_wait=("wait_time", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
    )

    if drop_partial and not summary.empty:
        # A week counts as complete only if observations reach its final day.
        last_period = work["_period"].max()
        if work["ts_local"].max() < last_period.end_time.normalize():
            summary = summary[summary["_period"] != last_period]

    summary["week"] = summary["_period"].astype(str)
    return summary.drop(columns=["_period"]).tail(weeks).reset_index(drop=True)


def verify_week(predictions: pd.DataFrame, silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full verification pass over one published forecast week."""
    joined = join_predictions_to_actuals(predictions, silver)
    if joined.empty:
        log.warning("no predicted slots matched an actual; nothing to verify yet")
        return {"summary": pd.DataFrame(), "by_ride": pd.DataFrame(), "joined": joined}

    summary = accuracy_summary(joined)
    overall = summary[summary.segment == "overall"].iloc[0]
    log.info(
        "production accuracy: MAE %.2f | RMSE %.2f | within 10 min %.1f%% | band coverage %.1f%%",
        overall["mae"],
        overall["rmse"],
        overall["within_10min_pct"],
        overall["interval_coverage_pct"],
    )
    return {"summary": summary, "by_ride": accuracy_by_ride(joined), "joined": joined}
