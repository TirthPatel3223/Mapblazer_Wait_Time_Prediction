"""Batch scoring: the next seven days of wait times for every attraction.

The forecast is generated ahead of time rather than served on demand because the consumer
is a time-dependent travelling-salesman solver, which needs the whole cost surface at once
-- every ride at every arrival time -- not one point per request. Precomputing turns that
into a single indexed table read.

Bounds travel with every row. A route optimised purely on point estimates has no way to
prefer a reliable 20-minute queue over a volatile one averaging the same.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .config import pipeline
from .filters import PARK_CONSTRAINTS
from .models.base import WaitTimeModel
from .timeutils import to_utc

log = logging.getLogger(__name__)

PREDICTION_SCHEMA = [
    "park_name",
    "ride_key",
    "ride_name",
    "ts_local",
    "ts_utc",
    "predicted_wait_min",
    "lower_bound",
    "upper_bound",
    "model_name",
    "model_version",
    "forecast_week",
    "generated_at",
]


class CoverageError(RuntimeError):
    """The champion cannot serve enough of the fleet to publish a forecast."""


def active_entities(silver: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    """Attractions seen recently enough to still be operating.

    Guards against forecasting rides that closed months ago -- v1's dataset still carried
    a whole superseded SeaWorld park.
    """
    cutoff = silver["ts_local"].max() - pd.Timedelta(days=lookback_days)
    recent = silver[silver["ts_local"] > cutoff]
    return (
        recent[["park_name", "ride_key", "ride_name"]]
        .drop_duplicates(subset=["park_name", "ride_key"])
        .reset_index(drop=True)
    )


def build_forecast_grid(
    entities: pd.DataFrame,
    start: pd.Timestamp | None = None,
    horizon_days: int | None = None,
    grid_minutes: int | None = None,
) -> pd.DataFrame:
    """Every (attraction, local time slot) pair inside park operating hours.

    Built in park-local time because that is when guests actually visit; the UTC column is
    derived afterwards so downstream consumers never have to do the conversion themselves
    -- which is the mistake that produced the defect this pipeline was built to fix.
    """
    cfg = pipeline()
    horizon = horizon_days or cfg.horizon_days
    step = grid_minutes or cfg.grid_minutes

    start = (start or pd.Timestamp.now()).normalize()
    slots = pd.date_range(start, start + pd.Timedelta(days=horizon), freq=f"{step}min", inclusive="left")

    frames = []
    for park, constraint in PARK_CONSTRAINTS.items():
        park_entities = entities[entities["park_name"] == park]
        if park_entities.empty:
            continue
        open_slots = slots[
            (slots.hour >= constraint.open_hour) & (slots.hour < constraint.close_hour)
        ]
        if len(open_slots) == 0:
            continue
        frames.append(park_entities.merge(pd.DataFrame({"ts_local": open_slots}), how="cross"))

    if not frames:
        return pd.DataFrame(columns=["park_name", "ride_key", "ride_name", "ts_local"])

    grid = pd.concat(frames, ignore_index=True)
    log.info(
        "forecast grid: %d rows across %d attractions, %s to %s local",
        len(grid),
        len(entities),
        grid.ts_local.min(),
        grid.ts_local.max(),
    )
    return grid


def score_grid(
    model: WaitTimeModel,
    grid: pd.DataFrame,
    model_version: str = "unknown",
    min_coverage: float | None = None,
) -> pd.DataFrame:
    """Score the grid and shape it for the serving table.

    Refuses to publish a forecast the champion cannot mostly cover. In v1 an unresolved
    ride produced a silent NaN that was later dropped; here it fails the job, because a
    routing optimiser handed a partial map will confidently plan around the gaps.
    """
    threshold = pipeline().min_coverage if min_coverage is None else min_coverage
    coverage = model.coverage(grid)
    if coverage < threshold:
        raise CoverageError(
            f"champion {model.name} covers only {coverage:.1%} of the forecast grid "
            f"(floor {threshold:.0%}). Refusing to publish a partial forecast -- this is "
            "the tripwire for entity-resolution drift between training and serving."
        )

    preds = model.predict(grid[["park_name", "ride_key", "ts_local"]])
    now = datetime.now(timezone.utc)

    out = grid.copy()
    # Whole minutes: sub-minute precision on a queue estimate is false precision.
    out["predicted_wait_min"] = preds["yhat"].round().astype(int)
    out["lower_bound"] = preds["yhat_lower"].round().astype(int)
    out["upper_bound"] = preds["yhat_upper"].round().astype(int)
    out["ts_utc"] = to_utc(out["ts_local"])
    out["model_name"] = model.name
    out["model_version"] = str(model_version)
    out["forecast_week"] = out["ts_local"].dt.to_period("W").astype(str)
    out["generated_at"] = now

    log.info(
        # No "v" prefix: model_version carries whatever identifier the caller registered,
        # which for a Unity Catalog logged model is a `models:/...` URI, not a number.
        "scored %d rows with %s (%s), coverage %.3f, mean predicted wait %.1f min",
        len(out),
        model.name,
        model_version,
        coverage,
        out.predicted_wait_min.mean(),
    )
    return out[PREDICTION_SCHEMA]


def generate_forecast(
    model: WaitTimeModel,
    silver: pd.DataFrame,
    model_version: str = "unknown",
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: active attractions -> grid -> scored forecast."""
    return score_grid(model, build_forecast_grid(active_entities(silver), start=start), model_version)
