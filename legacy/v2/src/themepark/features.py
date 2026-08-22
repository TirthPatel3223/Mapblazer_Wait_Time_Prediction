"""The single feature builder.

v1 defined `create_features` five separate times across the training, evaluation, KPI and
real-time scripts, with two different input conventions (a `ds` column in some, the
DatetimeIndex in others). Duplication like that is how the training and serving paths
drift apart. There is one implementation now.

Every feature is derived from *park-local* wall time. Feeding this UTC timestamps
reproduces the v1 defect where Saturday evening was labelled Sunday.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

# Feature order is part of the model contract: XGBoost binds by position when handed a
# bare array, so this list must stay stable across training and serving.
TEMPORAL_FEATURES: list[str] = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "is_holiday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURES: list[str] = ["park_name", "ride_key"]

GLOBAL_FEATURES: list[str] = CATEGORICAL_FEATURES + TEMPORAL_FEATURES

_US_HOLIDAYS = holidays.US()


def build_features(df: pd.DataFrame, ts_col: str = "ts_local") -> pd.DataFrame:
    """Attach the 10 temporal features, derived from park-local timestamps.

    Cyclical sin/cos pairs give the trees a continuous representation of hour-of-day and
    month-of-year, so that 23:30 and 00:00 sit next to each other rather than at opposite
    ends of an ordinal axis.
    """
    if ts_col not in df.columns:
        raise KeyError(
            f"build_features expects a '{ts_col}' column of park-local timestamps; "
            f"got columns {list(df.columns)}"
        )

    out = df.copy()
    ts = pd.to_datetime(out[ts_col])

    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError(
            f"'{ts_col}' is timezone-aware. Features must be built from tz-naive "
            "park-local wall time -- use themepark.timeutils.to_park_local first."
        )

    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    # Holiday lookup is the slow part; do it once per distinct date rather than per row.
    dates = ts.dt.normalize()
    unique_dates = dates.drop_duplicates()
    holiday_map = {d: int(d.date() in _US_HOLIDAYS) for d in unique_dates}
    out["is_holiday"] = dates.map(holiday_map).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def is_high_traffic(df: pd.DataFrame) -> pd.Series:
    """Weekend or US public holiday -- the segment the parks actually care about."""
    return (df["is_weekend"] == 1) | (df["is_holiday"] == 1)


def is_peak_hour(df: pd.DataFrame, start: int = 11, end: int = 20) -> pd.Series:
    """Local hours when guests are genuinely queuing, not the opening/closing tails."""
    return (df["hour"] >= start) & (df["hour"] < end)
