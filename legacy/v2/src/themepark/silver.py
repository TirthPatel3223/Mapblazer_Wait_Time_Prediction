"""Bronze -> silver: the one transform that defines what "clean" means.

Kept in the shared package rather than inside a Databricks notebook so that the local dry
run, the CI tests and the scheduled job all produce byte-identical silver. A notebook-only
transform is how training and serving quietly diverge.
"""

from __future__ import annotations

import logging

import pandas as pd

from .filters import EXCLUDED_RIDE_NAMES, MIN_OBSERVATIONS_PER_RIDE, apply_operating_filter
from .naming import canonical_ride_key

log = logging.getLogger(__name__)

SILVER_SCHEMA = ["park_name", "ride_key", "ride_name", "ts_local", "ts_utc", "wait_time"]


def build_silver(
    raw: pd.DataFrame,
    grid_minutes: int = 30,
    min_observations: int = MIN_OBSERVATIONS_PER_RIDE,
) -> pd.DataFrame:
    """Clean, timezone-correct, gridded observations ready for modelling.

    Steps, in order, because the order matters:

    1. Drop duplicate `wait_time_id` -- the collector is deliberately idempotent and
       overlapping batches are normal.
    2. Drop known-bad attraction names (the source table contains one literally named "0").
    3. Convert UTC to park-local, *then* apply operating-hour windows. Doing this the other
       way round is the defect that cost v1 the entire evening peak.
    4. Resample each attraction onto a fixed 30-minute grid. The raw feed arrives every
       25-30 minutes at irregular offsets; a fixed grid is what makes predictions joinable
       back to actuals during verification.
    5. Drop attractions with too little history to model.
    """
    work = raw.copy()
    start_rows = len(work)

    if "wait_time_id" in work.columns:
        work = work.drop_duplicates(subset=["wait_time_id"])

    work["ride_name"] = work["ride_name"].astype(str).str.strip()
    work = work[~work["ride_name"].isin(EXCLUDED_RIDE_NAMES)]

    # Timezone conversion plus operating-hour and sentinel filtering. Adds `ts_local`.
    work = apply_operating_filter(work)
    if work.empty:
        raise ValueError("operating filter removed every row; check timestamps and park names")

    work["ride_key"] = work["ride_name"].map(canonical_ride_key)
    work = work[work["ride_key"] != ""]

    gridded = (
        work.set_index("ts_local")
        .groupby(["park_name", "ride_key", "ride_name"])["wait_time"]
        .resample(f"{grid_minutes}min")
        .mean()
        .dropna()
        .reset_index()
    )

    counts = gridded.groupby(["park_name", "ride_key"])["wait_time"].transform("size")
    gridded = gridded[counts >= min_observations].reset_index(drop=True)

    # Recovered from local time so the two columns can never disagree.
    from .timeutils import to_utc

    gridded["ts_utc"] = to_utc(gridded["ts_local"])

    log.info(
        "silver: %d raw rows -> %d gridded rows, %d attractions, %s to %s local",
        start_rows,
        len(gridded),
        gridded.groupby(["park_name", "ride_key"]).ngroups,
        gridded.ts_local.min(),
        gridded.ts_local.max(),
    )
    return gridded[SILVER_SCHEMA]


def silver_quality_report(silver: pd.DataFrame) -> dict:
    """Numbers worth asserting on before a training run consumes this table."""
    return {
        "rows": len(silver),
        "entities": int(silver.groupby(["park_name", "ride_key"]).ngroups),
        "parks": int(silver["park_name"].nunique()),
        "start": str(silver["ts_local"].min()),
        "end": str(silver["ts_local"].max()),
        "mean_wait": float(silver["wait_time"].mean()),
        "pct_zero": float((silver["wait_time"] == 0).mean() * 100),
        "p95_wait": float(silver["wait_time"].quantile(0.95)),
        "max_wait": float(silver["wait_time"].max()),
    }
