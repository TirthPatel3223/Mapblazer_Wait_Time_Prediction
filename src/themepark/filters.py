"""Operating-window and data-quality filters.

The hour bounds below are *park-local* wall-clock hours and are only ever applied to
park-local timestamps. See `themepark.timeutils` for why that distinction matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .timeutils import PARK_TZ, to_park_local


@dataclass(frozen=True)
class ParkConstraint:
    """Operating window for one park, in local wall-clock hours.

    `close_hour` is exclusive. A park closing at 24 means "through 23:59 local".
    `start_date` marks the first local date with trustworthy collection: Six Flags was
    not being scraped reliably before 2026-02-15.
    """

    open_hour: int
    close_hour: int
    start_date: str
    tz: str = PARK_TZ


PARK_CONSTRAINTS: dict[str, ParkConstraint] = {
    "Disneyland": ParkConstraint(open_hour=8, close_hour=24, start_date="2025-12-06"),
    "Disney California Adventure Park": ParkConstraint(8, 22, "2025-12-06"),
    "Universal Studios Hollywood": ParkConstraint(8, 22, "2025-12-06"),
    "SeaWorld San Diego": ParkConstraint(10, 20, "2025-12-06"),
    "Six Flags Magic Mountain": ParkConstraint(10, 21, "2026-02-15"),
}

# Upstream data-quality exclusions.
#   'SeaWorld San Diego Obsolete' is a duplicate of the live SeaWorld park carrying a
#   superseded attraction set under a different tp_id; keeping both double-counts and
#   pollutes the global model's park category.
EXCLUDED_PARKS: frozenset[str] = frozenset({"SeaWorld San Diego Obsolete"})

#   A Six Flags attraction is literally named "0" in the source table.
EXCLUDED_RIDE_NAMES: frozenset[str] = frozenset({"0", ""})

# Sentinel / impossible readings. 999 is the source system's "unknown" marker.
MAX_PLAUSIBLE_WAIT_MIN = 900

# A ride needs this many 30-minute observations before it is worth modelling.
MIN_OBSERVATIONS_PER_RIDE = 100


def apply_operating_filter(
    df: pd.DataFrame,
    ts_utc_col: str = "ts_utc",
    park_col: str = "park_name",
    wait_col: str = "wait_time",
) -> pd.DataFrame:
    """Convert to local time, then keep only rows inside each park's operating window.

    Adds a `ts_local` column. Rows whose park is unknown or excluded are dropped, as are
    implausible wait readings.

    Zero-minute waits are *kept*: during operating hours a zero is a real walk-on, which
    is exactly the state a routing optimiser most wants to find. Only closed-park zeros
    are removed, and they are removed by the time window rather than by thresholding the
    target.
    """
    out = df.copy()
    out[ts_utc_col] = pd.to_datetime(out[ts_utc_col], errors="coerce")
    out = out[out[ts_utc_col].notna()]

    out = out[~out[park_col].isin(EXCLUDED_PARKS)]
    out = out[out[park_col].isin(PARK_CONSTRAINTS)]

    out = out[(out[wait_col] >= 0) & (out[wait_col] < MAX_PLAUSIBLE_WAIT_MIN)]

    out["ts_local"] = to_park_local(out[ts_utc_col])

    local_hour = out["ts_local"].dt.hour
    local_date = out["ts_local"].dt.normalize()

    keep = pd.Series(False, index=out.index)
    for park, c in PARK_CONSTRAINTS.items():
        in_park = out[park_col] == park
        keep |= (
            in_park
            & (local_hour >= c.open_hour)
            & (local_hour < c.close_hour)
            & (local_date >= pd.Timestamp(c.start_date))
        )

    return out[keep].reset_index(drop=True)


def operating_hours(park: str) -> tuple[int, int]:
    """(open_hour, close_hour) in local time. Used to bound the forecast grid."""
    c = PARK_CONSTRAINTS[park]
    return c.open_hour, c.close_hour
