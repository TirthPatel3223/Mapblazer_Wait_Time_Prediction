"""Timezone handling.

The source database stores `wait_time_upd_dt` in UTC. Every park in the dataset is in
America/Los_Angeles. The v1 pipeline never converted, then applied operating-hour
windows that were plainly written in local terms (Disneyland 08:00-24:00) directly to
UTC hours. Because the west coast is UTC-8/-7, "keep UTC hours 8 through 23" actually
kept 00:00-15:00 local -- seven hours of guaranteed closed-park zeros -- and threw away
the entire 16:00-23:00 local evening peak.

The same defect poisoned the features: `dayofweek` and `is_weekend` were derived from UTC,
so a Saturday 20:00 local observation was labelled Sunday. That is the single most
predictive feature in the model, mislabelled at exactly the hours that matter.

Everything downstream of this module works in park-local wall time.
"""

from __future__ import annotations

import pandas as pd

PARK_TZ = "America/Los_Angeles"


def to_park_local(values, tz: str = PARK_TZ) -> pd.Series:
    """Convert UTC timestamps to tz-naive park-local wall time.

    Naive input is assumed UTC, matching the source database. The result is deliberately
    tz-naive: Prophet rejects tz-aware `ds`, and downstream calendar features want wall
    time, not an offset. DST transitions are handled by the tz database, so a spring-
    forward day correctly has 23 local hours.
    """
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def to_utc(values, tz: str = PARK_TZ) -> pd.Series:
    """Inverse of `to_park_local`: tz-naive park-local wall time back to tz-naive UTC.

    Used when writing forecasts, which are generated on a local-time grid but must also
    carry a UTC column so downstream consumers never have to guess.

    Ambiguous local times (the repeated hour at the autumn DST fallback) resolve to the
    first, i.e. still-DST, occurrence; nonexistent local times (the skipped hour in
    spring) shift forward. Neither is meaningful at 30-minute forecast granularity, but
    silently raising in a scheduled job would be worse than either.
    """
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_convert(tz).dt.tz_localize(None)
    localized = s.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)
