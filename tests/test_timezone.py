"""Regression tests for the timezone defect.

The source database stores UTC. v1 never converted, then applied operating windows that
were written in local terms (Disneyland 08:00-24:00) directly to UTC hours, and derived
every calendar feature from UTC as well.

Two distinct failures follow, and both are tested here:

1. The operating filter kept 00:00-15:00 local and discarded the 16:00-23:00 local
   evening peak -- the hours that carry almost all of the queue.
2. `dayofweek` / `is_weekend` were wrong for every evening observation, because after
   16:00 local the UTC date has already rolled over. Sunday evening was labelled Monday
   (weekend flag lost) and Friday evening was labelled Saturday (weekend flag invented).
"""

import pandas as pd
import pytest

from themepark.features import build_features
from themepark.filters import apply_operating_filter
from themepark.timeutils import PARK_TZ, to_park_local, to_utc


def test_utc_to_local_offset():
    """Winter is UTC-8."""
    local = to_park_local(pd.Series([pd.Timestamp("2026-01-04 03:00:00")]))
    assert local.iloc[0] == pd.Timestamp("2026-01-03 19:00:00")
    assert local.dt.tz is None, "features need tz-naive wall time; Prophet rejects tz-aware ds"


def test_dst_offset_changes():
    """Summer is UTC-7. A fixed offset would silently skew half the year by an hour."""
    winter = to_park_local(pd.Series([pd.Timestamp("2026-01-15 20:00:00")])).iloc[0]
    summer = to_park_local(pd.Series([pd.Timestamp("2026-07-15 20:00:00")])).iloc[0]
    assert winter.hour == 12
    assert summer.hour == 13


def test_roundtrip_local_to_utc():
    original = pd.Series([pd.Timestamp("2026-03-15 14:30:00"), pd.Timestamp("2026-08-02 21:00:00")])
    assert list(to_utc(to_park_local(original))) == list(original)


@pytest.mark.parametrize(
    "local_ts,expected_dow,expected_weekend,utc_dow_v1",
    [
        # Sunday evening: v1 read it as Monday and threw the weekend flag away.
        ("2026-01-04 20:00:00", 6, 1, 0),
        # Friday evening: v1 read it as Saturday and invented a weekend flag.
        ("2026-01-02 20:00:00", 4, 0, 5),
        # Saturday evening: right day-of-week only by luck of the weekend pair.
        ("2026-01-03 20:00:00", 5, 1, 6),
        # Midday is unaffected -- the UTC date has not rolled over yet.
        ("2026-01-04 11:00:00", 6, 1, 6),
    ],
)
def test_weekend_flag_is_correct_in_local_time(local_ts, expected_dow, expected_weekend, utc_dow_v1):
    local = pd.Timestamp(local_ts)
    feats = build_features(pd.DataFrame({"ts_local": [local]}))

    assert feats["dayofweek"].iloc[0] == expected_dow
    assert feats["is_weekend"].iloc[0] == expected_weekend

    # Confirm the premise: building from UTC really would have given a different answer.
    utc = local.tz_localize(PARK_TZ).tz_convert("UTC").tz_localize(None)
    assert utc.dayofweek == utc_dow_v1


def test_build_features_rejects_tz_aware_input():
    """Fail loudly rather than silently producing offset-shifted hours."""
    aware = pd.DataFrame({"ts_local": pd.to_datetime(["2026-01-04 20:00:00"]).tz_localize("UTC")})
    with pytest.raises(ValueError, match="timezone-aware"):
        build_features(aware)


def test_build_features_requires_the_named_column():
    with pytest.raises(KeyError):
        build_features(pd.DataFrame({"wrong": [pd.Timestamp("2026-01-01")]}))


def test_evening_peak_survives_the_operating_filter():
    """The regression that matters most.

    19:00 local at Disneyland is the busiest hour of the day. In UTC it is 03:00 the next
    morning, so v1's `hour >= 8` test dropped it outright.
    """
    peak_utc = pd.Timestamp("2026-01-04 03:00:00")  # 19:00 local, Saturday
    df = pd.DataFrame(
        {"ts_utc": [peak_utc], "park_name": ["Disneyland"], "wait_time": [55], "ride_name": ["X"]}
    )

    kept = apply_operating_filter(df)

    assert len(kept) == 1, "the evening peak must survive the operating-hours filter"
    assert kept["ts_local"].iloc[0] == pd.Timestamp("2026-01-03 19:00:00")
    assert peak_utc.hour < 8, "premise: v1's UTC hour>=8 test dropped this row"


def test_closed_park_overnight_rows_are_removed():
    """03:00 local is genuinely closed. Those zeros are artefacts, not walk-ons."""
    closed_local_utc = pd.Timestamp("2026-01-04 11:00:00")  # 03:00 local
    df = pd.DataFrame(
        {"ts_utc": [closed_local_utc], "park_name": ["Disneyland"], "wait_time": [0], "ride_name": ["X"]}
    )
    assert len(apply_operating_filter(df)) == 0


def test_walk_on_zeros_during_operating_hours_are_kept():
    """A zero at 14:00 local is a real walk-on and the most useful signal a router has."""
    df = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2026-01-03 22:00:00")],  # 14:00 local
            "park_name": ["Disneyland"],
            "wait_time": [0],
            "ride_name": ["X"],
        }
    )
    assert len(apply_operating_filter(df)) == 1


def test_sentinel_and_excluded_rows_are_dropped():
    df = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2026-01-03 22:00:00")] * 3,
            "park_name": ["Disneyland", "Disneyland", "SeaWorld San Diego Obsolete"],
            "wait_time": [999, 20, 20],
            "ride_name": ["A", "B", "C"],
        }
    )
    kept = apply_operating_filter(df)
    assert list(kept["ride_name"]) == ["B"]


def test_six_flags_start_date_is_enforced():
    """Six Flags collection was unreliable before 2026-02-15."""
    df = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2026-01-20 22:00:00"), pd.Timestamp("2026-03-20 22:00:00")],
            "park_name": ["Six Flags Magic Mountain"] * 2,
            "wait_time": [15, 15],
            "ride_name": ["X", "X"],
        }
    )
    kept = apply_operating_filter(df)
    assert len(kept) == 1
    assert kept["ts_local"].iloc[0].month == 3
