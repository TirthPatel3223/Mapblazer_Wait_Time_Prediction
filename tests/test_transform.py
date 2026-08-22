"""Unit tests for the bronze -> silver transform and the quality checks."""

from __future__ import annotations

import pandas as pd
import pytest

import pipeline


class TestCanonicalKey:
    def test_punctuation_stripped(self):
        assert (
            pipeline.canonical_key("Guardians of the Galaxy - Mission: BREAKOUT!")
            == "Guardians_of_the_Galaxy_-_Mission_BREAKOUT"
        )

    def test_apostrophe(self):
        assert pipeline.canonical_key("Soarin' Around the World") == "Soarin_Around_the_World"

    def test_whitespace_collapsed(self):
        assert pipeline.canonical_key("  A   B  ") == "A_B"

    def test_none_and_empty(self):
        assert pipeline.canonical_key(None) == ""
        assert pipeline.canonical_key("") == ""


class TestTimezone:
    def test_utc_to_local_winter(self):
        # 04:00 UTC Sunday is 20:00 PST Saturday: the evening-peak row the v1 defect
        # deleted.
        local = pipeline.to_local(pd.Series([pd.Timestamp("2026-01-11 04:00:00")]))
        assert local.iloc[0] == pd.Timestamp("2026-01-10 20:00:00")
        assert local.iloc[0].dayofweek == 5  # Saturday

    def test_utc_to_local_summer(self):
        local = pipeline.to_local(pd.Series([pd.Timestamp("2026-07-05 03:00:00")]))
        assert local.iloc[0] == pd.Timestamp("2026-07-04 20:00:00")

    def test_roundtrip(self):
        wall = pd.Series(pd.to_datetime(["2026-01-10 20:00:00", "2026-07-04 12:30:00"]))
        assert (pipeline.to_local(pipeline.to_utc(wall)) == wall).all()


def _raw_row(wait_id, wait, ts_utc, ride="Alpha Coaster", park="Disneyland"):
    return {
        "wait_time_id": wait_id,
        "wait_time": wait,
        "ts_utc": pd.Timestamp(ts_utc),
        "ride_name": ride,
        "park_name": park,
    }


class TestBuildSilver:
    @pytest.fixture(autouse=True)
    def _small_thresholds(self, monkeypatch):
        monkeypatch.setattr(pipeline, "MIN_OBS_PER_RIDE", 1)

    def test_convert_then_filter_order(self):
        raw = pd.DataFrame(
            [
                # 20:00 local Saturday == 04:00 UTC Sunday. A UTC-hour filter would
                # wrongly drop this row; the local-hour filter must keep it.
                _raw_row(1, 25, "2026-01-11 04:00:00"),
                # 03:00 local == 11:00 UTC. A UTC-hour filter would wrongly keep this
                # closed-park row; the local-hour filter must drop it.
                _raw_row(2, 0, "2026-01-10 11:00:00"),
            ]
        )
        silver = pipeline.build_silver(raw)
        assert len(silver) == 1
        assert silver["ts_local"].iloc[0] == pd.Timestamp("2026-01-10 20:00:00")
        assert silver["ts_utc"].iloc[0] == pd.Timestamp("2026-01-11 04:00:00")

    def test_bad_rows_dropped(self):
        good = _raw_row(1, 25, "2026-01-10 20:00:00")  # 12:00 local
        raw = pd.DataFrame(
            [
                good,
                _raw_row(2, -5, "2026-01-10 20:00:00"),
                _raw_row(3, 999, "2026-01-10 20:00:00"),
                _raw_row(4, 10, "2026-01-10 20:00:00", ride="0"),
                _raw_row(5, 10, "2026-01-10 20:00:00", park="SeaWorld San Diego Obsolete"),
                _raw_row(6, 10, "2026-01-10 20:00:00", park="Some Unknown Park"),
                dict(good, wait_time=999999),  # duplicate wait_time_id 1
            ]
        )
        silver = pipeline.build_silver(raw)
        assert len(silver) == 1
        assert silver["wait_time"].iloc[0] == 25.0

    def test_grid_and_canonical_key(self):
        # Two raw readings inside one 30-minute slot are averaged; punctuated names
        # canonicalise.
        raw = pd.DataFrame(
            [
                _raw_row(1, 10, "2026-01-10 20:05:00", ride="Soarin' Around the World"),
                _raw_row(2, 20, "2026-01-10 20:20:00", ride="Soarin' Around the World"),
            ]
        )
        silver = pipeline.build_silver(raw)
        assert len(silver) == 1
        assert silver["ride_key"].iloc[0] == "Soarin_Around_the_World"
        assert silver["wait_time"].iloc[0] == 15.0

    def test_min_observations_filter(self, monkeypatch):
        monkeypatch.setattr(pipeline, "MIN_OBS_PER_RIDE", 3)
        raw = pd.DataFrame(
            [_raw_row(i, 10, f"2026-01-10 {18 + i}:00:00") for i in range(2)]
        )
        assert pipeline.build_silver(raw).empty


class TestChronoSplit:
    def test_split_is_chronological(self):
        ts = pd.date_range("2026-01-01 12:00", periods=100, freq="h")
        silver = pd.DataFrame({"ts_local": ts, "wait_time": range(100)})
        train, test = pipeline.chrono_split(silver)
        assert 78 <= len(train) <= 82
        assert train["ts_local"].max() < test["ts_local"].min()


def _silver_frame(hours, waits, park="Disneyland", ride="Alpha"):
    ts = [pd.Timestamp(f"2026-01-{5 + i % 20:02d} {h:02d}:00:00") for i, h in enumerate(hours)]
    return pd.DataFrame(
        {
            "park_name": park,
            "ride_key": ride,
            "ride_name": ride,
            "ts_local": ts,
            "ts_utc": ts,
            "wait_time": waits,
        }
    )


class TestSilverChecks:
    def test_clean_data_passes_tz_check(self):
        hours = list(range(8, 24)) * 20
        waits = [30.0 if 13 <= h <= 15 else 5.0 for h in hours]
        problems = pipeline.check_silver(_silver_frame(hours, waits), None)
        assert not any("inverted" in p for p in problems)

    def test_inverted_timezone_detected(self):
        # Inverted data relocates the afternoon peak to the late evening hours.
        hours = list(range(8, 24)) * 20
        waits = [30.0 if h >= 22 else 1.0 for h in hours]
        problems = pipeline.check_silver(_silver_frame(hours, waits), None)
        assert any("inverted" in p for p in problems)

    def test_zero_share_detected(self):
        hours = list(range(8, 24)) * 20
        waits = [0.0] * len(hours)
        problems = pipeline.check_silver(_silver_frame(hours, waits), None)
        assert any("zero-wait share" in p for p in problems)

    def test_row_collapse_detected(self):
        hours = list(range(8, 24)) * 20
        waits = [10.0] * len(hours)
        frame = _silver_frame(hours, waits)
        problems = pipeline.check_silver(frame, last_row_count=len(frame) * 3)
        assert any("collapsed" in p for p in problems)

    def test_future_timestamps_detected(self):
        frame = _silver_frame([12, 13, 14], [5.0, 6.0, 7.0])
        frame.loc[0, "ts_local"] = pipeline.now_local() + pd.Timedelta(days=30)
        problems = pipeline.check_silver(frame, None)
        assert any("future" in p for p in problems)


class TestPredictionChecks:
    def _forecast_frame(self, n=10):
        ts = pd.date_range("2026-08-23 12:00", periods=n, freq="30min")
        return pd.DataFrame(
            {
                "row_kind": "forecast",
                "park_name": "Disneyland",
                "ride_key": "Alpha",
                "ride_name": "Alpha",
                "ts_local": ts,
                "ts_utc": ts,
                "predicted_wait_min": 10.0,
                "lower_bound": 5.0,
                "upper_bound": 15.0,
                "actual_wait_min": float("nan"),
                "error_min": float("nan"),
                "model_name": "prophet_fleet",
            }
        )

    def test_negative_prediction_detected(self):
        frame = self._forecast_frame()
        frame.loc[0, "predicted_wait_min"] = -1.0
        problems = pipeline.check_predictions(frame, n_active_entities=1)
        assert any("negative" in p for p in problems)

    def test_bound_ordering_detected(self):
        frame = self._forecast_frame()
        frame.loc[0, "lower_bound"] = 99.0
        problems = pipeline.check_predictions(frame, n_active_entities=1)
        assert any("lower <= predicted <= upper" in p for p in problems)

    def test_coverage_detected(self):
        frame = self._forecast_frame()
        problems = pipeline.check_predictions(frame, n_active_entities=10)
        assert any("active rides" in p for p in problems)

    def test_null_detected(self):
        frame = self._forecast_frame()
        frame.loc[0, "predicted_wait_min"] = float("nan")
        problems = pipeline.check_predictions(frame, n_active_entities=1)
        assert any("nulls in forecast.predicted_wait_min" in p for p in problems)
