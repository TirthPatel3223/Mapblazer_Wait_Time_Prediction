"""Uploaded Parquet must be readable by Spark.

pandas holds datetimes as `datetime64[ns]` and writes Parquet `TIMESTAMP(NANOS)`. Spark
cannot read that type at all: `[PARQUET_TYPE_ILLEGAL] Illegal Parquet type: INT64
(TIMESTAMP(NANOS,false))`. The failure surfaces only once a Databricks job reads the file,
long after the upload reported success -- so it is checked here, on the writing side.
"""

import io

import pandas as pd
import pyarrow.parquet as pq
import pytest

from themepark.config import DatabricksSettings
from themepark.sources import Databricks

# Spark reads MILLIS and MICROS. NANOS is rejected outright.
SPARK_READABLE_UNITS = {"ms", "us"}


class CapturingLake(Databricks):
    """Databricks, with the network call replaced by a buffer."""

    def __init__(self):
        super().__init__(DatabricksSettings(host="https://x.databricks.com", token="t"))
        self.written = {}

    def _workspace(self):
        lake = self

        class Files:
            def upload(self, path, buffer, overwrite=False):
                lake.written[path] = buffer.read()

        class W:
            files = Files()

        return W()


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "wait_time_id": [1, 2, 3],
            "wait_time": [0, 15, 45],
            "ts_utc": pd.to_datetime(
                ["2026-01-03 20:00:00", "2026-01-03 20:30:00", "2026-01-03 21:00:00"]
            ),
            "ride_name": ["Rise of the Resistance", "Peter Pan's Flight", "Soarin'"],
        }
    )


def schema_of(lake):
    (payload,) = lake.written.values()
    return pq.read_schema(io.BytesIO(payload))


def test_pandas_default_would_have_been_unreadable(frame):
    """Documents the defect: this is what the old code wrote."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    assert pq.read_schema(buffer).field("ts_utc").type.unit == "ns"


def test_uploaded_timestamps_are_spark_readable(frame):
    lake = CapturingLake()
    lake.upload_parquet(frame, "part-000.parquet")

    unit = schema_of(lake).field("ts_utc").type.unit
    assert unit in SPARK_READABLE_UNITS, f"Spark cannot read TIMESTAMP({unit.upper()})"


def test_every_datetime_column_is_coerced(frame):
    """A second timestamp column must not slip through."""
    frame["ingested_at"] = pd.to_datetime(["2026-01-04 01:00:00"] * 3)
    lake = CapturingLake()
    lake.upload_parquet(frame, "part-000.parquet")

    schema = schema_of(lake)
    for name in ("ts_utc", "ingested_at"):
        assert schema.field(name).type.unit in SPARK_READABLE_UNITS


def test_timestamp_values_survive_the_downcast(frame):
    """Minute-granularity data loses nothing going from nanoseconds to microseconds."""
    lake = CapturingLake()
    lake.upload_parquet(frame, "part-000.parquet")

    (payload,) = lake.written.values()
    roundtripped = pq.read_table(io.BytesIO(payload)).to_pandas()
    pd.testing.assert_series_equal(
        roundtripped["ts_utc"].astype("datetime64[ns]"), frame["ts_utc"], check_names=False
    )


def test_rows_and_columns_are_unchanged(frame):
    lake = CapturingLake()
    lake.upload_parquet(frame, "part-000.parquet")

    (payload,) = lake.written.values()
    roundtripped = pq.read_table(io.BytesIO(payload)).to_pandas()
    assert list(roundtripped.columns) == list(frame.columns)
    assert len(roundtripped) == len(frame)
