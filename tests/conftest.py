"""Shared test fixtures.

The pipeline talks to Spark through six module-level functions (read_table,
write_table, drop_table, table_exists, table_count, promote_table). Substituting a
dict-backed FakeLake for those six exercises the entire orchestration -- including the
failure paths that would cost a metered Databricks run to discover -- locally.
"""

from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402

_RUN_COUNTER = itertools.count(1)


class FakeLake:
    """In-memory stand-in for Unity Catalog tables."""

    def __init__(self):
        self.tables: dict[str, pd.DataFrame] = {}

    def read(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"table not found: {name}")
        return self.tables[name].copy()

    def write(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df.copy()

    def drop(self, name: str) -> None:
        self.tables.pop(name, None)

    def exists(self, name: str) -> bool:
        return name in self.tables

    def count(self, name: str) -> int:
        return len(self.tables[name])

    def promote(self, src: str, dst: str) -> None:
        self.tables[dst] = self.tables[src].copy()

    def snapshot(self) -> dict[str, pd.DataFrame]:
        return {k: v.copy() for k, v in self.tables.items()}


def patch_pipeline(mp: pytest.MonkeyPatch, lake: FakeLake, models_root: Path) -> None:
    mp.setattr(pipeline, "read_table", lake.read)
    mp.setattr(pipeline, "write_table", lake.write)
    mp.setattr(pipeline, "drop_table", lake.drop)
    mp.setattr(pipeline, "table_exists", lake.exists)
    mp.setattr(pipeline, "table_count", lake.count)
    mp.setattr(pipeline, "promote_table", lake.promote)
    mp.setattr(pipeline, "ensure_workspace", lambda: None)
    mp.setattr(pipeline, "MODELS_ROOT", models_root)
    # Synthetic data has 3 attractions; the production threshold is 50.
    mp.setattr(pipeline, "MIN_ATTRACTIONS", 3)
    # make_run_id has one-second resolution; tests run several pipelines per second.
    mp.setattr(pipeline, "make_run_id", lambda: f"20260822T{next(_RUN_COUNTER):06d}Z")


RIDES = ("Alpha Coaster", "Beta's Flight!", "Gamma: The Ride")


def make_bronze(
    seed: int = 1,
    days: int = 21,
    park: str = "Disneyland",
    rides: tuple[str, ...] = RIDES,
    start: str = "2026-01-05",
    all_zero: bool = False,
) -> pd.DataFrame:
    """Synthetic bronze rows on a 30-minute local grid, 08:00-23:30, with an afternoon
    peak around 14:30 so the timezone quality check has a real signal."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    slots = pd.date_range(start_ts + pd.Timedelta(hours=8),
                          start_ts + pd.Timedelta(days=days), freq="30min")
    slots = slots[(slots.hour >= 8)]
    rows = []
    wait_id = itertools.count(seed * 10_000_000 + 1)
    for ride_i, ride in enumerate(rides):
        hour_f = np.asarray(slots.hour, dtype=float) + np.asarray(slots.minute, dtype=float) / 60.0
        base = 2.0 + 22.0 * np.exp(-((hour_f - 14.5) ** 2) / 18.0)
        weekend = np.isin(slots.dayofweek, [5, 6]).astype(float) * 6.0
        noise = rng.normal(0, 3, len(slots))
        wait = np.clip(base + weekend + noise + ride_i * 2.0, 0, None)
        wait[wait < 4.0] = 0.0
        if all_zero:
            wait[:] = 0.0
        ts_utc = (
            pd.Series(slots)
            .dt.tz_localize(pipeline.PARK_TZ, ambiguous=True, nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        rows.append(
            pd.DataFrame(
                {
                    "wait_time_id": [next(wait_id) for _ in range(len(slots))],
                    "wait_time": wait.round().astype("int64"),
                    "ts_utc": ts_utc,
                    "at_id": ride_i + 1,
                    "ride_name": ride,
                    "tp_id": 1,
                    "park_name": park,
                    "_ingested_at": pd.Timestamp("2026-08-01 00:00:00"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def base_state(tmp_path_factory):
    """One real end-to-end successful run (trains actual Prophet and XGBoost models),
    snapshotted so each test starts from the same post-first-run world."""
    mp = pytest.MonkeyPatch()
    lake = FakeLake()
    models_root = tmp_path_factory.mktemp("base_models")
    patch_pipeline(mp, lake, models_root)
    lake.write(pipeline.BRONZE_TABLE, make_bronze(seed=1))
    try:
        summary = pipeline.main()
    finally:
        mp.undo()
    return {"tables": lake.snapshot(), "models_root": models_root, "summary": summary}


@pytest.fixture
def env(base_state, monkeypatch, tmp_path):
    """A fresh, isolated copy of the post-first-run world."""
    lake = FakeLake()
    lake.tables = {k: v.copy() for k, v in base_state["tables"].items()}
    models_root = tmp_path / "models"
    shutil.copytree(base_state["models_root"], models_root)
    patch_pipeline(monkeypatch, lake, models_root)
    return lake


@pytest.fixture
def empty_env(monkeypatch, tmp_path):
    """A world with nothing in it but bronze -- the true first-run scenario."""
    lake = FakeLake()
    models_root = tmp_path / "models"
    patch_pipeline(monkeypatch, lake, models_root)
    return lake
