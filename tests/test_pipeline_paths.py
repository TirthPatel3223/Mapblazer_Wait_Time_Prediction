"""Integration tests over the full orchestration with a fake lake.

These prove the safety nets actually trip:
  - a failed quality check drops _current and leaves _last byte-identical
  - any failure still refreshes the forecast window via the previous champion
  - a first-ever failure with nothing to fall back to changes nothing and says so
  - a failed run leaves champion.json untouched and its run directory orphaned
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest
from conftest import make_bronze

import pipeline

DATA_COLS = [
    "row_kind", "park_name", "ride_key", "ride_name", "ts_local", "ts_utc",
    "predicted_wait_min", "lower_bound", "upper_bound", "actual_wait_min",
    "error_min", "model_name",
]


def read_pointer(models_root):
    return json.loads((models_root / "champion.json").read_text(encoding="utf-8"))


def sorted_backtest(preds):
    bt = preds[preds["row_kind"] == "backtest"]
    return bt[DATA_COLS].sort_values(["model_name", "ride_key", "ts_local"]).reset_index(drop=True)


class TestFirstRunSuccess:
    def test_summary_and_tables(self, env, base_state):
        summary = base_state["summary"]
        assert summary["run_status"] == "fresh_model"
        for table in (pipeline.SILVER_TABLE, pipeline.KPI_TABLE, pipeline.PRED_TABLE):
            assert env.exists(table + pipeline.LAST)
            # _current is deliberately left in place after a successful promotion.
            assert env.exists(table + pipeline.CURRENT)

    def test_champion_pointer_and_run_dir(self, env, base_state, tmp_path):
        pointer = read_pointer(tmp_path / "models")
        assert pointer["model_name"] in pipeline.FAMILY_CLASSES
        run_dir = tmp_path / "models" / "runs" / pointer["run_id"]
        assert (run_dir / "manifest.json").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["ride_count"] == len(manifest["entities"]) == pointer["ride_count"]

    def test_gold_contents(self, env):
        preds = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        assert set(preds["row_kind"].unique()) == {"forecast", "backtest"}
        fc = preds[preds["row_kind"] == "forecast"]
        assert (fc["lower_bound"] <= fc["predicted_wait_min"]).all()
        assert (fc["predicted_wait_min"] <= fc["upper_bound"]).all()
        assert (fc["predicted_wait_min"] >= 0).all()

        # The forecast is the champion's alone; the backtest carries every candidate
        # so the dashboard can compare their test-set error distributions.
        assert fc["model_name"].nunique() == 1
        bt = preds[preds["row_kind"] == "backtest"]
        assert set(bt["model_name"].unique()) == set(pipeline.CANDIDATES)

        kpis = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        assert set(kpis["model"].unique()) == {
            pipeline.MODEL_PROPHET,
            pipeline.MODEL_XGB_GLOBAL,
            pipeline.MODEL_XGB_LOCAL,
            pipeline.MODEL_BASELINE,
        }
        assert kpis[kpis["is_champion"]]["model"].nunique() == 1

    def test_gold_tables_agree_with_pointer(self, env, tmp_path):
        pointer = read_pointer(tmp_path / "models")
        preds = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        kpis = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        assert (preds["run_id"] == pointer["run_id"]).all()
        assert (kpis["run_id"] == pointer["run_id"]).all()


class TestQualityFailure:
    def test_last_survives_and_forecast_refreshed(self, env, tmp_path):
        models_root = tmp_path / "models"
        silver_before = env.read(pipeline.SILVER_TABLE + pipeline.LAST)
        kpis_before = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        preds_before = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        pointer_before = read_pointer(models_root)

        # All-zero waits trip the zero-share check (and the tz peak check) in silver.
        env.write(pipeline.BRONZE_TABLE, make_bronze(seed=2, all_zero=True))
        with pytest.raises(pipeline.QualityCheckError, match="zero-wait share"):
            pipeline.main()

        # _current dropped, _last byte-identical for silver and KPIs.
        for table in (pipeline.SILVER_TABLE, pipeline.KPI_TABLE, pipeline.PRED_TABLE):
            assert not env.exists(table + pipeline.CURRENT)
        pd.testing.assert_frame_equal(
            env.read(pipeline.SILVER_TABLE + pipeline.LAST), silver_before
        )
        pd.testing.assert_frame_equal(env.read(pipeline.KPI_TABLE + pipeline.LAST), kpis_before)

        # The forecast window was refreshed by the previous champion; backtest rows
        # are untouched.
        preds_after = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        fc = preds_after[preds_after["row_kind"] == "forecast"]
        assert (fc["run_status"] == "fallback_after_failure").all()
        assert (fc["run_id"] == pointer_before["run_id"]).all()
        assert fc["generated_at"].iloc[0] > preds_before["generated_at"].iloc[0]
        pd.testing.assert_frame_equal(
            sorted_backtest(preds_after), sorted_backtest(preds_before)
        )

        # champion.json points exactly where it did before.
        assert read_pointer(models_root) == pointer_before

    def test_gold_failure_orphans_run_dir(self, env, tmp_path, monkeypatch):
        models_root = tmp_path / "models"
        pointer_before = read_pointer(models_root)
        kpis_before = env.read(pipeline.KPI_TABLE + pipeline.LAST)

        env.write(pipeline.BRONZE_TABLE, make_bronze(seed=3))
        monkeypatch.setattr(pipeline, "check_kpis", lambda df: ["forced gold failure"])
        with pytest.raises(pipeline.QualityCheckError, match="forced gold failure"):
            pipeline.main()

        # The failed run persisted its artifacts, but its directory is orphaned:
        # champion.json still points at the old run, and the fallback served the old
        # model (run_id proves which artifacts were loaded).
        run_dirs = {p.name for p in (models_root / "runs").iterdir() if p.is_dir()}
        assert pointer_before["run_id"] in run_dirs
        assert len(run_dirs) == 2
        assert read_pointer(models_root) == pointer_before

        preds_after = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        fc = preds_after[preds_after["row_kind"] == "forecast"]
        assert (fc["run_id"] == pointer_before["run_id"]).all()
        assert (fc["run_status"] == "fallback_after_failure").all()
        pd.testing.assert_frame_equal(env.read(pipeline.KPI_TABLE + pipeline.LAST), kpis_before)


class TestKeptPreviousModel:
    def test_gate_keeps_incumbent_and_rescores(self, env, tmp_path):
        models_root = tmp_path / "models"
        pointer_before = read_pointer(models_root)

        # Doctor the incumbent's recorded MAE so no retrain can beat it by 1 percent.
        kpis = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        mask = kpis["is_champion"] & (kpis["kpi_name"] == "mae")
        kpis.loc[mask, "kpi_value"] = 0.5
        env.write(pipeline.KPI_TABLE + pipeline.LAST, kpis)

        env.write(pipeline.BRONZE_TABLE, make_bronze(seed=4))
        summary = pipeline.main()

        assert summary["run_status"] == "kept_previous_model"
        assert summary["champion"] == pointer_before["model_name"]
        # No new artifacts were persisted and the pointer did not move.
        run_dirs = {p.name for p in (models_root / "runs").iterdir() if p.is_dir()}
        assert run_dirs == {pointer_before["run_id"]}
        assert read_pointer(models_root) == pointer_before

        # Previous KPI values kept serving (the doctored 0.5 survived promotion).
        kpis_after = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        mask = kpis_after["is_champion"] & (kpis_after["kpi_name"] == "mae")
        assert (kpis_after.loc[mask, "kpi_value"] == 0.5).all()
        assert (kpis_after["run_status"] == "kept_previous_model").all()

        preds_after = env.read(pipeline.PRED_TABLE + pipeline.LAST)
        fc = preds_after[preds_after["row_kind"] == "forecast"]
        assert (fc["model_name"] == pointer_before["model_name"]).all()
        assert (fc["run_id"] == pointer_before["run_id"]).all()


class TestNothingToFallBackTo:
    def test_first_run_failure_changes_nothing(self, empty_env, tmp_path, caplog):
        empty_env.write(pipeline.BRONZE_TABLE, make_bronze(seed=5, all_zero=True))
        with caplog.at_level(logging.WARNING):
            with pytest.raises(pipeline.QualityCheckError):
                pipeline.main()
        # Nothing was created: no silver/gold tables, no models, no pointer.
        assert set(empty_env.tables) == {pipeline.BRONZE_TABLE}
        assert not (tmp_path / "models" / "champion.json").exists()
        assert not (tmp_path / "models" / "runs").exists()
        assert "nothing to fall back to" in caplog.text


class TestPointerMismatch:
    def test_mismatch_detected_and_nothing_changed(self, env, tmp_path, caplog):
        # Simulate a crash between table promotion and the pointer write: the tables
        # carry a run_id champion.json has never heard of.
        kpis = env.read(pipeline.KPI_TABLE + pipeline.LAST)
        kpis["run_id"] = "19990101T000000Z"
        env.write(pipeline.KPI_TABLE + pipeline.LAST, kpis)
        preds_before = env.read(pipeline.PRED_TABLE + pipeline.LAST)

        env.write(pipeline.BRONZE_TABLE, make_bronze(seed=6, all_zero=True))
        with caplog.at_level(logging.ERROR):
            with pytest.raises(pipeline.QualityCheckError):
                pipeline.main()

        assert "disagrees with champion.json" in caplog.text
        # The fallback changed NOTHING: serving predictions are exactly as they were.
        pd.testing.assert_frame_equal(
            env.read(pipeline.PRED_TABLE + pipeline.LAST), preds_before
        )
