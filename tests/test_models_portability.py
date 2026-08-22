"""Model artifacts are plain files that must reload into identical predictions.

The previous system pickled repo classes into MLflow models; the artifact was then only
loadable by a process that could import the repo, which the Databricks job could not.
These tests prove each family round-trips through its plain-file form.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_bronze

import pipeline


@pytest.fixture(scope="module")
def small_split():
    silver = pipeline.build_silver(make_bronze(seed=11, days=10, rides=("Alpha Coaster", "Beta's Flight!")))
    return pipeline.chrono_split(silver)


class TestXGBGlobalRoundTrip:
    def test_save_load_predict_identical(self, small_split, tmp_path):
        train, test = small_split
        model = pipeline.XGBGlobal.train(train)
        before = model.predict(test)
        model.save(tmp_path)
        manifest = {"trained_at": model.trained_at}
        loaded = pipeline.XGBGlobal.load(tmp_path, manifest)
        after = loaded.predict(test)
        np.testing.assert_allclose(before["yhat"], after["yhat"], rtol=1e-5)
        np.testing.assert_allclose(before["lo"], after["lo"], rtol=1e-5)
        assert loaded.entities == model.entities


class TestXGBLocalRoundTrip:
    def test_save_load_predict_identical(self, small_split, tmp_path):
        train, test = small_split
        model = pipeline.XGBLocalFleet.train(train)
        before = model.predict(test)
        model.save(tmp_path)
        manifest = {"trained_at": model.trained_at, "entities": sorted(model.entities)}
        loaded = pipeline.XGBLocalFleet.load(tmp_path, manifest)
        after = loaded.predict(test)
        np.testing.assert_allclose(before["yhat"], after["yhat"], rtol=1e-5)
        assert loaded.entities == model.entities


class TestProphetRoundTrip:
    def test_save_load_predict_close(self, small_split, tmp_path):
        train, test = small_split
        model = pipeline.ProphetFleet.train(train)
        before = model.predict(test)
        model.save(tmp_path)
        manifest = {"trained_at": model.trained_at, "entities": sorted(model.entities)}
        loaded = pipeline.ProphetFleet.load(tmp_path, manifest)
        after = loaded.predict(test)
        # Point predictions are deterministic; interval bounds are Monte Carlo and
        # legitimately differ between calls.
        np.testing.assert_allclose(before["yhat"], after["yhat"], rtol=1e-4, atol=1e-3)
        assert loaded.entities == model.entities


class TestChampionStore:
    def _persist(self, small_split, models_root, monkeypatch, run_id="20260801T000000Z"):
        monkeypatch.setattr(pipeline, "MODELS_ROOT", models_root)
        train, _ = small_split
        model = pipeline.XGBLocalFleet.train(train)
        pipeline.persist_champion(run_id, model)
        pipeline.write_champion_pointer(run_id, model.family, model.trained_at, len(model.entities))
        return model

    def test_pointer_roundtrip_and_load(self, small_split, tmp_path, monkeypatch):
        model = self._persist(small_split, tmp_path / "models", monkeypatch)
        pointer = pipeline.read_champion_pointer()
        assert pointer["model_name"] == model.family
        loaded, manifest = pipeline.load_champion_model(pointer)
        assert loaded is not None
        assert loaded.entities == model.entities

    def test_missing_artifact_refused(self, small_split, tmp_path, monkeypatch):
        self._persist(small_split, tmp_path / "models", monkeypatch)
        pointer = pipeline.read_champion_pointer()
        run_dir = pipeline.runs_root() / pointer["run_id"]
        victim = next((run_dir / "xgb_local").glob("*.json"))
        victim.unlink()
        loaded, manifest = pipeline.load_champion_model(pointer)
        assert loaded is None

    def test_tampered_manifest_refused(self, small_split, tmp_path, monkeypatch):
        self._persist(small_split, tmp_path / "models", monkeypatch)
        pointer = pipeline.read_champion_pointer()
        run_dir = pipeline.runs_root() / pointer["run_id"]
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["ride_count"] += 1
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        loaded, _ = pipeline.load_champion_model(pointer)
        assert loaded is None

    def test_missing_pointer_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "MODELS_ROOT", tmp_path / "models")
        assert pipeline.read_champion_pointer() is None


class TestRetention:
    def test_keeps_recent_plus_champion(self, tmp_path, monkeypatch):
        models_root = tmp_path / "models"
        monkeypatch.setattr(pipeline, "MODELS_ROOT", models_root)
        runs = [f"2026080{i}T000000Z" for i in range(1, 8)]
        for run_id in runs:
            (models_root / "runs" / run_id).mkdir(parents=True)
        # Champion is the OLDEST run: retention must keep it anyway.
        pipeline.write_champion_pointer(runs[0], "xgb_local_fleet", "2026-08-01", 2)
        pipeline.cleanup_runs()
        remaining = {p.name for p in (models_root / "runs").iterdir()}
        assert remaining == set(runs[-4:]) | {runs[0]}
