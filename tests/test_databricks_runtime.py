"""Local simulations of the Databricks serverless execution conditions that have each
cost a metered job run to discover."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pipeline

SOURCE = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text(encoding="utf-8")


class TestExecCompatibility:
    def test_exec_without_dunder_file(self):
        # A serverless spark_python_task runs exec(compile(source, path, "exec")) with
        # no __file__ binding. Importing the module that way must not raise, must not
        # need Spark, and must not need credentials.
        namespace = {"__name__": "pipeline_exec_check"}
        exec(compile(SOURCE, "pipeline.py", "exec"), namespace)
        assert callable(namespace["main"])

    def test_no_environment_credentials_required(self):
        # A job inside Databricks is already authenticated; the task must not demand
        # DATABRICKS_HOST/TOKEN just to spell table names.
        assert "os.environ" not in SOURCE
        assert "DATABRICKS_TOKEN" not in SOURCE


class TestExitSemantics:
    def test_no_sys_exit_in_source(self):
        # SystemExit inside the IPython kernel marks the task FAILED even with code 0.
        assert "sys.exit" not in SOURCE
        assert "raise SystemExit" not in SOURCE

    def test_success_returns_normally(self, base_state):
        # The fixture ran main() to completion: it returned a summary rather than
        # raising anything.
        assert isinstance(base_state["summary"], dict)

    def test_failure_raises_plain_exception(self, empty_env):
        from conftest import make_bronze

        empty_env.write(pipeline.BRONZE_TABLE, make_bronze(seed=7, all_zero=True))
        with pytest.raises(Exception) as excinfo:
            pipeline.main()
        assert not isinstance(excinfo.value, SystemExit)


class TestTimestampCompatibility:
    def test_nanosecond_timestamps_downcast(self):
        # pandas defaults to TIMESTAMP(NANOS), which Spark cannot read at all.
        df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 12:00:00"]), "x": [1]})
        assert df["ts"].dtype == "datetime64[ns]"
        out = pipeline.timestamps_to_us(df)
        assert out["ts"].dtype == "datetime64[us]"
        assert out["x"].dtype == df["x"].dtype

    def test_downcast_without_as_unit(self, monkeypatch):
        # The Databricks serverless runtime ships a pandas older than 2.2, where
        # Series.dt.as_unit does not exist -- this cost run 167970874790110. The
        # astype fallback must produce the same result.
        import pandas.core.indexes.accessors as accessors

        monkeypatch.delattr(accessors.DatetimeProperties, "as_unit", raising=False)
        df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 12:00:00"])})
        assert not hasattr(df["ts"].dt, "as_unit")
        out = pipeline.timestamps_to_us(df)
        assert out["ts"].dtype == "datetime64[us]"
