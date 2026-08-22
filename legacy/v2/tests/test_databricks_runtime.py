"""Settings must construct inside a Databricks task, where there are no credentials.

The four jobs run *inside* the workspace. They are already authenticated and use
DatabricksSettings only to spell table and volume names. Requiring DATABRICKS_HOST there
is not "failing loudly", it is failing wrongly -- and it is what killed all four tasks in
run 579827130793859, immediately after the __file__ defect was fixed.

The boundary these tests pin: naming works with no environment at all; credentials are
demanded only where something actually dials the workspace from outside it.
"""

import os
from unittest import mock

import pytest

from themepark.config import (
    DatabricksSettings,
    SourceDBSettings,
    SupabaseSettings,
    pipeline,
)

# What a Databricks serverless task sees: none of the runner's secrets.
INSIDE_DATABRICKS = {"DATABRICKS_RUNTIME_VERSION": "client.2.9"}


@pytest.fixture
def inside_databricks():
    pipeline.cache_clear()
    with mock.patch.dict(os.environ, INSIDE_DATABRICKS, clear=True):
        yield
    pipeline.cache_clear()


def test_settings_construct_with_no_credentials(inside_databricks):
    """The regression. This raised RuntimeError and failed the task during main()."""
    cfg = DatabricksSettings()
    assert cfg.host == ""
    assert cfg.token == ""


def test_table_naming_works_with_no_credentials(inside_databricks):
    """What the jobs actually need from this object."""
    cfg = DatabricksSettings()
    assert cfg.bronze_table == "themepark.bronze.wait_times_raw"
    assert cfg.silver_table == "themepark.silver.wait_times"
    assert cfg.predictions_table == "themepark.gold.predictions"
    assert cfg.landing_path == "/Volumes/themepark/bronze/landing"


def test_pipeline_settings_construct_with_no_credentials(inside_databricks):
    """train_job reads these; every field must have a default."""
    p = pipeline()
    assert p.horizon_days == 7
    assert p.min_coverage == 0.95


def test_missing_credentials_are_reported_together(inside_databricks):
    """One error naming both beats two round trips."""
    with pytest.raises(RuntimeError) as err:
        DatabricksSettings().require_api_credentials()
    message = str(err.value)
    assert "DATABRICKS_HOST" in message and "DATABRICKS_TOKEN" in message


def test_the_error_says_a_job_should_never_see_it(inside_databricks):
    """Whoever hits this next needs to know which side of the boundary they are on."""
    with pytest.raises(RuntimeError, match="inside Databricks does not need"):
        DatabricksSettings().require_api_credentials()


def test_blank_is_treated_as_missing(inside_databricks):
    """An empty GitHub secret expands to '', which set-but-useless must not pass."""
    with mock.patch.dict(os.environ, {"DATABRICKS_HOST": "   ", "DATABRICKS_TOKEN": ""}):
        with pytest.raises(RuntimeError):
            DatabricksSettings().require_api_credentials()


def test_credentials_pass_when_present(inside_databricks):
    with mock.patch.dict(
        os.environ, {"DATABRICKS_HOST": "https://x.databricks.com", "DATABRICKS_TOKEN": "dapi"}
    ):
        DatabricksSettings().require_api_credentials()  # must not raise


@pytest.mark.parametrize("settings_class", [SourceDBSettings, SupabaseSettings])
def test_runner_only_settings_still_fail_loudly(inside_databricks, settings_class):
    """The boundary is deliberate: no Databricks task constructs either of these, so a
    missing secret there is a real misconfiguration and must not be softened."""
    with pytest.raises(RuntimeError, match="not set"):
        settings_class()
