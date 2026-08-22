"""Configuration normalisation tests."""

import os
from unittest import mock

from themepark.config import DatabricksSettings

REQUIRED = {"DATABRICKS_HOST": "https://dbc-demo.cloud.databricks.com", "DATABRICKS_TOKEN": "x"}


def settings(**extra):
    with mock.patch.dict(os.environ, {**REQUIRED, **extra}, clear=False):
        return DatabricksSettings()


def test_http_path_accepts_a_bare_warehouse_id():
    assert settings(DATABRICKS_WAREHOUSE_ID="862f1da61ea3f0f5").http_path == (
        "/sql/1.0/warehouses/862f1da61ea3f0f5"
    )


def test_http_path_accepts_the_full_path_pasted_from_the_ui():
    """Copying the whole 'Connection details' line is the obvious mistake to make."""
    pasted = "/sql/1.0/warehouses/862f1da61ea3f0f5"
    assert settings(DATABRICKS_WAREHOUSE_ID=pasted).http_path == pasted


def test_http_path_tolerates_whitespace_and_trailing_slash():
    assert settings(DATABRICKS_WAREHOUSE_ID="  /sql/1.0/warehouses/abc123/  ").http_path == (
        "/sql/1.0/warehouses/abc123"
    )


def test_server_hostname_strips_the_scheme():
    """databricks-sql-connector wants a bare hostname, not a URL."""
    assert settings().server_hostname == "dbc-demo.cloud.databricks.com"
