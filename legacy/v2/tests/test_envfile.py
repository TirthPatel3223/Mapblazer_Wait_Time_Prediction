"""Tests for the `.env` loader.

`config.py` has always told the user that "local runs read it from .env". Nothing
implemented that, so `python jobs/seed_from_csv.py` failed on DATABRICKS_HOST with a
fully populated `.env` two directories up. These tests pin the behaviour that fixes it,
and the two rules that keep it from causing a worse problem than it solved.
"""

import os
from unittest import mock

from themepark import envfile


def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_values_reach_the_environment(tmp_path):
    path = write(tmp_path, "DATABRICKS_HOST=https://example.databricks.com\n")
    with mock.patch.dict(os.environ, {}, clear=True):
        assert envfile.load(path) == ["DATABRICKS_HOST"]
        assert os.environ["DATABRICKS_HOST"] == "https://example.databricks.com"


def test_a_real_environment_variable_wins(tmp_path):
    """The rule that keeps a stale working copy from overriding CI or systemd."""
    path = write(tmp_path, "DATABRICKS_HOST=https://stale.databricks.com\n")
    with mock.patch.dict(os.environ, {"DATABRICKS_HOST": "https://real.databricks.com"}, clear=True):
        assert envfile.load(path) == []
        assert os.environ["DATABRICKS_HOST"] == "https://real.databricks.com"


def test_override_is_available_but_not_the_default(tmp_path):
    path = write(tmp_path, "PG_HOST=from-file\n")
    with mock.patch.dict(os.environ, {"PG_HOST": "from-env"}, clear=True):
        assert envfile.load(path, override=True) == ["PG_HOST"]
        assert os.environ["PG_HOST"] == "from-file"


def test_a_hash_inside_a_value_is_not_a_comment():
    """Stripping inline comments would silently truncate a password at the '#'."""
    assert envfile.parse("PG_PASSWORD=hunter2#prod\n")["PG_PASSWORD"] == "hunter2#prod"
    assert envfile.parse("PG_PASSWORD=hunter2 # prod\n")["PG_PASSWORD"] == "hunter2 # prod"


def test_comments_blank_lines_quotes_and_export():
    parsed = envfile.parse(
        "\n"
        "# --- Upstream Postgres ---\n"
        "PG_HOST=db.example.com\n"
        "\n"
        'PG_USER="reader"\n'
        "PG_PASSWORD='s3cret'\n"
        "export PG_PORT=5432\n"
        "   \n"
        "not a variable line\n"
    )
    assert parsed == {
        "PG_HOST": "db.example.com",
        "PG_USER": "reader",
        "PG_PASSWORD": "s3cret",
        "PG_PORT": "5432",
    }


def test_a_jwt_survives_intact():
    """Supabase keys carry '.' and '-'; nothing in the parser may reshape them."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiYW5vbiJ9.abc-DEF_123"
    assert envfile.parse(f"SUPABASE_SERVICE_KEY={jwt}\n")["SUPABASE_SERVICE_KEY"] == jwt


def test_an_empty_value_is_kept_as_empty_not_dropped():
    """`_opt` treats '' and unset differently; the file must not blur them."""
    assert envfile.parse("DATABRICKS_WAREHOUSE_ID=\n") == {"DATABRICKS_WAREHOUSE_ID": ""}


def test_a_missing_file_is_not_an_error(tmp_path):
    """Databricks and GitHub runners have no .env and are already configured."""
    assert envfile.load(tmp_path / "nope.env") == []


def test_found_from_a_subdirectory(tmp_path):
    """Jobs get run from the repo root and from jobs/; both must resolve the same file."""
    path = write(tmp_path, "PG_HOST=db.example.com\n")
    nested = tmp_path / "jobs" / "nested"
    nested.mkdir(parents=True)
    assert envfile.find_dotenv(nested) == path


def test_settings_construct_from_a_dotenv_alone(tmp_path):
    """The actual reported failure: seed_from_csv died on DATABRICKS_HOST with a good .env."""
    from themepark.config import DatabricksSettings

    path = write(
        tmp_path,
        "DATABRICKS_HOST=https://example.databricks.com\nDATABRICKS_TOKEN=dapi-x\n",
    )
    with mock.patch.dict(os.environ, {}, clear=True):
        envfile.load(path)
        assert DatabricksSettings().server_hostname == "example.databricks.com"
