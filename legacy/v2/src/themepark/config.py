"""Runtime configuration, entirely from the environment.

v1 hard-coded a relative CSV path in every script, so nothing ran unless the working
directory happened to be the repo root. The same code now has to run in three places --
a laptop, a GitHub Actions runner and a Databricks serverless task -- so every location
is injected.

Nothing here has a credential default. A missing secret must fail loudly at startup, not
silently fall back to something that half-works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from .envfile import load as load_dotenv

# At import, because every settings class below reads os.environ the moment it is
# constructed and there is no earlier hook they all share. Doing it here means no entry
# point -- job, script or test -- can forget to, and a real environment variable still
# wins over the file, so CI and Databricks behave exactly as they did before.
load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "Local runs read it from .env; CI reads it from GitHub Secrets."
        )
    return value


def _opt(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class SourceDBSettings:
    """The upstream Mapblazer Postgres. Read-only, always."""

    host: str = field(default_factory=lambda: _env("PG_HOST"))
    port: int = field(default_factory=lambda: _int("PG_PORT", 5432))
    database: str = field(default_factory=lambda: _env("PG_DATABASE"))
    user: str = field(default_factory=lambda: _env("PG_USER"))
    password: str = field(default_factory=lambda: _env("PG_PASSWORD"))
    sslmode: str = field(default_factory=lambda: _opt("PG_SSLMODE", "require"))

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password} sslmode={self.sslmode}"
        )

    def __repr__(self) -> str:  # never let a password reach a log line
        return f"SourceDBSettings(host={self.host!r}, database={self.database!r}, user={self.user!r})"


@dataclass(frozen=True)
class DatabricksSettings:
    """Where the lakehouse is, and -- only from outside it -- how to authenticate.

    `host` and `token` are deliberately optional. This object has two very different
    users. Code on a laptop or a GitHub runner needs credentials to reach the workspace
    API. Code running *inside* a Databricks task is already authenticated and uses this
    object purely to spell table and volume names, where demanding a token it does not
    need is not "failing loudly" -- it is failing wrongly, which is how the first four
    tasks died after their imports were fixed.

    So the credentials are validated at the point of use, by `require_api_credentials`,
    rather than at construction.
    """

    host: str = field(default_factory=lambda: _opt("DATABRICKS_HOST"))
    token: str = field(default_factory=lambda: _opt("DATABRICKS_TOKEN"))
    warehouse_id: str = field(default_factory=lambda: _opt("DATABRICKS_WAREHOUSE_ID"))
    catalog: str = field(default_factory=lambda: _opt("DATABRICKS_CATALOG", "themepark"))
    bronze_schema: str = field(default_factory=lambda: _opt("BRONZE_SCHEMA", "bronze"))
    silver_schema: str = field(default_factory=lambda: _opt("SILVER_SCHEMA", "silver"))
    gold_schema: str = field(default_factory=lambda: _opt("GOLD_SCHEMA", "gold"))
    landing_volume: str = field(default_factory=lambda: _opt("LANDING_VOLUME", "landing"))

    @property
    def http_path(self) -> str:
        """SQL warehouse HTTP path, accepting either the bare ID or the whole path.

        The Databricks UI shows this as `/sql/1.0/warehouses/<id>` under "Connection
        details", and copying the whole line is the obvious thing to do. Naively
        interpolating that produces `/sql/1.0/warehouses//sql/1.0/warehouses/<id>` and
        fails every query with an error that points nowhere near the cause, so normalise
        to the last segment instead.
        """
        return f"/sql/1.0/warehouses/{self.warehouse_id.strip().rstrip('/').rsplit('/', 1)[-1]}"

    @property
    def server_hostname(self) -> str:
        return self.host.replace("https://", "").replace("http://", "").rstrip("/")

    @property
    def landing_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.bronze_schema}/{self.landing_volume}"

    def table(self, schema: str, name: str) -> str:
        return f"{self.catalog}.{schema}.{name}"

    @property
    def bronze_table(self) -> str:
        return self.table(self.bronze_schema, "wait_times_raw")

    @property
    def silver_table(self) -> str:
        return self.table(self.silver_schema, "wait_times")

    @property
    def predictions_table(self) -> str:
        return self.table(self.gold_schema, "predictions")

    @property
    def metrics_table(self) -> str:
        return self.table(self.gold_schema, "model_metrics")

    @property
    def accuracy_table(self) -> str:
        return self.table(self.gold_schema, "prediction_accuracy")

    @property
    def promotion_table(self) -> str:
        return self.table(self.gold_schema, "promotion_log")

    def require_api_credentials(self) -> None:
        """Fail before building a client, not deep inside an SDK call.

        Raised only on the paths that genuinely dial the workspace from outside it.
        """
        missing = [
            name
            for name, value in (("DATABRICKS_HOST", self.host), ("DATABRICKS_TOKEN", self.token))
            if not value.strip()
        ]
        if missing:
            raise RuntimeError(
                f"{' and '.join(missing)} must be set to reach the Databricks API from "
                "outside the workspace. Local runs read it from .env; CI reads it from "
                "GitHub Secrets. A task running inside Databricks does not need either -- "
                "if you are seeing this in a job run, something is calling themepark."
                "sources.Databricks where it should be using Spark directly."
            )

    def __repr__(self) -> str:
        return f"DatabricksSettings(host={self.host!r}, catalog={self.catalog!r})"


@dataclass(frozen=True)
class SupabaseSettings:
    url: str = field(default_factory=lambda: _env("SUPABASE_URL"))
    service_key: str = field(default_factory=lambda: _env("SUPABASE_SERVICE_KEY"))

    @property
    def rest_url(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1"

    def __repr__(self) -> str:
        return f"SupabaseSettings(url={self.url!r})"


@dataclass(frozen=True)
class PipelineSettings:
    """Knobs that change behaviour rather than location."""

    # Forecast horizon handed to the downstream routing optimiser.
    horizon_days: int = field(default_factory=lambda: _int("HORIZON_DAYS", 7))
    grid_minutes: int = field(default_factory=lambda: _int("GRID_MINUTES", 30))

    # Trailing holdout for the champion/challenger backtest. A rolling origin, not a
    # random split -- it mirrors what a weekly-retrain system actually experiences.
    backtest_days: int = field(default_factory=lambda: _int("BACKTEST_DAYS", 14))

    # Promotion gate thresholds. See themepark.promote.
    min_improvement: float = field(default_factory=lambda: float(_opt("MIN_IMPROVEMENT", "0.02")))
    max_high_wait_regression: float = field(
        default_factory=lambda: float(_opt("MAX_HIGH_WAIT_REGRESSION", "0.05"))
    )
    min_coverage: float = field(default_factory=lambda: float(_opt("MIN_COVERAGE", "0.95")))

    # A ride whose mean wait clears this is "high-wait": where accuracy has real value.
    high_wait_threshold_min: float = field(
        default_factory=lambda: float(_opt("HIGH_WAIT_THRESHOLD_MIN", "10"))
    )

    registered_model_name: str = field(
        default_factory=lambda: _opt("REGISTERED_MODEL_NAME", "themepark.gold.wait_time_forecaster")
    )
    mlflow_experiment: str = field(
        default_factory=lambda: _opt("MLFLOW_EXPERIMENT", "/Shared/themepark-wait-times")
    )
    keep_model_versions: int = field(default_factory=lambda: _int("KEEP_MODEL_VERSIONS", 4))


@lru_cache(maxsize=1)
def pipeline() -> PipelineSettings:
    return PipelineSettings()
