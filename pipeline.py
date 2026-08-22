"""Theme-park wait-time weekly pipeline. One file, one Databricks job task.

End to end: bronze -> silver -> train three model families -> KPIs -> gold tables ->
quality checks -> promote current tables to last -> update the serving pointer.

Serving discipline (the part that must never be broken):

  - Every silver and gold table exists twice: <name>_current and <name>_last.
  - The run writes only _current. Quality checks run against _current.
  - Any failure: drop _current, leave _last untouched, run the fallback, re-raise so the
    job goes red. The fallback re-scores the upcoming week with the previous model so
    the forecast window is fresh even when the run failed.
  - Only a fully successful run promotes _current into _last (atomic DEEP CLONE per
    table, all promotions back to back at the very end).
  - Model artifacts get the same guarantee: every run writes runs/<run_id>/ in the
    models volume, and serving.json (the pointer that says which run serves) is written
    last, after the tables are promoted. A failed run's directory is orphaned, never
    read, and removed by retention.
  - Supabase and the dashboard read _last only. That happens outside Databricks (Free
    Edition jobs have no outbound internet) via publish.py.

Databricks serverless constraints honoured here:
  - No __file__ usage (spark_python_task is exec()'d and __file__ is undefined).
  - Success returns normally; SystemExit would mark the task FAILED even with code 0.
  - No credentials required: a job inside Databricks is already authenticated.
  - Datetime columns are cast to microseconds before handing them to Spark.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("themepark")

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------

CATALOG = "themepark"
BRONZE_TABLE = f"{CATALOG}.bronze.wait_times_raw"
SILVER_TABLE = f"{CATALOG}.silver.wait_times"
KPI_TABLE = f"{CATALOG}.gold.kpis"
PRED_TABLE = f"{CATALOG}.gold.predictions"
CURRENT = "_current"
LAST = "_last"

MODELS_ROOT = Path("/Volumes/themepark/gold/models")

PARK_TZ = "America/Los_Angeles"

# park -> (open_hour, close_hour_exclusive, first trustworthy local date). Local
# wall-clock hours; close 24 means "through 23:59". Six Flags was not scraped reliably
# before 2026-02-15.
PARK_HOURS: dict[str, tuple[int, int, str]] = {
    "Disneyland": (8, 24, "2025-12-06"),
    "Disney California Adventure Park": (8, 22, "2025-12-06"),
    "Universal Studios Hollywood": (8, 22, "2025-12-06"),
    "SeaWorld San Diego": (10, 20, "2025-12-06"),
    "Six Flags Magic Mountain": (10, 21, "2026-02-15"),
}

# 'SeaWorld San Diego Obsolete' duplicates the live SeaWorld park under another tp_id;
# one Six Flags attraction is literally named "0" in the source table.
EXCLUDED_PARKS = {"SeaWorld San Diego Obsolete"}
EXCLUDED_RIDE_NAMES = {"0", ""}

MAX_WAIT = 900  # 900+ is the source system's sentinel, not a wait
GRID_MINUTES = 30
MIN_OBS_PER_RIDE = 100  # 30-minute observations required in silver
MIN_TRAIN_ROWS = 50  # rows required in the train split to fit a per-ride model
TEST_FRACTION = 0.20
FORECAST_DAYS = 7
IMPROVEMENT_FACTOR = 0.99  # a new model must be at least 1 percent better to take over
HIGH_WAIT_MEAN_MIN = 10.0  # rides averaging above this are the high-wait segment
PEAK_HOURS = (11, 20)  # local, end exclusive
SEVERE_MISS_MIN = 15.0
WITHIN_MIN = 10.0
INTERVAL_WIDTH = 0.80
RETAIN_RUNS = 4

# Quality-check thresholds (module-level so tests can tighten or relax them).
MIN_ATTRACTIONS = 50
MAX_ZERO_PCT = 75.0
MIN_ROW_RATIO_VS_LAST = 0.5
MIN_FORECAST_RIDE_COVERAGE = 0.95

MODEL_PROPHET = "prophet_fleet"
MODEL_XGB_GLOBAL = "xgb_global"
MODEL_XGB_LOCAL = "xgb_local_fleet"
MODEL_BASELINE = "baseline_ride_mean"
CANDIDATES = (MODEL_PROPHET, MODEL_XGB_GLOBAL, MODEL_XGB_LOCAL)

# Fixed hyperparameters, no search.
XGB_PARAMS = dict(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

TEMPORAL_FEATURES = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "is_holiday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

STATUS_FRESH = "fresh_model"
STATUS_KEPT = "kept_previous_model"
STATUS_FALLBACK = "fallback_after_failure"


class PipelineError(Exception):
    """A failure the pipeline diagnosed itself, with a message worth reading."""


class QualityCheckError(PipelineError):
    """One or more quality checks failed against a _current table."""


# ------------------------------------------------------------------------------------
# Spark IO. Everything that touches Spark goes through these few functions, so local
# tests can substitute a dict-backed store and exercise the full orchestration.
# ------------------------------------------------------------------------------------

_SPARK = None


def get_spark():
    global _SPARK
    if _SPARK is None:
        from pyspark.sql import SparkSession

        _SPARK = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        try:
            # Pin the session timezone so tz-naive wall-time columns round-trip
            # unchanged through TimestampType.
            _SPARK.conf.set("spark.sql.session.timeZone", "UTC")
        except Exception:
            log.warning("could not pin spark session timezone; assuming UTC default")
    return _SPARK


def series_to_us(s: pd.Series) -> pd.Series:
    """Downcast a datetime series to microseconds across pandas versions.

    .dt.as_unit exists only from pandas 2.2; astype("datetime64[us]") works on 2.0-2.1.
    On pandas 1.x (some Databricks runtimes) everything is nanoseconds and neither
    exists -- that is fine, because spark.createDataFrame converts ns itself; the
    TIMESTAMP(NANOS) trap is only in direct pandas-to-Parquet writes, which this
    pipeline never does.
    """
    if hasattr(s.dt, "as_unit"):
        return s.dt.as_unit("us")
    try:
        return s.astype("datetime64[us]")
    except (TypeError, ValueError):
        return s


def timestamps_to_us(df: pd.DataFrame) -> pd.DataFrame:
    """pandas defaults to nanosecond timestamps, which Parquet readers can choke on."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = series_to_us(out[col])
    return out


def read_table(name: str) -> pd.DataFrame:
    return get_spark().table(name).toPandas()


def write_table(name: str, df: pd.DataFrame) -> None:
    """Overwrite `name` with `df`. A Delta overwrite is a single atomic commit."""
    sdf = get_spark().createDataFrame(timestamps_to_us(df))
    sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
    log.info("wrote %d rows to %s", len(df), name)


def drop_table(name: str) -> None:
    get_spark().sql(f"DROP TABLE IF EXISTS {name}")


def table_exists(name: str) -> bool:
    try:
        get_spark().sql(f"DESCRIBE TABLE {name}")
        return True
    except Exception:
        return False


def table_count(name: str) -> int:
    return int(get_spark().sql(f"SELECT COUNT(*) AS c FROM {name}").collect()[0][0])


def promote_table(src: str, dst: str) -> None:
    """Atomically replace dst with the contents of src.

    DEEP CLONE is a single Delta commit, so a reader never sees a partial table.
    CREATE OR REPLACE ... AS SELECT is the documented fallback where clone is refused.
    """
    spark = get_spark()
    try:
        spark.sql(f"CREATE OR REPLACE TABLE {dst} DEEP CLONE {src}")
        log.info("promoted %s -> %s (deep clone)", src, dst)
    except Exception as exc:
        log.warning("deep clone %s -> %s refused (%s); using CREATE OR REPLACE AS SELECT", src, dst, exc)
        spark.sql(f"CREATE OR REPLACE TABLE {dst} AS SELECT * FROM {src}")
        log.info("promoted %s -> %s (create or replace as select)", src, dst)


def ensure_workspace() -> None:
    spark = get_spark()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.gold.models")


# ------------------------------------------------------------------------------------
# Time and naming
# ------------------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def now_local() -> pd.Timestamp:
    """Current park-local wall time, tz-naive. Module-level so tests can pin it."""
    return pd.Timestamp(utcnow()).tz_convert(PARK_TZ).tz_localize(None)


def make_run_id() -> str:
    return utcnow().strftime("%Y%m%dT%H%M%SZ")


_PUNCT = re.compile(r"[^\w\s-]")


def canonical_key(name) -> str:
    """The ONE sanitizer. Two divergent sanitizers once silently dropped 43 of 120
    rides, disproportionately the busiest ones. Never re-derive a key inline."""
    if name is None:
        return ""
    safe = _PUNCT.sub("", str(name)).strip().replace(" ", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def entity_of(df: pd.DataFrame) -> pd.Series:
    """park::ride key. Two parks may legitimately share a ride name, so the ride key
    alone is not an identity."""
    return df["park_name"].map(canonical_key) + "::" + df["ride_key"].astype(str)


def entity_filename(entity: str) -> str:
    return entity.replace("::", "__") + ".json"


def to_local(values) -> pd.Series:
    """UTC (naive or aware) -> tz-naive park-local wall time.

    This must run BEFORE any operating-hours filter or calendar feature. Filtering on
    UTC hours silently deletes the evening peak and keeps seven hours of closed-park
    zeros.
    """
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s.dt.tz_convert(PARK_TZ).dt.tz_localize(None)


def to_utc(values) -> pd.Series:
    """tz-naive park-local wall time -> tz-naive UTC. DST-ambiguous local times resolve
    to the DST occurrence; skipped times shift forward."""
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    localized = s.dt.tz_localize(PARK_TZ, ambiguous=True, nonexistent="shift_forward")
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


# ------------------------------------------------------------------------------------
# Step 1: bronze -> silver
# ------------------------------------------------------------------------------------


def build_silver(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean, timezone-correct, 30-minute-gridded observations. Order matters."""
    work = raw.copy()
    n0 = len(work)

    # 1. Duplicate observations from overlapping ingestion batches.
    if "wait_time_id" in work.columns:
        work = work.drop_duplicates(subset=["wait_time_id"])

    # 2. Bad rows: negative waits, the 900+ sentinel, the ride named "0", the
    #    duplicated obsolete SeaWorld park.
    work = work[(work["wait_time"] >= 0) & (work["wait_time"] < MAX_WAIT)].copy()
    work["ride_name"] = work["ride_name"].astype(str).str.strip()
    work = work[~work["ride_name"].isin(EXCLUDED_RIDE_NAMES)]
    work = work[~work["park_name"].isin(EXCLUDED_PARKS)]
    work = work[work["park_name"].isin(PARK_HOURS)].copy()

    # 3. Convert to local time FIRST, then filter operating hours on the LOCAL clock.
    work["ts_local"] = to_local(work["ts_utc"])
    work = work[work["ts_local"].notna()]
    local_hour = work["ts_local"].dt.hour
    local_date = work["ts_local"].dt.normalize()
    keep = pd.Series(False, index=work.index)
    for park, (open_h, close_h, start_date) in PARK_HOURS.items():
        keep |= (
            (work["park_name"] == park)
            & (local_hour >= open_h)
            & (local_hour < close_h)
            & (local_date >= pd.Timestamp(start_date))
        )
    work = work[keep].copy()
    if work.empty:
        raise PipelineError("silver transform removed every row; check timestamps and park names")

    # 4. Canonical ride key.
    work["ride_key"] = work["ride_name"].map(canonical_key)
    work = work[work["ride_key"] != ""]

    # 5. Fixed 30-minute grid per attraction (mean within each slot).
    grid = (
        work.set_index("ts_local")
        .groupby(["park_name", "ride_key", "ride_name"])["wait_time"]
        .resample(f"{GRID_MINUTES}min")
        .mean()
        .dropna()
        .reset_index()
    )
    # Two raw spellings can canonicalise to the same key; collapse them.
    grid = grid.groupby(["park_name", "ride_key", "ts_local"], as_index=False).agg(
        ride_name=("ride_name", "first"), wait_time=("wait_time", "mean")
    )

    # 6. Attractions with too little history to model.
    counts = grid.groupby(["park_name", "ride_key"])["wait_time"].transform("size")
    grid = grid[counts >= MIN_OBS_PER_RIDE].reset_index(drop=True)

    grid["ts_utc"] = to_utc(grid["ts_local"])
    grid = grid[["park_name", "ride_key", "ride_name", "ts_local", "ts_utc", "wait_time"]]

    n_entities = grid.groupby(["park_name", "ride_key"]).ngroups
    log.info(
        "silver: %d raw -> %d gridded rows, %d attractions, %d parks, mean wait %.1f, %.1f%% zeros, %s to %s local",
        n0,
        len(grid),
        n_entities,
        grid["park_name"].nunique(),
        grid["wait_time"].mean(),
        (grid["wait_time"] == 0).mean() * 100,
        grid["ts_local"].min(),
        grid["ts_local"].max(),
    )
    return grid


# ------------------------------------------------------------------------------------
# Features and split
# ------------------------------------------------------------------------------------

_HOLIDAYS = None


def _us_holidays():
    global _HOLIDAYS
    if _HOLIDAYS is None:
        import holidays

        _HOLIDAYS = holidays.US()
    return _HOLIDAYS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from park-LOCAL wall time. Feeding this UTC reproduces the
    defect where Saturday evening was labelled Sunday."""
    out = df.copy()
    ts = pd.to_datetime(out["ts_local"])
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    dates = ts.dt.normalize()
    hmap = {d: int(d.date() in _us_holidays()) for d in dates.drop_duplicates()}
    out["is_holiday"] = dates.map(hmap).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def chrono_split(silver: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First 80 percent by time trains, last 20 percent tests. Never random: random
    folds leak the future into validation on a time series."""
    cutoff = silver["ts_local"].quantile(1.0 - TEST_FRACTION)
    train = silver[silver["ts_local"] <= cutoff].copy()
    test = silver[silver["ts_local"] > cutoff].copy()
    log.info(
        "split: %d train rows through %s, %d test rows after (cutoff %s)",
        len(train),
        train["ts_local"].max(),
        len(test),
        cutoff,
    )
    return train, test


# ------------------------------------------------------------------------------------
# Model families. Each trains from a dataframe, predicts yhat/lo/hi for covered rows
# (NaN for entities it does not know), and persists as PLAIN FILES: Prophet via
# model_to_json, XGBoost via Booster.save_model. Never pickles of repo classes -- a
# pickle is only loadable by a process that can import the class that made it, which is
# exactly what kept breaking inside Databricks jobs.
# ------------------------------------------------------------------------------------


class ProphetFleet:
    family = MODEL_PROPHET

    def __init__(self):
        self.models: dict[str, object] = {}
        self.trained_at: str | None = None

    @property
    def entities(self) -> set[str]:
        return set(self.models)

    @classmethod
    def train(cls, train_df: pd.DataFrame) -> ProphetFleet:
        from prophet import Prophet

        self = cls()
        work = train_df.copy()
        work["_entity"] = entity_of(work)
        skipped = 0
        for entity, group in work.groupby("_entity", sort=False):
            series = (
                group[["ts_local", "wait_time"]]
                .rename(columns={"ts_local": "ds", "wait_time": "y"})
                .dropna()
                .sort_values("ds")
            )
            if len(series) < MIN_TRAIN_ROWS:
                skipped += 1
                continue
            model = Prophet(
                growth="flat",
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=INTERVAL_WIDTH,
                # The default 1000 trend draws dominate predict time and buy nothing
                # at 30-minute granularity.
                uncertainty_samples=200,
            )
            model.add_country_holidays(country_name="US")
            try:
                self.models[str(entity)] = model.fit(series)
            except Exception as exc:
                log.warning("prophet fit failed for %s: %s", entity, exc)
                skipped += 1
        self.trained_at = utcnow().isoformat()
        log.info("prophet_fleet: fitted %d entities, skipped %d", len(self.models), skipped)
        if not self.models:
            raise PipelineError("prophet_fleet fitted zero entities")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {"yhat": np.nan, "lo": np.nan, "hi": np.nan}, index=df.index, dtype=float
        )
        keys = entity_of(df)
        for entity, idx in keys.groupby(keys).groups.items():
            model = self.models.get(str(entity))
            if model is None:
                continue
            future = pd.DataFrame({"ds": pd.to_datetime(df.loc[idx, "ts_local"]).values})
            fc = model.predict(future)
            out.loc[idx, "yhat"] = fc["yhat"].to_numpy()
            out.loc[idx, "lo"] = fc["yhat_lower"].to_numpy()
            out.loc[idx, "hi"] = fc["yhat_upper"].to_numpy()
        return out

    def save(self, run_dir: Path) -> None:
        from prophet.serialize import model_to_json

        target = run_dir / "prophet"
        target.mkdir(parents=True, exist_ok=True)
        for entity, model in self.models.items():
            (target / entity_filename(entity)).write_text(model_to_json(model), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path, manifest: dict) -> ProphetFleet:
        from prophet.serialize import model_from_json

        self = cls()
        for entity in manifest["entities"]:
            path = run_dir / "prophet" / entity_filename(entity)
            self.models[entity] = model_from_json(path.read_text(encoding="utf-8"))
        self.trained_at = manifest["trained_at"]
        return self


def _residual_quantiles(entities: pd.Series, residuals: np.ndarray) -> tuple[dict, dict]:
    lo_q = (1.0 - INTERVAL_WIDTH) / 2.0
    frame = pd.DataFrame({"e": entities.to_numpy(), "r": residuals})
    grouped = frame.groupby("e")["r"]
    lo = {k: float(v) for k, v in grouped.quantile(lo_q).items()}
    hi = {k: float(v) for k, v in grouped.quantile(1.0 - lo_q).items()}
    return lo, hi


class XGBGlobal:
    family = MODEL_XGB_GLOBAL
    FEATURES = ["park_name", "ride_key", *TEMPORAL_FEATURES]

    def __init__(self):
        self.model = None
        self.park_categories: list[str] = []
        self.ride_categories: list[str] = []
        self._entities: set[str] = set()
        self.res_lo: dict[str, float] = {}
        self.res_hi: dict[str, float] = {}
        self.trained_at: str | None = None

    @property
    def entities(self) -> set[str]:
        return self._entities

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pinned category lists keep training and serving aligned: pandas assigns
        # category codes by order of appearance, so letting each batch infer its own
        # would silently remap ride identities at inference.
        feats = build_features(df)
        feats["park_name"] = feats["park_name"].astype(
            pd.CategoricalDtype(categories=self.park_categories)
        )
        feats["ride_key"] = feats["ride_key"].astype(
            pd.CategoricalDtype(categories=self.ride_categories)
        )
        return feats[self.FEATURES]

    @classmethod
    def train(cls, train_df: pd.DataFrame) -> XGBGlobal:
        import xgboost as xgb

        self = cls()
        work = train_df.copy()
        ents = entity_of(work)
        self.park_categories = sorted(work["park_name"].astype(str).unique())
        self.ride_categories = sorted(work["ride_key"].astype(str).unique())
        X = self._encode(work)
        y = work["wait_time"].astype(float)
        self.model = xgb.XGBRegressor(enable_categorical=True, **XGB_PARAMS).fit(X, y)
        self.res_lo, self.res_hi = _residual_quantiles(ents, y.to_numpy() - self.model.predict(X))
        self._entities = set(ents.unique())
        self.trained_at = utcnow().isoformat()
        log.info("xgb_global: fitted on %d rows, %d entities", len(work), len(self._entities))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        yhat = self.model.predict(self._encode(df)).astype(float)
        keys = entity_of(df)
        known = keys.isin(self._entities).to_numpy()
        yhat[~known] = np.nan
        lo_off = keys.map(self.res_lo).fillna(0.0).to_numpy()
        hi_off = keys.map(self.res_hi).fillna(0.0).to_numpy()
        return pd.DataFrame(
            {"yhat": yhat, "lo": yhat + lo_off, "hi": yhat + hi_off}, index=df.index
        )

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(run_dir / "xgb_global.json"))
        meta = {
            "park_categories": self.park_categories,
            "ride_categories": self.ride_categories,
            "entities": sorted(self._entities),
            "residual_lo": self.res_lo,
            "residual_hi": self.res_hi,
            "features": self.FEATURES,
        }
        (run_dir / "xgb_global_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path, manifest: dict) -> XGBGlobal:
        import xgboost as xgb

        self = cls()
        meta = json.loads((run_dir / "xgb_global_meta.json").read_text(encoding="utf-8"))
        self.model = xgb.XGBRegressor(enable_categorical=True)
        self.model.load_model(str(run_dir / "xgb_global.json"))
        self.park_categories = meta["park_categories"]
        self.ride_categories = meta["ride_categories"]
        self._entities = set(meta["entities"])
        self.res_lo = meta["residual_lo"]
        self.res_hi = meta["residual_hi"]
        self.trained_at = manifest["trained_at"]
        return self


class XGBLocalFleet:
    family = MODEL_XGB_LOCAL

    def __init__(self):
        self.models: dict[str, object] = {}
        self.res_lo: dict[str, float] = {}
        self.res_hi: dict[str, float] = {}
        self.trained_at: str | None = None

    @property
    def entities(self) -> set[str]:
        return set(self.models)

    @classmethod
    def train(cls, train_df: pd.DataFrame) -> XGBLocalFleet:
        import xgboost as xgb

        self = cls()
        work = build_features(train_df)
        work["_entity"] = entity_of(work)
        skipped = 0
        for entity, group in work.groupby("_entity", sort=False):
            if len(group) < MIN_TRAIN_ROWS:
                skipped += 1
                continue
            X = group[TEMPORAL_FEATURES]
            y = group["wait_time"].astype(float)
            model = xgb.XGBRegressor(**XGB_PARAMS).fit(X, y)
            residual = y.to_numpy() - model.predict(X)
            lo_q = (1.0 - INTERVAL_WIDTH) / 2.0
            self.models[str(entity)] = model
            self.res_lo[str(entity)] = float(np.quantile(residual, lo_q))
            self.res_hi[str(entity)] = float(np.quantile(residual, 1.0 - lo_q))
        self.trained_at = utcnow().isoformat()
        log.info("xgb_local_fleet: fitted %d entities, skipped %d", len(self.models), skipped)
        if not self.models:
            raise PipelineError("xgb_local_fleet fitted zero entities")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        work = build_features(df)
        out = pd.DataFrame(
            {"yhat": np.nan, "lo": np.nan, "hi": np.nan}, index=df.index, dtype=float
        )
        keys = entity_of(work)
        for entity, idx in keys.groupby(keys).groups.items():
            model = self.models.get(str(entity))
            if model is None:
                continue
            yhat = model.predict(work.loc[idx, TEMPORAL_FEATURES]).astype(float)
            out.loc[idx, "yhat"] = yhat
            out.loc[idx, "lo"] = yhat + self.res_lo[str(entity)]
            out.loc[idx, "hi"] = yhat + self.res_hi[str(entity)]
        return out

    def save(self, run_dir: Path) -> None:
        target = run_dir / "xgb_local"
        target.mkdir(parents=True, exist_ok=True)
        for entity, model in self.models.items():
            model.save_model(str(target / entity_filename(entity)))
        meta = {"residual_lo": self.res_lo, "residual_hi": self.res_hi}
        (run_dir / "xgb_local_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path, manifest: dict) -> XGBLocalFleet:
        import xgboost as xgb

        self = cls()
        meta = json.loads((run_dir / "xgb_local_meta.json").read_text(encoding="utf-8"))
        for entity in manifest["entities"]:
            model = xgb.XGBRegressor()
            model.load_model(str(run_dir / "xgb_local" / entity_filename(entity)))
            self.models[entity] = model
        self.res_lo = meta["residual_lo"]
        self.res_hi = meta["residual_hi"]
        self.trained_at = manifest["trained_at"]
        return self


class BaselineMean:
    """Per-ride historical mean. Never serves; the floor every candidate must beat."""

    family = MODEL_BASELINE

    def __init__(self):
        self.means: dict[str, float] = {}

    @property
    def entities(self) -> set[str]:
        return set(self.means)

    @classmethod
    def train(cls, train_df: pd.DataFrame) -> BaselineMean:
        self = cls()
        ents = entity_of(train_df)
        self.means = train_df.groupby(ents)["wait_time"].mean().astype(float).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        yhat = entity_of(df).map(self.means)
        return pd.DataFrame({"yhat": yhat, "lo": yhat, "hi": yhat}, index=df.index)


FAMILY_CLASSES = {
    MODEL_PROPHET: ProphetFleet,
    MODEL_XGB_GLOBAL: XGBGlobal,
    MODEL_XGB_LOCAL: XGBLocalFleet,
}


# ------------------------------------------------------------------------------------
# Step 2b: model artifact store on the volume
# ------------------------------------------------------------------------------------


def runs_root() -> Path:
    return MODELS_ROOT / "runs"


def serving_pointer_path() -> Path:
    return MODELS_ROOT / "serving.json"


def read_serving_pointer() -> dict | None:
    path = serving_pointer_path()
    try:
        if not path.exists():
            return None
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if not all(k in pointer for k in ("run_id", "model_name", "trained_at")):
            log.error("serving.json is missing required fields: %s", sorted(pointer))
            return None
        return pointer
    except Exception as exc:
        log.error("serving.json unreadable: %s", exc)
        return None


def write_serving_pointer(run_id: str, model_name: str, trained_at: str, ride_count: int) -> None:
    """The last write of a successful run. One tiny file, so switching serving_models can
    never be left half-done the way copying a 58 MB Prophet fleet could."""
    pointer = {
        "run_id": run_id,
        "model_name": model_name,
        "trained_at": trained_at,
        "ride_count": ride_count,
    }
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    path = serving_pointer_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        # Some FUSE mounts refuse rename; the file is small enough that a direct
        # rewrite is an acceptable fallback.
        path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        tmp.unlink(missing_ok=True)
    log.info("serving.json -> run %s (%s)", run_id, model_name)


def persist_serving_model(run_id: str, model) -> Path:
    """Write the serving family's artifacts into runs/<run_id>/. A failed run's
    directory is orphaned, never read, and cleaned up by retention -- so this can
    happen before the quality checks without risking the serving model."""
    run_dir = runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir)
    manifest = {
        "run_id": run_id,
        "model_name": model.family,
        "trained_at": model.trained_at,
        "entities": sorted(model.entities),
        "ride_count": len(model.entities),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("persisted %s artifacts for run %s (%d entities)", model.family, run_id, len(model.entities))
    return run_dir


def load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("manifest unreadable at %s: %s", path, exc)
        return None
    required = ("run_id", "model_name", "trained_at", "entities", "ride_count")
    missing = [k for k in required if k not in manifest]
    if missing:
        log.error("manifest at %s missing fields %s", path, missing)
        return None
    if manifest["ride_count"] != len(manifest["entities"]):
        log.error(
            "manifest at %s inconsistent: ride_count %d but %d entities",
            path,
            manifest["ride_count"],
            len(manifest["entities"]),
        )
        return None
    return manifest


def load_serving_model(pointer: dict):
    """Resolve serving.json -> runs/<run_id>/, validate the manifest, load the model,
    and confirm the loaded entity set matches the manifest before trusting it."""
    run_dir = runs_root() / pointer["run_id"]
    if not run_dir.is_dir():
        log.error("serving run directory missing: %s", run_dir)
        return None, None
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None, None
    if manifest["run_id"] != pointer["run_id"] or manifest["model_name"] != pointer["model_name"]:
        log.error(
            "manifest disagrees with serving.json: manifest says %s/%s, pointer says %s/%s",
            manifest["run_id"],
            manifest["model_name"],
            pointer["run_id"],
            pointer["model_name"],
        )
        return None, None
    family = FAMILY_CLASSES.get(manifest["model_name"])
    if family is None:
        log.error("unknown model family in manifest: %s", manifest["model_name"])
        return None, None
    try:
        model = family.load(run_dir, manifest)
    except Exception as exc:
        log.error("failed to load serving artifacts from %s: %s", run_dir, exc)
        return None, None
    if model.entities != set(manifest["entities"]):
        log.error(
            "loaded model entities do not match manifest (%d loaded vs %d listed)",
            len(model.entities),
            len(manifest["entities"]),
        )
        return None, None
    return model, manifest


def cleanup_runs() -> None:
    """Keep the RETAIN_RUNS most recent run directories plus whatever serving.json
    points at. Prophet is roughly 58 MB per run and free-tier storage is finite."""
    root = runs_root()
    if not root.is_dir():
        return
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    keep = {p.name for p in dirs[:RETAIN_RUNS]}
    pointer = read_serving_pointer()
    if pointer:
        keep.add(pointer["run_id"])
    for p in dirs:
        if p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)
            log.info("retention: removed old run directory %s", p.name)


# ------------------------------------------------------------------------------------
# Step 3: KPIs
# ------------------------------------------------------------------------------------


def compute_kpis(
    test_df: pd.DataFrame, preds: pd.DataFrame, high_wait_entities: set[str]
) -> dict[str, float]:
    """KPIs on the covered test rows. `coverage_pct` records how many rows the model
    actually answered for, so a model cannot look good by skipping hard rides."""
    feats = build_features(test_df)
    mask = preds["yhat"].notna()
    covered = feats[mask.to_numpy()]
    yhat = preds.loc[mask, "yhat"].to_numpy()
    actual = covered["wait_time"].to_numpy(dtype=float)
    err = yhat - actual
    abs_err = np.abs(err)
    ents = entity_of(covered)
    high = ents.isin(high_wait_entities).to_numpy()
    peak = ((covered["hour"] >= PEAK_HOURS[0]) & (covered["hour"] < PEAK_HOURS[1])).to_numpy()
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "pct_severe_miss": float((abs_err > SEVERE_MISS_MIN).mean() * 100),
        "pct_within_10min": float((abs_err <= WITHIN_MIN).mean() * 100),
        "bias": float(err.mean()),
        "high_wait_mae": float(abs_err[high].mean()) if high.any() else float("nan"),
        "peak_hours_mae": float(abs_err[peak].mean()) if peak.any() else float("nan"),
        "coverage_pct": float(mask.mean() * 100),
        "test_rows": float(mask.sum()),
    }


def build_kpi_table(
    kpis_by_model: dict[str, dict[str, float]],
    serving: str,
    trained_at_by_model: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for model_name, kpis in kpis_by_model.items():
        for kpi_name, value in kpis.items():
            rows.append(
                {
                    "model": model_name,
                    "kpi_name": kpi_name,
                    "kpi_value": float(value),
                    "is_serving": model_name == serving,
                    "trained_at": trained_at_by_model.get(model_name, ""),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------
# Step 4: gold predictions (forecast grid + backtest rows)
# ------------------------------------------------------------------------------------


def forecast_slots_per_park() -> dict[str, int]:
    slots = {}
    for park, (open_h, close_h, _) in PARK_HOURS.items():
        slots[park] = (close_h - open_h) * (60 // GRID_MINUTES) * FORECAST_DAYS
    return slots


def active_entities_frame(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per active entity with its park and latest display name."""
    work = silver.copy()
    work["_entity"] = entity_of(work)
    latest = work.sort_values("ts_local").groupby("_entity").tail(1)
    return latest[["_entity", "park_name", "ride_key", "ride_name"]].reset_index(drop=True)


def build_forecast(model, silver: pd.DataFrame, start_local: pd.Timestamp) -> pd.DataFrame:
    """Next 7 days x 30-minute slots x every active ride the model covers, bounded to
    each park's local operating hours."""
    entities = active_entities_frame(silver)
    covered = entities[entities["_entity"].isin(model.entities)]
    dropped = len(entities) - len(covered)
    if dropped:
        log.warning("forecast: model does not cover %d of %d active entities", dropped, len(entities))

    grid_start = (start_local + pd.Timedelta(minutes=GRID_MINUTES)).floor(f"{GRID_MINUTES}min")
    grid_end = start_local + pd.Timedelta(days=FORECAST_DAYS)
    all_slots = pd.date_range(grid_start, grid_end, freq=f"{GRID_MINUTES}min")

    frames = []
    for park, group in covered.groupby("park_name"):
        open_h, close_h, _ = PARK_HOURS[park]
        park_slots = all_slots[(all_slots.hour >= open_h) & (all_slots.hour < close_h)]
        frames.append(
            group.merge(pd.DataFrame({"ts_local": park_slots}), how="cross")
        )
    if not frames:
        raise PipelineError("forecast grid is empty: model covers no active entities")
    grid = pd.concat(frames, ignore_index=True)

    preds = model.predict(grid)
    grid["predicted_wait_min"] = preds["yhat"].to_numpy()
    grid["lower_bound"] = preds["lo"].to_numpy()
    grid["upper_bound"] = preds["hi"].to_numpy()
    grid = grid[grid["predicted_wait_min"].notna()].copy()

    grid["predicted_wait_min"] = grid["predicted_wait_min"].clip(lower=0.0)
    grid["lower_bound"] = grid["lower_bound"].clip(lower=0.0)
    grid["lower_bound"] = np.minimum(grid["lower_bound"], grid["predicted_wait_min"])
    grid["upper_bound"] = np.maximum(grid["upper_bound"], grid["predicted_wait_min"])

    grid["ts_utc"] = to_utc(grid["ts_local"])
    grid["row_kind"] = "forecast"
    grid["actual_wait_min"] = np.nan
    grid["error_min"] = np.nan
    grid["model_name"] = model.family
    return grid[
        [
            "row_kind",
            "park_name",
            "ride_key",
            "ride_name",
            "ts_local",
            "ts_utc",
            "predicted_wait_min",
            "lower_bound",
            "upper_bound",
            "actual_wait_min",
            "error_min",
            "model_name",
        ]
    ]


def build_backtest(test_df: pd.DataFrame, preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Test-set rows: predicted, actual, difference. The dashboard plots come from
    these, so they live in the predictions table alongside the forecast."""
    mask = preds["yhat"].notna()
    out = test_df[mask.to_numpy()].copy()
    out["predicted_wait_min"] = preds.loc[mask, "yhat"].clip(lower=0.0).to_numpy()
    out["lower_bound"] = np.minimum(
        preds.loc[mask, "lo"].clip(lower=0.0).to_numpy(), out["predicted_wait_min"].to_numpy()
    )
    out["upper_bound"] = np.maximum(
        preds.loc[mask, "hi"].to_numpy(), out["predicted_wait_min"].to_numpy()
    )
    out["actual_wait_min"] = out["wait_time"].astype(float)
    out["error_min"] = out["predicted_wait_min"] - out["actual_wait_min"]
    out["row_kind"] = "backtest"
    out["model_name"] = model_name
    return out[
        [
            "row_kind",
            "park_name",
            "ride_key",
            "ride_name",
            "ts_local",
            "ts_utc",
            "predicted_wait_min",
            "lower_bound",
            "upper_bound",
            "actual_wait_min",
            "error_min",
            "model_name",
        ]
    ]


def stamp(df: pd.DataFrame, run_id: str, run_status: str, model_trained_at: str) -> pd.DataFrame:
    """Provenance columns both gold tables carry. run_id is the run that TRAINED the
    serving model (it must agree with serving.json); generated_at is this run."""
    out = df.copy()
    out["run_id"] = run_id
    out["run_status"] = run_status
    out["generated_at"] = pd.Timestamp(utcnow()).tz_localize(None)
    out["model_trained_at"] = model_trained_at
    return out


# ------------------------------------------------------------------------------------
# Step 5b: quality checks
# ------------------------------------------------------------------------------------


def check_silver(silver: pd.DataFrame, last_row_count: int | None) -> list[str]:
    problems = []
    if last_row_count is not None and len(silver) < MIN_ROW_RATIO_VS_LAST * last_row_count:
        problems.append(
            f"silver row count collapsed: {len(silver)} vs {last_row_count} in _last"
        )
    n_entities = silver.groupby(["park_name", "ride_key"]).ngroups
    if n_entities < MIN_ATTRACTIONS:
        problems.append(f"only {n_entities} distinct attractions (need {MIN_ATTRACTIONS})")
    pct_zero = float((silver["wait_time"] == 0).mean() * 100)
    if pct_zero > MAX_ZERO_PCT:
        problems.append(
            f"zero-wait share {pct_zero:.1f}% exceeds {MAX_ZERO_PCT}% (timezone order suspect)"
        )
    for col in ("ride_key", "ts_local", "wait_time"):
        nulls = int(silver[col].isna().sum())
        if nulls:
            problems.append(f"{nulls} nulls in silver.{col}")
    ts_min, ts_max = silver["ts_local"].min(), silver["ts_local"].max()
    if ts_min < pd.Timestamp("2025-12-01"):
        problems.append(f"ts_local minimum {ts_min} predates trustworthy collection")
    if ts_max > now_local() + pd.Timedelta(days=1):
        problems.append(f"ts_local maximum {ts_max} is in the future")
    # The check that proves the timezone conversion ran the right way round. Row counts
    # per hour are nearly uniform on a fixed collection grid, so the discriminating
    # signal is which hour carries the highest MEAN wait: correctly converted data
    # peaks in the afternoon (11:00-20:00 local); inverted data relocates the true
    # afternoon peak to 20:00-23:00 and fills 08:00-16:00 with closed-park zeros.
    peak_hour = int(silver.groupby(silver["ts_local"].dt.hour)["wait_time"].mean().idxmax())
    if not (11 <= peak_hour <= 20):
        problems.append(
            f"mean wait peaks at local hour {peak_hour}:00; expected 11:00-20:00 "
            "(timezone conversion is likely inverted)"
        )
    return problems


def check_kpis(kpis: pd.DataFrame) -> list[str]:
    problems = []
    present = set(kpis["model"].unique())
    for model_name in CANDIDATES:
        if model_name not in present:
            problems.append(f"model {model_name} missing from KPI table")
    mae = kpis[kpis["kpi_name"] == "mae"].set_index("model")["kpi_value"]
    for model_name, value in mae.items():
        if not (math.isfinite(value) and value > 0):
            problems.append(f"MAE for {model_name} is {value}; must be finite and positive")
    serving_models = kpis[kpis["is_serving"]]["model"].unique()
    if len(serving_models) != 1:
        problems.append(f"expected exactly one serving model, found {list(serving_models)}")
    elif MODEL_BASELINE in mae.index and serving_models[0] in mae.index:
        serving_mae, base_mae = float(mae[serving_models[0]]), float(mae[MODEL_BASELINE])
        if not serving_mae < base_mae:
            problems.append(
                f"serving model {serving_models[0]} MAE {serving_mae:.2f} does not beat "
                f"the per-ride mean baseline {base_mae:.2f}"
            )
    return problems


def check_predictions(preds: pd.DataFrame, n_active_entities: int) -> list[str]:
    problems = []
    fc = preds[preds["row_kind"] == "forecast"]
    bt = preds[preds["row_kind"] == "backtest"]
    if fc.empty:
        return ["predictions table has no forecast rows"]

    covered = fc.groupby(["park_name", "ride_key"]).ngroups
    if n_active_entities and covered < MIN_FORECAST_RIDE_COVERAGE * n_active_entities:
        problems.append(
            f"forecast covers {covered} of {n_active_entities} active rides "
            f"(need {MIN_FORECAST_RIDE_COVERAGE:.0%})"
        )
    for col in ("park_name", "ride_key", "ride_name", "ts_local", "ts_utc",
                "predicted_wait_min", "lower_bound", "upper_bound", "model_name"):
        nulls = int(fc[col].isna().sum())
        if nulls:
            problems.append(f"{nulls} nulls in forecast.{col}")
    if (fc["predicted_wait_min"] < 0).any() or (fc["lower_bound"] < 0).any():
        problems.append("negative values in forecast predictions or lower bounds")
    bad_bounds = int(
        (
            (fc["lower_bound"] > fc["predicted_wait_min"])
            | (fc["predicted_wait_min"] > fc["upper_bound"])
        ).sum()
    )
    if bad_bounds:
        problems.append(f"{bad_bounds} forecast rows violate lower <= predicted <= upper")

    slots = forecast_slots_per_park()
    expected = sum(
        slots[park] for park, _ in fc.groupby(["park_name", "ride_key"]).groups.keys()
    )
    if not (0.75 * expected <= len(fc) <= 1.05 * expected):
        problems.append(
            f"forecast row count {len(fc)} outside sane band around {expected} (rides x slots)"
        )

    if not bt.empty:
        for col in ("predicted_wait_min", "actual_wait_min", "error_min"):
            nulls = int(bt[col].isna().sum())
            if nulls:
                problems.append(f"{nulls} nulls in backtest.{col}")
        if (bt["predicted_wait_min"] < 0).any():
            problems.append("negative predictions in backtest rows")
    return problems


# ------------------------------------------------------------------------------------
# Step 5: the promotion decision (a number compared against a number in a table)
# ------------------------------------------------------------------------------------


def previous_serving_from_kpis() -> dict | None:
    """Serving model name and MAE as recorded in gold.kpis_last, or None on first run."""
    if not table_exists(KPI_TABLE + LAST):
        return None
    kpis = read_table(KPI_TABLE + LAST)
    serving_rows = kpis[(kpis["is_serving"]) & (kpis["kpi_name"] == "mae")]
    if serving_rows.empty:
        log.warning("gold.kpis_last exists but records no serving MAE")
        return None
    row = serving_rows.iloc[0]
    return {
        "model_name": str(row["model"]),
        "mae": float(row["kpi_value"]),
        "run_id": str(row["run_id"]) if "run_id" in serving_rows.columns else None,
    }


def decide_promotion(new_mae: float, previous: dict | None) -> tuple[bool, str]:
    if previous is None:
        return True, "first run: no incumbent, shipping the new model"
    threshold = previous["mae"] * IMPROVEMENT_FACTOR
    if new_mae < threshold:
        return True, (
            f"new MAE {new_mae:.3f} beats incumbent {previous['mae']:.3f} "
            f"by more than 1 percent (threshold {threshold:.3f})"
        )
    return False, (
        f"new MAE {new_mae:.3f} does not beat incumbent {previous['mae']:.3f} "
        f"by 1 percent (threshold {threshold:.3f}); keeping the previous model"
    )


# ------------------------------------------------------------------------------------
# The pipeline
# ------------------------------------------------------------------------------------


def drop_current_tables() -> None:
    """A failed run may not leave partial _current tables around to confuse anyone.
    _last is never touched here."""
    for table in (SILVER_TABLE, KPI_TABLE, PRED_TABLE):
        try:
            drop_table(table + CURRENT)
            log.info("dropped %s", table + CURRENT)
        except Exception as exc:
            log.error("failed to drop %s: %s", table + CURRENT, exc)


def run_pipeline(run_id: str) -> dict:
    ensure_workspace()

    # ---- Step 1: bronze -> silver ------------------------------------------------
    raw = read_table(BRONZE_TABLE)
    log.info("bronze: %d rows read from %s", len(raw), BRONZE_TABLE)
    silver = build_silver(raw)

    last_count = table_count(SILVER_TABLE + LAST) if table_exists(SILVER_TABLE + LAST) else None
    write_table(SILVER_TABLE + CURRENT, silver)
    failures = check_silver(silver, last_count)
    if failures:
        for f in failures:
            log.error("silver check failed: %s", f)
        raise QualityCheckError("; ".join(failures))
    log.info("silver checks passed")

    # ---- Step 2: train -----------------------------------------------------------
    train_df, test_df = chrono_split(silver)
    high_wait = {
        e for e, m in train_df.groupby(entity_of(train_df))["wait_time"].mean().items()
        if m > HIGH_WAIT_MEAN_MIN
    }

    models = {
        MODEL_PROPHET: ProphetFleet.train(train_df),
        MODEL_XGB_GLOBAL: XGBGlobal.train(train_df),
        MODEL_XGB_LOCAL: XGBLocalFleet.train(train_df),
    }
    baseline = BaselineMean.train(train_df)

    # ---- Step 3: KPIs on the held-out 20 percent ---------------------------------
    kpis_by_model = {}
    test_preds = {}
    for name, model in {**models, MODEL_BASELINE: baseline}.items():
        preds = model.predict(test_df)
        test_preds[name] = preds
        kpis_by_model[name] = compute_kpis(test_df, preds, high_wait)
        log.info(
            "%s: mae %.3f rmse %.3f within10 %.1f%% coverage %.1f%%",
            name,
            kpis_by_model[name]["mae"],
            kpis_by_model[name]["rmse"],
            kpis_by_model[name]["pct_within_10min"],
            kpis_by_model[name]["coverage_pct"],
        )

    best_candidate = min(CANDIDATES, key=lambda m: kpis_by_model[m]["mae"])
    new_mae = kpis_by_model[best_candidate]["mae"]
    if not (math.isfinite(new_mae) and new_mae > 0):
        raise PipelineError(f"best new MAE is {new_mae}; evaluation is broken, refusing to decide")
    log.info("best new model: %s (mae %.3f)", best_candidate, new_mae)

    # ---- Step 5: promotion decision ----------------------------------------------
    previous = previous_serving_from_kpis()
    ship_new, reason = decide_promotion(new_mae, previous)
    log.info("promotion decision: %s", reason)

    start_local = now_local()
    if ship_new:
        serving_model = models[best_candidate]
        run_status = STATUS_FRESH
        serving_run_id = run_id
        # Artifacts go to runs/<run_id>/ now; if anything later fails, the directory
        # is orphaned and serving.json still points at the old run.
        persist_serving_model(run_id, serving_model)
        trained_at_by_model = {name: m.trained_at for name, m in models.items()}
        trained_at_by_model[MODEL_BASELINE] = serving_model.trained_at
        kpi_df = build_kpi_table(kpis_by_model, best_candidate, trained_at_by_model)
        forecast = build_forecast(serving_model, silver, start_local)
        # Backtest rows for every candidate plus the per-ride mean baseline: the
        # dashboard plots each error distribution side by side, and the baseline
        # panel must use the same train-mean baseline the KPI table was measured on.
        backtest = pd.concat(
            [
                build_backtest(test_df, test_preds[name], name)
                for name in (*CANDIDATES, MODEL_BASELINE)
            ],
            ignore_index=True,
        )
        model_trained_at = serving_model.trained_at
    else:
        pointer = read_serving_pointer()
        if pointer is None:
            raise PipelineError(
                "promotion gate chose the previous model but serving.json is missing or "
                "unreadable; refusing to guess (unconditional promotion is the failure "
                "mode this pipeline exists to prevent)"
            )
        if previous and previous.get("run_id") and previous["run_id"] != pointer["run_id"]:
            raise PipelineError(
                f"kpis_last.run_id {previous['run_id']} disagrees with serving.json "
                f"run_id {pointer['run_id']}; refusing to serve mismatched model and KPIs"
            )
        serving_model, manifest = load_serving_model(pointer)
        if serving_model is None:
            raise PipelineError(
                f"could not reload the previous serving model from run {pointer['run_id']}; "
                "job fails rather than promoting unvalidated output"
            )
        run_status = STATUS_KEPT
        serving_run_id = pointer["run_id"]
        model_trained_at = pointer["trained_at"]
        # Previous KPI values keep serving: the model has not changed, so its measured
        # KPIs are still the correct ones to display.
        prev_kpis = read_table(KPI_TABLE + LAST)
        kpi_df = prev_kpis[["model", "kpi_name", "kpi_value", "is_serving", "trained_at"]].copy()
        forecast = build_forecast(serving_model, silver, start_local)
        prev_preds = read_table(PRED_TABLE + LAST)
        backtest = prev_preds[prev_preds["row_kind"] == "backtest"][
            [
                "row_kind", "park_name", "ride_key", "ride_name", "ts_local", "ts_utc",
                "predicted_wait_min", "lower_bound", "upper_bound", "actual_wait_min",
                "error_min", "model_name",
            ]
        ].copy()

    # ---- Step 4: gold tables -----------------------------------------------------
    kpi_df = stamp(kpi_df, serving_run_id, run_status, model_trained_at)
    preds_df = stamp(pd.concat([forecast, backtest], ignore_index=True), serving_run_id, run_status, model_trained_at)
    write_table(KPI_TABLE + CURRENT, kpi_df)
    write_table(PRED_TABLE + CURRENT, preds_df)

    # ---- Step 5b: gold checks, then promote --------------------------------------
    n_active = silver.groupby(["park_name", "ride_key"]).ngroups
    failures = check_kpis(kpi_df) + check_predictions(preds_df, n_active)
    if failures:
        for f in failures:
            log.error("gold check failed: %s", f)
        raise QualityCheckError("; ".join(failures))
    log.info("gold checks passed")

    for table in (SILVER_TABLE, KPI_TABLE, PRED_TABLE):
        promote_table(table + CURRENT, table + LAST)

    if ship_new:
        write_serving_pointer(
            run_id, serving_model.family, serving_model.trained_at, len(serving_model.entities)
        )
    else:
        log.info(
            "serving.json unchanged: still run %s (%s)", serving_run_id, serving_model.family
        )
    cleanup_runs()

    return {
        "run_id": run_id,
        "run_status": run_status,
        "serving_model": serving_model.family,
        "serving_run_id": serving_run_id,
        "new_model_mae": round(new_mae, 3),
        "previous_mae": round(previous["mae"], 3) if previous else None,
        "decision": reason,
        "silver_rows": len(silver),
        "attractions": n_active,
        "forecast_rows": int((preds_df["row_kind"] == "forecast").sum()),
        "backtest_rows": int((preds_df["row_kind"] == "backtest").sum()),
    }


# ------------------------------------------------------------------------------------
# Step 5c: fallback. Runs on ANY pipeline failure so the forecast window is never
# stale. Protects the dashboard; never papers over the failure (main still re-raises).
# ------------------------------------------------------------------------------------


def run_fallback() -> bool:
    """Re-score the upcoming week with the previous serving model and swap ONLY the forecast
    rows of gold.predictions_last. Returns True if the forecast was refreshed. Never
    raises: a broken fallback must not mask the original failure."""
    try:
        if not table_exists(SILVER_TABLE + LAST):
            log.warning("fallback: no silver _last table exists; nothing to fall back to")
            return False
        pointer = read_serving_pointer()
        if pointer is None:
            log.warning("fallback: no serving.json; nothing to fall back to")
            return False

        if table_exists(KPI_TABLE + LAST):
            kpis_last = read_table(KPI_TABLE + LAST)
            if "run_id" in kpis_last.columns and not kpis_last.empty:
                kpi_run_id = str(kpis_last["run_id"].iloc[0])
                if kpi_run_id != pointer["run_id"]:
                    log.error(
                        "fallback: kpis_last.run_id %s disagrees with serving.json run_id %s "
                        "-- a previous run promoted tables but never moved the pointer. "
                        "Changing NOTHING; resolve by editing serving.json or re-running "
                        "a full successful pipeline.",
                        kpi_run_id,
                        pointer["run_id"],
                    )
                    return False

        model, manifest = load_serving_model(pointer)
        if model is None:
            log.warning("fallback: serving artifacts incomplete; changing nothing")
            return False

        silver_last = read_table(SILVER_TABLE + LAST)
        forecast = build_forecast(model, silver_last, now_local())
        forecast = stamp(forecast, pointer["run_id"], STATUS_FALLBACK, pointer["trained_at"])

        n_active = silver_last.groupby(["park_name", "ride_key"]).ngroups
        failures = check_predictions(forecast, n_active)
        if failures:
            for f in failures:
                log.error("fallback forecast failed checks: %s", f)
            log.warning("fallback: refusing to publish an unvalidated forecast; changing nothing")
            return False

        if table_exists(PRED_TABLE + LAST):
            prev = read_table(PRED_TABLE + LAST)
            backtest = prev[prev["row_kind"] == "backtest"].copy()
            if not backtest.empty:
                # Keep the measured backtest exactly as it was; only restamp the
                # table-level provenance so the whole table tells one story.
                backtest["run_id"] = pointer["run_id"]
                backtest["run_status"] = STATUS_FALLBACK
                backtest["generated_at"] = forecast["generated_at"].iloc[0]
                backtest["model_trained_at"] = pointer["trained_at"]
                backtest = backtest[list(forecast.columns)]
        else:
            backtest = forecast.iloc[0:0]

        combined = pd.concat([forecast, backtest], ignore_index=True)
        # A Delta overwrite is one atomic commit; readers see old or new, never partial.
        write_table(PRED_TABLE + LAST, combined)
        log.info(
            "fallback: refreshed forecast window with the previous serving model %s from run %s "
            "(%d forecast rows). gold.kpis_last untouched.",
            pointer["model_name"],
            pointer["run_id"],
            len(forecast),
        )
        return True
    except Exception:
        log.exception("fallback itself failed; nothing was changed")
        return False


def main() -> dict:
    run_id = make_run_id()
    log.info("pipeline run %s starting", run_id)
    try:
        summary = run_pipeline(run_id)
    except Exception as exc:
        log.error("pipeline run %s FAILED: %s", run_id, exc)
        try:
            drop_current_tables()
        except Exception:
            log.exception("dropping _current tables failed")
        refreshed = run_fallback()
        log.error(
            "run %s failed; _last tables untouched; forecast %s; re-raising so the job "
            "reports failure",
            run_id,
            "refreshed by fallback" if refreshed else "NOT refreshed (nothing to fall back to)",
        )
        raise
    log.info("pipeline run %s succeeded: %s", run_id, json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
