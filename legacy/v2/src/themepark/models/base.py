"""The contract every model family implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..naming import entity_key

# Columns a caller must supply.
FIT_COLUMNS = ["park_name", "ride_key", "ts_local", "wait_time"]
PREDICT_COLUMNS = ["park_name", "ride_key", "ts_local"]

# Columns every model returns. Bounds are always present, even for families with no
# native notion of uncertainty, because the downstream routing optimiser needs a
# worst-case wait to plan against and should not have to special-case the model in use.
PREDICTION_COLUMNS = ["yhat", "yhat_lower", "yhat_upper"]


class WaitTimeModel(ABC):
    """Predicts wait minutes for (park, ride, local timestamp).

    Implementations must be picklable: MLflow serialises the fitted instance as the model
    artefact.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.entities_: set[str] = set()
        self.fitted_: bool = False
        self.train_rows_: int = 0

    @staticmethod
    def entities_of(df: pd.DataFrame) -> pd.Series:
        """Fully-qualified entity key per row. Two parks may share a ride name."""
        return df.apply(lambda r: entity_key(r["park_name"], r["ride_key"]), axis=1)

    def _validate_fit(self, df: pd.DataFrame) -> None:
        missing = [c for c in FIT_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"{self.name}.fit is missing required columns: {missing}")

    def _validate_predict(self, df: pd.DataFrame) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} has not been fitted")
        missing = [c for c in PREDICT_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"{self.name}.predict is missing required columns: {missing}")

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> WaitTimeModel:
        """Train on park-local observations. Returns self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return yhat / yhat_lower / yhat_upper, indexed like `X`."""

    def coverage(self, X: pd.DataFrame) -> float:
        """Share of requested rows this model can actually serve from a fitted entity.

        This is the tripwire for the v1 entity-resolution defect. There, unmatched rides
        fell into a bare `except: pass` and were dropped from the metrics, so accuracy
        appeared to improve as coverage silently collapsed. Coverage is now measured and
        gated on -- see `themepark.promote`.
        """
        if X.empty:
            return 1.0
        return float(self.entities_of(X).isin(self.entities_).mean())

    def _empty_predictions(self, X: pd.DataFrame, fill: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {"yhat": fill, "yhat_lower": fill, "yhat_upper": fill}, index=X.index, dtype="float64"
        )

    @staticmethod
    def _finalize(preds: pd.DataFrame) -> pd.DataFrame:
        """Clip to physically possible values and keep the bounds ordered.

        A negative queue is meaningless, and an optimiser handed `yhat_lower > yhat` will
        produce nonsense routes.
        """
        out = preds[PREDICTION_COLUMNS].astype("float64").clip(lower=0.0)
        out["yhat_lower"] = out[["yhat_lower", "yhat"]].min(axis=1)
        out["yhat_upper"] = out[["yhat_upper", "yhat"]].max(axis=1)
        return out

    def __repr__(self) -> str:
        state = f"{len(self.entities_)} entities, {self.train_rows_:,} rows" if self.fitted_ else "unfitted"
        return f"<{type(self).__name__} {self.name} ({state})>"
