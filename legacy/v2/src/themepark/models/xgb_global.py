"""One gradient-boosted model across every park and ride.

Park and ride are passed as native XGBoost categoricals rather than one-hot columns, so
the trees can split on ride identity directly and rides with sparse history borrow
structure from the rest of the fleet. That pooling is why this family was the strongest
performer on the v1 weekend/holiday subset.

Point estimates only -- gradient boosting has no native predictive interval -- so bounds
come from the empirical distribution of training residuals per ride.
"""

from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb

from ..features import TEMPORAL_FEATURES, build_features
from .base import WaitTimeModel

log = logging.getLogger(__name__)

FEATURES = ["park_name", "ride_key", *TEMPORAL_FEATURES]

DEFAULT_PARAMS = dict(
    enable_categorical=True,
    tree_method="hist",
    max_depth=8,
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)


class XGBGlobal(WaitTimeModel):
    name = "xgb_global"

    def __init__(self, params: dict | None = None, residual_interval: float = 0.80) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.residual_interval = residual_interval
        self.model_: xgb.XGBRegressor | None = None
        self.park_categories_: list[str] = []
        self.ride_categories_: list[str] = []
        self.residual_lo_: dict[str, float] = {}
        self.residual_hi_: dict[str, float] = {}

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the categorical ontology captured at fit time.

        Pinning the category lists is what keeps training and serving aligned: pandas
        assigns category codes by order of appearance, so letting each batch infer its own
        would silently remap ride identities at inference.
        """
        feats = build_features(df)
        feats["park_name"] = feats["park_name"].astype(
            pd.CategoricalDtype(categories=self.park_categories_)
        )
        feats["ride_key"] = feats["ride_key"].astype(
            pd.CategoricalDtype(categories=self.ride_categories_)
        )
        return feats[FEATURES]

    def fit(self, df: pd.DataFrame) -> XGBGlobal:
        self._validate_fit(df)
        work = df.copy()
        work["_entity"] = self.entities_of(work)

        self.park_categories_ = sorted(work["park_name"].astype(str).unique())
        self.ride_categories_ = sorted(work["ride_key"].astype(str).unique())

        X = self._encode(work)
        y = work["wait_time"].astype(float)

        self.model_ = xgb.XGBRegressor(**self.params).fit(X, y)

        residual = y.to_numpy() - self.model_.predict(X)
        lo_q = (1.0 - self.residual_interval) / 2.0
        by_entity = pd.DataFrame(
            {"_entity": work["_entity"].to_numpy(), "r": residual}
        ).groupby("_entity")["r"]
        self.residual_lo_ = by_entity.quantile(lo_q).to_dict()
        self.residual_hi_ = by_entity.quantile(1.0 - lo_q).to_dict()

        self.entities_ = set(work["_entity"].unique())
        self.train_rows_ = len(work)
        self.fitted_ = True
        log.info("XGBGlobal fitted on %d rows, %d entities", self.train_rows_, len(self.entities_))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_predict(X)
        yhat = self.model_.predict(self._encode(X))
        keys = self.entities_of(X)
        lo = keys.map(self.residual_lo_).fillna(0.0).to_numpy()
        hi = keys.map(self.residual_hi_).fillna(0.0).to_numpy()
        return self._finalize(
            pd.DataFrame(
                {"yhat": yhat, "yhat_lower": yhat + lo, "yhat_upper": yhat + hi}, index=X.index
            )
        )

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            return pd.Series(dtype=float)
        return pd.Series(self.model_.feature_importances_, index=FEATURES).sort_values(
            ascending=False
        )
