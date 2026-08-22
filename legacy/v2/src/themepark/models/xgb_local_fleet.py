"""One gradient-boosted model per attraction.

Kept as a challenger because it genuinely won on the v1 live snapshot (9.78 min MAE
versus Prophet's 12.09), even though it lost on the offline holdout. That disagreement
between offline and live is precisely why the weekly gate re-scores every family on fresh
data instead of trusting a ranking fixed months ago.

Two deliberate departures from v1:

* **No hyperparameter search.** v1 ran `RandomizedSearchCV(cv=3)`, which is plain K-Fold:
  on a time series it puts future rows in the validation fold and selects parameters that
  look better than they are. Fixed, conservative parameters remove the leak and cut
  training time substantially.
* **Shallower trees.** v1 searched up to `max_depth=10` with 300 estimators on roughly 4k
  rows per ride, producing 3.8 MB of model for each one -- 320 MB of artefacts for a
  problem this size. Depth 6 with 200 estimators fits the data without memorising it.
"""

from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb

from ..features import TEMPORAL_FEATURES, build_features
from .base import WaitTimeModel

log = logging.getLogger(__name__)

MIN_ROWS_PER_ENTITY = 100

DEFAULT_PARAMS = dict(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)


class XGBLocalFleet(WaitTimeModel):
    name = "xgb_local_fleet"

    def __init__(self, params: dict | None = None, residual_interval: float = 0.80) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.residual_interval = residual_interval
        self.models_: dict[str, xgb.XGBRegressor] = {}
        self.residual_lo_: dict[str, float] = {}
        self.residual_hi_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> XGBLocalFleet:
        self._validate_fit(df)
        work = df.copy()
        work["_entity"] = self.entities_of(work)

        lo_q = (1.0 - self.residual_interval) / 2.0
        trained, skipped = 0, 0

        for entity, group in work.groupby("_entity", sort=False):
            if len(group) < MIN_ROWS_PER_ENTITY:
                skipped += 1
                continue
            feats = build_features(group)
            X, y = feats[TEMPORAL_FEATURES], group["wait_time"].astype(float)
            try:
                model = xgb.XGBRegressor(**self.params).fit(X, y)
            except Exception as exc:
                log.warning("XGB local fit failed for %s: %s", entity, exc)
                skipped += 1
                continue

            residual = pd.Series(y.to_numpy() - model.predict(X))
            self.models_[entity] = model
            self.residual_lo_[entity] = float(residual.quantile(lo_q))
            self.residual_hi_[entity] = float(residual.quantile(1.0 - lo_q))
            trained += 1

        self.entities_ = set(self.models_)
        self.train_rows_ = len(work)
        self.fitted_ = True
        log.info("XGBLocalFleet fitted %d entities (%d skipped)", trained, skipped)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_predict(X)
        keys = self.entities_of(X)
        out = self._empty_predictions(X)
        feats = build_features(X)

        for entity, idx in keys.groupby(keys).groups.items():
            model = self.models_.get(str(entity))
            if model is None:
                continue
            yhat = model.predict(feats.loc[idx, TEMPORAL_FEATURES])
            out.loc[idx, "yhat"] = yhat
            out.loc[idx, "yhat_lower"] = yhat + self.residual_lo_.get(str(entity), 0.0)
            out.loc[idx, "yhat_upper"] = yhat + self.residual_hi_.get(str(entity), 0.0)

        return self._finalize(out)
