"""Per-ride historical mean: the floor every challenger has to clear.

v1 recovered this number by reaching into a serialised Prophet artefact and averaging its
stored training history, which meant the baseline only existed for rides that happened to
have a Prophet model. It is a first-class candidate here.

On the live snapshot in the v1 repo this baseline carried a -15.2 minute bias -- it
systematically under-predicted, because a mean taken over mostly-closed hours is far below
the mean during operating hours. That bias is invisible to MAE alone, which is why the
scorecard reports signed error too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import WaitTimeModel


class BaselineModel(WaitTimeModel):
    name = "baseline"

    def __init__(self, interval_width: float = 0.80) -> None:
        super().__init__()
        self.interval_width = interval_width
        self.means_: dict[str, float] = {}
        self.lower_: dict[str, float] = {}
        self.upper_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame) -> BaselineModel:
        self._validate_fit(df)
        work = df.copy()
        work["_entity"] = self.entities_of(work)

        lo_q = (1.0 - self.interval_width) / 2.0
        grouped = work.groupby("_entity")["wait_time"]
        self.means_ = grouped.mean().to_dict()
        self.lower_ = grouped.quantile(lo_q).to_dict()
        self.upper_ = grouped.quantile(1.0 - lo_q).to_dict()

        self.global_mean_ = float(work["wait_time"].mean())
        self.entities_ = set(self.means_)
        self.train_rows_ = len(work)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_predict(X)
        keys = self.entities_of(X)
        g = self.global_mean_
        return self._finalize(
            pd.DataFrame(
                {
                    "yhat": keys.map(self.means_).fillna(g).astype(float),
                    "yhat_lower": keys.map(self.lower_).fillna(g).astype(float),
                    "yhat_upper": keys.map(self.upper_).fillna(g).astype(float),
                },
                index=X.index,
            ).replace([np.inf, -np.inf], g)
        )
