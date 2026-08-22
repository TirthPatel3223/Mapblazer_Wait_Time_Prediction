"""One Prophet per attraction, wrapped as a single addressable model.

v1 wrote 120 loose JSON files and looked them up by constructing a filesystem path from
the ride name -- the mechanism that lost 43 rides. The fleet holds its members in a dict
keyed by `naming.entity_key`, so resolution is exact and a miss is measurable via
`coverage()` rather than silent.

Prophet is the only family here that produces genuine predictive intervals. Those bounds
are what the downstream time-dependent routing optimiser should plan against: a route
built on the point estimate alone has no notion of the risk it is taking.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from .base import WaitTimeModel

log = logging.getLogger(__name__)

# Prophet needs a couple of days of 30-minute observations before a weekly seasonality
# term means anything.
MIN_ROWS_PER_ENTITY = 100


class ProphetFleet(WaitTimeModel):
    name = "prophet_fleet"

    def __init__(self, interval_width: float = 0.80, uncertainty_samples: int = 200) -> None:
        super().__init__()
        self.interval_width = interval_width
        # The default of 1000 simulated trend draws dominates prediction time and buys
        # nothing at 30-minute granularity.
        self.uncertainty_samples = uncertainty_samples
        self._serialized: dict[str, str] = {}
        self._cache: dict[str, Prophet] = {}

    # Prophet holds a compiled Stan backend that does not pickle. Keep only the JSON
    # payloads in the pickled state and rebuild on demand in the worker that needs them.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def _build(self) -> Prophet:
        model = Prophet(
            growth="flat",
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
        )
        model.add_country_holidays(country_name="US")
        return model

    def fit(self, df: pd.DataFrame) -> ProphetFleet:
        self._validate_fit(df)
        work = df.copy()
        work["_entity"] = self.entities_of(work)

        trained, skipped = 0, 0
        for entity, group in work.groupby("_entity", sort=False):
            series = (
                group[["ts_local", "wait_time"]]
                .rename(columns={"ts_local": "ds", "wait_time": "y"})
                .dropna()
                .sort_values("ds")
            )
            if len(series) < MIN_ROWS_PER_ENTITY:
                skipped += 1
                continue
            try:
                model = self._build().fit(series)
            except Exception as exc:
                # One pathological ride must not take down the weekly retrain. It simply
                # drops out of coverage, which the promotion gate can see.
                log.warning("Prophet fit failed for %s: %s", entity, exc)
                skipped += 1
                continue
            self._serialized[entity] = model_to_json(model)
            trained += 1

        self.entities_ = set(self._serialized)
        self.train_rows_ = len(work)
        self.fitted_ = True
        log.info("ProphetFleet fitted %d entities (%d skipped)", trained, skipped)
        return self

    def _model_for(self, entity: str) -> Prophet | None:
        if entity not in self._serialized:
            return None
        if entity not in self._cache:
            self._cache[entity] = model_from_json(self._serialized[entity])
        return self._cache[entity]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_predict(X)
        keys = self.entities_of(X)
        out = self._empty_predictions(X)

        for entity, idx in keys.groupby(keys).groups.items():
            model = self._model_for(str(entity))
            if model is None:
                continue  # unseen ride: leaves zeros, and lowers coverage()
            future = pd.DataFrame({"ds": pd.to_datetime(X.loc[idx, "ts_local"]).values})
            forecast = model.predict(future)
            out.loc[idx, "yhat"] = forecast["yhat"].to_numpy()
            out.loc[idx, "yhat_lower"] = forecast["yhat_lower"].to_numpy()
            out.loc[idx, "yhat_upper"] = forecast["yhat_upper"].to_numpy()

        return self._finalize(out)

    def artifact_bytes(self) -> int:
        """Rough serialised size -- the fleet is the largest artefact in the registry."""
        return sum(len(v) for v in self._serialized.values())

    def to_json(self) -> str:
        return json.dumps(self._serialized)
