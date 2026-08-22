"""Model families, all behind one interface.

v1 kept these apart: Prophet models were JSON files on disk looked up by path, XGBoost
models were a parallel tree of JSON files, and the baseline was recovered by reaching
into a Prophet artefact's stored history. Comparing them meant three different code paths
and a filesystem lookup that silently returned nothing when a name did not match.

Here every family implements `WaitTimeModel`, so the backtest, the promotion gate and the
scoring job are written once and work against whichever family currently holds champion.
"""

from .base import WaitTimeModel
from .baseline import BaselineModel
from .prophet_fleet import ProphetFleet
from .xgb_global import XGBGlobal
from .xgb_local_fleet import XGBLocalFleet

CANDIDATE_REGISTRY: dict[str, type[WaitTimeModel]] = {
    "baseline": BaselineModel,
    "prophet_fleet": ProphetFleet,
    "xgb_global": XGBGlobal,
    "xgb_local_fleet": XGBLocalFleet,
}

__all__ = [
    "WaitTimeModel",
    "BaselineModel",
    "ProphetFleet",
    "XGBGlobal",
    "XGBLocalFleet",
    "CANDIDATE_REGISTRY",
]
