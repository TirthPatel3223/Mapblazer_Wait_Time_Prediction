"""Model families, all behind one interface.

v1 kept these apart: Prophet models were JSON files on disk looked up by path, XGBoost
models were a parallel tree of JSON files, and the baseline was recovered by reaching
into a Prophet artefact's stored history. Comparing them meant three different code paths
and a filesystem lookup that silently returned nothing when a name did not match.

Here every family implements `WaitTimeModel`, so the backtest, the promotion gate and the
scoring job are written once and work against whichever family currently holds champion.

Imports are lazy, and that is load-bearing rather than tidiness. Only `train` needs
Prophet and XGBoost, so only its task installs them; `verify` runs on a lean environment
with just `holidays`. But `evaluate.py` does `from .models.base import WaitTimeModel`, and
importing a submodule executes this file first -- so eagerly importing the four families
here dragged Prophet into every task that touched the interface, and failed the verify
task with `ModuleNotFoundError: No module named 'prophet'` after it had installed nothing
wrong. Resolving the classes on attribute access instead keeps `base` free of the heavy
dependencies its consumers do not need.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base import WaitTimeModel

# name -> submodule holding it. The submodule is imported only when the name is used.
_FAMILIES = {
    "BaselineModel": ".baseline",
    "ProphetFleet": ".prophet_fleet",
    "XGBGlobal": ".xgb_global",
    "XGBLocalFleet": ".xgb_local_fleet",
}

# Candidate name as it appears in the scorecard -> the class implementing it.
_REGISTRY = {
    "baseline": "BaselineModel",
    "prophet_fleet": "ProphetFleet",
    "xgb_global": "XGBGlobal",
    "xgb_local_fleet": "XGBLocalFleet",
}

if TYPE_CHECKING:  # for type checkers and editors only; never executed at runtime
    from .baseline import BaselineModel
    from .prophet_fleet import ProphetFleet
    from .xgb_global import XGBGlobal
    from .xgb_local_fleet import XGBLocalFleet


def __getattr__(name: str):
    """PEP 562 module-level attribute access, so `from .models import ProphetFleet` works."""
    if name in _FAMILIES:
        return getattr(importlib.import_module(_FAMILIES[name], __name__), name)
    if name == "CANDIDATE_REGISTRY":
        return candidate_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


def candidate_registry() -> dict[str, type[WaitTimeModel]]:
    """The families the weekly job trains, resolved on demand.

    A function rather than a module-level dict: building the dict requires importing every
    class, which is exactly the eager import this module exists to avoid.
    """
    return {name: __getattr__(attr) for name, attr in _REGISTRY.items()}


def get_candidate(name: str) -> type[WaitTimeModel]:
    """One family by its scorecard name, importing only that family's dependencies.

    Preferred over `candidate_registry()` on any path that trains a subset: asking for
    `baseline` alone should not import Prophet.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown candidate {name!r}; known: {', '.join(_REGISTRY)}")
    return __getattr__(_REGISTRY[name])


def candidate_names() -> list[str]:
    """Candidate names without importing a single model implementation."""
    return list(_REGISTRY)


__all__ = [
    "WaitTimeModel",
    "BaselineModel",
    "ProphetFleet",
    "XGBGlobal",
    "XGBLocalFleet",
    "CANDIDATE_REGISTRY",
    "candidate_registry",
    "get_candidate",
    "candidate_names",
]
