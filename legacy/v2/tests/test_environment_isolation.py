"""Each Databricks task must import under its own environment, not the union of all three.

The weekly job deliberately splits into three environments so that only `train` pays for
Prophet and cmdstan -- the dominant cost on the free tier, where the actual compute is
under a minute. That saving is only real if the lean tasks genuinely do not need the heavy
packages.

`verify` failed with `ModuleNotFoundError: No module named 'prophet'` despite importing
nothing of the sort. The chain: verify -> themepark.verify -> .evaluate ->
`from .models.base import WaitTimeModel`. Importing a submodule executes the parent
package's __init__ first, and that eagerly imported all four families.

Static import analysis missed it precisely because it resolved `.models.base` straight to
base.py. So these tests do not analyse -- they hide the packages an environment does not
install and import for real, which is the only check that models what Databricks does.
"""

import builtins
import importlib
import sys

import pytest

# From databricks.yml. `light` and `metrics` must survive without prophet or xgboost.
ENVIRONMENTS = {
    "light": {"prophet", "xgboost", "mlflow"},
    "metrics": {"prophet", "xgboost", "mlflow"},
    "ml": set(),
}

# The themepark modules each task's entry point imports at module level.
TASK_IMPORTS = {
    "light": ["themepark.config", "themepark.silver", "themepark.filters"],
    "metrics": ["themepark.config", "themepark.verify"],
    "ml": ["themepark.config", "themepark.train", "themepark.score", "themepark.promote"],
}


@pytest.fixture
def hide_packages():
    """Make named top-level packages unimportable, as a leaner environment would.

    Replacing `builtins.__import__` is enough: an `import x` statement always routes
    through it, so the guard fires before sys.modules is ever consulted. Only themepark's
    own modules are evicted, so they re-execute their imports under the guard -- numpy and
    pandas are left alone, because reloading a C extension raises "cannot load module more
    than once per process" and would fail the test for a reason that has nothing to do
    with what it checks.
    """
    real_import = builtins.__import__
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "themepark"}

    def blocker(blocked):
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        for module in list(sys.modules):
            if module.split(".")[0] == "themepark":
                sys.modules.pop(module, None)

    yield blocker

    builtins.__import__ = real_import
    for module in list(sys.modules):
        if module.split(".")[0] == "themepark":
            sys.modules.pop(module, None)
    sys.modules.update(saved)


@pytest.mark.parametrize("environment", sorted(ENVIRONMENTS))
def test_task_modules_import_under_their_own_environment(environment, hide_packages):
    """The regression. `metrics` had no prophet, and verify imported it transitively."""
    blocked = ENVIRONMENTS[environment]
    if blocked:
        hide_packages(blocked)

    for module in TASK_IMPORTS[environment]:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            pytest.fail(
                f"{environment} environment cannot import {module}: {exc}. "
                f"Either the import is genuinely needed -- add it to the environment in "
                f"databricks.yml -- or it leaked in transitively and should be made lazy."
            )


def test_the_model_interface_costs_nothing_to_import(hide_packages):
    """`WaitTimeModel` is the shared contract; needing Prophet to see it defeats the split."""
    hide_packages({"prophet", "xgboost"})
    importlib.import_module("themepark.models.base")
    importlib.import_module("themepark.models")  # the __init__ that used to be eager


def test_candidate_names_needs_no_model_dependencies(hide_packages):
    """Listing what could be trained must not import what trains it."""
    hide_packages({"prophet", "xgboost"})
    models = importlib.import_module("themepark.models")
    assert models.candidate_names() == [
        "baseline",
        "prophet_fleet",
        "xgb_global",
        "xgb_local_fleet",
    ]


def test_baseline_resolves_without_prophet(hide_packages):
    """get_candidate must import one family, not all four."""
    hide_packages({"prophet", "xgboost"})
    models = importlib.import_module("themepark.models")
    assert models.get_candidate("baseline").__name__ == "BaselineModel"


def test_heavy_families_still_resolve_when_available():
    """Laziness must not break the path that actually trains."""
    from themepark.models import get_candidate

    assert get_candidate("prophet_fleet").__name__ == "ProphetFleet"
    assert get_candidate("xgb_global").__name__ == "XGBGlobal"


def test_unknown_candidate_names_are_reported_clearly():
    from themepark.models import get_candidate

    with pytest.raises(KeyError, match="unknown candidate"):
        get_candidate("no_such_model")
