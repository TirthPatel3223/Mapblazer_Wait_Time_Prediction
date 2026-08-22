"""A registered model must load without this repo already on sys.path.

MLflow records `code: null` unless told otherwise, so a pyfunc model that pickles classes
from `themepark` is loadable only by a process that can already import `themepark`. That
is not a theoretical concern: it silently disabled the promotion gate. `load_champion()`
raised ModuleNotFoundError, the caller treats an unloadable champion as "none registered",
and the run promoted unconditionally with `no champion registered` -- a gate that can
never refuse. It is also what would break model serving, where nothing but the artifact
exists.

These tests load in a subprocess with the package deliberately invisible, because an
in-process check would pass on the strength of the already-imported module.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from themepark.train import ChampionLoadError, load_champion

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

pytest.importorskip("mlflow")
pytest.importorskip("prophet")


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=900,
    )


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """A champion saved exactly as train.py saves it, code_paths included."""
    path = tmp_path_factory.mktemp("model") / "champion"
    result = _run(
        f'''
        import sys, warnings, logging
        warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
        sys.path.insert(0, r"{SRC}")
        import pandas as pd, mlflow
        from themepark.models import get_candidate
        from themepark.train import pyfunc_adapter, PACKAGE_ROOT

        ts = pd.date_range("2026-01-01 08:00", periods=400, freq="30min")
        frame = pd.DataFrame({{
            "park_name": "Disneyland", "ride_key": "Space_Mountain",
            "ride_name": "Space Mountain", "ts_local": ts,
            "wait_time": [5, 10, 20, 15] * 100,
        }})
        model = get_candidate("prophet_fleet")().fit(frame)
        mlflow.pyfunc.save_model(
            path=r"{path}",
            python_model=pyfunc_adapter(model),
            code_paths=[PACKAGE_ROOT],
        )
        print("saved")
        ''',
        cwd=ROOT,
    )
    assert path.exists(), f"model was not saved: {result.stderr[-2000:]}"
    return path


def test_the_package_is_bundled_into_the_artifact(saved_model):
    """`code: null` in MLmodel is the shape of the bug; the package must be present."""
    mlmodel = (saved_model / "MLmodel").read_text(encoding="utf-8")
    assert "code: null" not in mlmodel
    bundled = list(saved_model.rglob("themepark/models/prophet_fleet.py"))
    assert bundled, "themepark was not carried with the model"


def test_loads_with_the_package_invisible(saved_model, tmp_path):
    """The regression: this is what a restored environment and model serving both see."""
    result = _run(
        f'''
        import warnings, logging
        warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
        import mlflow
        loaded = mlflow.pyfunc.load_model(r"{saved_model}")
        print("NAME", loaded.unwrap_python_model().wrapped.name)
        ''',
        cwd=tmp_path,  # not the repo, so a stray relative import cannot rescue it
    )
    assert "NAME prophet_fleet" in result.stdout, (
        f"champion could not load without the repo on sys.path.\n"
        f"stdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-2500:]}"
    )


def test_it_still_predicts_after_a_bare_load(saved_model, tmp_path):
    """Loading is not enough -- the unpickled fleet has to answer."""
    result = _run(
        f'''
        import warnings, logging
        warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
        import pandas as pd, mlflow
        loaded = mlflow.pyfunc.load_model(r"{saved_model}")
        out = loaded.predict(pd.DataFrame({{
            "park_name": ["Disneyland"], "ride_key": ["Space_Mountain"],
            "ts_local": [pd.Timestamp("2026-02-01 13:00:00")],
        }}))
        print("COLUMNS", ",".join(map(str, out.columns)))
        print("ROWS", len(out))
        ''',
        cwd=tmp_path,
    )
    assert "ROWS 1" in result.stdout, result.stderr[-2500:]
    assert "yhat" in result.stdout, result.stdout


# --- load_champion must not confuse "absent" with "broken" ---------------------------
#
# The portability bug was only half the failure. The other half was that load_champion
# swallowed every exception and returned None, which the caller reads as "first
# deployment". So an unloadable incumbent produced [PROMOTED] ... no champion registered
# on every run: a gate that emitted a decision it was no longer capable of making.

def fake_mlflow(exception):
    """An mlflow whose load_model raises `exception`."""
    module = mock.MagicMock()
    module.pyfunc.load_model.side_effect = exception
    return module


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_DOES_NOT_EXIST: Routine or Model 'themepark.gold.x' does not exist.",
        "Registered model alias champion not found",
        "no such registered model",
    ],
)
def test_a_missing_champion_is_a_first_deployment(message):
    with mock.patch.dict(sys.modules, {"mlflow": fake_mlflow(RuntimeError(message))}):
        assert load_champion() is None


@pytest.mark.parametrize(
    "exception",
    [
        ModuleNotFoundError("No module named 'themepark.models.prophet_fleet'"),
        RuntimeError("connection reset by peer"),
        ValueError("unpickling failed: unsupported protocol"),
    ],
)
def test_a_broken_champion_raises_rather_than_promoting(exception):
    """The regression. Any of these silently disabled the gate."""
    with mock.patch.dict(sys.modules, {"mlflow": fake_mlflow(exception)}):
        with pytest.raises(ChampionLoadError, match="could not be loaded"):
            load_champion()


def test_the_error_explains_the_consequence():
    """Whoever reads this at 3am needs to know why it refused instead of continuing."""
    broken = ModuleNotFoundError("No module named 'themepark.models.prophet_fleet'")
    with mock.patch.dict(sys.modules, {"mlflow": fake_mlflow(broken)}):
        with pytest.raises(ChampionLoadError, match="without ever comparing it"):
            load_champion()
