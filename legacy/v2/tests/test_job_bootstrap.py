"""The Databricks entry points must resolve their own location without `__file__`.

A serverless `spark_python_task` is executed as `exec(compile(source, path, "exec"))`.
That namespace has no `__file__`, so `Path(__file__).resolve().parent` raises NameError
and the task dies during import -- which is exactly how the first deployed run failed,
before a single row was read. The real path does survive, as the code object's filename.

These tests run each entry point's prologue under those conditions. They are cheap and
local, and they are the check that stops a Databricks-only import error from being
discovered by spending free-tier compute on a failed run.
"""

import sys
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
DATABRICKS_ENTRY_POINTS = ["bronze_load.py", "silver_build.py", "train_job.py", "verify_job.py"]

# The one line that has to be identical everywhere. Pinned so a well-meaning tidy-up of
# one file cannot quietly reintroduce the fragile form in the other three.
BOOTSTRAP = (
    'HERE = Path(globals().get("__file__", sys._getframe().f_code.co_filename)).resolve().parent'
)

# Where the bundle actually syncs the repo, from the traceback of the failed run.
WORKSPACE = "/Workspace/Users/tirthpatel3223@gmail.com/.bundle/themepark/files/jobs/{}"


def prologue(name):
    """Source up to and including the bootstrap -- imports only, no pyspark, no Spark."""
    source = (JOBS / name).read_text(encoding="utf-8")
    marker = "sys.path[:0] ="
    assert marker in source, f"{name} has no sys.path bootstrap"
    return source[: source.index("\n", source.index(marker)) + 1]


@pytest.fixture(autouse=True)
def _restore_sys_path():
    """The prologue mutates the real sys.path; put it back."""
    saved = list(sys.path)
    yield
    sys.path[:] = saved


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_resolves_its_location_without_dunder_file(name):
    """The regression: no __file__ in globals, exactly as Databricks executes it."""
    namespace = {"__name__": "__main__"}
    exec(compile(prologue(name), WORKSPACE.format(name), "exec"), namespace)

    assert namespace["HERE"].name == "jobs"
    assert namespace["HERE"].parent.name == "files", "must resolve inside the synced bundle"


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_puts_both_jobs_and_src_on_the_path(name):
    """`_databricks` lives in jobs/, `themepark` in src/; both imports follow immediately."""
    exec(compile(prologue(name), WORKSPACE.format(name), "exec"), {"__name__": "__main__"})

    assert Path(sys.path[0]).name == "jobs"
    assert Path(sys.path[1]).name == "src"


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_still_works_when_dunder_file_is_defined(name):
    """A laptop and a GitHub runner must keep taking the __file__ branch."""
    real = JOBS / name
    namespace = {"__name__": "__main__", "__file__": str(real)}
    exec(compile(prologue(name), str(real), "exec"), namespace)

    assert namespace["HERE"] == JOBS


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_no_entry_point_reintroduces_the_fragile_form(name):
    source = (JOBS / name).read_text(encoding="utf-8")
    assert BOOTSTRAP in source, f"{name} drifted from the pinned bootstrap"
    assert "Path(__file__)" not in source, (
        f"{name} uses Path(__file__) directly, which raises NameError on Databricks serverless"
    )


# --- exit handling ------------------------------------------------------------------
#
# A Databricks task runs the entry point inside an IPython kernel, which reports any
# SystemExit as an error regardless of its code. `raise SystemExit(main())` therefore
# marked bronze_load FAILED after it had successfully written all 842,539 rows, and
# max_retries then ran the completed job twice more against the free-tier quota.

def epilogue(name):
    """The `if __name__ == "__main__":` block, with main() stubbed out."""
    source = (JOBS / name).read_text(encoding="utf-8")
    marker = 'if __name__ == "__main__":'
    assert marker in source, f"{name} has no main guard"
    return source[source.index(marker) :]


def run_epilogue(name, returns):
    """Execute the guard as __main__, with main() returning `returns`."""
    namespace = {"__name__": "__main__", "main": lambda: returns}
    exec(compile(epilogue(name), str(JOBS / name), "exec"), namespace)


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_success_does_not_raise_systemexit(name):
    """The regression: a green run must not look like a failure to Databricks."""
    run_epilogue(name, 0)  # must simply return


@pytest.mark.parametrize("name", DATABRICKS_ENTRY_POINTS)
def test_failure_still_raises(name):
    """Softening the success path must not swallow a real failure."""
    with pytest.raises(SystemExit) as err:
        run_epilogue(name, 1)
    assert err.value.code == 1
