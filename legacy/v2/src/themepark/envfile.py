"""Load `.env` into the process environment for local runs.

`config.py` tells you that "local runs read it from .env", and until this module existed
nothing made that true: a fully populated `.env` sat in the repo root and every entry
point still died on the first required variable. Only `check_upstream.py` could read it,
via a private copy of the parser -- the same duplicated-implementation shape that caused
the ride-key defect. So there is one parser here and everything calls it.

No new dependency. python-dotenv would mean an extra runtime install on the collector VM
and inside every Databricks task, to parse twelve lines of `KEY=value`.

Two deliberate rules:

* **A real environment variable always wins.** GitHub Actions, systemd and Databricks all
  inject configuration properly. A stale `.env` left in a working copy must never quietly
  override what the platform supplied -- that failure looks like the platform lying to you.
* **Inline comments are not stripped.** `PG_PASSWORD=hunter2 # prod` keeps the whole
  string rather than silently truncating a password at the `#`. A credential that is
  wrong and fails loudly beats one that is subtly shortened.
"""

from __future__ import annotations

import os
from pathlib import Path

FILENAME = ".env"


def find_dotenv(start: Path | None = None) -> Path | None:
    """The nearest `.env` at or above `start`, else at or above this package.

    Searching upward is what lets `python jobs/collect.py` and `python collect.py` and a
    systemd unit with its own WorkingDirectory all resolve the same file.
    """
    origins = [(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for origin in origins:
        for directory in (origin, *origin.parents):
            candidate = directory / FILENAME
            if candidate.is_file():
                return candidate
    return None


def parse(text: str) -> dict[str, str]:
    """`KEY=value` lines to a dict. Tolerates `export`, quotes, comments and blank lines."""
    values: dict[str, str] = {}
    for raw in text.lstrip("\ufeff").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key.isidentifier():
            continue  # not a variable assignment; ignore rather than guess
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load(path: Path | None = None, override: bool = False) -> list[str]:
    """Populate `os.environ` from a `.env` file. Returns the names actually set.

    A missing file is not an error: on Databricks and on a runner there is no `.env`, and
    the environment is already populated. Returning the applied names rather than nothing
    lets a caller report where its configuration came from.
    """
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return []

    applied = []
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied
