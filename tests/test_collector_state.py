"""Tests for the collector's local watermark cache.

The cache exists to protect the Databricks free-tier compute quota: querying the SQL
warehouse for `max(wait_time_id)` on every 30-minute run means 48 wake-ups a day of a
warehouse that scales to zero, which costs far more than the weekly training it supports.

Correctness requirement: the cache may cost an extra warehouse query, but must never
cause a row to be skipped.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))

from collect import RESYNC_AFTER, read_watermark, write_watermark  # noqa: E402


class FakeLake:
    """Stands in for Databricks, counting how often the warehouse is queried."""

    def __init__(self, watermark=5000):
        self._watermark = watermark
        self.queries = 0

    def watermark(self):
        self.queries += 1
        return self._watermark


def test_cold_start_queries_databricks(tmp_path):
    lake = FakeLake(4242)
    value, origin = read_watermark(lake, tmp_path / "state.json")
    assert (value, origin, lake.queries) == (4242, "Databricks", 1)


def test_cold_start_persists_state_for_next_run(tmp_path):
    state = tmp_path / "state.json"
    lake = FakeLake(4242)
    read_watermark(lake, state)

    value, origin = read_watermark(lake, state)
    assert (value, origin) == (4242, "local state")
    assert lake.queries == 1, "the second run must not touch the warehouse"


def test_fresh_state_avoids_the_warehouse_entirely(tmp_path):
    state = tmp_path / "state.json"
    write_watermark(state, 9000)
    lake = FakeLake(1)

    for _ in range(48):  # a full day of 30-minute runs
        value, origin = read_watermark(lake, state)

    assert (value, origin) == (9000, "local state")
    assert lake.queries == 0


def test_stale_state_triggers_a_resync(tmp_path):
    state = tmp_path / "state.json"
    stale = datetime.now(timezone.utc) - RESYNC_AFTER - timedelta(minutes=1)
    state.write_text(
        json.dumps({"watermark": 100, "updated_at": stale.isoformat(), "last_resync": stale.isoformat()})
    )

    lake = FakeLake(7777)
    value, origin = read_watermark(lake, state)
    assert (value, origin, lake.queries) == (7777, "Databricks", 1)


def test_resync_flag_ignores_local_state(tmp_path):
    state = tmp_path / "state.json"
    write_watermark(state, 100)

    lake = FakeLake(8888)
    value, origin = read_watermark(lake, state, force_resync=True)
    assert (value, origin, lake.queries) == (8888, "Databricks", 1)


def test_corrupt_state_falls_back_rather_than_crashing(tmp_path):
    """A truncated write must cost a warehouse query, not a failed ingestion run."""
    state = tmp_path / "state.json"
    state.write_text("{ this is not json")

    lake = FakeLake(1234)
    value, origin = read_watermark(lake, state)
    assert (value, origin, lake.queries) == (1234, "Databricks", 1)


def test_batch_updates_do_not_extend_the_resync_deadline(tmp_path):
    """Per-batch writes must not defer reconciliation indefinitely.

    Otherwise a continuously busy collector would never re-check Databricks, and local
    state could drift from reality without anything noticing.
    """
    state = tmp_path / "state.json"
    stale = datetime.now(timezone.utc) - RESYNC_AFTER - timedelta(minutes=1)
    state.write_text(
        json.dumps({"watermark": 100, "updated_at": stale.isoformat(), "last_resync": stale.isoformat()})
    )

    write_watermark(state, 200, resynced=False)
    assert json.loads(state.read_text())["last_resync"] == stale.isoformat()

    lake = FakeLake(9999)
    _, origin = read_watermark(lake, state)
    assert origin == "Databricks"
