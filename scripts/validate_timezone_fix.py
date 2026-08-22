"""Side-by-side proof that the timezone fix is real, run against the raw source extract.

v1 applied park-local operating hours to UTC timestamps. This script reproduces the v1
filter verbatim, runs the corrected one, and prints both. The giveaway is the hourly
profile: a real theme park ramps up in the morning, plateaus midday to early evening and
decays to close. v1's profile is that curve shifted eight hours into a UTC frame, with
the evening peak amputated.

    python scripts/validate_timezone_fix.py [path/to/extract.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from themepark.filters import apply_operating_filter  # noqa: E402

DEFAULT_CSV = "data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv"

# Verbatim from v1 data_utils.PARK_CONSTRAINTS.
V1_CONSTRAINTS = {
    "Disneyland": (8, 24, "2025-12-06"),
    "Disney California Adventure Park": (8, 22, "2025-12-06"),
    "Universal Studios Hollywood": (8, 22, "2025-12-06"),
    "SeaWorld San Diego": (10, 20, "2025-12-06"),
    "SeaWorld San Diego Obsolete": (10, 20, "2025-12-06"),
    "Six Flags Magic Mountain": (10, 21, "2026-02-15"),
}


def v1_filter(df: pd.DataFrame) -> pd.DataFrame:
    """The original logic: local-hour windows tested against UTC hours."""
    out = df.copy()
    out["hour"] = out["ts_utc"].dt.hour
    out = out[(out.wait_time >= 0) & (out.wait_time < 900)]
    parts = []
    for park, (open_h, close_h, start) in V1_CONSTRAINTS.items():
        d = out[out.park_name == park]
        parts.append(d[(d.ts_utc >= start) & (d.hour >= open_h) & (d.hour < close_h)])
    return pd.concat(parts, ignore_index=True)


def main(csv_path: str = DEFAULT_CSV) -> int:
    df = pd.read_csv(csv_path, usecols=["wait_time_upd_dt", "tp_name", "at_name", "wait_time"])
    df["wait_time_upd_dt"] = pd.to_datetime(df["wait_time_upd_dt"])
    df = df.rename(
        columns={"wait_time_upd_dt": "ts_utc", "tp_name": "park_name", "at_name": "ride_name"}
    )

    v1, v2 = v1_filter(df), apply_operating_filter(df)
    total_mass = df[(df.wait_time >= 0) & (df.wait_time < 900)].wait_time.sum()

    print(f"source rows: {len(df):,}\n")
    print(f"{'':<18}{'v1 (UTC bug)':>14}{'v2 (local)':>14}")
    print(f"{'rows kept':<18}{len(v1):>14,}{len(v2):>14,}")
    print(f"{'mean wait (min)':<18}{v1.wait_time.mean():>14.2f}{v2.wait_time.mean():>14.2f}")
    print(
        f"{'zeros':<18}{(v1.wait_time == 0).mean() * 100:>13.1f}%"
        f"{(v2.wait_time == 0).mean() * 100:>13.1f}%"
    )
    print(
        f"{'wait-minute mass':<18}{v1.wait_time.sum() / total_mass * 100:>13.1f}%"
        f"{v2.wait_time.sum() / total_mass * 100:>13.1f}%"
    )

    dl1 = v1[v1.park_name == "Disneyland"]
    dl2 = v2[v2.park_name == "Disneyland"]
    p1 = dl1.groupby(dl1.ts_utc.dt.hour).wait_time.mean()
    p2 = dl2.groupby(dl2.ts_local.dt.hour).wait_time.mean()

    print(f"\nDisneyland peak: v1 at UTC hour {p1.idxmax():02d}, v2 at LOCAL hour {p2.idxmax():02d}")
    print("\nlocal-hour profile after the fix:")
    for hour, value in p2.items():
        print(f"  {hour:02d}:00  {value:5.1f}  {'#' * int(value)}")

    # The fix must recover the overwhelming majority of the queue signal.
    recovered = v2.wait_time.sum() / total_mass
    if recovered < 0.90:
        print(f"\nFAIL: only {recovered:.1%} of wait-minute mass retained")
        return 1
    print(f"\nPASS: {recovered:.1%} of wait-minute mass retained (v1 kept {v1.wait_time.sum() / total_mass:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
