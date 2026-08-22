"""Score historical weeks with the champion so verification has actuals to grade.

On a freshly deployed pipeline `gold.prediction_accuracy` is empty, and it stays empty
until a published forecast has had a week to come true. The dashboard's most valuable
panel -- measured accuracy, rather than the flattering offline number -- therefore shows
nothing on day one, which is exactly when someone is most likely to look at it.

This scores weeks that sit *inside* the silver date range, so actuals already exist and
verify_job can grade them immediately.

What this is, precisely, so nobody oversells it: the champion was fitted on data ending
at the backtest split and has never seen these weeks, so the errors are genuinely
out-of-sample. It is not the same thing as a forecast published before the data existed.
The distinction is visible in the data itself -- a backfilled row has `generated_at`
*after* `ts_local`, a live one has it before -- and `--label` records it in model_version.

    python jobs/backfill_accuracy.py                 # the trailing holdout weeks
    python jobs/backfill_accuracy.py --weeks 4
    python jobs/backfill_accuracy.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Databricks serverless runs a spark_python_task as exec(compile(source, path, "exec")),
# which leaves __file__ undefined while still recording the real path on the code object.
# Both lookups are needed: __file__ on a laptop or a runner, the frame on Databricks.
# Repeated verbatim in each entry point -- see jobs/_databricks.py for why it cannot be
# imported from there.
HERE = Path(globals().get("__file__", sys._getframe().f_code.co_filename)).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from _databricks import log, read_delta, spark_session, write_delta  # noqa: E402
from themepark.config import DatabricksSettings, pipeline  # noqa: E402
from themepark.score import generate_forecast  # noqa: E402
from themepark.train import load_champion  # noqa: E402


def week_starts(observed_through: pd.Timestamp, weeks: int, horizon_days: int) -> list[pd.Timestamp]:
    """Week boundaries fully covered by observed data, most recent last.

    Anchored on the last day with actuals rather than on today, because a slot with no
    actual cannot be graded and would only inflate the published row count.
    """
    end = observed_through.normalize()
    return [end - pd.Timedelta(days=horizon_days * (n + 1)) for n in reversed(range(weeks))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=2, help="how many trailing weeks to score")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    parser.add_argument(
        "--label",
        default="backfill",
        help="recorded in model_version so these rows stay distinguishable from live ones",
    )
    parser.add_argument(
        "--after",
        help=(
            "drop slots at or before this local timestamp -- set it to the champion's "
            "training cutoff so no graded slot was ever trained on"
        ),
    )
    args = parser.parse_args()

    cfg = DatabricksSettings()
    settings = pipeline()
    spark = spark_session()

    silver = read_delta(spark, cfg.silver_table)
    silver["ts_local"] = pd.to_datetime(silver["ts_local"])
    observed_through = silver["ts_local"].max()
    log.info("silver holds %d rows, observed through %s", len(silver), observed_through)

    champion = load_champion()
    if champion is None:
        log.error("no @champion registered -- run the train task before backfilling")
        return 1
    log.info("scoring with champion %s", champion.name)

    cutoff = pd.Timestamp(args.after) if args.after else None
    if cutoff is not None:
        log.info("excluding slots at or before the training cutoff %s", cutoff)

    starts = week_starts(observed_through, args.weeks, settings.horizon_days)
    log.info(
        "will score %d week(s): %s",
        len(starts),
        ", ".join(s.strftime("%Y-%m-%d") for s in starts),
    )
    if args.dry_run:
        log.info("dry run: nothing written")
        return 0

    frames = []
    for start in starts:
        forecast = generate_forecast(
            champion, silver, model_version=args.label, start=start
        )
        # Drop anything past the last actual: an ungradeable row is noise in the table and
        # a silently missing row in every accuracy average computed from it.
        forecast = forecast[forecast["ts_local"] <= observed_through]
        if cutoff is not None:
            # Anything at or before the training cutoff was fitted on, and grading it would
            # quietly flatter the published accuracy with in-sample slots.
            forecast = forecast[forecast["ts_local"] > cutoff]
        log.info("  %s -> %d gradeable slots", start.strftime("%Y-%m-%d"), len(forecast))
        frames.append(forecast)

    published = pd.concat(frames, ignore_index=True)
    if published.empty:
        log.warning("no gradeable slots produced; nothing written")
        return 0

    write_delta(spark, published, cfg.predictions_table, mode="append")
    log.info(
        "backfilled %d prediction rows across %d week(s); run the verify task to grade them",
        len(published),
        len(starts),
    )
    return 0


if __name__ == "__main__":
    _status = main()
    if _status:
        raise SystemExit(_status)
