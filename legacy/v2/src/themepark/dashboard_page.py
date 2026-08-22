"""Assembles the dashboard page from the gold tables.

Separated from `dashboard.py` (the SVG primitives) so the chart code stays reusable and
this file reads as the page's outline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .dashboard import (
    PALETTE,
    STALE_HOURS,
    details,
    esc,
    forecast_band_chart,
    grouped_bar_chart,
    hbar_chart,
    line_chart,
    short_week,
    table,
)

log = logging.getLogger(__name__)

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light dark;
  --surface:%(l_surface)s; --plane:%(l_plane)s; --ink:%(l_ink)s; --ink2:%(l_ink2)s;
  --muted:%(l_muted)s; --grid:%(l_grid)s; --axis:%(l_axis)s; --border:%(l_border)s;
  --s1:%(l_s1)s; --s2:%(l_s2)s;
  --good:%(l_good)s; --warning:%(l_warning)s; --critical:%(l_critical)s;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --surface:%(d_surface)s; --plane:%(d_plane)s; --ink:%(d_ink)s; --ink2:%(d_ink2)s;
    --muted:%(d_muted)s; --grid:%(d_grid)s; --axis:%(d_axis)s; --border:%(d_border)s;
    --s1:%(d_s1)s; --s2:%(d_s2)s;
  }
}
:root[data-theme="dark"]{
  --surface:%(d_surface)s; --plane:%(d_plane)s; --ink:%(d_ink)s; --ink2:%(d_ink2)s;
  --muted:%(d_muted)s; --grid:%(d_grid)s; --axis:%(d_axis)s; --border:%(d_border)s;
  --s1:%(d_s1)s; --s2:%(d_s2)s;
}
body{margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 72px}
header h1{font-size:26px;margin:0 0 4px;letter-spacing:-0.01em}
header p{margin:0;color:var(--ink2);font-size:14px}
.banner{margin:20px 0 26px;padding:12px 16px;border-radius:10px;font-size:14px;
  display:flex;gap:10px;align-items:center;border:1px solid var(--border)}
.banner.ok{background:color-mix(in srgb,var(--good) 12%%,var(--surface))}
.banner.stale{background:color-mix(in srgb,var(--critical) 12%%,var(--surface))}
.banner b{font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px;margin-bottom:30px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.tile .k{font-size:12px;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted)}
.tile .v{font-size:30px;font-weight:600;margin:6px 0 2px;letter-spacing:-0.02em}
.tile .s{font-size:13px;color:var(--ink2)}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:20px 22px;margin-bottom:22px}
section h2{font-size:17px;margin:0 0 4px;letter-spacing:-0.01em}
section .sub{margin:0 0 16px;color:var(--ink2);font-size:13.5px;max-width:74ch}
.chart{width:100%%;height:auto;overflow:visible;display:block}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}
.lbl{fill:var(--ink2);font-size:12.5px}
.val{fill:var(--ink);font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums}
.bar path,.bar rect{fill:var(--s1);opacity:.55}
.bar-hi path,.bar-hi rect{fill:var(--s1)}
.s1 path,.s1 circle{fill:var(--s1)} .s2 path{fill:var(--s2)}
.s1-stroke{stroke:var(--s1)} .s1-fill{fill:var(--s1);opacity:.16}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot circle{stroke:var(--surface);stroke-width:2}
.band{stroke:none}
.legend{display:flex;gap:18px;margin-bottom:10px;font-size:13px;color:var(--ink2);flex-wrap:wrap}
.key{display:flex;gap:7px;align-items:center}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.sw.s1{background:var(--s1)} .sw.s2{background:var(--s2)}
.sw.band-sw{background:color-mix(in srgb,var(--s1) 22%%,transparent);
  border:1px solid color-mix(in srgb,var(--s1) 50%%,transparent)}
details{margin-top:14px;font-size:13px}
summary{cursor:pointer;color:var(--ink2);padding:4px 0}
.tablewrap,details>table{overflow-x:auto;display:block}
table{border-collapse:collapse;width:100%%;margin-top:10px;font-size:12.5px;
  font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 12px 6px 0;border-bottom:1px solid var(--border);
  white-space:nowrap}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:0.04em}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600}
.pill.yes{background:color-mix(in srgb,var(--good) 18%%,var(--surface));color:var(--ink)}
.pill.no{background:color-mix(in srgb,var(--warning) 26%%,var(--surface));color:var(--ink)}
.empty{color:var(--muted);font-size:13.5px;font-style:italic;margin:10px 0}
code{background:var(--plane);padding:1.5px 5px;border-radius:4px;font-size:12.5px}
footer{color:var(--muted);font-size:12.5px;margin-top:30px;text-align:center}
@media (max-width:640px){.tile .v{font-size:25px}.wrap{padding:20px 14px 48px}}
"""


def _tile(key: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="k">{esc(key)}</div>'
        f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'
    )


def _freshness(runs: pd.DataFrame) -> str:
    """Catches the silent failure: a pipeline that stopped but left stale data in place."""
    if runs is None or runs.empty:
        return '<div class="banner stale">&#9888; <b>No pipeline runs recorded yet.</b></div>'

    latest = runs.copy()
    latest["run_at"] = pd.to_datetime(latest["run_at"], utc=True, errors="coerce")
    newest = latest["run_at"].max()
    age = datetime.now(timezone.utc) - newest.to_pydatetime()

    per_job = (
        latest.sort_values("run_at").groupby("job").last()[["run_at", "status"]].reset_index()
    )
    detail = " &middot; ".join(
        f"{esc(r.job)} {esc(str(r.run_at)[:16])} ({esc(r.status)})" for r in per_job.itertuples()
    )

    if age > timedelta(hours=STALE_HOURS):
        return (
            f'<div class="banner stale">&#9888; <b>Pipeline is stale</b> &mdash; last run '
            f"{age.total_seconds() / 3600:.0f}h ago. {detail}</div>"
        )
    return (
        f'<div class="banner ok">&#10003; <b>Pipeline healthy</b> &mdash; last run '
        f"{age.total_seconds() / 3600:.1f}h ago. {detail}</div>"
    )


def render(data: dict[str, pd.DataFrame], generated_at: datetime | None = None) -> str:
    """Build the full page from the published gold tables."""
    now = generated_at or datetime.now(timezone.utc)
    accuracy = data.get("prediction_accuracy", pd.DataFrame())
    metrics = data.get("model_metrics", pd.DataFrame())
    by_ride = data.get("accuracy_by_ride", pd.DataFrame())
    drift = data.get("data_drift", pd.DataFrame())
    promotions = data.get("promotion_log", pd.DataFrame())
    predictions = data.get("predictions", pd.DataFrame())
    runs = data.get("pipeline_runs", pd.DataFrame())

    overall = accuracy[accuracy.segment == "overall"] if not accuracy.empty else pd.DataFrame()
    latest = overall.sort_values("forecast_week").iloc[-1] if not overall.empty else None

    champion = "unknown"
    if promotions is not None and not promotions.empty and "winner" in promotions:
        champion = str(promotions.sort_values("decided_at").iloc[-1]["winner"])
    elif latest is not None:
        champion = str(latest.get("model_name", "unknown"))

    n_rides = int(predictions["ride_key"].nunique()) if not predictions.empty else 0

    tiles = "".join(
        [
            _tile(
                "Production MAE",
                f"{latest['mae']:.2f} min" if latest is not None else "--",
                "measured against actuals, not a backtest",
            ),
            _tile(
                "Within 10 minutes",
                f"{latest['within_10min_pct']:.0f}%" if latest is not None else "--",
                "share of slots predicted within 10 min",
            ),
            _tile(
                "Interval coverage",
                f"{latest['interval_coverage_pct']:.0f}%" if latest is not None else "--",
                "actuals inside the 80% band",
            ),
            _tile("Champion model", champion, "selected by the weekly gate"),
            _tile("Attractions served", str(n_rides) if n_rides else "--", "forecast every 30 min"),
        ]
    )

    sections: list[str] = []

    # --- Production accuracy over time -------------------------------------------------
    if not overall.empty:
        weeks = overall.sort_values("forecast_week")
        sections.append(
            f"""<section><h2>Production accuracy over time</h2>
<p class="sub">Error of forecasts scored against observations that did not exist when the
forecast was published. This is the only measurement here with no possibility of leakage,
and it is the number worth trusting &mdash; a backtest can always be drawn favourably.</p>
{line_chart([short_week(w) for w in weeks.forecast_week], list(weeks.mae.astype(float)))}
{details("Table view", table(weeks[["forecast_week", "n", "mae", "rmse", "within_10min_pct", "bias", "interval_coverage_pct"]]))}
</section>"""
        )

    # --- Backtest vs production --------------------------------------------------------
    if not accuracy.empty and not metrics.empty:
        latest_week = accuracy.sort_values("forecast_week").forecast_week.iloc[-1]
        prod = accuracy[accuracy.forecast_week == latest_week].set_index("segment")["mae"]
        champ_metrics = metrics[metrics.model == champion]
        if not champ_metrics.empty:
            row = champ_metrics.sort_values("overall_mae").iloc[0]
            segs, back, live = [], [], []
            for segment, column in [
                ("overall", "overall_mae"),
                ("peak_hours", "peak_hours_mae"),
                ("weekend_holiday", "weekend_holiday_mae"),
            ]:
                if segment in prod.index and column in row:
                    segs.append(segment.replace("_", " "))
                    back.append(float(row[column]))
                    live.append(float(prod[segment]))
            if segs:
                sections.append(
                    f"""<section><h2>Backtest versus production</h2>
<p class="sub">The same champion, scored two ways. A large gap between these bars means the
offline number is not predictive of live behaviour &mdash; the previous version of this
project reported a 3.27&nbsp;min holdout MAE while its one live sample came in near 12.
Publishing both is the point.</p>
{grouped_bar_chart(segs, {"backtest holdout": back, "production (live)": live})}
{details("Table view", table(pd.DataFrame({"segment": segs, "backtest_mae": back, "production_mae": live})))}
</section>"""
                )

    # --- Model leaderboard -------------------------------------------------------------
    if not metrics.empty:
        board = metrics.sort_values("overall_mae")
        names = list(board["model"].astype(str))
        highlight = names.index(champion) if champion in names else None
        sections.append(
            f"""<section><h2>Candidate leaderboard &mdash; this week's backtest</h2>
<p class="sub">Every family is retrained and re-scored weekly on the same trailing holdout,
and so is the incumbent champion. Re-scoring the incumbent rather than trusting its stored
metrics is what makes the comparison like-for-like. Lower is better; the solid bar is the
current champion.</p>
{hbar_chart(names, list(board["overall_mae"].astype(float)), highlight=highlight)}
{details("Full scorecard", table(board))}
</section>"""
        )

    # --- Promotion history -------------------------------------------------------------
    if not promotions.empty:
        promo = promotions.sort_values("decided_at", ascending=False).head(12).copy()
        rows = "".join(
            "<tr>"
            f"<td>{esc(str(r.decided_at)[:16])}</td>"
            f'<td><span class="pill {"yes" if str(r.promoted).lower() in ("true", "1") else "no"}">'
            f'{"promoted" if str(r.promoted).lower() in ("true", "1") else "declined"}</span></td>'
            f"<td>{esc(r.winner)}</td><td>{esc(getattr(r, 'reason', ''))}</td></tr>"
            for r in promo.itertuples()
        )
        sections.append(
            f"""<section><h2>Promotion decisions</h2>
<p class="sub">A challenger replaces the champion only if it clears the coverage floor,
beats the historical-mean baseline, improves overall MAE by at least 2%, and does not
regress RMSE on high-wait rides. Declined promotions are recorded here too &mdash; a gate
that has never said no is not a gate.</p>
<div class="tablewrap"><table><thead><tr><th>decided</th><th>outcome</th><th>champion</th>
<th>reason</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>"""
        )

    # --- Worst-predicted attractions ---------------------------------------------------
    if not by_ride.empty:
        worst = by_ride.sort_values("mae", ascending=False).head(12)
        sections.append(
            f"""<section><h2>Hardest attractions to predict</h2>
<p class="sub">Ranked by production MAE. A large positive bias with a near-zero mean actual
is the signature of an unplanned closure &mdash; the model has no ride-status input, which
is currently the largest known gap.</p>
{hbar_chart(list(worst.ride_name.astype(str)), list(worst.mae.astype(float)))}
{details("Table view", table(worst))}
</section>"""
        )

    # --- Sample forecast ---------------------------------------------------------------
    if not predictions.empty:
        busiest = (
            predictions.groupby(["ride_name"])["predicted_wait_min"].mean().sort_values().index[-1]
        )
        sample = predictions[predictions.ride_name == busiest].sort_values("ts_local").head(96)
        if len(sample) > 2:
            sections.append(
                f"""<section><h2>Published forecast &mdash; {esc(busiest)}</h2>
<p class="sub">The next few days at 30-minute resolution, with the 80% interval the routing
optimiser plans against. A route built on the point estimate alone cannot distinguish a
reliable 20-minute queue from a volatile one averaging the same.</p>
{forecast_band_chart(list(sample.ts_local.astype(str)), list(sample.predicted_wait_min.astype(float)), list(sample.lower_bound.astype(float)), list(sample.upper_bound.astype(float)))}
{details("Table view", table(sample[["ts_local", "predicted_wait_min", "lower_bound", "upper_bound"]]))}
</section>"""
            )

    # --- Drift -------------------------------------------------------------------------
    if not drift.empty:
        d = drift.sort_values("week")
        sections.append(
            f"""<section><h2>Ingestion health</h2>
<p class="sub">Volume and the share of zero-wait observations, weekly. A collector that
half-fails still posts a respectable MAE on the rows it does return &mdash; these two move
first. Two measures, two charts: never two y-axes on one plot.</p>
<h3 style="font-size:13.5px;color:var(--ink2);margin:14px 0 6px">Rows ingested per week</h3>
{line_chart([short_week(w) for w in d.week], list(d.rows.astype(float)), unit="rows", digits=0)}
<h3 style="font-size:13.5px;color:var(--ink2);margin:20px 0 6px">Share of zero-wait observations (%)</h3>
{line_chart([short_week(w) for w in d.week], list(d.pct_zero.astype(float)), unit="%", digits=1)}
{details("Table view", table(d))}
</section>"""
        )

    if not sections:
        sections.append(
            '<section><p class="empty">No published results yet. The dashboard fills in '
            "after the first scheduled run completes.</p></section>"
        )

    tokens = {f"l_{k}": v for k, v in PALETTE["light"].items()}
    tokens |= {f"d_{k}": v for k, v in PALETTE["dark"].items()}

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theme Park Wait Time Forecasting &mdash; Production Monitor</title>
<meta name="description" content="Live accuracy, model leaderboard and promotion history for an automated weekly wait-time forecasting pipeline.">
<style>{CSS % tokens}</style>
</head><body>
<div class="wrap">
<header>
  <h1>Theme park wait-time forecasting</h1>
  <p>Automated weekly retraining, champion/challenger promotion, and out-of-sample accuracy
     monitoring. Regenerated by the pipeline &mdash; no manual step.</p>
</header>
{_freshness(runs)}
<div class="tiles">{tiles}</div>
{"".join(sections)}
<footer>Generated {now:%Y-%m-%d %H:%M} UTC &middot; every chart above has a table view &middot;
forecasts served from Postgres via REST</footer>
</div></body></html>"""
