"""Render the static dashboard (site/index.html) from the serving gold tables.

Self-contained HTML: inline CSS, inline SVG, zero JavaScript, no external assets, so it
survives a strict Content-Security-Policy on GitHub Pages. Hover values are native SVG
<title> tooltips. Numeric tiles come from themepark.gold.kpis_last; plots come from the
backtest rows of themepark.gold.predictions_last (predicted vs actual vs difference).
The run provenance strip makes a stale or fallback dashboard visibly so.

Usage:
    python dashboard.py --out site/index.html                 fetch from warehouse
    python dashboard.py --from-parquet DIR --out site/index.html
"""

from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import publish

ROOT = Path(__file__).resolve().parent

STATUS_LABELS = {
    "fresh_model": ("Fresh model", "good"),
    "kept_previous_model": ("Kept previous model", "neutral"),
    "fallback_after_failure": ("Fallback after a failed run", "serious"),
}

TILE_SPECS = [
    ("mae", "MAE", "min", "Mean absolute error on the held-out test window"),
    ("rmse", "RMSE", "min", "Root mean squared error"),
    ("pct_within_10min", "Within 10 min", "%", "Share of predictions within 10 minutes of the actual wait"),
    ("pct_severe_miss", "Severe misses", "%", "Share of predictions off by more than 15 minutes"),
    ("bias", "Bias", "min", "Mean signed error; positive means over-forecasting"),
    ("high_wait_mae", "High-wait MAE", "min", "MAE on rides averaging over 10 minutes"),
    ("peak_hours_mae", "Peak-hours MAE", "min", "MAE during 11:00-20:00 local"),
]

CSS = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --actual: #2a78d6; --predicted: #eb6834;
  --under: #2a78d6; --over: #e34948; --zero: #c3c2b7;
  --seq: #2a78d6;
  --good: #0ca30c; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --actual: #3987e5; --predicted: #d95926;
    --under: #3987e5; --over: #e66767; --zero: #52514e;
    --seq: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 24px 20px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.prov {
  display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 20px; font-size: 13px;
}
.prov b { font-weight: 600; }
.prov .k { color: var(--muted); margin-right: 6px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px; font-weight: 600;
  padding: 2px 10px; border-radius: 999px; border: 1px solid var(--border);
}
.badge .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ink-2); }
.badge.good .dot { background: var(--good); }
.badge.serious .dot { background: var(--serious); }
.badge.serious { border-color: var(--serious); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .v { font-size: 26px; font-weight: 650; letter-spacing: -0.01em; }
.tile .u { font-size: 13px; color: var(--muted); font-weight: 400; }
.tile .l { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; }
.card .why { color: var(--muted); font-size: 12px; margin: 0 0 10px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .grid2, .grid3 { grid-template-columns: 1fr; } }
.panel h3 { font-size: 13px; margin: 0; }
.panel .stat { color: var(--muted); font-size: 12px; margin: 0 0 6px; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2); margin-bottom: 6px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
svg { display: block; width: 100%; height: auto; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); }
svg .endlab { fill: var(--ink-2); font-weight: 600; }
svg .val { fill: var(--ink-2); font-variant-numeric: tabular-nums; }
svg .ridelab { fill: var(--ink); font-size: 11.5px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
td:first-child { color: var(--ink); }
tr.champ td { font-weight: 650; }
details { margin-top: 10px; }
summary { color: var(--muted); font-size: 12px; cursor: pointer; }
details table { font-size: 12px; margin-top: 8px; }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


# ------------------------------------------------------------------------------------
# SVG helpers
# ------------------------------------------------------------------------------------


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = np.ceil(lo / step) * step
    return [float(t) for t in np.arange(start, hi + step * 0.01, step)]


def line_chart(x_labels, series, width=920, height=260, y_unit="min"):
    """Two-series line chart. series = [(name, css_var, values), ...]."""
    ml, mr, mt, mb = 44, 90, 12, 26
    pw, ph = width - ml - mr, height - mt - mb
    all_vals = [v for _, _, vals in series for v in vals if v is not None]
    y_hi = max(all_vals) * 1.08
    ticks = nice_ticks(0, y_hi)
    y_top = max(y_hi, ticks[-1])

    def sx(i):
        return ml + (pw * i / max(len(x_labels) - 1, 1))

    def sy(v):
        return mt + ph * (1 - v / y_top)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for t in ticks:
        y = sy(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 3.5:.1f}" text-anchor="end">{t:g}</text>')
    parts.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="var(--axis)" stroke-width="1"/>')
    n_x_ticks = min(len(x_labels), 8)
    for j in range(n_x_ticks):
        i = round(j * (len(x_labels) - 1) / max(n_x_ticks - 1, 1))
        parts.append(f'<text x="{sx(i):.1f}" y="{mt + ph + 16}" text-anchor="middle">{esc(x_labels[i])}</text>')
    for name, var, vals in series:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{ml + pw + 8}" y="{sy(vals[-1]) + 3.5:.1f}" class="endlab">{esc(name)}</text>'
        )
    # Hover layer: transparent hit circles with native tooltips.
    for i, label in enumerate(x_labels):
        tip = ", ".join(f"{name} {fmt(vals[i])} {y_unit}" for name, _, vals in series)
        parts.append(
            f'<circle cx="{sx(i):.1f}" cy="{mt + ph / 2:.1f}" r="{max(pw / len(x_labels) / 2, 6):.1f}" '
            f'fill="transparent"><title>{esc(label)}: {esc(tip)}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


def rounded_top_bar(x, y, w, h, r=3):
    r = min(r, w / 2, h) if h > 0 else 0
    return (
        f"M {x:.1f} {y + h:.1f} L {x:.1f} {y + r:.1f} Q {x:.1f} {y:.1f} {x + r:.1f} {y:.1f} "
        f"L {x + w - r:.1f} {y:.1f} Q {x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f} "
        f"L {x + w:.1f} {y + h:.1f} Z"
    )


def histogram(edges, counts, width=450, height=240, y_top=None):
    """Residual histogram; bars colored by polarity (under / near zero / over).
    Pass y_top to share one y-scale across small multiples."""
    ml, mr, mt, mb = 44, 10, 12, 30
    pw, ph = width - ml - mr, height - mt - mb
    total = sum(counts)
    y_top = (y_top if y_top else max(counts)) * 1.08
    ticks = nice_ticks(0, y_top)
    y_top = max(y_top, ticks[-1])
    n = len(counts)
    bw = pw / n

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for t in ticks:
        y = mt + ph * (1 - t / y_top)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        lab = f"{t / 1000:g}k" if y_top >= 2000 else f"{t:g}"
        parts.append(f'<text x="{ml - 8}" y="{y + 3.5:.1f}" text-anchor="end">{lab}</text>')
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        mid = (lo + hi) / 2
        var = "--under" if mid < -2.5 else ("--over" if mid > 2.5 else "--zero")
        x = ml + i * bw + 1  # 2px gap between fills
        h = ph * c / y_top
        y = mt + ph - h
        share = 100 * c / total if total else 0
        lo_lab = "under" if lo == -np.inf else f"{lo:g}"
        hi_lab = "over" if hi == np.inf else f"{hi:g}"
        parts.append(
            f'<path d="{rounded_top_bar(x, y, bw - 2, h)}" fill="var({var})">'
            f"<title>error {lo_lab} to {hi_lab} min: {c:,} rows ({share:.1f}%)</title></path>"
        )
    parts.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="var(--axis)" stroke-width="1"/>')
    for value in (-30, -15, 0, 15, 30):
        frac = (value - edges[1]) / (edges[-2] - edges[1])
        x = ml + bw + frac * (pw - 2 * bw)
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 16}" text-anchor="middle">{value:+g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hbar_chart(rows, width=460, height=None):
    """Horizontal bars: (label, value, tooltip). Single hue: magnitude only."""
    rh = 26
    ml, mr, mt, mb = 4, 46, 4, 4
    height = height or (mt + mb + rh * len(rows))
    label_w = 195
    pw = width - ml - mr - label_w
    v_max = max(v for _, v, _ in rows) * 1.05

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i, (label, value, tip) in enumerate(rows):
        y = mt + i * rh
        bw = pw * value / v_max
        short = label if len(label) <= 28 else label[:27] + "…"
        parts.append(
            f'<text x="{ml}" y="{y + rh / 2 + 4:.1f}" class="ridelab">{esc(short)}</text>'
        )
        bar_x = ml + label_w
        parts.append(
            f'<path d="M {bar_x} {y + 5} L {bar_x + bw - 3:.1f} {y + 5} Q {bar_x + bw:.1f} {y + 5} '
            f'{bar_x + bw:.1f} {y + 8} L {bar_x + bw:.1f} {y + rh - 8} Q {bar_x + bw:.1f} {y + rh - 5} '
            f'{bar_x + bw - 3:.1f} {y + rh - 5} L {bar_x} {y + rh - 5} Z" fill="var(--seq)">'
            f"<title>{esc(tip)}</title></path>"
        )
        parts.append(
            f'<text x="{bar_x + bw + 6:.1f}" y="{y + rh / 2 + 4:.1f}" class="val">{value:.1f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def data_table(headers, rows) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<details><summary>Data table</summary><table><tr>{head}</tr>{body}</table></details>"


# ------------------------------------------------------------------------------------
# Page assembly
# ------------------------------------------------------------------------------------


def build_page(kpis: pd.DataFrame, preds: pd.DataFrame) -> str:
    champ = kpis[kpis["is_champion"]]
    champ_name = champ["model"].iloc[0]
    champ_wide = champ.set_index("kpi_name")["kpi_value"]
    run_id = str(kpis["run_id"].iloc[0])
    # The badge describes what is being SERVED, and the forecast rows are where a
    # fallback is recorded (a fallback refreshes predictions_last but leaves kpis_last
    # alone, so the KPI table still says kept_previous_model).
    run_status = str(preds["run_status"].iloc[0])
    status_label, status_class = STATUS_LABELS.get(run_status, (run_status, "neutral"))
    generated_at = pd.Timestamp(preds["generated_at"].iloc[0])
    trained_at = str(preds["model_trained_at"].iloc[0])[:19].replace("T", " ")

    bt = preds[preds["row_kind"] == "backtest"].copy()
    fc = preds[preds["row_kind"] == "forecast"]
    window = f"{fc['ts_local'].min():%b %d} to {fc['ts_local'].max():%b %d, %Y}"

    # The backtest carries every candidate's test-set predictions; the time-series
    # charts show the champion, the histogram grid compares all of them.
    bt_champ = bt[bt["model_name"] == champ_name]
    if bt_champ.empty:
        bt_champ = bt

    # Tiles
    tiles = []
    for key, label, unit, why in TILE_SPECS:
        value = champ_wide.get(key)
        digits = 1
        tiles.append(
            f'<div class="tile" title="{esc(why)}"><div class="v">{fmt(value, digits)}'
            f'<span class="u"> {unit}</span></div><div class="l">{esc(label)}</div></div>'
        )

    # Model comparison table
    order = ["prophet_fleet", "xgb_global", "xgb_local_fleet", "baseline_ride_mean"]
    cols = [
        ("mae", "MAE"), ("rmse", "RMSE"), ("pct_within_10min", "Within 10 min %"),
        ("pct_severe_miss", "Severe miss %"), ("bias", "Bias"),
        ("high_wait_mae", "High-wait MAE"), ("peak_hours_mae", "Peak MAE"),
    ]
    wide = kpis.pivot_table(index="model", columns="kpi_name", values="kpi_value")
    model_rows = []
    for model in order:
        if model not in wide.index:
            continue
        cells = "".join(f"<td>{fmt(wide.loc[model, k])}</td>" for k, _ in cols)
        cls = ' class="champ"' if model == champ_name else ""
        marker = " (champion)" if model == champ_name else ""
        model_rows.append(f"<tr{cls}><td>{esc(model)}{marker}</td>{cells}</tr>")
    model_table = (
        '<table><tr><th>Model</th>'
        + "".join(f"<th>{h}</th>" for _, h in cols)
        + "</tr>"
        + "".join(model_rows)
        + "</table>"
    )

    # Chart A: daily mean, predicted vs actual (champion)
    bt_champ = bt_champ.copy()
    bt_champ["date"] = bt_champ["ts_local"].dt.normalize()
    daily = bt_champ.groupby("date")[["actual_wait_min", "predicted_wait_min"]].mean()
    daily_labels = [d.strftime("%b %d") for d in daily.index]
    chart_daily = line_chart(
        daily_labels,
        [
            ("Actual", "--actual", daily["actual_wait_min"].round(2).tolist()),
            ("Predicted", "--predicted", daily["predicted_wait_min"].round(2).tolist()),
        ],
    )
    daily_table = data_table(
        ["Date", "Actual (min)", "Predicted (min)"],
        # strict: all three come from the same groupby index, so a length mismatch would
        # mean the table had silently dropped rows the chart still plotted.
        [(lab, fmt(a), fmt(p)) for lab, a, p in
         zip(daily_labels, daily["actual_wait_min"], daily["predicted_wait_min"], strict=True)],
    )

    # Chart B: mean wait by local hour (champion)
    hourly = bt_champ.groupby(bt_champ["ts_local"].dt.hour)[
        ["actual_wait_min", "predicted_wait_min"]
    ].mean()
    hour_labels = [f"{h:02d}:00" for h in hourly.index]
    chart_hourly = line_chart(
        hour_labels,
        [
            ("Actual", "--actual", hourly["actual_wait_min"].round(2).tolist()),
            ("Predicted", "--predicted", hourly["predicted_wait_min"].round(2).tolist()),
        ],
        width=450,
        height=240,
    )

    # Chart C: per-model error distributions (predicted - actual), legacy-v1 style:
    # one panel per candidate, MAE and RMSE in the panel title, shared bins and a
    # shared y-scale so the panels compare honestly.
    inner = np.arange(-40, 45, 5).astype(float)
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    model_order = [
        m
        for m in ["prophet_fleet", "xgb_global", "xgb_local_fleet", "baseline_ride_mean"]
        if m in set(bt["model_name"])
    ]
    hist_counts = {
        m: np.histogram(bt.loc[bt["model_name"] == m, "error_min"].to_numpy(), bins=edges)[0]
        for m in model_order
    }
    shared_y = max(c.max() for c in hist_counts.values()) if hist_counts else 0
    hist_panels = []
    for m in model_order:
        mae = fmt(wide.loc[m, "mae"], 2) if m in wide.index else "n/a"
        rmse = fmt(wide.loc[m, "rmse"], 2) if m in wide.index else "n/a"
        if m == champ_name:
            marker = " (champion)"
        elif m == "baseline_ride_mean":
            marker = " (the floor every model must beat)"
        else:
            marker = ""
        svg = histogram(list(edges), list(hist_counts[m]), width=460, height=230, y_top=shared_y)
        hist_panels.append(
            f'<div class="panel"><h3>{esc(m)}{marker}</h3>'
            f'<p class="stat">MAE = {mae} min | RMSE = {rmse}</p>{svg}</div>'
        )
    chart_hists = f'<div class="grid2">{"".join(hist_panels)}</div>'

    # Chart D: highest-error rides (champion)
    per_ride = (
        bt_champ.groupby(["park_name", "ride_key"])
        .agg(
            mae=("error_min", lambda e: float(np.abs(e).mean())),
            n=("error_min", "size"),
            ride_name=("ride_name", "first"),
        )
        .sort_values("mae", ascending=False)
        .head(15)
    )
    ride_rows = [
        (row.ride_name, row.mae, f"{row.ride_name} ({park}): MAE {row.mae:.1f} min over {row.n:,} test slots")
        for (park, _), row in per_ride.iterrows()
    ]
    chart_rides = hbar_chart(ride_rows)

    legend_two = (
        '<div class="legend">'
        '<span><span class="sw" style="background:var(--actual)"></span>Actual</span>'
        '<span><span class="sw" style="background:var(--predicted)"></span>Predicted</span>'
        "</div>"
    )

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    test_n = f"{len(bt_champ):,}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:;">
<title>Mapblazer Wait-Time Forecast</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>Mapblazer Wait-Time Forecast</h1>
<p class="sub">Weekly wait-time predictions for 5 California theme parks, 30-minute resolution.</p>

<div class="prov">
  <span class="badge {status_class}"><span class="dot"></span>{esc(status_label)}</span>
  <span><span class="k">Serving forecast</span><b>{esc(window)}</b></span>
  <span><span class="k">Model</span><b>{esc(champ_name)}</b></span>
  <span><span class="k">Run</span>{esc(run_id)}</span>
  <span><span class="k">Trained</span>{esc(trained_at)} UTC</span>
  <span><span class="k">Published</span>{generated_at:%Y-%m-%d %H:%M} UTC</span>
</div>

<div class="tiles">{"".join(tiles)}</div>

<div class="card">
  <h2>Model comparison</h2>
  <p class="why">Backtest on the held-out final 20 percent of history ({test_n} observations). The champion must beat the per-ride mean baseline.</p>
  {model_table}
</div>

<div class="card">
  <h2>Predicted vs actual, daily mean</h2>
  <p class="why">Champion backtest: mean wait across all rides per day. Hover for values.</p>
  {legend_two}
  {chart_daily}
  {daily_table}
</div>

<div class="card">
  <h2>Backtest error distributions by model</h2>
  <p class="why">Predicted minus actual on the held-out test set, minutes. Blue under-forecasts,
  red over-forecasts. Shared axes, so a wider spread means a genuinely worse model. The
  baseline predicts each ride's historical average wait, so the difference between its
  panel and the others is what the models actually learned.</p>
  {chart_hists}
</div>

<div class="grid2">
  <div class="card">
    <h2>Mean wait by local hour</h2>
    <p class="why">The evening peak is the part a timezone mistake silently deletes.</p>
    {legend_two}
    {chart_hourly}
  </div>
  <div class="card">
    <h2>Hardest rides to predict</h2>
    <p class="why">Highest champion backtest MAE. Long queues move in bursts.</p>
    {chart_rides}
  </div>
</div>

<footer>Run {esc(run_id)} | status {esc(run_status)} | page built {built}.
Data: Databricks gold tables (predictions_last, kpis_last), served via Supabase.</footer>
</div>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-parquet", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "site" / "index.html")
    args = parser.parse_args(argv)

    publish.load_env(ROOT / ".env")
    if args.from_parquet:
        kpis, preds = publish.fetch_from_parquet(args.from_parquet)
    else:
        kpis, preds = publish.fetch_from_warehouse()
    publish.validate_pair(kpis, preds)

    page = build_page(kpis, preds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    publish.log.info("dashboard written to %s (%d bytes)", args.out, len(page.encode("utf-8")))


if __name__ == "__main__":
    main()
