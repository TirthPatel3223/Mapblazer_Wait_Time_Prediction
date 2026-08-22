"""Static dashboard generator.

Renders one self-contained HTML file with inline SVG. No build step, no JavaScript
framework, no external requests -- so it can be served from GitHub Pages for free, never
sleeps, and cannot break because a CDN changed. Databricks Apps would have been the
obvious home for this, but on Free Edition an App stops 24 hours after it is started, and
a dashboard that is down whenever someone actually looks at it is worse than no dashboard.

Every chart ships a table view beneath it, so nothing here depends on colour alone.
"""

from __future__ import annotations

import html
import logging
import math

import pandas as pd

log = logging.getLogger(__name__)

# Validated two-slot categorical palette (blue, orange) plus status colours.
# Verified with the palette validator in both modes, all pairs:
#   light worst pair CVD dE 24.7, normal-vision dE 33.6; dark 26.8 / 31.8. All checks pass.
PALETTE = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "s1": "#3987e5",
        "s2": "#d95926",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}

STALE_HOURS = 36


def esc(value) -> str:
    return html.escape(str(value))


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else esc(value)


# --------------------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------------------


def nice_ticks(vmax: float, vmin: float = 0.0, target: int = 4) -> list[float]:
    """Round tick values on a 1/2/5 x 10^n ladder.

    Axis labels like `24033.8` make a reader decode the number instead of reading the
    shape. Rounded steps cost nothing and are what every plotting library does for you --
    hand-rolled SVG has to do it explicitly.
    """
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next((m * magnitude for m in (1, 2, 5, 10) if raw <= m * magnitude), 10 * magnitude)

    # The top tick must sit at or above vmax, or marks scaled against it overflow the
    # plot and get clipped at the frame.
    ticks, value = [], math.floor(vmin / step) * step
    while value < vmax - step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    ticks.append(round(value, 10))
    return ticks


def short_week(label: str) -> str:
    """Compress a pandas weekly period label for an axis tick.

    `Period[W]` stringifies as `2026-03-16/2026-03-22`, which is 21 characters and
    guarantees overlapping ticks on any chart with more than three points. Only the week
    start matters for reading a trend.
    """
    text = str(label)
    start = text.split("/")[0]
    try:
        return pd.Timestamp(start).strftime("%b %-d")
    except (ValueError, TypeError):
        try:
            return pd.Timestamp(start).strftime("%b %d").replace(" 0", " ")
        except (ValueError, TypeError):
            return text[:10]


def _tick_label(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def _rounded_bar(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Horizontal bar: square at the baseline, rounded at the data end.

    Rounding only the value end keeps the baseline crisp so bars remain comparable, while
    the soft end reads as a measured quantity rather than a hard cut.
    """
    r = max(0.0, min(r, w, h / 2))
    if w <= r:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1):.1f}" height="{h:.1f}" />'
    return (
        f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} '
        f"V{y + h - r:.1f} A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z\" />"
    )


def hbar_chart(
    labels: list[str],
    values: list[float],
    unit: str = "min",
    highlight: int | None = None,
    height_per_row: int = 30,
    label_width: int = 210,
) -> str:
    """Horizontal bars for one measure across categories.

    One measure, so one hue -- categorical colours would imply the rows are different
    kinds of thing rather than the same thing measured across entities. Every bar is
    directly labelled, which is also the secondary encoding that keeps this readable
    without colour.
    """
    if not labels:
        return '<p class="empty">No data yet.</p>'

    rows = len(labels)
    width, pad_r = 720, 66
    height = rows * height_per_row + 16
    plot_w = width - label_width - pad_r
    vmax = max(max(values), 1e-9)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMinYMin meet">'
    ]
    for i, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = i * height_per_row + 8
        bar_h = height_per_row - 12  # 2px+ surface gap between adjacent bars
        w = (value / vmax) * plot_w
        cls = "bar-hi" if highlight is not None and i == highlight else "bar"
        parts.append(
            f'<text x="{label_width - 10}" y="{y + bar_h / 2 + 4}" class="lbl" '
            f'text-anchor="end">{esc(label)}</text>'
        )
        parts.append(
            f'<g class="{cls}"><title>{esc(label)}: {_fmt(value)} {esc(unit)}</title>'
            + _rounded_bar(label_width, y, w, bar_h)
            + "</g>"
        )
        parts.append(
            f'<text x="{label_width + w + 8}" y="{y + bar_h / 2 + 4}" class="val">'
            f"{_fmt(value)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    unit: str = "min",
) -> str:
    """Two series across shared categories -- one shared y scale, never a second axis."""
    if not categories:
        return '<p class="empty">No data yet.</p>'

    names = list(series)
    width, height = 720, 260
    pad_l, pad_b, pad_t = 46, 46, 14
    plot_w, plot_h = width - pad_l - 12, height - pad_b - pad_t
    data_max = max([v for values in series.values() for v in values] + [1e-9])
    ticks = nice_ticks(data_max * 1.08)
    vmax = max(max(ticks), 1e-9)

    group_w = plot_w / len(categories)
    bar_w = min(30.0, (group_w - 10) / len(names) - 2)  # 2px gap between adjacent bars

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']

    for tick in ticks:
        y = pad_t + plot_h * (1 - tick / vmax)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - 12}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_tick_label(tick)}</text>"
        )

    for gi, category in enumerate(categories):
        gx = pad_l + gi * group_w
        for si, name in enumerate(names):
            value = series[name][gi]
            bar_h = (value / vmax) * plot_h
            x = gx + (group_w - bar_w * len(names) - 2 * (len(names) - 1)) / 2 + si * (bar_w + 2)
            y = pad_t + plot_h - bar_h
            parts.append(
                f'<g class="s{si + 1}"><title>{esc(category)} — {esc(name)}: '
                f"{_fmt(value)} {esc(unit)}</title>"
                f'<path d="M{x:.1f},{y + bar_h:.1f} V{y + 4:.1f} '
                f"A4,4 0 0 1 {x + 4:.1f},{y:.1f} H{x + bar_w - 4:.1f} "
                f'A4,4 0 0 1 {x + bar_w:.1f},{y + 4:.1f} V{y + bar_h:.1f} Z" /></g>'
            )
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{height - pad_b + 20}" class="tick" '
            f'text-anchor="middle">{esc(category)}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - 12}" '
        f'y2="{pad_t + plot_h}" />'
    )
    parts.append("</svg>")

    legend = "".join(
        f'<span class="key"><i class="sw s{i + 1}"></i>{esc(name)}</span>'
        for i, name in enumerate(names)
    )
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def line_chart(
    x_labels: list[str], values: list[float], unit: str = "min", digits: int = 2
) -> str:
    """Single series over time. No legend box -- the title names the series."""
    if len(values) < 2:
        return (
            '<p class="empty">Needs at least two completed weeks. '
            "The first scheduled run seeds this.</p>"
        )

    width, height = 720, 240
    pad_l, pad_b, pad_t = 46, 42, 14
    plot_w, plot_h = width - pad_l - 16, height - pad_b - pad_t
    ticks = nice_ticks(max(values) * 1.02, min(min(values) * 0.92, 0), target=5)
    vmin, vmax = min(ticks), max(ticks)
    span = max(vmax - vmin, 1e-9)

    def px(i: int) -> float:
        return pad_l + (i / (len(values) - 1)) * plot_w

    def py(v: float) -> float:
        return pad_t + plot_h * (1 - (v - vmin) / span)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for tick in ticks:
        y = py(tick)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_tick_label(tick)}</text>"
        )

    points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline class="line s1-stroke" points="{points}" />')

    for i, v in enumerate(values):
        parts.append(
            f'<g class="dot s1"><title>{esc(x_labels[i])}: {_fmt(v, digits)} {esc(unit)}</title>'
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" /></g>'
        )

    step = max(1, len(x_labels) // 8)
    for i in range(0, len(x_labels), step):
        parts.append(
            f'<text x="{px(i):.1f}" y="{height - pad_b + 20}" class="tick" '
            f'text-anchor="middle">{esc(x_labels[i])}</text>'
        )

    # Direct-label the most recent point only -- a number on every point is noise.
    parts.append(
        f'<text x="{px(len(values) - 1):.1f}" y="{py(values[-1]) - 12:.1f}" class="val" '
        f'text-anchor="end">{_fmt(values[-1], digits)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def forecast_band_chart(ts: list, yhat: list[float], lo: list[float], hi: list[float]) -> str:
    """Point forecast inside its uncertainty band -- what the router actually consumes."""
    if len(yhat) < 2:
        return '<p class="empty">No forecast published yet.</p>'

    width, height = 720, 250
    pad_l, pad_b, pad_t = 46, 40, 14
    plot_w, plot_h = width - pad_l - 16, height - pad_b - pad_t
    ticks = nice_ticks(max(max(hi), 1.0) * 1.06)
    vmax = max(max(ticks), 1e-9)

    def px(i: int) -> float:
        return pad_l + (i / (len(yhat) - 1)) * plot_w

    def py(v: float) -> float:
        return pad_t + plot_h * (1 - v / vmax)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for tick in ticks:
        y = py(tick)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_tick_label(tick)}</text>"
        )

    upper = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(hi))
    lower = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in reversed(list(enumerate(lo))))
    parts.append(f'<polygon class="band s1-fill" points="{upper} {lower}" />')
    parts.append(
        '<polyline class="line s1-stroke" points="'
        + " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(yhat))
        + '" />'
    )

    step = max(1, len(ts) // 7)
    for i in range(0, len(ts), step):
        parts.append(
            f'<text x="{px(i):.1f}" y="{height - pad_b + 20}" class="tick" '
            f'text-anchor="middle">{esc(str(ts[i])[5:16])}</text>'
        )
    parts.append("</svg>")
    return (
        '<div class="legend"><span class="key"><i class="sw s1"></i>predicted wait</span>'
        '<span class="key"><i class="sw band-sw"></i>80% interval</span></div>' + "".join(parts)
    )


def table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """The table view every chart carries, so nothing relies on colour alone."""
    if df is None or df.empty:
        return '<p class="empty">No data.</p>'
    view = df.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in view.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v) if isinstance(v, float) else esc(v)}</td>" for v in row) + "</tr>"
        for row in view.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def details(summary_text: str, content: str) -> str:
    return f"<details><summary>{esc(summary_text)}</summary>{content}</details>"
