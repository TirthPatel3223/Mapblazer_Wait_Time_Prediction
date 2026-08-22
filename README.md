# Theme Park Wait-Time Forecasting

Weekly retrained wait-time forecasts for 109 attractions across 5 California parks, at
30-minute resolution, seven days ahead. Runs unattended on Databricks Free Edition,
serves through Supabase and GitHub Pages, and costs $0/month.

[Live dashboard](https://TirthPatel3223.github.io/Mapblazer_Wait_Time_Prediction/) |
[Known issues](KNOWN_ISSUES.md)

## Architecture

One Python file, `pipeline.py`, runs as a single Databricks serverless job task every
Sunday and does the whole thing end to end:

    bronze.wait_times_raw
      -> silver.wait_times_current      clean, timezone-correct, 30-minute grid
      -> train prophet_fleet, xgb_global, xgb_local_fleet (fixed hyperparameters)
      -> KPIs on a chronological 80/20 holdout, vs a per-ride mean baseline
      -> gold.kpis_current + gold.predictions_current (forecast + backtest rows)
      -> quality checks
      -> promote every _current table to _last (atomic DEEP CLONE)
      -> champion.json pointer update in /Volumes/themepark/gold/models

Serving reads the `_last` tables only. A failed or half-finished run can therefore
never take down the dashboard: any failure drops the `_current` tables, leaves `_last`
untouched, re-scores the upcoming week with the previous champion (Step 5c fallback) so
the forecast window stays fresh, and still fails the job so the failure alerts.

The champion model is promoted only if its holdout MAE beats the incumbent's recorded
MAE by at least 1 percent; otherwise the previous model is reloaded from plain-file
artifacts (Prophet JSON, XGBoost `save_model`) and re-scored. No MLflow, no pickles,
no registry: a number compared against a number in a table, plus a JSON pointer.

Databricks Free Edition jobs have no outbound internet, so `publish.py` (GitHub
Actions, Sundays 08:00 UTC, or a laptop) pulls the gold `_last` tables over the SQL
warehouse, pushes both to Supabase atomically through the `publish_serving()` RPC (one
Postgres transaction), and `dashboard.py` renders the static dashboard for Pages.

## Layout

    pipeline.py            the entire Databricks job (single spark_python_task)
    publish.py             gold _last -> Supabase, atomic swap via RPC
    dashboard.py           gold _last -> static site/index.html (inline SVG, no JS)
    databricks.yml         asset bundle: one job, one task, one environment
    supabase_schema.sql    serving tables, staging twins, swap RPC, grants (run once)
    tests/                 unit + integration tests incl. every failure path
    legacy/                previous implementations, kept for reference only

## Results (holdout: final 20 percent of Dec 2025 - May 2026, 78,110 observations)

| Model | MAE | RMSE | Within 10 min | High-wait MAE |
|---|---|---|---|---|
| prophet_fleet (champion) | 7.01 | 11.49 | 76.6% | 10.3 |
| xgb_global | 8.29 | 12.93 | 71.7% | 12.2 |
| xgb_local_fleet | 8.46 | 14.72 | 72.0% | 12.9 |
| baseline (per-ride mean) | 9.51 | 14.47 | 64.5% | 14.4 |

## Running it

    pip install -r requirements.txt -r requirements-dev.txt
    pytest -q                                  # includes the failure-path proofs
    databricks bundle validate --target prod
    databricks bundle deploy   --target prod
    databricks bundle run themepark_weekly --target prod
    python publish.py                          # push serving tables to Supabase
    python dashboard.py --out site/index.html  # render the dashboard

Secrets live in `.env` at the repo root (gitignored); see `.env.example` for the keys.
`supabase_schema.sql` must be run once in the Supabase SQL editor before the first
publish (newer Supabase projects grant no PostgREST access implicitly).
