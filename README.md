# Theme Park Wait-Time Forecasting

Weekly retrained wait-time forecasts for 109 attractions across 5 California theme
parks (Disneyland, Disney California Adventure, Universal Studios Hollywood, SeaWorld
San Diego, Six Flags Magic Mountain), at 30-minute resolution, seven days ahead. Runs
unattended on Databricks Free Edition, serves through Supabase and GitHub Pages, and
costs $0/month.

[Live dashboard](https://TirthPatel3223.github.io/Mapblazer_Wait_Time_Prediction/)

## How it works

Three components, each running where it has to:

    EC2 (source database)  ->  Databricks (bronze -> gold)  ->  Supabase + GitHub Pages
    deploy/ec2/ingest.py       pipeline.py                      publish.py, dashboard.py
    hourly                     Sunday 06:00 UTC                 Sunday 08:00 UTC

### Ingestion: the EC2 agent

The upstream wait-time database lives on an AWS EC2 instance that is not ours, so the
agent runs *there* and pushes outbound rather than being pulled from — see
[`deploy/ec2/`](deploy/ec2) for the full deployment. Pulling would mean opening Postgres
to the internet for a rotating set of runner IPs; pushing means the database connection
stays on the loopback interface, no firewall rule changes, and the only network access
needed is outbound HTTPS.

One run, hourly, on a systemd timer:

    1. watermark   MAX(wait_time_id) already in themepark.bronze.wait_times_raw
    2. read        SELECT ... FROM wait_times JOIN attractions JOIN themeparks
                   WHERE wait_time_id > watermark ORDER BY wait_time_id LIMIT 200000
    3. land        write the batch as Parquet into /Volumes/themepark/bronze/landing/
    4. merge       MERGE that file into bronze on wait_time_id, insert-only

The watermark is read from bronze itself rather than from a cursor file on the box, so
it always describes what actually landed. A run killed at any point re-reads the same
range next time and the `MERGE ... WHEN NOT MATCHED` absorbs the overlap; there is no
local state that can drift out of sync and nothing to repair by hand. Bronze is
append-only and never edited — it is the audit trail, and every correction happens
downstream in silver.

### Training and serving: the weekly Databricks job

The entire weekly job is one Python file, `pipeline.py`, executed as a single
Databricks serverless task every Sunday at 06:00 UTC:

    themepark.bronze.wait_times_raw          842,539 raw queue observations
      |
      |  clean: drop duplicate readings, sentinel waits (>=900), negative waits,
      |  known-bad attractions; convert UTC to America/Los_Angeles FIRST, then
      |  filter to each park's local operating hours; canonical ride keys;
      |  resample to a fixed 30-minute grid; drop rides with under 100 observations
      v
    themepark.silver.wait_times_current      ~390K rows, 109 attractions
      |
      |  chronological 80/20 split (never random: random folds leak the future)
      |  train three candidates with fixed hyperparameters:
      |    prophet_fleet    one Prophet per ride (flat growth, daily+weekly
      |                     seasonality, US holidays, 80% intervals)
      |    xgb_global       one XGBoost across all rides (ride as categorical)
      |    xgb_local_fleet  one XGBoost per ride
      |  plus a per-ride historical-mean baseline as the floor
      v
    themepark.gold.kpis_current              MAE, RMSE, within-10-min, severe-miss,
                                             bias, high-wait MAE, peak-hours MAE
                                             per model, serving model flagged
    themepark.gold.predictions_current       7-day forecast (serving model) +
                                             test-set backtest rows for every model
                                             and the baseline (predicted, actual,
                                             error), row_kind distinguishes them

### The current / last serving pattern

Every silver and gold table exists twice: `_current` and `_last`. A run writes only
`_current`; quality checks run against `_current`; and only when everything passes are
the tables promoted `_current -> _last` with atomic `CREATE OR REPLACE ... DEEP CLONE`
commits. Supabase and the dashboard read `_last` only, so a failed or half-finished
run can never take down a working dashboard.

Quality checks include: silver row count vs last week, attraction count, zero-wait
share, null checks, and a timezone tripwire (the local hour with the highest mean wait
must fall between 11:00 and 20:00 -- inverted timezone handling relocates the
afternoon peak into the late evening); gold checks cover model presence, finite MAEs,
serving-model-beats-baseline, forecast ride coverage, bound ordering, and row-count
bands.

### Promotion and rollback

Model artifacts are plain files in the `themepark.gold.models` volume -- Prophet
serialized to JSON per ride, XGBoost via `save_model` -- written to an immutable
`runs/<run_id>/` directory with a manifest. A tiny `serving.json` pointer names the
serving run and is written last, only after the tables have been promoted.

A newly trained model ships only if its holdout MAE beats the incumbent's recorded MAE
by at least 1 percent. Otherwise the incumbent is reloaded from disk and
re-scored over the upcoming week, so the forecast window is fresh either way. Rolling
back is editing `serving.json` to an earlier `run_id`.

### When ingestion stops

Bronze going quiet is not a failure the shape checks can see: the row count is unchanged,
so silver validates and the run would happily retrain on last week's data and report
success. So the run asks the one question the checks do not -- how old is the newest
observation? Past `MAX_INPUT_AGE_HOURS` (36) it skips training entirely, re-scores the
coming week with the model already serving, and promotes gold as usual. The forecast
window stays live for the dashboard and the API, `run_status` is `stale_input` so the
dashboard says why, and no compute is spent fitting three models on a repeat of last
week. The check runs before training because those fits are the expensive part.

This keeps the forecast current; it cannot keep it correct. A model scoring a week it has
no recent data for is exactly as good as it was when it was trained, and blind to
anything that changed since.

### Failure behavior

Any failure -- a quality check, a training error, unreadable bronze -- drops the
`_current` tables, leaves `_last` untouched, re-scores the upcoming week with the
previous serving model (validated against the same prediction checks) so the dashboard
never serves a forecast window that has slid into the past, and still fails the job
so the failure alerts. Both gold tables carry `run_id`, `run_status` (`fresh_model`,
`kept_previous_model`, `stale_input`, or `fallback_after_failure`) and `generated_at`, so
a fallback week is visibly a fallback. A first-ever run with nothing to fall back to
changes nothing and says so.

Publishing has the matching guard on the other side: `publish.py` refuses a serving pair
whose forecast window has already elapsed. A weekly job that never started leaves gold
perfectly self-consistent and simply old, which is otherwise indistinguishable from a
good week until someone reads the dates.

### Serving

Databricks Free Edition jobs have no outbound internet, so serving is pulled, not
pushed. On Sundays at 08:00 UTC (or on demand) a GitHub Actions workflow:

1. reads `gold.kpis_last` and `gold.predictions_last` over the Databricks SQL
   warehouse (`publish.py`),
2. loads both into Supabase staging tables in idempotent chunks, then swaps staging
   into serving with a table-rename inside a single Postgres transaction
   (`publish_serving()` in `supabase_schema.sql`) -- readers always see one
   consistent pair, and Supabase's PostgREST exposes the tables as a public
   read-only REST API,
3. renders the dashboard (`dashboard.py`) -- a single static HTML file with inline
   SVG charts, no JavaScript, built from the same data -- and deploys it to GitHub
   Pages. The dashboard deploys even if the Supabase push fails; the job still goes
   red so the failure is visible.

## Results (holdout: final 20 percent of Dec 2025 - May 2026, 78,110 observations)

| Model | MAE | RMSE | Within 10 min | High-wait MAE |
|---|---|---|---|---|
| prophet_fleet (serving) | 7.01 | 11.49 | 76.6% | 10.3 |
| xgb_global | 8.26 | 12.88 | 71.8% | 12.2 |
| xgb_local_fleet | 8.45 | 14.70 | 71.9% | 12.9 |
| baseline (per-ride mean) | 9.51 | 14.47 | 64.5% | 14.4 |

## Layout

    pipeline.py            the entire Databricks job (single spark_python_task)
    publish.py             gold _last -> Supabase, atomic rename swap via RPC
    dashboard.py           gold _last -> static site/index.html (inline SVG, no JS)
    databricks.yml         asset bundle: one job, one task, one environment
    supabase_schema.sql    serving tables, staging twins, swap RPC, grants
    deploy/ec2/            the ingestion agent installed on the source-database host

The repository carries only what deploys. The superseded v1/v2 implementations and the
local test suite are kept outside it.

## Running it

Ingestion, on the EC2 instance that hosts the source database:

    sudo bash deploy/ec2/install.sh            # then fill in /opt/themepark/.env
    python deploy/ec2/ingest.py --dry-run      # reports the gap, writes nothing
    python deploy/ec2/ingest.py --drain        # backfill in one pass

Training and serving, from anywhere:

    pip install -r requirements.txt -r requirements-dev.txt
    ruff check .
    databricks bundle validate --target prod
    databricks bundle deploy   --target prod
    databricks bundle run themepark_weekly --target prod
    python publish.py                          # push serving tables to Supabase
    python dashboard.py --out site/index.html  # render the dashboard

Secrets live in `.env` at the repo root (gitignored); see `.env.example` for the keys,
which are also the GitHub Actions secret names. The EC2 agent has its own
`/opt/themepark/.env` with the `PG_*` credentials — those exist only on that machine,
because nothing outside it ever touches the source database.
`supabase_schema.sql` must be run once in the Supabase SQL editor before the first
publish.

## Querying the forecast API

    curl "https://xkwbvgfkazqgvkzqwxvg.supabase.co/rest/v1/predictions?row_kind=eq.forecast&park_name=eq.Disneyland&limit=5" \
         -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2J2Z2ZrYXpxZ3ZrenF3eHZnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NzM1NDY2NywiZXhwIjoyMTAyOTMwNjY3fQ.WRac3sWazNU5aXzWsnL7z4UHQfGKi6b1ieKkBd-IZWE"

Every forecast row carries `predicted_wait_min` with `lower_bound`/`upper_bound`
(80 percent interval), park-local and UTC timestamps, and the run provenance columns.
