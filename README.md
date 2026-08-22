# Theme Park Wait-Time Forecasting — Production Pipeline

An automated forecasting system that ingests live queue data every 30 minutes, retrains
and re-evaluates four model families every Sunday, promotes a champion through an MLflow
registry gate, publishes a 7-day forecast for every attraction to a REST-accessible
Postgres, and measures last week's predictions against what actually happened.

Nobody touches it. It runs at **$0/month**.

📊 **[Live dashboard](https://TirthPatel3223.github.io/Mapblazer_Wait_Time_Prediction/)** ·
🔌 [REST API](#rest-api) · 🐛 [Known issues](KNOWN_ISSUES.md)

---

## Current production numbers

Measured on **109 attractions across 5 parks**, forecasting a full week ahead at
30-minute resolution.

| | Backtest (14-day holdout) | **Production (live)** |
|---|---|---|
| MAE | 6.81 min | **6.84 min** |
| RMSE | 11.31 min | **10.99 min** |
| Within 10 minutes | 77.1% | **77.2%** |
| 80% interval coverage | — | **85.1%** |
| Coverage (rides servable) | 1.000 | 1.000 |

"Production" means the forecast was published **before any of those observations
existed**, then scored against them a week later. It is the only number here with no
possibility of leakage, and the backtest tracks it at 1.00× — the offline estimate is
honest.

**Candidate leaderboard** (same holdout, retrained weekly):

| Model | MAE | RMSE | p95 abs err | Within 10 min | High-wait MAE |
|---|---|---|---|---|---|
| **prophet_fleet** *(champion)* | **6.81** | 11.31 | 24.3 | 77.1% | 9.98 |
| xgb_global | 7.06 | 12.00 | 26.7 | 76.3% | 10.38 |
| xgb_local_fleet | 7.18 | 12.68 | 28.7 | 75.8% | 10.68 |
| baseline (per-ride mean) | 9.65 | 14.70 | 30.5 | 63.9% | 14.56 |

> **On the previous version's numbers.** An earlier iteration of this project reported a
> Prophet MAE of **3.27 min**. That figure was wrong — it was computed on a 77-ride
> low-wait subsample with the busiest hours of every day filtered out, because of two
> defects described below. **6.84 min on the whole problem is the real number**, and it
> is worth more than the flattering one.

---

## The two defects this rewrite fixed

Both were found by auditing the evaluation path rather than the models. Both are now
pinned by regression tests in [`tests/`](tests); the original code is in
[`legacy/v1/`](legacy/v1).

### 1 · Park-local operating hours applied to UTC timestamps

The source database stores UTC. The filter declared hours that are plainly local
(`Disneyland: open 8, close 24`) and applied them straight to UTC hours. The parks are
UTC−8/−7, so "keep hours 8 through 23" actually kept **00:00–15:00 local** — seven hours
of guaranteed closed-park zeros — and discarded the entire **16:00–23:00 evening peak**.

It corrupted the features too: `dayofweek` and `is_weekend` came from UTC, so Sunday
20:00 local was labelled Monday (weekend flag lost) and Friday 20:00 was labelled
Saturday (weekend flag invented) — the most predictive feature in the model, wrong at
exactly the hours that matter.

```
                     v1 (UTC bug)    v2 (park-local)
rows kept                 423,947            393,821
mean wait (min)              7.06              15.30
zeros                       69.8%              36.8%
wait-minute mass            47.1%              95.0%   ← more than half the signal was gone
Disneyland peak      "UTC hour 20"     local hour 12
```

After the fix the daily profile is finally a theme park: ramp from 08:00, plateau
11:00–19:00, decay to close.

```
python scripts/validate_timezone_fix.py     # runs both filters side by side
```

### 2 · Two different ride-name sanitizers

Training wrote artefacts with `re.sub(r'[^\w\s-]','',name)`. Evaluation looked them up
with `name.replace(' ','_')`. Any attraction containing an apostrophe, colon, comma, `!`
or `&` produced a path that did not exist, hit a bare `except: pass`, and was dropped
from the metrics.

**43 of 120 rides vanished** — and they were the marquee ones:

| | Rides | Mean wait |
|---|---|---|
| Included in v1 metrics | 77 | 5.88 min |
| **Silently dropped** | **43** | **10.03 min** |

Rise of the Resistance, Guardians — Mission: BREAKOUT!, Toy Story Midway Mania!, Soarin'
Around the World, Peter Pan's Flight, Tiana's Bayou Adventure. Every published accuracy
figure was measured with the hard cases removed.

The fix is structural, not a patched string: `themepark.naming.canonical_ride_key` is the
single implementation, models hold their members in a dict keyed by it, and
**`coverage` is now a gated metric** — a model that cannot serve 95% of the fleet is
ineligible for promotion, and the scoring job fails rather than publishing a partial
forecast.

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│  SOURCE VM (EC2)                                   │
│    Postgres ── localhost ──► collector (systemd)   │
│                              every 30 min          │
│    psycopg → Parquet → Databricks SDK upload       │
└────────────┬───────────────────────────────────────┘
             │ OUTBOUND https only — Postgres is never exposed
             ▼
┌──────────────────────────────────────────────────────┐
│  DATABRICKS  (serverless, Unity Catalog, Delta)      │
│                                                      │
│   /Volumes/themepark/bronze/landing/                 │
│        └─► bronze.wait_times_raw    append-only      │
│              └─► silver.wait_times  UTC→local, 30min │
│                                                      │
│   ══ WEEKLY WORKFLOW — Sunday 06:00 UTC ══           │
│     train   4 candidates → MLflow                    │
│     gate    re-score incumbent, promote or decline   │
│     score   next 7d × 30min × all rides              │
│     verify  last week's forecast vs actuals          │
└────────────┬─────────────────────────────────────────┘
             │ GitHub Actions pulls (Databricks has no
             │ outbound internet on the free tier)
             ▼
   ┌──────────────────┐        ┌──────────────────┐
   │  Supabase        │        │  GitHub Pages    │
   │  PostgREST API   │        │  dashboard       │
   └────────┬─────────┘        └──────────────────┘
            └──► time-dependent routing optimiser
```

**Why ingestion pushes instead of pulls.** The source database runs on a VM alongside
other services. Pulling would mean exposing Postgres to the internet for a rotating set of
runner IPs; pushing means the VM makes an ordinary outbound HTTPS call it can already
make. `listen_addresses` stays `localhost`, no security-group rule changes, no SSH tunnel,
and credentials never cross the internet. Databricks Free Edition also blocks outbound
internet, so it could not have dialled the database itself in any case.

**Bootstrapping without that VM.** `jobs/seed_from_csv.py` writes a historical extract
into the landing volume in the identical schema the collector emits, so the entire
pipeline can be deployed and verified before live ingestion exists. The collector's
watermark then resumes exactly where the seed stopped — no gap, no overlap, no manual
reconciliation.

---

## The promotion gate

The part that makes this a deployment rather than a cron entry. Every Sunday all four
candidates are retrained **and the incumbent champion is re-scored on the same fresh
holdout** — comparing against its stored metrics from a previous week would let a quiet
week masquerade as a model improvement.

A challenger is promoted only if **all** hold:

| Check | Threshold | Why |
|---|---|---|
| `coverage` | ≥ 95% of the fleet | the defect-2 tripwire — accuracy must not improve by shrinking the population |
| beats baseline | strictly | if nothing beats a per-ride mean, something upstream is broken; the run **fails** rather than shipping |
| MAE improvement | ≥ 2% | a smaller gain is holdout noise, and churning the champion invalidates the accuracy history |
| high-wait RMSE | ≤ 105% of incumbent | better on average but worse where queues are long is not better |

Every decision — promoted or declined — is written to `gold.promotion_log` and shown on
the dashboard. *A gate that has never said no is not a gate*, so the test suite asserts
refusal in five distinct scenarios.

---

## REST API

PostgREST exposes the forecast table directly — there is no API service in this repo to
deploy, monitor or patch.

```bash
curl "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2J2Z2ZrYXpxZ3ZrenF3eHZnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODczNTQ2NjcsImV4cCI6MjEwMjkzMDY2N30.LNL9hTJEZXTMpaWyjPTIynvJE88WXJuO3ZYUhaRZD5M/rest/v1/predictions?\
park_name=eq.Disneyland&\
ts_local=gte.2026-08-24T09:00:00&ts_local=lt.2026-08-24T21:00:00&\
select=ride_name,ts_local,predicted_wait_min,lower_bound,upper_bound&\
order=ts_local" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2J2Z2ZrYXpxZ3ZrenF3eHZnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODczNTQ2NjcsImV4cCI6MjEwMjkzMDY2N30.LNL9hTJEZXTMpaWyjPTIynvJE88WXJuO3ZYUhaRZD5M"
```

```json
[{"ride_name": "Space Mountain", "ts_local": "2026-08-24T09:00:00",
  "predicted_wait_min": 22, "lower_bound": 11, "upper_bound": 38}]
```

Bounds ship with every row. The consumer is a time-dependent travelling-salesman solver,
and a route optimised on point estimates alone cannot distinguish a reliable 20-minute
queue from a volatile one averaging the same.

The anon key is read-only (RLS `SELECT` policy); the service-role key exists only in
GitHub Secrets.

---

## Repository layout

```
src/themepark/          the library — imported identically by laptop, runner and Databricks
  naming.py             canonical_ride_key — ONE implementation (kills defect 2)
  timeutils.py          UTC → America/Los_Angeles (kills defect 1)
  features.py           the single feature builder (was duplicated in 5 files)
  filters.py            operating windows, in local time
  silver.py             bronze → silver transform, shared by job and tests
  models/               baseline · prophet_fleet · xgb_global · xgb_local_fleet
  evaluate.py           segmented backtest scorecard
  promote.py            the champion/challenger gate
  score.py              7-day forecast grid
  verify.py             production accuracy: predictions vs what happened
  dashboard*.py         static dashboard generator

jobs/                   thin entry points (Databricks tasks, VM collector, runner jobs)
  collect.py            source Postgres → Parquet → Unity Catalog volume
  seed_from_csv.py      warm-start the lakehouse from a historical extract
deploy/ec2/             systemd unit, timer and installer for the source VM
.github/workflows/      collect · train · publish · ci
databricks.yml          Asset Bundle — the weekly job as code
infra/supabase_schema.sql
scripts/                local_pipeline.py · validate_timezone_fix.py
tests/                  70 tests, mostly regressions for the two defects
legacy/v1/              superseded scripts, kept for reference
```

---

## Running it

```bash
pip install -r requirements-dev.txt
pytest -q                                    # 70 tests
python scripts/validate_timezone_fix.py      # proves defect 1 is fixed
python scripts/local_pipeline.py --fast      # full cycle offline, ~1 min
python jobs/build_dashboard.py --source local --dir artifacts/local_run
```

`scripts/local_pipeline.py` is a genuine dress rehearsal: it builds silver, trains all
four candidates, runs the gate, generates a forecast, then scores that forecast against a
withheld future week — the same thing the scheduled job does every Sunday.

**Deploying** — see [`.env.example`](.env.example) for required secrets, run
[`infra/supabase_schema.sql`](infra/supabase_schema.sql) once, then:

```bash
databricks bundle deploy --target prod
python jobs/seed_from_csv.py                 # warm start from the local extract
databricks bundle run themepark_weekly
```

Live ingestion is attached separately, on the VM that hosts the source database — see
[`deploy/ec2/`](deploy/ec2). `scripts/check_upstream.py` verifies connectivity layer by
layer before you trust it.

---

## Stack

**Databricks** (Unity Catalog · Delta Lake medallion · Workflows · serverless) ·
**MLflow** (tracking · Model Registry · `@champion` alias) ·
**GitHub Actions** (ingestion · CI · Pages) ·
**Databricks Asset Bundles** (jobs as code) ·
**Supabase / PostgREST** (serving + REST) ·
Prophet · XGBoost · pandas · pytest · ruff

---

*Wait-time data is sourced from a partner operational database and is never committed to
this repository. Park and attraction names are public information.*
