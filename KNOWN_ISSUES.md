# Known issues and deliberate trade-offs

Written down because a system that documents its own limits is easier to trust than one
that claims not to have any. Ordered by how much they would change the numbers.

---

## Open — would materially improve accuracy

### 1. No ride-status signal (largest known gap)

The model has no input for "this attraction is closed for refurbishment." The signature
is unmistakable in the per-ride errors: bias equal to MAE with a mean actual of zero, on
attractions that were shut for the whole holdout window. The model confidently predicts a
normal queue for a ride that was not operating.

This is not really a modelling failure — it is a missing feature. The upstream
`attractions` table carries a `status` column that neither the ingestion agent nor the
pipeline currently reads.

**Fix:** add `status` to the ingestion query and to bronze, then either suppress
forecasts for non-operating rides in silver or feed it in as a feature. Roughly half a
day, most of it a bronze schema change and a backfill.

### 2. No lag or autoregressive features

All four families are pure calendar-clock functions: hour, minute, day of week, month,
holiday flag, and cyclical encodings. There is no lag-1 wait, no rolling mean, no
park-level concurrent load, no weather, and none of the `person_capacity` /
`show_duration` columns available upstream.

This is the single largest available accuracy lift and it is deliberately out of scope:
lags make a 7-day-ahead forecast recursive, which is a different and much larger piece of
work than the deployment this project is about. A hybrid — calendar-only beyond 24 hours,
autoregressive within it — is the sensible next step.

### 3. Zero-inflation is handled by the metric, not the model

About 37 percent of silver observations are exactly zero even after the timezone fix. A
single regressor is being asked to model both "is there a queue at all" and "how long is
it."

A two-stage model — classify operating/non-zero, then regress conditional on non-zero —
is the standard treatment. The segmented KPIs (`high_wait_mae`, `peak_hours_mae`) exist
so this weakness stays visible instead of disappearing into an average.

---

## Open — operational

### 4. The landing volume grows without bound

`deploy/ec2/ingest.py` merges each batch from its own Parquet file by exact path, so
nothing rescans the volume and the merge cost does not grow with history. But the files
are never removed either. At roughly 5 MB/week that is fine for years; past that, expire
`dt=` prefixes older than a few months. Bronze is the audit trail, so deleting landed
files loses nothing.

### 5. Park operating hours are hard-coded

`PARK_HOURS` in `pipeline.py` carries fixed local open/close hours per park. Real parks
vary hours by season and by day. Wrong bounds do not corrupt training — they only trim
the forecast grid — but a park extending its summer hours will have its late evening
unforecast until the constant is updated. Inferring the window per park per weekday from
observed activity is the fix.

### 6. No automated test gate in CI

The unit and failure-path suite is developed and run locally; it is not published in this
repository, which carries only what deploys. CI therefore checks lint and imports, not
behaviour. The failure paths it covers — a failed quality check dropping `_current` while
`_last` survives, the fallback refreshing the forecast on a failed run, a first-ever run
with nothing to fall back to — are exactly the ones a regression would be quietest about,
so they have to be re-run by hand before a change to the promotion or fallback logic
ships.

### 7. Git history still contains the v1 model artefacts

`trained_models/` (422 MB, 245 JSON files, no LFS) is untracked going forward, but it is
still in the history and on the remote — the repo clones at roughly 122 MB. Purging it
needs a history rewrite and a force-push, which has not been done because it is
destructive and rewrites commit hashes anyone else may have pulled.

### 8. The Supabase keepalive has no margin

Supabase pauses a free project after 7 days without a database request, and the publish
workflow runs every 7 days. Any skipped or failed Sunday can therefore let the serving
API go to sleep, which looks like an outage rather than a paused project. A manual
`workflow_dispatch` wakes it. A scheduled read midweek would remove the coincidence.

---

## Deliberate trade-offs

### 9. Hyperparameter search removed rather than fixed

v1 used `RandomizedSearchCV(cv=3)`, which is random K-fold — on a time series, future
rows leak into the validation folds and the selected parameters are optimistic. The fix
would be `TimeSeriesSplit`. Instead the search was dropped entirely in favour of fixed
conservative parameters (`max_depth=6, n_estimators=200`).

That removes the leak, cuts training time substantially, and shrinks the per-ride
artefacts considerably. A proper rolling-origin search is worth adding once there is more
than the current nine months of history.

### 10. Ingestion and publishing both run outside Databricks

Databricks Free Edition restricts outbound internet, so a job cannot reach the upstream
Postgres or Supabase. Every network hop is therefore initiated from outside: the EC2
agent pushes bronze in, and GitHub Actions pulls gold out.

On a paid workspace ingestion would be a Lakeflow pipeline. The separation is defensible
on its own terms — an ingestion failure cannot consume training compute, and the source
database is never exposed to the internet — but it is a constraint, not a preference.

### 11. Ingestion cadence is set by warehouse cost, not by data freshness

Each ingestion run wakes a serverless SQL warehouse, so the hourly timer is a compute-quota
decision rather than a data one. The feed has 30-minute granularity and the model retrains
weekly, so nothing downstream notices; but if this ever fed a real-time consumer, the
warehouse round trip per batch would be the first thing to replace.

The quota is not theoretical. Ingestion stopped writing bronze on 2026-08-25; the weekly
job then never started on 2026-08-30, and the publish workflow could not open a warehouse
session at all (`BAD_REQUEST: Cannot create the resource, please try again later`) --
every symptom of a workspace that cannot provision serverless compute, with ~24 warehouse
wake-ups a day as the plausible cause. The pipeline now degrades rather than stopping
(see issue 14), but nothing in it can make compute available: widening the timer from 1h
to 6h is the lever, and the unit file already says so.

### 12. Point forecasts are precomputed, not served on demand

The consumer is a time-dependent routing solver, which needs the entire cost surface
(every ride at every arrival time) rather than one point per request. A weekly batch into
an indexed Postgres table is the right shape, and it means no model-serving endpoint has
to stay warm. The cost is that an intra-week correction requires a manual job run.

### 13. Alerting is email plus visible run provenance

GitHub emails on workflow failure, Databricks emails on job failure, and the dashboard
carries the run id, the run status (`fresh_model`, `kept_previous_model`, `stale_input`,
`fallback_after_failure`) and the publish timestamp, so a fallback week is visibly a
fallback. There is no paging and no on-call. For a system whose worst failure mode is
serving last week's forecast for another week, that is proportionate — but it is the
first thing that would change if anything depended on this operationally.

The gap this does not cover: a weekly job that never *starts* sends no failure email,
because nothing failed. That is why `publish.py` checks the forecast window rather than
trusting the tables to be recent — an elapsed window is the only signal a run that never
happened leaves behind.

### 14. A stale-input week is silent about what it does not know

When ingestion stops, the run skips training and re-scores the coming week with the
serving model (`run_status` = `stale_input`), so the forecast window stays live. What it
publishes is a model's opinion of a week it has no recent data for: as good as the model
was, and blind to anything that has changed since. The dashboard flags the status and the
KPI tiles still show the values measured when the model was trained, which is honest but
easy to skim past. A real fix is an accuracy backfill that scores those forecasts against
actuals once ingestion recovers, so a stale week is judged rather than just labelled.

---

## Fixed in this version

Both were found by auditing the v1 evaluation path.

| | Defect | Effect |
|---|---|---|
| **1** | Park-local operating hours applied to UTC timestamps | Kept 00:00-15:00 local and discarded the entire evening peak. Mean wait in the training set read 7.06 min; it is actually 15.4. `dayofweek` and `is_weekend` were wrong for every evening observation. |
| **2** | Two different ride-name sanitizers between training and evaluation | 43 of 120 rides silently resolved to nothing and were dropped from every reported metric — the excluded set averaged 10.03 min wait versus 5.88 for those kept. Every v1 headline number was computed on a low-wait subsample. |

The first is pinned by a quality check that runs on every pipeline run: the local hour
with the highest mean wait must fall between 11:00 and 20:00, which an inverted timezone
conversion cannot satisfy. The second is pinned by there being exactly one ride-key
function in `pipeline.py`, used by training, scoring and the forecast grid alike.
