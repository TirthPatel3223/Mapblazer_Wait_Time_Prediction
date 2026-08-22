# Known issues and deliberate trade-offs

Written down because a system that documents its own limits is easier to trust than one
that claims not to have any. Ordered by how much they would change the numbers.

---

## Open — would materially improve accuracy

### 1. No ride-status signal (largest known gap)

The model has no input for "this attraction is closed for refurbishment." Two of the ten
worst-predicted attractions in the last verification run were closures:

| Attraction | Production MAE | Bias | Mean actual |
|---|---|---|---|
| Pirates of the Caribbean | 18.49 | **+18.49** | **0.00** |
| Buzz Lightyear Astro Blasters | 17.22 | **+17.22** | **0.00** |

Bias equal to MAE with a mean actual of zero is the signature: the model confidently
predicts a normal queue for a ride that was shut all week. This is not a modelling
failure so much as a missing feature — the upstream `attractions` table carries a
`status` column that the pipeline does not currently read.

**Fix:** join `attractions.status` into silver and suppress forecasts for non-operating
rides, or add it as a feature. Roughly half a day.

### 2. No lag or autoregressive features

All four families are pure calendar-clock functions: hour, day of week, month, holiday
flag, and cyclical encodings. There is no lag-1 wait, no rolling mean, no park-level
concurrent load, no weather, and none of the `person_capacity` / `show_duration` columns
available in the upstream `attractions` table.

This is the single largest available accuracy lift and it is deliberately out of scope:
lags make a 7-day-ahead forecast recursive, which is a different and much larger piece of
work than the deployment this project is about. A hybrid — calendar-only beyond 24 hours,
autoregressive within it — is the sensible next step.

### 3. Zero-inflation is handled by the metric, not the model

About 37% of observations are exactly zero even after the timezone fix (down from 70%
before it — most of those "zeros" were closed-park hours being trained on). A single
regressor is being asked to model both "is there a queue at all" and "how long is it."

A two-stage model — classify operating/non-zero, then regress conditional on non-zero —
is the standard treatment. The segmented scorecard (`nonzero_actual`, `long_queues`)
exists so this weakness is visible rather than hidden inside an average.

---

## Open — operational

### 4. Bronze load rescans the whole landing volume

`jobs/bronze_load.py` reads every Parquet file in the volume and anti-joins on
`wait_time_id`. Correct and idempotent, but the scan grows with history. At current
volume (~20k rows/week) this is fine for well over a year; past that, partition-prune on
the `dt=` prefix or archive loaded files.

### 5. Park operating hours are hard-coded

`themepark.filters.PARK_CONSTRAINTS` carries fixed local open/close hours per park. Real
parks vary hours by season and by day. Wrong bounds do not corrupt training — they only
trim the forecast grid — but a park extending summer hours will have its late evening
unforecast until the constant is updated. Inferring the window per park per weekday from
observed activity is the fix.

### 6. Git history still contains the v1 model artefacts

`trained_models/` (422 MB, 245 JSON files, no LFS) is untracked going forward, but it is
still in the history and on the remote — the repo clones at ~122 MB. Purging it needs a
history rewrite and a force-push, which has not been done because it is destructive and
rewrites commit hashes anyone else may have pulled.

---

## Deliberate trade-offs

### 7. Hyperparameter search removed rather than fixed

v1 used `RandomizedSearchCV(cv=3)`, which is random K-Fold — on a time series, future
rows leak into the validation folds and the selected parameters are optimistic. The fix
would be `TimeSeriesSplit`. Instead the search was dropped entirely in favour of fixed
conservative parameters (`max_depth=6, n_estimators=200`).

That removes the leak, cuts training time substantially, and shrinks the per-ride
artefacts from 3.8 MB to a fraction of that. A proper rolling-origin search is worth
adding once there is more than the current ~9 months of history.

### 8. Ingestion runs on GitHub Actions, not Databricks

Databricks Free Edition restricts outbound internet to an allowlist of trusted domains,
so a Databricks job cannot reach the upstream Postgres or Supabase. Every network hop is
therefore inbound to Databricks, brokered by a runner.

On a paid workspace this would be a Lakeflow ingestion pipeline. The separation is
defensible on its own terms — ingestion failures cannot consume training compute — but it
is a constraint, not a preference.

### 9. Point forecasts are precomputed, not served on demand

The consumer is a time-dependent routing solver, which needs the entire cost surface
(every ride at every arrival time) rather than one point per request. A weekly batch into
an indexed Postgres table is the right shape, and it means no model-serving endpoint has
to stay warm. The cost is that an intra-week correction requires a manual job run.

### 10. Alerting is email plus a staleness banner

GitHub emails on workflow failure, Databricks emails on job failure, and the dashboard
turns red if the newest run is more than 36 hours old. There is no paging and no
on-call. For a system whose worst failure mode is serving last week's forecast for
another week, that is proportionate — but it is the first thing that would change if
anything depended on this operationally.

---

## Fixed in this version

Both were found by auditing the v1 evaluation path; see [`legacy/v1/`](legacy/v1) for the
original code and [`tests/`](tests) for the regression tests that pin them.

| | Defect | Effect |
|---|---|---|
| **1** | Park-local operating hours applied to UTC timestamps | Kept 00:00–15:00 local, discarded the entire evening peak. Only **47%** of wait-minute mass survived, versus **95%** now. Mean wait in the training set was 7.06 min; it is actually 15.30. `dayofweek` and `is_weekend` were wrong for every evening observation. |
| **2** | Two different ride-name sanitizers between training and evaluation | 43 of 120 rides silently resolved to nothing and were dropped from every reported metric — the excluded set averaged 10.03 min wait versus 5.88 for those kept. Every v1 headline number was computed on a low-wait subsample. |

Reproduce the first with `python scripts/validate_timezone_fix.py`, which runs the v1
filter and the corrected one side by side against the same extract.
