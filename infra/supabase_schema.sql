-- Supabase serving schema.
--
-- Run once in the Supabase SQL editor. Creates the tables the publish job upserts into
-- and opens read-only access so PostgREST can serve them.
--
-- PostgREST is why there is no API service in this repo: it exposes every table below as
-- a filtered, paginated, authenticated REST endpoint with no application code to write,
-- containerise, deploy or patch. For a read-mostly forecast table consumed by one
-- downstream optimiser, hand-rolling a service would be strictly worse.

-- ---------------------------------------------------------------------------------
-- Forecasts. The primary product: what every attraction's queue is expected to be,
-- at 30-minute resolution, for the next seven days.
-- ---------------------------------------------------------------------------------
create table if not exists predictions (
    park_name           text        not null,
    ride_key            text        not null,
    ride_name           text        not null,
    ts_local            timestamp   not null,
    ts_utc              timestamp   not null,
    predicted_wait_min  integer     not null,
    -- Bounds travel with every row. A route optimised on point estimates alone cannot
    -- distinguish a reliable 20-minute queue from a volatile one averaging the same.
    lower_bound         integer     not null,
    upper_bound         integer     not null,
    model_name          text        not null,
    model_version       text,
    forecast_week       text,
    generated_at        timestamptz not null default now(),
    -- Natural key, so a republished week upserts in place rather than duplicating.
    primary key (park_name, ride_key, ts_local)
);

-- The routing solver's access pattern: one park, one time window, ordered by time.
create index if not exists predictions_park_time_idx on predictions (park_name, ts_local);
create index if not exists predictions_ride_time_idx on predictions (ride_key, ts_local);

-- ---------------------------------------------------------------------------------
-- Weekly backtest scorecard, one row per candidate per run.
-- ---------------------------------------------------------------------------------
create table if not exists model_metrics (
    model                    text not null,
    overall_mae              double precision,
    overall_rmse             double precision,
    overall_p95_abs_error    double precision,
    overall_within_10min_pct double precision,
    overall_bias             double precision,
    high_wait_rides_mae      double precision,
    high_wait_rides_rmse     double precision,
    peak_hours_mae           double precision,
    weekend_holiday_mae      double precision,
    coverage                 double precision,
    interval_coverage_pct    double precision,
    n_entities_fitted        integer,
    train_rows               bigint,
    computed_at              timestamptz not null default now(),
    primary key (model, computed_at)
);

-- ---------------------------------------------------------------------------------
-- Production accuracy: forecasts scored against observations that did not exist when
-- the forecast was published. The only genuinely leak-free measurement in the system.
-- ---------------------------------------------------------------------------------
create table if not exists prediction_accuracy (
    forecast_week         text not null,
    segment               text not null,
    model_name            text,
    model_version         text,
    n                     bigint,
    mae                   double precision,
    rmse                  double precision,
    p95_abs_error         double precision,
    within_10min_pct      double precision,
    bias                  double precision,
    mean_actual           double precision,
    interval_coverage_pct double precision,
    primary key (forecast_week, segment)
);

create table if not exists accuracy_by_ride (
    park_name             text not null,
    ride_key              text not null,
    ride_name             text,
    n                     bigint,
    mae                   double precision,
    rmse                  double precision,
    bias                  double precision,
    mean_actual           double precision,
    interval_coverage_pct double precision,
    computed_at           timestamptz not null default now(),
    primary key (park_name, ride_key)
);

-- ---------------------------------------------------------------------------------
-- Ingestion drift. A collector that half-fails still posts a respectable MAE on the
-- rows it does return; volume and the share of zeros move first.
-- ---------------------------------------------------------------------------------
create table if not exists data_drift (
    week       text primary key,
    rows       bigint,
    entities   integer,
    mean_wait  double precision,
    pct_zero   double precision,
    p95_wait   double precision
);

-- ---------------------------------------------------------------------------------
-- Every promotion decision, taken or refused. A gate nobody can audit is not a gate.
-- ---------------------------------------------------------------------------------
create table if not exists promotion_log (
    decided_at   timestamptz not null,
    promoted     text,
    winner       text,
    incumbent    text,
    reason       text,
    winner_mae   double precision,
    baseline_mae double precision,
    incumbent_mae double precision,
    git_sha      text,
    primary key (decided_at, winner)
);

-- ---------------------------------------------------------------------------------
-- Run history. Also the keepalive: Supabase pauses a free project after 7 days with no
-- database request, and the collector writes here every 30 minutes.
-- ---------------------------------------------------------------------------------
create table if not exists pipeline_runs (
    job           text        not null,
    run_at        timestamptz not null,
    status        text        not null,
    detail        text,
    rows_affected bigint,
    primary key (job, run_at)
);

create index if not exists pipeline_runs_recent_idx on pipeline_runs (run_at desc);

-- ---------------------------------------------------------------------------------
-- Read-only public access.
--
-- IMPORTANT: projects created after 2026-05-30 no longer grant PostgREST access
-- implicitly -- without the explicit grants below the REST API returns an empty schema
-- and every request 404s, with nothing in the logs to explain why.
-- ---------------------------------------------------------------------------------
grant usage on schema public to anon, authenticated;

grant select on
    predictions, model_metrics, prediction_accuracy,
    accuracy_by_ride, data_drift, promotion_log, pipeline_runs
to anon, authenticated;

-- RLS on with a permissive select policy: the anon key can read and nothing else.
-- Writes use the service-role key, which lives only in GitHub Secrets.
do $$
declare t text;
begin
    foreach t in array array[
        'predictions', 'model_metrics', 'prediction_accuracy',
        'accuracy_by_ride', 'data_drift', 'promotion_log', 'pipeline_runs'
    ] loop
        execute format('alter table %I enable row level security', t);
        execute format('drop policy if exists "public read" on %I', t);
        execute format(
            'create policy "public read" on %I for select to anon, authenticated using (true)', t
        );
    end loop;
end $$;

-- Verify:
--   curl "$SUPABASE_URL/rest/v1/predictions?park_name=eq.Disneyland&limit=5" \
--        -H "apikey: $SUPABASE_ANON_KEY"
