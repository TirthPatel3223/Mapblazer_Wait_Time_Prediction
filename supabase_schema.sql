-- Supabase serving schema. Run in the Supabase SQL editor.
--
-- Safe to re-run: it drops and recreates the serving tables (they hold no unique
-- state -- every publish fully replaces them), so after re-running this file, run
-- `python publish.py` once to refill them.
--
-- publish.py loads both serving tables atomically: it fills the *_staging tables in
-- chunks, then calls publish_serving(), whose DELETE + INSERT of both serving tables
-- executes inside ONE Postgres transaction (every PostgREST function call is a single
-- transaction). If anything fails, the transaction rolls back and Supabase keeps the
-- previous consistent pair -- the dashboard can never see this week's KPIs beside last
-- week's forecast.
--
-- Notes for this project:
--   * Projects created after 2026-05-30 do not auto-grant PostgREST access. Without
--     the explicit grants at the bottom every REST request 404s.
--   * The pg-safeupdate guard is active, so every DELETE carries WHERE true.

drop table if exists kpis_staging cascade;
drop table if exists predictions_staging cascade;
drop table if exists kpis cascade;
drop table if exists predictions cascade;

create table kpis (
    model            text not null,
    kpi_name         text not null,
    kpi_value        double precision,
    is_champion      boolean not null default false,
    trained_at       text,
    run_id           text,
    run_status       text,
    generated_at     timestamptz,
    model_trained_at text,
    primary key (model, kpi_name)
);

create table predictions (
    row_kind           text not null,           -- 'forecast' or 'backtest'
    park_name          text not null,
    ride_key           text not null,
    ride_name          text,
    ts_local           timestamp not null,      -- park-local wall time
    ts_utc             timestamp not null,
    predicted_wait_min double precision not null,
    lower_bound        double precision,
    upper_bound        double precision,
    actual_wait_min    double precision,        -- backtest rows only
    error_min          double precision,        -- predicted - actual, backtest only
    model_name         text not null,
    run_id             text,
    run_status         text,
    generated_at       timestamptz,
    model_trained_at   text,
    -- model_name is part of the key: backtest rows exist for every candidate model,
    -- so (ride, timestamp) alone is not unique.
    primary key (row_kind, model_name, park_name, ride_key, ts_local)
);

create index predictions_park_time_idx on predictions (park_name, ts_local);
create index predictions_kind_model_idx on predictions (row_kind, model_name);

-- Staging twins: loaded in chunks over REST, then swapped into serving in one
-- transaction by publish_serving(). Never exposed to anon.
create table kpis_staging (like kpis including all);
create table predictions_staging (like predictions including all);

create or replace function stage_reset()
returns void
language plpgsql
security definer set search_path = public
as $$
begin
    truncate kpis_staging;
    truncate predictions_staging;
end $$;

create or replace function publish_serving()
returns jsonb
language plpgsql
security definer set search_path = public
as $$
declare
    n_kpis bigint;
    n_predictions bigint;
begin
    -- One transaction: readers see the old pair or the new pair, never a mix.
    -- TRUNCATE, not DELETE: DELETE walks every row through MVCC and hit the
    -- statement timeout at 255K rows on free-tier compute; TRUNCATE is instant on
    -- any table size, still rolls back with the transaction, and is not subject to
    -- the pg-safeupdate guard.
    truncate kpis;
    insert into kpis select * from kpis_staging;
    truncate predictions;
    insert into predictions select * from predictions_staging;
    select count(*) into n_kpis from kpis;
    select count(*) into n_predictions from predictions;
    if n_kpis = 0 or n_predictions = 0 then
        raise exception 'refusing to publish an empty serving table (kpis=%, predictions=%)',
            n_kpis, n_predictions;
    end if;
    return jsonb_build_object('kpis', n_kpis, 'predictions', n_predictions);
end $$;

-- The swap inserts ~255K rows in one statement; the platform's default statement
-- timeout for API roles is too short for that on free-tier compute. PostgREST
-- applies impersonated-role settings per request, and the reload notify makes it
-- pick this up without a restart.
alter role service_role set statement_timeout = '5min';
notify pgrst, 'reload config';

-- Read-only public access to the SERVING tables only.
grant usage on schema public to anon, authenticated;
grant select on kpis, predictions to anon, authenticated;
revoke all on kpis_staging, predictions_staging from anon, authenticated;
revoke execute on function stage_reset(), publish_serving() from anon, authenticated;

alter table kpis enable row level security;
alter table predictions enable row level security;
alter table kpis_staging enable row level security;
alter table predictions_staging enable row level security;

create policy "public read" on kpis for select to anon, authenticated using (true);
create policy "public read" on predictions for select to anon, authenticated using (true);

-- Verify:
--   curl "$SUPABASE_URL/rest/v1/kpis?select=model,kpi_name,kpi_value&limit=5" \
--        -H "apikey: $SUPABASE_ANON_KEY"
