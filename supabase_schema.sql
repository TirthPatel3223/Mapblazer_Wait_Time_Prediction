-- Supabase serving schema. Run in the Supabase SQL editor.
--
-- Safe to re-run: it drops and recreates the serving tables (they hold no unique
-- state -- every publish fully replaces them), so after re-running this file, run
-- `python publish.py` once to refill them.
--
-- How publishing stays atomic on free-tier compute:
--   1. publish.py chunk-loads the *_staging tables (idempotent inserts, so a
--      connection-level retry of a chunk cannot create duplicates).
--   2. publish_serving() swaps serving and staging BY TABLE RENAME inside one
--      function call -- one Postgres transaction, catalog-only updates, instant at
--      any table size. Readers see the old pair or the new pair, never a mix.
--      (Earlier versions moved every row through DELETE/INSERT and hit the
--      free-tier statement timeout at 255K rows.)
-- Because the names trade places, serving and staging objects carry identical
-- structure, grants and policies.
--
-- Notes for this project:
--   * Projects created after 2026-05-30 do not auto-grant PostgREST access; without
--     the explicit grants below every REST request 404s.
--   * The pg-safeupdate guard is active; nothing here issues a bare DELETE.

drop table if exists kpis_staging cascade;
drop table if exists predictions_staging cascade;
drop table if exists kpis cascade;
drop table if exists predictions cascade;
drop table if exists kpis_retiring cascade;
drop table if exists predictions_retiring cascade;

create table kpis (
    model            text not null,
    kpi_name         text not null,
    kpi_value        double precision,
    is_serving      boolean not null default false,
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
    -- model_name is part of the key: backtest rows exist for every candidate model
    -- and the baseline, so (ride, timestamp) alone is not unique.
    primary key (row_kind, model_name, park_name, ride_key, ts_local)
);

create index predictions_park_time_idx on predictions (park_name, ts_local);
create index predictions_kind_model_idx on predictions (row_kind, model_name);

-- Staging twins (LIKE copies structure and indexes; grants and policies are added
-- explicitly below because LIKE does not copy them).
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
    select count(*) into n_kpis from kpis_staging;
    select count(*) into n_predictions from predictions_staging;
    if n_kpis = 0 or n_predictions = 0 then
        raise exception 'refusing to publish: staging is empty (kpis=%, predictions=%)',
            n_kpis, n_predictions;
    end if;

    -- The atomic swap: rename staging into serving. Catalog updates only.
    alter table kpis rename to kpis_retiring;
    alter table kpis_staging rename to kpis;
    alter table kpis_retiring rename to kpis_staging;
    alter table predictions rename to predictions_retiring;
    alter table predictions_staging rename to predictions;
    alter table predictions_retiring rename to predictions_staging;

    -- Discard the retired serving rows so the store stays small.
    truncate kpis_staging;
    truncate predictions_staging;

    -- PostgREST caches table metadata by name; tell it the names moved.
    notify pgrst, 'reload schema';

    return jsonb_build_object('kpis', n_kpis, 'predictions', n_predictions);
end $$;

-- Longer statement timeout for API sessions as headroom on small compute.
alter role service_role set statement_timeout = '5min';
notify pgrst, 'reload config';

-- Read-only public access. Serving and staging get IDENTICAL grants and policies so
-- the rename swap never changes what the anon role can see (staging only ever holds
-- the same public data mid-load).
grant usage on schema public to anon, authenticated;
grant select on kpis, predictions, kpis_staging, predictions_staging to anon, authenticated;
revoke execute on function stage_reset(), publish_serving() from anon, authenticated;

alter table kpis enable row level security;
alter table predictions enable row level security;
alter table kpis_staging enable row level security;
alter table predictions_staging enable row level security;

create policy "public read" on kpis for select to anon, authenticated using (true);
create policy "public read" on predictions for select to anon, authenticated using (true);
create policy "public read" on kpis_staging for select to anon, authenticated using (true);
create policy "public read" on predictions_staging for select to anon, authenticated using (true);

-- Verify:
--   curl "$SUPABASE_URL/rest/v1/kpis?select=model,kpi_name,kpi_value&limit=5" \
--        -H "apikey: $SUPABASE_ANON_KEY"
