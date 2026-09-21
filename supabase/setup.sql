-- Paste this whole file into Supabase > SQL Editor > New query > Run.

-- 1. Private storage bucket for the price history (parquet files).
insert into storage.buckets (id, name, public)
values ('market-data', 'market-data', false)
on conflict (id) do nothing;

-- 2. Log of every pipeline run (probe, backfill, nightly update).
create table if not exists public.pipeline_runs (
  id       bigint generated always as identity primary key,
  run_at   timestamptz not null default now(),
  job      text not null,
  status   text not null,
  details  jsonb
);

-- Row level security ON with no policies = only the service key can read/write.
alter table public.pipeline_runs enable row level security;
