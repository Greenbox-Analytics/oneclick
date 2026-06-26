-- ============================================================
-- royalty_statement_rows — reconcile existing table
--   Per-line-item rows from an uploaded royalty statement.
--   Powers the OneClick Earnings Breakdown tab (by time period /
--   vendor / country / source).
--
--   The table already exists in the database with the full column set
--   and indexes. This migration only reconciles the two differences the
--   Earnings Breakdown feature needs, and is safe to re-run:
--     1. calculation_id FK: on delete SET NULL -> on delete CASCADE, so
--        deleting/superseding a calculation removes its statement rows
--        instead of orphaning them (calculation_id -> NULL).
--     2. Owner-only RLS + policies (mirroring royalty_calculations).
--
--   For fresh environments without the table, create it first.
-- ============================================================

create table if not exists public.royalty_statement_rows (
  id uuid primary key default gen_random_uuid(),
  calculation_id uuid references public.royalty_calculations(id) on delete cascade,
  user_id uuid not null,
  song_title text not null,
  vendor text,
  country text,
  country_code text,
  delivery_type text,
  delivery_format text,
  sale_date date,
  units_sold numeric,
  net_units numeric,
  sales numeric,
  net_income numeric,
  distribution numeric,
  net_payable numeric not null,
  isrc text,
  upc text,
  currency text not null default 'USD',
  created_at timestamptz not null default now()
);

create index if not exists idx_royalty_statement_rows_calculation_id
  on public.royalty_statement_rows(calculation_id);
create index if not exists idx_royalty_statement_rows_calc_song
  on public.royalty_statement_rows(calculation_id, song_title);
create index if not exists idx_royalty_statement_rows_sale_date
  on public.royalty_statement_rows(calculation_id, sale_date)
  where sale_date is not null;

-- ------------------------------------------------------------
-- 1. calculation_id FK: ON DELETE SET NULL -> ON DELETE CASCADE.
--    Drop the existing constraint (by its known name, if present) and
--    re-add it with cascade semantics.
-- ------------------------------------------------------------
alter table public.royalty_statement_rows
  drop constraint if exists royalty_statement_rows_calculation_id_fkey;

alter table public.royalty_statement_rows
  add constraint royalty_statement_rows_calculation_id_fkey
  foreign key (calculation_id) references public.royalty_calculations(id) on delete cascade;

-- ------------------------------------------------------------
-- 2. RLS — owner-only, mirroring royalty_calculations.
-- ------------------------------------------------------------
alter table public.royalty_statement_rows enable row level security;

drop policy if exists "Users can view their own statement rows" on public.royalty_statement_rows;
create policy "Users can view their own statement rows"
  on public.royalty_statement_rows for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own statement rows" on public.royalty_statement_rows;
create policy "Users can insert their own statement rows"
  on public.royalty_statement_rows for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can delete their own statement rows" on public.royalty_statement_rows;
create policy "Users can delete their own statement rows"
  on public.royalty_statement_rows for delete
  using (auth.uid() = user_id);
