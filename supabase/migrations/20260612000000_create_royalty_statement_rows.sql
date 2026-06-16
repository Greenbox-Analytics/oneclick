-- ============================================================
-- royalty_statement_rows
--   Per-line-item rows from an uploaded royalty statement.
--   Powers the OneClick Earnings Breakdown tab (by country / month /
--   delivery format / vendor) and supplies the period range used for
--   payout deduplication once the contacts/payout ledger lands.
--
--   Statement is aggregated to song-level totals elsewhere; this table
--   preserves the row-level dimensions the aggregator currently discards.
-- ============================================================

create table public.royalty_statement_rows (
  id uuid primary key default gen_random_uuid(),
  calculation_id uuid not null references public.royalty_calculations(id) on delete cascade,
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

create index idx_royalty_statement_rows_calculation_id
  on public.royalty_statement_rows(calculation_id);
create index idx_royalty_statement_rows_calc_song
  on public.royalty_statement_rows(calculation_id, song_title);
create index idx_royalty_statement_rows_sale_date
  on public.royalty_statement_rows(calculation_id, sale_date)
  where sale_date is not null;

-- ============================================================
-- RLS — owner-only, mirroring royalty_calculations.
-- ============================================================

alter table public.royalty_statement_rows enable row level security;

create policy "Users can view their own statement rows"
  on public.royalty_statement_rows for select
  using (auth.uid() = user_id);

create policy "Users can insert their own statement rows"
  on public.royalty_statement_rows for insert
  with check (auth.uid() = user_id);

create policy "Users can delete their own statement rows"
  on public.royalty_statement_rows for delete
  using (auth.uid() = user_id);
