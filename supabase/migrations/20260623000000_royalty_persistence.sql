-- ============================================================
-- royalty_persistence (Phase 1)
--   Payees, per-collaborator earning lines, payout records,
--   payout coverage, and a shared FX rate cache.
--
--   Also adds the `currency` column to project_files for
--   statement-level currency tracking.
-- ============================================================

alter table public.project_files add column if not exists currency text;

-- ============================================================
-- 1) royalty_payees — workspace-scoped payee identity.
-- ============================================================

create table public.royalty_payees (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  normalized_name text not null,
  display_name text not null,
  payout_currency text not null default 'USD',
  registry_user_id uuid references auth.users(id) on delete set null,
  email text,
  email_source text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, normalized_name)
);

alter table public.royalty_payees enable row level security;

create policy "Users can view their own payees"
  on public.royalty_payees for select
  using (auth.uid() = user_id);

create policy "Users can insert their own payees"
  on public.royalty_payees for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own payees"
  on public.royalty_payees for update
  using (auth.uid() = user_id);

create policy "Users can delete their own payees"
  on public.royalty_payees for delete
  using (auth.uid() = user_id);

-- ============================================================
-- 2) royalty_lines — per-collaborator earning line.
-- ============================================================

create table public.royalty_lines (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  calculation_id uuid not null references public.royalty_calculations(id) on delete cascade,
  royalty_statement_id uuid not null references public.project_files(id) on delete cascade,
  payee_id uuid not null references public.royalty_payees(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  work_id uuid references public.works_registry(id) on delete set null,
  song_title text not null,
  role text,
  royalty_type text,
  percentage numeric,
  song_revenue numeric,
  amount_owed numeric not null,
  statement_currency text not null default 'USD',
  period_start date not null,
  period_end date not null,
  created_at timestamptz not null default now()
);

alter table public.royalty_lines enable row level security;

create policy "Users can view their own royalty lines"
  on public.royalty_lines for select
  using (auth.uid() = user_id);

create policy "Users can insert their own royalty lines"
  on public.royalty_lines for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own royalty lines"
  on public.royalty_lines for update
  using (auth.uid() = user_id);

create policy "Users can delete their own royalty lines"
  on public.royalty_lines for delete
  using (auth.uid() = user_id);

-- ============================================================
-- 3) royalty_payouts — invoice / payout record (one per payee).
-- ============================================================

create table public.royalty_payouts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  payee_id uuid not null references public.royalty_payees(id) on delete cascade,
  status text not null default 'draft' check (status in ('draft','paid')),
  pay_currency text not null,
  fx_rate_date date not null,
  total_amount numeric not null,
  breakdown_snapshot jsonb not null,
  idempotency_key text,
  note text,
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  unique (user_id, idempotency_key)
);

alter table public.royalty_payouts enable row level security;

create policy "Users can view their own payouts"
  on public.royalty_payouts for select
  using (auth.uid() = user_id);

create policy "Users can insert their own payouts"
  on public.royalty_payouts for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own payouts"
  on public.royalty_payouts for update
  using (auth.uid() = user_id);

create policy "Users can delete their own payouts"
  on public.royalty_payouts for delete
  using (auth.uid() = user_id);

-- ============================================================
-- 4) royalty_payout_coverage — which statements a payout settles.
-- ============================================================

create table public.royalty_payout_coverage (
  payout_id uuid not null references public.royalty_payouts(id) on delete cascade,
  payee_id uuid not null references public.royalty_payees(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  royalty_statement_id uuid not null references public.project_files(id) on delete cascade,
  covered_amount numeric not null,
  primary key (payout_id, royalty_statement_id)
);

alter table public.royalty_payout_coverage enable row level security;

create policy "Users can view their own payout coverage"
  on public.royalty_payout_coverage for select
  using (
    exists (
      select 1 from public.royalty_payouts p
      where p.id = payout_id
        and p.user_id = auth.uid()
    )
  );

create policy "Users can insert their own payout coverage"
  on public.royalty_payout_coverage for insert
  with check (
    exists (
      select 1 from public.royalty_payouts p
      where p.id = payout_id
        and p.user_id = auth.uid()
    )
  );

create policy "Users can delete their own payout coverage"
  on public.royalty_payout_coverage for delete
  using (
    exists (
      select 1 from public.royalty_payouts p
      where p.id = payout_id
        and p.user_id = auth.uid()
    )
  );

-- ============================================================
-- 5) fx_rate_snapshots — shared FX rate cache (service-role managed).
-- ============================================================

create table public.fx_rate_snapshots (
  base text not null,
  rate_date date not null,
  rates jsonb not null,
  fetched_at timestamptz not null default now(),
  primary key (base, rate_date)
);

alter table public.fx_rate_snapshots enable row level security;

create policy "Anyone can read FX rate snapshots"
  on public.fx_rate_snapshots for select
  using (true);

-- ============================================================
-- Indexes
-- ============================================================

create index on public.royalty_lines (user_id, payee_id);
create index on public.royalty_lines (project_id);
create index on public.royalty_lines (royalty_statement_id, project_id);
create index on public.royalty_lines (payee_id, royalty_statement_id);
create index on public.royalty_payouts (user_id, payee_id);
create index on public.royalty_payout_coverage (payee_id, royalty_statement_id);
