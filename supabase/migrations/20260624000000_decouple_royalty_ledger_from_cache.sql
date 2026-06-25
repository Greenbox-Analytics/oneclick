-- ============================================================
-- Decouple the royalty ledger from the OneClick calc cache.
--
-- The deploy "Clear OneClick cache" step (clear_oneclick_cache.py) deletes EVERY
-- row in royalty_calculations whenever OneClick code changes. royalty_lines and
-- royalty_statement_rows referenced royalty_calculations with ON DELETE CASCADE,
-- so that cache wipe also destroyed the derived royalty ledger (per-collaborator
-- earnings + statement detail) — leaving payees/invoices intact but with $0
-- earned/owed and orphaned invoices.
--
-- Fix: switch those FKs to ON DELETE SET NULL. A cache wipe now deletes only the
-- calc results; the ledger persists (calculation_id just becomes NULL). Re-runs
-- still replace lines by (royalty_statement_id, project_id), so this is safe.
--
-- Idempotent: the existing FK constraint name is looked up dynamically (so it
-- works regardless of how it was auto-named), and the royalty_statement_rows part
-- is guarded since that table may not have been applied yet on every DB.
-- ============================================================

-- ---- royalty_lines.calculation_id -> ON DELETE SET NULL ----
alter table public.royalty_lines alter column calculation_id drop not null;

do $$
declare cname text;
begin
  select tc.constraint_name into cname
  from information_schema.table_constraints tc
  join information_schema.key_column_usage kcu
    on tc.constraint_name = kcu.constraint_name
   and tc.table_schema = kcu.table_schema
  where tc.table_schema = 'public'
    and tc.table_name = 'royalty_lines'
    and tc.constraint_type = 'FOREIGN KEY'
    and kcu.column_name = 'calculation_id'
  limit 1;
  if cname is not null then
    execute format('alter table public.royalty_lines drop constraint %I', cname);
  end if;
end $$;

alter table public.royalty_lines
  add constraint royalty_lines_calculation_id_fkey
  foreign key (calculation_id) references public.royalty_calculations(id) on delete set null;

-- ---- royalty_statement_rows.calculation_id -> ON DELETE SET NULL (guarded) ----
do $$
declare cname text;
begin
  if exists (
    select 1 from information_schema.tables
    where table_schema = 'public' and table_name = 'royalty_statement_rows'
  ) then
    alter table public.royalty_statement_rows alter column calculation_id drop not null;

    select tc.constraint_name into cname
    from information_schema.table_constraints tc
    join information_schema.key_column_usage kcu
      on tc.constraint_name = kcu.constraint_name
     and tc.table_schema = kcu.table_schema
    where tc.table_schema = 'public'
      and tc.table_name = 'royalty_statement_rows'
      and tc.constraint_type = 'FOREIGN KEY'
      and kcu.column_name = 'calculation_id'
    limit 1;
    if cname is not null then
      execute format('alter table public.royalty_statement_rows drop constraint %I', cname);
    end if;

    alter table public.royalty_statement_rows
      add constraint royalty_statement_rows_calculation_id_fkey
      foreign key (calculation_id) references public.royalty_calculations(id) on delete set null;
  end if;
end $$;
