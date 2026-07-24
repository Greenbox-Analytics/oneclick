-- Royalty ledger reconciliation: provenance, stable identity, history, tombstones.
-- Spec: docs/superpowers/specs/2026-07-23-oneclick-ledger-reconciliation-design.md

-- ---- 1. royalty_lines: new columns -----------------------------------------
alter table public.royalty_lines
  add column if not exists source_contracts jsonb not null default '[]',
  add column if not exists song_key text,
  add column if not exists royalty_type_key text,
  add column if not exists statement_content_hash text,
  add column if not exists statement_file_name text,
  add column if not exists payee_locked boolean not null default false,
  add column if not exists locked_party_key text;

-- ---- 2. Backfill ------------------------------------------------------------
-- 2a. song_key from song_title (lower + collapse whitespace)
update public.royalty_lines
set song_key = lower(regexp_replace(trim(song_title), '\s+', ' ', 'g'))
where song_key is null;

-- 2b. royalty_type_key: every existing line passed the streaming filter.
update public.royalty_lines
set royalty_type_key = 'streaming'
where royalty_type_key is null;

-- 2c. source_contracts from the calc junction (name/hash only for live files)
update public.royalty_lines rl
set source_contracts = sub.sources
from (
  select rcc.calculation_id,
         jsonb_agg(distinct jsonb_strip_nulls(jsonb_build_object(
           'id', rcc.contract_id,
           'name', pf.file_name,
           'hash', pf.content_hash))) as sources
  from public.royalty_calculation_contracts rcc
  left join public.project_files pf on pf.id = rcc.contract_id
  group by rcc.calculation_id
) sub
where rl.calculation_id = sub.calculation_id
  and rl.source_contracts = '[]'::jsonb;

-- 2d. statement hash/name for still-live statement files
update public.royalty_lines rl
set statement_content_hash = pf.content_hash,
    statement_file_name = pf.file_name
from public.project_files pf
where pf.id = rl.royalty_statement_id
  and rl.statement_content_hash is null;

alter table public.royalty_lines
  alter column song_key set not null,
  alter column royalty_type_key set not null;

-- ---- 3. History table (needed by the dedupe step below) ---------------------
create table if not exists public.royalty_ledger_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  action text not null check (action in
    ('updated','deleted','source_removed','adopted','superseded','coverage_moved',
     'migration_merged','manual_purge')),
  old_row jsonb not null,
  cause text not null,
  occurred_at timestamptz not null default now()
);
alter table public.royalty_ledger_history enable row level security;
create policy "history_select_own" on public.royalty_ledger_history
  for select using (auth.uid() = user_id);
-- No insert/update/delete policies: writes go through the service role only.

-- ---- 4. Dedupe duplicate identities, then build the unique index ------------
-- Pre-existing data can hold >1 row per identity (the old merge kept 30%+35%
-- as two rows). Merge: keep the earliest row, SUM amount_owed/percentage,
-- union sources. SEMANTICS NOTE (deliberate): this turns an old double-count
-- into one combined line (30%+35% -> 65%). It preserves the totals users have
-- already been shown; the next recalculation of that statement re-detects the
-- disagreement and gates it for explicit resolution. Sequential statements
-- with temp tables — run-once SQL should be boring.
create temp table _ranked as
select id, user_id, royalty_statement_id, project_id, payee_id, song_key,
       royalty_type_key, amount_owed, percentage, source_contracts,
       row_number() over (
         partition by royalty_statement_id, project_id, payee_id, song_key, royalty_type_key
         order by created_at, id) as rn
from public.royalty_lines;

create temp table _absorbed as
select r.*, k.id as keeper_id
from _ranked r
join _ranked k
  on k.rn = 1
 and k.royalty_statement_id = r.royalty_statement_id
 and k.project_id = r.project_id and k.payee_id = r.payee_id
 and k.song_key = r.song_key and k.royalty_type_key = r.royalty_type_key
where r.rn > 1;

insert into public.royalty_ledger_history (user_id, action, old_row, cause)
select a.user_id, 'migration_merged', to_jsonb(rl.*), 'migration_20260723'
from _absorbed a join public.royalty_lines rl on rl.id = a.id;

update public.royalty_lines k
set amount_owed = k.amount_owed + s.extra_amount,
    percentage  = case when k.percentage is null or s.pct_has_null
                       then null else k.percentage + s.extra_pct end
from (
  select keeper_id,
         sum(amount_owed) as extra_amount,
         sum(coalesce(percentage, 0)) as extra_pct,
         bool_or(percentage is null) as pct_has_null
  from _absorbed group by keeper_id
) s
where k.id = s.keeper_id;

-- Union each keeper's sources with all its absorbed rows' sources.
-- (Built as a grouped derived table — a FROM-subquery may not reference the
-- UPDATE target without LATERAL, which is why the obvious correlated form fails.)
update public.royalty_lines k
set source_contracts = m.merged
from (
  select u.keeper_id, coalesce(jsonb_agg(distinct u.e), '[]'::jsonb) as merged
  from (
    select a.keeper_id, jsonb_array_elements(a.source_contracts) as e
    from _absorbed a
    union
    select ak.keeper_id, jsonb_array_elements(rl.source_contracts) as e
    from (select distinct keeper_id from _absorbed) ak
    join public.royalty_lines rl on rl.id = ak.keeper_id
  ) u
  group by u.keeper_id
) m
where k.id = m.keeper_id;

delete from public.royalty_lines where id in (select id from _absorbed);
drop table _ranked;
drop table _absorbed;

create unique index if not exists royalty_lines_identity_ux
  on public.royalty_lines
  (royalty_statement_id, project_id, payee_id, song_key, royalty_type_key);
create index if not exists royalty_lines_sources_gin
  on public.royalty_lines using gin (source_contracts);

-- ---- 5. Supersession tombstones + not-related dismissals ---------------------
create table if not exists public.royalty_statement_supersessions (
  user_id uuid not null,
  old_statement_id uuid not null,
  new_statement_id uuid not null,
  kind text not null default 'superseded' check (kind in ('superseded','not_related')),
  created_at timestamptz not null default now(),
  primary key (user_id, old_statement_id, new_statement_id)
);
-- A statement is superseded at most once; may be 'not_related' to many.
create unique index if not exists royalty_supersessions_one_supersede_ux
  on public.royalty_statement_supersessions (user_id, old_statement_id)
  where kind = 'superseded';
alter table public.royalty_statement_supersessions enable row level security;
create policy "supersessions_select_own" on public.royalty_statement_supersessions
  for select using (auth.uid() = user_id);

-- ---- 6. Coverage: stable row handle + moved_from marker ----------------------
-- The composite PK (payout_id, royalty_statement_id, project_id) is the only
-- key today; re-allocation/re-pointing must mutate individual rows, so add id.
alter table public.royalty_payout_coverage
  add column if not exists id uuid not null default gen_random_uuid(),
  add column if not exists moved_from jsonb;
create unique index if not exists royalty_payout_coverage_id_ux
  on public.royalty_payout_coverage (id);

-- ---- 7. Drop the destructive CASCADEs ---------------------------------------
-- The statement id becomes an opaque bucket key that outlives the file.
do $$
declare c record;
begin
  for c in
    select con.conname, rel.relname
    from pg_constraint con
    join pg_class rel on rel.oid = con.conrelid
    join pg_attribute att on att.attrelid = rel.oid and att.attnum = any(con.conkey)
    where rel.relname in ('royalty_lines', 'royalty_payout_coverage')
      and con.contype = 'f'
      and att.attname = 'royalty_statement_id'
  loop
    execute format('alter table public.%I drop constraint %I', c.relname, c.conname);
  end loop;
end $$;
