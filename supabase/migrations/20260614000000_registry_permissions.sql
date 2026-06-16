-- 20260614000000_registry_permissions.sql
-- Work collaborator permissions: grants, access_level, stake linking, parse cache.
-- Additive only. See docs/superpowers/specs/2026-06-13-work-collaborator-permissions-design.md

-- 1. registry_collaborators: permission tier + per-person terms
alter table registry_collaborators
  add column if not exists access_level text not null default 'viewer'
    check (access_level in ('viewer','admin')),
  add column if not exists terms jsonb not null default '[]';

-- 2. registry_access_grants: the per-collaborator visibility source of truth
create table if not exists registry_access_grants (
  id              uuid primary key default gen_random_uuid(),
  work_id         uuid not null references works_registry(id) on delete cascade,
  collaborator_id uuid not null references registry_collaborators(id) on delete cascade,
  resource_type   text not null check (resource_type in
                    ('project_file','audio_file','license','agreement','ownership_breakdown')),
  resource_id     uuid,            -- NULL only for work-wide grants (ownership_breakdown)
  granted_by      uuid not null references auth.users(id),
  created_at      timestamptz not null default now()
);
-- Dedupe item grants:
create unique index if not exists registry_access_grants_item_uq
  on registry_access_grants (collaborator_id, resource_type, resource_id)
  where resource_id is not null;
-- Exactly one ownership_breakdown row per collaborator (NULL resource_id):
create unique index if not exists registry_access_grants_workwide_uq
  on registry_access_grants (collaborator_id, resource_type)
  where resource_id is null;
create index if not exists registry_access_grants_work_idx on registry_access_grants (work_id);
create index if not exists registry_access_grants_collab_idx on registry_access_grants (collaborator_id);

alter table registry_access_grants enable row level security;
-- Interim policy: deny all direct client access (backend uses service-role and bypasses
-- RLS). REPLACED by an owner/project-admin select policy in the RLS migration (Task 9).
create policy "grants_no_direct_client_access" on registry_access_grants
  for all using (false);

-- 3. ownership_stakes: explicit person linking
alter table ownership_stakes
  add column if not exists collaborator_id uuid references registry_collaborators(id) on delete set null,
  add column if not exists is_owner_stake  boolean not null default false;
-- NAND, not XOR: a stake must not be BOTH the owner's and collaborator-linked.
-- Neither set is allowed and intentional — that's a third-party stake.
alter table ownership_stakes
  add constraint stake_not_both_owner_and_collab
    check (not (is_owner_stake and collaborator_id is not null));
-- Conflict target for the derived-stake upsert (one stake per collaborator per type);
-- leaves owner/third-party (null collaborator_id) stakes unconstrained.
create unique index if not exists ownership_stakes_collab_type_uq
  on ownership_stakes (collaborator_id, stake_type)
  where collaborator_id is not null;

-- 4. contract_parse_cache: parse-once by file content hash. Backend (service-role) ONLY.
-- Enable RLS with NO policies: Supabase auto-exposes public tables via PostgREST and grants
-- anon/authenticated by default, so without RLS this (sensitive: parsed contract splits)
-- would be client-readable. RLS-on + no-policies denies all clients; service-role bypasses RLS.
create table if not exists contract_parse_cache (
  content_hash text primary key,
  parsed       jsonb not null,
  created_at   timestamptz not null default now()
);
alter table contract_parse_cache enable row level security;

-- 5. Backfill the new linking columns from existing data (safety net; expected no-op
--    on a DB with no collaborators yet — see deploy gate below).
--    a) collaborator_id from the legacy first-only registry_collaborators.stake_id link
update ownership_stakes os
set collaborator_id = rc.id
from registry_collaborators rc
where rc.stake_id = os.id and os.collaborator_id is null;
--    b) is_owner_stake: there is NO reliable signal for "the owner's stake" in legacy
--       rows (ownership_stakes.user_id = work owner for ALL rows; holder_email is often
--       a stage name/blank). We therefore do NOT auto-guess it. Going forward it is set
--       at the creation source (Task 5 backend + Task 20 Add Work wizard). The deploy
--       gate below keeps this safe.

-- 6. Deploy hard-gate: fail closed if confirmed work-only collaborators already exist
do $$
declare n int; n_stakes int; n_works int;
begin
  select count(*) into n
  from registry_collaborators rc
  where rc.status = 'confirmed'
    and not exists (
      select 1 from works_registry w
      join project_members pm on pm.project_id = w.project_id
      where w.id = rc.work_id and pm.user_id = rc.collaborator_user_id);
  if n > 0 then
    raise exception
      'Found % confirmed work-only collaborator(s); decide grandfathering before closed-by-default rollout', n;
  end if;
  -- Pre-existing works/stakes will NOT have is_owner_stake set (no reliable legacy signal;
  -- Task 20 only flags rows created/edited after rollout). If non-zero, owners must re-save
  -- their own split row OR a manual backfill is needed, else viewers won't see the owner slice.
  select count(*) into n_stakes from ownership_stakes;
  select count(*) into n_works  from works_registry;
  raise notice 'registry_permissions: % pre-existing works, % stakes (none flagged is_owner_stake). If >0, plan owner re-save/backfill.', n_works, n_stakes;
end $$;
