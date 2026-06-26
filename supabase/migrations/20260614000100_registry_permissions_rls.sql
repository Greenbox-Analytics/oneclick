-- 20260614000100_registry_permissions_rls.sql
-- RLS mirror for the work-collaborator permission model (defense-in-depth; the service-role
-- backend is authoritative). Helpers are SECURITY DEFINER querying OTHER tables to avoid the
-- recursion class fixed in 20260403000009_fix_rls_recursion.sql.

-- ============ Helpers ============
create or replace function public.work_project_id(p_work uuid)
returns uuid security definer set search_path = public, auth, extensions
language sql stable as $$
  select project_id from public.works_registry where id = p_work;
$$;

create or replace function public.work_role(p_work uuid)
returns text security definer set search_path = public, auth, extensions
language plpgsql stable as $$
declare owner_id uuid; lvl text;
begin
  select user_id into owner_id from public.works_registry where id = p_work;
  if owner_id = auth.uid() then return 'owner'; end if;
  select access_level into lvl from public.registry_collaborators
    where work_id = p_work and collaborator_user_id = auth.uid() and status = 'confirmed' limit 1;
  if lvl = 'admin' then return 'admin'; end if;
  if lvl is not null then return 'viewer'; end if;
  return 'none';
end $$;

create or replace function public.can_view_grant(p_work uuid, p_type text, p_rid uuid)
returns boolean security definer set search_path = public, auth, extensions
language plpgsql stable as $$
begin
  return exists (
    select 1 from public.registry_access_grants g
    join public.registry_collaborators c on c.id = g.collaborator_id
    where c.work_id = p_work and c.collaborator_user_id = auth.uid() and c.status = 'confirmed'
      and g.resource_type = p_type
      and (g.resource_id = p_rid or (g.resource_id is null and p_rid is null)));
end $$;

create or replace function public.can_view_project_file_grant(p_file uuid)
returns boolean security definer set search_path = public, auth, extensions
language sql stable as $$
  select exists (
    select 1 from public.work_files wf
    join public.registry_collaborators c on c.work_id = wf.work_id
    join public.registry_access_grants g on g.collaborator_id = c.id
    where wf.file_id = p_file
      and c.collaborator_user_id = auth.uid() and c.status = 'confirmed'
      and g.resource_type = 'project_file' and g.resource_id = p_file);
$$;

create or replace function public.can_view_audio_file_grant(p_audio uuid)
returns boolean security definer set search_path = public, auth, extensions
language sql stable as $$
  select exists (
    select 1 from public.work_audio_links wal
    join public.registry_collaborators c on c.work_id = wal.work_id
    join public.registry_access_grants g on g.collaborator_id = c.id
    where wal.audio_file_id = p_audio
      and c.collaborator_user_id = auth.uid() and c.status = 'confirmed'
      and g.resource_type = 'audio_file' and g.resource_id = p_audio);
$$;

-- ============ registry_access_grants: replace interim deny-all ============
drop policy if exists "grants_no_direct_client_access" on registry_access_grants;
create policy "grants_select" on registry_access_grants for select using (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin')
);

-- ============ ownership_stakes ============
drop policy if exists "Owner or collaborator can read stakes" on ownership_stakes;
create policy "stakes_select" on ownership_stakes for select using (
  is_project_member(public.work_project_id(work_id))
  or public.work_role(work_id) in ('owner','admin')
  or is_owner_stake
  or collaborator_id in (
       select id from registry_collaborators
       where collaborator_user_id = auth.uid() and status = 'confirmed')
  or public.can_view_grant(work_id, 'ownership_breakdown', null)
);
drop policy if exists "Owner can insert stakes" on ownership_stakes;
drop policy if exists "Owner can update stakes" on ownership_stakes;
drop policy if exists "Owner can delete stakes" on ownership_stakes;
create policy "stakes_insert" on ownership_stakes for insert with check (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));
create policy "stakes_update" on ownership_stakes for update using (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));
create policy "stakes_delete" on ownership_stakes for delete using (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));

-- ============ licensing_rights ============
drop policy if exists "Owner or collaborator can read licenses" on licensing_rights;
create policy "licenses_select" on licensing_rights for select using (
  is_project_member(public.work_project_id(work_id))
  or public.work_role(work_id) in ('owner','admin')
  or public.can_view_grant(work_id, 'license', id)
);
drop policy if exists "Owner can insert licenses" on licensing_rights;
drop policy if exists "Owner can update licenses" on licensing_rights;
drop policy if exists "Owner can delete licenses" on licensing_rights;
create policy "licenses_insert" on licensing_rights for insert with check (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));
create policy "licenses_update" on licensing_rights for update using (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));
create policy "licenses_delete" on licensing_rights for delete using (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));

-- ============ registry_agreements (immutable: insert only) ============
drop policy if exists "Owner or collaborator can read agreements" on registry_agreements;
create policy "agreements_select" on registry_agreements for select using (
  is_project_member(public.work_project_id(work_id))
  or public.work_role(work_id) in ('owner','admin')
  or public.can_view_grant(work_id, 'agreement', id)
);
drop policy if exists "Owner can insert agreements" on registry_agreements;
create policy "agreements_insert" on registry_agreements for insert with check (
  public.work_role(work_id) in ('owner','admin')
  or get_project_role(public.work_project_id(work_id)) in ('owner','admin'));

-- ============ project_files / audio_files: tighten ONLY the work-collab read path ============
drop policy if exists "project_files_select_via_work_collab" on project_files;
create policy "project_files_select_via_work_collab" on project_files for select using (
  public.can_view_project_file_grant(id)
);
drop policy if exists "audio_files_select_via_work_collab" on audio_files;
create policy "audio_files_select_via_work_collab" on audio_files for select using (
  public.can_view_audio_file_grant(id)
);

-- ============ works_registry writes -> elevated (reads + insert unchanged) ============
drop policy if exists "Owner can update works" on works_registry;
drop policy if exists "works_update_editors" on works_registry;
drop policy if exists "Owner can delete works" on works_registry;
drop policy if exists "works_delete_owner_admin" on works_registry;
create policy "works_update_elevated" on works_registry for update using (
  public.work_role(id) in ('owner','admin')
  or get_project_role(project_id) in ('owner','admin'));
create policy "works_delete_elevated" on works_registry for delete using (
  public.work_role(id) in ('owner','admin')
  or get_project_role(project_id) in ('owner','admin'));
