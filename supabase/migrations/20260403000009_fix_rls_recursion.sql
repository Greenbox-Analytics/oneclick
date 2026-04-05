-- Fix infinite recursion in project_members RLS policies.
-- The original policies referenced project_members in their own WHERE clause,
-- causing Postgres to enter an infinite loop when evaluating access.
-- Solution: SECURITY DEFINER helper functions that bypass RLS for membership checks.

-- 1. Helper function: check if current user is a member of a project
CREATE OR REPLACE FUNCTION public.is_project_member(p_id UUID)
RETURNS BOOLEAN
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM project_members
    WHERE project_id = p_id AND user_id = auth.uid()
  );
END;
$$;

-- 2. Helper function: get current user's role on a project
CREATE OR REPLACE FUNCTION public.get_project_role(p_id UUID)
RETURNS TEXT
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
DECLARE
  r TEXT;
BEGIN
  SELECT role INTO r FROM project_members
  WHERE project_id = p_id AND user_id = auth.uid();
  RETURN r;
END;
$$;

-- 3. Replace recursive project_members policies
DROP POLICY IF EXISTS "project_members_select_members" ON project_members;
DROP POLICY IF EXISTS "project_members_insert_admins" ON project_members;
DROP POLICY IF EXISTS "project_members_update_admins" ON project_members;
DROP POLICY IF EXISTS "project_members_delete_admins" ON project_members;

CREATE POLICY "project_members_select_members" ON project_members
  FOR SELECT USING (is_project_member(project_id));

CREATE POLICY "project_members_insert_admins" ON project_members
  FOR INSERT WITH CHECK (get_project_role(project_id) IN ('owner', 'admin'));

CREATE POLICY "project_members_update_admins" ON project_members
  FOR UPDATE USING (get_project_role(project_id) IN ('owner', 'admin'));

CREATE POLICY "project_members_delete_admins" ON project_members
  FOR DELETE USING (
    get_project_role(project_id) IN ('owner', 'admin')
    OR (user_id = auth.uid() AND role != 'owner')
  );

-- 4. Replace projects policy
DROP POLICY IF EXISTS "projects_select_via_member" ON projects;
CREATE POLICY "projects_select_via_member" ON projects
  FOR SELECT USING (is_project_member(id));

-- 5. Replace works_registry policies
DROP POLICY IF EXISTS "works_select_via_project_member" ON works_registry;
DROP POLICY IF EXISTS "works_insert_editors" ON works_registry;
DROP POLICY IF EXISTS "works_update_editors" ON works_registry;
DROP POLICY IF EXISTS "works_delete_owner_admin" ON works_registry;

CREATE POLICY "works_select_via_project_member" ON works_registry
  FOR SELECT USING (is_project_member(project_id));

CREATE POLICY "works_insert_editors" ON works_registry
  FOR INSERT WITH CHECK (get_project_role(project_id) IN ('owner', 'admin', 'editor'));

CREATE POLICY "works_update_editors" ON works_registry
  FOR UPDATE USING (get_project_role(project_id) IN ('owner', 'admin', 'editor'));

CREATE POLICY "works_delete_owner_admin" ON works_registry
  FOR DELETE USING (get_project_role(project_id) IN ('owner', 'admin'));

-- 6. Replace project_files policy
DROP POLICY IF EXISTS "project_files_select_via_project_member" ON project_files;
CREATE POLICY "project_files_select_via_project_member" ON project_files
  FOR SELECT USING (is_project_member(project_id));

-- 7. Replace audio_files policy
DROP POLICY IF EXISTS "audio_files_select_via_project_member" ON audio_files;
CREATE POLICY "audio_files_select_via_project_member" ON audio_files
  FOR SELECT USING (
    id IN (SELECT audio_file_id FROM project_audio_links WHERE is_project_member(project_id))
  );

-- 8. Replace work_files policies
DROP POLICY IF EXISTS "work_files_select" ON work_files;
DROP POLICY IF EXISTS "work_files_insert_editors" ON work_files;
DROP POLICY IF EXISTS "work_files_delete_editors" ON work_files;

CREATE POLICY "work_files_select" ON work_files
  FOR SELECT USING (
    work_id IN (SELECT id FROM works_registry WHERE is_project_member(project_id))
    OR work_id IN (SELECT work_id FROM registry_collaborators WHERE collaborator_user_id = auth.uid() AND status = 'confirmed')
  );

CREATE POLICY "work_files_insert_editors" ON work_files
  FOR INSERT WITH CHECK (
    work_id IN (SELECT id FROM works_registry WHERE get_project_role(project_id) IN ('owner', 'admin', 'editor'))
  );

CREATE POLICY "work_files_delete_editors" ON work_files
  FOR DELETE USING (
    work_id IN (SELECT id FROM works_registry WHERE get_project_role(project_id) IN ('owner', 'admin', 'editor'))
  );

-- 9. Replace work_audio_links policies
DROP POLICY IF EXISTS "work_audio_links_select" ON work_audio_links;
DROP POLICY IF EXISTS "work_audio_links_insert_editors" ON work_audio_links;
DROP POLICY IF EXISTS "work_audio_links_delete_editors" ON work_audio_links;

CREATE POLICY "work_audio_links_select" ON work_audio_links
  FOR SELECT USING (
    work_id IN (SELECT id FROM works_registry WHERE is_project_member(project_id))
    OR work_id IN (SELECT work_id FROM registry_collaborators WHERE collaborator_user_id = auth.uid() AND status = 'confirmed')
  );

CREATE POLICY "work_audio_links_insert_editors" ON work_audio_links
  FOR INSERT WITH CHECK (
    work_id IN (SELECT id FROM works_registry WHERE get_project_role(project_id) IN ('owner', 'admin', 'editor'))
  );

CREATE POLICY "work_audio_links_delete_editors" ON work_audio_links
  FOR DELETE USING (
    work_id IN (SELECT id FROM works_registry WHERE get_project_role(project_id) IN ('owner', 'admin', 'editor'))
  );

-- 10. Replace pending_project_invites policies
DROP POLICY IF EXISTS "pending_invites_select" ON pending_project_invites;
DROP POLICY IF EXISTS "pending_invites_insert" ON pending_project_invites;
DROP POLICY IF EXISTS "pending_invites_delete" ON pending_project_invites;

CREATE POLICY "pending_invites_select" ON pending_project_invites
  FOR SELECT USING (get_project_role(project_id) IN ('owner', 'admin'));

CREATE POLICY "pending_invites_insert" ON pending_project_invites
  FOR INSERT WITH CHECK (get_project_role(project_id) IN ('owner', 'admin'));

CREATE POLICY "pending_invites_delete" ON pending_project_invites
  FOR DELETE USING (get_project_role(project_id) IN ('owner', 'admin'));
