-- 1. Projects table: members can read
CREATE POLICY "projects_select_via_member" ON projects
  FOR SELECT USING (
    id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );

-- 2. Invite visibility by email (case-insensitive, before claim)
DROP POLICY IF EXISTS "Collaborator can read own invitations" ON registry_collaborators;

CREATE POLICY "collaborators_select_by_email_or_id" ON registry_collaborators
  FOR SELECT USING (
    LOWER(email) = LOWER((SELECT email FROM auth.users WHERE id = auth.uid()))
    OR collaborator_user_id = auth.uid()
    OR invited_by = auth.uid()
  );

-- 3. File access via work collaboration (confirmed only)
CREATE POLICY "project_files_select_via_work_collab" ON project_files
  FOR SELECT USING (
    id IN (
      SELECT file_id FROM work_files WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );

-- 4. Audio access via work collaboration (confirmed only)
CREATE POLICY "audio_files_select_via_work_collab" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM work_audio_links WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );

-- 5. Project member access to works
CREATE POLICY "works_select_via_project_member" ON works_registry
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- 6. Project member access to files
CREATE POLICY "project_files_select_via_project_member" ON project_files
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- 7. Project member access to audio (via project_audio_links)
CREATE POLICY "audio_files_select_via_project_member" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM project_audio_links WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
  );

-- 8. Editors+ can create/update works
CREATE POLICY "works_insert_editors" ON works_registry
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

CREATE POLICY "works_update_editors" ON works_registry
  FOR UPDATE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

-- 9. Owner/admin can delete works
CREATE POLICY "works_delete_owner_admin" ON works_registry
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );
