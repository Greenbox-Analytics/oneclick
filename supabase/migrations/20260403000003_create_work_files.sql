CREATE TABLE work_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  file_id UUID NOT NULL REFERENCES project_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, file_id)
);

ALTER TABLE work_files ENABLE ROW LEVEL SECURITY;

CREATE POLICY "work_files_select" ON work_files
  FOR SELECT USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
    OR
    work_id IN (
      SELECT work_id FROM registry_collaborators
      WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
    )
  );

CREATE POLICY "work_files_insert_editors" ON work_files
  FOR INSERT WITH CHECK (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE POLICY "work_files_delete_editors" ON work_files
  FOR DELETE USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE INDEX idx_work_files_work_id ON work_files(work_id);
CREATE INDEX idx_work_files_file_id ON work_files(file_id);
