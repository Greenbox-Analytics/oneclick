CREATE TABLE work_audio_links (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  audio_file_id UUID NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, audio_file_id)
);

ALTER TABLE work_audio_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "work_audio_links_select" ON work_audio_links
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

CREATE POLICY "work_audio_links_insert_editors" ON work_audio_links
  FOR INSERT WITH CHECK (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE POLICY "work_audio_links_delete_editors" ON work_audio_links
  FOR DELETE USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE INDEX idx_work_audio_links_work_id ON work_audio_links(work_id);
CREATE INDEX idx_work_audio_links_audio_file_id ON work_audio_links(audio_file_id);
