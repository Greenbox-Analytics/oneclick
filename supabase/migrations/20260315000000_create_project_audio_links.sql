-- Junction table linking portfolio projects to audio files (many-to-many)
CREATE TABLE project_audio_links (
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE NOT NULL,
  audio_file_id UUID REFERENCES audio_files(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, audio_file_id)
);

-- Index for reverse lookups (find all projects an audio file is linked to)
CREATE INDEX idx_project_audio_links_audio_file_id ON project_audio_links(audio_file_id);

-- Enable RLS
ALTER TABLE project_audio_links ENABLE ROW LEVEL SECURITY;

-- RLS policies (access through project → artist ownership)
CREATE POLICY "Users can view their project audio links"
  ON project_audio_links FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM projects p
      JOIN artists a ON a.id = p.artist_id
      WHERE p.id = project_audio_links.project_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create project audio links"
  ON project_audio_links FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM projects p
      JOIN artists a ON a.id = p.artist_id
      WHERE p.id = project_audio_links.project_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete their project audio links"
  ON project_audio_links FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM projects p
      JOIN artists a ON a.id = p.artist_id
      WHERE p.id = project_audio_links.project_id
      AND a.user_id = auth.uid()
    )
  );
