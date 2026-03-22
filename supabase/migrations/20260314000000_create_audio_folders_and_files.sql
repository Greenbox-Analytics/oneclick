-- Audio folders table for hierarchical folder structure (linked to artist, not project)
CREATE TABLE audio_folders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  parent_id UUID REFERENCES audio_folders(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audio files table for storing audio file metadata
CREATE TABLE audio_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  folder_id UUID NOT NULL REFERENCES audio_folders(id) ON DELETE CASCADE,
  file_name TEXT NOT NULL,
  file_url TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  file_type TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_audio_folders_artist_id ON audio_folders(artist_id);
CREATE INDEX idx_audio_folders_parent_id ON audio_folders(parent_id);
CREATE INDEX idx_audio_files_folder_id ON audio_files(folder_id);

-- Unique folder name within same parent (prevent duplicate folder names)
CREATE UNIQUE INDEX idx_audio_folders_unique_name ON audio_folders(artist_id, parent_id, name)
  WHERE parent_id IS NOT NULL;
CREATE UNIQUE INDEX idx_audio_folders_unique_name_root ON audio_folders(artist_id, name)
  WHERE parent_id IS NULL;

-- Unique file name within same folder
CREATE UNIQUE INDEX idx_audio_files_unique_name ON audio_files(folder_id, file_name);

-- Enable RLS
ALTER TABLE audio_folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_files ENABLE ROW LEVEL SECURITY;

-- RLS policies for audio_folders (access through artist ownership)
CREATE POLICY "Users can view audio folders for their artists"
  ON audio_folders FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM artists a
      WHERE a.id = audio_folders.artist_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create audio folders for their artists"
  ON audio_folders FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM artists a
      WHERE a.id = audio_folders.artist_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete audio folders for their artists"
  ON audio_folders FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM artists a
      WHERE a.id = audio_folders.artist_id
      AND a.user_id = auth.uid()
    )
  );

-- RLS policies for audio_files
CREATE POLICY "Users can view audio files for their artists"
  ON audio_files FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM audio_folders af
      JOIN artists a ON a.id = af.artist_id
      WHERE af.id = audio_files.folder_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create audio files for their artists"
  ON audio_files FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM audio_folders af
      JOIN artists a ON a.id = af.artist_id
      WHERE af.id = audio_files.folder_id
      AND a.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete audio files for their artists"
  ON audio_files FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM audio_folders af
      JOIN artists a ON a.id = af.artist_id
      WHERE af.id = audio_files.folder_id
      AND a.user_id = auth.uid()
    )
  );
