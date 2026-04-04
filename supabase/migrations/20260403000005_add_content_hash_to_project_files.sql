ALTER TABLE project_files ADD COLUMN content_hash TEXT;
CREATE INDEX idx_project_files_content_hash ON project_files(content_hash) WHERE content_hash IS NOT NULL;
