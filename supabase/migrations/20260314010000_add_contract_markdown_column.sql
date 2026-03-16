-- Add contract_markdown column to project_files table
-- Stores the full markdown text of contract PDFs for full-document context
ALTER TABLE project_files
ADD COLUMN IF NOT EXISTS contract_markdown TEXT DEFAULT NULL;

-- Add a comment for documentation
COMMENT ON COLUMN project_files.contract_markdown IS 'Full markdown text of contract PDF, used for full-document LLM context. Populated during upload or lazily on first access.';
