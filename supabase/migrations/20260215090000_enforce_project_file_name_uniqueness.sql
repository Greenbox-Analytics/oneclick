-- Enforce unique file names within the same project across all categories
-- Uses normalized name (trim + lowercase) so case/whitespace variants are treated as duplicates
CREATE UNIQUE INDEX IF NOT EXISTS project_files_project_id_normalized_file_name_unique
ON public.project_files (project_id, lower(btrim(file_name)));