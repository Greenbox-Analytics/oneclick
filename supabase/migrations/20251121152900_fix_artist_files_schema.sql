-- Fix project_files table schema issues

-- 1. Drop existing CHECK constraint
ALTER TABLE public.project_files
DROP CONSTRAINT IF EXISTS project_files_folder_category_check;

-- 2. Add file_path column for storage reference
ALTER TABLE public.project_files
ADD COLUMN IF NOT EXISTS file_path TEXT;

-- 3. Update folder_category constraint to match frontend code
ALTER TABLE public.project_files
ADD CONSTRAINT project_files_folder_category_check 
CHECK (folder_category IN ('contract', 'split_sheet', 'royalty_statement', 'other'));

-- 4. Create storage bucket for project files
INSERT INTO storage.buckets (id, name, public)
VALUES ('project-files', 'project-files', true)
ON CONFLICT (id) DO NOTHING;

-- 5. Set up storage policies for project-files bucket
CREATE POLICY "Anyone can upload project files"
ON storage.objects FOR INSERT
TO public
WITH CHECK (bucket_id = 'project-files');

CREATE POLICY "Anyone can view project files"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'project-files');

CREATE POLICY "Anyone can update project files"
ON storage.objects FOR UPDATE
TO public
USING (bucket_id = 'project-files');

CREATE POLICY "Anyone can delete project files"
ON storage.objects FOR DELETE
TO public
USING (bucket_id = 'project-files');

-- 6. Add index for faster file lookups
CREATE INDEX IF NOT EXISTS idx_project_files_project_id 
ON public.project_files(project_id);

CREATE INDEX IF NOT EXISTS idx_project_files_folder_category 
ON public.project_files(folder_category);

-- 7. Add comment for documentation
COMMENT ON COLUMN public.project_files.folder_category IS 
'File category: contract, split_sheet, royalty_statement, or other';

COMMENT ON COLUMN public.project_files.file_path IS 
'Storage path for the file in Supabase Storage';

COMMENT ON COLUMN public.project_files.file_url IS 
'Public URL for accessing the file';
