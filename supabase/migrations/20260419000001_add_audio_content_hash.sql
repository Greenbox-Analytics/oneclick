-- Add content_hash column to audio_files for SHA-256 based deduplication.
-- A file with the same hash within a folder is treated as the same upload
-- and the existing audio_files row is reused.

ALTER TABLE public.audio_files
ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_audio_files_folder_content_hash
ON public.audio_files(folder_id, content_hash)
WHERE content_hash IS NOT NULL;
