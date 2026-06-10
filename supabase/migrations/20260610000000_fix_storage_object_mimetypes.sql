-- Fix MIME types on storage objects uploaded through the backend.
--
-- The Python storage client defaults uploads to text/plain when no
-- content-type is passed, so every file uploaded via the backend
-- (/upload, /contracts/upload, Drive import, split sheet save) was stored
-- with mimetype 'text/plain'. Signed URLs then serve PDFs as text/plain
-- and the browser's inline PDF viewer refuses to render them
-- ("unable to load PDF" in the Zoe contract slide-over).
--
-- The backend now passes an explicit content-type on upload; this syncs the
-- DB-side metadata for objects that already exist. NOTE: this alone does NOT
-- fix what signed URLs serve — Supabase Storage serves Content-Type from the
-- underlying S3 object, so the real backfill is a re-upload in place via
-- src/backend/scripts/backfill_storage_mimetypes.py (already run). This
-- migration keeps storage.objects.metadata consistent with the S3 layer.
-- Idempotent: rows already corrected don't match the WHERE clause.
--
-- Survey at time of writing (project-files bucket): 144 .pdf, 24 .xlsx,
-- 1 .docx stored as text/plain.

UPDATE storage.objects
SET metadata = jsonb_set(metadata, '{mimetype}', '"application/pdf"')
WHERE bucket_id = 'project-files'
  AND metadata->>'mimetype' = 'text/plain'
  AND lower(name) LIKE '%.pdf';

UPDATE storage.objects
SET metadata = jsonb_set(
    metadata, '{mimetype}', '"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"'
  )
WHERE bucket_id = 'project-files'
  AND metadata->>'mimetype' = 'text/plain'
  AND lower(name) LIKE '%.xlsx';

UPDATE storage.objects
SET metadata = jsonb_set(
    metadata, '{mimetype}', '"application/vnd.openxmlformats-officedocument.wordprocessingml.document"'
  )
WHERE bucket_id = 'project-files'
  AND metadata->>'mimetype' = 'text/plain'
  AND lower(name) LIKE '%.docx';
