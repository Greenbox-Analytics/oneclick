-- 20260614000200_lock_storage_buckets.sql
-- !!! DO NOT APPLY until the new frontend (signed-URL readers) is deployed to prod. !!!
-- This makes the buckets private; any client still relying on public URLs / raw paths will
-- break until it requests signed URLs. The backend (service-role) is unaffected.
--
-- Context: dev and prod share one Supabase database. Applying this before the signed-URL
-- frontend is live in prod will break file/audio downloads in the running prod app.
-- (audio-files is already private — included here for completeness/idempotency.)
update storage.buckets set public = false where id in ('project-files','audio-files');

-- Drop the over-permissive project-files policies (these were TO public — unauthenticated!).
drop policy if exists "Anyone can view project files" on storage.objects;
drop policy if exists "Anyone can upload project files" on storage.objects;
drop policy if exists "Anyone can update project files" on storage.objects;
drop policy if exists "Anyone can delete project files" on storage.objects;
-- Drop the blanket authenticated SELECT on audio (reads now go via signed URLs / backend).
drop policy if exists "Authenticated users can read audio files" on storage.objects;

-- Re-add project-files WRITE policies scoped to authenticated so the app's own uploads keep
-- working (the dropped ones were TO public). Reads are served via signed URLs, so no client
-- SELECT policy is added.
create policy "authed upload project files" on storage.objects
  for insert to authenticated with check (bucket_id = 'project-files');
create policy "authed update project files" on storage.objects
  for update to authenticated using (bucket_id = 'project-files');
create policy "authed delete project files" on storage.objects
  for delete to authenticated using (bucket_id = 'project-files');
-- audio-files already has TO authenticated insert/update/delete (keep them).
