-- supabase/migrations/20260805000004_file_size_and_archived_org_guards.sql
-- ============================================================================
-- Two small guards from the same audit:
--
-- 1. file_size was an unconstrained client-writable BIGINT on project_files
--    and audio_files. The storage triggers (_bump_storage) trust it, so one
--    INSERT with a huge NEGATIVE file_size zeroes out
--    organizations.storage_bytes (defeating the hard storage cap) or, with a
--    huge positive one, bricks an org's uploads. CHECK >= 0, added NOT VALID
--    so a pre-existing bad row cannot brick the migration, then VALIDATEd
--    opportunistically — the constraint binds every future write either way.
--
-- 2. artists_insert_team (20260803000002) required active membership but not
--    a live org: a member of an ARCHIVED org could still insert an artist
--    into it — a row nobody can ever access, since can_access_artist denies
--    on archived_at. Recreated with the archived check, matching what
--    can_access_artist already enforces for every read.
-- ============================================================================

BEGIN;

-- ------------------------------------------------------- 1. file_size >= 0 --
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'project_files_file_size_check') THEN
    ALTER TABLE project_files ADD CONSTRAINT project_files_file_size_check
      CHECK (file_size IS NULL OR file_size >= 0) NOT VALID;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'audio_files_file_size_check') THEN
    ALTER TABLE audio_files ADD CONSTRAINT audio_files_file_size_check
      CHECK (file_size IS NULL OR file_size >= 0) NOT VALID;
  END IF;
END $$;

-- Separate blocks so one table's bad legacy rows don't stop the other's
-- VALIDATE. A failure here is worth chasing (someone already wrote a negative
-- size — storage_bytes is off by that much), but the NOT VALID constraint
-- still binds every new write, which is what closes the hole.
DO $$
BEGIN
  ALTER TABLE project_files VALIDATE CONSTRAINT project_files_file_size_check;
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'project_files: negative file_size rows exist (storage attribution is off); the CHECK still binds new writes';
END $$;

DO $$
BEGIN
  ALTER TABLE audio_files VALIDATE CONSTRAINT audio_files_file_size_check;
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'audio_files: negative file_size rows exist (storage attribution is off); the CHECK still binds new writes';
END $$;

-- ----------------------------------- 2. no new artists in an archived org --
-- Same body as 20260803000002 plus the organizations join. WITH CHECK, not
-- USING: artists are created CLIENT-SIDE straight against PostgREST, so this
-- policy is the only thing standing between a caller and someone else's (or a
-- dead) team. user_id stays pinned to the caller.
DROP POLICY IF EXISTS "artists_insert_team" ON artists;
CREATE POLICY "artists_insert_team" ON artists
  FOR INSERT WITH CHECK (
    team_id IS NOT NULL
    AND auth.uid() = user_id
    AND EXISTS (
      SELECT 1 FROM org_members m
      JOIN organizations o ON o.id = m.org_id
       WHERE m.org_id = artists.team_id
         AND m.user_id = auth.uid()
         AND m.status = 'active'
         AND o.archived_at IS NULL
    )
  );

COMMIT;
