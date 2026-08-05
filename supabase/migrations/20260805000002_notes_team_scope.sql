-- supabase/migrations/20260805000002_notes_team_scope.sql
-- ============================================================================
-- notes / note_folders missed the 20260803000002 creator re-scope.
--
-- Their policies (20260329100000) grant on `artists.user_id = auth.uid()` (and
-- a projects-join variant) with no `team_id IS NULL`, plus row-level user_id
-- ownership for read/update/delete. Consequence on a TEAM artist: the CREATOR
-- — including after offboarding — keeps full create/read/update/delete on
-- every note under the team subtree, while actual active members have no
-- access at all. Exactly the backdoor 20260803000002 closed for the other six
-- artist-scoped tables.
--
-- Same two treatments as that migration:
--   1. Each owner policy is dropped and re-created with the personal scope
--      made explicit (`AND a.team_id IS NULL`, resolved through artist_id or
--      project_id -> projects.artist_id — a note carries exactly one of the
--      two, enforced by the tables' CHECK constraint). Byte-identical for
--      every personal artist.
--   2. One ADDITIVE team-layer policy per table delegating to
--      can_access_artist, so access cannot drift from the other eight tables.
--
-- The two "Collaborators can read project ..." policies are untouched: they
-- grant through registry_collaborators, which is the work-collaborator layer,
-- orthogonal to artist ownership.
-- ============================================================================

BEGIN;

-- ======================================================== note_folders ======
DROP POLICY IF EXISTS "Owner can read own folders"               ON note_folders;
DROP POLICY IF EXISTS "Owner can insert folders with valid scope" ON note_folders;
DROP POLICY IF EXISTS "Owner can update own folders"             ON note_folders;
DROP POLICY IF EXISTS "Owner can delete own folders"             ON note_folders;

CREATE POLICY "note_folders_select_personal" ON note_folders FOR SELECT USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = note_folders.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = note_folders.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "note_folders_insert_personal" ON note_folders FOR INSERT WITH CHECK (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = note_folders.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = note_folders.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL))
  )
);

CREATE POLICY "note_folders_update_personal" ON note_folders FOR UPDATE USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = note_folders.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = note_folders.project_id AND a.team_id IS NULL))
  )
) WITH CHECK (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = note_folders.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = note_folders.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "note_folders_delete_personal" ON note_folders FOR DELETE USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = note_folders.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = note_folders.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "note_folders_team_artist" ON note_folders
  FOR ALL USING (
    can_access_artist(artist_id, auth.uid())
    OR EXISTS (SELECT 1 FROM projects p
                WHERE p.id = note_folders.project_id
                  AND can_access_artist(p.artist_id, auth.uid()))
  ) WITH CHECK (
    can_access_artist(artist_id, auth.uid())
    OR EXISTS (SELECT 1 FROM projects p
                WHERE p.id = note_folders.project_id
                  AND can_access_artist(p.artist_id, auth.uid()))
  );

-- =============================================================== notes ======
DROP POLICY IF EXISTS "Owner can read own notes"               ON notes;
DROP POLICY IF EXISTS "Owner can insert notes with valid scope" ON notes;
DROP POLICY IF EXISTS "Owner can update own notes"             ON notes;
DROP POLICY IF EXISTS "Owner can delete own notes"             ON notes;

CREATE POLICY "notes_select_personal" ON notes FOR SELECT USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = notes.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = notes.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "notes_insert_personal" ON notes FOR INSERT WITH CHECK (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = notes.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = notes.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL))
  )
);

CREATE POLICY "notes_update_personal" ON notes FOR UPDATE USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = notes.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = notes.project_id AND a.team_id IS NULL))
  )
) WITH CHECK (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = notes.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = notes.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "notes_delete_personal" ON notes FOR DELETE USING (
  auth.uid() = user_id
  AND (
    (artist_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM artists a
       WHERE a.id = notes.artist_id AND a.team_id IS NULL))
    OR
    (project_id IS NOT NULL AND EXISTS (
      SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
       WHERE p.id = notes.project_id AND a.team_id IS NULL))
  )
);

CREATE POLICY "notes_team_artist" ON notes
  FOR ALL USING (
    can_access_artist(artist_id, auth.uid())
    OR EXISTS (SELECT 1 FROM projects p
                WHERE p.id = notes.project_id
                  AND can_access_artist(p.artist_id, auth.uid()))
  ) WITH CHECK (
    can_access_artist(artist_id, auth.uid())
    OR EXISTS (SELECT 1 FROM projects p
                WHERE p.id = notes.project_id
                  AND can_access_artist(p.artist_id, auth.uid()))
  );

COMMIT;
