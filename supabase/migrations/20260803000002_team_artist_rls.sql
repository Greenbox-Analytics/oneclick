-- supabase/migrations/20260803000002_team_artist_rls.sql
-- ============================================================================
-- Team access for everything that hangs off an artist.
--
-- THE CENTRAL PROBLEM, and why this migration is much bigger than "add some
-- team policies": `artists.user_id` keeps holding the CREATOR after an artist
-- becomes team-owned. Every pre-existing policy on every artist-scoped table
-- is written as
--
--     EXISTS (SELECT 1 FROM artists a WHERE a.id = <fk> AND a.user_id = auth.uid())
--
-- so the creator keeps PERSONAL-OWNER rights over the whole subtree of a team
-- artist -- forever, including after `org_members.status` flips to suspended.
-- `can_access_artist` correctly denies an offboarded member; the old policy
-- grants anyway, and permissive policies OR. Adding team policies beside them
-- would leave that wide open.
--
-- So there are TWO treatments here:
--
--   1. Every creator-keyed policy (21 of them, across projects, project_files,
--      audio_folders, audio_files, project_audio_links and artist_credentials)
--      is dropped and re-created with `AND a.team_id IS NULL`. Byte-identical
--      for every artist that exists today -- all of them have team_id NULL --
--      and it makes each policy say what it now means: this is the PERSONAL
--      path.
--
--      (Rejected alternative: null out `artists.user_id` when an artist becomes
--      team-owned, which makes all 21 fail closed for free. Ten lines instead
--      of a hundred, but implicit -- a future change that re-populates user_id
--      silently reopens 21 holes at once, and nothing in the policy text would
--      warn you. A security boundary is the wrong place to be clever.)
--
--   2. Team policies are then purely ADDITIVE beside them. Postgres RLS
--      policies are PERMISSIVE and OR together, so `personal OR team` needs no
--      further edits. One `FOR ALL` policy per table rather than four
--      verb-specific ones: the predicate is identical for all four verbs, and
--      four copies is four places to fix a mistake.
--
-- Every team policy delegates to can_access_artist rather than repeating the
-- join, so access cannot drift table to table.
-- ============================================================================

BEGIN;

-- =========================================================== artists ========
-- Replacing 20260112000000_add_user_id_to_artists.sql's four policies.
DROP POLICY IF EXISTS "Users can view their own artists"   ON artists;
DROP POLICY IF EXISTS "Users can create their own artists" ON artists;
DROP POLICY IF EXISTS "Users can update their own artists" ON artists;
DROP POLICY IF EXISTS "Users can delete their own artists" ON artists;

CREATE POLICY "artists_select_personal" ON artists
  FOR SELECT USING (team_id IS NULL AND auth.uid() = user_id);

CREATE POLICY "artists_insert_personal" ON artists
  FOR INSERT WITH CHECK (team_id IS NULL AND auth.uid() = user_id);

CREATE POLICY "artists_update_personal" ON artists
  FOR UPDATE USING (team_id IS NULL AND auth.uid() = user_id)
          WITH CHECK (team_id IS NULL AND auth.uid() = user_id);

CREATE POLICY "artists_delete_personal" ON artists
  FOR DELETE USING (team_id IS NULL AND auth.uid() = user_id);

CREATE POLICY "artists_select_team" ON artists
  FOR SELECT USING (team_id IS NOT NULL AND can_access_artist(id, auth.uid()));

-- WITH CHECK, not USING: artists are created CLIENT-SIDE straight against
-- PostgREST (src/pages/NewArtist.tsx, src/components/NewArtistDialog.tsx), so
-- this is the only thing stopping a caller inserting a roster row into someone
-- else's team. user_id is pinned to the caller so the creator stamp cannot be
-- forged either.
CREATE POLICY "artists_insert_team" ON artists
  FOR INSERT WITH CHECK (
    team_id IS NOT NULL
    AND auth.uid() = user_id
    AND EXISTS (
      SELECT 1 FROM org_members m
       WHERE m.org_id = artists.team_id
         AND m.user_id = auth.uid()
         AND m.status = 'active'
    )
  );

-- WITH CHECK mirrors USING so a member cannot move an artist to a second team
-- they happen to belong to. The artists_lock_team_id trigger from
-- 20260803000001 is the backstop that catches every other route.
CREATE POLICY "artists_update_team" ON artists
  FOR UPDATE USING (team_id IS NOT NULL AND can_access_artist(id, auth.uid()))
          WITH CHECK (team_id IS NOT NULL AND can_access_artist(id, auth.uid()));

-- Admin-only, and this is the one place it genuinely matters: ten tables
-- cascade off artists, so deleting the row deletes the team's whole catalogue.
CREATE POLICY "artists_delete_team" ON artists
  FOR DELETE USING (team_id IS NOT NULL AND can_access_artist(id, auth.uid(), TRUE));

-- =========================================================== projects =======
-- From 20260320200000_secure_projects_and_files_rls.sql. Same bodies, plus the
-- clause that stops them applying to a team artist's subtree.
DROP POLICY IF EXISTS "Users can view own projects"   ON public.projects;
DROP POLICY IF EXISTS "Users can create own projects" ON public.projects;
DROP POLICY IF EXISTS "Users can update own projects" ON public.projects;
DROP POLICY IF EXISTS "Users can delete own projects" ON public.projects;

CREATE POLICY "projects_select_personal" ON projects FOR SELECT USING (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = projects.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "projects_insert_personal" ON projects FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = projects.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "projects_update_personal" ON projects FOR UPDATE USING (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = projects.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "projects_delete_personal" ON projects FOR DELETE USING (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = projects.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);

CREATE POLICY "projects_team_artist" ON projects
  FOR ALL USING (can_access_artist(artist_id, auth.uid()))
      WITH CHECK (can_access_artist(artist_id, auth.uid()));

-- ====================================================== project_files =======
DROP POLICY IF EXISTS "Users can view own project files"   ON public.project_files;
DROP POLICY IF EXISTS "Users can create own project files" ON public.project_files;
DROP POLICY IF EXISTS "Users can update own project files" ON public.project_files;
DROP POLICY IF EXISTS "Users can delete own project files" ON public.project_files;

CREATE POLICY "project_files_select_personal" ON project_files FOR SELECT USING (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_files.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "project_files_insert_personal" ON project_files FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_files.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "project_files_update_personal" ON project_files FOR UPDATE USING (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_files.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "project_files_delete_personal" ON project_files FOR DELETE USING (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_files.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);

CREATE POLICY "project_files_team_artist" ON project_files
  FOR ALL USING (
    EXISTS (SELECT 1 FROM projects p
             WHERE p.id = project_files.project_id
               AND can_access_artist(p.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM projects p
             WHERE p.id = project_files.project_id
               AND can_access_artist(p.artist_id, auth.uid()))
  );

-- ====================================================== audio_folders =======
DROP POLICY IF EXISTS "Users can view audio folders for their artists"   ON audio_folders;
DROP POLICY IF EXISTS "Users can create audio folders for their artists" ON audio_folders;
DROP POLICY IF EXISTS "Users can delete audio folders for their artists" ON audio_folders;

CREATE POLICY "audio_folders_select_personal" ON audio_folders FOR SELECT USING (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = audio_folders.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "audio_folders_insert_personal" ON audio_folders FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = audio_folders.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "audio_folders_delete_personal" ON audio_folders FOR DELETE USING (
  EXISTS (SELECT 1 FROM artists a
           WHERE a.id = audio_folders.artist_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);

CREATE POLICY "audio_folders_team_artist" ON audio_folders
  FOR ALL USING (can_access_artist(artist_id, auth.uid()))
      WITH CHECK (can_access_artist(artist_id, auth.uid()));

-- ======================================================== audio_files =======
DROP POLICY IF EXISTS "Users can view audio files for their artists"   ON audio_files;
DROP POLICY IF EXISTS "Users can create audio files for their artists" ON audio_files;
DROP POLICY IF EXISTS "Users can delete audio files for their artists" ON audio_files;

CREATE POLICY "audio_files_select_personal" ON audio_files FOR SELECT USING (
  EXISTS (SELECT 1 FROM audio_folders af JOIN artists a ON a.id = af.artist_id
           WHERE af.id = audio_files.folder_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "audio_files_insert_personal" ON audio_files FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM audio_folders af JOIN artists a ON a.id = af.artist_id
           WHERE af.id = audio_files.folder_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "audio_files_delete_personal" ON audio_files FOR DELETE USING (
  EXISTS (SELECT 1 FROM audio_folders af JOIN artists a ON a.id = af.artist_id
           WHERE af.id = audio_files.folder_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);

CREATE POLICY "audio_files_team_artist" ON audio_files
  FOR ALL USING (
    EXISTS (SELECT 1 FROM audio_folders f
             WHERE f.id = audio_files.folder_id
               AND can_access_artist(f.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM audio_folders f
             WHERE f.id = audio_files.folder_id
               AND can_access_artist(f.artist_id, auth.uid()))
  );

-- ================================================= project_audio_links ======
-- From 20260315000000. This table had NO team policy at all in the first draft
-- of this migration -- a team member could see the audio and the project but
-- not the link between them.
DROP POLICY IF EXISTS "Users can view their project audio links"   ON project_audio_links;
DROP POLICY IF EXISTS "Users can create project audio links"       ON project_audio_links;
DROP POLICY IF EXISTS "Users can delete their project audio links" ON project_audio_links;

CREATE POLICY "project_audio_links_select_personal" ON project_audio_links FOR SELECT USING (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_audio_links.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "project_audio_links_insert_personal" ON project_audio_links FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_audio_links.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);
CREATE POLICY "project_audio_links_delete_personal" ON project_audio_links FOR DELETE USING (
  EXISTS (SELECT 1 FROM projects p JOIN artists a ON a.id = p.artist_id
           WHERE p.id = project_audio_links.project_id AND a.user_id = auth.uid() AND a.team_id IS NULL)
);

CREATE POLICY "project_audio_links_team_artist" ON project_audio_links
  FOR ALL USING (
    EXISTS (SELECT 1 FROM projects p
             WHERE p.id = project_audio_links.project_id
               AND can_access_artist(p.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM projects p
             WHERE p.id = project_audio_links.project_id
               AND can_access_artist(p.artist_id, auth.uid()))
  );

-- ====================================================== works_registry ======
-- No creator-keyed policy to scope here: works_registry access has always run
-- through project_members (20260403000008/9), never through artists.user_id.
-- works_registry carries artist_id directly, so this needs no hop through
-- projects.
CREATE POLICY "works_team_artist" ON works_registry
  FOR ALL USING (can_access_artist(artist_id, auth.uid()))
      WITH CHECK (can_access_artist(artist_id, auth.uid()));

-- ================================================== artist_credentials ======
-- These are DSP logins -- the single most valuable thing to walk off with, and
-- rotating them locks everyone else out. TWO changes:
--
--   * the four owner policies (20260420000002) key on the CREDENTIAL ROW's own
--     user_id, so whoever added a credential kept it after being offboarded.
--     Scoped to a personal artist, same as everything above.
--   * team access is ADMIN-ONLY, read included. There is no reason every member
--     of a label needs the Spotify-for-Artists password, and member-wide read
--     was the highest-value hole in the first draft of this migration.
DROP POLICY IF EXISTS "artist_credentials_owner_select" ON artist_credentials;
DROP POLICY IF EXISTS "artist_credentials_owner_insert" ON artist_credentials;
DROP POLICY IF EXISTS "artist_credentials_owner_update" ON artist_credentials;
DROP POLICY IF EXISTS "artist_credentials_owner_delete" ON artist_credentials;

CREATE POLICY "artist_credentials_personal" ON artist_credentials
  FOR ALL USING (
    auth.uid() = user_id
    AND EXISTS (SELECT 1 FROM artists a WHERE a.id = artist_credentials.artist_id AND a.team_id IS NULL)
  ) WITH CHECK (
    auth.uid() = user_id
    AND EXISTS (SELECT 1 FROM artists a WHERE a.id = artist_credentials.artist_id AND a.team_id IS NULL)
  );

CREATE POLICY "artist_credentials_team_admin" ON artist_credentials
  FOR ALL USING (can_access_artist(artist_id, auth.uid(), TRUE))
      WITH CHECK (can_access_artist(artist_id, auth.uid(), TRUE));

-- ============================================ asset moves across owners ======
-- A member with legitimate read on a team file could otherwise walk it out with
-- ONE statement:
--
--     UPDATE project_files SET project_id = <my own personal project> WHERE ...
--
-- The team policy's USING passes on the OLD row (they are a member) and its
-- WITH CHECK passes on the NEW row (can_access_artist on their own personal
-- artist is true). Permissive policies OR, and WITH CHECK cannot see OLD, so no
-- policy can express "the owner must not change" -- same reason artists.team_id
-- needed a trigger in 20260803000001.
--
-- Comparing team_id rather than artist_id is deliberate: moving a file between
-- two projects of the SAME team is ordinary work, and moving between two of
-- your OWN personal artists (NULL -> NULL) stays a personal-RLS question.
-- ============================================================================
CREATE OR REPLACE FUNCTION public.lock_asset_owner_move()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_old_team UUID;
  v_new_team UUID;
BEGIN
  IF TG_TABLE_NAME = 'project_files' THEN
    SELECT a.team_id INTO v_old_team
      FROM projects p JOIN artists a ON a.id = p.artist_id WHERE p.id = OLD.project_id;
    SELECT a.team_id INTO v_new_team
      FROM projects p JOIN artists a ON a.id = p.artist_id WHERE p.id = NEW.project_id;
  ELSE
    SELECT a.team_id INTO v_old_team
      FROM audio_folders f JOIN artists a ON a.id = f.artist_id WHERE f.id = OLD.folder_id;
    SELECT a.team_id INTO v_new_team
      FROM audio_folders f JOIN artists a ON a.id = f.artist_id WHERE f.id = NEW.folder_id;
  END IF;

  IF v_old_team IS DISTINCT FROM v_new_team THEN
    RAISE EXCEPTION 'a file cannot be moved between owners; transfer the artist instead'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS project_files_lock_owner_move ON project_files;
CREATE TRIGGER project_files_lock_owner_move
  BEFORE UPDATE OF project_id ON project_files
  FOR EACH ROW EXECUTE FUNCTION public.lock_asset_owner_move();

DROP TRIGGER IF EXISTS audio_files_lock_owner_move ON audio_files;
CREATE TRIGGER audio_files_lock_owner_move
  BEFORE UPDATE OF folder_id ON audio_files
  FOR EACH ROW EXECUTE FUNCTION public.lock_asset_owner_move();

COMMIT;
