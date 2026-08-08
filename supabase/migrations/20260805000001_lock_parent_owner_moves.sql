-- supabase/migrations/20260805000001_lock_parent_owner_moves.sql
-- ============================================================================
-- Close the parent-pointer walk-out one level above the files.
--
-- 20260803000002's lock_asset_owner_move guards project_files.project_id and
-- audio_files.folder_id, but the identical one-statement attack works on the
-- parents themselves. An active org member runs
--
--     UPDATE projects SET artist_id = <their own personal artist> WHERE ...
--
-- USING passes on the OLD row (they are a member of the owning team), WITH
-- CHECK passes on the NEW row (can_access_artist on their own personal artist
-- is true), permissive policies OR, and WITH CHECK cannot see OLD — so no
-- policy can express "the owner must not change". One statement walks out an
-- entire project WITH every file in it. Same shape for audio_folders.artist_id
-- (a whole folder of audio) and works_registry.artist_id / project_id (a work
-- plus its stakes, collaborators and licensing).
--
-- Rule: the RESOLVED OWNER may not change under an end-user JWT. The owner is
-- the artist's team_id when set, else the personal owner (artists.user_id) —
-- so moving between two projects/artists of the SAME team is ordinary work,
-- and moving between two of your OWN personal artists still works. The
-- service role (auth.uid() IS NULL) stays free, same as artists_lock_team_id:
-- the transfer endpoint re-points nothing here today, but must not be broken.
--
-- Deliberately STRICTER than lock_asset_owner_move's team_id-only comparison:
-- for a FILE a personal->personal cross-user move is already stopped by
-- personal RLS, but projects and works are also writable by project MEMBERS
-- who are not the artist owner — so the personal owner has to be part of the
-- comparison here, not just the team.
-- ============================================================================

BEGIN;

CREATE OR REPLACE FUNCTION public.lock_parent_owner_move()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_old_team UUID;
  v_new_team UUID;
  v_old_user UUID;
  v_new_user UUID;
  v_moved BOOLEAN;
BEGIN
  -- Service role: backend endpoints run with auth.uid() IS NULL and stay
  -- free, mirroring artists_lock_team_id.
  IF auth.uid() IS NULL THEN
    RETURN NEW;
  END IF;

  -- All three guarded tables carry artist_id directly.
  SELECT a.team_id, a.user_id INTO v_old_team, v_old_user
    FROM artists a WHERE a.id = OLD.artist_id;
  SELECT a.team_id, a.user_id INTO v_new_team, v_new_user
    FROM artists a WHERE a.id = NEW.artist_id;
  v_moved := v_old_team IS DISTINCT FROM v_new_team
          OR (v_old_team IS NULL AND v_new_team IS NULL
              AND v_old_user IS DISTINCT FROM v_new_user);

  -- works_registry has a SECOND parent edge: reparenting a work into another
  -- owner's project walks it out just as surely as re-pointing its artist.
  IF NOT v_moved AND TG_TABLE_NAME = 'works_registry' THEN
    SELECT a.team_id, a.user_id INTO v_old_team, v_old_user
      FROM projects p JOIN artists a ON a.id = p.artist_id
      WHERE p.id = OLD.project_id;
    SELECT a.team_id, a.user_id INTO v_new_team, v_new_user
      FROM projects p JOIN artists a ON a.id = p.artist_id
      WHERE p.id = NEW.project_id;
    v_moved := v_old_team IS DISTINCT FROM v_new_team
            OR (v_old_team IS NULL AND v_new_team IS NULL
                AND v_old_user IS DISTINCT FROM v_new_user);
  END IF;

  IF v_moved THEN
    RAISE EXCEPTION 'a % row cannot be moved between owners; transfer the artist instead', TG_TABLE_NAME
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS projects_lock_owner_move ON projects;
CREATE TRIGGER projects_lock_owner_move
  BEFORE UPDATE OF artist_id ON projects
  FOR EACH ROW EXECUTE FUNCTION public.lock_parent_owner_move();

DROP TRIGGER IF EXISTS audio_folders_lock_owner_move ON audio_folders;
CREATE TRIGGER audio_folders_lock_owner_move
  BEFORE UPDATE OF artist_id ON audio_folders
  FOR EACH ROW EXECUTE FUNCTION public.lock_parent_owner_move();

DROP TRIGGER IF EXISTS works_registry_lock_owner_move ON works_registry;
CREATE TRIGGER works_registry_lock_owner_move
  BEFORE UPDATE OF artist_id, project_id ON works_registry
  FOR EACH ROW EXECUTE FUNCTION public.lock_parent_owner_move();

COMMIT;
