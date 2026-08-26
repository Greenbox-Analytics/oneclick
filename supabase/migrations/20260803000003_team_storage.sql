-- supabase/migrations/20260803000003_team_storage.sql
-- ============================================================================
-- Storage follows artist ownership.
--
-- usage_counters.total_storage_bytes is ONE number per user, which was fine
-- while every artist was personal. With team-owned artists a manager who joins
-- a label would otherwise carry the label's catalogue on their personal
-- counter.
--
-- The fix has to land in the TRIGGERS, not in recalc_user_storage. The triggers
-- (20260509000001:167-290) are the live path: they fire on every file write and
-- resolve the owner through artists.user_id, so teaching only the repair
-- function about team_id would leave every real upload still charging a person
-- and would leave organizations.storage_bytes permanently at zero -- a cap that
-- never fires.
--
-- The two owner helpers are replaced by artist-id lookups plus ONE _bump_storage
-- that decides where a delta lands. That also collapses ~120 lines of repeated
-- UPDATE blocks in the two trigger bodies into two four-line bodies, and gets
-- the cross-project / cross-owner move case right for free (it is just a
-- negative delta on the old artist and a positive one on the new).
-- ============================================================================

BEGIN;

ALTER TABLE organizations
  ADD COLUMN IF NOT EXISTS storage_bytes BIGINT NOT NULL DEFAULT 0;

-- Resolve the ARTIST a file hangs off, not its owner: who pays is
-- _bump_storage's decision now, and it is the only place that makes it.
CREATE OR REPLACE FUNCTION _artist_for_pf(pf_project_id UUID)
RETURNS UUID AS $$
  SELECT p.artist_id FROM projects p WHERE p.id = pf_project_id;
$$ LANGUAGE sql STABLE SECURITY DEFINER SET search_path TO 'public';

CREATE OR REPLACE FUNCTION _artist_for_af(af_folder_id UUID)
RETURNS UUID AS $$
  SELECT afo.artist_id FROM audio_folders afo WHERE afo.id = af_folder_id;
$$ LANGUAGE sql STABLE SECURITY DEFINER SET search_path TO 'public';

-- ONE place decides whose counter a byte delta lands on.
CREATE OR REPLACE FUNCTION _bump_storage(p_artist_id UUID, p_delta BIGINT)
RETURNS VOID AS $$
DECLARE a RECORD;
BEGIN
  IF p_artist_id IS NULL OR p_delta = 0 THEN RETURN; END IF;
  SELECT user_id, team_id INTO a FROM artists WHERE id = p_artist_id;
  IF NOT FOUND THEN RETURN; END IF;

  IF a.team_id IS NOT NULL THEN
    UPDATE organizations
       SET storage_bytes = GREATEST(storage_bytes + p_delta, 0), updated_at = now()
     WHERE id = a.team_id;
  ELSIF a.user_id IS NOT NULL THEN
    UPDATE usage_counters
       SET total_storage_bytes = GREATEST(total_storage_bytes + p_delta, 0), updated_at = now()
     WHERE user_id = a.user_id;
  END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public';

-- Both triggers reduce to: take the old row's bytes off its artist, put the new
-- row's bytes on its artist. INSERT has no OLD, DELETE has no NEW, and an
-- UPDATE that moves a file between projects/folders is handled without a
-- special case because the two artists are simply looked up separately.
CREATE OR REPLACE FUNCTION trigger_storage_pf_change()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP <> 'INSERT' THEN
    PERFORM _bump_storage(_artist_for_pf(OLD.project_id), -COALESCE(OLD.file_size, 0));
  END IF;
  IF TG_OP <> 'DELETE' THEN
    PERFORM _bump_storage(_artist_for_pf(NEW.project_id), COALESCE(NEW.file_size, 0));
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public';

CREATE OR REPLACE FUNCTION trigger_storage_af_change()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP <> 'INSERT' THEN
    PERFORM _bump_storage(_artist_for_af(OLD.folder_id), -COALESCE(OLD.file_size, 0));
  END IF;
  IF TG_OP <> 'DELETE' THEN
    PERFORM _bump_storage(_artist_for_af(NEW.folder_id), COALESCE(NEW.file_size, 0));
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public';

-- Dead once the trigger bodies above no longer call them (grep confirms those
-- two triggers were their only callers). Dropped rather than left in place: a
-- helper called _project_owner_for_pf sitting next to _artist_for_pf is an
-- invitation to call the wrong one later.
DROP FUNCTION IF EXISTS _project_owner_for_pf(UUID);
DROP FUNCTION IF EXISTS _audio_owner_for_af(UUID);

-- The personal recompute gains ONE clause: team-owned artists are not yours.
-- Body is otherwise the definition from 20260509000001.
CREATE OR REPLACE FUNCTION recalc_user_storage(p_user_id UUID)
RETURNS VOID AS $$
DECLARE
  total BIGINT;
BEGIN
  IF p_user_id IS NULL THEN RETURN; END IF;
  SELECT
    COALESCE((SELECT SUM(pf.file_size)
              FROM project_files pf
              JOIN projects p ON p.id = pf.project_id
              JOIN artists a ON a.id = p.artist_id
              WHERE a.user_id = p_user_id AND a.team_id IS NULL), 0)
    +
    COALESCE((SELECT SUM(af.file_size)
              FROM audio_files af
              JOIN audio_folders afo ON afo.id = af.folder_id
              JOIN artists a ON a.id = afo.artist_id
              WHERE a.user_id = p_user_id AND a.team_id IS NULL), 0)
  INTO total;

  UPDATE usage_counters
  SET total_storage_bytes = total, updated_at = now()
  WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public';

-- The team mirror: identical shape, keyed on team_id. Repair path only -- the
-- triggers keep the number live; this is what the transfer endpoint and support
-- call to re-derive it from scratch.
CREATE OR REPLACE FUNCTION recalc_team_storage(p_org_id UUID)
RETURNS BIGINT AS $$
DECLARE
  total BIGINT;
BEGIN
  IF p_org_id IS NULL THEN RETURN 0; END IF;
  SELECT
    COALESCE((SELECT SUM(pf.file_size)
              FROM project_files pf
              JOIN projects p ON p.id = pf.project_id
              JOIN artists a ON a.id = p.artist_id
              WHERE a.team_id = p_org_id), 0)
    +
    COALESCE((SELECT SUM(af.file_size)
              FROM audio_files af
              JOIN audio_folders afo ON afo.id = af.folder_id
              JOIN artists a ON a.id = afo.artist_id
              WHERE a.team_id = p_org_id), 0)
  INTO total;

  UPDATE organizations SET storage_bytes = total, updated_at = now() WHERE id = p_org_id;
  RETURN total;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public';

REVOKE EXECUTE ON FUNCTION public.recalc_team_storage(UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.recalc_team_storage(UUID) TO service_role;

-- Seed both sides for anything that already exists (no teams in production yet,
-- but a dev/QA database may have some, and recalc_user_storage's new clause
-- means every personal counter that held team bytes is now wrong).
DO $$
DECLARE r RECORD;
BEGIN
  FOR r IN SELECT DISTINCT team_id AS id FROM artists WHERE team_id IS NOT NULL LOOP
    PERFORM recalc_team_storage(r.id);
  END LOOP;
  FOR r IN SELECT DISTINCT user_id AS id FROM artists WHERE user_id IS NOT NULL LOOP
    PERFORM recalc_user_storage(r.id);
  END LOOP;
END $$;

COMMIT;
