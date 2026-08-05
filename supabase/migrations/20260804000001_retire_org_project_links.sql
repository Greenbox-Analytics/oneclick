-- supabase/migrations/20260804000001_retire_org_project_links.sql
-- ============================================================================
-- One payer edge. org_project_links answered "which org pays for this project"
-- one project at a time; artists.team_id answers it once per roster entry and
-- covers everything below the artist. Keeping both would leave two sources of
-- truth for who gets charged.
--
-- THIS MIGRATION DOES NOT BACKFILL, DELIBERATELY. A link is a per-PROJECT
-- BILLING edge; team_id is an artist-wide ACCESS edge. Auto-promoting one to
-- the other would silently hand every active member of the org read access to
-- every OTHER project, file, audio file and DSP credential under that artist --
-- projects nobody ever linked. That is an access-widening event and it belongs
-- to a person, not to a migration: each artist gets transferred through
-- POST /orgs/{id}/artists/{id}/transfer, deliberately, by its owner.
--
-- So this refuses to run while any link is not already covered, and prints what
-- to look at. With licensing off in production the expected row count is zero
-- and this degenerates to a clean DROP.
--
-- RUN ORDER: only after the code that stops reading this table has deployed.
-- ============================================================================

BEGIN;

DO $$
DECLARE
  v_n INTEGER;
  v_sample TEXT;
BEGIN
  SELECT count(*), string_agg(DISTINCT l.project_id::text, ', ')
    INTO v_n, v_sample
    FROM org_project_links l
    JOIN projects p ON p.id = l.project_id
    JOIN artists a ON a.id = p.artist_id
   WHERE a.team_id IS DISTINCT FROM l.org_id;

  IF v_n > 0 THEN
    RAISE EXCEPTION
      'refusing to drop org_project_links: % link(s) are not covered by artist ownership. '
      'Transfer each project''s artist to its org first '
      '(POST /orgs/{id}/artists/{id}/transfer), then re-run. Projects: %', v_n, v_sample;
  END IF;
END $$;

DROP TABLE org_project_links;

COMMIT;
