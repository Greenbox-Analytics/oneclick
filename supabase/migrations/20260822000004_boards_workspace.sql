-- ============================================================================
-- Workspace scoping, follow-up 2/2: boards get a workspace filing label.
--
-- CAREFUL — boards now carry TWO org-valued columns that mean different things:
--
--   boards.team_id -> organizations(id)  SHARING: an org-owned board, visible
--                                        to the org's live seats (20260818000001,
--                                        can_access_board). NULL = personal.
--   boards.org_id  -> organizations(id)  FILING: which workspace a PERSONAL
--                                        board is filed under, so the workspace
--                                        switcher can show it in the right view.
--
-- org_id is a FILING LABEL, not a sharing grant. A personal board filed under
-- an org stays visible to its owner alone; sharing is team_id's job. For an
-- artist-keyed board org_id is DERIVED from artists.team_id, never from the
-- request scope — ensure_personal_board is keyed (owner_id, artist_id), so
-- stamping it from the ambient scope would create a second board for the same
-- artist and split its tasks in two.
--
-- This file must sort AFTER 20260818000001_boards_to_orgs.sql: it recreates two
-- of that migration's policies with an extra org_id guard, and it uses the
-- is_live_org_member predicate that file defines.
-- ============================================================================
BEGIN;

-- ON DELETE SET NULL, never CASCADE: dissolving an org reverts its artists to
-- people (docs/licensing.md, "Archive vs. dissolve") and must not take members'
-- boards with it. Losing the filing label is recoverable; losing the board is not.
ALTER TABLE boards ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id) ON DELETE SET NULL;

COMMENT ON COLUMN boards.org_id IS
  'Workspace this board is filed under. NULL = Personal. For an artist-keyed '
  'board it is DERIVED from artists.team_id, never from the request scope -- '
  'ensure_personal_board is keyed (owner_id, artist_id), so stamping it from '
  'the ambient scope would create a second board for the same artist and split '
  'its tasks in two. A filing label only: sharing is team_id''s job.';

-- Backfill: derive from the artist for artist-keyed boards. Everything else
-- stays NULL (Personal), which is where those boards effectively live today.
-- `b.org_id IS NULL` makes a re-run a no-op.
UPDATE boards b
SET org_id = a.team_id
FROM artists a
WHERE b.artist_id = a.id
  AND a.team_id IS NOT NULL
  AND b.org_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_boards_owner_org ON boards(owner_id, org_id);

-- ============================================================
-- RLS: pin org_id to a live membership.
--
-- Defense-in-depth (all board traffic is service-role today), closing the
-- direct-PostgREST path: without the guard a caller could file a board under
-- any org id, planting rows in a workspace they cannot even see. The two
-- policies recreated here are the CURRENT ones from 20260818000001 — same
-- names, same predicates — each extended with the org_id clause. Liveness is
-- is_live_org_member (never bare is_org_member): filing under an archived or
-- lapsed org fails closed, matching every other org predicate.
--
-- The UPDATE policy also gains an explicit WITH CHECK. 20260818000001 left it
-- USING-only (reused as the check), which the org_id guard must extend anyway;
-- restating both halves keeps the owner-or-admin write gate intact. The
-- boards_lock_team_id trigger still guards team_id moves; org_id may be
-- re-filed by whoever passes this policy — it is a label, not access.
-- ============================================================
DROP POLICY IF EXISTS "boards_insert_owner_or_member" ON boards;
CREATE POLICY "boards_insert_owner_or_member" ON boards
  FOR INSERT WITH CHECK (
    (
      (team_id IS NULL AND owner_id = auth.uid())
      OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id))
    )
    AND (org_id IS NULL OR is_live_org_member(auth.uid(), org_id))
  );

DROP POLICY IF EXISTS "boards_update_owner_or_admin" ON boards;
CREATE POLICY "boards_update_owner_or_admin" ON boards
  FOR UPDATE USING (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id)
        AND (owner_id = auth.uid() OR is_org_admin(auth.uid(), team_id)))
  ) WITH CHECK (
    (
      (team_id IS NULL AND owner_id = auth.uid())
      OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id)
          AND (owner_id = auth.uid() OR is_org_admin(auth.uid(), team_id)))
    )
    AND (org_id IS NULL OR is_live_org_member(auth.uid(), org_id))
  );

COMMIT;
