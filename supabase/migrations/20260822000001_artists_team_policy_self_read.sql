-- supabase/migrations/20260822000001_artists_team_policy_self_read.sql
-- ============================================================================
-- Creating an artist inside a team failed. The INSERT was never the problem --
-- the read-back was.
--
-- Both artist-creation paths are client-side, straight against PostgREST
-- (src/pages/NewArtist.tsx, src/components/NewArtistDialog.tsx), and both end
-- with `.insert({...}).select().single()`. That is an INSERT ... RETURNING, and
-- Postgres applies the table's SELECT policy to the returned row. The two
-- policies from 20260803000002 behave differently under it:
--
--   artists_select_personal  team_id IS NULL AND auth.uid() = user_id
--       -> tests the NEW ROW's own columns. RETURNING passes.
--
--   artists_select_team      team_id IS NOT NULL AND can_access_artist(id, auth.uid())
--       -> can_access_artist does `SELECT ... FROM artists WHERE a.id = ...`.
--          It RE-READS THE TABLE. The function is STABLE, so it runs under the
--          calling statement's snapshot, and a row inserted by that same
--          statement is not in it (cmin = current command id). It returns
--          false, the row is denied, and the INSERT errors with 42501 instead
--          of returning the new artist.
--
-- So a policy on `artists` must never be written as a re-read of `artists`.
-- Fix: express the four team policies against the ROW'S OWN team_id, using the
-- predicate that already exists for exactly this shape --
-- is_live_org_member(user, org) from 20260818000001, which is verbatim
-- can_access_artist's team branch (ACTIVE seat, org not archived, not lapsed)
-- but keyed on the org id, so it never touches `artists`.
--
-- Semantics are unchanged for every existing row. For a row with team_id NOT
-- NULL, can_access_artist(id, uid) IS is_live_org_member(uid, team_id), and the
-- require_admin variant IS that plus the admin role check. Only the EVALUATION
-- changes -- no self-read, so INSERT ... RETURNING works.
--
-- One deliberate behaviour change, from the same audit: artists_insert_team
-- (20260805000004) checked o.archived_at but NOT o.status <> 'lapsed', which
-- can_access_artist gained in 20260816000002. A member of a lapsed org could
-- create an artist nobody -- admins included -- could ever read. Folding the
-- policy onto is_live_org_member closes that as a side effect.
--
-- NOT touched: the child-table policies (projects, works_registry,
-- project_files, notes, ...). They call can_access_artist about a DIFFERENT
-- table's row, which is already committed and therefore visible under any
-- snapshot -- they have no self-read problem, and the one-predicate rule in
-- docs/licensing.md keeps holding for them. `boards` has the same self-read
-- shape (boards_select -> can_access_board -> reads boards) but is safe today
-- because boards are created server-side on the service-role client, where RLS
-- is bypassed entirely; if a board ever gains a client-side create path, it
-- needs this same treatment.
-- ============================================================================

BEGIN;

-- ------------------------------------------------- one liveness definition --
-- can_access_artist's team branch was a second, hand-inlined copy of
-- is_live_org_member's body. Refold it onto the function so the two cannot
-- drift (identical output; the personal branch and the p_require_admin role
-- check are untouched). The org_members LEFT JOIN survives only to supply
-- m.role for the admin check -- liveness now comes entirely from the helper.
CREATE OR REPLACE FUNCTION public.can_access_artist(
  p_artist_id UUID,
  p_user_id UUID,
  p_require_admin BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1
      FROM artists a
      LEFT JOIN org_members m
             ON m.org_id = a.team_id
            AND m.user_id = p_user_id
            AND m.status = 'active'
     WHERE a.id = p_artist_id
       AND (
         -- personal: the creator, and only while it is not team-owned
         (a.team_id IS NULL AND a.user_id = p_user_id)
         -- team: a live seat (active member, org neither archived nor lapsed),
         -- plus the admin role when the caller asked for it
         OR (a.team_id IS NOT NULL
             AND is_live_org_member(p_user_id, a.team_id)
             AND (NOT p_require_admin OR m.role = 'admin'))
       )
  );
$$;

REVOKE EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) TO authenticated, service_role;

-- --------------------------------------- artists team policies, row-keyed --
-- Every one of these is now a predicate over the row's own team_id. None of
-- them reads `artists`, so all four are correct during INSERT/UPDATE, not just
-- against already-committed rows.

DROP POLICY IF EXISTS "artists_select_team" ON artists;
CREATE POLICY "artists_select_team" ON artists
  FOR SELECT USING (
    team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id)
  );

-- WITH CHECK, not USING: artists are created CLIENT-SIDE straight against
-- PostgREST, so this is the only thing stopping a caller inserting a roster row
-- into someone else's (or a dead) team. user_id stays pinned to the caller so
-- the creator stamp cannot be forged either.
DROP POLICY IF EXISTS "artists_insert_team" ON artists;
CREATE POLICY "artists_insert_team" ON artists
  FOR INSERT WITH CHECK (
    team_id IS NOT NULL
    AND auth.uid() = user_id
    AND is_live_org_member(auth.uid(), team_id)
  );

-- WITH CHECK mirrors USING so a member cannot move an artist to a second team
-- they happen to belong to. The artists_lock_team_id trigger from
-- 20260803000001 is the backstop that catches every other route.
DROP POLICY IF EXISTS "artists_update_team" ON artists;
CREATE POLICY "artists_update_team" ON artists
  FOR UPDATE USING      (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id))
          WITH CHECK    (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id));

-- Admin-only, and this is the one place it genuinely matters: ten tables
-- cascade off artists, so deleting the row deletes the team's whole catalogue.
-- is_org_admin (20260721000001) does NOT check the org's own state, which is
-- why it is ANDed with is_live_org_member rather than used alone.
DROP POLICY IF EXISTS "artists_delete_team" ON artists;
CREATE POLICY "artists_delete_team" ON artists
  FOR DELETE USING (
    team_id IS NOT NULL
    AND is_live_org_member(auth.uid(), team_id)
    AND is_org_admin(auth.uid(), team_id)
  );

COMMIT;
