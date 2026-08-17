-- supabase/migrations/20260818000001_boards_to_orgs.sql
-- Boards on Teams (spec docs/superpowers/specs/2026-08-16-boards-on-teams-design.md):
-- board-teams merge into organizations.
--
-- DEPLOY ORDER — this migration is FORWARD-BREAKING for the old code. Once it
-- lands, the pre-merge backend cannot create a team board (its team_id points
-- at `teams`, which now violates the FK) and its list_boards returns nothing
-- for a team. Apply it INSIDE the deploy window, immediately after the new
-- backend is live on Cloud Run (Vercel can lag; the frontend only reads what
-- the backend gives it). 20260818000002 (the DROP) comes later still.
BEGIN;

-- 0. Pre-flight: uq_boards_personal_artist is UNIQUE (owner_id, artist_id)
--    WHERE team_id IS NULL AND artist_id IS NOT NULL, so a team board carrying
--    an artist_id could collide the moment step 1 NULLs its team_id — with a
--    bare 23505 halfway through a one-way migration. Nothing in the app
--    prevents that combination (BoardCreate accepts team_id and artist_id
--    together), so assert instead of hoping. Verified 0 such rows on
--    2026-08-16; this exists so a later environment fails LOUDLY and early.
DO $$
DECLARE clashes INTEGER;
BEGIN
  SELECT count(*) INTO clashes
    FROM boards b
   WHERE b.team_id IS NOT NULL AND b.artist_id IS NOT NULL
     AND b.team_id NOT IN (SELECT id FROM organizations)  -- LEGACY rows only; see step 1
     AND (
       EXISTS (SELECT 1 FROM boards p
                WHERE p.team_id IS NULL AND p.owner_id = b.owner_id AND p.artist_id = b.artist_id)
       OR EXISTS (SELECT 1 FROM boards o
                   WHERE o.team_id IS NOT NULL AND o.id <> b.id
                     AND o.owner_id = b.owner_id AND o.artist_id = b.artist_id)
     );
  IF clashes > 0 THEN
    RAISE EXCEPTION
      'boards_to_orgs: % team board(s) would collide with uq_boards_personal_artist once detached. '
      'Clear or re-point their artist_id first.', clashes;
  END IF;
END $$;

-- 1. Legacy board-team boards become PERSONAL boards of their creator (owner_id).
--    Assignees who are not that owner lose the assignment (a personal board has
--    exactly one possible assignee); board-team invite notifications are gone.
--    See Step 0: the owner has seen these counts before this runs.
DELETE FROM board_task_assignees a
 USING board_tasks t, boards b
 WHERE a.task_id = t.id AND t.board_id = b.id
   AND b.team_id IS NOT NULL AND a.user_id <> b.owner_id
   AND b.team_id NOT IN (SELECT id FROM organizations);  -- LEGACY boards only (see step 1)

DELETE FROM notifications WHERE type = 'team_invite' OR entity_type = 'team';

DO $$
DECLARE n INTEGER;
BEGIN
  -- `NOT IN (SELECT id FROM organizations)` is LOAD-BEARING, not belt-and-braces.
  -- Before step 2 runs, team_id points at a legacy `teams` row; AFTER it, the
  -- same column points at an `organizations` row. A bare
  -- `WHERE team_id IS NOT NULL` therefore means "legacy board" on the first
  -- apply and "every real team board" on a second one — re-running this file
  -- would silently detach every org board in the product, leave `restricted`
  -- set and `board_members` orphaned, and report it as a success. Every other
  -- statement in this migration is written to be re-run safe; this is the one
  -- that could not be, so it is scoped explicitly instead.
  UPDATE boards SET team_id = NULL
   WHERE team_id IS NOT NULL
     AND team_id NOT IN (SELECT id FROM organizations);
  GET DIAGNOSTICS n = ROW_COUNT;
  RAISE NOTICE 'boards_to_orgs: % board-team boards reparented to their creators', n;
END $$;

-- 2. Repoint the FK. RESTRICT, not CASCADE: orgs are archived/dissolved, never
--    deleted (20260803000001 uses the same posture for artists.team_id).
ALTER TABLE boards DROP CONSTRAINT IF EXISTS boards_team_id_fkey;
ALTER TABLE boards
  ADD CONSTRAINT boards_team_id_fkey FOREIGN KEY (team_id)
  REFERENCES organizations(id) ON DELETE RESTRICT;

-- 3. Visibility narrowing.
ALTER TABLE boards ADD COLUMN IF NOT EXISTS restricted BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS board_members (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  board_id   UUID NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  added_by   UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (board_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_board_members_user_id ON board_members(user_id);

-- 4a. ONE liveness definition, shared by every policy below and mirrored by
--     artist_access.live_org_ids in Python: an ACTIVE seat in an org that is
--     neither archived nor lapsed. is_org_member alone is NOT enough — it
--     ignores the org's own state, which is how a member of an archived org
--     could still INSERT into it.
CREATE OR REPLACE FUNCTION public.is_live_org_member(p_user_id UUID, p_org_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1 FROM org_members m JOIN organizations o ON o.id = m.org_id
     WHERE m.org_id = p_org_id AND m.user_id = p_user_id AND m.status = 'active'
       AND o.archived_at IS NULL AND o.status <> 'lapsed'
  );
$$;

-- 4b. ONE access predicate (mirrored by src/backend/boards/authz.py):
--     personal → owner; team → live seat AND (open board OR owner OR org admin
--     OR listed in board_members).
CREATE OR REPLACE FUNCTION public.can_access_board(p_board_id UUID, p_user_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1
      FROM boards b
      LEFT JOIN org_members m
             ON m.org_id = b.team_id AND m.user_id = p_user_id AND m.status = 'active'
     WHERE b.id = p_board_id
       AND (
         (b.team_id IS NULL AND b.owner_id = p_user_id)
         OR (
           b.team_id IS NOT NULL
           AND is_live_org_member(p_user_id, b.team_id)
           AND (
             NOT b.restricted
             OR b.owner_id = p_user_id
             OR m.role = 'admin'
             OR EXISTS (SELECT 1 FROM board_members bm WHERE bm.board_id = b.id AND bm.user_id = p_user_id)
           )
         )
       )
  );
$$;

REVOKE EXECUTE ON FUNCTION public.can_access_board(UUID, UUID) FROM PUBLIC;
GRANT  EXECUTE ON FUNCTION public.can_access_board(UUID, UUID) TO authenticated, service_role;
REVOKE EXECUTE ON FUNCTION public.is_live_org_member(UUID, UUID) FROM PUBLIC;
GRANT  EXECUTE ON FUNCTION public.is_live_org_member(UUID, UUID) TO authenticated, service_role;

-- 5. Rewrite the six team-keyed policies onto the predicate. All three write
--    policies use the SAME liveness helper as the read one.
DROP POLICY IF EXISTS "boards_select_owner_or_member" ON boards;
DROP POLICY IF EXISTS "boards_insert_owner_or_member" ON boards;
DROP POLICY IF EXISTS "boards_update_owner_or_member" ON boards;
DROP POLICY IF EXISTS "boards_delete_owner_or_admin"  ON boards;
-- The two RENAMED policies too: Postgres has no CREATE POLICY IF NOT EXISTS /
-- OR REPLACE, so without these a second apply aborts with 42710.
DROP POLICY IF EXISTS "boards_select_accessible"      ON boards;
DROP POLICY IF EXISTS "boards_update_owner_or_admin"  ON boards;

CREATE POLICY "boards_select_accessible" ON boards
  FOR SELECT USING (can_access_board(id, auth.uid()));
CREATE POLICY "boards_insert_owner_or_member" ON boards
  FOR INSERT WITH CHECK (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id))
  );
-- UPDATE is owner-or-admin, NOT can_access_board: a policy without WITH CHECK
-- reuses USING as the check, so "anyone who can see it can write it" would let
-- a plain member flip `restricted` (locking colleagues out) or repoint
-- `team_id` at another org they belong to, straight from the anon-key client.
-- The backend gates both; this closes the direct-client path.
CREATE POLICY "boards_update_owner_or_admin" ON boards
  FOR UPDATE USING (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id)
        AND (owner_id = auth.uid() OR is_org_admin(auth.uid(), team_id)))
  );
CREATE POLICY "boards_delete_owner_or_admin" ON boards
  FOR DELETE USING (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_live_org_member(auth.uid(), team_id)
        AND is_org_admin(auth.uid(), team_id))
  );

-- 5b. WITH CHECK can't see OLD and permissive policies OR together, so no
--     policy can express "the owning org must not change". Same problem, same
--     answer as artists_lock_team_id (20260803000001): a BEFORE UPDATE trigger
--     that refuses the move under an end-user JWT (auth.uid() IS NOT NULL) and
--     lets the service-role backend through.
CREATE OR REPLACE FUNCTION public.boards_lock_team_id()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  IF NEW.team_id IS DISTINCT FROM OLD.team_id AND auth.uid() IS NOT NULL THEN
    RAISE EXCEPTION 'boards.team_id is set at creation, not changed directly'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS boards_lock_team_id_trg ON boards;
CREATE TRIGGER boards_lock_team_id_trg
  BEFORE UPDATE OF team_id ON boards
  FOR EACH ROW EXECUTE FUNCTION public.boards_lock_team_id();

DROP POLICY IF EXISTS "board_task_assignees_select_board_reachable" ON board_task_assignees;
CREATE POLICY "board_task_assignees_select_board_reachable" ON board_task_assignees
  FOR SELECT USING (
    EXISTS (SELECT 1 FROM board_tasks t WHERE t.id = board_task_assignees.task_id
              AND can_access_board(t.board_id, auth.uid()))
  );

DROP POLICY IF EXISTS "board_task_works_select_board_reachable" ON board_task_works;
CREATE POLICY "board_task_works_select_board_reachable" ON board_task_works
  FOR SELECT USING (
    EXISTS (SELECT 1 FROM board_tasks t WHERE t.id = board_task_works.task_id
              AND can_access_board(t.board_id, auth.uid()))
  );

ALTER TABLE board_members ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "board_members_select_board_reachable" ON board_members;
CREATE POLICY "board_members_select_board_reachable" ON board_members
  FOR SELECT USING (can_access_board(board_id, auth.uid()));
-- No client write policies: writes go through the backend's service-role client.

-- 6. Assertions.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM boards WHERE team_id IS NOT NULL
               AND team_id NOT IN (SELECT id FROM organizations)) THEN
    RAISE EXCEPTION 'boards_to_orgs: boards.team_id still points outside organizations';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_policy p JOIN pg_class c ON c.oid = p.polrelid
              WHERE c.relname IN ('boards','board_task_assignees','board_task_works')
                AND (pg_get_expr(p.polqual, p.polrelid) ILIKE '%is_team_%'
                     OR pg_get_expr(p.polwithcheck, p.polrelid) ILIKE '%is_team_%')) THEN
    RAISE EXCEPTION 'boards_to_orgs: a board policy still references is_team_*';
  END IF;
END $$;

COMMIT;
