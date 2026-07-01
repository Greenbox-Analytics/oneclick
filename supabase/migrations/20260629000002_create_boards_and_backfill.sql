-- Team Boards Phase 1 — the boards entity + unify existing single-user columns/tasks.
-- Spec §4 (boards table) and §11 (migration). Personal board = team_id NULL.
-- board_id stays NULLABLE: live insert paths (create_column/create_task/
-- create_parent_task/Notion sync) omit it. The NOT NULL lockdown is deferred to
-- Phase 3, after those paths are updated. See the plan's Production-safety callout.
-- Wrapped in an explicit transaction so an abort at the DO-block rolls everything back.
BEGIN;

-- ============================================================
-- 1. boards table
-- ============================================================
CREATE TABLE boards (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  team_id     UUID REFERENCES teams(id) ON DELETE CASCADE,   -- NULL = personal board
  owner_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  artist_id   UUID REFERENCES artists(id) ON DELETE SET NULL,  -- personal-board artist scope; NULL = team board OR the unscoped Personal board
  name        TEXT NOT NULL,
  description TEXT,
  archived    BOOLEAN NOT NULL DEFAULT false,
  position    INTEGER NOT NULL DEFAULT 0,                     -- ordered within (team_id, owner_id)
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_boards_team_id ON boards(team_id);
CREATE INDEX idx_boards_owner_id ON boards(owner_id);
CREATE INDEX idx_boards_owner_artist ON boards(owner_id, artist_id);  -- Phase 3 ensure_personal_board(user, artist) lookups

CREATE TRIGGER boards_updated_at
  BEFORE UPDATE ON boards
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- ============================================================
-- 2. Add NULLABLE board_id to columns + tasks
-- ============================================================
ALTER TABLE board_columns ADD COLUMN board_id UUID REFERENCES boards(id) ON DELETE CASCADE;
ALTER TABLE board_tasks   ADD COLUMN board_id UUID REFERENCES boards(id) ON DELETE CASCADE;

-- ============================================================
-- 3. Create one personal board per (user, artist_id) across columns ∪ tasks.
--    boards.artist_id is a PERMANENT column: it persists the personal-board artist scope so
--    the backend can resolve a user's board later (Phase 3 ensure_personal_board). Team boards
--    and the unscoped Personal board carry artist_id = NULL.
-- ============================================================
INSERT INTO boards (team_id, owner_id, name, artist_id)
SELECT NULL, p.user_id, COALESCE(a.name, 'Personal'), p.artist_id
FROM (
  SELECT DISTINCT user_id, artist_id FROM board_columns
  UNION
  SELECT DISTINCT user_id, artist_id FROM board_tasks
) p
LEFT JOIN artists a ON a.id = p.artist_id;

-- ============================================================
-- 4. Backfill board_id (existing rows only)
-- ============================================================
-- 4a. Columns → their (owner, artist) board. IS NOT DISTINCT FROM matches NULL=NULL.
UPDATE board_columns c
SET board_id = b.id
FROM boards b
WHERE b.owner_id = c.user_id
  AND b.artist_id IS NOT DISTINCT FROM c.artist_id;

-- 4b. Tasks WITH a column → that column's board (column wins, even if task.artist_id differs).
UPDATE board_tasks t
SET board_id = c.board_id
FROM board_columns c
WHERE t.column_id = c.id
  AND t.column_id IS NOT NULL;

-- 4c. Column-less tasks with a non-null artist_id → board for (owner, artist_id).
UPDATE board_tasks t
SET board_id = b.id
FROM boards b
WHERE t.column_id IS NULL AND t.board_id IS NULL
  AND t.artist_id IS NOT NULL
  AND b.owner_id = t.user_id
  AND b.artist_id IS NOT DISTINCT FROM t.artist_id;

-- 4d. Column-less NULL-artist parents WITH children → first child's board (subtree cohesion).
--     Step 4e is the catch-all, so completeness (and the DO-block below) holds regardless of
--     nesting depth. In practice the UI keeps nesting single-level (TaskSubtasks.tsx renders
--     only for is_parent tasks, so a subtask isn't given children); deeper nesting — not a DB
--     constraint — would only surface more split-subtree rows in the Step-4 check, never NULLs.
UPDATE board_tasks p
SET board_id = child.board_id
FROM (
  SELECT DISTINCT ON (parent_task_id) parent_task_id, board_id
  FROM board_tasks
  WHERE parent_task_id IS NOT NULL AND board_id IS NOT NULL
  ORDER BY parent_task_id, position NULLS LAST, created_at
) child
WHERE p.id = child.parent_task_id
  AND p.column_id IS NULL AND p.board_id IS NULL
  AND p.artist_id IS NULL;

-- 4e. Any remaining column-less tasks (NULL artist, no children) → owner's Personal board.
UPDATE board_tasks t
SET board_id = b.id
FROM boards b
WHERE t.column_id IS NULL AND t.board_id IS NULL
  AND b.owner_id = t.user_id
  AND b.artist_id IS NULL;

-- ============================================================
-- 5. Assert backfill completeness (integrity gate, NOT a lock). boards.artist_id is now a
--    permanent column, so there is no temp column to drop.
--    Aborts (rolls back the whole migration) if any EXISTING row was missed.
-- ============================================================
-- Scope the assertion to rows that existed when this transaction began: now() is fixed at
-- transaction start, so a row whose insert transaction STARTS after the migration begins
-- (NULL board_id, since old code doesn't set it) sorts after now() and is excluded — making a
-- spurious abort very unlikely. Not airtight: a transaction already in-flight at our start can
-- carry an earlier created_at and commit mid-migration into the count; the low-traffic-window
-- guidance covers that tiny window (an airtight version would briefly lock the tables).
-- Those interim NULLs are fine regardless (board_id stays nullable until Phase 3).
DO $$
DECLARE n_cols INT; n_tasks INT;
BEGIN
  SELECT count(*) INTO n_cols  FROM board_columns WHERE board_id IS NULL AND created_at < now();
  SELECT count(*) INTO n_tasks FROM board_tasks   WHERE board_id IS NULL AND created_at < now();
  IF n_cols > 0 OR n_tasks > 0 THEN
    RAISE EXCEPTION 'Backfill incomplete: % columns and % tasks have NULL board_id', n_cols, n_tasks;
  END IF;
END $$;

CREATE INDEX idx_board_columns_board_id ON board_columns(board_id);
CREATE INDEX idx_board_tasks_board_id   ON board_tasks(board_id);

-- ============================================================
-- 6. RLS on boards (owner or team member can read; teams via SECURITY DEFINER helper).
--    Arg order is (user, team): is_team_member(auth.uid(), team_id).
-- ============================================================
ALTER TABLE boards ENABLE ROW LEVEL SECURITY;

CREATE POLICY "boards_select_owner_or_member" ON boards
  FOR SELECT USING (
    owner_id = auth.uid()
    OR (team_id IS NOT NULL AND is_team_member(auth.uid(), team_id))
  );
CREATE POLICY "boards_insert_owner_or_member" ON boards
  FOR INSERT WITH CHECK (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_team_member(auth.uid(), team_id))
  );
CREATE POLICY "boards_update_owner_or_member" ON boards
  FOR UPDATE USING (
    owner_id = auth.uid()
    OR (team_id IS NOT NULL AND is_team_member(auth.uid(), team_id))
  );
CREATE POLICY "boards_delete_owner_or_admin" ON boards
  FOR DELETE USING (
    (team_id IS NULL AND owner_id = auth.uid())
    OR (team_id IS NOT NULL AND is_team_admin(auth.uid(), team_id))
  );

COMMIT;
