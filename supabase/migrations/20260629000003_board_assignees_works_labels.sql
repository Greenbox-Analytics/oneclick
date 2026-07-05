-- Team Boards Phase 1 — real task assignment, works linking, label-only chips. Spec §4.1/§4.2/§8.
BEGIN;

-- Real assignment (replaces free-text board_tasks.assignee_name; that column is kept for legacy display).
CREATE TABLE board_task_assignees (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id     UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  assigned_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(task_id, user_id)
);
CREATE INDEX idx_board_task_assignees_user ON board_task_assignees(user_id);  -- powers "Assigned to me"
CREATE INDEX idx_board_task_assignees_task ON board_task_assignees(task_id);

-- Task <-> works_registry link (new in v1).
CREATE TABLE board_task_works (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id    UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  work_id    UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  label      TEXT,   -- denormalized work title at link time
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(task_id, work_id)
);
CREATE INDEX idx_board_task_works_task ON board_task_works(task_id);

-- Denormalized labels on existing link tables (so teammates render chips without RLS access).
ALTER TABLE board_task_artists   ADD COLUMN label TEXT;
ALTER TABLE board_task_projects  ADD COLUMN label TEXT;
ALTER TABLE board_task_contracts ADD COLUMN label TEXT;

-- RLS on the new tables (SELECT-only; writes via service-role — see plan note).
-- A user may read a row if they can reach the underlying task's board (owner or team member).
ALTER TABLE board_task_assignees ENABLE ROW LEVEL SECURITY;
ALTER TABLE board_task_works ENABLE ROW LEVEL SECURITY;

CREATE POLICY "board_task_assignees_select_board_reachable" ON board_task_assignees
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM board_tasks t JOIN boards b ON b.id = t.board_id
      WHERE t.id = board_task_assignees.task_id
        AND (b.owner_id = auth.uid() OR (b.team_id IS NOT NULL AND is_team_member(auth.uid(), b.team_id)))
    )
  );

CREATE POLICY "board_task_works_select_board_reachable" ON board_task_works
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM board_tasks t JOIN boards b ON b.id = t.board_id
      WHERE t.id = board_task_works.task_id
        AND (b.owner_id = auth.uid() OR (b.team_id IS NOT NULL AND is_team_member(auth.uid(), b.team_id)))
    )
  );

COMMIT;
