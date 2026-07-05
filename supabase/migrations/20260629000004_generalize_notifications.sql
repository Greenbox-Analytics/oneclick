-- Team Boards Phase 1 — generalize the notifications surface. Spec §4.3.
-- registry_notifications becomes the unified `notifications` table; a back-compat
-- view keeps old backend instances working during a rolling deploy.
BEGIN;

-- 1. Rename (RLS policies stay attached to the table across rename).
ALTER TABLE registry_notifications RENAME TO notifications;

-- 2. Widen the type CHECK to admit the two new types.
--    registry_notifications_type_check is Postgres's auto-name for the original inline CHECK.
ALTER TABLE notifications DROP CONSTRAINT IF EXISTS registry_notifications_type_check;
ALTER TABLE notifications ADD CONSTRAINT notifications_type_check
  CHECK (type IN ('invitation','confirmation','dispute','status_change','verification','task_assigned','team_invite'));

-- 3. Generalized polymorphic reference (work_id retained for back-compat).
ALTER TABLE notifications ADD COLUMN entity_type TEXT;
ALTER TABLE notifications ADD COLUMN entity_id   UUID;

-- 4. Backfill existing registry rows.
UPDATE notifications
SET entity_type = 'work', entity_id = work_id
WHERE work_id IS NOT NULL AND entity_type IS NULL;

-- 5. Back-compat view for rolling deploys. security_invoker = true so the view enforces the
--    querying role's RLS (no privilege-escalation shim). Auto-updatable (simple SELECT *),
--    so old instances writing db.table("registry_notifications").insert(...) pass through.
CREATE VIEW registry_notifications WITH (security_invoker = true) AS
  SELECT * FROM notifications;

-- 5b. A new view does NOT inherit the renamed table's grants — and the original table had no
--     explicit grant (it relied on Supabase default privileges), so the view's coverage isn't
--     guaranteed. Grant explicitly so the service-role backend (and authenticated) can
--     read/write the shim; without this the back-compat insert silently fails.
GRANT SELECT, INSERT, UPDATE, DELETE ON registry_notifications TO authenticated, service_role;

-- 6. Nudge PostgREST to reload its schema cache so the new view/columns are exposed immediately.
NOTIFY pgrst, 'reload schema';

COMMIT;
