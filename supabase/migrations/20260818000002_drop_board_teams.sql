-- supabase/migrations/20260818000002_drop_board_teams.sql
-- Boards on Teams, part 2: remove the board-team tables. APPLY ONLY AFTER the
-- code from the 2026-08-16 plan is live (no code path references these objects
-- any more; the old code did) AND 20260818000001 has been verified. Nothing
-- here is reversible, so it is deliberately a separate, later migration.
BEGIN;

-- Triggers first — process_pending_team_invites_on_signup lives on auth.users
-- and would SURVIVE the DROP TABLEs below (its function reads the team tables,
-- so every signup would then fail).
DROP TRIGGER IF EXISTS process_pending_team_invites_on_signup ON auth.users;
DROP TRIGGER IF EXISTS team_member_removal_cleanup_trigger ON team_members;
DROP TRIGGER IF EXISTS team_archive_if_empty_trigger ON team_members;
DROP TRIGGER IF EXISTS team_members_admin_guard_trigger ON team_members;
DROP TRIGGER IF EXISTS auto_create_team_admin_trigger ON teams;
DROP TRIGGER IF EXISTS team_members_updated_at ON team_members;
DROP TRIGGER IF EXISTS teams_updated_at ON teams;

DROP FUNCTION IF EXISTS process_pending_team_invites();
DROP FUNCTION IF EXISTS team_member_removal_cleanup();
DROP FUNCTION IF EXISTS team_archive_if_empty();
DROP FUNCTION IF EXISTS team_members_admin_guard();
DROP FUNCTION IF EXISTS auto_create_team_admin();

-- Tables BEFORE is_team_member/is_team_admin: their own RLS policies call those
-- functions, so dropping the functions first would need a CASCADE. 0001 already
-- rewrote the boards / board_task_assignees policies off is_team_*, and
-- repointed boards.team_id at organizations, so nothing outside these three
-- tables depends on them any more.
DROP TABLE IF EXISTS pending_team_invites;
DROP TABLE IF EXISTS team_members;
DROP TABLE IF EXISTS teams;
DROP FUNCTION IF EXISTS is_team_member(UUID, UUID);
DROP FUNCTION IF EXISTS is_team_admin(UUID, UUID);

-- Repeat 0001's delete: between the two migrations the OLD backend may still be
-- serving and can still write a 'team_invite' row. ADD CONSTRAINT validates
-- existing rows, so one straggler would abort this whole migration.
DELETE FROM notifications WHERE type = 'team_invite' OR entity_type = 'team';
ALTER TABLE notifications DROP CONSTRAINT IF EXISTS notifications_type_check;
ALTER TABLE notifications ADD CONSTRAINT notifications_type_check
  CHECK (type IN ('invitation','confirmation','dispute','status_change','verification','task_assigned'));

COMMIT;
