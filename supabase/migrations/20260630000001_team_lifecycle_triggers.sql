-- Team Boards Phase 2 — membership lifecycle (spec §5.5) + signup auto-convert (§6).
-- Requires Phase 1 migrations 0001-0004 applied (team_members, boards,
-- board_task_assignees, notifications). Wrapped in a transaction (see Phase 1 apply note).
BEGIN;

-- ============================================================
-- 0. Auto-add the creator as the team's first admin. AFTER INSERT on teams.
--    Keyed on NEW.created_by (NOT auth.uid(), which is NULL under the service role — that's
--    why auto_create_project_owner needed backfill 20260607000000). Atomic with the team
--    insert, so a team can never exist without an admin (no orphan-team risk).
-- ============================================================
CREATE OR REPLACE FUNCTION auto_create_team_admin()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  IF NEW.created_by IS NOT NULL THEN
    INSERT INTO team_members (team_id, user_id, role, invited_by)
    VALUES (NEW.id, NEW.created_by, 'admin', NEW.created_by)
    ON CONFLICT (team_id, user_id) DO NOTHING;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER auto_create_team_admin_trigger
  AFTER INSERT ON teams
  FOR EACH ROW EXECUTE FUNCTION auto_create_team_admin();

-- ============================================================
-- 1. Last-admin guard + auto-promote. BEFORE DELETE OR UPDATE on team_members.
--    A team must always have >=1 admin while it has >=1 member.
-- ============================================================
CREATE OR REPLACE FUNCTION team_members_admin_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  losing_admin  BOOLEAN;
  is_cascade    BOOLEAN;
  other_admins  INT;
  other_members INT;
BEGIN
  IF TG_OP = 'UPDATE' THEN
    losing_admin := (OLD.role = 'admin' AND NEW.role <> 'admin');
  ELSE  -- DELETE
    losing_admin := (OLD.role = 'admin');
  END IF;

  -- Not an admin-loss (promotion, or touching a non-admin row): nothing to guard.
  -- This is also why the auto-promote UPDATE below can't recurse — promotion returns here.
  IF NOT losing_admin THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  -- Cascade context: inside a parent delete (RI cascade / account deletion) or the
  -- member's user no longer exists. Must NOT raise here; must still auto-promote.
  is_cascade := pg_trigger_depth() > 1
                OR NOT EXISTS (SELECT 1 FROM auth.users WHERE id = OLD.user_id);

  SELECT count(*) INTO other_admins
  FROM team_members
  WHERE team_id = OLD.team_id AND role = 'admin' AND id <> OLD.id;

  IF other_admins > 0 THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  -- Removing/demoting the LAST admin.
  SELECT count(*) INTO other_members
  FROM team_members
  WHERE team_id = OLD.team_id AND id <> OLD.id;

  IF other_members = 0 THEN
    IF is_cascade THEN
      -- Sole member being account/team-deleted: allow; trigger #2 archives the team.
      IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
    END IF;
    RAISE EXCEPTION 'You are the only admin of this team — promote another member first';
  END IF;

  -- Other members remain → promote the longest-tenured to admin (stable on ties via id).
  UPDATE team_members
  SET role = 'admin'
  WHERE id = (
    SELECT id FROM team_members
    WHERE team_id = OLD.team_id AND id <> OLD.id
    ORDER BY created_at, id
    LIMIT 1
  );

  IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;

CREATE TRIGGER team_members_admin_guard_trigger
  BEFORE DELETE OR UPDATE ON team_members
  FOR EACH ROW EXECUTE FUNCTION team_members_admin_guard();

-- ============================================================
-- 2. Archive an orphaned team when its last member leaves. AFTER DELETE on team_members.
--    No-ops if the team itself is being hard-deleted (the UPDATE finds no row).
-- ============================================================
CREATE OR REPLACE FUNCTION team_archive_if_empty()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM team_members WHERE team_id = OLD.team_id) THEN
    UPDATE teams SET archived_at = now()
    WHERE id = OLD.team_id AND archived_at IS NULL;
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER team_archive_if_empty_trigger
  AFTER DELETE ON team_members
  FOR EACH ROW EXECUTE FUNCTION team_archive_if_empty();

-- ============================================================
-- 3. Removal cleanup. AFTER DELETE on team_members. Drops the removed user's footprint
--    for THIS team — assignments on the team's boards + team-scoped notifications.
--    Forward-compatible: assignees arrive in Phase 3, task_assigned notifications in Phase 4.
-- ============================================================
CREATE OR REPLACE FUNCTION team_member_removal_cleanup()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  DELETE FROM board_task_assignees a
  USING board_tasks t, boards b
  WHERE a.task_id = t.id
    AND t.board_id = b.id
    AND b.team_id = OLD.team_id
    AND a.user_id = OLD.user_id;

  DELETE FROM notifications n
  WHERE n.user_id = OLD.user_id
    AND (
      (n.entity_type = 'team' AND n.entity_id = OLD.team_id)
      OR (n.type = 'task_assigned' AND (n.metadata ->> 'team_id') = OLD.team_id::text)
    );

  RETURN OLD;
END;
$$;

CREATE TRIGGER team_member_removal_cleanup_trigger
  AFTER DELETE ON team_members
  FOR EACH ROW EXECUTE FUNCTION team_member_removal_cleanup();

-- ============================================================
-- 4. Signup auto-convert. AFTER INSERT on auth.users. Mirrors
--    process_pending_project_invites (filters expires_at > now()).
-- ============================================================
CREATE OR REPLACE FUNCTION process_pending_team_invites()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  INSERT INTO team_members (team_id, user_id, role, invited_by)
  SELECT pi.team_id, NEW.id, pi.role, pi.invited_by
  FROM pending_team_invites pi
  WHERE LOWER(pi.email) = LOWER(NEW.email)
    AND pi.status = 'pending'
    AND pi.expires_at > now()
  ON CONFLICT (team_id, user_id) DO NOTHING;

  UPDATE pending_team_invites
  SET status = 'accepted'
  WHERE LOWER(email) = LOWER(NEW.email)
    AND status = 'pending'
    AND expires_at > now();

  RETURN NEW;
END;
$$;

CREATE TRIGGER process_pending_team_invites_on_signup
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION process_pending_team_invites();

COMMIT;
