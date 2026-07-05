-- Fix: a hard `DELETE FROM teams` cascades to team_members, and the last-admin auto-promote
-- in team_members_admin_guard (a BEFORE-DELETE trigger that UPDATEs a sibling row to admin)
-- then targets a member the SAME cascade is also deleting → SQLSTATE 27000
-- ("tuple to be updated was already modified by an operation triggered by the current command").
-- This made hard team deletion impossible when a team had ≥2 members.
--
-- The app uses SOFT delete (archive_team), so this only affected the ops/hard-delete path.
-- Normal account-deletion cascade promotes a NON-deleted sibling and is unaffected. When the
-- whole team is torn down there is no admin to preserve, so we swallow exactly that condition.
BEGIN;

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

  IF NOT losing_admin THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  is_cascade := pg_trigger_depth() > 1
                OR NOT EXISTS (SELECT 1 FROM auth.users WHERE id = OLD.user_id);

  SELECT count(*) INTO other_admins
  FROM team_members
  WHERE team_id = OLD.team_id AND role = 'admin' AND id <> OLD.id;

  IF other_admins > 0 THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  SELECT count(*) INTO other_members
  FROM team_members
  WHERE team_id = OLD.team_id AND id <> OLD.id;

  IF other_members = 0 THEN
    IF is_cascade THEN
      IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
    END IF;
    RAISE EXCEPTION 'You are the only admin of this team — promote another member first';
  END IF;

  -- Other members remain → promote the longest-tenured to admin. If the promote target is
  -- itself being removed in the same command (hard team teardown), there is no admin to
  -- preserve — swallow SQLSTATE 27000 and let the delete proceed.
  BEGIN
    UPDATE team_members
    SET role = 'admin'
    WHERE id = (
      SELECT id FROM team_members
      WHERE team_id = OLD.team_id AND id <> OLD.id
      ORDER BY created_at, id
      LIMIT 1
    );
  EXCEPTION
    WHEN triggered_data_change_violation THEN
      NULL;
  END;

  IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;

COMMIT;
