-- Fix v2: hard `DELETE FROM teams` with >=2 members still raised SQLSTATE 27000 after
-- 20260701000002. That migration wrapped the auto-promote UPDATE in an EXCEPTION handler,
-- but the error is NOT raised inside the UPDATE — it surfaces AFTER the trigger returns,
-- when the outer cascade reaches the sibling row the trigger just modified ("tuple to be
-- updated was already modified by an operation triggered by the current command"). A
-- BEFORE-trigger exception handler can never catch that. Verified live 2026-07-03.
--
-- Correct fix: detect whole-team teardown and skip BOTH the last-admin rejection and the
-- auto-promote — when the entire team is being deleted there is no admin-ship to preserve.
-- Detection: during an ON DELETE CASCADE from `teams`, the parent row is already deleted
-- before the cascade fires, so it is invisible here. This is the same row-visibility trick
-- the guard already uses for account-deletion cascades (NOT EXISTS auth.users), which is
-- live-proven. Account deletion keeps its promote path: the teams row still exists there.
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

  -- Whole-team teardown: the parent teams row is already gone (deleted earlier in this
  -- command; the cascade fires after). Nothing to preserve — allow without promoting,
  -- otherwise the promote UPDATE hits a sibling this same cascade is deleting (27000).
  IF NOT EXISTS (SELECT 1 FROM teams WHERE id = OLD.team_id) THEN
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

  -- Other members remain and the team itself survives → promote the longest-tenured.
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

COMMIT;
