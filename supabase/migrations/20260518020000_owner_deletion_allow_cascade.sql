-- ============================================================================
-- Fix: user deletion fails with "Cannot remove the project owner"
-- (SQLSTATE P0001) when the user owns at least one project.
--
-- The prevent_owner_deletion() BEFORE-DELETE trigger on public.project_members
-- previously had a single guard: allow the delete if the project row no
-- longer exists. That handles "DELETE FROM projects" cascading down to its
-- project_members rows. It does NOT handle "DELETE FROM auth.users"
-- cascading down via the project_members.user_id FK directly — the project
-- still exists at that moment.
--
-- Fix: use pg_trigger_depth() to detect cascade context. When this trigger
-- fires at depth > 1, it was invoked from inside another (parent) delete —
-- either an RI cascade or a parent user-defined trigger. In that case the
-- parent statement is the source of truth, so allow the owner row to be
-- removed. At depth = 1 (a top-level DELETE FROM project_members from the
-- application), keep the original protection.
--
-- Also pins search_path to public for the same reason as the prior fix
-- migration (functions called from auth.admin.delete_user inherit the
-- supabase_auth_admin role's search_path which doesn't include public).
-- ============================================================================

CREATE OR REPLACE FUNCTION public.prevent_owner_deletion()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  -- Cascade context: the parent delete already decided this row goes away.
  -- Don't second-guess. Includes:
  --   * auth.users delete → project_members.user_id CASCADE
  --   * auth.users delete → artists → projects → project_members.project_id CASCADE
  --   * Future cascades from any other table touching project_members
  IF pg_trigger_depth() > 1 THEN
    RETURN OLD;
  END IF;

  -- Top-level delete: enforce the original rule.
  -- Owner rows can only be deleted if the project itself is also gone.
  IF OLD.role = 'owner' AND EXISTS (
    SELECT 1 FROM public.projects WHERE id = OLD.project_id
  ) THEN
    RAISE EXCEPTION 'Cannot remove the project owner';
  END IF;
  RETURN OLD;
END;
$$;
