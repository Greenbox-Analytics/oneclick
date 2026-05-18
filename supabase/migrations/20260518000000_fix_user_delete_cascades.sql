-- ============================================================================
-- Fix user deletion cascade
--
-- Problem: deleting a user from auth.users fails with "Database error deleting
-- user" because two FK constraints to auth.users(id) lack ON DELETE clauses,
-- and one large data table (royalty_calculations) has no FK at all so its
-- rows orphan instead of cascading.
--
-- Tables touched:
--   * public.project_members.invited_by         (was NO ACTION → SET NULL)
--   * public.pending_project_invites.invited_by (was NO ACTION → CASCADE)
--   * public.royalty_calculations.user_id       (had no FK → add with CASCADE)
--
-- See: src/backend/.../REFERENCES auth.users   for full FK inventory.
-- ============================================================================

-- 1. project_members.invited_by
-- Membership ROW should survive the inviter leaving (the workspace member is
-- still valid — they just lose provenance about who invited them).
ALTER TABLE public.project_members
  DROP CONSTRAINT IF EXISTS project_members_invited_by_fkey;
ALTER TABLE public.project_members
  ADD CONSTRAINT project_members_invited_by_fkey
    FOREIGN KEY (invited_by)
    REFERENCES auth.users(id)
    ON DELETE SET NULL;

-- 2. pending_project_invites.invited_by
-- A pending invite without a sender is meaningless; cascade-delete the invite
-- when the inviter is removed. invited_by is NOT NULL; CASCADE removes the
-- whole row so the constraint stays satisfied.
ALTER TABLE public.pending_project_invites
  DROP CONSTRAINT IF EXISTS pending_project_invites_invited_by_fkey;
ALTER TABLE public.pending_project_invites
  ADD CONSTRAINT pending_project_invites_invited_by_fkey
    FOREIGN KEY (invited_by)
    REFERENCES auth.users(id)
    ON DELETE CASCADE;

-- 3. royalty_calculations.user_id
-- Currently a bare UUID column ("keeping as UUID for flexibility" per the
-- 20260207 migration). That flexibility lets data orphan when a user is
-- deleted. Clean existing orphans first, then add the FK with CASCADE so
-- future deletes sweep them.
DELETE FROM public.royalty_calculations
  WHERE user_id IS NULL
     OR user_id NOT IN (SELECT id FROM auth.users);

ALTER TABLE public.royalty_calculations
  ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE public.royalty_calculations
  ADD CONSTRAINT royalty_calculations_user_id_fkey
    FOREIGN KEY (user_id)
    REFERENCES auth.users(id)
    ON DELETE CASCADE;
