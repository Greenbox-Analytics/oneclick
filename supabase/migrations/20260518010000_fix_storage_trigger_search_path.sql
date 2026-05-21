-- ============================================================================
-- Fix: user deletion fails with "relation 'projects' does not exist"
-- (SQLSTATE 42P01).
--
-- Root cause (same class as 20260517010000_fix_signup_trigger_search_path.sql):
-- the auth cascade chain auth.users → public.artists → public.projects
-- → public.project_files → public.audio_files fires several trigger
-- functions during the cascade. Those functions were created without a
-- `SET search_path` clause and reference public-schema tables unqualified
-- (`FROM projects`, `FROM artists`, etc).
--
-- GoTrue calls auth.admin.delete_user() as the supabase_auth_admin role.
-- That role's session search_path is `auth`. When the cascade fires
-- a trigger function with no search_path of its own, the function
-- inherits the caller's search_path. Unqualified `projects` resolves
-- to `auth.projects` (which doesn't exist) → 42P01 → whole delete
-- rolls back.
--
-- Fix: ALTER each affected function to pin its search_path to public.
-- This is non-destructive — the function bodies are unchanged, just the
-- per-function GUC is set. Matches the pattern used to fix the signup
-- triggers yesterday.
--
-- Function inventory (origin migration):
--   * 20260509000001_subscription_foundation.sql
--       recalc_user_storage, _project_owner_for_pf, _audio_owner_for_af,
--       trigger_storage_pf_change, trigger_storage_af_change
--   * 20260510000002_subproject3_gating.sql
--       effective_storage_cap, enforce_storage_cap_pf, enforce_storage_cap_af
--   * 20260412000001_fix_owner_deletion_trigger.sql
--       prevent_owner_deletion
-- ============================================================================

ALTER FUNCTION public.recalc_user_storage(UUID)           SET search_path TO 'public';
ALTER FUNCTION public._project_owner_for_pf(UUID)         SET search_path TO 'public';
ALTER FUNCTION public._audio_owner_for_af(UUID)           SET search_path TO 'public';
ALTER FUNCTION public.trigger_storage_pf_change()         SET search_path TO 'public';
ALTER FUNCTION public.trigger_storage_af_change()         SET search_path TO 'public';
ALTER FUNCTION public.effective_storage_cap(UUID)         SET search_path TO 'public';
ALTER FUNCTION public.enforce_storage_cap_pf()            SET search_path TO 'public';
ALTER FUNCTION public.enforce_storage_cap_af()            SET search_path TO 'public';
ALTER FUNCTION public.prevent_owner_deletion()            SET search_path TO 'public';
