-- ============================================================================
-- Fix: user deletion fails with "permission denied for table
-- royalty_calculations" (SQLSTATE 42501).
--
-- Migration 20260518000000 added a FK royalty_calculations.user_id ->
-- auth.users(id) ON DELETE CASCADE. Cascades use the calling role's
-- table-level privileges, and GoTrue's supabase_auth_admin didn't have
-- DELETE on this table — it's the only cascade target without it (other
-- chains via artists/projects already have the grant by virtue of being
-- inherited from public-schema defaults).
--
-- Fix: grant DELETE on royalty_calculations to supabase_auth_admin so
-- auth.users delete can cascade through.
-- ============================================================================

GRANT DELETE ON public.royalty_calculations TO supabase_auth_admin;
