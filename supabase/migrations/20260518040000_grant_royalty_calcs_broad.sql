-- ============================================================================
-- Fix: user deletion still failing with "permission denied for table
-- royalty_calculations" after the narrow GRANT in 20260518030000.
--
-- Root cause: the cascade chain reaches royalty_calculation_contracts too
-- (via FK calculation_id ON DELETE CASCADE), and the role context for the
-- cascade isn't strictly supabase_auth_admin — the auth audit log shows
-- `actor_username: "service_role"`. Without certainty on which role the
-- cascade actually runs under, grant the full DML verb set to both
-- supabase_auth_admin AND service_role, on both the parent and child.
-- RLS still gates row-level access; table-level GRANTs are required for
-- cascade ops independently of RLS.
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON public.royalty_calculations
  TO supabase_auth_admin, service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.royalty_calculation_contracts
  TO supabase_auth_admin, service_role;
