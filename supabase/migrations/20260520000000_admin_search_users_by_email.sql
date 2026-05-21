-- ============================================================================
-- Admin email search RPC.
--
-- The Supabase auth admin API does not expose a server-side email search —
-- only paginated list_users(). The previous backend implementation fetched
-- the first page and filtered in Python, which silently missed any matching
-- user past the first page. This RPC scans auth.users.email directly and
-- returns the top N matches.
--
-- Security: SECURITY DEFINER lets the function read auth.users (which
-- regular roles can't). EXECUTE is REVOKEd from PUBLIC and granted ONLY
-- to service_role — so only the backend (which uses VITE_SUPABASE_SECRET_KEY
-- = the service role key) can call it. Admin enforcement happens at the
-- FastAPI layer via require_admin.
--
-- search_path is pinned to 'public', 'auth' so the function resolves
-- table names correctly regardless of the caller's session search_path
-- (same defensive pattern as the storage triggers in
-- 20260518010000_fix_storage_trigger_search_path.sql).
-- ============================================================================

CREATE OR REPLACE FUNCTION public.admin_search_users_by_email(
  p_search TEXT,
  p_limit  INT DEFAULT 10
)
RETURNS TABLE (id UUID, email TEXT, created_at TIMESTAMPTZ)
SECURITY DEFINER
SET search_path TO 'public', 'auth'
LANGUAGE sql
AS $$
  SELECT u.id, u.email::text, u.created_at
  FROM auth.users u
  WHERE u.email ILIKE '%' || p_search || '%'
  ORDER BY u.email
  LIMIT GREATEST(LEAST(p_limit, 100), 1);
$$;

REVOKE EXECUTE ON FUNCTION public.admin_search_users_by_email(TEXT, INT) FROM PUBLIC;
GRANT  EXECUTE ON FUNCTION public.admin_search_users_by_email(TEXT, INT) TO service_role;
