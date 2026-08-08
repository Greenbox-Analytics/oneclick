-- supabase/migrations/20260720000000_protect_profiles_is_admin.sql
-- ============================================================================
-- Close the profiles.is_admin self-escalation hole.
--
-- THE VULNERABILITY: profiles has "Users can update own profile." FOR UPDATE
-- USING (auth.uid() = id) with no column-level control (RLS is row-level by
-- design — it cannot restrict WHICH columns a permitted row-write touches).
-- The browser talks to PostgREST with a generic client, so any signed-in user
-- can PATCH their own row with {"is_admin": true} and receive the admin
-- console, implicit-Pro entitlements, and the admin bypass of every credit
-- gate. 20260517000000_add_profiles_is_admin.sql documented this exact risk
-- and prescribed this exact fix ("add a BEFORE UPDATE trigger that rejects
-- is_admin changes when the writer isn't service_role") but never shipped it.
--
-- THE FIX: a BEFORE INSERT OR UPDATE trigger that rejects privileged-column
-- changes unless the writer is a backend/system role. Trigger (not column
-- GRANTs) because it is surgical: every existing user-editable profile flow
-- (username, avatar, phone, billing_context_org_id when licensing lands, ...)
-- keeps working untouched, and the guard names its intent in the error.
--
-- Roles allowed to write is_admin:
--   service_role        — the backend's Supabase client (AdminService promote/demote)
--   postgres            — migrations / psql maintenance
--   supabase_admin      — Supabase platform operations
--   supabase_auth_admin — GoTrue (signup trigger inserts profile rows)
-- PostgREST requests from browsers run as 'authenticated' / 'anon' and are
-- rejected when they touch is_admin. INSERT is guarded too: the row-level
-- INSERT policy (auth.uid() = id) would otherwise let a user whose profile
-- row is missing insert one with is_admin = true.
--
-- If more privileged columns are ever added to profiles (is_staff, tier
-- overrides, etc.), ADD THEM TO THIS TRIGGER. profiles is a user-writable
-- table; any column users must not control needs an explicit guard here.
--
-- SECURITY DEFINER is deliberately NOT used — the function runs with the
-- caller's privileges and only inspects/raises. search_path is pinned anyway
-- per repo convention (lesson: 20260517010000_fix_signup_trigger_search_path).
-- ============================================================================

CREATE OR REPLACE FUNCTION public.protect_profiles_privileged_columns()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path TO 'public'
AS $$
BEGIN
  IF current_user IN ('service_role', 'postgres', 'supabase_admin', 'supabase_auth_admin') THEN
    RETURN NEW;
  END IF;

  IF TG_OP = 'INSERT' THEN
    IF COALESCE(NEW.is_admin, false) THEN
      RAISE EXCEPTION 'profiles.is_admin can only be set by the backend'
        USING ERRCODE = 'insufficient_privilege';
    END IF;
  ELSIF NEW.is_admin IS DISTINCT FROM OLD.is_admin THEN
    RAISE EXCEPTION 'profiles.is_admin can only be changed by the backend'
      USING ERRCODE = 'insufficient_privilege';
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS protect_profiles_privileged_columns ON public.profiles;
CREATE TRIGGER protect_profiles_privileged_columns
  BEFORE INSERT OR UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.protect_profiles_privileged_columns();

-- ============================================================================
-- Post-apply verification (run manually; both must fail with
-- 'insufficient_privilege' when executed as an authenticated user):
--   PATCH /rest/v1/profiles?id=eq.<own-id>  {"is_admin": true}
--   (and as a user with no profile row) POST /rest/v1/profiles
--     {"id": "<own-id>", "is_admin": true}
-- Backend regression check: admin promote/demote via /admin/users still works
-- (service_role client passes the role gate).
-- ============================================================================
