-- ============================================================================
-- DB-backed admin roles (companion to ADMIN_EMAILS env-var bootstrap).
--
-- profiles.is_admin = true grants the same access as being in ADMIN_EMAILS.
-- ADMIN_EMAILS stays the bootstrap path; day-to-day admin grants happen via
-- the /admin/users UI which writes this column via service-role.
--
-- SECURITY NOTE: We intentionally do NOT add a WITH CHECK RLS guard on
-- this column. The only writer is the backend's service-role client (via
-- AdminService.promote_user / demote_user), which bypasses RLS regardless
-- of policy. The frontend has no path to write profiles.is_admin. If a
-- user-scoped write path is ever added, add a BEFORE UPDATE trigger that
-- rejects is_admin changes when session_user <> 'service_role'.
-- ============================================================================

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

-- Partial index on is_admin (not id): id is already PK-indexed, so per-user
-- lookups don't benefit from an extra index. The "any admin exists?" check
-- (SELECT 1 FROM profiles WHERE is_admin = true LIMIT 1) does benefit.
CREATE INDEX IF NOT EXISTS idx_profiles_is_admin_true
  ON public.profiles (is_admin)
  WHERE is_admin = true;
