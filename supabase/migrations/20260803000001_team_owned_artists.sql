-- supabase/migrations/20260803000001_team_owned_artists.sql
-- ============================================================================
-- Team-owned artist profiles. artists.team_id NULL = personal (today's
-- behaviour, untouched); NOT NULL = owned by that organization and visible to
-- every ACTIVE member.
--
-- ONE nullable column rather than a polymorphic owner_type/owner_id pair:
-- artists is the deepest table in the schema (ten tables cascade off it) and a
-- discriminator there is a much larger commitment than it looks. user_id stays
-- as the creator; when team_id is set it stops meaning "owner" -- which is
-- precisely why 20260803000002 has to rewrite the policies that read it that
-- way.
--
-- ON DELETE RESTRICT, not CASCADE: deleting an organization must never take a
-- roster with it. Orgs are archived (archived_at), never deleted, so this is a
-- tripwire rather than a workflow.
-- ============================================================================

BEGIN;

ALTER TABLE artists
  ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES organizations(id) ON DELETE RESTRICT,
  -- Transfer is money-adjacent: it changes who pays for an artist's AI usage
  -- and whose storage its files occupy. Stamped on the row itself -- a separate
  -- audit table would be a second thing to keep consistent for two columns.
  ADD COLUMN IF NOT EXISTS transferred_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS transferred_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- Partial index: every team-scoped read filters on this, personal reads never do.
CREATE INDEX IF NOT EXISTS idx_artists_team_id ON artists (team_id) WHERE team_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- artists.user_id was ON DELETE CASCADE (20260112000000). That was correct
-- while every artist was personal: deleting the account deleted the roster.
-- With team ownership it is a data-loss bug -- the creator of a team artist
-- deleting their own account would take the LABEL's artist, and the ten tables
-- that cascade off it, with them.
--
-- SET NULL instead. For a team artist user_id is already only a creator stamp
-- (team_id is the owner), so losing it costs nothing. For a personal artist a
-- NULL user_id fails every policy closed, which is the right outcome for a row
-- whose owner no longer exists -- and account deletion enumerates and removes
-- those rows explicitly first (users/account_deletion_service.py), so this is a
-- backstop, not the cleanup path.
-- ---------------------------------------------------------------------------
ALTER TABLE artists DROP CONSTRAINT IF EXISTS artists_user_id_fkey;
ALTER TABLE artists
  ADD CONSTRAINT artists_user_id_fkey
  FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL;

-- ---------------------------------------------------------------------------
-- can_access_artist -- the ONE definition of artist authority. Every policy in
-- 20260803000002 calls this instead of repeating the join, so access can never
-- drift between the ten tables that hang off an artist.
--
-- p_require_admin is a third argument rather than a second function because the
-- admin variant is this body plus one join clause; two near-identical
-- SECURITY DEFINER functions are two things to keep in step.
--
-- SECURITY DEFINER because it reads org_members, whose own RLS would otherwise
-- recurse when this is called from inside an artists policy.
--
-- Org status is NOT checked, only archived_at: a 'pending' org (created, not
-- yet paid up) is a team that is still being set up, and its members should be
-- able to build the roster they are about to pay for. Billing has its own
-- status='active' gate in resolve_billing_org_for_project -- access and payment
-- are different questions.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.can_access_artist(
  p_artist_id UUID,
  p_user_id UUID,
  p_require_admin BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1
      FROM artists a
      LEFT JOIN organizations o ON o.id = a.team_id
      LEFT JOIN org_members m
             ON m.org_id = a.team_id
            AND m.user_id = p_user_id
            AND m.status = 'active'
            AND (NOT p_require_admin OR m.role = 'admin')
     WHERE a.id = p_artist_id
       AND (
         -- personal: the creator, and only while it is not team-owned
         (a.team_id IS NULL AND a.user_id = p_user_id)
         -- team: an active member (admin, when required) of a live org
         OR (a.team_id IS NOT NULL AND m.id IS NOT NULL AND o.archived_at IS NULL)
       )
  );
$$;

-- ---------------------------------------------------------------------------
-- team_id is set by ONE code path: the transfer endpoint, which runs on the
-- service-role client (auth.uid() IS NULL, RLS bypassed). Anything holding an
-- end-user JWT is refused here regardless of which policy let the UPDATE
-- through -- a policy WITH CHECK cannot see OLD, so it cannot tell "renamed the
-- artist" from "moved it to another team" or "yanked it back out of one".
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.artists_lock_team_id()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  IF NEW.team_id IS DISTINCT FROM OLD.team_id AND auth.uid() IS NOT NULL THEN
    RAISE EXCEPTION 'artists.team_id is changed by the transfer endpoint, not directly'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS artists_lock_team_id_trg ON artists;
CREATE TRIGGER artists_lock_team_id_trg
  BEFORE UPDATE OF team_id ON artists
  FOR EACH ROW EXECUTE FUNCTION public.artists_lock_team_id();

REVOKE EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) TO authenticated, service_role;

COMMIT;
