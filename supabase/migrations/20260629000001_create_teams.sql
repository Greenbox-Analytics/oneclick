-- Team Boards Phase 1 — team membership foundation.
-- Standalone teams that own many boards (spec §4.1). Roles: admin | member.
-- Behavioral lifecycle triggers (auto-promote / removal cleanup) land in Phase 2.
-- Reuses the repo-canonical public.update_updated_at_column() (20251120060318).
-- Wrapped in an explicit transaction so the whole migration applies atomically.
BEGIN;

-- ============================================================
-- Tables
-- ============================================================
CREATE TABLE teams (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  description TEXT,
  created_by  UUID REFERENCES auth.users(id) ON DELETE SET NULL,  -- informational; team survives via auto-promote (Phase 2)
  archived_at TIMESTAMPTZ,                                          -- soft-delete: NULL = active
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE team_members (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  team_id    UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role       TEXT NOT NULL CHECK (role IN ('admin', 'member')),
  invited_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(team_id, user_id)
);

CREATE TABLE pending_team_invites (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  team_id    UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
  email      TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('admin', 'member')) DEFAULT 'member',
  token      UUID NOT NULL DEFAULT gen_random_uuid(),
  status     TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'declined')) DEFAULT 'pending',
  invited_by UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days')
);

CREATE INDEX idx_team_members_team_id ON team_members(team_id);
CREATE INDEX idx_team_members_user_id ON team_members(user_id);
-- Case-insensitive uniqueness so the §6 re-invite dedup and the LOWER(email) lookup/RLS agree.
CREATE UNIQUE INDEX uq_pending_team_invites_team_email ON pending_team_invites (team_id, LOWER(email));
CREATE INDEX idx_pending_team_invites_email ON pending_team_invites (LOWER(email));

-- ============================================================
-- updated_at triggers (reuse the canonical function)
-- ============================================================
CREATE TRIGGER teams_updated_at
  BEFORE UPDATE ON teams
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER team_members_updated_at
  BEFORE UPDATE ON team_members
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- ============================================================
-- Membership-check helpers (SECURITY DEFINER to avoid recursive RLS).
-- Arg order (p_user_id, p_team_id) matches the Python helpers in teams/authz.py.
-- ============================================================
CREATE OR REPLACE FUNCTION is_team_member(p_user_id UUID, p_team_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1 FROM team_members
    WHERE team_id = p_team_id AND user_id = p_user_id
  );
$$;

CREATE OR REPLACE FUNCTION is_team_admin(p_user_id UUID, p_team_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1 FROM team_members
    WHERE team_id = p_team_id AND user_id = p_user_id AND role = 'admin'
  );
$$;

-- ============================================================
-- Row Level Security
-- ============================================================
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE pending_team_invites ENABLE ROW LEVEL SECURITY;

-- teams: members can read; admins can update; anyone authenticated can create.
CREATE POLICY "teams_select_members" ON teams
  FOR SELECT USING (is_team_member(auth.uid(), id));
CREATE POLICY "teams_insert_authenticated" ON teams
  FOR INSERT WITH CHECK (auth.uid() IS NOT NULL);
CREATE POLICY "teams_update_admins" ON teams
  FOR UPDATE USING (is_team_admin(auth.uid(), id));

-- team_members: members read the roster; admins write; a member can remove themselves.
CREATE POLICY "team_members_select_members" ON team_members
  FOR SELECT USING (is_team_member(auth.uid(), team_id));
CREATE POLICY "team_members_insert_admins" ON team_members
  FOR INSERT WITH CHECK (is_team_admin(auth.uid(), team_id));
CREATE POLICY "team_members_update_admins" ON team_members
  FOR UPDATE USING (is_team_admin(auth.uid(), team_id));
CREATE POLICY "team_members_delete_admin_or_self" ON team_members
  FOR DELETE USING (is_team_admin(auth.uid(), team_id) OR user_id = auth.uid());

-- pending_team_invites: admins manage; an invitee can see their own invite by email.
CREATE POLICY "pending_team_invites_admins" ON pending_team_invites
  FOR ALL USING (is_team_admin(auth.uid(), team_id))
  WITH CHECK (is_team_admin(auth.uid(), team_id));
CREATE POLICY "pending_team_invites_select_own_email" ON pending_team_invites
  FOR SELECT USING (LOWER(email) = LOWER(COALESCE(auth.jwt() ->> 'email', '')));

COMMIT;
