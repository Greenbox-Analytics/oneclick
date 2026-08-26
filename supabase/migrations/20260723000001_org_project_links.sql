-- supabase/migrations/20260723000001_org_project_links.sql
-- ============================================================================
-- Licensing Phase C — org-linked project permissions (Task 1: schema).
--
-- Spec: docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §6
-- Plan: docs/superpowers/plans/2026-07-22-licensing-phase-c-permissions.md Task 1
--
-- Load-bearing rules restated (full numbered list lives at the top of the
-- plan; only the ones this migration is directly responsible for):
--  1. Linking = consent. Only the project OWNER (holding an active seat in
--     that org) creates or removes a link — enforced in the service layer
--     (Task 2); this migration only shapes the table so that invariant is
--     representable (one row per project, owner-readable).
--  2. Provenance is stamped on fresh INSERT only. project_members.org_id
--     (added below) is NULL for organic memberships and set to the granting
--     org's id ONLY the first time an org admin grants access to a seat with
--     no existing row — never overwritten on an already-organic row. That
--     stamping is a service-layer behavior (Task 3); this column is its
--     storage.
--  3. Revocation is subtractive and provenance-scoped: services delete only
--     project_members rows where org_id = the revoking org. Organic rows
--     (org_id IS NULL) and other orgs' rows are never touched by one org's
--     revocation. The owner row can't carry an org_id in practice (owners
--     aren't org-granted), so it is never a revocation target either way.
--  8. ONE org per project: org_project_links is UNIQUE(project_id) — see the
--     inline comment on that constraint for the full rationale.
--
-- OPERATOR NOTE: apply this AFTER 20260722000001_org_members_email.sql.
-- This migration is WRITTEN ONLY — never run it from this task.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. org_project_links — owner-consented link from a project to the ONE org
-- that bills and administers access to it (spec §6).
-- ---------------------------------------------------------------------------
CREATE TABLE org_project_links (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  -- Nullable: the linking user may later be deleted without destroying the
  -- link itself (same ON DELETE SET NULL precedent as
  -- project_members.invited_by in 20260518000000_fix_user_delete_cascades —
  -- provenance of WHO linked it is nice-to-have, not load-bearing).
  linked_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- Rule 8 (round 2): ONE org per project. This UNIQUE constraint is the
  -- entire enforcement mechanism for "a project links to at most one org" —
  -- a money decision (which org's pool/caps a project's work draws against)
  -- must not hide in a read-time tie-breaker (e.g. "oldest link wins" or
  -- resolving across multiple linked orgs at query time). The spec's own
  -- language is singular ("link their project to THE org"). Widening this
  -- later (allowing multiple links) is a one-line migration; narrowing it
  -- after real link rows exist across multiple orgs is not — so the
  -- constraint goes in now, before any data can violate the singular model.
  -- Named explicitly (rather than relying on Postgres's implicit
  -- "<table>_<column>_key" default) so the COMMENT below has a stable target.
  CONSTRAINT org_project_links_project_id_key UNIQUE(project_id)
);

COMMENT ON CONSTRAINT org_project_links_project_id_key ON org_project_links IS
  'Rule 8: one org per project, by design — a money decision (whose pool/caps '
  'this project''s work bills against) must not be resolved at read time via a '
  'tie-breaker across multiple links. Widening to allow multiple orgs later is '
  'easy; narrowing after multi-link data exists is not, so the singular model '
  'is enforced from the start.';

-- No separate index on project_id: the UNIQUE(project_id) constraint above
-- already creates a unique btree index on that column, which Postgres uses
-- for both the uniqueness check and ordinary lookups by project_id (e.g. the
-- owner-link and unlink paths in Task 2). A duplicate non-unique index would
-- be redundant.
CREATE INDEX idx_org_project_links_org_id ON org_project_links(org_id);

-- ---------------------------------------------------------------------------
-- 2. project_members.org_id — provenance column (rule 2). NULL = organic
-- membership (the user joined/was invited independently of any org). A
-- non-NULL value records which org's admin granted this row via Task 3's
-- membership endpoints, stamped ONLY on fresh INSERT — never on an update to
-- an already-organic row (rule 2: an org "add" on a seat that already holds
-- organic access is a no-op, otherwise offboarding would delete access the
-- member earned independently of the org).
-- ---------------------------------------------------------------------------
ALTER TABLE project_members
  ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id) ON DELETE SET NULL;

COMMENT ON COLUMN project_members.org_id IS
  'Provenance, not access control: NULL = organic membership (untouched by any '
  'org). Non-NULL = this row was granted by that org''s admin (stamped once, on '
  'fresh INSERT only — rule 2). ON DELETE SET NULL (not CASCADE) is deliberate: '
  'if the granting org itself is deleted, this row degrades to organic instead '
  'of silently disappearing along with the member''s project access. Actual '
  'revocation (rule 3 — deleting exactly the rows one org granted, on offboard/'
  'unlink/archive) is the SERVICE''s job with this column as its filter '
  '(orgs/projects.py revoke_org_granted_memberships, Task 4); this FK is only '
  'a referential backstop against a dangling org_id, not the revocation '
  'mechanism itself.';

-- ---------------------------------------------------------------------------
-- 3. Row Level Security on org_project_links. SELECT-only: readable by (a)
-- members of the linked org, so org admins can see which projects are linked
-- to them, and (b) the project's owner, so the owner who created the link
-- can see/manage it. No write policies — links are created/removed only by
-- the backend's service-role client (Task 2's owner-consent + active-seat
-- checks ARE the authz, per the repo's authz model; see
-- 20260721000001_licensing_core.sql section 7 for the same posture on
-- organizations/org_members).
-- ---------------------------------------------------------------------------
ALTER TABLE org_project_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_project_links_select_org_members" ON org_project_links
  FOR SELECT USING (is_org_member(auth.uid(), org_id));

-- Project-owner predicate reused EXACTLY from the project_members RLS idiom
-- (get_project_role, the SECURITY DEFINER helper added in
-- 20260403000009_fix_rls_recursion.sql to avoid recursive RLS) — narrowed to
-- 'owner' only, since org admins already have their own policy above and
-- this branch exists so the OWNER (who alone can create/remove the link,
-- rule 1) can always see it regardless of org membership.
CREATE POLICY "org_project_links_select_project_owner" ON org_project_links
  FOR SELECT USING (get_project_role(project_id) = 'owner');

COMMIT;
