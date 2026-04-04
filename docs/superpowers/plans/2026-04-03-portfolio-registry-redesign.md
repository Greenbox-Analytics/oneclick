# Portfolio + Registry Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Portfolio as a project grid, introduce a tabbed Project Detail page, redesign Registry as an ownership tracking dashboard, and add dual-layer access control (project members + work collaborators).

**Architecture:** New Project Detail page (`/projects/{projectId}`) becomes the central hub with 5 tabs (Works, Files, Audio, Members, Settings). Portfolio simplifies to a card grid. Registry becomes read-only tracking. Backend gains project membership CRUD, work-file/audio linking, and enhanced invite-with-stakes. All access controlled via RLS on `project_members` and `registry_collaborators`.

**Tech Stack:** React 18 + TypeScript, FastAPI, Supabase (PostgreSQL + RLS), TanStack React Query, Radix/shadcn UI, Tailwind CSS, Resend (email).

**Spec:** `docs/superpowers/specs/2026-04-03-portfolio-registry-redesign.md`

---

### Task 1: Database Migrations

**Goal:** Create all migration files for the new tables, columns, triggers, and RLS policies. User runs these manually.

**Files:**
- Create: `supabase/migrations/20260403000001_create_project_members.sql`
- Create: `supabase/migrations/20260403000002_create_pending_project_invites.sql`
- Create: `supabase/migrations/20260403000003_create_work_files.sql`
- Create: `supabase/migrations/20260403000004_create_work_audio_links.sql`
- Create: `supabase/migrations/20260403000005_add_content_hash_to_project_files.sql`
- Create: `supabase/migrations/20260403000006_update_works_registry.sql`
- Create: `supabase/migrations/20260403000007_simplify_collaborator_status.sql`
- Create: `supabase/migrations/20260403000008_update_rls_policies.sql`
- Modify: `src/integrations/supabase/types.ts` — add types for new tables

**Acceptance Criteria:**
- [ ] `project_members` table created with role CHECK, UNIQUE(project_id, user_id)
- [ ] Auto-owner trigger (SECURITY DEFINER) inserts owner row on project creation
- [ ] Owner protection: prevent deletion trigger + prevent role change trigger + partial unique index
- [ ] `pending_project_invites` table with auth.users INSERT trigger (SECURITY DEFINER)
- [ ] `work_files` join table with UNIQUE(work_id, file_id)
- [ ] `work_audio_links` join table with UNIQUE(work_id, audio_file_id)
- [ ] `content_hash` column added to `project_files`
- [ ] `custom_work_type` column + 'other' added to work_type CHECK
- [ ] Disputed status removed from works_registry and registry_collaborators (with data migration)
- [ ] `dispute_reason` column dropped
- [ ] All RLS policies from spec applied (email-based invite visibility, project member access, confirmed-only file access, join table SELECT, write policies, DELETE for works)
- [ ] TypeScript types updated for new tables

**Verify:** All migration files exist with correct SQL. User runs `supabase db push` or applies manually.

**Steps:**

- [ ] **Step 1: Create project_members migration**

Write `supabase/migrations/20260403000001_create_project_members.sql`:

```sql
-- project_members: dual-layer access control (project-level roles)
CREATE TABLE project_members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
  invited_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(project_id, user_id)
);

-- Ensure at most one owner per project
CREATE UNIQUE INDEX one_owner_per_project ON project_members (project_id) WHERE role = 'owner';

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_project_members_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER project_members_updated_at
  BEFORE UPDATE ON project_members
  FOR EACH ROW EXECUTE FUNCTION update_project_members_updated_at();

-- Prevent owner deletion
CREATE OR REPLACE FUNCTION prevent_owner_deletion()
RETURNS TRIGGER AS $$
BEGIN
  IF OLD.role = 'owner' THEN
    RAISE EXCEPTION 'Cannot remove the project owner';
  END IF;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER prevent_owner_deletion_trigger
  BEFORE DELETE ON project_members
  FOR EACH ROW EXECUTE FUNCTION prevent_owner_deletion();

-- Prevent owner role changes and promotion to owner
CREATE OR REPLACE FUNCTION protect_owner_role()
RETURNS TRIGGER AS $$
BEGIN
  IF OLD.role = 'owner' AND NEW.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot change the project owner role';
  END IF;
  IF NEW.role = 'owner' AND OLD.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot promote to owner';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER protect_owner_role_trigger
  BEFORE UPDATE ON project_members
  FOR EACH ROW EXECUTE FUNCTION protect_owner_role();

-- Auto-create owner when project is created (SECURITY DEFINER to bypass RLS)
CREATE OR REPLACE FUNCTION auto_create_project_owner()
RETURNS TRIGGER SECURITY DEFINER AS $$
BEGIN
  IF auth.uid() IS NOT NULL THEN
    INSERT INTO project_members (project_id, user_id, role)
    VALUES (NEW.id, auth.uid(), 'owner');
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_create_project_owner_trigger
  AFTER INSERT ON projects
  FOR EACH ROW EXECUTE FUNCTION auto_create_project_owner();

-- RLS
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY "project_members_select_members" ON project_members
  FOR SELECT USING (
    project_id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );

CREATE POLICY "project_members_insert_admins" ON project_members
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_update_admins" ON project_members
  FOR UPDATE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_delete_admins" ON project_members
  FOR DELETE USING (
    (project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    ))
    OR
    (user_id = auth.uid() AND role != 'owner')
  );

-- Indexes
CREATE INDEX idx_project_members_project_id ON project_members(project_id);
CREATE INDEX idx_project_members_user_id ON project_members(user_id);
```

- [ ] **Step 2: Create pending_project_invites migration**

Write `supabase/migrations/20260403000002_create_pending_project_invites.sql`:

```sql
-- For inviting users who don't have an account yet
CREATE TABLE pending_project_invites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
  invited_by UUID NOT NULL REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ DEFAULT now() + interval '7 days',
  UNIQUE(project_id, email)
);

ALTER TABLE pending_project_invites ENABLE ROW LEVEL SECURITY;

-- Admins+ on the project can manage pending invites
CREATE POLICY "pending_invites_select" ON pending_project_invites
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "pending_invites_insert" ON pending_project_invites
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "pending_invites_delete" ON pending_project_invites
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

-- On signup: convert pending invites to project_members
CREATE OR REPLACE FUNCTION process_pending_project_invites()
RETURNS TRIGGER SECURITY DEFINER AS $$
BEGIN
  INSERT INTO project_members (project_id, user_id, role, invited_by)
  SELECT pi.project_id, NEW.id, pi.role, pi.invited_by
  FROM pending_project_invites pi
  WHERE LOWER(pi.email) = LOWER(NEW.email)
    AND pi.expires_at > now();

  DELETE FROM pending_project_invites
  WHERE LOWER(email) = LOWER(NEW.email);

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER process_pending_invites_on_signup
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION process_pending_project_invites();
```

- [ ] **Step 3: Create work_files migration**

Write `supabase/migrations/20260403000003_create_work_files.sql`:

```sql
CREATE TABLE work_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  file_id UUID NOT NULL REFERENCES project_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, file_id)
);

ALTER TABLE work_files ENABLE ROW LEVEL SECURITY;

-- SELECT: project members OR confirmed collaborators
CREATE POLICY "work_files_select" ON work_files
  FOR SELECT USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
    OR
    work_id IN (
      SELECT work_id FROM registry_collaborators
      WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
    )
  );

-- INSERT/DELETE: editors+
CREATE POLICY "work_files_insert_editors" ON work_files
  FOR INSERT WITH CHECK (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE POLICY "work_files_delete_editors" ON work_files
  FOR DELETE USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE INDEX idx_work_files_work_id ON work_files(work_id);
CREATE INDEX idx_work_files_file_id ON work_files(file_id);
```

- [ ] **Step 4: Create work_audio_links migration**

Write `supabase/migrations/20260403000004_create_work_audio_links.sql` — same pattern as work_files but for audio:

```sql
CREATE TABLE work_audio_links (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  audio_file_id UUID NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, audio_file_id)
);

ALTER TABLE work_audio_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "work_audio_links_select" ON work_audio_links
  FOR SELECT USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
    OR
    work_id IN (
      SELECT work_id FROM registry_collaborators
      WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
    )
  );

CREATE POLICY "work_audio_links_insert_editors" ON work_audio_links
  FOR INSERT WITH CHECK (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE POLICY "work_audio_links_delete_editors" ON work_audio_links
  FOR DELETE USING (
    work_id IN (
      SELECT id FROM works_registry WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE INDEX idx_work_audio_links_work_id ON work_audio_links(work_id);
CREATE INDEX idx_work_audio_links_audio_file_id ON work_audio_links(audio_file_id);
```

- [ ] **Step 5: Add content_hash to project_files**

Write `supabase/migrations/20260403000005_add_content_hash_to_project_files.sql`:

```sql
ALTER TABLE project_files ADD COLUMN content_hash TEXT;
CREATE INDEX idx_project_files_content_hash ON project_files(content_hash) WHERE content_hash IS NOT NULL;
```

- [ ] **Step 6: Update works_registry**

Write `supabase/migrations/20260403000006_update_works_registry.sql`:

```sql
-- Add custom work type column
ALTER TABLE works_registry ADD COLUMN custom_work_type TEXT;

-- Data migration: move disputed works to draft BEFORE changing constraint
UPDATE works_registry SET status = 'draft' WHERE status = 'disputed';

-- Update work_type CHECK to include 'other'
ALTER TABLE works_registry DROP CONSTRAINT IF EXISTS works_registry_work_type_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_work_type_check
  CHECK (work_type IN ('single', 'ep_track', 'album_track', 'composition', 'other'));

-- Update status CHECK to remove 'disputed'
ALTER TABLE works_registry DROP CONSTRAINT IF EXISTS works_registry_status_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_status_check
  CHECK (status IN ('draft', 'pending_approval', 'registered'));
```

- [ ] **Step 7: Simplify collaborator status**

Write `supabase/migrations/20260403000007_simplify_collaborator_status.sql`:

```sql
-- Data migration: move disputed collaborators to declined
UPDATE registry_collaborators SET status = 'declined' WHERE status = 'disputed';

-- Update status CHECK
ALTER TABLE registry_collaborators DROP CONSTRAINT IF EXISTS registry_collaborators_status_check;
ALTER TABLE registry_collaborators ADD CONSTRAINT registry_collaborators_status_check
  CHECK (status IN ('invited', 'confirmed', 'declined', 'revoked'));

-- Drop dispute_reason column
ALTER TABLE registry_collaborators DROP COLUMN IF EXISTS dispute_reason;
```

- [ ] **Step 8: Update RLS policies**

Write `supabase/migrations/20260403000008_update_rls_policies.sql`:

```sql
-- 1. Projects table: members can read
CREATE POLICY "projects_select_via_member" ON projects
  FOR SELECT USING (
    id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );

-- 2. Invite visibility by email (case-insensitive, before claim)
-- Drop existing select policies on registry_collaborators first, then recreate
DROP POLICY IF EXISTS "Collaborator can read own invitations" ON registry_collaborators;

CREATE POLICY "collaborators_select_by_email_or_id" ON registry_collaborators
  FOR SELECT USING (
    LOWER(email) = LOWER((SELECT email FROM auth.users WHERE id = auth.uid()))
    OR collaborator_user_id = auth.uid()
    OR invited_by = auth.uid()
  );

-- 3. File access via work collaboration (confirmed only)
CREATE POLICY "project_files_select_via_work_collab" ON project_files
  FOR SELECT USING (
    id IN (
      SELECT file_id FROM work_files WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );

-- 4. Audio access via work collaboration (confirmed only)
CREATE POLICY "audio_files_select_via_work_collab" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM work_audio_links WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );

-- 5. Project member access to works
CREATE POLICY "works_select_via_project_member" ON works_registry
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- 6. Project member access to files
CREATE POLICY "project_files_select_via_project_member" ON project_files
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- 7. Project member access to audio (via project_audio_links)
CREATE POLICY "audio_files_select_via_project_member" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM project_audio_links WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
  );

-- 8. Editors+ can create/update works
CREATE POLICY "works_insert_editors" ON works_registry
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

CREATE POLICY "works_update_editors" ON works_registry
  FOR UPDATE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

-- 9. Owner/admin can delete works
CREATE POLICY "works_delete_owner_admin" ON works_registry
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );
```

- [ ] **Step 9: Update TypeScript types**

Add new table types to `src/integrations/supabase/types.ts`. Add interfaces for `project_members`, `pending_project_invites`, `work_files`, and `work_audio_links` following the existing Row/Insert/Update pattern used by other tables in the file. Also add `content_hash: string | null` to `project_files.Row` and `custom_work_type: string | null` to `works_registry.Row`.

- [ ] **Step 10: Commit**

```bash
git add supabase/migrations/20260403*.sql src/integrations/supabase/types.ts
git commit -m "feat: add database migrations for portfolio/registry redesign

8 migration files: project_members, pending_project_invites,
work_files, work_audio_links, content_hash, works_registry updates,
collaborator status simplification, and RLS policy updates."
```

---

### Task 2: Backend — Project Members Service

**Goal:** Build the backend CRUD endpoints for managing project members and pending invites.

**Files:**
- Create: `src/backend/projects/__init__.py`
- Create: `src/backend/projects/models.py`
- Create: `src/backend/projects/service.py`
- Create: `src/backend/projects/router.py`
- Modify: `src/backend/main.py` — register new router

**Acceptance Criteria:**
- [ ] `GET /projects/{projectId}/members` returns all members with role
- [ ] `POST /projects/{projectId}/members` adds member (admin+ only), auto-adds if account exists, creates pending invite if not
- [ ] `PUT /projects/{projectId}/members/{memberId}` updates role (admin+ only, cannot change owner)
- [ ] `DELETE /projects/{projectId}/members/{memberId}` removes member (admin+ or self-remove for non-owners)
- [ ] `GET /projects/{projectId}/pending-invites` lists pending invites
- [ ] `DELETE /projects/{projectId}/pending-invites/{inviteId}` cancels a pending invite
- [ ] Email sent via Resend for pending invites to non-existing users

**Verify:** `curl -X GET "http://localhost:8000/projects/{id}/members?user_id={uid}"` returns member list

**Steps:**

- [ ] **Step 1: Create models**

Write `src/backend/projects/__init__.py` (empty file) and `src/backend/projects/models.py`:

```python
from pydantic import BaseModel, EmailStr
from typing import Optional


class MemberAdd(BaseModel):
    email: EmailStr
    role: str  # admin, editor, viewer (not owner)


class MemberUpdate(BaseModel):
    role: str  # admin, editor, viewer (not owner)
```

- [ ] **Step 2: Create service layer**

Write `src/backend/projects/service.py`:

```python
from supabase import Client


async def get_members(db: Client, user_id: str, project_id: str):
    """List all members of a project. Caller must be a member."""
    result = (
        db.table("project_members")
        .select("*, auth_user:user_id(email)")
        .eq("project_id", project_id)
        .order("created_at")
        .execute()
    )
    return result.data or []


async def get_user_role(db: Client, user_id: str, project_id: str) -> str | None:
    """Get the caller's role on a project, or None if not a member."""
    result = (
        db.table("project_members")
        .select("role")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .maybeSingle()
        .execute()
    )
    return result.data["role"] if result.data else None


async def add_member(db: Client, user_id: str, project_id: str, email: str, role: str):
    """Add a member by email. Auto-adds if account exists, creates pending invite if not."""
    # Validate caller is admin+
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can add members")
    if role not in ("admin", "editor", "viewer"):
        raise ValueError("Invalid role")

    # Check if user exists
    existing = db.table("profiles").select("id, email").ilike("email", email).maybeSingle().execute()

    if existing.data:
        # User exists — add directly
        result = (
            db.table("project_members")
            .insert({
                "project_id": project_id,
                "user_id": existing.data["id"],
                "role": role,
                "invited_by": user_id,
            })
            .execute()
        )
        return {"type": "added", "member": result.data[0] if result.data else None}
    else:
        # User doesn't exist — create pending invite
        result = (
            db.table("pending_project_invites")
            .insert({
                "project_id": project_id,
                "email": email.lower(),
                "role": role,
                "invited_by": user_id,
            })
            .execute()
        )
        return {"type": "pending", "invite": result.data[0] if result.data else None}


async def update_member_role(db: Client, user_id: str, project_id: str, member_id: str, role: str):
    """Update a member's role. Caller must be admin+."""
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can change roles")
    if role not in ("admin", "editor", "viewer"):
        raise ValueError("Invalid role — cannot set to owner")

    result = (
        db.table("project_members")
        .update({"role": role})
        .eq("id", member_id)
        .eq("project_id", project_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def remove_member(db: Client, user_id: str, project_id: str, member_id: str):
    """Remove a member. Admin+ can remove others; non-owners can remove themselves."""
    # Get the target member
    target = (
        db.table("project_members")
        .select("*")
        .eq("id", member_id)
        .eq("project_id", project_id)
        .single()
        .execute()
    )
    if not target.data:
        raise ValueError("Member not found")

    is_self = target.data["user_id"] == user_id
    caller_role = await get_user_role(db, user_id, project_id)

    if target.data["role"] == "owner":
        raise PermissionError("Cannot remove the project owner")
    if not is_self and caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can remove other members")

    db.table("project_members").delete().eq("id", member_id).execute()
    return {"deleted": member_id}


async def get_pending_invites(db: Client, user_id: str, project_id: str):
    """List pending invites for non-existing users."""
    result = (
        db.table("pending_project_invites")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


async def delete_pending_invite(db: Client, user_id: str, project_id: str, invite_id: str):
    """Cancel a pending invite."""
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can cancel invites")
    db.table("pending_project_invites").delete().eq("id", invite_id).execute()
    return {"deleted": invite_id}
```

- [ ] **Step 3: Create router**

Write `src/backend/projects/router.py`:

```python
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from projects import service
from projects.models import MemberAdd, MemberUpdate

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("/{project_id}/members")
async def list_members(project_id: str, user_id: str = Query(...)):
    members = await service.get_members(_get_supabase(), user_id, project_id)
    return {"members": members}


@router.post("/{project_id}/members")
async def add_member(project_id: str, body: MemberAdd, user_id: str = Query(...)):
    try:
        result = await service.add_member(
            _get_supabase(), user_id, project_id, body.email, body.role
        )
        # If pending, send invitation email
        if result["type"] == "pending":
            try:
                from projects.emails import send_project_invite_email
                project = _get_supabase().table("projects").select("name").eq("id", project_id).single().execute()
                inviter = _get_supabase().table("profiles").select("full_name, email").eq("id", user_id).single().execute()
                send_project_invite_email(
                    recipient_email=body.email,
                    project_name=project.data["name"] if project.data else "Unknown",
                    inviter_name=inviter.data.get("full_name", "Someone") if inviter.data else "Someone",
                    role=body.role,
                )
            except Exception as e:
                print(f"Warning: Failed to send project invite email: {e}")
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{project_id}/members/{member_id}")
async def update_member(project_id: str, member_id: str, body: MemberUpdate, user_id: str = Query(...)):
    try:
        result = await service.update_member_role(
            _get_supabase(), user_id, project_id, member_id, body.role
        )
        return {"member": result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{project_id}/members/{member_id}")
async def remove_member(project_id: str, member_id: str, user_id: str = Query(...)):
    try:
        return await service.remove_member(_get_supabase(), user_id, project_id, member_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/pending-invites")
async def list_pending_invites(project_id: str, user_id: str = Query(...)):
    invites = await service.get_pending_invites(_get_supabase(), user_id, project_id)
    return {"invites": invites}


@router.delete("/{project_id}/pending-invites/{invite_id}")
async def cancel_pending_invite(project_id: str, invite_id: str, user_id: str = Query(...)):
    try:
        return await service.delete_pending_invite(_get_supabase(), user_id, project_id, invite_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
```

- [ ] **Step 4: Create project invite email**

Write `src/backend/projects/emails.py` following the pattern in `src/backend/registry/emails.py`:

```python
import html
import os
import resend


def send_project_invite_email(
    recipient_email: str,
    project_name: str,
    inviter_name: str,
    role: str,
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping project invite email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    safe_project = html.escape(project_name)
    safe_inviter = html.escape(inviter_name)
    safe_role = html.escape(role)

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <h1 style="color: #1a3a2a; font-size: 24px; text-align: center;">Msanii</h1>
      <p style="font-size: 16px; color: #333;">You've been invited to a project!</p>
      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has invited you as a <strong>{safe_role}</strong>
        on the project <strong>&ldquo;{safe_project}&rdquo;</strong>.
      </p>
      <p style="font-size: 14px; color: #888;">
        Sign up at <a href="{frontend_url}/auth">{frontend_url}</a> to get started.
        You'll automatically be added to the project once you create your account.
      </p>
    </div>
    """

    try:
        return resend.Emails.send({
            "from": from_address,
            "to": [recipient_email],
            "subject": f"{safe_inviter} invited you to \"{safe_project}\" on Msanii",
            "html": html_body,
        })
    except Exception as e:
        print(f"Warning: Failed to send project invite email: {e}")
        return None
```

- [ ] **Step 5: Register router in main.py**

Add to `src/backend/main.py` after the existing router imports:

```python
from projects.router import router as projects_router
```

And in the `include_router` section:

```python
app.include_router(projects_router, prefix="/projects", tags=["Projects"])
```

- [ ] **Step 6: Commit**

```bash
git add src/backend/projects/ src/backend/main.py
git commit -m "feat: add project members backend (CRUD + pending invites + email)"
```

---

### Task 3: Backend — Work Links & Enhanced Collaboration

**Goal:** Add endpoints for linking files/audio to works, the composite invite-with-stakes endpoint, decline, accept-from-dashboard, and auto-transition logic.

**Files:**
- Create: `src/backend/registry/work_links_service.py`
- Modify: `src/backend/registry/models.py` — add new models
- Modify: `src/backend/registry/service.py` — update confirm/revoke/submit logic
- Modify: `src/backend/registry/router.py` — add new routes
- Modify: `src/backend/registry/emails.py` — enhance invite email with stake info

**Acceptance Criteria:**
- [ ] `GET/POST/DELETE /works/{workId}/files` manages work-file links
- [ ] `GET/POST/DELETE /works/{workId}/audio` manages work-audio links
- [ ] `POST /registry/collaborators/invite-with-stakes` creates collaborator + stakes atomically
- [ ] `POST /registry/collaborators/{id}/decline` with email validation
- [ ] `POST /registry/collaborators/{id}/accept-from-dashboard` with email validation, claim + confirm atomic
- [ ] `GET /registry/collaborators/my-invites` returns invites by email match
- [ ] Confirm endpoint auto-transitions work to `registered` when all confirmed
- [ ] Revoke endpoint deletes associated stakes and reverts registered → draft
- [ ] Submit-for-approval only resets collaborators with changed stakes
- [ ] Invite email includes stake %, role, terms, work title, project name, artist name

**Verify:** Test invite-with-stakes creates both records; test confirm auto-transitions work status.

**Steps:**

- [ ] **Step 1: Add new models to registry/models.py**

Add to `src/backend/registry/models.py`:

```python
class StakeInput(BaseModel):
    stake_type: str  # master, publishing
    percentage: float


class CollaboratorInviteWithStakes(BaseModel):
    work_id: str
    email: EmailStr
    name: str
    role: str
    stakes: List[StakeInput] = []
    notes: Optional[str] = None
```

- [ ] **Step 2: Create work_links_service.py**

Write `src/backend/registry/work_links_service.py`:

```python
from supabase import Client


async def get_work_files(db: Client, work_id: str):
    result = (
        db.table("work_files")
        .select("*, project_files(*)")
        .eq("work_id", work_id)
        .execute()
    )
    return result.data or []


async def link_file_to_work(db: Client, work_id: str, file_id: str):
    result = (
        db.table("work_files")
        .insert({"work_id": work_id, "file_id": file_id})
        .execute()
    )
    return result.data[0] if result.data else None


async def unlink_file_from_work(db: Client, link_id: str):
    db.table("work_files").delete().eq("id", link_id).execute()
    return {"deleted": link_id}


async def get_work_audio(db: Client, work_id: str):
    result = (
        db.table("work_audio_links")
        .select("*, audio_files(*)")
        .eq("work_id", work_id)
        .execute()
    )
    return result.data or []


async def link_audio_to_work(db: Client, work_id: str, audio_file_id: str):
    result = (
        db.table("work_audio_links")
        .insert({"work_id": work_id, "audio_file_id": audio_file_id})
        .execute()
    )
    return result.data[0] if result.data else None


async def unlink_audio_from_work(db: Client, link_id: str):
    db.table("work_audio_links").delete().eq("id", link_id).execute()
    return {"deleted": link_id}
```

- [ ] **Step 3: Update registry/service.py — invite-with-stakes, decline, accept-from-dashboard, auto-transition**

Add these functions to `src/backend/registry/service.py`:

```python
async def invite_with_stakes(db: Client, user_id: str, data):
    """Create collaborator + ownership stakes atomically."""
    import secrets
    from datetime import datetime, timedelta, timezone

    # Validate user owns the work
    work = db.table("works_registry").select("*").eq("id", data.work_id).single().execute()
    if not work.data or work.data["user_id"] != user_id:
        raise PermissionError("Not the work owner")

    # Create collaborator
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
    collab = db.table("registry_collaborators").insert({
        "work_id": data.work_id,
        "invited_by": user_id,
        "email": data.email,
        "name": data.name,
        "role": data.role,
        "status": "invited",
        "invite_token": token,
        "expires_at": expires,
    }).execute()
    collab_row = collab.data[0] if collab.data else None

    # Create stakes
    created_stakes = []
    for stake in data.stakes:
        s = db.table("ownership_stakes").insert({
            "work_id": data.work_id,
            "user_id": user_id,
            "stake_type": stake.stake_type,
            "holder_name": data.name,
            "holder_role": data.role,
            "percentage": stake.percentage,
            "holder_email": data.email,
        }).execute()
        if s.data:
            created_stakes.append(s.data[0])

    # Link first stake to collaborator if exists
    if created_stakes and collab_row:
        db.table("registry_collaborators").update({
            "stake_id": created_stakes[0]["id"]
        }).eq("id", collab_row["id"]).execute()

    # If work is registered, revert to draft (ownership changed)
    if work.data["status"] == "registered":
        db.table("works_registry").update({"status": "draft"}).eq("id", data.work_id).execute()

    return {"collaborator": collab_row, "stakes": created_stakes, "invite_token": token}


async def decline_invitation(db: Client, user_id: str, collaborator_id: str):
    """Decline an invitation. Validates email match."""
    # Get collaborator record
    collab = db.table("registry_collaborators").select("*").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise ValueError("Collaborator not found")

    # Validate email match
    user = db.rpc("get_user_email", {"uid": user_id}).execute()
    user_email = user.data if isinstance(user.data, str) else None
    if not user_email:
        # Fallback: check profiles table
        profile = db.table("profiles").select("email").eq("id", user_id).maybeSingle().execute()
        user_email = profile.data["email"] if profile.data else None

    if not user_email or user_email.lower() != collab.data["email"].lower():
        raise PermissionError("Email does not match invitation")

    db.table("registry_collaborators").update({
        "status": "declined",
        "collaborator_user_id": user_id,
        "responded_at": "now()",
    }).eq("id", collaborator_id).execute()

    return {"declined": collaborator_id}


async def accept_from_dashboard(db: Client, user_id: str, collaborator_id: str):
    """Claim + confirm atomically from the registry dashboard."""
    collab = db.table("registry_collaborators").select("*").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise ValueError("Collaborator not found")

    # Validate email match
    profile = db.table("profiles").select("email").eq("id", user_id).maybeSingle().execute()
    user_email = profile.data["email"] if profile.data else None
    if not user_email or user_email.lower() != collab.data["email"].lower():
        raise PermissionError("Email does not match invitation")

    # Claim + confirm
    db.table("registry_collaborators").update({
        "collaborator_user_id": user_id,
        "status": "confirmed",
        "responded_at": "now()",
    }).eq("id", collaborator_id).execute()

    # Check auto-transition
    await _check_auto_register(db, collab.data["work_id"])

    return {"accepted": collaborator_id}


async def get_my_invites(db: Client, user_id: str):
    """Get invites for current user by email match (for Action Required tab)."""
    profile = db.table("profiles").select("email").eq("id", user_id).maybeSingle().execute()
    if not profile.data or not profile.data.get("email"):
        return []

    email = profile.data["email"].lower()
    result = (
        db.table("registry_collaborators")
        .select("*, works_registry(id, title, project_id, status, projects(name), artists:artist_id(name))")
        .ilike("email", email)
        .eq("status", "invited")
        .order("invited_at", desc=True)
        .execute()
    )
    return result.data or []


async def _check_auto_register(db: Client, work_id: str):
    """After a confirm, check if ALL collaborators are confirmed → auto-register."""
    collabs = (
        db.table("registry_collaborators")
        .select("status")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .neq("status", "declined")
        .execute()
    )
    if not collabs.data:
        return
    all_confirmed = all(c["status"] == "confirmed" for c in collabs.data)
    if all_confirmed:
        db.table("works_registry").update({"status": "registered"}).eq("id", work_id).execute()
```

Also update the existing `confirm_stake` function to call `_check_auto_register` after confirming, and update the `revoke_collaborator` function to delete associated stakes and revert registered works to draft.

- [ ] **Step 4: Update registry/router.py — add new routes**

Add to `src/backend/registry/router.py`:

```python
from registry import work_links_service
from registry.models import CollaboratorInviteWithStakes

# Work file links
@router.get("/works/{work_id}/files")
async def list_work_files(work_id: str, user_id: str = Query(...)):
    files = await work_links_service.get_work_files(_get_supabase(), work_id)
    return {"files": files}

@router.post("/works/{work_id}/files")
async def link_file(work_id: str, file_id: str = Query(...), user_id: str = Query(...)):
    result = await work_links_service.link_file_to_work(_get_supabase(), work_id, file_id)
    return {"link": result}

@router.delete("/works/{work_id}/files/{link_id}")
async def unlink_file(work_id: str, link_id: str, user_id: str = Query(...)):
    return await work_links_service.unlink_file_from_work(_get_supabase(), link_id)

# Work audio links
@router.get("/works/{work_id}/audio")
async def list_work_audio(work_id: str, user_id: str = Query(...)):
    audio = await work_links_service.get_work_audio(_get_supabase(), work_id)
    return {"audio": audio}

@router.post("/works/{work_id}/audio")
async def link_audio(work_id: str, audio_file_id: str = Query(...), user_id: str = Query(...)):
    result = await work_links_service.link_audio_to_work(_get_supabase(), work_id, audio_file_id)
    return {"link": result}

@router.delete("/works/{work_id}/audio/{link_id}")
async def unlink_audio(work_id: str, link_id: str, user_id: str = Query(...)):
    return await work_links_service.unlink_audio_from_work(_get_supabase(), link_id)

# Enhanced collaboration
@router.post("/collaborators/invite-with-stakes")
async def invite_with_stakes(body: CollaboratorInviteWithStakes, user_id: str = Query(...)):
    try:
        result = await service.invite_with_stakes(_get_supabase(), user_id, body)
        # Send rich invite email
        try:
            from registry.emails import send_rich_invitation_email
            work = _get_supabase().table("works_registry").select("title, projects(name), artists:artist_id(name)").eq("id", body.work_id).single().execute()
            inviter = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
            send_rich_invitation_email(
                recipient_email=body.email,
                recipient_name=body.name,
                inviter_name=inviter.data.get("full_name", "Someone") if inviter.data else "Someone",
                work_title=work.data["title"] if work.data else "Unknown",
                project_name=work.data["projects"]["name"] if work.data and work.data.get("projects") else "Unknown",
                artist_name=work.data["artists"]["name"] if work.data and work.data.get("artists") else "Unknown",
                role=body.role,
                stakes=body.stakes,
                notes=body.notes,
                invite_token=result["invite_token"],
            )
        except Exception as e:
            print(f"Warning: Failed to send invitation email: {e}")
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.post("/collaborators/{collaborator_id}/decline")
async def decline_invitation(collaborator_id: str, user_id: str = Query(...)):
    try:
        return await service.decline_invitation(_get_supabase(), user_id, collaborator_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.post("/collaborators/{collaborator_id}/accept-from-dashboard")
async def accept_from_dashboard(collaborator_id: str, user_id: str = Query(...)):
    try:
        return await service.accept_from_dashboard(_get_supabase(), user_id, collaborator_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/collaborators/my-invites")
async def my_invites(user_id: str = Query(...)):
    invites = await service.get_my_invites(_get_supabase(), user_id)
    return {"invites": invites}
```

- [ ] **Step 5: Enhance invite email with stake details**

Add `send_rich_invitation_email` to `src/backend/registry/emails.py` that includes work title, project name, artist name, stake percentages, role, notes/terms, and accept/decline action buttons. Follow the existing `send_invitation_email` pattern but with richer HTML content per the spec's "Invitation Email Content" section.

- [ ] **Step 6: Commit**

```bash
git add src/backend/registry/
git commit -m "feat: add work file/audio links, invite-with-stakes, decline, accept-from-dashboard, auto-transition"
```

---

### Task 4: Frontend — Shared Hooks & Components

**Goal:** Build the reusable hooks and components that multiple pages will use.

**Files:**
- Create: `src/hooks/useProjectMembers.ts`
- Create: `src/hooks/useWorkFiles.ts`
- Create: `src/hooks/useWorkAudio.ts`
- Create: `src/components/InlineEdit.tsx`

**Acceptance Criteria:**
- [ ] `useProjectMembers` hook provides CRUD queries/mutations for project members
- [ ] `useWorkFiles` hook provides link/unlink queries/mutations
- [ ] `useWorkAudio` hook provides link/unlink queries/mutations
- [ ] `InlineEdit` component renders text that becomes an input on click, saves on Enter/blur, cancels on Escape

**Verify:** InlineEdit renders correctly in isolation; hooks compile without errors.

**Steps:**

- [ ] **Step 1: Create useProjectMembers hook**

Write `src/hooks/useProjectMembers.ts` following the pattern in `src/hooks/useRegistry.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface ProjectMember {
  id: string;
  project_id: string;
  user_id: string;
  role: "owner" | "admin" | "editor" | "viewer";
  invited_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface PendingInvite {
  id: string;
  project_id: string;
  email: string;
  role: string;
  invited_by: string;
  created_at: string;
  expires_at: string;
}

export function useProjectMembers(projectId?: string) {
  const { user } = useAuth();
  return useQuery<ProjectMember[]>({
    queryKey: ["project-members", projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ members: ProjectMember[] }>(
        `${API_URL}/projects/${projectId}/members?user_id=${user.id}`
      );
      return data.members;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useMyRole(projectId?: string) {
  const { data: members } = useProjectMembers(projectId);
  const { user } = useAuth();
  if (!members || !user) return null;
  const me = members.find((m) => m.user_id === user.id);
  return me?.role ?? null;
}

export function useAddProjectMember() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, email, role }: { projectId: string; email: string; role: string }) => {
      return apiFetch(`${API_URL}/projects/${projectId}/members?user_id=${user!.id}`, {
        method: "POST",
        body: JSON.stringify({ email, role }),
      });
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Member added");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUpdateMemberRole() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, memberId, role }: { projectId: string; memberId: string; role: string }) => {
      return apiFetch(`${API_URL}/projects/${projectId}/members/${memberId}?user_id=${user!.id}`, {
        method: "PUT",
        body: JSON.stringify({ role }),
      });
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Role updated");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useRemoveProjectMember() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, memberId }: { projectId: string; memberId: string }) => {
      return apiFetch(`${API_URL}/projects/${projectId}/members/${memberId}?user_id=${user!.id}`, {
        method: "DELETE",
      });
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Member removed");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
```

- [ ] **Step 2: Create useWorkFiles and useWorkAudio hooks**

Write `src/hooks/useWorkFiles.ts` and `src/hooks/useWorkAudio.ts` following the same pattern — each provides a `useQuery` for listing and `useMutation`s for link/unlink. Use query keys `["work-files", workId]` and `["work-audio", workId]` respectively.

- [ ] **Step 3: Create InlineEdit component**

Write `src/components/InlineEdit.tsx`:

```typescript
import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Pencil } from "lucide-react";

interface InlineEditProps {
  value: string;
  onSave: (newValue: string) => Promise<void>;
  className?: string;
  inputClassName?: string;
  disabled?: boolean;
}

export function InlineEdit({ value, onSave, className = "", inputClassName = "", disabled = false }: InlineEditProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setDraft(value); }, [value]);
  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);

  const handleSave = async () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === value) {
      setDraft(value);
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      await onSave(trimmed);
      setEditing(false);
    } catch {
      setDraft(value);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") { setDraft(value); setEditing(false); }
  };

  if (editing) {
    return (
      <Input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={handleSave}
        onKeyDown={handleKeyDown}
        disabled={saving}
        className={inputClassName}
      />
    );
  }

  return (
    <span
      className={`group inline-flex items-center gap-1.5 ${disabled ? "" : "cursor-pointer"} ${className}`}
      onClick={() => !disabled && setEditing(true)}
    >
      {value}
      {!disabled && <Pencil className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />}
    </span>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add src/hooks/useProjectMembers.ts src/hooks/useWorkFiles.ts src/hooks/useWorkAudio.ts src/components/InlineEdit.tsx
git commit -m "feat: add shared hooks (project members, work files/audio) and InlineEdit component"
```

---

### Task 5: Frontend — Project Detail Page

**Goal:** Rewrite ProjectDetail.tsx as a 5-tab layout (Works, Files, Audio, Members, Settings) with all CRUD functionality.

**Files:**
- Modify: `src/pages/ProjectDetail.tsx` — complete rewrite
- Create: `src/components/project/WorksTab.tsx`
- Create: `src/components/project/FilesTab.tsx`
- Create: `src/components/project/AudioTab.tsx`
- Create: `src/components/project/MembersTab.tsx`
- Create: `src/components/project/SettingsTab.tsx`
- Create: `src/components/project/AddWorkDialog.tsx`

**Acceptance Criteria:**
- [ ] Project header with inline-editable name, role badge, "+ Add Work" button
- [ ] Works tab lists work cards with status/type badges, click navigates to WorkDetail
- [ ] AddWorkDialog creates works with type dropdown (including Other + free text), audio file selector
- [ ] Files tab shows 4 collapsible folders with "Relevant works" labels, file upload, work linking
- [ ] Audio tab lists project audio with work linking, upload button
- [ ] Members tab shows project members (role badges, role dropdown) and work-only collaborators
- [ ] Settings tab with project name/description edit, danger zone delete
- [ ] Role-based UI: editors+ see create/edit, admins+ see member management, owner sees delete

**Verify:** Navigate to `/projects/{id}` — all 5 tabs render and function correctly.

**Steps:**

- [ ] **Step 1: Create the tab components**

Create `src/components/project/` directory. Build each tab as a focused component:

**WorksTab.tsx** — receives `projectId`, `userRole`. Fetches works via `useWorksByProject(projectId)`. Renders work cards with status badges (Draft=gray, Pending=amber, Registered=green), type badges, ISRC, linked audio name, collaborator count. Click navigates to `/tools/registry/{workId}`. Shows "+ Add Work" button for editors+. Uses `InlineEdit` for work titles.

**AddWorkDialog.tsx** — form with: title (required), work type dropdown (Single, EP Track, Album Track, Composition, Other → shows free text input), ISRC (optional), audio file dropdown (filtered to project's audio) + "Upload New" button. On submit: calls `useCreateWork` then optionally `linkAudioToWork`.

**FilesTab.tsx** — receives `projectId`, `userRole`. Shows 4 collapsible sections (Contracts, Split Sheets, Royalty Statements, Other). Each file row shows name, "Relevant works: [links]" (fetched via work_files join), upload date. Upload button per folder. After upload, prompt to link to works. SHA-256 dedup check within the same project before uploading.

**AudioTab.tsx** — receives `projectId`, `userRole`. Lists audio files with format/duration info and "Relevant works" labels. Upload button. "Link to work" action on unlinked files.

**MembersTab.tsx** — receives `projectId`, `userRole`. Two sections: "Project Members" (avatar, name, email, role badge dropdown for admins+, remove button) and "Work-Only Collaborators" (read-only list from registry_collaborators filtered to this project's works). Uses `useProjectMembers` hook.

**SettingsTab.tsx** — receives `projectId`, `userRole`, `project` data. Editable name + description inputs. Read-only artist display. Danger zone with delete button (owner only) and "Leave project" button (non-owners).

- [ ] **Step 2: Rewrite ProjectDetail.tsx**

Rewrite `src/pages/ProjectDetail.tsx` as the shell page:

```typescript
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { useMyRole } from "@/hooks/useProjectMembers";
import { InlineEdit } from "@/components/InlineEdit";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Plus, Loader2 } from "lucide-react";
import { WorksTab } from "@/components/project/WorksTab";
import { FilesTab } from "@/components/project/FilesTab";
import { AudioTab } from "@/components/project/AudioTab";
import { MembersTab } from "@/components/project/MembersTab";
import { SettingsTab } from "@/components/project/SettingsTab";
import { AddWorkDialog } from "@/components/project/AddWorkDialog";
import { useState } from "react";

const ROLE_COLORS: Record<string, string> = {
  owner: "bg-purple-500/20 text-purple-400",
  admin: "bg-blue-500/20 text-blue-400",
  editor: "bg-amber-500/20 text-amber-400",
  viewer: "bg-emerald-500/20 text-emerald-400",
};

const ProjectDetail = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const role = useMyRole(projectId);
  const [showAddWork, setShowAddWork] = useState(false);

  const { data: project, isLoading } = useQuery({
    queryKey: ["project-detail", projectId],
    queryFn: async () => {
      if (!projectId) return null;
      const { data } = await supabase
        .from("projects")
        .select("*, artists(name, user_id)")
        .eq("id", projectId)
        .single();
      return data;
    },
    enabled: !!projectId,
  });

  if (isLoading) return <div className="flex justify-center p-12"><Loader2 className="h-8 w-8 animate-spin" /></div>;
  if (!project) return <div className="p-8 text-center text-muted-foreground">Project not found or access denied.</div>;

  const canEdit = role === "owner" || role === "admin" || role === "editor";

  const handleRename = async (newName: string) => {
    await supabase.from("projects").update({ name: newName }).eq("id", projectId!);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate("/portfolio")} className="text-sm text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4 inline mr-1" />Portfolio
          </button>
          <span className="text-muted-foreground">|</span>
          <InlineEdit
            value={project.name}
            onSave={handleRename}
            disabled={!canEdit}
            className="text-xl font-bold"
          />
          {role && <Badge className={ROLE_COLORS[role]}>{role}</Badge>}
        </div>
        {canEdit && (
          <Button size="sm" onClick={() => setShowAddWork(true)}>
            <Plus className="h-4 w-4 mr-1" />Add Work
          </Button>
        )}
      </div>

      {/* Tabs */}
      <Tabs defaultValue="works" className="px-6 pt-2">
        <TabsList>
          <TabsTrigger value="works">Works</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="audio">Audio</TabsTrigger>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>
        <TabsContent value="works"><WorksTab projectId={projectId!} userRole={role} /></TabsContent>
        <TabsContent value="files"><FilesTab projectId={projectId!} userRole={role} /></TabsContent>
        <TabsContent value="audio"><AudioTab projectId={projectId!} userRole={role} /></TabsContent>
        <TabsContent value="members"><MembersTab projectId={projectId!} userRole={role} /></TabsContent>
        <TabsContent value="settings"><SettingsTab projectId={projectId!} userRole={role} project={project} /></TabsContent>
      </Tabs>

      <AddWorkDialog
        open={showAddWork}
        onOpenChange={setShowAddWork}
        projectId={projectId!}
        artistId={project.artist_id}
      />
    </div>
  );
};

export default ProjectDetail;
```

- [ ] **Step 3: Build each tab component with full CRUD**

Implement each tab component in `src/components/project/`. Each should be self-contained, fetch its own data via hooks, and handle its own CRUD operations. Follow the existing component patterns (shadcn Dialog for modals, toast for feedback, Badge for status indicators).

- [ ] **Step 4: Commit**

```bash
git add src/pages/ProjectDetail.tsx src/components/project/
git commit -m "feat: rewrite ProjectDetail with 5-tab layout (works, files, audio, members, settings)"
```

---

### Task 6: Frontend — Portfolio Simplification

**Goal:** Simplify Portfolio.tsx from ~1300 lines to a clean project card grid with "My Projects" and "Shared with Me" sections.

**Files:**
- Modify: `src/pages/Portfolio.tsx` — major simplification
- Modify: `src/hooks/usePortfolioData.ts` — add shared projects query

**Acceptance Criteria:**
- [ ] "My Projects" section shows projects grouped by artist → year
- [ ] "Shared with Me" section shows projects where user is member but not owner, with role badge
- [ ] Each project card shows: name, work count, member count, last updated
- [ ] Click card navigates to `/projects/{projectId}`
- [ ] Keep: search, filter by artist, sort, date range
- [ ] Keep: "+ Create Project" button
- [ ] Remove: file management, audio management, board tasks, inline project editing

**Verify:** Navigate to `/portfolio` — shows card grid, click card goes to Project Detail.

**Steps:**

- [ ] **Step 1: Update usePortfolioData to include shared projects**

Add a query for projects where user is a member (from `project_members` table) to `src/hooks/usePortfolioData.ts`. Return both `myProjects` (owner) and `sharedProjects` (member, not owner).

- [ ] **Step 2: Rewrite Portfolio.tsx**

Strip out the accordion/expansion UI, file management, audio management, and board task display. Replace with a card grid. Keep the existing search/filter/sort bar. Add the "Shared with Me" section below "My Projects". Each card is a simple `<Card>` with project name, work count badge, member count badge, last updated date, and the `onClick={() => navigate(`/projects/${project.id}`)}` handler.

- [ ] **Step 3: Commit**

```bash
git add src/pages/Portfolio.tsx src/hooks/usePortfolioData.ts
git commit -m "feat: simplify Portfolio to card grid with My Projects + Shared with Me sections"
```

---

### Task 7: Frontend — Registry Dashboard Redesign

**Goal:** Redesign Registry.tsx as a read-only ownership tracking dashboard with summary cards and 4 tabs.

**Files:**
- Modify: `src/pages/Registry.tsx` — full redesign
- Modify: `src/hooks/useRegistry.ts` — add my-invites hook, update types

**Acceptance Criteria:**
- [ ] Summary cards: My Works count, Registered count, Pending count, Collaborations count
- [ ] "Action Required" tab: pending invites with Accept/Decline buttons (directly from dashboard)
- [ ] "My Works" tab: works grouped by year → project, with master/publishing percentages
- [ ] "Collaborations" tab: work-scoped view of works user is invited to
- [ ] "All Activity" tab: chronological feed of status changes
- [ ] Remove: "Create Work" button and dialog (creation now in Project Detail)
- [ ] Search bar across all tabs

**Verify:** Navigate to `/tools/registry` — summary cards render, tabs switch correctly, Accept/Decline work from Action Required tab.

**Steps:**

- [ ] **Step 1: Add useMyInvites hook**

Add to `src/hooks/useRegistry.ts`:

```typescript
export function useMyInvites() {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["registry-my-invites", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<{ invites: any[] }>(
        `${API_URL}/registry/collaborators/my-invites?user_id=${user.id}`
      );
      return data.invites;
    },
    enabled: !!user?.id,
  });
}

export function useAcceptFromDashboard() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (collaboratorId: string) => {
      return apiFetch(
        `${API_URL}/registry/collaborators/${collaboratorId}/accept-from-dashboard?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-my-invites"] });
      queryClient.invalidateQueries({ queryKey: ["registry-works"] });
      toast.success("Invitation accepted");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useDeclineInvitation() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (collaboratorId: string) => {
      return apiFetch(
        `${API_URL}/registry/collaborators/${collaboratorId}/decline?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-my-invites"] });
      toast.success("Invitation declined");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
```

- [ ] **Step 2: Redesign Registry.tsx**

Rewrite `src/pages/Registry.tsx` with:
- 4 summary cards at top (purple=My Works, green=Registered, amber=Pending, blue=Collaborations)
- Tabs component with Action Required (default if invites exist), My Works, Collaborations, All Activity
- Action Required: map over `useMyInvites()` data, render invite cards with Accept/Decline buttons
- My Works: group works by year (collapsible) → by project, show master/publishing % per work row
- Collaborations: map over `useMyCollaborations()`, show work-scoped cards
- All Activity: use existing `useRegistryNotifications()` or build from registry_notifications table

- [ ] **Step 3: Commit**

```bash
git add src/pages/Registry.tsx src/hooks/useRegistry.ts
git commit -m "feat: redesign Registry as ownership tracking dashboard with 4 tabs"
```

---

### Task 8: Frontend — WorkDetail Updates + Enhanced Invite

**Goal:** Update WorkDetail to remove disputed status, add linked files/audio display, inline rename, and enhanced invite form.

**Files:**
- Modify: `src/pages/WorkDetail.tsx`
- Modify: `src/components/registry/CollaborationStatus.tsx`
- Modify: `src/components/registry/InviteCollaboratorModal.tsx`
- Modify: `src/components/registry/OwnershipPanel.tsx`

**Acceptance Criteria:**
- [ ] Status badges use 3 states only: Draft (gray), Pending (amber), Registered (green)
- [ ] No dispute flow — collaborators see Accept/Decline only
- [ ] Work title uses InlineEdit
- [ ] Linked files section shows contracts/files from work_files
- [ ] Linked audio section shows audio from work_audio_links
- [ ] Work type displays custom text for "Other" type
- [ ] Enhanced invite form: email, name, role (with Other free text), stake type, percentages, notes
- [ ] "Register" button for zero-collaborator works (direct draft → registered)
- [ ] CollaborationStatus shows Accept/Decline (no dispute)

**Verify:** Navigate to `/tools/registry/{workId}` — all updates visible, invite form comprehensive.

**Steps:**

- [ ] **Step 1: Update WorkDetail.tsx**

Remove all dispute-related code (dispute dialog, dispute reason display, disputed status badge). Add `InlineEdit` for work title. Add sections for linked files and audio (fetch via `useWorkFiles` and `useWorkAudio` hooks). Add "Register" button visible when work is `draft` and has 0 collaborators. Update status color map to remove `disputed`.

- [ ] **Step 2: Update InviteCollaboratorModal.tsx**

Rewrite the form to include: stake type selector (Master / Publishing / Both), percentage inputs per selected type, notes/terms field. On submit, call `useInviteWithStakes` mutation instead of the old `useInviteCollaborator`. Add "Other" option to role dropdown with free text input.

- [ ] **Step 3: Update CollaborationStatus.tsx**

Remove dispute button and dispute reason display. Replace with Decline button. Update status badges: `invited` → "Pending" (amber), `confirmed` → "Accepted" (green), `declined` → "Declined" (gray). Remove dispute-related filtering.

- [ ] **Step 4: Commit**

```bash
git add src/pages/WorkDetail.tsx src/components/registry/
git commit -m "feat: update WorkDetail — remove dispute, add file/audio links, enhanced invite form"
```

---

### Task 9: Frontend — Zoe & OneClick Integration

**Goal:** Add "From Shared Works" document source to Zoe and "From Works" source to OneClick.

**Files:**
- Modify: `src/pages/Zoe.tsx` — add shared works contract source
- Modify: `src/pages/OneClick.tsx` — add works as document source

**Acceptance Criteria:**
- [ ] Zoe document selector includes "From Shared Works" option
- [ ] Selecting it shows works where user is confirmed collaborator, filtered to those with linked files
- [ ] Picking a work shows its linked contracts for selection
- [ ] OneClick document selection step includes "From Works" source
- [ ] Selecting it shows works grouped by project for the selected artist

**Verify:** Open Zoe → see "From Shared Works" in document picker. Open OneClick → see "From Works" in document sources.

**Steps:**

- [ ] **Step 1: Update Zoe.tsx**

Add a new document source option in Zoe's document selector. When "From Shared Works" is selected, fetch works via `useMyCollaborations()` filtered to `status = 'confirmed'`. For the selected work, fetch linked files via `GET /works/{workId}/files`. Pass the selected file to Zoe's existing document processing pipeline.

- [ ] **Step 2: Update OneClick.tsx**

Add "From Works" as a document source option in the document selection step. When selected, show works for the chosen artist grouped by project (using `useWorksByProject`). When a work is selected, show its linked files. Selected file feeds into OneClick's existing analysis pipeline.

- [ ] **Step 3: Commit**

```bash
git add src/pages/Zoe.tsx src/pages/OneClick.tsx
git commit -m "feat: add shared works document source to Zoe and works source to OneClick"
```

---

### Implementation Notes (from spec)

Handle these during execution:

1. **pending_project_invites management:** 7-day expiry, cleanup on access
2. **AddWorkDialog atomicity:** Sequential frontend calls are fine; orphaned audio is harmless
3. **Admin demoting other admins:** Intentional — document in UI tooltip
4. **updated_at trigger on project_members:** Added in migration step 1
5. **auto_create_project_owner and service role:** Only fires for authenticated users (auth.uid() check)
6. **Existing upload policies:** Verify project_files and audio_files already allow editors+ INSERT during Task 1
7. **Case-insensitive email matching:** Apply LOWER() in all backend email comparisons
8. **CLAUDE.md:** Already created at repo root with full project documentation
