# Portfolio + Registry Redesign: Project Detail, Ownership Tracking & Access Control

**Date:** 2026-04-03
**Status:** Approved
**Branch:** msaniiV2-tasks-licensing-artistmanagement

---

## Overview

Redesign the Portfolio and Rights Registry to introduce a dedicated Project Detail page, dual-layer access control, and an ownership tracking dashboard. Creation flows through Portfolio → Project Detail. The Registry becomes a read-only tracking hub. OneClick and Zoe gain the ability to read data from works and shared contracts.

---

## Page Structure

### 1. Portfolio Page (Simplified)

**Path:** `/portfolio`
**Role:** Project grid — lightweight entry point.

**Changes from current:**
- Remove inline project expansion/accordion (currently ~1300 lines of nested UI)
- Replace with a clean card grid in two sections:
  - **My Projects** — grouped by artist → year (projects where user is owner)
  - **Shared with Me** — projects where user is a member but not owner, grouped by artist → year. Shows the role badge on each card.
- Each project card shows: name, work count, member count, last updated
- Click card → navigates to `/projects/{projectId}`
- Keep: search, filter by artist, sort (A-Z, newest, oldest), date range filters
- Keep: "+ Create Project" button (opens ProjectFormDialog)
- Add: optional "Add Works" step during project creation (hybrid flow — add multiple works upfront, or skip and add later)

**What moves out:**
- File management → Project Detail Files tab
- Audio management → Project Detail Audio tab
- Board tasks display → removed from Portfolio; board tasks are managed in Workspace and are not part of Project Detail
- Inline project editing → Project Detail Settings tab

### 2. Project Detail Page (NEW)

**Path:** `/projects/{projectId}`
**Role:** Central hub for everything inside a project.

**Header:**
- Back link: "← Portfolio"
- Project name — inline click-to-edit (click title, type, press Enter to save)
- Role badge (Owner/Admin/Editor/Viewer) — color-coded
- "+ Add Work" button
- Overflow menu (⋮)

**5 Tabs:**

#### Tab 1: Works
- List of all works in this project as cards
- Each card shows:
  - Work title (inline click-to-edit)
  - Status badge: Draft (gray), Pending (amber), Registered (green)
  - Type badge: Single, EP Track, Album Track, Composition, or custom text (via "Other")
  - ISRC (if set)
  - Linked audio file name
  - Collaborator count + allocation status
- Click card → navigates to `/tools/registry/{workId}` (existing WorkDetail page)
- "+ Add Work" button opens dialog:
  - Work Title (required)
  - Work Type dropdown: Single, EP Track, Album Track, Composition, Other → free text input appears when Other selected
  - ISRC (optional)
  - Link Audio File: dropdown of project-scoped audio files + "or Upload New" action
  - Create button

#### Tab 2: Files
- 4 folder categories: Contracts, Split Sheets, Royalty Statements, Other
- Each folder is collapsible with file count
- Each file row shows:
  - File name
  - "Relevant works: [Work A], [Work B]" — clickable links
  - Upload date
- Project-level search bar across all files
- Upload button per folder
- File linking: after upload, prompt to link to specific works
- SHA-256 content hash computed on upload, stored on file record
  - On upload: check if a file with the same hash already exists **within the same project** (not cross-project — cross-project files are independent even if identical)
  - If duplicate found within project: prompt "This file already exists in this project. Link the existing file to additional works instead of uploading a duplicate?"
  - If no match: upload normally

#### Tab 3: Audio
- List of all audio files in this project
- Each row shows: file name, format/bitrate/duration, "Relevant works: [Work Name]" or "Not linked to any work"
- "+ Upload Audio" button — uploads and auto-links to project
- "Link to work" action on unlinked files
- Audio files are project-scoped: only files linked to THIS project appear in work creation dropdowns

#### Tab 4: Members
- Split into two sections:

**Project Members** (can see all works):
- Avatar circle (initials, colored by role), name, email
- Role badge dropdown (clickable to change): Owner (purple), Admin (blue), Editor (amber), Viewer (green)
- Remove button (×) — admin+ can remove others, any non-owner member can remove themselves. Owner cannot be removed.
- "+ Invite Member" button

**Project Member Invite Flow:**
- Admin+ clicks "+ Invite Member" → enters email and selects role
- If the email matches an existing user account: member is added directly (no accept/decline ceremony — project membership is simpler than work collaboration)
- If no account exists for that email: an email is sent inviting them to sign up. On signup, they are auto-added to the project with the assigned role.
- Backend: `POST /projects/{projectId}/members` → validates caller is admin+, creates `project_members` row, sends email if needed via Resend

**Work-Only Collaborators** (can see only their linked work):
- Avatar circle, name, email
- "on [Work Name]" link
- Status badge: Accepted (green) / Pending (amber) / Declined (gray)
- These entries are read-only here — managed through the invite flow on WorkDetail

**Role permissions:**

| Capability | Owner | Admin | Editor | Viewer |
|---|---|---|---|---|
| See all works & files | ✓ | ✓ | ✓ | ✓ |
| Create/edit works | ✓ | ✓ | ✓ | ✗ |
| Upload files & audio | ✓ | ✓ | ✓ | ✗ |
| Manage members | ✓ | ✓ | ✗ | ✗ |
| Delete project | ✓ | ✗ | ✗ | ✗ |
| Edit project settings | ✓ | ✓ | ✗ | ✗ |

#### Tab 5: Settings
- Project Name (editable input)
- Description (editable text area)
- Artist (read-only, cannot be changed after creation)
- Danger Zone:
  - Delete project button with confirmation (owner only)
  - "Leave project" button for non-owner members (removes themselves from `project_members`)

### 3. Work Detail Page (Existing — Minor Updates)

**Path:** `/tools/registry/{workId}`
**Role:** Per-work ownership, licensing, agreements, collaboration.

**Changes:**
- Remove "Disputed" status — work statuses are now: Draft, Pending, Registered
- Collaborator flow: Accept or Decline only (no dispute/dispute_reason)
- Inline rename on work title (click-to-edit)
- Show linked files section: contracts/files linked to this work via work_files join table
- Show linked audio file
- Work type: support "Other" with custom text display

**Work Status Transition Rules:**
- `draft` → `pending_approval`: Owner clicks "Submit for Approval". Requires at least 1 collaborator with status `invited`. Endpoint: `POST /registry/works/{workId}/submit-for-approval` (already exists in codebase).
- `pending_approval` → `registered`: **Backend logic in the confirm endpoint** (`src/backend/registry/service.py` → `confirm_stake()`). After each `POST /registry/collaborators/{id}/confirm`, the backend checks: are ALL collaborators on this work now `confirmed`? If yes, auto-transition work to `registered`. If not, stay in `pending_approval`.
- `pending_approval` + new collaborator added: Work stays in `pending_approval`. The new collaborator receives an invite email automatically (no need for owner to re-submit). Only the new collaborator needs to confirm — **existing confirmed collaborators keep their status.** The auto-transition check still requires ALL collaborators to be confirmed before moving to `registered`.
- `registered` → `draft`: Triggered ONLY by changes that affect ownership/collaboration:
  - Adding a new collaborator → reverts to `draft` (new person needs to confirm)
  - Revoking a collaborator → reverts to `draft` (ownership changed)
  - Changing ownership stake percentages → reverts to `draft`
  - **Does NOT revert for:** title rename, ISRC/ISWC/UPC edits, work type change, release date, notes. These are metadata-only changes that don't affect who owns what.
  - **Implementation location:** This logic lives in the **backend service layer**, not DB triggers. Specifically:
    - `POST /registry/collaborators/invite-with-stakes` → if work is `registered`, set to `draft`
    - `POST /registry/collaborators/{id}/revoke` → if work is `registered`, set to `draft`
    - `PUT /registry/stakes/{stakeId}` → if work is `registered`, set to `draft`
    - `PUT /registry/works/{workId}` → does NOT check or revert status (metadata-only edits)
- **Zero collaborators:** A work with no collaborators stays in `draft`. The owner can manually register it by clicking "Register" (no approval needed if there's nobody to approve). This is a direct `draft` → `registered` transition available only when collaborator count = 0.
- **Collaborator self-decline after accepting:** Not allowed. Once `confirmed`, a collaborator's status is locked. Only the owner can revoke.

**Revoke Workflow:**
- Owner or admin can revoke a collaborator from the Work Detail → Collaboration panel (existing revoke button in `CollaborationStatus`).
- `POST /registry/collaborators/{id}/revoke` → sets status to `revoked`.
- On revoke: the associated ownership stake(s) linked to that collaborator are deleted (`DELETE FROM ownership_stakes WHERE id = collaborator.stake_id`).
- If the work was `registered` and a collaborator is revoked, work reverts to `draft` (ownership has changed, needs re-confirmation from remaining collaborators).
- Revoked collaborator loses access to the work and its files (RLS only grants access to `confirmed` status).

**Invite Collaborator Form (enhanced):**
The existing InviteCollaboratorModal becomes more comprehensive:
- Email (required)
- Name (required)
- Role dropdown: Artist, Producer, Songwriter, Composer, Publisher, Label, Other (free text)
- Stake Type: Master, Publishing, or Both
- If Master selected: Master percentage input
- If Publishing selected: Publishing percentage input
- If Both: both percentage inputs
- Link to specific stake (optional, for existing stakes)
- Notes/terms (optional text field)
- On submit: creates collaborator record + ownership stake(s) in one action. Backend needs a new composite endpoint (POST `/registry/collaborators/invite-with-stakes`) that creates both the collaborator and stake records atomically.

**Invitation Email Content:**
The invite email must contain all information the collaborator needs to make a decision, since they cannot access the work's files until they accept:
- Work title and type
- Project name and artist name
- Their assigned role (e.g., Producer)
- Their listed stake(s): "Master: 15%" / "Publishing: 25%" / both
- Any notes/terms the owner included
- Who invited them (inviter name + email)
- Accept / Decline buttons (links to claim endpoint)
- 48-hour expiry notice

This way the collaborator can accept or decline based on the email alone, without needing to log in or view files first. Files become accessible only after acceptance.

### 4. Rights Registry Dashboard (Redesigned)

**Path:** `/tools/registry`
**Role:** Cross-project ownership tracking. No creation — tracking only.

**Header:**
- Title: "Rights Registry"
- Subtitle: "Track your ownership, stakes, and collaborations across all projects"
- Search bar (works, artists, projects)
- Filter dropdown

**Summary Cards Row:**
- My Works (count, across N projects)
- Registered (count, fully confirmed)
- Pending (count, awaiting collaborator response)
- Collaborations (count, works you're invited to)

**4 Tabs:**

#### Action Required
- Pending invites that need your response
- **Invite visibility before claim:** Invites are matched by email. RLS policy on `registry_collaborators` allows `SELECT` where `email = (SELECT email FROM auth.users WHERE id = auth.uid())` AND `status = 'invited'`. This lets users see their invites before clicking the email claim link.
- **Accepting from this page:** When a user clicks Accept on an unclaimed invite, the backend performs claim + confirm atomically: sets `collaborator_user_id = auth.uid()` and `status = 'confirmed'` in one operation.
- Each card shows:
  - Who invited you, project name, work name
  - Your listed stake (master %, publishing %, role)
  - Accept / Decline / View Work buttons
- Accept and Decline can be done directly from this page (no need to navigate to WorkDetail)

#### My Works
- All works you own, organized by year (collapsible dropdowns: 2026 ▼, 2025 ▼, etc.)
- Within each year, grouped by project with "Open project →" link
- Each work row shows: title, status badge, master %, publishing %, collaborator count
- Click row → navigates to WorkDetail

#### Collaborations
- Works other people invited you to
- Scoped view: you see only the work and its details, NOT the project or other works
- Each card shows: work title, status, inviter name, project name (as context only, not clickable), your role, your stake percentage
- Click → navigates to WorkDetail (where you see work-level details only)

#### All Activity
- Chronological feed of: new invites received, acceptances, declines, status changes, new works created
- Each entry: timestamp, description, link to relevant work

---

## Data Model Changes

**Important: Do not run these directly. Create migration files only. User will run manually.**

### New Tables

#### `project_members`
```sql
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
```

#### `work_files`
Join table linking files to works:
```sql
CREATE TABLE work_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  file_id UUID NOT NULL REFERENCES project_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, file_id)
);
```

#### `work_audio_links`
Join table linking audio files to works:
```sql
CREATE TABLE work_audio_links (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  audio_file_id UUID NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(work_id, audio_file_id)
);
```

#### `pending_project_invites`
For project member invites to users who don't have an account yet:
```sql
CREATE TABLE pending_project_invites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
  invited_by UUID NOT NULL REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(project_id, email)
);
```
**Mechanism:** When `POST /projects/{projectId}/members` is called with an email that doesn't match any `auth.users.email`:
1. Insert into `pending_project_invites` (not `project_members`)
2. Send signup invitation email via Resend

**On signup trigger** (Supabase auth hook):
```sql
CREATE OR REPLACE FUNCTION process_pending_project_invites()
RETURNS TRIGGER SECURITY DEFINER AS $$
BEGIN
  INSERT INTO project_members (project_id, user_id, role, invited_by)
  SELECT pi.project_id, NEW.id, pi.role, pi.invited_by
  FROM pending_project_invites pi
  WHERE pi.email = NEW.email;

  DELETE FROM pending_project_invites WHERE email = NEW.email;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER process_pending_invites_on_signup
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION process_pending_project_invites();
```

### Table Modifications

#### `project_files` — add content hash
```sql
ALTER TABLE project_files ADD COLUMN content_hash TEXT;
```

#### `works_registry` — add custom work type
```sql
ALTER TABLE works_registry ADD COLUMN custom_work_type TEXT;
```
The `work_type` column uses a CHECK constraint (not a Postgres enum). Replace the constraint to add 'other':
```sql
ALTER TABLE works_registry DROP CONSTRAINT works_registry_work_type_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_work_type_check
  CHECK (work_type IN ('single', 'ep_track', 'album_track', 'composition', 'other'));
```
When `work_type = 'other'`, `custom_work_type` stores the user's free text.

#### `works_registry` — status constraint update
**Data migration first** — migrate existing disputed records before changing the constraint:
```sql
UPDATE works_registry SET status = 'draft' WHERE status = 'disputed';
```
Then replace the constraint:
```sql
ALTER TABLE works_registry DROP CONSTRAINT works_registry_status_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_status_check
  CHECK (status IN ('draft', 'pending_approval', 'registered'));
```

#### `registry_collaborators` — simplify status
**Data migration first:**
```sql
UPDATE registry_collaborators SET status = 'declined' WHERE status = 'disputed';
```
Then replace the constraint:
```sql
ALTER TABLE registry_collaborators DROP CONSTRAINT registry_collaborators_status_check;
ALTER TABLE registry_collaborators ADD CONSTRAINT registry_collaborators_status_check
  CHECK (status IN ('invited', 'confirmed', 'declined', 'revoked'));
```
Drop the `dispute_reason` column:
```sql
ALTER TABLE registry_collaborators DROP COLUMN IF EXISTS dispute_reason;
```

### RLS Policy Updates

#### Invite visibility by email (before claim)
```sql
-- Users can see invites addressed to their email, even before claiming.
-- Uses LOWER() on both sides for case-insensitive matching
-- (e.g., Mike@Producer.com matches mike@producer.com).
CREATE POLICY "collaborators_select_by_email" ON registry_collaborators
  FOR SELECT USING (
    LOWER(email) = LOWER((SELECT email FROM auth.users WHERE id = auth.uid()))
    OR collaborator_user_id = auth.uid()
    OR invited_by = auth.uid()
  );
```

#### Projects table — member access
```sql
-- Project members can read the project itself (needed for Shared with Me, Project Detail header)
CREATE POLICY "projects_select_via_member" ON projects
  FOR SELECT USING (
    id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );
```

#### Project members table — mutual visibility
```sql
-- All project members can see each other (needed for Members tab)
CREATE POLICY "project_members_select_members" ON project_members
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );
```

#### File access via work collaboration
```sql
-- ONLY confirmed collaborators can read files linked to their works.
-- Invited (unclaimed/unconfirmed) collaborators do NOT get file access —
-- the invite email contains the key terms (stake %, role) so they can
-- make an informed decision without needing to access the actual files.
-- This prevents the loophole of indefinitely sitting on an invite to
-- view files without ever confirming.
CREATE POLICY "project_files_select_via_work_collab" ON project_files
  FOR SELECT USING (
    id IN (
      SELECT file_id FROM work_files WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );
```

#### Audio access via work collaboration
```sql
-- ONLY confirmed collaborators can read audio linked to their works (same rationale)
CREATE POLICY "audio_files_select_via_work_collab" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM work_audio_links WHERE work_id IN (
        SELECT work_id FROM registry_collaborators
        WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
      )
    )
  );
```

#### Join tables — SELECT policies
```sql
-- work_files: readable by project members and confirmed work collaborators
-- (Required for subqueries in project_files/audio_files policies to return results)
CREATE POLICY "work_files_select_members" ON work_files
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

-- work_audio_links: same pattern
CREATE POLICY "work_audio_links_select_members" ON work_audio_links
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
```

#### Works access for work-only collaborators
```sql
-- NOTE: The existing policy "Owner or collaborator can read works" (from migration
-- 20260329000000) already covers this case:
--   ON works_registry FOR SELECT USING (
--     auth.uid() = user_id
--     OR id IN (SELECT work_id FROM registry_collaborators
--               WHERE collaborator_user_id = auth.uid() AND status != 'revoked')
--   )
-- This lets claimed collaborators (invited or confirmed) read the work record itself.
-- No new policy needed for works_registry SELECT — just confirming the existing one
-- covers work-only collaborators like Mike after claim.
--
-- For pre-claim invite display (Action Required tab on Registry), the backend endpoint
-- GET /registry/collaborators/my-invites runs with service role and returns work
-- title/project name embedded in the response — no direct Supabase query needed.
```

#### Project member access
```sql
-- Project members can read all works in their project
CREATE POLICY "works_select_via_project_member" ON works_registry
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- Project members can read all files in their project
CREATE POLICY "project_files_select_via_project_member" ON project_files
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members WHERE user_id = auth.uid()
    )
  );

-- Project members can read all audio in their project (via project_audio_links)
CREATE POLICY "audio_files_select_via_project_member" ON audio_files
  FOR SELECT USING (
    id IN (
      SELECT audio_file_id FROM project_audio_links WHERE project_id IN (
        SELECT project_id FROM project_members WHERE user_id = auth.uid()
      )
    )
  );
```

#### Role-based write access
```sql
-- Editors+ can create/update works
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

-- Owner/admin can delete works
CREATE POLICY "works_delete_owner_admin" ON works_registry
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

-- Editors+ can link/unlink files and audio to works
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

-- Same pattern for work_audio_links (editors+ can INSERT/DELETE)
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

-- Admins+ can manage project members (but cannot delete owner)
CREATE POLICY "project_members_insert_admins" ON project_members
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_update_admins" ON project_members
  FOR UPDATE USING (
    -- Admins+ can change roles (but not the owner's role — enforced by trigger)
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_delete_admins" ON project_members
  FOR DELETE USING (
    -- Admins+ can remove others (but not the owner — enforced by trigger)
    (project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    ))
    OR
    -- Any non-owner member can remove themselves
    (user_id = auth.uid() AND role != 'owner')
  );
```

#### Owner protection
```sql
-- Trigger: prevent owner deletion from project_members
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

-- Trigger: prevent changing owner's role AND prevent anyone from becoming owner
CREATE OR REPLACE FUNCTION protect_owner_role()
RETURNS TRIGGER AS $$
BEGIN
  -- Can't demote the owner
  IF OLD.role = 'owner' AND NEW.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot change the project owner role';
  END IF;
  -- Can't promote to owner (prevents multiple owners)
  IF NEW.role = 'owner' AND OLD.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot promote to owner — use ownership transfer instead';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER protect_owner_role_trigger
  BEFORE UPDATE ON project_members
  FOR EACH ROW EXECUTE FUNCTION protect_owner_role();

-- Belt-and-suspenders: partial unique index ensures at most one owner per project
CREATE UNIQUE INDEX one_owner_per_project
  ON project_members (project_id) WHERE role = 'owner';
```

#### Auto-owner trigger (SECURITY DEFINER)
```sql
-- When a project is created, auto-insert the creator as owner.
-- Must be SECURITY DEFINER because the INSERT policy on project_members
-- requires the user to already be an admin+ member, creating a chicken-and-egg problem.
CREATE OR REPLACE FUNCTION auto_create_project_owner()
RETURNS TRIGGER SECURITY DEFINER AS $$
BEGIN
  INSERT INTO project_members (project_id, user_id, role)
  VALUES (NEW.id, auth.uid(), 'owner');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_create_project_owner_trigger
  AFTER INSERT ON projects
  FOR EACH ROW EXECUTE FUNCTION auto_create_project_owner();
```

---

## Backend Endpoint Inventory

### Existing Endpoints (no changes needed)
| Method | Path | Purpose |
|---|---|---|
| GET | `/registry/works` | List user's works |
| GET | `/registry/works/{workId}` | Get single work |
| GET | `/registry/works/{workId}/full` | Work + stakes + licenses + agreements + collaborators |
| GET | `/registry/works/by-project/{projectId}` | Works in a project |
| GET | `/registry/works/my-collaborations` | Works where user is collaborator |
| POST | `/registry/works` | Create work |
| PUT | `/registry/works/{workId}` | Update work (resets registered → draft) |
| DELETE | `/registry/works/{workId}` | Delete work |
| POST | `/registry/works/{workId}/submit-for-approval` | Draft → pending_approval. Validates: at least 1 collaborator, work in `draft` status. **Only resets collaborators whose stakes changed** (new collaborators, or existing ones with modified percentages). Already-confirmed collaborators with unchanged stakes keep their `confirmed` status. Resends emails only to those reset to `invited`. |
| GET | `/registry/works/{workId}/export` | Export proof of ownership PDF |
| GET | `/registry/stakes` | List stakes for a work |
| POST | `/registry/stakes` | Create stake |
| PUT | `/registry/stakes/{stakeId}` | Update stake |
| DELETE | `/registry/stakes/{stakeId}` | Delete stake |
| GET | `/registry/licenses` | List licenses for a work |
| POST | `/registry/licenses` | Create license |
| PUT | `/registry/licenses/{licenseId}` | Update license |
| DELETE | `/registry/licenses/{licenseId}` | Delete license |
| GET | `/registry/agreements` | List agreements for a work |
| POST | `/registry/agreements` | Create agreement |
| POST | `/registry/collaborators/claim` | Claim invitation by token (sets collaborator_user_id) |
| POST | `/registry/collaborators/{id}/confirm` | Accept invitation. **Updated:** after confirming, check if all collaborators are confirmed → auto-transition work to `registered`. |
| POST | `/registry/collaborators/{id}/revoke` | Revoke collaborator. **Updated:** deletes associated ownership stakes, reverts `registered` work to `draft`. |
| POST | `/registry/collaborators/{id}/resend` | Resend expired invitation |

### New Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/registry/collaborators/invite-with-stakes` | Composite: creates collaborator record + ownership stake(s) atomically. Body: `{work_id, email, name, role, stakes: [{type, percentage}], notes?}` |
| POST | `/registry/collaborators/{id}/decline` | Decline invitation. Sets status to `declined`. Notifies work owner. **Backend MUST validate** that `auth.uid()`'s email matches the collaborator record's email (same as accept-from-dashboard). |
| GET | `/registry/collaborators/my-invites` | Get invites for current user by email match (for Action Required tab before claim). |
| POST | `/registry/collaborators/{id}/accept-from-dashboard` | Claim + confirm atomically (for accepting directly from Registry dashboard without visiting the email link first). Sets `collaborator_user_id` and `status = 'confirmed'` in one call. **Backend MUST validate** that `auth.uid()`'s email matches the collaborator record's email before processing — prevents someone who knows a collaborator ID from accepting someone else's invite. |
| GET | `/projects/{projectId}/members` | List project members |
| POST | `/projects/{projectId}/members` | Add project member (admin+ only). Body: `{email, role}`. Auto-adds if account exists, sends invite email if not. |
| PUT | `/projects/{projectId}/members/{memberId}` | Update member role (admin+ only, cannot change owner role) |
| DELETE | `/projects/{projectId}/members/{memberId}` | Remove member (admin+ can remove others, any non-owner can remove self, owner cannot be removed) |
| GET | `/works/{workId}/files` | List files linked to a work |
| POST | `/works/{workId}/files` | Link file to work. Body: `{file_id}` |
| DELETE | `/works/{workId}/files/{linkId}` | Unlink file from work |
| GET | `/works/{workId}/audio` | List audio files linked to a work |
| POST | `/works/{workId}/audio` | Link audio to work. Body: `{audio_file_id}` |
| DELETE | `/works/{workId}/audio/{linkId}` | Unlink audio from work |

### Modified Endpoints
| Method | Path | Change |
|---|---|---|
| POST | `/registry/collaborators/invite` | **Deprecated** — replaced by `/invite-with-stakes`. Keep for backwards compatibility but prefer the composite endpoint. |

---

## Tool Integrations

### OneClick
- **No functional changes** to OneClick's core workflow
- **New data sources:** When selecting documents for analysis, OneClick can browse:
  - Portfolio project files (existing)
  - Files linked to specific works (via work_files)
  - Artist profile documents
- Implementation: Update the document selection step to include a "From Works" source option

### Zoe
- **New contract source:** "From Shared Works" option in the contract/document selector
- When selected, shows works the user is a collaborator on, filtered to those with linked files
- User picks a work → sees its linked contracts → selects one for Zoe to analyze
- Implementation: Add a new fetch path in Zoe's document loading that queries work_files joined with registry_collaborators

---

## UI/UX Specifications

### Color System

**Role badges (project members):**
- Owner: purple (`#a78bfa` / `rgba(139,92,246,0.2)`)
- Admin: blue (`#3b82f6` / `rgba(59,130,246,0.2)`)
- Editor: amber (`#f59e0b` / `rgba(245,158,11,0.2)`)
- Viewer: green (`#10b981` / `rgba(16,185,129,0.2)`)

**Status badges (works):**
- Draft: gray (`#9ca3af` / `rgba(107,114,128,0.15)`)
- Pending: amber (`#f59e0b` / `rgba(245,158,11,0.15)`)
- Registered: green (`#10b981` / `rgba(16,185,129,0.15)`)

**Collaborator status (UI label → DB value):**
- Accepted → `confirmed`: green (`#10b981`)
- Pending → `invited`: amber (`#f59e0b`)
- Declined → `declined`: gray (`#9ca3af`)

**Work status (UI label → DB value):**
- Draft → `draft`
- Pending → `pending_approval`
- Registered → `registered`

### Inline Rename
- Project names and work titles are click-to-edit
- On click: text becomes an input field with current value
- On Enter or blur: saves via API, reverts on error
- On Escape: cancels edit
- Small pencil icon (✎) appears on hover as affordance

### Navigation Flow
```
Portfolio (grid) → Project Detail (tabs) → Work Detail (existing)
                                        ↗
Registry (dashboard) ──────────────────┘
```

### File Labels
- Use "Relevant works" (not "Linked to") when showing which works a file is connected to

---

## Migration Files to Create

All migrations go in `supabase/migrations/`. Naming convention: `YYYYMMDD######_description.sql`. **User runs these manually — do not execute directly.**

1. **`20260403000001_create_project_members.sql`**
   - Create `project_members` table with role CHECK constraint
   - Owner protection triggers: `prevent_owner_deletion` (DELETE) + `prevent_owner_role_change` (UPDATE). Both as regular functions (not SECURITY DEFINER).
   - Auto-create owner entry trigger: `auto_create_project_owner` — **SECURITY DEFINER** (bypasses RLS because no member row exists yet at INSERT time)
   - Enable RLS
   - SELECT policy: all members can see each other
   - INSERT policy: admins+ can add members
   - UPDATE policy: admins+ can change roles
   - DELETE policy: admins+ can remove others, non-owners can remove self

2. **`20260403000002_create_pending_project_invites.sql`**
   - Create `pending_project_invites` table (for inviting non-existing users)
   - Auth trigger: `process_pending_project_invites` — **SECURITY DEFINER** — fires on `auth.users` INSERT, converts pending invites to `project_members` rows
   - Enable RLS (inviter can manage their own invites)

3. **`20260403000003_create_work_files.sql`**
   - Create `work_files` join table (work_id → file_id, UNIQUE constraint)
   - Enable RLS
   - **SELECT policies:** project members can read all; **confirmed** work collaborators only (no access at `invited` status — invite email contains key terms instead)
   - Write policies: editors+ can INSERT/DELETE

4. **`20260403000004_create_work_audio_links.sql`**
   - Create `work_audio_links` join table (work_id → audio_file_id, UNIQUE constraint)
   - Enable RLS
   - **SELECT policies:** project members via project_audio_links; **confirmed** work collaborators only
   - Write policies: editors+ can INSERT/DELETE

5. **`20260403000005_add_content_hash_to_project_files.sql`**
   - Add `content_hash TEXT` column to `project_files`

6. **`20260403000006_update_works_registry.sql`**
   - Add `custom_work_type TEXT` column
   - **Data migration:** `UPDATE works_registry SET status = 'draft' WHERE status = 'disputed'`
   - Drop and recreate `work_type` CHECK constraint (add 'other')
   - Drop and recreate `status` CHECK constraint (remove 'disputed')

7. **`20260403000007_simplify_collaborator_status.sql`**
   - **Data migration:** `UPDATE registry_collaborators SET status = 'declined' WHERE status = 'disputed'`
   - Drop and recreate `status` CHECK constraint: invited, confirmed, declined, revoked
   - Drop `dispute_reason` column

8. **`20260403000008_update_rls_policies.sql`**
   - **Projects table:** SELECT policy for project members (needed for Shared with Me + Project Detail header)
   - Invite visibility by email on `registry_collaborators` (before claim)
   - File access for work collaborators: `confirmed` status only (no access at `invited` — invite email contains key terms instead)
   - Audio file access via work collaboration (was missing entirely)
   - **SELECT on join tables:** `work_files` and `work_audio_links` need their own SELECT policies (subqueries in other policies reference them)
   - Project member cascading access to works, files, audio
   - Role-based write permissions for works (INSERT, UPDATE, **DELETE for owner/admin**)
   - Write permissions for `work_files` and `work_audio_links` (editors+ INSERT/DELETE)

**Note on existing unique constraint:** The `registry_collaborators` table already has a partial unique index `ON (work_id, email) WHERE status != 'revoked'`. This means re-inviting a revoked collaborator to the same work works correctly — the revoked row doesn't conflict with the new invite. No migration needed for gap #8.

---

## Components to Create/Modify

### Modified Pages (major rewrite)
- `src/pages/ProjectDetail.tsx` — already exists at `/projects/:projectId` with basic notes/about tabs. Rewrite to 5-tab layout
- `src/components/project/WorksTab.tsx` — works list + add work dialog
- `src/components/project/FilesTab.tsx` — 4-folder file manager with work linking
- `src/components/project/AudioTab.tsx` — audio list with work linking
- `src/components/project/MembersTab.tsx` — project members + work collaborators
- `src/components/project/SettingsTab.tsx` — project settings + danger zone
- `src/components/project/AddWorkDialog.tsx` — work creation form with audio dropdown
- `src/components/InlineEdit.tsx` — reusable click-to-edit component

### Modified Components
- `src/pages/Portfolio.tsx` — simplify to project card grid
- `src/pages/Registry.tsx` — redesign as tracking dashboard with summary cards + 4 tabs
- `src/pages/WorkDetail.tsx` — remove disputed status, add linked files/audio display, inline rename
- `src/components/registry/CollaborationStatus.tsx` — remove dispute flow, accept/decline only
- `src/components/registry/InviteCollaboratorModal.tsx` — enhanced with splits, stake type, terms
- `src/pages/Zoe.tsx` — add "From Shared Works" contract source
- `src/pages/OneClick.tsx` — add works/portfolio as document source options

### New Hooks
- `src/hooks/useProjectMembers.ts` — CRUD for project_members
- `src/hooks/useWorkFiles.ts` — link/unlink files to works, dedup check
- `src/hooks/useWorkAudio.ts` — link/unlink audio to works

---

## Workflow Examples

These end-to-end scenarios show exactly how a user moves through the system. Each step maps to a specific component and API call.

### Workflow 1: Independent Artist Creates a Project with Works

**Persona:** Yash is an independent artist managing his own music.

1. **Yash opens Portfolio** (`/portfolio`) — sees a grid of his existing projects grouped by artist.
2. **Clicks "+ Create Project"** — `ProjectFormDialog` opens.
   - Selects artist: "Yash Khapre"
   - Enters project name: "Midnight Sessions EP"
   - Clicks "Create"
   - API: `INSERT INTO projects` + trigger auto-creates `project_members` row with role=`owner`
3. **Redirected to Project Detail** (`/projects/{newProjectId}`) — Works tab is empty.
4. **Clicks "+ Add Work"** — `AddWorkDialog` opens.
   - Title: "Late Night Drive"
   - Type: "Single"
   - ISRC: (leaves blank)
   - Audio: clicks "Upload New" → uploads `late_night_drive_master.wav`
   - API: upload to Supabase Storage → `INSERT INTO audio_files` → `INSERT INTO project_audio_links` → `INSERT INTO works_registry` → `INSERT INTO work_audio_links`
5. **Adds two more works** the same way: "Echoes" (EP Track) and "Midnight Interlude" (Other → types "Spoken Word").
6. **Switches to Files tab** — uploads `Producer_Agreement_v2.pdf` into Contracts folder.
   - After upload, prompted: "Link to which works?" → selects "Late Night Drive" and "Echoes"
   - API: `INSERT INTO project_files` (with SHA-256 hash) → `INSERT INTO work_files` (×2)
7. **Result:** Project has 3 works, 1 contract linked to 2 works, 3 audio files each linked to their work.

### Workflow 2: Manager Invites a Collaborator to a Specific Work

**Persona:** Yash manages artist "Amara" and needs to add producer Mike to one track.

1. **Yash opens Project Detail** for Amara's "Summer Vibes LP" → Works tab shows 5 works.
2. **Clicks "Neon Lights"** → navigates to Work Detail (`/tools/registry/{workId}`).
3. **Clicks "Invite Collaborator"** — enhanced `InviteCollaboratorModal` opens.
   - Email: mike@producer.com
   - Name: Mike Peters
   - Role: Producer
   - Stake Type: Master
   - Master %: 15%
   - Notes: "Production fee covered separately, see agreement"
   - Clicks "Send Invite"
   - API: `POST /registry/collaborators/invite-with-stakes` → creates `registry_collaborators` row (status=`invited`) + `ownership_stakes` row (master, 15%) atomically. Sends email via Resend with invite token.
4. **Work status stays Draft** — Yash can continue editing.
5. **Yash clicks "Submit for Approval"** on the CollaborationStatus panel.
   - API: `POST /registry/works/{workId}/submit-for-approval` → status changes to `pending_approval`, invitation email resent.
6. **Result:** Mike has a pending invite. Work is in "Pending" state. Yash sees "1 pending" on the work card in Project Detail.

### Workflow 3: Collaborator Accepts an Invite via Registry

**Persona:** Mike is a producer who received an invite from Yash.

**Path A — via email link (standard flow):**
1. **Mike clicks the email link** → `/tools/registry/invite/{token}` → `InviteClaim` page.
   - If not logged in: redirected to `/auth` then back.
   - API: `POST /registry/collaborators/claim` → sets `collaborator_user_id` to Mike's ID.
   - Redirected to Work Detail for "Neon Lights".
2. **Mike sees a banner:** "You've been listed as Producer on this work. Please review and confirm or decline."
   - Sees his listed stake: Master 15%
   - Does NOT see linked files yet (RLS only grants file access to `confirmed` collaborators)
   - The invite email already contained the stake details, role, and terms — Mike has enough context to decide
3. **Mike clicks "Accept"** — API: `POST /registry/collaborators/{id}/confirm` → status changes to `confirmed`.
   - **Now Mike can see linked files** (contract, audio) — RLS grants access at `confirmed` status.
   - Backend checks: are all collaborators on this work confirmed? If yes → work auto-transitions to `registered`. If not → stays `pending_approval`.

**Path B — via Registry Dashboard (without clicking email link):**
1. **Mike opens Registry Dashboard** (`/tools/registry`).
   - **Action Required tab** shows his pending invite. How: RLS policy matches `registry_collaborators.email` against `auth.users.email` for `invited` status records. No `collaborator_user_id` needed yet.
   - Card shows: "You've been invited as Producer on 'Neon Lights' — invited by Yash Khapre — Master: 15%"
2. **Mike clicks Accept** directly from this page.
   - API: `POST /registry/collaborators/{id}/accept-from-dashboard` → performs claim + confirm atomically (sets `collaborator_user_id = auth.uid()` AND `status = 'confirmed'` in one call).
   - Same auto-transition check as Path A.
3. **Or Mike clicks Decline** → API: `POST /registry/collaborators/{id}/decline` → status changes to `declined`, Yash is notified.

**After acceptance (either path):**
4. **Mike's Registry Dashboard now shows:**
   - Collaborations tab: "Neon Lights" with status "Registered" (or "Pending" if other collaborators remain), his 15% master stake
   - He can click through to see work details, linked contracts, linked audio
   - He does NOT see Amara's project, other works, or other project members

### Workflow 4: Project-Level Member vs Work-Only Collaborator

**Persona:** Yash adds his assistant Sarah to the full project, but producer Mike only has access to one track.

1. **Yash opens Project Detail** → Members tab.
2. **Clicks "+ Invite Member"** — enters Sarah's email, selects role "Viewer".
   - API: `POST /projects/{projectId}/members` with `{email: "sarah@mgmt.com", role: "viewer"}`
   - Backend checks: Sarah's email matches an existing user → auto-adds to `project_members` (no accept/decline flow for project membership).
   - If Sarah had no account: email sent inviting her to sign up. On signup, auto-added with role=viewer.
3. **Sarah logs in** and navigates to her Portfolio.
   - She sees "Summer Vibes LP" under the **"Shared with Me"** section (she's a member, not the owner).
   - The card shows her role badge: Viewer (green).
   - She opens it → sees ALL 5 works, ALL files, ALL audio, ALL members.
   - She cannot edit anything (Viewer role), but she can view everything.
4. **Meanwhile, Mike** (work-only collaborator from Workflow 3):
   - Opens his Registry → Collaborations tab → sees only "Neon Lights".
   - Clicks into Work Detail → sees the work's ownership, linked contracts, linked audio.
   - He does NOT see the project, other works, Sarah, or any project-level data.
   - If Mike tries to navigate to `/projects/{projectId}`, the page shows an access denied message (RLS blocks the project query since he's not in `project_members`).
5. **RLS check order:**
   - Sarah accessing a file: `project_members` check → role=viewer → ✓ read access to all project files
   - Mike accessing a file: `project_members` check → ✗ not a member → `work_files` + `registry_collaborators` check → file is linked to his work AND he's confirmed → ✓ read access to that file only

### Workflow 5: Uploading a Duplicate Contract (Same Project)

**Persona:** Yash uploads the same contract twice within the same project.

1. **Yash is in Project Detail** for "Summer Vibes LP" → Files tab → Contracts folder.
2. **Clicks Upload** → selects `Producer_Agreement_v2.pdf` from his computer.
3. **Frontend computes SHA-256 hash** of the file before uploading.
4. **API checks within the same project only:** `SELECT * FROM project_files WHERE content_hash = '{hash}' AND project_id = '{currentProjectId}'`
5. **Match found:** The same file already exists in this project's Contracts folder.
6. **Prompt appears:** "This file already exists in this project. Link the existing file to additional works instead of uploading a duplicate?"
   - **"Link existing"** → shows work picker, creates `work_files` reference(s) to the existing file (no new upload)
   - **"Upload anyway"** → uploads as a new file with its own ID

**Cross-project note:** If Yash uploads the same file in a different project, no dedup check fires — cross-project files are independent. Each project gets its own `project_files` row pointing to its own storage object, even if the content is identical. This avoids RLS conflicts where a file's `project_id` wouldn't match the accessing project.

### Workflow 6: Zoe Analyzes a Contract from a Shared Work

**Persona:** Mike wants Zoe to analyze the producer agreement for "Neon Lights".

1. **Mike opens Zoe** (`/tools/zoe`).
2. **In the document selector**, Mike sees a new source option: "From Shared Works".
3. **Clicks "From Shared Works"** — shows works where Mike is a confirmed collaborator:
   - "Neon Lights" (Summer Vibes LP) — 1 contract
4. **Selects "Neon Lights"** → sees linked files: `Producer_Agreement_v2.pdf`.
5. **Selects the contract** → Zoe loads the full document markdown.
6. **Mike asks:** "What are my royalty split terms in this contract?"
7. **Zoe analyzes** and responds with extracted terms, payment schedules, etc.
8. **Data flow:** Zoe query → `work_files` JOIN `project_files` WHERE `work_id` IN (Mike's confirmed collaborations) → fetches file content from Supabase Storage.

### Workflow 7: OneClick Reads from Portfolio Data

**Persona:** Yash wants to run royalty calculations for "Late Night Drive".

1. **Yash opens OneClick** (`/tools/oneclick`).
2. **Selects artist:** "Yash Khapre".
3. **Document selection step** now shows sources:
   - "Project files" (existing) — shows all projects for this artist
   - "From Works" (new) — shows all works grouped by project
   - "Artist documents" (existing)
4. **Yash selects "From Works"** → sees "Midnight Sessions EP" → "Late Night Drive".
5. **Sees files linked to this work:** `Producer_Agreement_v2.pdf`, any split sheets.
6. **Selects the contract** → OneClick proceeds with its existing analysis pipeline.
7. **No changes to OneClick's core logic** — only the document source selector is updated.

### Workflow 8: Creating a Project with Multiple Works (Hybrid Flow)

**Persona:** Yash is setting up a new album project.

1. **Yash clicks "+ Create Project"** on Portfolio.
2. **ProjectFormDialog** opens — enters name: "Urban Nights Album", selects artist.
3. **After clicking "Create"**, redirected to the new Project Detail page.
4. **Works tab is empty** with a call-to-action: "No works yet. Add your first work."
5. **Yash clicks "+ Add Work" repeatedly** to create 10 tracks:
   - Each time the dialog opens, previous entries are saved. He can add works one-by-one at his own pace.
   - For 3 tracks he uploads audio files. For 7 he leaves audio empty (to add later).
   - For track 8 he selects type "Other" and types "Remix".
6. **Result:** 10 draft works, 3 with audio linked. Yash can now upload contracts, invite collaborators, and submit works for approval at his own pace.

### Workflow 9: Inline Renaming a Work

**Persona:** Yash realizes "Untitled Track 3" needs a proper name.

1. **In Project Detail → Works tab**, Yash hovers over "Untitled Track 3" — a small ✎ icon appears next to the title.
2. **Clicks the title** — text becomes an editable input field, pre-filled with "Untitled Track 3".
3. **Types "Midnight Interlude"** and presses Enter.
   - API: `PUT /registry/works/{workId}` with `{ title: "Midnight Interlude" }`
   - On success: input reverts to text display with the new name.
   - On error: input reverts to the old name, toast shows error message.
4. **Same interaction works on the Project Detail header** for renaming the project itself.

### Workflow 10: Revoking a Collaborator

**Persona:** Yash needs to remove Mike from "Neon Lights" after a business disagreement.

1. **Yash opens Work Detail** for "Neon Lights" → Collaboration panel.
2. **Clicks the revoke button (×)** next to Mike's name. Confirmation dialog: "Remove Mike Peters as collaborator? Their 15% master stake will be deleted."
3. **Confirms** — API: `POST /registry/collaborators/{id}/revoke`
   - Backend: sets collaborator status to `revoked`, deletes the associated `ownership_stakes` row (15% master).
   - If work was `registered`, it reverts to `draft` (ownership changed, remaining collaborators need to re-confirm).
4. **Mike loses access immediately** — RLS only grants access to `confirmed` collaborators. Mike's Registry Collaborations tab no longer shows "Neon Lights".
5. **Yash sees the freed-up 15%** and can either redistribute it or invite a new collaborator.

### Workflow 11: Registering a Work with Zero Collaborators

**Persona:** Yash has a solo work with no collaborators — he owns 100%.

1. **Yash creates "Solo Interlude"** in his project — it's in `draft` status.
2. **He adds ownership stakes:** Master 100% (himself), Publishing 100% (himself).
3. **No collaborators to invite** — the "Submit for Approval" button is hidden (it requires at least 1 collaborator).
4. **Instead, a "Register" button appears** — available only when collaborator count = 0.
5. **Yash clicks "Register"** — API: `PUT /registry/works/{workId}` with `{ status: 'registered' }`. Direct `draft` → `registered` transition, no approval flow needed.
6. **Work is now Registered.** If Yash later adds a collaborator, the work reverts to `draft` and enters the normal approval flow.

---

## Implementation Notes

Handle these during execution — not spec-level issues but details to get right:

1. **pending_project_invites management:** Add a DELETE endpoint (`DELETE /projects/{projectId}/pending-invites/{id}`) so admins can retract invites to wrong emails. Add an expiry column (default 7 days) and a cleanup job or check-on-access pattern.

2. **pending_project_invites RLS:** Admins+ on the project should be able to SELECT and DELETE pending invites. The inviter should also be able to manage their own.

3. **AddWorkDialog atomicity:** The 5-step sequence (upload → audio_files → project_audio_links → works_registry → work_audio_links) should handle partial failures gracefully. If work creation fails after audio upload, the orphaned audio file is harmless (it's still linked to the project via project_audio_links). Consider a composite backend endpoint if this becomes a UX issue, but for now, sequential frontend calls with error handling are acceptable.

4. **Admin demoting other admins:** This is intentional — admins can change each other's roles. Owner is the only protected role. Worth documenting in a user-facing tooltip on the role dropdown.

5. **updated_at trigger on project_members:** Add a `BEFORE UPDATE` trigger to auto-set `updated_at = now()`. Same pattern as other tables in the codebase.

6. **auto_create_project_owner and service role:** The trigger uses `auth.uid()` which is NULL for service-role operations. If projects are ever created via backend scripts or admin tools, the owner row won't be created. For now this is fine — projects are only created via the frontend (authenticated users). If service-role project creation is needed later, use `INSERT INTO project_members` explicitly in the backend code.

7. **Existing upload policies:** Verify that existing RLS policies on `project_files` and `audio_files` already allow editors+ to INSERT. If not, add INSERT policies during the RLS migration.

8. **Case-insensitive email matching:** The `LOWER()` fix in the RLS policy should also be applied in all backend email comparison logic (claim, accept-from-dashboard, decline). Use `LOWER()` consistently.

---

## Out of Scope

- AI-powered contract extraction to auto-fill splits (future enhancement)
- Dispute/flag flow for collaborators (future enhancement if requested)
- Real-time notifications / push (uses existing notification system)
- Changes to board/task management
- Changes to team cards
- Direct Supabase database changes (migration files only)
- Ownership transfer (changing project owner) — not addressed, can be added later
- Collaborator self-decline after accepting — not allowed by design; only owner can revoke
- Cross-project file deduplication — files are independent per project to avoid RLS conflicts
