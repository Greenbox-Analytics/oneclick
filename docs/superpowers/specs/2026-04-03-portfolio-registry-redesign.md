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
- Replace with a clean card grid grouped by artist → year
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
  - On upload: check if user already has access to a file with same hash
  - If duplicate found: prompt "You already have access to this file via [Work X]. Link it instead of uploading a duplicate?"
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
- Remove button (×) — owner only
- "+ Invite Member" button

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
- Danger Zone: Delete project button with confirmation (owner only)

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

### Table Modifications

#### `project_files` — add content hash
```sql
ALTER TABLE project_files ADD COLUMN content_hash TEXT;
```

#### `works_registry` — add custom work type
```sql
ALTER TABLE works_registry ADD COLUMN custom_work_type TEXT;
```
The `work_type` enum gains 'other'. When `work_type = 'other'`, `custom_work_type` stores the user's free text.

#### `works_registry` — status enum update
Remove 'disputed' from the status check constraint. New valid values: `draft`, `pending_approval`, `registered`.

#### `registry_collaborators` — simplify status
Remove 'disputed' status. Valid values: `invited`, `confirmed`, `declined`, `revoked`.
Remove `dispute_reason` column.

### RLS Policy Updates

#### File access via work collaboration
```
-- Collaborators can read files linked to their works
SELECT on project_files WHERE id IN (
  SELECT file_id FROM work_files WHERE work_id IN (
    SELECT work_id FROM registry_collaborators
    WHERE collaborator_user_id = auth.uid() AND status = 'confirmed'
  )
)
```

#### Project member access
```
-- Project members can read all works in their project
SELECT on works_registry WHERE project_id IN (
  SELECT project_id FROM project_members WHERE user_id = auth.uid()
)

-- Project members can read all files in their project
SELECT on project_files WHERE project_id IN (
  SELECT project_id FROM project_members WHERE user_id = auth.uid()
)
```

#### Role-based write access
```
-- Editors+ can create/update works
INSERT/UPDATE on works_registry WHERE project_id IN (
  SELECT project_id FROM project_members
  WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
)

-- Admins+ can manage members
INSERT/UPDATE/DELETE on project_members WHERE project_id IN (
  SELECT project_id FROM project_members
  WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
)
```

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

All migrations go in `supabase/migrations/`. Naming convention: `YYYYMMDD######_description.sql`.

1. **`20260403000001_create_project_members.sql`**
   - Create `project_members` table
   - RLS policies for project member access
   - Auto-create owner entry when project is created (trigger)

2. **`20260403000002_create_work_files.sql`**
   - Create `work_files` join table
   - RLS: project members can read all; work collaborators can read their work's files

3. **`20260403000003_create_work_audio_links.sql`**
   - Create `work_audio_links` join table
   - RLS: same pattern as work_files

4. **`20260403000004_add_content_hash_to_project_files.sql`**
   - Add `content_hash` TEXT column to `project_files`

5. **`20260403000005_update_works_registry.sql`**
   - Add `custom_work_type` TEXT column
   - Update `work_type` check constraint to include 'other'
   - Update `status` check constraint to remove 'disputed'

6. **`20260403000006_simplify_collaborator_status.sql`**
   - Update `registry_collaborators` status check to: invited, confirmed, declined, revoked
   - Drop `dispute_reason` column

7. **`20260403000007_update_rls_policies.sql`**
   - File access via work collaboration
   - Project member cascading access to works, files, audio
   - Role-based write permissions

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

1. **Mike clicks the email link** → `/tools/registry/invite/{token}` → `InviteClaim` page.
   - If not logged in: redirected to `/auth` then back.
   - API: `POST /registry/collaborators/claim` → sets `collaborator_user_id` to Mike's ID.
   - Redirected to Work Detail for "Neon Lights".
2. **Mike sees a banner:** "You've been listed as Producer on this work. Please review and confirm or decline."
   - Sees his listed stake: Master 15%
   - Sees linked contract: `Producer_Agreement_v2.pdf` (readable because he's a confirmed collaborator via RLS)
3. **Mike clicks "Accept"** — API: `POST /registry/collaborators/{id}/confirm` → status changes to `confirmed`.
4. **Alternatively, Mike opens Registry Dashboard** (`/tools/registry`) without clicking the email link.
   - **Action Required tab** shows: "You've been invited as Producer on 'Neon Lights' — invited by Yash Khapre — Master: 15%"
   - Mike clicks **Accept** directly from this page (no navigation needed).
   - Or clicks **Decline** → status changes to `declined`, Yash is notified.
5. **After acceptance:** If all collaborators have accepted, work status automatically transitions to `registered`.
6. **Mike's Registry Dashboard now shows:**
   - Collaborations tab: "Neon Lights" with status "Registered", his 15% master stake
   - He can click through to see work details, linked contracts, linked audio
   - He does NOT see Amara's project, other works, or other project members

### Workflow 4: Project-Level Member vs Work-Only Collaborator

**Persona:** Yash adds his assistant Sarah to the full project, but producer Mike only has access to one track.

1. **Yash opens Project Detail** → Members tab.
2. **Clicks "+ Invite Member"** — enters Sarah's email, selects role "Viewer".
   - API: `INSERT INTO project_members` (role=`viewer`)
   - Sarah gets an email invite to join the project.
3. **Sarah logs in** and navigates to her Portfolio.
   - She sees "Summer Vibes LP" in her portfolio (because she's a project member).
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

### Workflow 5: Uploading a Duplicate Contract

**Persona:** Yash uploads the same contract to a different project.

1. **Yash is in Project Detail** for "Debut Album" → Files tab → Contracts folder.
2. **Clicks Upload** → selects `Producer_Agreement_v2.pdf` from his computer.
3. **Frontend computes SHA-256 hash** of the file before uploading.
4. **API checks:** `SELECT * FROM project_files WHERE content_hash = '{hash}' AND project_id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())`
5. **Match found:** The same file exists in "Summer Vibes LP" project.
6. **Prompt appears:** "You already have this file in 'Summer Vibes LP → Contracts'. Would you like to link the existing file instead of uploading a duplicate?"
   - **"Link existing"** → creates a `work_files` reference to the existing file (no new upload, no storage duplication)
   - **"Upload anyway"** → uploads as a new file with its own ID (for cases where the user wants separate copies)

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

---

## Out of Scope

- AI-powered contract extraction to auto-fill splits (future enhancement)
- Dispute/flag flow for collaborators (future enhancement if requested)
- Real-time notifications / push (uses existing notification system)
- Changes to board/task management
- Changes to team cards
- Direct Supabase database changes (migration files only)
