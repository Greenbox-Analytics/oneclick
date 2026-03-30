# Rights & Ownership Registry v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a collaborative Rights & Ownership Registry with TeamCard-based identity, Notion-like rich notes, and per-work scoped collaboration. Artists are per-user but verified via TeamCard linking. Collaborators see each other's curated identity (TeamCard) while keeping private notes separate (Option C merge).

**Architecture:** Projects are the container; works are the granular unit of ownership. Five registry tables (`works_registry`, `ownership_stakes`, `licensing_rights`, `registry_agreements`, `registry_collaborators`) with collaborator-aware RLS. A `team_cards` table stores each user's shareable identity — populated from onboarding defaults (first name, last name, preferred name, email), configurable in profile settings. When a collaborator verifies via email, their TeamCard data populates the identity fields on the inviter's artist entry (Option C: TeamCard wins on shared fields, private notes stay yours). Rich notes use BlockNote (Notion-like block editor) with folder organization, available on both artist profiles ("My Notes" section) and project workspaces (About page + Notes section). Verified badges appear in artist profiles and workspace. Notifications surface in the Workspace tool's notifications tab.

**Collaboration Flow:**
1. Creator registers a work (status: `draft`) and adds ownership stakes
2. Creator invites collaborators by email for each stake
3. Collaborator receives email — if they have an account, they also get an in-app notification
4. Collaborator logs in (or signs up), claims invitation, gets linked to the work
5. On first link, a verification request is sent — collaborator's TeamCard populates the inviter's artist entry
6. Verified badge appears on artist profile and in workspace
7. Creator clicks "Submit for Approval" → status: `pending_approval`
8. Each collaborator confirms or disputes their stake
9. When ALL confirm → `registered`. If ANY dispute → `disputed`

**Tech Stack:** Supabase (PostgreSQL + RLS), FastAPI (Python backend), Resend (invitation/verification emails), React + TypeScript + React Query (frontend), ReportLab (PDF generation), BlockNote + @blocknote/shadcn (rich notes editor), shadcn/ui + Tailwind (UI components)

**Known Limitations (out of scope for this plan):**
- **Auth model:** The existing backend uses `user_id` as a query parameter (not JWT extraction). This is a pre-existing pattern across all endpoints in main.py. Migrating to proper JWT auth is a separate security initiative that should cover the entire backend, not just these endpoints.
- **Pagination:** Works, notes, and collaborator lists are unpaginated. Fine for MVP; add cursor-based pagination when lists exceed ~100 items.
- **Rate limiting:** No rate limiting on invitation emails. Add application-level rate limiting (e.g., max 10 invites per work per hour) as a follow-up.
- **V1 plan file dependency:** Tasks 10-12 and 15 reference `docs/superpowers/plans/2026-03-28-rights-registry.md` for component source code (unchanged from v1). **Do not delete the v1 plan file** until all tasks are implemented.
- **Expired invitation cleanup:** Expired `registry_collaborators` rows remain in the DB. Add a scheduled cleanup job (e.g., `DELETE FROM registry_collaborators WHERE status = 'invited' AND expires_at < now() - interval '30 days'`) as a follow-up.
- **CORS:** Already configured in `main.py` (lines 54-59) with `CORSMiddleware`, `allow_credentials=True`, `allow_methods=["*"]`. No action needed.
- **Async on sync supabase calls:** Service functions are `async def` but supabase-py `.execute()` is synchronous. This is a pre-existing pattern across all backend endpoints. Migrating to the async supabase client is a separate initiative.
- **Transaction boundaries:** `submit_for_approval` performs multiple DB operations without a transaction. supabase-py doesn't support transactions natively. Acceptable for MVP; refactor to Postgres RPC function if atomicity issues arise.

---

## File Structure

```
supabase/migrations/
  20260329000000_create_rights_registry.sql     # 7 tables: 5 registry + team_cards + notifications
  20260329100000_add_notes_and_verification.sql  # notes, note_folders, artists verification cols, project about

src/backend/registry/
  __init__.py
  models.py                                      # Pydantic models: registry + TeamCard + notes
  service.py                                     # CRUD + collaboration + TeamCard + notes + verification
  router.py                                      # All endpoints
  emails.py                                      # Invitation + verification emails via Resend
  pdf_generator.py                               # Proof of ownership with approval status

src/backend/main.py                              # Mount registry router (modify)

src/lib/apiFetch.ts                                # Shared fetch helper (used by all registry hooks)
src/hooks/useRegistry.ts                         # React Query hooks for registry CRUD + collaboration
src/hooks/useTeamCard.ts                         # TeamCard CRUD + visibility settings
src/hooks/useNotes.ts                            # Notes + folders CRUD
src/hooks/useRegistryNotifications.ts            # Notification queries + mark-read

src/components/notes/
  NotesEditor.tsx                                # BlockNote rich editor wrapper
  NotesSidebar.tsx                               # Folder tree + note list
  NotesView.tsx                                  # Combined sidebar + editor (reusable)

src/components/registry/
  OwnershipPanel.tsx                             # Master + publishing with approval status
  LicensingPanel.tsx                             # Licensing rights management
  AgreementsPanel.tsx                            # Timestamped agreements timeline
  CollaborationStatus.tsx                        # Approval workflow status display
  InviteCollaboratorModal.tsx                    # Invite stakeholders by email or from roster
  ProofOfOwnership.tsx                           # Export button

src/components/workspace/RegistryNotifications.tsx
src/components/profile/TeamCardSettings.tsx      # TeamCard field visibility + preview

src/pages/Registry.tsx                           # Work listing + pending reviews
src/pages/WorkDetail.tsx                         # Work detail + approval workflow
src/pages/InviteClaim.tsx                        # Invitation claim handler
src/pages/ProjectDetail.tsx                      # Project about + notes (NEW)

src/App.tsx                                      # Add routes (modify)
src/pages/Tools.tsx                              # Add Registry card (modify)
src/pages/Profile.tsx                            # Add TeamCard settings section (modify)
src/pages/ArtistProfile.tsx                      # Verified badge + Option C merge + notes (modify)
```

---

### Task 1: Database Schema — Registry, TeamCards & Notifications

**Goal:** Create the 5 registry tables, team_cards table, and registry_notifications table with collaborator-aware RLS.

**Files:**
- Create: `supabase/migrations/20260329000000_create_rights_registry.sql`

**Acceptance Criteria:**
- [ ] `works_registry` with `project_id NOT NULL` and status flow: draft → pending_approval → registered / disputed
- [ ] `ownership_stakes` with percentage validation
- [ ] `licensing_rights` with status tracking
- [ ] `registry_agreements` with immutable timestamps
- [ ] `registry_collaborators` with invite/confirm/dispute workflow, invite tokens, 48h expiry
- [ ] `team_cards` with one-per-user constraint, default-populated fields, visibility settings
- [ ] `registry_notifications` for in-app collaboration notifications
- [ ] RLS policies: collaborators can SELECT works/stakes they're involved in; creator retains full CRUD
- [ ] TeamCard RLS: users can only read/write their own; collaborators can read linked TeamCards

**Verify:** `supabase db push` succeeds

**Steps:**

- [ ] **Step 0: Remove old v1 migration files**

The v1 plan created migration files with the same timestamps but different content. Delete them before creating v2 files:

```bash
rm -f supabase/migrations/20260329000000_create_rights_registry.sql
rm -f supabase/migrations/20260329100000_add_artist_claim.sql
```

- [ ] **Step 1: Create the migration file**

Create `supabase/migrations/20260329000000_create_rights_registry.sql`:

```sql
-- ============================================================
-- Rights & Ownership Registry with Collaboration Layer
-- + TeamCards + Notifications
-- ============================================================

-- 1. team_cards — each user's shareable collaboration identity
create table team_cards (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null unique references auth.users(id) on delete cascade,
  -- Core identity (populated from onboarding, cannot be empty)
  -- display_name = preferred name (from profiles.given_name) or first+last fallback
  display_name text not null,
  first_name text not null,
  last_name text not null,
  email text not null,
  -- Optional shareable fields
  avatar_url text,
  bio text,
  phone text,
  website text,
  company text,
  industry text,
  -- Structured links
  social_links jsonb not null default '{}',
  dsp_links jsonb not null default '{}',
  custom_links jsonb not null default '[]',
  -- Which fields are visible to collaborators
  visible_fields jsonb not null default '["display_name", "email", "first_name", "last_name"]',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2. works_registry
create table works_registry (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  artist_id uuid not null references artists(id) on delete cascade,
  project_id uuid not null references projects(id) on delete cascade,
  title text not null,
  work_type text not null default 'single'
    check (work_type in ('single', 'ep_track', 'album_track', 'composition')),
  isrc text,
  iswc text,
  upc text,
  release_date date,
  status text not null default 'draft'
    check (status in ('draft', 'pending_approval', 'registered', 'disputed')),
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 3. ownership_stakes
create table ownership_stakes (
  id uuid primary key default gen_random_uuid(),
  work_id uuid not null references works_registry(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  stake_type text not null check (stake_type in ('master', 'publishing')),
  holder_name text not null,
  holder_role text not null,
  percentage numeric(5,2) not null check (percentage > 0 and percentage <= 100),
  holder_email text,
  holder_ipi text,
  publisher_or_label text,
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 4. licensing_rights
create table licensing_rights (
  id uuid primary key default gen_random_uuid(),
  work_id uuid not null references works_registry(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  license_type text not null
    check (license_type in ('sync', 'mechanical', 'performance', 'print', 'digital', 'exclusive', 'non_exclusive', 'other')),
  licensee_name text not null,
  licensee_email text,
  territory text default 'worldwide',
  start_date date not null,
  end_date date,
  terms text,
  status text not null default 'active'
    check (status in ('active', 'expired', 'terminated', 'pending')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 5. registry_agreements (immutable — no update/delete)
create table registry_agreements (
  id uuid primary key default gen_random_uuid(),
  work_id uuid not null references works_registry(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  agreement_type text not null
    check (agreement_type in ('ownership_transfer', 'split_agreement', 'license_grant', 'amendment', 'termination')),
  title text not null,
  description text,
  effective_date date not null,
  parties jsonb not null default '[]',
  file_id uuid references project_files(id) on delete set null,
  document_hash text,
  created_at timestamptz not null default now()
);

-- 6. registry_collaborators — invitation + approval + verification tracking
create table registry_collaborators (
  id uuid primary key default gen_random_uuid(),
  work_id uuid not null references works_registry(id) on delete cascade,
  stake_id uuid references ownership_stakes(id) on delete set null,
  invited_by uuid not null references auth.users(id),
  collaborator_user_id uuid references auth.users(id),
  email text not null,
  name text not null,
  role text not null,
  status text not null default 'invited'
    check (status in ('invited', 'confirmed', 'disputed', 'revoked')),
  invite_token uuid not null default gen_random_uuid(),
  dispute_reason text,
  expires_at timestamptz not null default (now() + interval '48 hours'),
  invited_at timestamptz not null default now(),
  responded_at timestamptz
);

-- 7. registry_notifications — in-app collaboration notifications
create table registry_notifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  work_id uuid references works_registry(id) on delete set null,
  type text not null
    check (type in ('invitation', 'confirmation', 'dispute', 'status_change', 'verification')),
  title text not null,
  message text not null,
  read boolean not null default false,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

-- ============================================================
-- Row Level Security
-- ============================================================

-- team_cards: own read/write, collaborators can read linked cards
alter table team_cards enable row level security;

create policy "Users can manage their own team card"
  on team_cards for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Collaborators can read linked team cards"
  on team_cards for select
  using (
    user_id in (
      select collaborator_user_id from registry_collaborators
      where invited_by = auth.uid() and collaborator_user_id is not null and status != 'revoked'
      union
      select invited_by from registry_collaborators
      where collaborator_user_id = auth.uid() and status != 'revoked'
    )
  );

-- works_registry: owner + collaborators can read, only owner can write
alter table works_registry enable row level security;

create policy "Owner or collaborator can read works"
  on works_registry for select
  using (
    auth.uid() = user_id
    or id in (
      select work_id from registry_collaborators
      where collaborator_user_id = auth.uid()
        and status != 'revoked'
    )
  );

create policy "Owner can insert works"
  on works_registry for insert
  with check (auth.uid() = user_id);

create policy "Owner can update works"
  on works_registry for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete works"
  on works_registry for delete
  using (auth.uid() = user_id);

-- ownership_stakes: owner + collaborators can read, only owner can write
alter table ownership_stakes enable row level security;

create policy "Owner or collaborator can read stakes"
  on ownership_stakes for select
  using (
    auth.uid() = user_id
    or work_id in (
      select work_id from registry_collaborators
      where collaborator_user_id = auth.uid()
        and status != 'revoked'
    )
  );

create policy "Owner can insert stakes"
  on ownership_stakes for insert
  with check (auth.uid() = user_id);

create policy "Owner can update stakes"
  on ownership_stakes for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete stakes"
  on ownership_stakes for delete
  using (auth.uid() = user_id);

-- licensing_rights: owner + collaborators can read, only owner can write
alter table licensing_rights enable row level security;

create policy "Owner or collaborator can read licenses"
  on licensing_rights for select
  using (
    auth.uid() = user_id
    or work_id in (
      select work_id from registry_collaborators
      where collaborator_user_id = auth.uid()
        and status != 'revoked'
    )
  );

create policy "Owner can insert licenses"
  on licensing_rights for insert
  with check (auth.uid() = user_id);

create policy "Owner can update licenses"
  on licensing_rights for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete licenses"
  on licensing_rights for delete
  using (auth.uid() = user_id);

-- registry_agreements: owner + collaborators can read, only owner can insert
alter table registry_agreements enable row level security;

create policy "Owner or collaborator can read agreements"
  on registry_agreements for select
  using (
    auth.uid() = user_id
    or work_id in (
      select work_id from registry_collaborators
      where collaborator_user_id = auth.uid()
        and status != 'revoked'
    )
  );

create policy "Owner can insert agreements"
  on registry_agreements for insert
  with check (auth.uid() = user_id);

-- registry_collaborators: inviter can manage, collaborator can read/update own
alter table registry_collaborators enable row level security;

create policy "Inviter can manage collaborators"
  on registry_collaborators for all
  using (auth.uid() = invited_by);

create policy "Collaborator can read own invitations"
  on registry_collaborators for select
  using (auth.uid() = collaborator_user_id);

create policy "Collaborator can update own invitation status"
  on registry_collaborators for update
  using (auth.uid() = collaborator_user_id)
  with check (auth.uid() = collaborator_user_id);

-- Restrict what collaborators can change on their own row.
-- Only status, dispute_reason, and responded_at are mutable by the collaborator.
-- All other fields are frozen to prevent impersonation or scope changes.
create or replace function restrict_collaborator_self_update()
returns trigger as $$
begin
  if auth.uid() = old.collaborator_user_id and auth.uid() != old.invited_by then
    new.work_id := old.work_id;
    new.stake_id := old.stake_id;
    new.invited_by := old.invited_by;
    new.collaborator_user_id := old.collaborator_user_id;
    new.email := old.email;
    new.name := old.name;
    new.role := old.role;
    new.invite_token := old.invite_token;
    new.expires_at := old.expires_at;
    new.invited_at := old.invited_at;
  end if;
  return new;
end;
$$ language plpgsql;

create trigger registry_collaborators_restrict_self_update
  before update on registry_collaborators
  for each row execute function restrict_collaborator_self_update();

-- registry_notifications: user can manage own notifications
alter table registry_notifications enable row level security;

create policy "Users can manage own notifications"
  on registry_notifications for all
  using (auth.uid() = user_id);

-- ============================================================
-- Indexes
-- ============================================================
create index idx_team_cards_user_id on team_cards(user_id);
create index idx_works_registry_user_id on works_registry(user_id);
create index idx_works_registry_artist_id on works_registry(artist_id);
create index idx_works_registry_project_id on works_registry(project_id);
create index idx_ownership_stakes_work_id on ownership_stakes(work_id);
create index idx_ownership_stakes_user_id on ownership_stakes(user_id);
create index idx_licensing_rights_work_id on licensing_rights(work_id);
create index idx_registry_agreements_work_id on registry_agreements(work_id);
create index idx_registry_collaborators_work_id on registry_collaborators(work_id);
create index idx_registry_collaborators_user_id on registry_collaborators(collaborator_user_id);
create index idx_registry_collaborators_token on registry_collaborators(invite_token);
create index idx_registry_collaborators_email on registry_collaborators(email);
-- Prevent duplicate invitations: one email per work (excluding revoked)
create unique index idx_registry_collaborators_unique_invite
  on registry_collaborators(work_id, email)
  where status != 'revoked';
create index idx_registry_notifications_user_id on registry_notifications(user_id);

-- ============================================================
-- Updated-at triggers
-- ============================================================
create or replace function update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger team_cards_updated_at
  before update on team_cards
  for each row execute function update_updated_at_column();

create trigger works_registry_updated_at
  before update on works_registry
  for each row execute function update_updated_at_column();

create trigger ownership_stakes_updated_at
  before update on ownership_stakes
  for each row execute function update_updated_at_column();

-- Enforce stake percentage totals at the DB level (prevents race conditions).
-- Application-layer validation is still useful for user-facing error messages,
-- but this trigger is the authoritative constraint.
create or replace function validate_stake_total()
returns trigger as $$
declare
  total numeric;
begin
  select coalesce(sum(percentage), 0) into total
  from ownership_stakes
  where work_id = new.work_id
    and stake_type = new.stake_type
    and id != coalesce(new.id, '00000000-0000-0000-0000-000000000000'::uuid);

  if (total + new.percentage) > 100.0 then
    raise exception 'Total %% for % would exceed 100%% (current: %%%, adding: %%%)',
      new.stake_type, total, new.percentage;
  end if;
  return new;
end;
$$ language plpgsql;

create trigger ownership_stakes_validate_total
  before insert or update on ownership_stakes
  for each row execute function validate_stake_total();

create trigger licensing_rights_updated_at
  before update on licensing_rights
  for each row execute function update_updated_at_column();

-- ============================================================
-- Utility: look up user ID by email (O(1) via auth.users index)
-- Used by the backend instead of iterating list_users()
-- ============================================================
-- SECURITY NOTE: This function is SECURITY DEFINER, meaning any authenticated user
-- can check if an email exists on the platform (user enumeration). This is a deliberate
-- trade-off required for the collaboration invite flow. If this becomes a concern,
-- add rate limiting at the application layer on the invite endpoint.
create or replace function get_user_id_by_email(lookup_email text)
returns uuid as $$
  select id from auth.users where email = lower(lookup_email) limit 1;
$$ language sql security definer;

-- ============================================================
-- Auto-create TeamCard on user signup (via trigger)
-- Populated from profiles table (onboarding data)
-- Email sourced from auth.users (profiles does not store email)
-- display_name = preferred name (given_name) or first+last fallback
-- ============================================================
create or replace function create_default_team_card()
returns trigger as $$
declare
  user_email text;
begin
  -- Email lives in auth.users, not profiles
  select email into user_email from auth.users where id = new.id;

  insert into team_cards (user_id, display_name, first_name, last_name, email)
  values (
    new.id,
    coalesce(nullif(new.given_name, ''), new.first_name || ' ' || new.last_name, new.full_name, ''),
    coalesce(new.first_name, ''),
    coalesce(new.last_name, ''),
    coalesce(user_email, '')
  )
  on conflict (user_id) do nothing;
  return new;
end;
$$ language plpgsql security definer;

create trigger on_profile_created_create_team_card
  after insert on profiles
  for each row execute function create_default_team_card();

-- Backfill TeamCards for all existing users (trigger only fires on new INSERTs).
-- Without this, every existing user gets 404 on /registry/teamcard.
insert into team_cards (user_id, display_name, first_name, last_name, email)
select
  p.id,
  coalesce(nullif(p.given_name, ''), p.first_name || ' ' || p.last_name, p.full_name, ''),
  coalesce(p.first_name, ''),
  coalesce(p.last_name, ''),
  coalesce(u.email, '')
from profiles p
join auth.users u on u.id = p.id
on conflict (user_id) do nothing;
```

- [ ] **Step 2: Apply migration**

Run: `supabase db push`
Expected: 7 tables created with all policies, indexes, and triggers

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260329000000_create_rights_registry.sql
git commit -m "feat: add rights registry schema with team cards, notifications, and collaboration-aware RLS"
```

---

### Task 2: Database Schema — Notes, Verification & Project About

**Goal:** Create notes/folders tables for Notion-like rich notes, add verification columns to artists, and add rich about field to projects.

**Files:**
- Create: `supabase/migrations/20260329100000_add_notes_and_verification.sql`

**Acceptance Criteria:**
- [ ] `note_folders` with nested folder support (parent_folder_id), scoped to artist or project
- [ ] `notes` with BlockNote JSON content, scoped to artist or project, linkable to folders
- [ ] Artists gain `linked_user_id`, `verified`, `verified_at` columns (replacing old claim columns)
- [ ] Projects gain `about_content` jsonb column for rich about page
- [ ] RLS: only the note/folder owner can CRUD; verified collaborators can read project-scoped notes
- [ ] Updated artists RLS: verified (linked) user can view + edit their artist profile
- [ ] Updated projects/project_files RLS: verified user can view projects and files

**Verify:** `supabase db push` succeeds

**Steps:**

- [ ] **Step 1: Create the migration file**

Create `supabase/migrations/20260329100000_add_notes_and_verification.sql`:

```sql
-- ============================================================
-- Notes System (Notion-like) + Artist Verification + Project About
-- ============================================================

-- 1. note_folders — nested folder structure for notes
create table note_folders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  artist_id uuid references artists(id) on delete cascade,
  project_id uuid references projects(id) on delete cascade,
  name text not null,
  parent_folder_id uuid references note_folders(id) on delete cascade,
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (
    (artist_id is not null and project_id is null)
    or (artist_id is null and project_id is not null)
  )
);

-- 2. notes — rich content stored as BlockNote JSON blocks
create table notes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  folder_id uuid references note_folders(id) on delete set null,
  artist_id uuid references artists(id) on delete cascade,
  project_id uuid references projects(id) on delete cascade,
  title text not null default 'Untitled',
  content jsonb not null default '[]',
  pinned boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (
    (artist_id is not null and project_id is null)
    or (artist_id is null and project_id is not null)
  )
);

-- 3. Add verification columns to artists (replaces old claim system)
alter table public.artists
  add column if not exists linked_user_id uuid references auth.users(id) on delete set null,
  add column if not exists verified boolean not null default false,
  add column if not exists verified_at timestamptz;

-- 4. Add rich about content to projects
alter table public.projects
  add column if not exists about_content jsonb not null default '[]';

-- ============================================================
-- RLS for notes system
-- ============================================================

alter table note_folders enable row level security;
alter table notes enable row level security;

-- Note folders: owner can manage; collaborators can read project-scoped folders.
-- NOTE: Collaborator read access requires at least one work on the project with them
-- as a collaborator (via registry_collaborators). If a project has no works yet,
-- only the project owner can see its notes. This is intentional — collaboration
-- access is always scoped through works, not granted at the project level directly.
create policy "Owner can read own folders"
  on note_folders for select
  using (auth.uid() = user_id);

create policy "Owner can insert folders with valid scope"
  on note_folders for insert
  with check (
    auth.uid() = user_id
    and (
      (artist_id is not null and exists (
        select 1 from artists where id = note_folders.artist_id and user_id = auth.uid()
      ))
      or
      (project_id is not null and exists (
        select 1 from projects join artists on artists.id = projects.artist_id
        where projects.id = note_folders.project_id and artists.user_id = auth.uid()
      ))
    )
  );

create policy "Owner can update own folders"
  on note_folders for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete own folders"
  on note_folders for delete
  using (auth.uid() = user_id);

create policy "Collaborators can read project note folders"
  on note_folders for select
  using (
    project_id is not null
    and exists (
      select 1 from works_registry w
      join registry_collaborators rc on rc.work_id = w.id
      where w.project_id = note_folders.project_id
        and rc.collaborator_user_id = auth.uid()
        and rc.status != 'revoked'
    )
  );

-- Notes: owner can manage; verified collaborators can read project-scoped notes
-- Owner can manage their own notes, but only for artists/projects they own
create policy "Owner can read own notes"
  on notes for select
  using (auth.uid() = user_id);

create policy "Owner can insert notes with valid scope"
  on notes for insert
  with check (
    auth.uid() = user_id
    and (
      -- If artist-scoped, caller must own the artist
      (artist_id is not null and exists (
        select 1 from artists where id = notes.artist_id and user_id = auth.uid()
      ))
      or
      -- If project-scoped, caller must own the project (via artist)
      (project_id is not null and exists (
        select 1 from projects join artists on artists.id = projects.artist_id
        where projects.id = notes.project_id and artists.user_id = auth.uid()
      ))
    )
  );

create policy "Owner can update own notes"
  on notes for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete own notes"
  on notes for delete
  using (auth.uid() = user_id);

create policy "Collaborators can read project notes"
  on notes for select
  using (
    project_id is not null
    and exists (
      select 1 from works_registry w
      join registry_collaborators rc on rc.work_id = w.id
      where w.project_id = notes.project_id
        and rc.collaborator_user_id = auth.uid()
        and rc.status != 'revoked'
    )
  );

-- ============================================================
-- Updated artists RLS: verified (linked) user can view + edit
-- ============================================================

drop policy if exists "Users can view their own artists" on public.artists;
drop policy if exists "Users can update their own artists" on public.artists;

create policy "Users can view their own artists"
  on public.artists for select
  using (
    auth.uid() = user_id
    or auth.uid() = linked_user_id
  );

create policy "Users can update their own artists"
  on public.artists for update
  using (
    auth.uid() = user_id
    or auth.uid() = linked_user_id
  );

-- ============================================================
-- Updated projects RLS: verified artists can view projects
-- ============================================================

drop policy if exists "Users can view own projects" on public.projects;

create policy "Users can view own projects"
  on public.projects for select
  using (
    exists (
      select 1 from public.artists
      where artists.id = projects.artist_id
      and (artists.user_id = auth.uid() or artists.linked_user_id = auth.uid())
    )
  );

-- ============================================================
-- Updated project_files RLS: verified artists can view files
-- ============================================================

drop policy if exists "Users can view own project files" on public.project_files;

create policy "Users can view own project files"
  on public.project_files for select
  using (
    exists (
      select 1 from public.projects
      join public.artists on artists.id = projects.artist_id
      where projects.id = project_files.project_id
      and (artists.user_id = auth.uid() or artists.linked_user_id = auth.uid())
    )
  );

-- ============================================================
-- Indexes
-- ============================================================
create index idx_note_folders_user_id on note_folders(user_id);
create index idx_note_folders_artist_id on note_folders(artist_id);
create index idx_note_folders_project_id on note_folders(project_id);
create index idx_notes_user_id on notes(user_id);
create index idx_notes_folder_id on notes(folder_id);
create index idx_notes_artist_id on notes(artist_id);
create index idx_notes_project_id on notes(project_id);
create index idx_artists_linked_user_id on public.artists(linked_user_id);
create index idx_artists_email on public.artists(email);

-- Triggers
create trigger note_folders_updated_at
  before update on note_folders
  for each row execute function update_updated_at_column();

create trigger notes_updated_at
  before update on notes
  for each row execute function update_updated_at_column();

-- Reset verified flag when linked_user_id becomes null (e.g. user deletes account)
create or replace function reset_verified_on_unlink()
returns trigger as $$
begin
  if new.linked_user_id is null and old.linked_user_id is not null then
    new.verified := false;
    new.verified_at := null;
  end if;
  return new;
end;
$$ language plpgsql;

create trigger artists_reset_verified
  before update on public.artists
  for each row execute function reset_verified_on_unlink();
```

- [ ] **Step 2: Apply migration**

Run: `supabase db push`
Expected: 2 new tables (notes, note_folders), 3 columns on artists, 1 column on projects

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260329100000_add_notes_and_verification.sql
git commit -m "feat: add notes system, artist verification columns, and project about content"
```

---

### Task 3: Backend Models & Service — Registry + TeamCard + Notes + Verification

**Goal:** Pydantic models and service layer for all CRUD operations: registry, collaboration, TeamCard, notes, and verification.

**Files:**
- Create: `src/backend/registry/__init__.py`
- Create: `src/backend/registry/models.py`
- Create: `src/backend/registry/service.py`

**Acceptance Criteria:**
- [ ] Models for works, stakes, licenses, agreements, collaborator invitations
- [ ] Models for TeamCard create/update, note create/update, folder create/update
- [ ] CRUD service for all registry tables
- [ ] `invite_collaborator` creates collaborator row, returns invite token, checks if user exists
- [ ] `claim_invitation` links a user_id to an invitation by token
- [ ] `confirm_stake` / `dispute_stake` update collaborator status
- [ ] `check_and_update_work_status` auto-transitions: all confirmed → registered, any disputed → disputed
- [ ] `submit_for_approval` validates all stakes have collaborators and transitions to pending_approval
- [ ] TeamCard CRUD: get, update, get visible fields for collaborator view
- [ ] Notes CRUD: create/update/delete notes and folders, list by artist or project
- [ ] `verify_artist_link`: links a collaborator's user_id to an artist entry, sets verified=true
- [ ] Notification service: create, list, mark read

**Pre-requisite:** `pip install pydantic[email]` (required for `EmailStr` in CollaboratorInvite)

**Verify:** `cd src/backend && python -c "from registry.models import WorkCreate, CollaboratorInvite, TeamCardUpdate, NoteCreate; print('OK')"`

**Steps:**

- [ ] **Step 1: Create module init**

Create `src/backend/registry/__init__.py` (empty file).

- [ ] **Step 2: Create models**

Create `src/backend/registry/models.py`:

```python
"""Pydantic models for the Rights & Ownership Registry, TeamCards, and Notes."""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date


# --- Works ---

class WorkCreate(BaseModel):
    artist_id: str
    project_id: str
    title: str
    work_type: str = "single"
    isrc: Optional[str] = None
    iswc: Optional[str] = None
    upc: Optional[str] = None
    release_date: Optional[date] = None
    notes: Optional[str] = None


class WorkUpdate(BaseModel):
    title: Optional[str] = None
    work_type: Optional[str] = None
    project_id: Optional[str] = None
    isrc: Optional[str] = None
    iswc: Optional[str] = None
    upc: Optional[str] = None
    release_date: Optional[date] = None
    status: Optional[str] = None
    notes: Optional[str] = None


# --- Ownership Stakes ---

class StakeCreate(BaseModel):
    work_id: str
    stake_type: str
    holder_name: str
    holder_role: str
    percentage: float
    holder_email: Optional[str] = None
    holder_ipi: Optional[str] = None
    publisher_or_label: Optional[str] = None
    notes: Optional[str] = None


class StakeUpdate(BaseModel):
    stake_type: Optional[str] = None
    holder_name: Optional[str] = None
    holder_role: Optional[str] = None
    percentage: Optional[float] = None
    holder_email: Optional[str] = None
    holder_ipi: Optional[str] = None
    publisher_or_label: Optional[str] = None
    notes: Optional[str] = None


# --- Licensing Rights ---

class LicenseCreate(BaseModel):
    work_id: str
    license_type: str
    licensee_name: str
    licensee_email: Optional[str] = None
    territory: str = "worldwide"
    start_date: date
    end_date: Optional[date] = None
    terms: Optional[str] = None


class LicenseUpdate(BaseModel):
    license_type: Optional[str] = None
    licensee_name: Optional[str] = None
    licensee_email: Optional[str] = None
    territory: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    terms: Optional[str] = None
    status: Optional[str] = None


# --- Agreements ---

class PartyInput(BaseModel):
    name: str
    role: str
    email: Optional[str] = None


class AgreementCreate(BaseModel):
    work_id: str
    agreement_type: str
    title: str
    description: Optional[str] = None
    effective_date: date
    parties: List[PartyInput]
    file_id: Optional[str] = None
    document_hash: Optional[str] = None


# --- Collaboration ---

class CollaboratorInvite(BaseModel):
    work_id: str
    stake_id: Optional[str] = None
    email: EmailStr
    name: str
    role: str


class DisputeRequest(BaseModel):
    reason: str


# --- TeamCard ---

class TeamCardUpdate(BaseModel):
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    social_links: Optional[dict] = None
    dsp_links: Optional[dict] = None
    custom_links: Optional[list] = None
    visible_fields: Optional[list] = None


# --- Notes ---

class NoteCreate(BaseModel):
    title: str = "Untitled"
    content: list = []
    artist_id: Optional[str] = None
    project_id: Optional[str] = None
    folder_id: Optional[str] = None
    pinned: bool = False


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[list] = None
    folder_id: Optional[str] = None
    pinned: Optional[bool] = None


class FolderCreate(BaseModel):
    name: str
    artist_id: Optional[str] = None
    project_id: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: int = 0


class FolderUpdate(BaseModel):
    name: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: Optional[int] = None


# --- Project About ---

class ProjectAboutUpdate(BaseModel):
    about_content: list = []
```

- [ ] **Step 3: Create service layer**

Create `src/backend/registry/service.py`:

```python
"""Service layer for the Rights & Ownership Registry with collaboration, TeamCards, notes, and verification."""

import hashlib
from supabase import Client


# ============================================================
# Works
# ============================================================

async def get_works(db: Client, user_id: str, artist_id: str = None):
    query = db.table("works_registry").select("*").eq("user_id", user_id)
    if artist_id:
        query = query.eq("artist_id", artist_id)
    result = query.order("created_at", desc=True).execute()
    return result.data


async def get_works_as_collaborator(db: Client, user_id: str):
    """Get ALL works where user is a collaborator (not the creator) — any status."""
    collab_rows = (
        db.table("registry_collaborators")
        .select("work_id")
        .eq("collaborator_user_id", user_id)
        .neq("status", "revoked")
        .execute()
    )
    work_ids = [r["work_id"] for r in (collab_rows.data or [])]
    if not work_ids:
        return []
    result = (
        db.table("works_registry")
        .select("*")
        .in_("id", work_ids)
        .neq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data


async def get_works_by_project(db: Client, user_id: str, project_id: str):
    result = (
        db.table("works_registry")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at")
        .execute()
    )
    return result.data


async def get_work(db: Client, user_id: str, work_id: str):
    result = (
        db.table("works_registry")
        .select("*")
        .eq("id", work_id)
        .single()
        .execute()
    )
    return result.data


async def create_work(db: Client, user_id: str, data: dict):
    # Verify the artist belongs to this user before allowing work creation
    artist = db.table("artists").select("id").eq("id", data.get("artist_id")).eq("user_id", user_id).single().execute()
    if not artist.data:
        return None  # artist_id doesn't belong to this user
    data["user_id"] = user_id
    result = db.table("works_registry").insert(data).execute()
    return result.data[0] if result.data else None


async def update_work(db: Client, user_id: str, work_id: str, data: dict):
    # Block edits on registered works — requires re-approval if modified
    work = db.table("works_registry").select("status").eq("id", work_id).eq("user_id", user_id).single().execute()
    if not work.data:
        return None
    if work.data["status"] == "registered":
        # Reset to draft if owner edits a registered work — forces re-approval
        data["status"] = "draft"
    result = (
        db.table("works_registry")
        .update(data)
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_work(db: Client, user_id: str, work_id: str):
    # Notify collaborators before deletion so they don't silently lose access
    collabs = (
        db.table("registry_collaborators")
        .select("collaborator_user_id, work_id")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    work = db.table("works_registry").select("title").eq("id", work_id).single().execute()
    work_title = (work.data or {}).get("title", "Untitled")
    for c in (collabs.data or []):
        if c.get("collaborator_user_id"):
            await create_notification(
                db,
                user_id=c["collaborator_user_id"],
                work_id=None,  # work is being deleted
                notification_type="status_change",
                title="Work deleted",
                message=f'The work "{work_title}" has been deleted by its owner.',
                metadata={"work_title": work_title},
            )
    return db.table("works_registry").delete().eq("id", work_id).eq("user_id", user_id).execute().data


# ============================================================
# Ownership Stakes
# ============================================================

async def get_stakes(db: Client, user_id: str, work_id: str):
    result = (
        db.table("ownership_stakes")
        .select("*")
        .eq("work_id", work_id)
        .order("stake_type")
        .order("percentage", desc=True)
        .execute()
    )
    return result.data


async def validate_stake_percentage(
    db: Client, user_id: str, work_id: str, stake_type: str,
    new_percentage: float, exclude_stake_id: str = None,
):
    existing = (
        db.table("ownership_stakes")
        .select("id, percentage")
        .eq("work_id", work_id)
        .eq("user_id", user_id)
        .eq("stake_type", stake_type)
        .execute()
    )
    total = sum(
        row["percentage"] for row in (existing.data or [])
        if row["id"] != exclude_stake_id
    )
    return (total + new_percentage) <= 100.0


async def create_stake(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("ownership_stakes").insert(data).execute()
    return result.data[0] if result.data else None


async def update_stake(db: Client, user_id: str, stake_id: str, data: dict):
    result = (
        db.table("ownership_stakes")
        .update(data)
        .eq("id", stake_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_stake(db: Client, user_id: str, stake_id: str):
    return db.table("ownership_stakes").delete().eq("id", stake_id).eq("user_id", user_id).execute().data


# ============================================================
# Licensing Rights
# ============================================================

async def get_licenses(db: Client, user_id: str, work_id: str):
    result = (
        db.table("licensing_rights")
        .select("*")
        .eq("work_id", work_id)
        .order("start_date", desc=True)
        .execute()
    )
    return result.data


async def create_license(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("licensing_rights").insert(data).execute()
    return result.data[0] if result.data else None


async def update_license(db: Client, user_id: str, license_id: str, data: dict):
    result = (
        db.table("licensing_rights").update(data)
        .eq("id", license_id).eq("user_id", user_id).execute()
    )
    return result.data[0] if result.data else None


async def delete_license(db: Client, user_id: str, license_id: str):
    return db.table("licensing_rights").delete().eq("id", license_id).eq("user_id", user_id).execute().data


# ============================================================
# Agreements
# ============================================================

async def get_agreements(db: Client, user_id: str, work_id: str):
    result = (
        db.table("registry_agreements")
        .select("*")
        .eq("work_id", work_id)
        .order("effective_date", desc=True)
        .execute()
    )
    return result.data


async def create_agreement(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("registry_agreements").insert(data).execute()
    return result.data[0] if result.data else None


# ============================================================
# Collaboration
# ============================================================

async def get_collaborators(db: Client, work_id: str):
    result = (
        db.table("registry_collaborators")
        .select("*")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .order("invited_at", desc=True)
        .execute()
    )
    return result.data


async def check_user_exists(db: Client, email: str):
    """Check if a user with this email exists on the platform. Returns user_id or None.
    Uses a database function that queries auth.users directly (O(1) via index)
    instead of iterating paginated list_users() results."""
    result = db.rpc("get_user_id_by_email", {"lookup_email": email}).execute()
    if result.data:
        return result.data
    return None


async def invite_collaborator(db: Client, invited_by: str, data: dict, work_title: str = ""):
    """Create a collaborator invitation. Auto-links if user exists on platform."""
    data["invited_by"] = invited_by

    existing_user_id = await check_user_exists(db, data["email"])
    if existing_user_id:
        data["collaborator_user_id"] = existing_user_id

    result = db.table("registry_collaborators").insert(data).execute()
    collab = result.data[0] if result.data else None

    if collab and existing_user_id:
        inviter_profile = db.table("profiles").select("full_name").eq("id", invited_by).single().execute()
        inviter_name = (inviter_profile.data or {}).get("full_name") or "Someone"
        await create_notification(
            db,
            user_id=existing_user_id,
            work_id=data.get("work_id"),
            notification_type="invitation",
            title="New collaboration request",
            message=f'{inviter_name} listed you as {data.get("role", "collaborator")} on "{work_title}"',
            metadata={"inviter_name": inviter_name, "work_title": work_title, "role": data.get("role")},
        )

        # Auto-verify: link collaborator's user_id to matching artist entries
        await auto_verify_artist(db, invited_by, data["email"], existing_user_id)

    return collab


async def is_invite_expired(collab: dict) -> bool:
    """Check if an invitation has passed its 48h expiry."""
    from datetime import datetime, timezone
    expires_at = collab.get("expires_at")
    if not expires_at:
        return False
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    return datetime.now(timezone.utc) > expires_at


async def claim_invitation(db: Client, invite_token: str, user_id: str):
    """Link a logged-in user to their invitation by token. Returns (collab, error)."""
    result = (
        db.table("registry_collaborators")
        .select("*")
        .eq("invite_token", invite_token)
        .single()
        .execute()
    )
    if not result.data:
        return None, "not_found"

    collab = result.data
    if collab.get("collaborator_user_id") and collab["collaborator_user_id"] != user_id:
        return None, "already_claimed"

    if await is_invite_expired(collab):
        return None, "expired"

    updated = (
        db.table("registry_collaborators")
        .update({"collaborator_user_id": user_id})
        .eq("id", collab["id"])
        .execute()
    )
    claimed = updated.data[0] if updated.data else collab

    # Auto-verify artist link on claim
    await auto_verify_artist(db, collab["invited_by"], collab["email"], user_id)

    # Notify the work creator that someone claimed their invitation
    work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
    if work_row.data:
        await create_notification(
            db,
            user_id=work_row.data["user_id"],
            work_id=collab["work_id"],
            notification_type="invitation",
            title="Invitation claimed",
            message=f'{collab["name"]} accepted your invitation for "{work_row.data["title"]}"',
            metadata={"collaborator_name": collab["name"]},
        )

    return claimed, None


async def confirm_stake(db: Client, collaborator_id: str, user_id: str):
    """Collaborator confirms their stake. Can change from disputed→confirmed
    as long as the work is still in pending_approval status."""
    from datetime import datetime, timezone
    # Verify the work is still pending (allow changing decisions)
    collab_check = (
        db.table("registry_collaborators")
        .select("work_id, status")
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .single()
        .execute()
    )
    if not collab_check.data:
        return None
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", collab_check.data["work_id"])
        .single()
        .execute()
    )
    if work.data and work.data["status"] == "registered":
        return None  # Can't change after fully registered

    result = (
        db.table("registry_collaborators")
        .update({
            "status": "confirmed",
            "dispute_reason": None,
            "responded_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .execute()
    )
    if result.data:
        collab = result.data[0]
        await check_and_update_work_status(db, collab["work_id"])
        # Notify the work creator that a collaborator responded
        work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
        if work_row.data:
            await create_notification(
                db,
                user_id=work_row.data["user_id"],
                work_id=collab["work_id"],
                notification_type="confirmation",
                title="Stake confirmed",
                message=f'{collab["name"]} confirmed their stake on "{work_row.data["title"]}"',
                metadata={"collaborator_name": collab["name"]},
            )
    return result.data[0] if result.data else None


async def dispute_stake(db: Client, collaborator_id: str, user_id: str, reason: str):
    """Collaborator disputes their stake. Can change from confirmed→disputed
    as long as the work is still in pending_approval status."""
    from datetime import datetime, timezone
    # Verify the work is still pending (allow changing decisions)
    collab_check = (
        db.table("registry_collaborators")
        .select("work_id, status")
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .single()
        .execute()
    )
    if not collab_check.data:
        return None
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", collab_check.data["work_id"])
        .single()
        .execute()
    )
    if work.data and work.data["status"] == "registered":
        return None  # Can't change after fully registered

    result = (
        db.table("registry_collaborators")
        .update({
            "status": "disputed",
            "dispute_reason": reason,
            "responded_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .execute()
    )
    if result.data:
        collab = result.data[0]
        await check_and_update_work_status(db, collab["work_id"])
        # Notify the work creator about the dispute
        work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
        if work_row.data:
            await create_notification(
                db,
                user_id=work_row.data["user_id"],
                work_id=collab["work_id"],
                notification_type="dispute",
                title="Stake disputed",
                message=f'{collab["name"]} disputed their stake on "{work_row.data["title"]}": {reason}',
                metadata={"collaborator_name": collab["name"], "reason": reason},
            )
    return result.data[0] if result.data else None


async def check_and_update_work_status(db: Client, work_id: str):
    collabs = (
        db.table("registry_collaborators")
        .select("status")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    if not collabs.data:
        return
    statuses = [c["status"] for c in collabs.data]
    if any(s == "disputed" for s in statuses):
        db.table("works_registry").update({"status": "disputed"}).eq("id", work_id).execute()
    elif all(s == "confirmed" for s in statuses):
        db.table("works_registry").update({"status": "registered"}).eq("id", work_id).execute()


async def resend_invitation(db: Client, user_id: str, collaborator_id: str):
    """Resend an expired or pending invitation — generates new token, resets expiry."""
    import uuid
    from datetime import datetime, timezone, timedelta
    # Verify the caller is the inviter
    collab = (
        db.table("registry_collaborators")
        .select("*")
        .eq("id", collaborator_id)
        .eq("invited_by", user_id)
        .single()
        .execute()
    )
    if not collab.data:
        return None, "not_found"
    if collab.data["status"] not in ("invited", "revoked"):
        return None, f"Cannot resend: collaborator status is {collab.data['status']}"

    new_token = str(uuid.uuid4())
    updated = (
        db.table("registry_collaborators")
        .update({
            "invite_token": new_token,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
            "status": "invited",
        })
        .eq("id", collaborator_id)
        .execute()
    )
    return updated.data[0] if updated.data else None, None


async def revoke_collaborator(db: Client, user_id: str, collaborator_id: str):
    """Revoke a collaborator invitation. Only the inviter can do this."""
    result = (
        db.table("registry_collaborators")
        .update({"status": "revoked"})
        .eq("id", collaborator_id)
        .eq("invited_by", user_id)
        .execute()
    )
    if result.data:
        collab = result.data[0]
        await check_and_update_work_status(db, collab["work_id"])
    return result.data[0] if result.data else None


async def submit_for_approval(db: Client, user_id: str, work_id: str):
    # NOTE: This function performs multiple DB operations (reset collabs, update status,
    # send notifications) without a transaction. supabase-py doesn't support transactions
    # natively. If atomicity is critical, refactor into a Postgres RPC function.
    # For MVP, partial failure is acceptable — the worst case is collabs reset without
    # the status changing, which the user can retry.
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", work_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not work.data:
        return None, "Work not found"
    if work.data["status"] not in ("draft", "disputed"):
        return None, f"Cannot submit: work is already {work.data['status']}"

    collabs = (
        db.table("registry_collaborators")
        .select("id")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    if not collabs.data:
        return None, "No collaborators invited — add at least one before submitting"

    db.table("registry_collaborators").update(
        {"status": "invited", "dispute_reason": None, "responded_at": None}
    ).eq("work_id", work_id).eq("status", "disputed").execute()

    result = (
        db.table("works_registry")
        .update({"status": "pending_approval"})
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    updated_work = result.data[0] if result.data else None

    # Re-notify all collaborators (especially those reset from disputed→invited)
    if updated_work:
        work_title = updated_work.get("title", "Untitled")
        all_collabs = (
            db.table("registry_collaborators")
            .select("collaborator_user_id, email, name, invite_token")
            .eq("work_id", work_id)
            .eq("status", "invited")
            .execute()
        )
        inviter_profile = db.table("profiles").select("full_name").eq("id", user_id).single().execute()
        inviter_name = (inviter_profile.data or {}).get("full_name") or "The project owner"
        for c in (all_collabs.data or []):
            if c.get("collaborator_user_id"):
                await create_notification(
                    db,
                    user_id=c["collaborator_user_id"],
                    work_id=work_id,
                    notification_type="invitation",
                    title="Work resubmitted for approval",
                    message=f'{inviter_name} resubmitted "{work_title}" for your review',
                    metadata={"inviter_name": inviter_name, "work_title": work_title},
                )
        # Attach collaborator list so the router can re-send emails
        updated_work["_renotify_collabs"] = all_collabs.data or []

    return updated_work, None


# ============================================================
# Full Work Data
# ============================================================

async def get_work_full(db: Client, user_id: str, work_id: str):
    work = await get_work(db, user_id, work_id)
    if not work:
        return None
    stakes = await get_stakes(db, user_id, work_id)
    licenses = await get_licenses(db, user_id, work_id)
    agreements = await get_agreements(db, user_id, work_id)
    collaborators = await get_collaborators(db, work_id)
    return {
        **work,
        "stakes": stakes or [],
        "licenses": licenses or [],
        "agreements": agreements or [],
        "collaborators": collaborators or [],
    }


# ============================================================
# Verification — Option C merge logic
# ============================================================

async def auto_verify_artist(db: Client, manager_user_id: str, email: str, collaborator_user_id: str):
    """When a collaborator is linked, find any artist entries the manager created with
    that email and set linked_user_id + verified. This powers Option C merge.

    NOTE: This intentionally verifies ALL artist profiles the manager has with this email.
    If a manager has multiple artist entries for the same person (different stage names),
    all get verified at once. This is by design — one real person, one identity.

    Fallback: if the invite email doesn't match any artist profile, also tries
    the collaborator's auth email (from auth.users) in case the invite was sent
    to a different address than what the manager put on the artist profile."""
    from datetime import datetime, timezone

    # Primary match: artist email matches invite email
    artists = (
        db.table("artists")
        .select("id, linked_user_id")
        .eq("user_id", manager_user_id)
        .eq("email", email)
        .execute()
    )

    # Fallback: try the collaborator's actual auth email
    if not artists.data:
        tc = db.table("team_cards").select("email").eq("user_id", collaborator_user_id).single().execute()
        auth_email = (tc.data or {}).get("email", "")
        if auth_email and auth_email.lower() != email.lower():
            artists = (
                db.table("artists")
                .select("id, linked_user_id")
                .eq("user_id", manager_user_id)
                .eq("email", auth_email)
                .execute()
            )
    for artist in (artists.data or []):
        if not artist.get("linked_user_id"):
            db.table("artists").update({
                "linked_user_id": collaborator_user_id,
                "verified": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", artist["id"]).execute()

            # Notify the collaborator about verification
            await create_notification(
                db,
                user_id=collaborator_user_id,
                work_id=None,
                notification_type="verification",
                title="Artist profile verified",
                message=f"Your identity has been verified on an artist profile. Your TeamCard info is now visible to collaborators.",
                metadata={},
            )


async def get_artists_with_teamcards(db: Client, user_id: str):
    """Batch fetch: all artists for a user with TeamCard overlays in 2 queries (not N+1)."""
    artists_result = db.table("artists").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    artists = artists_result.data or []
    if not artists:
        return []

    # Batch fetch all relevant TeamCards in one query
    linked_ids = [a["linked_user_id"] for a in artists if a.get("linked_user_id") and a.get("verified")]
    tc_map = {}
    if linked_ids:
        tc_result = db.table("team_cards").select("*").in_("user_id", linked_ids).execute()
        for tc in (tc_result.data or []):
            visible = tc.get("visible_fields") or []
            filtered = {"user_id": tc["user_id"]}
            for field in visible:
                if field in tc:
                    filtered[field] = tc[field]
            tc_map[tc["user_id"]] = filtered

    results = []
    for a in artists:
        merged = {**a, "teamcard": None}
        if a.get("linked_user_id") and a.get("verified") and a["linked_user_id"] in tc_map:
            merged["teamcard"] = tc_map[a["linked_user_id"]]
        results.append(merged)
    return results


async def get_artist_with_teamcard(db: Client, artist_id: str):
    """Get an artist with Option C merge: if verified, overlay TeamCard fields on shared identity fields."""
    artist = db.table("artists").select("*").eq("id", artist_id).single().execute()
    if not artist.data:
        return None

    a = artist.data
    result = {**a, "teamcard": None}

    if a.get("linked_user_id") and a.get("verified"):
        tc = (
            db.table("team_cards")
            .select("*")
            .eq("user_id", a["linked_user_id"])
            .single()
            .execute()
        )
        if tc.data:
            card = tc.data
            visible = card.get("visible_fields") or []
            # Build filtered teamcard with only visible fields
            filtered_card = {"user_id": card["user_id"]}
            for field in visible:
                if field in card:
                    filtered_card[field] = card[field]
            result["teamcard"] = filtered_card
    return result


# ============================================================
# TeamCard
# ============================================================

async def get_team_card(db: Client, user_id: str):
    result = db.table("team_cards").select("*").eq("user_id", user_id).single().execute()
    return result.data


async def update_team_card(db: Client, user_id: str, data: dict):
    # email cannot be changed
    data.pop("email", None)
    # display_name, first_name, last_name cannot be empty
    for field in ("display_name", "first_name", "last_name"):
        if field in data and not data[field]:
            data.pop(field)
    result = (
        db.table("team_cards")
        .update(data)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def get_collaborator_team_card(db: Client, collaborator_user_id: str):
    """Get a collaborator's TeamCard filtered to only visible fields."""
    tc = db.table("team_cards").select("*").eq("user_id", collaborator_user_id).single().execute()
    if not tc.data:
        return None
    card = tc.data
    visible = card.get("visible_fields") or []
    filtered = {"user_id": card["user_id"], "email": card["email"]}
    for field in visible:
        if field in card:
            filtered[field] = card[field]
    return filtered


# ============================================================
# Notes
# ============================================================

async def get_notes(db: Client, user_id: str, artist_id: str = None, project_id: str = None, folder_id: str = None):
    """List notes. For artist-scoped notes, filters by user_id (private notes).
    For project-scoped notes, does NOT filter by user_id — relies on RLS to allow
    collaborator reads. RLS ensures users only see notes they own OR project notes
    where they're a work collaborator."""
    query = db.table("notes").select("*")
    if artist_id:
        # Artist notes are always private — filter by owner
        query = query.eq("user_id", user_id).eq("artist_id", artist_id)
    elif project_id:
        # Project notes: don't filter by user_id — RLS handles collaborator access
        query = query.eq("project_id", project_id)
    else:
        # No scope — return only user's own notes
        query = query.eq("user_id", user_id)
    if folder_id:
        query = query.eq("folder_id", folder_id)
    result = query.order("pinned", desc=True).order("updated_at", desc=True).execute()
    return result.data


async def get_note(db: Client, user_id: str, note_id: str):
    """Get a single note. RLS handles access control — if the caller can't see it,
    the query returns empty. No user_id filter needed for project-scoped notes."""
    result = db.table("notes").select("*").eq("id", note_id).single().execute()
    return result.data


async def create_note(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("notes").insert(data).execute()
    return result.data[0] if result.data else None


async def update_note(db: Client, user_id: str, note_id: str, data: dict):
    result = (
        db.table("notes")
        .update(data)
        .eq("id", note_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_note(db: Client, user_id: str, note_id: str):
    return db.table("notes").delete().eq("id", note_id).eq("user_id", user_id).execute().data


async def get_folders(db: Client, user_id: str, artist_id: str = None, project_id: str = None):
    """List folders. Same access pattern as get_notes: artist-scoped = private,
    project-scoped = RLS-controlled (collaborators can read)."""
    query = db.table("note_folders").select("*")
    if artist_id:
        query = query.eq("user_id", user_id).eq("artist_id", artist_id)
    elif project_id:
        query = query.eq("project_id", project_id)
    else:
        query = query.eq("user_id", user_id)
    result = query.order("sort_order").order("name").execute()
    return result.data


async def create_folder(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("note_folders").insert(data).execute()
    return result.data[0] if result.data else None


async def update_folder(db: Client, user_id: str, folder_id: str, data: dict):
    result = (
        db.table("note_folders")
        .update(data)
        .eq("id", folder_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_folder(db: Client, user_id: str, folder_id: str):
    return db.table("note_folders").delete().eq("id", folder_id).eq("user_id", user_id).execute().data


# ============================================================
# Project About
# ============================================================

async def get_project_about(db: Client, project_id: str):
    result = db.table("projects").select("about_content").eq("id", project_id).single().execute()
    return (result.data or {}).get("about_content", [])


async def update_project_about(db: Client, user_id: str, project_id: str, about_content: list):
    """Update project about content. Verifies the caller owns the project via artist ownership."""
    # Verify ownership: project → artist → user_id
    project = (
        db.table("projects")
        .select("id, artist_id")
        .eq("id", project_id)
        .single()
        .execute()
    )
    if not project.data:
        return None
    artist = (
        db.table("artists")
        .select("user_id")
        .eq("id", project.data["artist_id"])
        .single()
        .execute()
    )
    if not artist.data or artist.data["user_id"] != user_id:
        return None  # Not the owner

    result = (
        db.table("projects")
        .update({"about_content": about_content})
        .eq("id", project_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ============================================================
# Notifications
# ============================================================

async def create_notification(
    db: Client, user_id: str, work_id: str,
    notification_type: str, title: str, message: str,
    metadata: dict = None,
):
    db.table("registry_notifications").insert({
        "user_id": user_id,
        "work_id": work_id,
        "type": notification_type,
        "title": title,
        "message": message,
        "metadata": metadata or {},
    }).execute()


async def get_notifications(db: Client, user_id: str, unread_only: bool = False):
    query = db.table("registry_notifications").select("*").eq("user_id", user_id)
    if unread_only:
        query = query.eq("read", False)
    result = query.order("created_at", desc=True).limit(50).execute()
    return result.data


async def mark_notification_read(db: Client, user_id: str, notification_id: str):
    db.table("registry_notifications").update({"read": True}).eq("id", notification_id).eq("user_id", user_id).execute()


async def mark_all_notifications_read(db: Client, user_id: str):
    db.table("registry_notifications").update({"read": True}).eq("user_id", user_id).eq("read", False).execute()


# ============================================================
# Utility
# ============================================================

def compute_document_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
```

- [ ] **Step 4: Verify imports**

Run: `cd src/backend && python -c "from registry.models import WorkCreate, CollaboratorInvite, TeamCardUpdate, NoteCreate, FolderCreate, ProjectAboutUpdate; from registry.service import get_works, invite_collaborator, confirm_stake, get_team_card, get_notes; print('All OK')"`
Expected: `All OK`

- [ ] **Step 5: Commit**

```bash
git add src/backend/registry/
git commit -m "feat: add registry models and service with TeamCard, notes, verification, and collaboration"
```

---

### Task 4: Backend Router with All Endpoints

**Goal:** FastAPI router covering registry CRUD, collaboration, TeamCard, notes, and project about. Mounted in main.py.

**Files:**
- Create: `src/backend/registry/router.py`
- Modify: `src/backend/main.py`

**Acceptance Criteria:**
- [ ] Standard CRUD for works, stakes, licenses, agreements
- [ ] Collaboration endpoints: invite, claim, confirm, dispute, submit-for-approval
- [ ] TeamCard endpoints: GET/PUT own card, GET collaborator's visible card
- [ ] Notes endpoints: CRUD notes and folders, scoped by artist_id or project_id
- [ ] Project about endpoint: GET/PUT about_content
- [ ] Artist with TeamCard overlay: GET artist with merged TeamCard data
- [ ] Notifications: list, mark read, mark all read
- [ ] Router mounted at `/registry`

**Verify:** `cd src/backend && python -c "from registry.router import router; print('OK')"`

**Steps:**

- [ ] **Step 1: Create the router**

Create `src/backend/registry/router.py`:

```python
"""FastAPI router for the Rights & Ownership Registry."""

import re
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from registry import service
from registry.models import (
    WorkCreate, WorkUpdate,
    StakeCreate, StakeUpdate,
    LicenseCreate, LicenseUpdate,
    AgreementCreate,
    CollaboratorInvite, DisputeRequest,
    TeamCardUpdate,
    NoteCreate, NoteUpdate, FolderCreate, FolderUpdate,
    ProjectAboutUpdate,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


# ============================================================
# Works
# ============================================================

@router.get("/works")
async def list_works(user_id: str = Query(...), artist_id: Optional[str] = Query(None)):
    works = await service.get_works(_get_supabase(), user_id, artist_id)
    return {"works": works}


@router.get("/works/my-collaborations")
async def list_my_collaborations(user_id: str = Query(...)):
    works = await service.get_works_as_collaborator(_get_supabase(), user_id)
    return {"works": works}


@router.get("/works/by-project/{project_id}")
async def list_works_by_project(project_id: str, user_id: str = Query(...)):
    works = await service.get_works_by_project(_get_supabase(), user_id, project_id)
    return {"works": works}


@router.get("/works/{work_id}")
async def get_work(work_id: str, user_id: str = Query(...)):
    work = await service.get_work(_get_supabase(), user_id, work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.get("/works/{work_id}/full")
async def get_work_full(work_id: str, user_id: str = Query(...)):
    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    return data


@router.post("/works")
async def create_work(body: WorkCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "release_date" in data and data["release_date"]:
        data["release_date"] = data["release_date"].isoformat()
    work = await service.create_work(_get_supabase(), user_id, data)
    if not work:
        raise HTTPException(status_code=500, detail="Failed to create work")
    return work


@router.put("/works/{work_id}")
async def update_work(work_id: str, body: WorkUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "release_date" in data and data["release_date"]:
        data["release_date"] = data["release_date"].isoformat()
    work = await service.update_work(_get_supabase(), user_id, work_id, data)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.delete("/works/{work_id}")
async def delete_work(work_id: str, user_id: str = Query(...)):
    await service.delete_work(_get_supabase(), user_id, work_id)
    return {"ok": True}


@router.post("/works/{work_id}/submit-for-approval")
async def submit_for_approval(work_id: str, user_id: str = Query(...)):
    result, error = await service.submit_for_approval(_get_supabase(), user_id, work_id)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Re-send invitation emails to all collaborators (especially those reset from disputed)
    renotify = (result or {}).pop("_renotify_collabs", [])
    if renotify:
        from registry.emails import send_invitation_email
        profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
        inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
        work_title = (result or {}).get("title", "Untitled Work")
        for c in renotify:
            send_invitation_email(
                recipient_email=c["email"],
                recipient_name=c["name"],
                inviter_name=inviter_name,
                work_title=work_title,
                role="collaborator",
                invite_token=str(c.get("invite_token", "")),
            )

    return result


@router.get("/works/{work_id}/export")
async def export_proof_of_ownership(work_id: str, user_id: str = Query(...)):
    from registry.pdf_generator import generate_proof_of_ownership_pdf
    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    buffer = generate_proof_of_ownership_pdf(data)
    safe_title = re.sub(r"[^a-zA-Z0-9._-]", "_", data.get("title", "work"))
    return StreamingResponse(
        buffer, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Proof_of_Ownership_{safe_title}.pdf"'},
    )


# ============================================================
# Ownership Stakes
# ============================================================

@router.get("/stakes")
async def list_stakes(work_id: str = Query(...), user_id: str = Query(...)):
    stakes = await service.get_stakes(_get_supabase(), user_id, work_id)
    return {"stakes": stakes}


@router.post("/stakes")
async def create_stake(body: StakeCreate, user_id: str = Query(...)):
    valid = await service.validate_stake_percentage(
        _get_supabase(), user_id, body.work_id, body.stake_type, body.percentage
    )
    if not valid:
        raise HTTPException(status_code=400, detail=f"Adding {body.percentage}% would exceed 100% for {body.stake_type}")
    data = body.model_dump(exclude_none=True)
    stake = await service.create_stake(_get_supabase(), user_id, data)
    if not stake:
        raise HTTPException(status_code=500, detail="Failed to create stake")
    return stake


@router.put("/stakes/{stake_id}")
async def update_stake(stake_id: str, body: StakeUpdate, user_id: str = Query(...)):
    if body.percentage is not None:
        existing = (
            _get_supabase().table("ownership_stakes")
            .select("work_id, stake_type").eq("id", stake_id).eq("user_id", user_id)
            .single().execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Stake not found")
        stake_type = body.stake_type or existing.data["stake_type"]
        valid = await service.validate_stake_percentage(
            _get_supabase(), user_id, existing.data["work_id"],
            stake_type, body.percentage, exclude_stake_id=stake_id,
        )
        if not valid:
            raise HTTPException(status_code=400, detail=f"Exceeds 100% for {stake_type}")
    data = body.model_dump(exclude_none=True)
    stake = await service.update_stake(_get_supabase(), user_id, stake_id, data)
    if not stake:
        raise HTTPException(status_code=404, detail="Stake not found")
    return stake


@router.delete("/stakes/{stake_id}")
async def delete_stake(stake_id: str, user_id: str = Query(...)):
    await service.delete_stake(_get_supabase(), user_id, stake_id)
    return {"ok": True}


# ============================================================
# Licensing Rights
# ============================================================

@router.get("/licenses")
async def list_licenses(work_id: str = Query(...), user_id: str = Query(...)):
    licenses = await service.get_licenses(_get_supabase(), user_id, work_id)
    return {"licenses": licenses}


@router.post("/licenses")
async def create_license(body: LicenseCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    lic = await service.create_license(_get_supabase(), user_id, data)
    if not lic:
        raise HTTPException(status_code=500, detail="Failed to create license")
    return lic


@router.put("/licenses/{license_id}")
async def update_license(license_id: str, body: LicenseUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    lic = await service.update_license(_get_supabase(), user_id, license_id, data)
    if not lic:
        raise HTTPException(status_code=404, detail="License not found")
    return lic


@router.delete("/licenses/{license_id}")
async def delete_license(license_id: str, user_id: str = Query(...)):
    await service.delete_license(_get_supabase(), user_id, license_id)
    return {"ok": True}


# ============================================================
# Agreements (immutable)
# ============================================================

@router.get("/agreements")
async def list_agreements(work_id: str = Query(...), user_id: str = Query(...)):
    agreements = await service.get_agreements(_get_supabase(), user_id, work_id)
    return {"agreements": agreements}


@router.post("/agreements")
async def create_agreement(body: AgreementCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "effective_date" in data and data["effective_date"]:
        data["effective_date"] = data["effective_date"].isoformat()
    if "parties" in data:
        data["parties"] = [p if isinstance(p, dict) else p.model_dump() for p in data["parties"]]
    agreement = await service.create_agreement(_get_supabase(), user_id, data)
    if not agreement:
        raise HTTPException(status_code=500, detail="Failed to create agreement")
    return agreement


# ============================================================
# Collaboration
# ============================================================

@router.get("/collaborators")
async def list_collaborators(work_id: str = Query(...), user_id: str = Query(...)):
    """List collaborators. Only the work creator or an existing collaborator can view."""
    db = _get_supabase()
    work = db.table("works_registry").select("user_id").eq("id", work_id).single().execute()
    if not work.data:
        raise HTTPException(status_code=404, detail="Work not found")
    is_creator = work.data["user_id"] == user_id
    is_collab = db.table("registry_collaborators").select("id").eq("work_id", work_id).eq("collaborator_user_id", user_id).neq("status", "revoked").execute()
    if not is_creator and not (is_collab.data):
        raise HTTPException(status_code=403, detail="Not authorized to view collaborators")
    collabs = await service.get_collaborators(db, work_id)
    return {"collaborators": collabs}


@router.post("/collaborators/invite")
async def invite_collaborator(body: CollaboratorInvite, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    work = _get_supabase().table("works_registry").select("title").eq("id", body.work_id).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"

    collab = await service.invite_collaborator(_get_supabase(), user_id, data, work_title=work_title)
    if not collab:
        raise HTTPException(status_code=500, detail="Failed to invite collaborator")

    # Always send email
    from registry.emails import send_invitation_email
    profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
    inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
    send_invitation_email(
        recipient_email=body.email,
        recipient_name=body.name,
        inviter_name=inviter_name,
        work_title=work_title,
        role=body.role,
        invite_token=str(collab.get("invite_token", "")),
    )
    return collab


@router.post("/collaborators/claim")
async def claim_invitation(invite_token: str = Query(...), user_id: str = Query(...)):
    collab, error = await service.claim_invitation(_get_supabase(), invite_token, user_id)
    if error == "expired":
        raise HTTPException(status_code=410, detail="Invitation expired — ask the project owner to resend")
    if error:
        raise HTTPException(status_code=404, detail="Invitation not found or already claimed")
    return collab


@router.post("/collaborators/{collaborator_id}/confirm")
async def confirm_stake(collaborator_id: str, user_id: str = Query(...)):
    collab = await service.confirm_stake(_get_supabase(), collaborator_id, user_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/dispute")
async def dispute_stake(collaborator_id: str, body: DisputeRequest, user_id: str = Query(...)):
    collab = await service.dispute_stake(_get_supabase(), collaborator_id, user_id, body.reason)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/revoke")
async def revoke_collaborator(collaborator_id: str, user_id: str = Query(...)):
    """Revoke a collaborator invitation. Only the inviter can do this."""
    collab = await service.revoke_collaborator(_get_supabase(), user_id, collaborator_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/resend")
async def resend_invitation(collaborator_id: str, user_id: str = Query(...)):
    """Resend an expired or pending invitation with a fresh token and 48h expiry."""
    collab, error = await service.resend_invitation(_get_supabase(), user_id, collaborator_id)
    if error == "not_found":
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Re-send the email
    from registry.emails import send_invitation_email
    work = _get_supabase().table("works_registry").select("title").eq("id", collab["work_id"]).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"
    profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
    inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
    send_invitation_email(
        recipient_email=collab["email"],
        recipient_name=collab["name"],
        inviter_name=inviter_name,
        work_title=work_title,
        role=collab["role"],
        invite_token=str(collab.get("invite_token", "")),
    )
    return collab


# ============================================================
# TeamCard
# ============================================================

@router.get("/teamcard")
async def get_my_team_card(user_id: str = Query(...)):
    card = await service.get_team_card(_get_supabase(), user_id)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found — complete onboarding first")
    return card


@router.put("/teamcard")
async def update_team_card(body: TeamCardUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    card = await service.update_team_card(_get_supabase(), user_id, data)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found")
    return card


@router.get("/teamcard/{collaborator_user_id}")
async def get_collaborator_team_card(collaborator_user_id: str, user_id: str = Query(...)):
    """Get a collaborator's visible TeamCard fields. Requires the caller and target
    to share at least one collaboration link (either direction)."""
    db = _get_supabase()
    # Verify collaboration relationship exists
    shared = db.table("registry_collaborators").select("id").or_(
        f"and(invited_by.eq.{user_id},collaborator_user_id.eq.{collaborator_user_id}),"
        f"and(invited_by.eq.{collaborator_user_id},collaborator_user_id.eq.{user_id})"
    ).neq("status", "revoked").execute()
    if not shared.data:
        raise HTTPException(status_code=403, detail="No collaboration link with this user")
    card = await service.get_collaborator_team_card(db, collaborator_user_id)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found for this user")
    return card


# ============================================================
# Artist with TeamCard overlay (Option C merge)
# ============================================================

@router.get("/artists/{artist_id}/with-teamcard")
async def get_artist_with_teamcard(artist_id: str, user_id: str = Query(...)):
    data = await service.get_artist_with_teamcard(_get_supabase(), artist_id)
    if not data:
        raise HTTPException(status_code=404, detail="Artist not found")
    return data


@router.get("/artists/with-teamcards")
async def list_artists_with_teamcards(user_id: str = Query(...)):
    """Batch endpoint: returns all of a user's artists with TeamCard overlays applied.
    Uses a single query with LEFT JOIN to avoid N+1."""
    artists = await service.get_artists_with_teamcards(_get_supabase(), user_id)
    return {"artists": artists}


# ============================================================
# Notes
# ============================================================

@router.get("/notes")
async def list_notes(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    folder_id: Optional[str] = Query(None),
):
    notes = await service.get_notes(_get_supabase(), user_id, artist_id, project_id, folder_id)
    return {"notes": notes}


@router.get("/notes/{note_id}")
async def get_note(note_id: str, user_id: str = Query(...)):
    note = await service.get_note(_get_supabase(), user_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/notes")
async def create_note(body: NoteCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    note = await service.create_note(_get_supabase(), user_id, data)
    if not note:
        raise HTTPException(status_code=500, detail="Failed to create note")
    return note


@router.put("/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    note = await service.update_note(_get_supabase(), user_id, note_id, data)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str = Query(...)):
    await service.delete_note(_get_supabase(), user_id, note_id)
    return {"ok": True}


@router.get("/folders")
async def list_folders(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
):
    folders = await service.get_folders(_get_supabase(), user_id, artist_id, project_id)
    return {"folders": folders}


@router.post("/folders")
async def create_folder(body: FolderCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    folder = await service.create_folder(_get_supabase(), user_id, data)
    if not folder:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return folder


@router.put("/folders/{folder_id}")
async def update_folder(folder_id: str, body: FolderUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    folder = await service.update_folder(_get_supabase(), user_id, folder_id, data)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, user_id: str = Query(...)):
    await service.delete_folder(_get_supabase(), user_id, folder_id)
    return {"ok": True}


# ============================================================
# Project About
# ============================================================

@router.get("/projects/{project_id}/about")
async def get_project_about(project_id: str, user_id: str = Query(...)):
    """Get project about content. Requires auth — caller must be project owner
    or a collaborator on a work in this project."""
    db = _get_supabase()
    # Check access: owner or collaborator
    project = db.table("projects").select("artist_id").eq("id", project_id).single().execute()
    if not project.data:
        raise HTTPException(status_code=404, detail="Project not found")
    artist = db.table("artists").select("user_id, linked_user_id").eq("id", project.data["artist_id"]).single().execute()
    is_owner = artist.data and (artist.data["user_id"] == user_id or artist.data.get("linked_user_id") == user_id)
    is_collab = db.table("works_registry").select("id").eq("project_id", project_id).execute()
    collab_work_ids = [w["id"] for w in (is_collab.data or [])]
    has_collab_access = False
    if collab_work_ids:
        check = db.table("registry_collaborators").select("id").in_("work_id", collab_work_ids).eq("collaborator_user_id", user_id).neq("status", "revoked").execute()
        has_collab_access = bool(check.data)
    if not is_owner and not has_collab_access:
        raise HTTPException(status_code=403, detail="Not authorized to view this project")
    content = await service.get_project_about(db, project_id)
    return {"about_content": content}


@router.put("/projects/{project_id}/about")
async def update_project_about(project_id: str, body: ProjectAboutUpdate, user_id: str = Query(...)):
    result = await service.update_project_about(_get_supabase(), user_id, project_id, body.about_content)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


# ============================================================
# Notifications
# ============================================================

@router.get("/notifications")
async def list_notifications(user_id: str = Query(...), unread_only: bool = Query(False)):
    notifications = await service.get_notifications(_get_supabase(), user_id, unread_only)
    return {"notifications": notifications}


@router.post("/notifications/{notification_id}/read")
async def mark_read(notification_id: str, user_id: str = Query(...)):
    await service.mark_notification_read(_get_supabase(), user_id, notification_id)
    return {"ok": True}


@router.post("/notifications/read-all")
async def mark_all_read(user_id: str = Query(...)):
    await service.mark_all_notifications_read(_get_supabase(), user_id)
    return {"ok": True}
```

- [ ] **Step 2: Mount router in main.py**

In `src/backend/main.py`, after existing router imports (~line 40), add:

```python
from registry.router import router as registry_router
```

After existing `app.include_router` calls (~line 48), add:

```python
app.include_router(registry_router, prefix="/registry", tags=["Rights Registry"])
```

- [ ] **Step 3: Verify and commit**

Run: `cd src/backend && python -c "from registry.router import router; print('OK')"`
Expected: `OK`

```bash
git add src/backend/registry/router.py src/backend/main.py
git commit -m "feat: add registry router with collaboration, TeamCard, notes, and project about endpoints"
```

---

### Task 5: Email System — Invitation + Verification via Resend

**Goal:** Send invitation emails to collaborators and verification confirmation emails using the existing Resend integration.

**Files:**
- Create: `src/backend/registry/emails.py`

**Acceptance Criteria:**
- [ ] HTML invitation email with work title, inviter name, collaborator role, claim link
- [ ] Claim link: `{FRONTEND_URL}/tools/registry/invite/{invite_token}`
- [ ] Uses `RESEND_API_KEY` and `RESEND_FROM_EMAIL` env vars (existing pattern)
- [ ] Email ALWAYS sent — whether user exists on platform or not
- [ ] Invitation expires after 48h — claim returns 410 if expired

**Verify:** `cd src/backend && python -c "from registry.emails import send_invitation_email; print('OK')"`

**Steps:**

- [ ] **Step 1: Create emails module**

Create `src/backend/registry/emails.py`:

```python
"""Invitation and verification emails for the Rights Registry."""

import html
import os
import resend


def send_invitation_email(
    recipient_email: str,
    recipient_name: str,
    inviter_name: str,
    work_title: str,
    role: str,
    invite_token: str,
):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("Warning: RESEND_API_KEY not set — skipping invitation email")
        return None

    resend.api_key = api_key
    frontend_url = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
    claim_url = f"{frontend_url}/tools/registry/invite/{invite_token}"
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    # Escape all user-supplied values to prevent HTML injection
    safe_name = html.escape(recipient_name)
    safe_inviter = html.escape(inviter_name)
    safe_title = html.escape(work_title)
    safe_role = html.escape(role)

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii Rights Registry</h1>
      </div>

      <p style="font-size: 16px; color: #333;">Hi {safe_name},</p>

      <p style="font-size: 15px; color: #555;">
        <strong>{safe_inviter}</strong> has listed you as a <strong>{safe_role}</strong> on the work
        <strong>&ldquo;{safe_title}&rdquo;</strong> and is requesting you confirm your ownership stake.
      </p>

      <div style="text-align: center; margin: 32px 0;">
        <a href="{claim_url}"
           style="display: inline-block; background: #1a3a2a; color: white; padding: 14px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px;">
          Review &amp; Confirm Your Stake
        </a>
      </div>

      <p style="font-size: 13px; color: #888;">
        You'll be asked to sign in (or create an account) to view the full ownership details
        and confirm or dispute your stake. This invitation expires in 48 hours.
      </p>

      <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />

      <p style="font-size: 12px; color: #aaa; text-align: center;">
        Sent via Msanii Rights &amp; Ownership Registry
      </p>
    </div>
    """

    try:
        response = resend.Emails.send({
            "from": from_address,
            "to": [recipient_email],
            "subject": f"{safe_inviter} needs you to confirm your stake on \"{safe_title}\"",
            "html": html_body,
        })
        return response
    except Exception as e:
        print(f"Warning: Failed to send invitation email: {e}")
        return None
```

- [ ] **Step 2: Verify and commit**

Run: `cd src/backend && python -c "from registry.emails import send_invitation_email; print('OK')"`
Expected: `OK`

```bash
git add src/backend/registry/emails.py
git commit -m "feat: add invitation email with claim link via Resend"
```

---

### Task 6: Backend Proof of Ownership PDF

**Goal:** Generate PDF showing ownership, licensing, agreements, and approval status per stakeholder.

**Files:**
- Create: `src/backend/registry/pdf_generator.py`

**Acceptance Criteria:**
- [ ] PDF header: work title, ISRC/ISWC/UPC, type, release date
- [ ] Ownership tables show holder + percentage + approval status (Confirmed/Pending/Disputed)
- [ ] Licensing and agreement sections
- [ ] Footer: work status, generation timestamp, document hash

**Verify:** `cd src/backend && python -c "from registry.pdf_generator import generate_proof_of_ownership_pdf; print('OK')"`

**Steps:**

- [ ] **Step 1: Create PDF generator**

Create `src/backend/registry/pdf_generator.py`:

```python
"""Proof-of-ownership PDF with approval status per stakeholder."""

import io
import hashlib
from datetime import datetime, timezone

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

BRAND = colors.HexColor("#1a3a2a")
GREEN = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#d97706")


def generate_proof_of_ownership_pdf(work_data: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=24, spaceAfter=4, textColor=BRAND, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=20)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=14, textColor=BRAND, spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11)

    collaborators = work_data.get("collaborators", [])
    stake_approval = {}
    email_approval = {}
    for c in collaborators:
        if c.get("stake_id"):
            stake_approval[c["stake_id"]] = {"status": c["status"], "name": c["name"]}
        if c.get("email"):
            email_approval[c["email"].lower()] = {"status": c["status"], "name": c["name"]}

    # Header
    elements.append(Paragraph("PROOF OF OWNERSHIP", title_style))
    elements.append(Paragraph("Rights & Ownership Registry Certificate", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND, spaceAfter=20))

    # Status banner
    work_status = (work_data.get("status") or "draft").replace("_", " ").title()
    status_color = {"Registered": GREEN, "Disputed": RED, "Pending Approval": AMBER}.get(work_status, colors.HexColor("#666"))
    elements.append(Paragraph(f"<b>Registry Status: <font color='{status_color}'>{work_status}</font></b>", body_style))
    elements.append(Spacer(1, 8))

    # Work details
    elements.append(Paragraph("Work Details", section_style))
    details = [
        ["Title:", work_data.get("title", "—")],
        ["Type:", (work_data.get("work_type") or "single").replace("_", " ").title()],
        ["ISRC:", work_data.get("isrc") or "—"],
        ["ISWC:", work_data.get("iswc") or "—"],
        ["UPC:", work_data.get("upc") or "—"],
        ["Release Date:", str(work_data.get("release_date") or "—")],
    ]
    dt = Table(details, colWidths=[1.5 * inch, 5 * inch])
    dt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 12))

    # Ownership with approval
    stakes = work_data.get("stakes", [])

    def build_stakes_section(label, stake_list):
        elements.append(Paragraph(f"{label} Ownership", section_style))
        if not stake_list:
            elements.append(Paragraph(f"No {label.lower()} ownership recorded.", body_style))
            elements.append(Spacer(1, 8))
            return

        header = ["Holder", "Role", "%", "Publisher/Label", "Approval"]
        rows = [header]
        for s in stake_list:
            approval = "—"
            sid = s.get("id")
            hemail = (s.get("holder_email") or "").lower()
            if sid in stake_approval:
                approval = stake_approval[sid]["status"].title()
            elif hemail and hemail in email_approval:
                approval = email_approval[hemail]["status"].title()
            rows.append([
                s.get("holder_name", ""), s.get("holder_role", ""),
                f"{s.get('percentage', 0):.2f}%",
                s.get("publisher_or_label") or "—", approval,
            ])
        total = sum(s.get("percentage", 0) for s in stake_list)
        rows.append(["", "TOTAL", f"{total:.2f}%", "", ""])

        col_widths = [1.5 * inch, 1.0 * inch, 0.8 * inch, 1.5 * inch, 1.2 * inch]
        tbl = Table(rows, colWidths=col_widths)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("ALIGN", (4, 0), (4, -1), "CENTER"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
            ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#ccc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#999")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]
        for row_idx in range(1, len(rows) - 1):
            approval_text = rows[row_idx][4]
            if approval_text == "Confirmed":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), GREEN))
            elif approval_text == "Disputed":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), RED))
            elif approval_text == "Invited":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), AMBER))
        tbl.setStyle(TableStyle(style_cmds))
        elements.append(tbl)
        elements.append(Spacer(1, 12))

    master = [s for s in stakes if s.get("stake_type") == "master"]
    pub = [s for s in stakes if s.get("stake_type") == "publishing"]
    build_stakes_section("Master", master)
    build_stakes_section("Publishing", pub)

    # Licensing
    licenses = work_data.get("licenses", [])
    elements.append(Paragraph("Licensing Rights", section_style))
    if not licenses:
        elements.append(Paragraph("No licensing rights recorded.", body_style))
    else:
        rows = [["Type", "Licensee", "Territory", "Start", "End", "Status"]]
        for lic in licenses:
            rows.append([
                (lic.get("license_type") or "").replace("_", " ").title(),
                lic.get("licensee_name", ""), lic.get("territory", ""),
                str(lic.get("start_date", "—")), str(lic.get("end_date") or "Perpetual"),
                (lic.get("status") or "active").title(),
            ])
        tbl = Table(rows, colWidths=[1.0*inch, 1.4*inch, 1.0*inch, 0.9*inch, 0.9*inch, 0.8*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(tbl)
    elements.append(Spacer(1, 12))

    # Agreements
    agreements = work_data.get("agreements", [])
    elements.append(Paragraph("Agreement History", section_style))
    if not agreements:
        elements.append(Paragraph("No agreements recorded.", body_style))
    else:
        for agr in agreements:
            agr_type = (agr.get("agreement_type") or "").replace("_", " ").title()
            parties_list = agr.get("parties") or []
            party_names = ", ".join(p.get("name", "") for p in parties_list) if parties_list else "—"
            elements.append(Paragraph(f"<b>{agr.get('title', '')}</b> — {agr_type}", body_style))
            elements.append(Paragraph(f"Effective: {agr.get('effective_date', '—')} | Recorded: {agr.get('created_at', '—')}", small_style))
            elements.append(Paragraph(f"Parties: {party_names}", small_style))
            if agr.get("document_hash"):
                elements.append(Paragraph(f"Hash: {agr['document_hash']}", small_style))
            elements.append(Spacer(1, 6))

    # Footer
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc"), spaceAfter=8))
    content_str = f"{work_data.get('id', '')}|{work_data.get('title', '')}|{generated_at}"
    doc_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    elements.append(Paragraph(
        "This certificate reflects the ownership and rights information as recorded in the "
        "Msanii Rights & Ownership Registry. Stakeholder approval status indicates whether "
        "each party has confirmed their stake. A 'Registered' status means all parties have agreed.",
        ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11),
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        f"Generated: {generated_at} | Document ID: {doc_hash}",
        ParagraphStyle("Foot", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER),
    ))
    elements.append(Paragraph(
        "Msanii Rights & Ownership Registry",
        ParagraphStyle("Brand", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER),
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
```

- [ ] **Step 2: Verify and commit**

Run: `cd src/backend && python -c "from registry.pdf_generator import generate_proof_of_ownership_pdf; print('OK')"`
Expected: `OK`

```bash
git add src/backend/registry/pdf_generator.py
git commit -m "feat: add proof-of-ownership PDF with approval status per stakeholder"
```

---

### Task 7: Install BlockNote & Frontend Types, Hooks, Navigation

**Goal:** Install BlockNote editor packages, add TypeScript types for all new tables, create React Query hooks, and set up routing.

**Files:**
- Modify: `package.json` (install BlockNote)
- Modify: `src/integrations/supabase/types.ts`
- Create: `src/hooks/useRegistry.ts`
- Create: `src/hooks/useTeamCard.ts`
- Create: `src/hooks/useNotes.ts`
- Create: `src/hooks/useRegistryNotifications.ts`
- Modify: `src/App.tsx`
- Modify: `src/pages/Tools.tsx`

**Acceptance Criteria:**
- [ ] BlockNote packages installed: `@blocknote/core`, `@blocknote/react`, `@blocknote/shadcn`
- [ ] Types for works_registry, ownership_stakes, licensing_rights, registry_agreements, registry_collaborators, team_cards, notes, note_folders, registry_notifications
- [ ] useRegistry: all CRUD + collaboration hooks
- [ ] useTeamCard: get/update own card, get collaborator card
- [ ] useNotes: CRUD notes + folders, get project about
- [ ] useRegistryNotifications: list, mark read, unread count
- [ ] Routes: `/tools/registry`, `/tools/registry/:workId`, `/tools/registry/invite/:token`, `/projects/:projectId`
- [ ] Registry card on Tools page

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Install BlockNote**

```bash
npm install @blocknote/core@~0.17.0 @blocknote/react@~0.17.0 @blocknote/shadcn@~0.17.0
```

- [ ] **Step 2: Add TypeScript types**

In `src/integrations/supabase/types.ts`, add these table definitions inside `Tables: {}` (after the `projects` table, before the closing `}` of `Tables`):

```typescript
      works_registry: {
        Row: {
          id: string; user_id: string; artist_id: string; project_id: string
          title: string; work_type: string; isrc: string | null; iswc: string | null
          upc: string | null; release_date: string | null; status: string
          notes: string | null; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; artist_id: string; project_id: string
          title: string; work_type?: string; isrc?: string | null; iswc?: string | null
          upc?: string | null; release_date?: string | null; status?: string
          notes?: string | null; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; artist_id?: string; project_id?: string
          title?: string; work_type?: string; isrc?: string | null; iswc?: string | null
          upc?: string | null; release_date?: string | null; status?: string
          notes?: string | null; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      ownership_stakes: {
        Row: {
          id: string; work_id: string; user_id: string; stake_type: string
          holder_name: string; holder_role: string; percentage: number
          holder_email: string | null; holder_ipi: string | null
          publisher_or_label: string | null; notes: string | null
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; stake_type: string
          holder_name: string; holder_role: string; percentage: number
          holder_email?: string | null; holder_ipi?: string | null
          publisher_or_label?: string | null; notes?: string | null
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; stake_type?: string
          holder_name?: string; holder_role?: string; percentage?: number
          holder_email?: string | null; holder_ipi?: string | null
          publisher_or_label?: string | null; notes?: string | null
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      licensing_rights: {
        Row: {
          id: string; work_id: string; user_id: string; license_type: string
          licensee_name: string; licensee_email: string | null; territory: string
          start_date: string; end_date: string | null; terms: string | null
          status: string; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; license_type: string
          licensee_name: string; licensee_email?: string | null; territory?: string
          start_date: string; end_date?: string | null; terms?: string | null
          status?: string; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; license_type?: string
          licensee_name?: string; licensee_email?: string | null; territory?: string
          start_date?: string; end_date?: string | null; terms?: string | null
          status?: string; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      registry_agreements: {
        Row: {
          id: string; work_id: string; user_id: string; agreement_type: string
          title: string; description: string | null; effective_date: string
          parties: Json; file_id: string | null; document_hash: string | null
          created_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; agreement_type: string
          title: string; description?: string | null; effective_date: string
          parties?: Json; file_id?: string | null; document_hash?: string | null
          created_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; agreement_type?: string
          title?: string; description?: string | null; effective_date?: string
          parties?: Json; file_id?: string | null; document_hash?: string | null
          created_at?: string
        }
        Relationships: []
      }
      registry_collaborators: {
        Row: {
          id: string; work_id: string; stake_id: string | null; invited_by: string
          collaborator_user_id: string | null; email: string; name: string; role: string
          status: string; invite_token: string; dispute_reason: string | null
          expires_at: string; invited_at: string; responded_at: string | null
        }
        Insert: {
          id?: string; work_id: string; stake_id?: string | null; invited_by?: string
          collaborator_user_id?: string | null; email: string; name: string; role: string
          status?: string; invite_token?: string; dispute_reason?: string | null
          expires_at?: string; invited_at?: string; responded_at?: string | null
        }
        Update: {
          id?: string; work_id?: string; stake_id?: string | null; invited_by?: string
          collaborator_user_id?: string | null; email?: string; name?: string; role?: string
          status?: string; invite_token?: string; dispute_reason?: string | null
          expires_at?: string; invited_at?: string; responded_at?: string | null
        }
        Relationships: []
      }
      team_cards: {
        Row: {
          id: string; user_id: string; display_name: string; first_name: string
          last_name: string; email: string; avatar_url: string | null; bio: string | null
          phone: string | null; website: string | null; company: string | null
          industry: string | null; social_links: Json; dsp_links: Json; custom_links: Json
          visible_fields: Json; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id: string; display_name: string; first_name: string
          last_name: string; email: string; avatar_url?: string | null; bio?: string | null
          phone?: string | null; website?: string | null; company?: string | null
          industry?: string | null; social_links?: Json; dsp_links?: Json; custom_links?: Json
          visible_fields?: Json; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; display_name?: string; first_name?: string
          last_name?: string; email?: string; avatar_url?: string | null; bio?: string | null
          phone?: string | null; website?: string | null; company?: string | null
          industry?: string | null; social_links?: Json; dsp_links?: Json; custom_links?: Json
          visible_fields?: Json; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      notes: {
        Row: {
          id: string; user_id: string; folder_id: string | null
          artist_id: string | null; project_id: string | null
          title: string; content: Json; pinned: boolean
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; folder_id?: string | null
          artist_id?: string | null; project_id?: string | null
          title?: string; content?: Json; pinned?: boolean
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; folder_id?: string | null
          artist_id?: string | null; project_id?: string | null
          title?: string; content?: Json; pinned?: boolean
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      note_folders: {
        Row: {
          id: string; user_id: string; artist_id: string | null; project_id: string | null
          name: string; parent_folder_id: string | null; sort_order: number
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; artist_id?: string | null; project_id?: string | null
          name: string; parent_folder_id?: string | null; sort_order?: number
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; artist_id?: string | null; project_id?: string | null
          name?: string; parent_folder_id?: string | null; sort_order?: number
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      registry_notifications: {
        Row: {
          id: string; user_id: string; work_id: string | null; type: string
          title: string; message: string; read: boolean; metadata: Json; created_at: string
        }
        Insert: {
          id?: string; user_id?: string; work_id?: string | null; type: string
          title: string; message: string; read?: boolean; metadata?: Json; created_at?: string
        }
        Update: {
          id?: string; user_id?: string; work_id?: string | null; type?: string
          title?: string; message?: string; read?: boolean; metadata?: Json; created_at?: string
        }
        Relationships: []
      }
```

Also update the `artists` table types. The existing type is missing `user_id` (added in migration `20260112000000`) and the new verification columns. Add these to each section:

```typescript
      // Add to artists Row:
      user_id: string
      custom_social_links: Json | null
      custom_dsp_links: Json | null
      linked_user_id: string | null
      verified: boolean
      verified_at: string | null
      // Add to artists Insert:
      user_id?: string
      custom_social_links?: Json | null
      custom_dsp_links?: Json | null
      linked_user_id?: string | null
      verified?: boolean
      verified_at?: string | null
      // Add to artists Update:
      user_id?: string
      custom_social_links?: Json | null
      custom_dsp_links?: Json | null
      linked_user_id?: string | null
      verified?: boolean
      verified_at?: string | null
```

And update the `projects` table types:

```typescript
      // Add to projects Row:
      about_content: Json
      // Add to projects Insert:
      about_content?: Json
      // Add to projects Update:
      about_content?: Json
```

- [ ] **Step 3: Create shared apiFetch utility**

Create `src/lib/apiFetch.ts`:

```typescript
export const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export async function apiFetch<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 4: Create useRegistry hook**

Create `src/hooks/useRegistry.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// --- Types ---

export interface Work {
  id: string; user_id: string; artist_id: string; project_id: string;
  title: string; work_type: string; isrc: string | null; iswc: string | null;
  upc: string | null; release_date: string | null; status: string;
  notes: string | null; created_at: string; updated_at: string;
}

export interface OwnershipStake {
  id: string; work_id: string; user_id: string; stake_type: string;
  holder_name: string; holder_role: string; percentage: number;
  holder_email: string | null; holder_ipi: string | null;
  publisher_or_label: string | null; notes: string | null;
  created_at: string; updated_at: string;
}

export interface LicensingRight {
  id: string; work_id: string; user_id: string; license_type: string;
  licensee_name: string; licensee_email: string | null; territory: string;
  start_date: string; end_date: string | null; terms: string | null;
  status: string; created_at: string; updated_at: string;
}

export interface Agreement {
  id: string; work_id: string; user_id: string; agreement_type: string;
  title: string; description: string | null; effective_date: string;
  parties: Array<{ name: string; role: string; email?: string }>;
  file_id: string | null; document_hash: string | null; created_at: string;
}

export interface Collaborator {
  id: string; work_id: string; stake_id: string | null; invited_by: string;
  collaborator_user_id: string | null; email: string; name: string; role: string;
  status: string; invite_token: string; dispute_reason: string | null;
  expires_at: string; invited_at: string; responded_at: string | null;
}

export interface WorkFull extends Work {
  stakes: OwnershipStake[]; licenses: LicensingRight[];
  agreements: Agreement[]; collaborators: Collaborator[];
}

// --- Works ---

export function useWorks(artistId?: string) {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-works", user?.id, artistId],
    queryFn: async () => {
      if (!user?.id) return [];
      let url = `${API_URL}/registry/works?user_id=${user.id}`;
      if (artistId) url += `&artist_id=${artistId}`;
      const data = await apiFetch<{ works: Work[] }>(url);
      return data.works;
    },
    enabled: !!user?.id,
  });
}

export function useMyCollaborations() {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-my-collaborations", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<{ works: Work[] }>(
        `${API_URL}/registry/works/my-collaborations?user_id=${user.id}`
      );
      return data.works;
    },
    enabled: !!user?.id,
  });
}

export function useWorksByProject(projectId: string | undefined) {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-works-by-project", user?.id, projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ works: Work[] }>(
        `${API_URL}/registry/works/by-project/${projectId}?user_id=${user.id}`
      );
      return data.works;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useWorkFull(workId: string | undefined) {
  const { user } = useAuth();
  return useQuery<WorkFull | null>({
    queryKey: ["registry-work-full", user?.id, workId],
    queryFn: async () => {
      if (!user?.id || !workId) return null;
      return apiFetch<WorkFull>(
        `${API_URL}/registry/works/${workId}/full?user_id=${user.id}`
      );
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useCreateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      artist_id: string; project_id: string; title: string; work_type?: string;
      isrc?: string; iswc?: string; upc?: string; release_date?: string; notes?: string;
    }) =>
      apiFetch(`${API_URL}/registry/works?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-works"] }); toast.success("Work registered"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ workId, ...body }: { workId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/works/${workId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Work updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-works"] }); toast.success("Work deleted"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Stakes ---

export function useCreateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; stake_type: string; holder_name: string; holder_role: string;
      percentage: number; holder_email?: string; holder_ipi?: string;
      publisher_or_label?: string; notes?: string;
    }) =>
      apiFetch(`${API_URL}/registry/stakes?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake added"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ stakeId, ...body }: { stakeId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake updated"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (stakeId: string) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake removed"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Licenses ---

export function useCreateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; license_type: string; licensee_name: string;
      licensee_email?: string; territory?: string; start_date: string;
      end_date?: string; terms?: string;
    }) =>
      apiFetch(`${API_URL}/registry/licenses?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License added"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ licenseId, ...body }: { licenseId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License updated"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (licenseId: string) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License removed"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Agreements ---

export function useCreateAgreement() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; agreement_type: string; title: string; description?: string;
      effective_date: string; parties: Array<{ name: string; role: string; email?: string }>;
      file_id?: string; document_hash?: string;
    }) =>
      apiFetch(`${API_URL}/registry/agreements?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Agreement recorded"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Collaboration ---

export function useInviteCollaborator() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; email: string; name: string; role: string; stake_id?: string;
    }) =>
      apiFetch<Collaborator>(`${API_URL}/registry/collaborators/invite?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Invitation sent"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useClaimInvitation() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (inviteToken: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/claim?invite_token=${inviteToken}&user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useConfirmStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/confirm?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      toast.success("Stake confirmed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDisputeStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, reason }: { collaboratorId: string; reason: string }) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/dispute?user_id=${user!.id}`,
        {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason }),
        }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      toast.success("Dispute submitted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useSubmitForApproval() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}/submit-for-approval?user_id=${user!.id}`, {
        method: "POST",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Submitted for approval — invitations sent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRevokeCollaborator() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/revoke?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Collaborator revoked");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useResendInvitation() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/resend?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Invitation resent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Export ---

export function useExportProof() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (workId: string) => {
      const res = await fetch(`${API_URL}/registry/works/${workId}/export?user_id=${user!.id}`);
      if (!res.ok) throw new Error("Failed to generate proof of ownership");
      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition") || "";
      const match = disposition.match(/filename="?(.+?)"?$/);
      const filename = match ? match[1] : "Proof_of_Ownership.pdf";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
    },
    onSuccess: () => toast.success("Proof of ownership downloaded"),
    onError: (e: Error) => toast.error(e.message),
  });
}
```

- [ ] **Step 4: Create useTeamCard hook**

Create `src/hooks/useTeamCard.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface TeamCard {
  id: string;
  user_id: string;
  display_name: string;
  first_name: string;
  last_name: string;
  email: string;
  avatar_url: string | null;
  bio: string | null;
  phone: string | null;
  website: string | null;
  company: string | null;
  industry: string | null;
  social_links: Record<string, string>;
  dsp_links: Record<string, string>;
  custom_links: Array<{ label: string; url: string }>;
  visible_fields: string[];
  created_at: string;
  updated_at: string;
}

export function useMyTeamCard() {
  const { user } = useAuth();
  return useQuery<TeamCard | null>({
    queryKey: ["team-card", user?.id],
    queryFn: async () => {
      if (!user?.id) return null;
      return apiFetch<TeamCard>(`${API_URL}/registry/teamcard?user_id=${user.id}`);
    },
    enabled: !!user?.id,
  });
}

export function useUpdateTeamCard() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Partial<Omit<TeamCard, "id" | "user_id" | "email" | "created_at" | "updated_at">>) =>
      apiFetch<TeamCard>(`${API_URL}/registry/teamcard?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["team-card"] });
      toast.success("TeamCard updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useCollaboratorTeamCard(collaboratorUserId: string | undefined) {
  const { user } = useAuth();
  return useQuery<TeamCard | null>({
    queryKey: ["collaborator-team-card", collaboratorUserId],
    queryFn: async () => {
      if (!user?.id || !collaboratorUserId) return null;
      return apiFetch<TeamCard>(
        `${API_URL}/registry/teamcard/${collaboratorUserId}?user_id=${user.id}`
      );
    },
    enabled: !!user?.id && !!collaboratorUserId,
  });
}
```

- [ ] **Step 5: Create useNotes hook**

Create `src/hooks/useNotes.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface Note {
  id: string;
  user_id: string;
  folder_id: string | null;
  artist_id: string | null;
  project_id: string | null;
  title: string;
  content: unknown[];
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface NoteFolder {
  id: string;
  user_id: string;
  artist_id: string | null;
  project_id: string | null;
  name: string;
  parent_folder_id: string | null;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export function useNotes(scope: { artistId?: string; projectId?: string; folderId?: string }) {
  const { user } = useAuth();
  return useQuery<Note[]>({
    queryKey: ["notes", user?.id, scope.artistId, scope.projectId, scope.folderId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({ user_id: user.id });
      if (scope.artistId) params.set("artist_id", scope.artistId);
      if (scope.projectId) params.set("project_id", scope.projectId);
      if (scope.folderId) params.set("folder_id", scope.folderId);
      const data = await apiFetch<{ notes: Note[] }>(`${API_URL}/registry/notes?${params}`);
      return data.notes;
    },
    enabled: !!user?.id,
  });
}

export function useNote(noteId: string | undefined) {
  const { user } = useAuth();
  return useQuery<Note | null>({
    queryKey: ["note", user?.id, noteId],
    queryFn: async () => {
      if (!user?.id || !noteId) return null;
      return apiFetch<Note>(`${API_URL}/registry/notes/${noteId}?user_id=${user.id}`);
    },
    enabled: !!user?.id && !!noteId,
  });
}

export function useCreateNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      title?: string; content?: unknown[]; artist_id?: string;
      project_id?: string; folder_id?: string; pinned?: boolean;
    }) =>
      apiFetch<Note>(`${API_URL}/registry/notes?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Note created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ noteId, ...body }: {
      noteId: string; title?: string; content?: unknown[];
      folder_id?: string | null; pinned?: boolean;
    }) =>
      apiFetch<Note>(`${API_URL}/registry/notes/${noteId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      qc.invalidateQueries({ queryKey: ["note"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (noteId: string) =>
      apiFetch(`${API_URL}/registry/notes/${noteId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Note deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useFolders(scope: { artistId?: string; projectId?: string }) {
  const { user } = useAuth();
  return useQuery<NoteFolder[]>({
    queryKey: ["note-folders", user?.id, scope.artistId, scope.projectId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({ user_id: user.id });
      if (scope.artistId) params.set("artist_id", scope.artistId);
      if (scope.projectId) params.set("project_id", scope.projectId);
      const data = await apiFetch<{ folders: NoteFolder[] }>(`${API_URL}/registry/folders?${params}`);
      return data.folders;
    },
    enabled: !!user?.id,
  });
}

export function useCreateFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      name: string; artist_id?: string; project_id?: string;
      parent_folder_id?: string; sort_order?: number;
    }) =>
      apiFetch<NoteFolder>(`${API_URL}/registry/folders?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["note-folders"] });
      toast.success("Folder created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ folderId, ...body }: {
      folderId: string; name?: string; parent_folder_id?: string | null; sort_order?: number;
    }) =>
      apiFetch<NoteFolder>(`${API_URL}/registry/folders/${folderId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["note-folders"] }),
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (folderId: string) =>
      apiFetch(`${API_URL}/registry/folders/${folderId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["note-folders"] });
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Folder deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useProjectAbout(projectId: string | undefined) {
  const { user } = useAuth();
  return useQuery<unknown[]>({
    queryKey: ["project-about", user?.id, projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ about_content: unknown[] }>(
        `${API_URL}/registry/projects/${projectId}/about?user_id=${user.id}`
      );
      return data.about_content;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useUpdateProjectAbout() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ projectId, about_content }: { projectId: string; about_content: unknown[] }) =>
      apiFetch(`${API_URL}/registry/projects/${projectId}/about?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ about_content }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["project-about", vars.projectId] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
```

- [ ] **Step 6: Create useRegistryNotifications hook**

Create `src/hooks/useRegistryNotifications.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL } from "@/lib/apiFetch";

export interface RegistryNotification {
  id: string;
  user_id: string;
  work_id: string | null;
  type: string;
  title: string;
  message: string;
  read: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
}

export function useRegistryNotifications(unreadOnly = false) {
  const { user } = useAuth();
  return useQuery<RegistryNotification[]>({
    queryKey: ["registry-notifications", user?.id, unreadOnly],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(
        `${API_URL}/registry/notifications?user_id=${user.id}&unread_only=${unreadOnly}`
      );
      if (!res.ok) return [];
      const data = await res.json();
      return data.notifications || [];
    },
    enabled: !!user?.id,
    refetchInterval: 30000,
  });
}

export function useUnreadCount() {
  const { data } = useRegistryNotifications(true);
  return data?.length || 0;
}

export function useMarkNotificationRead() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (notificationId: string) => {
      await fetch(
        `${API_URL}/registry/notifications/${notificationId}/read?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["registry-notifications"] }),
  });
}

export function useMarkAllRead() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      await fetch(
        `${API_URL}/registry/notifications/read-all?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["registry-notifications"] }),
  });
}
```

- [ ] **Step 7: Add routes to App.tsx and card to Tools.tsx**

In `src/App.tsx`, add imports:

```typescript
import Registry from "./pages/Registry";
import WorkDetail from "./pages/WorkDetail";
import InviteClaim from "./pages/InviteClaim";
import ProjectDetail from "./pages/ProjectDetail";
```

Add routes inside `<Routes>`, before the `NotFound` catch-all. **Important:** `/tools/registry/invite/:token` must appear BEFORE `/tools/registry/:workId`:

```tsx
            <Route path="/tools/registry" element={<ProtectedRoute><Registry /></ProtectedRoute>} />
            <Route path="/tools/registry/invite/:token" element={<ProtectedRoute><InviteClaim /></ProtectedRoute>} />
            <Route path="/tools/registry/:workId" element={<ProtectedRoute><WorkDetail /></ProtectedRoute>} />
            <Route path="/projects/:projectId" element={<ProtectedRoute><ProjectDetail /></ProtectedRoute>} />
```

In `src/pages/Tools.tsx`, add `Shield` to lucide-react import and add the Registry card after the Split Sheet card:

```tsx
          <Card className="hover:border-primary/50 transition-colors cursor-pointer group" onClick={() => handleNavigate("/tools/registry", "Registry")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <CardTitle className="flex items-center gap-2">
                Rights Registry
                <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity transform translate-x-[-10px] group-hover:translate-x-0" />
              </CardTitle>
              <CardDescription>
                Track master ownership, publishing splits, licensing rights, and generate proof-of-ownership documents.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-start p-0 hover:bg-transparent hover:text-primary">
                Launch Tool
              </Button>
            </CardContent>
          </Card>
```

- [ ] **Step 8: Verify and commit**

Run: `npm run build`
Expected: Build succeeds (page stubs will be created in following tasks)

```bash
git add package.json package-lock.json src/integrations/supabase/types.ts src/hooks/useRegistry.ts src/hooks/useTeamCard.ts src/hooks/useNotes.ts src/hooks/useRegistryNotifications.ts src/App.tsx src/pages/Tools.tsx
git commit -m "feat: install BlockNote, add all hooks and types for registry, TeamCard, notes, and notifications"
```

---

### Task 8: Rich Notes Editor Component (BlockNote)

**Goal:** Reusable Notion-like notes editor with folder sidebar, built on BlockNote with shadcn theming. Used on both artist profiles and project pages.

**Files:**
- Create: `src/components/notes/NotesEditor.tsx`
- Create: `src/components/notes/NotesSidebar.tsx`
- Create: `src/components/notes/NotesView.tsx`

**Acceptance Criteria:**
- [ ] `NotesEditor` wraps BlockNote with shadcn theme, supports headings, bullets, numbered lists, code, images, dividers
- [ ] `NotesSidebar` shows folder tree with nested folders, note list, create/rename/delete folders
- [ ] `NotesView` combines sidebar + editor in a split layout, reusable via `scope` prop (artistId or projectId)
- [ ] Auto-saves content on change (debounced 1s)
- [ ] Pin notes to keep them at top
- [ ] Empty state when no notes exist

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create NotesEditor.tsx**

Create `src/components/notes/NotesEditor.tsx`:

```tsx
import { Component, type ReactNode, useEffect, useRef } from "react";
import { useCreateBlockNote } from "@blocknote/react";
import { BlockNoteView } from "@blocknote/shadcn";
import "@blocknote/shadcn/style.css";

// Error boundary — BlockNote errors won't crash the whole page
class EditorErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 border rounded-lg bg-destructive/10 text-destructive text-sm">
          Editor failed to load. Try refreshing the page.
        </div>
      );
    }
    return this.props.children;
  }
}

interface Props {
  initialContent?: unknown[];
  onChange?: (content: unknown[]) => void;
  editable?: boolean;
}

function NotesEditorInner({ initialContent, onChange, editable = true }: Props) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const editor = useCreateBlockNote({
    initialContent: initialContent && initialContent.length > 0
      ? (initialContent as Parameters<typeof useCreateBlockNote>[0]["initialContent"])
      : undefined,
  });

  useEffect(() => {
    if (!onChange) return;
    const handler = () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onChange(editor.document as unknown as unknown[]);
      }, 1000);
    };
    editor.onEditorContentChange(handler);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [editor, onChange]);

  // Respect system/app dark mode preference
  const prefersDark = typeof window !== "undefined" &&
    document.documentElement.classList.contains("dark");

  return (
    <div className="min-h-[300px] border rounded-lg overflow-hidden bg-background">
      <BlockNoteView editor={editor} editable={editable} theme={prefersDark ? "dark" : "light"} />
    </div>
  );
}

export default function NotesEditor(props: Props) {
  return (
    <EditorErrorBoundary>
      <NotesEditorInner {...props} />
    </EditorErrorBoundary>
  );
}
```

- [ ] **Step 2: Create NotesSidebar.tsx**

Create `src/components/notes/NotesSidebar.tsx`:

```tsx
import { useState } from "react";
import { type Note, type NoteFolder, useCreateNote, useCreateFolder, useDeleteFolder, useDeleteNote } from "@/hooks/useNotes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { FolderPlus, FilePlus, Folder, FileText, Trash2, Pin, ChevronRight, ChevronDown } from "lucide-react";

interface Props {
  folders: NoteFolder[];
  notes: Note[];
  selectedNoteId: string | null;
  onSelectNote: (noteId: string) => void;
  scope: { artistId?: string; projectId?: string };
}

export default function NotesSidebar({ folders, notes, selectedNoteId, onSelectNote, scope }: Props) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [newFolderName, setNewFolderName] = useState("");
  const [showNewFolder, setShowNewFolder] = useState(false);

  const createNote = useCreateNote();
  const createFolder = useCreateFolder();
  const deleteFolder = useDeleteFolder();
  const deleteNote = useDeleteNote();

  const toggleFolder = (folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;
    await createFolder.mutateAsync({
      name: newFolderName.trim(),
      artist_id: scope.artistId,
      project_id: scope.projectId,
    });
    setNewFolderName("");
    setShowNewFolder(false);
  };

  const handleCreateNote = async (folderId?: string) => {
    const note = await createNote.mutateAsync({
      artist_id: scope.artistId,
      project_id: scope.projectId,
      folder_id: folderId,
    });
    if (note?.id) onSelectNote(note.id);
  };

  const rootFolders = folders.filter((f) => !f.parent_folder_id);
  const unfolderedNotes = notes.filter((n) => !n.folder_id);

  const renderFolder = (folder: NoteFolder, depth: number = 0) => {
    const isExpanded = expandedFolders.has(folder.id);
    const children = folders.filter((f) => f.parent_folder_id === folder.id);
    const folderNotes = notes.filter((n) => n.folder_id === folder.id);

    return (
      <div key={folder.id}>
        <div
          className="flex items-center gap-1 py-1 px-2 rounded hover:bg-muted/50 cursor-pointer group"
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => toggleFolder(folder.id)}
        >
          {isExpanded ? <ChevronDown className="w-3 h-3 shrink-0" /> : <ChevronRight className="w-3 h-3 shrink-0" />}
          <Folder className="w-4 h-4 text-muted-foreground shrink-0" />
          <span className="text-sm truncate flex-1">{folder.name}</span>
          <div className="hidden group-hover:flex items-center gap-0.5">
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={(e) => { e.stopPropagation(); handleCreateNote(folder.id); }}>
              <FilePlus className="w-3 h-3" />
            </Button>
            <Button size="icon" variant="ghost" className="h-5 w-5 text-destructive" onClick={(e) => { e.stopPropagation(); deleteFolder.mutate(folder.id); }}>
              <Trash2 className="w-3 h-3" />
            </Button>
          </div>
        </div>
        {isExpanded && (
          <>
            {children.map((c) => renderFolder(c, depth + 1))}
            {folderNotes.map((n) => renderNote(n, depth + 1))}
          </>
        )}
      </div>
    );
  };

  const renderNote = (note: Note, depth: number = 0) => (
    <div
      key={note.id}
      className={cn(
        "flex items-center gap-1 py-1 px-2 rounded cursor-pointer group",
        selectedNoteId === note.id ? "bg-primary/10 text-primary" : "hover:bg-muted/50"
      )}
      style={{ paddingLeft: `${depth * 16 + 8}px` }}
      onClick={() => onSelectNote(note.id)}
    >
      <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
      {note.pinned && <Pin className="w-3 h-3 text-amber-500 shrink-0" />}
      <span className="text-sm truncate flex-1">{note.title}</span>
      <Button size="icon" variant="ghost" className="h-5 w-5 hidden group-hover:flex text-destructive shrink-0"
        onClick={(e) => { e.stopPropagation(); deleteNote.mutate(note.id); }}>
        <Trash2 className="w-3 h-3" />
      </Button>
    </div>
  );

  return (
    <div className="w-64 border-r bg-muted/30 flex flex-col h-full">
      <div className="p-3 border-b flex items-center justify-between">
        <span className="text-sm font-medium">Notes</span>
        <div className="flex items-center gap-1">
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => setShowNewFolder(true)}>
            <FolderPlus className="w-4 h-4" />
          </Button>
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => handleCreateNote()}>
            <FilePlus className="w-4 h-4" />
          </Button>
        </div>
      </div>
      {showNewFolder && (
        <div className="p-2 border-b flex gap-1">
          <Input value={newFolderName} onChange={(e) => setNewFolderName(e.target.value)}
            placeholder="Folder name" className="h-7 text-xs"
            onKeyDown={(e) => { if (e.key === "Enter") handleCreateFolder(); if (e.key === "Escape") setShowNewFolder(false); }}
            autoFocus />
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-1">
        {rootFolders.map((f) => renderFolder(f))}
        {unfolderedNotes.map((n) => renderNote(n))}
        {folders.length === 0 && notes.length === 0 && (
          <div className="text-center py-8 text-xs text-muted-foreground">
            <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No notes yet</p>
            <p>Click + to create one</p>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create NotesView.tsx**

Create `src/components/notes/NotesView.tsx`:

```tsx
import { useState, useRef, useCallback } from "react";
import { useNotes, useFolders, useNote, useUpdateNote } from "@/hooks/useNotes";
import NotesEditor from "./NotesEditor";
import NotesSidebar from "./NotesSidebar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Pin, PinOff } from "lucide-react";

interface Props {
  scope: { artistId?: string; projectId?: string };
  className?: string;
}

export default function NotesView({ scope, className }: Props) {
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [localTitle, setLocalTitle] = useState("");
  const titleDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { data: notes = [] } = useNotes(scope);
  const { data: folders = [] } = useFolders(scope);
  const { data: selectedNote } = useNote(selectedNoteId ?? undefined);
  const updateNote = useUpdateNote();

  // Sync local title from server when note selection changes
  const prevNoteIdRef = useRef<string | null>(null);
  if (selectedNote && selectedNoteId !== prevNoteIdRef.current) {
    prevNoteIdRef.current = selectedNoteId;
    setLocalTitle(selectedNote.title);
  }

  const handleContentChange = (content: unknown[]) => {
    if (!selectedNoteId) return;
    updateNote.mutate({ noteId: selectedNoteId, content });
  };

  const handleTitleChange = useCallback((title: string) => {
    setLocalTitle(title); // Update local state immediately (optimistic)
    if (!selectedNoteId) return;
    // Debounce the API call
    if (titleDebounceRef.current) clearTimeout(titleDebounceRef.current);
    titleDebounceRef.current = setTimeout(() => {
      if (title.trim()) {
        updateNote.mutate({ noteId: selectedNoteId, title: title.trim() });
      }
    }, 800);
  }, [selectedNoteId, updateNote]);

  const togglePin = () => {
    if (!selectedNoteId || !selectedNote) return;
    updateNote.mutate({ noteId: selectedNoteId, pinned: !selectedNote.pinned });
  };

  return (
    <div className={`flex border rounded-lg overflow-hidden bg-background ${className || ""}`} style={{ height: "600px" }}>
      <NotesSidebar
        folders={folders}
        notes={notes}
        selectedNoteId={selectedNoteId}
        onSelectNote={setSelectedNoteId}
        scope={scope}
      />
      <div className="flex-1 flex flex-col">
        {selectedNote ? (
          <>
            <div className="p-3 border-b flex items-center gap-2">
              <Input
                value={localTitle}
                onChange={(e) => handleTitleChange(e.target.value)}
                className="border-0 text-lg font-semibold p-0 h-auto focus-visible:ring-0"
                placeholder="Untitled"
              />
              <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={togglePin}>
                {selectedNote.pinned ? <PinOff className="w-4 h-4 text-amber-500" /> : <Pin className="w-4 h-4" />}
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <NotesEditor
                key={selectedNote.id}
                initialContent={selectedNote.content as unknown[]}
                onChange={handleContentChange}
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <p className="text-sm">Select a note or create a new one</p>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/components/notes/
git commit -m "feat: add Notion-like notes editor with BlockNote, folder sidebar, and auto-save"
```

---

### Task 9: TeamCard Settings UI

**Goal:** Add TeamCard configuration section to the user Profile page — field visibility toggles and a preview of what collaborators see.

**Files:**
- Create: `src/components/profile/TeamCardSettings.tsx`
- Modify: `src/pages/Profile.tsx`

**Acceptance Criteria:**
- [ ] "TeamCard" section in Profile page below Account Information
- [ ] Shows all configurable fields with toggle switches for visibility
- [ ] Email always visible (locked), display_name/first_name/last_name always visible (locked)
- [ ] Optional fields: avatar, bio, phone, website, company, industry, social_links, dsp_links, custom_links
- [ ] "Preview" tab shows what a collaborator would see
- [ ] Save updates TeamCard via API

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create TeamCardSettings.tsx**

Create `src/components/profile/TeamCardSettings.tsx`:

```tsx
import { useState } from "react";
import { useMyTeamCard, useUpdateTeamCard, type TeamCard } from "@/hooks/useTeamCard";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Eye, Settings, Lock } from "lucide-react";

const ALL_FIELDS = [
  { key: "display_name", label: "Display Name", locked: true },
  { key: "first_name", label: "First Name", locked: true },
  { key: "last_name", label: "Last Name", locked: true },
  { key: "email", label: "Email", locked: true },
  { key: "avatar_url", label: "Profile Photo", locked: false },
  { key: "bio", label: "Bio", locked: false },
  { key: "phone", label: "Phone", locked: false },
  { key: "website", label: "Website", locked: false },
  { key: "company", label: "Company", locked: false },
  { key: "industry", label: "Industry", locked: false },
  { key: "social_links", label: "Social Links", locked: false },
  { key: "dsp_links", label: "Streaming Platforms", locked: false },
  { key: "custom_links", label: "Custom Links", locked: false },
];

const LOCKED_FIELDS = ALL_FIELDS.filter((f) => f.locked).map((f) => f.key);

export default function TeamCardSettings() {
  const { data: card, isLoading } = useMyTeamCard();
  const updateCard = useUpdateTeamCard();
  const [editMode, setEditMode] = useState(false);
  const [displayName, setDisplayName] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [bio, setBio] = useState("");
  const [phone, setPhone] = useState("");
  const [website, setWebsite] = useState("");
  const [company, setCompany] = useState("");
  const [visibleFields, setVisibleFields] = useState<string[]>([]);

  const startEdit = () => {
    if (!card) return;
    setDisplayName(card.display_name);
    setFirstName(card.first_name);
    setLastName(card.last_name);
    setBio(card.bio || "");
    setPhone(card.phone || "");
    setWebsite(card.website || "");
    setCompany(card.company || "");
    setVisibleFields(card.visible_fields || LOCKED_FIELDS);
    setEditMode(true);
  };

  const handleSave = async () => {
    await updateCard.mutateAsync({
      display_name: displayName,
      first_name: firstName,
      last_name: lastName,
      bio: bio || undefined,
      phone: phone || undefined,
      website: website || undefined,
      company: company || undefined,
      visible_fields: [...new Set([...LOCKED_FIELDS, ...visibleFields])],
    });
    setEditMode(false);
  };

  const toggleVisibility = (key: string) => {
    setVisibleFields((prev) =>
      prev.includes(key) ? prev.filter((f) => f !== key) : [...prev, key]
    );
  };

  if (isLoading) {
    return (
      <Card><CardContent className="py-8 text-center"><Loader2 className="w-6 h-6 animate-spin mx-auto" /></CardContent></Card>
    );
  }

  if (!card) {
    return (
      <Card><CardContent className="py-8 text-center text-muted-foreground">
        TeamCard will be created after onboarding.
      </CardContent></Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">TeamCard</CardTitle>
            <CardDescription>
              Your collaboration identity — choose what collaborators see about you
            </CardDescription>
          </div>
          {!editMode && (
            <Button variant="outline" size="sm" onClick={startEdit}>
              <Settings className="w-4 h-4 mr-1" /> Configure
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue={editMode ? "edit" : "preview"}>
          <TabsList className="mb-4">
            <TabsTrigger value="preview"><Eye className="w-4 h-4 mr-1" /> Preview</TabsTrigger>
            {editMode && <TabsTrigger value="edit"><Settings className="w-4 h-4 mr-1" /> Edit</TabsTrigger>}
            {editMode && <TabsTrigger value="visibility">Visibility</TabsTrigger>}
          </TabsList>

          <TabsContent value="preview">
            <div className="border rounded-lg p-4 bg-muted/30">
              <div className="flex items-center gap-3 mb-3">
                {card.avatar_url && (card.visible_fields || []).includes("avatar_url") && (
                  <img src={card.avatar_url} alt="" className="w-12 h-12 rounded-full object-cover" />
                )}
                <div>
                  <p className="font-semibold">{card.display_name}</p>
                  <p className="text-sm text-muted-foreground">{card.email}</p>
                </div>
                <Badge className="bg-green-100 text-green-800 ml-auto">Verified</Badge>
              </div>
              {card.bio && (card.visible_fields || []).includes("bio") && (
                <p className="text-sm text-muted-foreground mb-2">{card.bio}</p>
              )}
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {card.company && (card.visible_fields || []).includes("company") && <span>{card.company}</span>}
                {card.industry && (card.visible_fields || []).includes("industry") && <span>· {card.industry}</span>}
                {card.website && (card.visible_fields || []).includes("website") && <span>· {card.website}</span>}
              </div>
            </div>
          </TabsContent>

          {editMode && (
            <TabsContent value="edit">
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium">Display Name *</label>
                  <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">First Name *</label>
                    <Input value={firstName} onChange={(e) => setFirstName(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Last Name *</label>
                    <Input value={lastName} onChange={(e) => setLastName(e.target.value)} />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Email</label>
                  <Input value={card.email} disabled className="bg-muted" />
                  <p className="text-xs text-muted-foreground mt-1">Email cannot be changed</p>
                </div>
                <div>
                  <label className="text-sm font-medium">Bio</label>
                  <Input value={bio} onChange={(e) => setBio(e.target.value)} placeholder="Short bio" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">Phone</label>
                    <Input value={phone} onChange={(e) => setPhone(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Website</label>
                    <Input value={website} onChange={(e) => setWebsite(e.target.value)} />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Company</label>
                  <Input value={company} onChange={(e) => setCompany(e.target.value)} />
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleSave} disabled={updateCard.isPending}>
                    {updateCard.isPending && <Loader2 className="w-4 h-4 mr-1 animate-spin" />}
                    Save Changes
                  </Button>
                  <Button variant="outline" onClick={() => setEditMode(false)}>Cancel</Button>
                </div>
              </div>
            </TabsContent>
          )}

          {editMode && (
            <TabsContent value="visibility">
              <p className="text-sm text-muted-foreground mb-3">Toggle which fields collaborators can see on your TeamCard.</p>
              <div className="space-y-3">
                {ALL_FIELDS.map((field) => (
                  <div key={field.key} className="flex items-center justify-between py-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{field.label}</span>
                      {field.locked && <Lock className="w-3 h-3 text-muted-foreground" />}
                    </div>
                    <Switch
                      checked={field.locked || visibleFields.includes(field.key)}
                      onCheckedChange={() => !field.locked && toggleVisibility(field.key)}
                      disabled={field.locked}
                    />
                  </div>
                ))}
              </div>
              <Button className="mt-4" onClick={handleSave} disabled={updateCard.isPending}>
                {updateCard.isPending && <Loader2 className="w-4 h-4 mr-1 animate-spin" />}
                Save Visibility
              </Button>
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Add TeamCardSettings to Profile page**

In `src/pages/Profile.tsx`, add the import:

```typescript
import TeamCardSettings from "@/components/profile/TeamCardSettings";
```

Add the component after the Account Information card (before the Appearance card):

```tsx
        <TeamCardSettings />
```

- [ ] **Step 3: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/components/profile/TeamCardSettings.tsx src/pages/Profile.tsx
git commit -m "feat: add TeamCard settings with field visibility toggles and collaborator preview"
```

---

### Task 10: Registry List Page

**Goal:** Registry page with "My Works", "Pending Your Review", and "Shared Works" sections.

**Files:**
- Create: `src/pages/Registry.tsx`

**Acceptance Criteria:**
- [ ] "My Works" section — works the user created, filterable by artist
- [ ] "Pending Your Review" section — works needing collaborator action
- [ ] "Shared Works" section — all collaborations
- [ ] Status badges with colors, search by title
- [ ] "Register Work" creation dialog requiring artist + project

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create Registry.tsx**

Create `src/pages/Registry.tsx`.

**Source:** Read `docs/superpowers/plans/2026-03-28-rights-registry.md`, search for `### Task 7:`, then find `Create \`src/pages/Registry.tsx\`:` — copy the entire component (lines 2688-3027 in v1). No v2 modifications needed — use verbatim.

**What the component does:** My Works grid, Pending Your Review horizontal scroll, Shared Works grid, artist/search filters, Register Work dialog requiring artist + project selection, status badges (draft=yellow, pending_approval=amber, registered=green, disputed=red).

- [ ] **Step 2: Verify and commit**

Run: `npm run build`

```bash
git add src/pages/Registry.tsx
git commit -m "feat: add Registry list page with pending reviews and project-required creation"
```

---

### Task 11: Work Detail with Approval Workflow

**Goal:** Work detail page with ownership/licensing/agreements tabs, collaboration status, and approval actions.

**Files:**
- Create: `src/pages/WorkDetail.tsx`
- Create: `src/components/registry/OwnershipPanel.tsx`
- Create: `src/components/registry/LicensingPanel.tsx`
- Create: `src/components/registry/AgreementsPanel.tsx`
- Create: `src/components/registry/CollaborationStatus.tsx`
- Create: `src/components/registry/InviteCollaboratorModal.tsx`
- Create: `src/components/registry/ProofOfOwnership.tsx`

**Acceptance Criteria:**
- [ ] Two view modes: owner (full CRUD + submit + invite) vs collaborator (read-only + confirm/dispute)
- [ ] CollaborationStatus shows progress bar and per-collaborator status with verified badges
- [ ] OwnershipPanel shows approval badge next to each stakeholder
- [ ] InviteCollaboratorModal lets you select from artist roster (pulls email) or enter manually
- [ ] "Submit for Approval" only enabled in draft/disputed status with >=1 collaborator
- [ ] ProofOfOwnership export button downloads PDF

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create all registry components**

Create the following files. Base source code is in `docs/superpowers/plans/2026-03-28-rights-registry.md` (v1 plan). For each file below, the **Source** line tells you exactly where to find the code in v1, and **v2 changes** lists what to modify.

**`src/components/registry/CollaborationStatus.tsx`**
- **Source:** v1 file, search `### Task 8:`, then `Step 1: Create CollaborationStatus.tsx` (lines 3070-3173)
- **v2 changes:** (1) Import `useResendInvitation, useRevokeCollaborator` from `@/hooks/useRegistry`. (2) Add a "Resend" button next to each collaborator row where `c.status === "invited"` and the invitation is expired (`new Date(c.expires_at) < new Date()`). (3) Add a "Revoke" button (trash icon) next to each collaborator row when `isOwner` is true. (4) Wire both buttons to their respective mutations.
- **Props:** `workId: string, workStatus: string, collaborators: Collaborator[], isOwner: boolean`

**`src/components/registry/OwnershipPanel.tsx`**
- **Source:** v1 file, search `Step 2: Create OwnershipPanel.tsx` (lines 3180-3408)
- **v2 changes:** None — use verbatim.
- **Props:** `workId: string, stakes: OwnershipStake[], collaborators: Collaborator[], isOwner: boolean`

**`src/components/registry/LicensingPanel.tsx`**
- **Source:** No v1 code (v1 said "structure matches OwnershipPanel pattern"). Build from scratch following the OwnershipPanel pattern.
- **Specification:** CRUD with add/edit/delete dialog. License type selector (sync/mechanical/performance/print/digital/exclusive/non_exclusive/other). Territory input, date range (start_date required, end_date optional), terms textarea, status badges. Wrap all CRUD buttons in `{isOwner && (...)}`.
- **Props:** `workId: string, licenses: LicensingRight[], isOwner: boolean`

**`src/components/registry/AgreementsPanel.tsx`**
- **Source:** No v1 code (same). Build from scratch.
- **Specification:** Timeline layout. Create dialog with: agreement type selector (ownership_transfer/split_agreement/license_grant/amendment/termination), title, description textarea, effective_date, parties array (name/role/email inputs, add/remove), file_id optional, document_hash display. Agreements are immutable — no edit/delete buttons. Wrap "Record Agreement" trigger in `{isOwner && (...)}`.
- **Props:** `workId: string, agreements: Agreement[], isOwner: boolean`

**`src/components/registry/InviteCollaboratorModal.tsx`**
- **Source:** v1 file, search `Step 1: Create InviteCollaboratorModal.tsx` (lines 3773-3868)
- **v2 changes:** (1) Add optional `artists?: Array<{id: string, name: string, email: string}>` prop. (2) When provided, show a "Select from roster" dropdown at the top of the form. When an artist is selected, auto-fill `email` and `name` fields from the artist entry. (3) Below the dropdown, show "Or enter details manually" label.
- **Props:** `workId: string, stakes: OwnershipStake[], artists?: Array<{id,name,email}>, open: boolean, onOpenChange: (open: boolean) => void`

**`src/components/registry/ProofOfOwnership.tsx`**
- **Source:** v1 file, search `Step 1: Create ProofOfOwnership.tsx` (lines 3967-3992)
- **v2 changes:** None — use verbatim. Simple button calling `useExportProof()`.
- **Props:** `workId: string`

- [ ] **Step 2: Create WorkDetail.tsx**

**Source:** v1 file, search `Step 5: Create WorkDetail.tsx` (lines 3443-3735) for the base component, plus v1 Task 10 Step 2 (lines 3997-4031) for ProofOfOwnership/Invite additions.

**v2 changes on top of v1 code:**
1. Import `useResendInvitation, useRevokeCollaborator` from `@/hooks/useRegistry`
2. Import `InviteCollaboratorModal` and `ProofOfOwnership`
3. Add `const [showInvite, setShowInvite] = useState(false)` state
4. In the owner actions header area, add: `<ProofOfOwnership workId={work.id} />`, then an "Invite" button (`<UserPlus>` icon) that opens `setShowInvite(true)`, then `<InviteCollaboratorModal workId={work.id} stakes={work.stakes || []} open={showInvite} onOpenChange={setShowInvite} />`
5. The edit dialog should NOT show `pending_approval` as a manual status option (it's set automatically)

**Props used:** `useParams<{ workId: string }>`, `useWorkFull`, `useUpdateWork`, `useDeleteWork`, `useConfirmStake`, `useDisputeStake`

- [ ] **Step 3: Verify and commit**

Run: `npm run build`

```bash
git add src/pages/WorkDetail.tsx src/components/registry/
git commit -m "feat: add Work Detail page with collaboration, approval workflow, and all registry panels"
```

---

### Task 12: Invitation Flow & Claim Handler

**Goal:** Handle invitation claim links and redirect flow for new/existing users.

**Files:**
- Create: `src/pages/InviteClaim.tsx`

**Acceptance Criteria:**
- [ ] `/tools/registry/invite/:token` claims invitation and redirects to work detail
- [ ] If not logged in, redirects to auth then back to claim
- [ ] Success toast on claim

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create InviteClaim.tsx**

Create `src/pages/InviteClaim.tsx`.

**Source:** v1 file, search `Step 2: Create InviteClaim.tsx` (lines 3875-3923). Copy verbatim — no v2 changes needed.

**What the component does:** Reads `token` from URL params via `useParams`. If user not logged in, redirects to `/auth?redirect=/tools/registry/invite/${token}`. If logged in, calls `useClaimInvitation` with the token. On success: toast + navigate to `/tools/registry/${data.work_id}`. On error: error toast + navigate to `/tools/registry`. Shows centered `<Loader2>` spinner while claiming. Uses `useState(false)` guard to prevent double-claims.

- [ ] **Step 2: Verify and commit**

Run: `npm run build`

```bash
git add src/pages/InviteClaim.tsx
git commit -m "feat: add invitation claim handler with auth redirect"
```

---

### Task 13: Artist Profile with Verified Badge & Option C Merge

**Goal:** Update artist profile to show verified badge when linked, overlay TeamCard data on identity fields (Option C), and add "My Notes" section using the rich notes editor.

**Files:**
- Modify: `src/pages/ArtistProfile.tsx`
- Modify: `src/pages/Artists.tsx`

**Acceptance Criteria:**
- [ ] Verified badge appears next to artist name when `verified === true`
- [ ] When verified, identity fields (name, bio, avatar, socials, DSPs) display TeamCard data instead of local artist data
- [ ] "My Notes" collapsible section at bottom using `NotesView` with `artistId` scope
- [ ] Private fields (your notes, genres you assigned, contract status) always show from local artist data
- [ ] Clear visual distinction between TeamCard zone and private zone
- [ ] Artists list page (Artists.tsx) uses batch overlay endpoint so verified artists show live TeamCard identity

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Add verified badge and TeamCard overlay to ArtistProfile**

In `src/pages/ArtistProfile.tsx`, add imports:

```typescript
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { CheckCircle } from "lucide-react";
import NotesView from "@/components/notes/NotesView";
```

Add a query to fetch the artist with TeamCard overlay:

```typescript
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

// Inside the component, after fetching artist:
const teamcardQuery = useQuery({
  queryKey: ["artist-teamcard", id],
  queryFn: async () => {
    if (!user?.id || !id) return null;
    const res = await fetch(`${API_URL}/registry/artists/${id}/with-teamcard?user_id=${user.id}`);
    if (!res.ok) return null;
    return res.json();
  },
  enabled: !!user?.id && !!id,
});

const teamcard = teamcardQuery.data?.teamcard;
const isVerified = teamcardQuery.data?.verified === true;
```

In the profile header area, add the verified badge:

```tsx
{isVerified && (
  <Badge className="bg-green-100 text-green-800 flex items-center gap-1">
    <CheckCircle className="w-3 h-3" /> Verified
  </Badge>
)}
```

When displaying identity fields, prefer TeamCard data if available:

```tsx
// For name display:
const displayName = (isVerified && teamcard?.display_name) || artist.name;

// For bio:
const displayBio = (isVerified && teamcard?.bio) || artist.bio;

// For avatar:
const displayAvatar = (isVerified && teamcard?.avatar_url) || artist.avatar_url;
```

- [ ] **Step 2: Add My Notes section**

At the bottom of the artist profile page, add a collapsible notes section:

```tsx
        {/* My Notes — private to you */}
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4">My Notes</h3>
          <NotesView scope={{ artistId: id }} />
        </div>
```

- [ ] **Step 3: Update Artists.tsx list page to show TeamCard data for verified artists**

In `src/pages/Artists.tsx`, update the artist fetch to use the batch overlay endpoint so verified artists display their live TeamCard identity (name, avatar) instead of stale local data:

```typescript
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

// Replace the existing artist fetch with:
const { data: artists, isLoading } = useQuery({
  queryKey: ["artists-with-teamcards", user?.id],
  queryFn: async () => {
    if (!user?.id) return [];
    const res = await fetch(`${API_URL}/registry/artists/with-teamcards?user_id=${user.id}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.artists || [];
  },
  enabled: !!user?.id,
});
```

When rendering artist cards, prefer TeamCard fields when available:

```tsx
const displayName = artist.teamcard?.display_name || artist.name;
const displayAvatar = artist.teamcard?.avatar_url || artist.avatar_url;
const isVerified = artist.verified === true;

// Show verified badge on card
{isVerified && (
  <Badge className="bg-green-100 text-green-800 text-xs">Verified</Badge>
)}
```

- [ ] **Step 4: Verify and commit**

Run: `npm run build`

```bash
git add src/pages/ArtistProfile.tsx src/pages/Artists.tsx
git commit -m "feat: add verified badge, TeamCard overlay, and private notes to artist profile and list"
```

---

### Task 14: Project Detail Page with About & Notes

**Goal:** New project detail page with a rich About section (BlockNote editor) and a Notes section, accessible to the project creator and linked collaborators.

**Files:**
- Create: `src/pages/ProjectDetail.tsx`

**Acceptance Criteria:**
- [ ] Route: `/projects/:projectId`
- [ ] About section using BlockNote editor (auto-saves to `projects.about_content`)
- [ ] Notes section using `NotesView` with `projectId` scope
- [ ] Only project creator can edit About and create notes
- [ ] Linked collaborators (via works on this project) can read About and project notes
- [ ] Project header shows name, description, artist name

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create ProjectDetail.tsx**

Create `src/pages/ProjectDetail.tsx`:

```tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useQuery } from "@tanstack/react-query";
import { useProjectAbout, useUpdateProjectAbout } from "@/hooks/useNotes";
import NotesEditor from "@/components/notes/NotesEditor";
import NotesView from "@/components/notes/NotesView";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Music, ArrowLeft, Loader2, FileText, StickyNote, Info } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

const ProjectDetail = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const updateAbout = useUpdateProjectAbout();

  // Fetch project with artist info
  const projectQuery = useQuery({
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

  const { data: aboutContent = [] } = useProjectAbout(projectId);

  const project = projectQuery.data;
  const isOwner = project?.artists?.user_id === user?.id;

  const handleAboutChange = (content: unknown[]) => {
    if (!projectId || !isOwner) return;
    updateAbout.mutate({ projectId, about_content: content });
  };

  if (projectQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Project not found</p>
        <Button variant="outline" onClick={() => navigate("/portfolio")}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Portfolio
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/portfolio")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Portfolio
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-foreground">{project.name}</h2>
          <p className="text-muted-foreground">
            {project.artists?.name || "Unknown Artist"}
            {project.description && ` — ${project.description}`}
          </p>
        </div>

        <Tabs defaultValue="about">
          <TabsList className="mb-6">
            <TabsTrigger value="about" className="gap-1.5">
              <Info className="w-4 h-4" /> About
            </TabsTrigger>
            <TabsTrigger value="notes" className="gap-1.5">
              <StickyNote className="w-4 h-4" /> Notes
            </TabsTrigger>
          </TabsList>

          <TabsContent value="about">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">About This Project</h3>
                {!isOwner && (
                  <span className="text-xs text-muted-foreground">Read only</span>
                )}
              </div>
              <NotesEditor
                key={`about-${projectId}`}
                initialContent={aboutContent as unknown[]}
                onChange={isOwner ? handleAboutChange : undefined}
                editable={isOwner}
              />
            </div>
          </TabsContent>

          <TabsContent value="notes">
            {projectId && <NotesView scope={{ projectId }} />}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default ProjectDetail;
```

- [ ] **Step 2: Verify and commit**

Run: `npm run build`

```bash
git add src/pages/ProjectDetail.tsx
git commit -m "feat: add project detail page with rich About section and notes"
```

---

### Task 15: Workspace Notifications Integration

**Goal:** Wire registry notifications into the Workspace tool's Notifications tab.

**Files:**
- Create: `src/components/workspace/RegistryNotifications.tsx`
- Modify: `src/pages/Workspace.tsx`

**Acceptance Criteria:**
- [ ] Notifications tab shows collaboration events: invitations, confirmations, disputes, verifications
- [ ] Each notification links to the relevant work detail page
- [ ] "Mark all as read" button
- [ ] Unread count badge on the Notifications tab trigger

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create RegistryNotifications.tsx**

Create `src/components/workspace/RegistryNotifications.tsx`.

**Source:** v1 file, search `Step 2: Create RegistryNotifications component` (lines 4148-4245). Copy verbatim — no v2 changes needed.

**What the component does:** Fetches all notifications via `useRegistryNotifications()`. Shows type badges (invitation=blue, confirmation=green, dispute=red, status_change=purple, verification=purple). Unread indicator dot on each item. Click marks as read and navigates to `/tools/registry/${n.work_id}`. "Mark all read" button at top. Empty state with Bell icon.

- [ ] **Step 2: Update Workspace.tsx**

In `src/pages/Workspace.tsx`, add imports:

```typescript
import { RegistryNotifications } from "@/components/workspace/RegistryNotifications";
import { useUnreadCount } from "@/hooks/useRegistryNotifications";
```

Add unread count inside the component:

```typescript
const unreadNotifications = useUnreadCount();
```

Update the Notifications tab trigger to show the badge:

```tsx
            <TabsTrigger value="notifications" className="gap-2">
              <Bell className="w-4 h-4" />
              Notifications
              {unreadNotifications > 0 && (
                <span className="ml-1 px-1.5 py-0.5 text-[10px] font-bold bg-destructive text-destructive-foreground rounded-full">
                  {unreadNotifications}
                </span>
              )}
            </TabsTrigger>
```

Replace the placeholder notifications content with:

```tsx
          <TabsContent value="notifications">
            <RegistryNotifications />
          </TabsContent>
```

- [ ] **Step 3: Verify and commit**

Run: `npm run build`

```bash
git add src/components/workspace/RegistryNotifications.tsx src/pages/Workspace.tsx
git commit -m "feat: add registry notifications to Workspace with unread badge"
```
