# Rights & Ownership Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a collaborative Rights & Ownership Registry that serves as the shared source of truth for master ownership, publishing splits, licensing rights, and timestamped agreements. Collaborators can be invited to view and confirm their stakes — the work only reaches "registered" status when all parties agree. Includes exportable proof-of-ownership documents showing approval status.

**Architecture:** Projects are the container; works are the granular unit of ownership. An album project has N tracks, each with their own stakes, licenses, agreements, and collaborators. Five Supabase tables (`works_registry`, `ownership_stakes`, `licensing_rights`, `registry_agreements`, `registry_collaborators`) with collaborator-aware RLS. `project_id` is required on `works_registry` — every work belongs to a project. A `registry/` backend module handles CRUD, invitation emails (via Resend), approval workflow (invite → confirm/dispute → auto-status), and PDF export. Only the project owner can create works and invite collaborators. Artist profiles are claimable — when you create an artist with an email, the system checks if that person has a platform account. If yes, the profile auto-claims and they can edit their own bio/links. If not, they get an invite; when they sign up, the profile claims to them. Invitations always send both email + in-app notification, and expire after 48h. When inviting a collaborator on a work, you can select from your artist roster (email pre-filled) or enter a new email. Six Supabase tables for the registry (adds `registry_notifications`), plus `artists` table gains `claimed_by`/`claim_status` columns. Frontend includes a registry listing with three views: "My Works" (created by user), "Pending Your Review" (needs action), and "Shared Works" (all collaborations across all artists/projects). Notifications also surface in the Workspace tool's notifications tab. Invitation claim flow lets collaborators join via email link.

**Tech Stack:** Supabase (PostgreSQL + RLS), FastAPI (Python backend), Resend (invitation emails), React + TypeScript + React Query (frontend), ReportLab (PDF generation), shadcn/ui + Tailwind (UI components)

**Collaboration Flow:**
1. Creator registers a work (status: `draft`) and adds ownership stakes
2. Creator invites collaborators by email for each stake
3. Creator clicks "Submit for Approval" → status: `pending_approval`, emails sent
4. Collaborators receive invitation emails with claim links
5. Collaborators log in, claim invitation, and see the full work + their stake
6. Each collaborator confirms or disputes their stake (dispute requires reason)
7. When ALL collaborators confirm → status auto-transitions to `registered`
8. If ANY collaborator disputes → status transitions to `disputed`

---

## File Structure

```
supabase/migrations/
  20260329000000_create_rights_registry.sql     # 6 tables + collaborator-aware RLS
  20260329100000_add_artist_claim.sql           # Add claimed_by/claim_status to artists + updated RLS

src/backend/registry/
  __init__.py
  models.py                                      # Pydantic models including collaboration
  service.py                                     # CRUD + invite/confirm/dispute + auto-status + notifications
  router.py                                      # All endpoints including collaboration + notifications
  emails.py                                      # Invitation email via Resend
  pdf_generator.py                               # Proof of ownership with approval status

src/backend/main.py                              # Mount registry router (modify)

src/hooks/useRegistryNotifications.ts            # Hook for notification queries + mark-read
src/components/workspace/RegistryNotifications.tsx # Notifications panel for Workspace tab

src/integrations/supabase/types.ts               # Add 5 table types (modify)
src/hooks/useRegistry.ts                         # React Query hooks for all CRUD + collaboration

src/pages/Registry.tsx                           # Work listing + pending reviews
src/pages/WorkDetail.tsx                         # Work detail + approval workflow
src/pages/InviteClaim.tsx                        # Invitation claim handler

src/components/registry/
  OwnershipPanel.tsx                             # Master + publishing with approval status per holder
  LicensingPanel.tsx                             # Licensing rights management
  AgreementsPanel.tsx                            # Timestamped agreements timeline
  CollaborationStatus.tsx                        # Approval workflow status display
  InviteCollaboratorModal.tsx                    # Invite stakeholders by email
  ProofOfOwnership.tsx                           # Export button

src/App.tsx                                      # Add routes (modify)
src/pages/Tools.tsx                              # Add Registry card (modify)
```

---

### Task 1: Database Schema with Collaboration Layer

**Goal:** Create all 5 Rights Registry tables with collaborator-aware RLS and approval workflow.

**Files:**
- Create: `supabase/migrations/20260329000000_create_rights_registry.sql`

**Acceptance Criteria:**
- [ ] `works_registry` with `project_id NOT NULL` and status flow: draft → pending_approval → registered / disputed
- [ ] `ownership_stakes` with percentage validation
- [ ] `licensing_rights` with status tracking
- [ ] `registry_agreements` with immutable timestamps
- [ ] `registry_collaborators` with invite/confirm/dispute workflow and invite tokens
- [ ] RLS policies allow collaborators to SELECT works/stakes they're involved in
- [ ] Creator retains full CRUD; collaborators get read-only + confirm/dispute

**Verify:** `supabase db push` succeeds

**Steps:**

- [ ] **Step 1: Create the migration file**

Create `supabase/migrations/20260329000000_create_rights_registry.sql`:

```sql
-- ============================================================
-- Rights & Ownership Registry with Collaboration Layer
-- ============================================================

-- 1. works_registry
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

-- 2. ownership_stakes
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

-- 3. licensing_rights
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

-- 4. registry_agreements (immutable — no update/delete)
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

-- 5. registry_collaborators — invitation + approval tracking
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
  invited_at timestamptz not null default now(),
  responded_at timestamptz
);

-- ============================================================
-- Row Level Security — Collaborator-Aware
-- ============================================================

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

-- ============================================================
-- Indexes
-- ============================================================
create index idx_works_registry_user_id on works_registry(user_id);
create index idx_works_registry_artist_id on works_registry(artist_id);
create index idx_ownership_stakes_work_id on ownership_stakes(work_id);
create index idx_ownership_stakes_user_id on ownership_stakes(user_id);
create index idx_licensing_rights_work_id on licensing_rights(work_id);
create index idx_registry_agreements_work_id on registry_agreements(work_id);
create index idx_registry_collaborators_work_id on registry_collaborators(work_id);
create index idx_registry_collaborators_user_id on registry_collaborators(collaborator_user_id);
create index idx_registry_collaborators_token on registry_collaborators(invite_token);
create index idx_registry_collaborators_email on registry_collaborators(email);

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

create trigger works_registry_updated_at
  before update on works_registry
  for each row execute function update_updated_at_column();

create trigger ownership_stakes_updated_at
  before update on ownership_stakes
  for each row execute function update_updated_at_column();

create trigger licensing_rights_updated_at
  before update on licensing_rights
  for each row execute function update_updated_at_column();
```

- [ ] **Step 2: Apply migration**

Run: `supabase db push`
Expected: 5 tables created with all policies

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260329000000_create_rights_registry.sql
git commit -m "feat: add rights registry schema with collaboration-aware RLS"
```

---

### Task 2: Backend Models & Service with Collaboration Logic

**Goal:** Pydantic models and service layer for CRUD + collaboration (invite, confirm, dispute, auto-status transitions).

**Files:**
- Create: `src/backend/registry/__init__.py`
- Create: `src/backend/registry/models.py`
- Create: `src/backend/registry/service.py`

**Acceptance Criteria:**
- [ ] Models for works, stakes, licenses, agreements, and collaborator invitations
- [ ] CRUD service for all 5 tables
- [ ] `invite_collaborator` creates collaborator row and returns invite token
- [ ] `claim_invitation` links a user_id to an invitation by token
- [ ] `confirm_stake` / `dispute_stake` update collaborator status
- [ ] `check_and_update_work_status` auto-transitions: all confirmed → registered, any disputed → disputed
- [ ] `submit_for_approval` validates all stakes have collaborators and transitions to pending_approval

**Verify:** `cd src/backend && python -c "from registry.models import WorkCreate, CollaboratorInvite; print('OK')"`

**Steps:**

- [ ] **Step 1: Create module init**

Create `src/backend/registry/__init__.py` (empty file).

- [ ] **Step 2: Create models**

Create `src/backend/registry/models.py`:

```python
"""Pydantic models for the Rights & Ownership Registry."""

from pydantic import BaseModel
from typing import Optional, List
from datetime import date


# --- Works ---

class WorkCreate(BaseModel):
    artist_id: str
    project_id: str  # required — every work belongs to a project
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
    email: str
    name: str
    role: str


class DisputeRequest(BaseModel):
    reason: str
```

- [ ] **Step 3: Create service layer**

Create `src/backend/registry/service.py`:

```python
"""Service layer for the Rights & Ownership Registry with collaboration."""

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
        .neq("user_id", user_id)  # exclude works they own
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data


async def get_works_by_project(db: Client, user_id: str, project_id: str):
    """Get all works inside a project (owner or collaborator)."""
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
    data["user_id"] = user_id
    result = db.table("works_registry").insert(data).execute()
    return result.data[0] if result.data else None


async def update_work(db: Client, user_id: str, work_id: str, data: dict):
    result = (
        db.table("works_registry")
        .update(data)
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_work(db: Client, user_id: str, work_id: str):
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
    """Check if a user with this email exists on the platform. Returns user_id or None."""
    result = db.auth.admin.list_users()
    for u in (result or []):
        if hasattr(u, 'email') and u.email and u.email.lower() == email.lower():
            return u.id
    return None


async def invite_collaborator(db: Client, invited_by: str, data: dict, work_title: str = ""):
    """Create a collaborator invitation. Auto-links if user exists on platform."""
    data["invited_by"] = invited_by

    # Check if the collaborator already has an account
    existing_user_id = await check_user_exists(db, data["email"])
    if existing_user_id:
        data["collaborator_user_id"] = existing_user_id

    result = db.table("registry_collaborators").insert(data).execute()
    collab = result.data[0] if result.data else None

    # Create in-app notification if user exists on platform
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

    # Email is ALWAYS sent (existing user or not) — router handles this
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
    return (updated.data[0] if updated.data else collab), None


async def confirm_stake(db: Client, collaborator_id: str, user_id: str):
    """Collaborator confirms their stake."""
    from datetime import datetime, timezone
    result = (
        db.table("registry_collaborators")
        .update({
            "status": "confirmed",
            "responded_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .execute()
    )
    if result.data:
        # Check if all collaborators confirmed → auto-register work
        collab = result.data[0]
        await check_and_update_work_status(db, collab["work_id"])
    return result.data[0] if result.data else None


async def dispute_stake(db: Client, collaborator_id: str, user_id: str, reason: str):
    """Collaborator disputes their stake with a reason."""
    from datetime import datetime, timezone
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
        # Mark work as disputed
        db.table("works_registry").update({"status": "disputed"}).eq("id", collab["work_id"]).execute()
    return result.data[0] if result.data else None


async def check_and_update_work_status(db: Client, work_id: str):
    """Auto-transition work status based on collaborator responses."""
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
    # else: some still "invited" — stay as pending_approval


async def submit_for_approval(db: Client, user_id: str, work_id: str):
    """Transition work from draft to pending_approval."""
    # Verify work exists and user owns it
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

    # Check there are collaborators
    collabs = (
        db.table("registry_collaborators")
        .select("id")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    if not collabs.data:
        return None, "No collaborators invited — add at least one before submitting"

    # Reset any disputed collaborators back to invited
    db.table("registry_collaborators").update(
        {"status": "invited", "dispute_reason": None, "responded_at": None}
    ).eq("work_id", work_id).eq("status", "disputed").execute()

    # Transition work status
    result = (
        db.table("works_registry")
        .update({"status": "pending_approval"})
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None, None


# ============================================================
# Full Work Data (for export / detail view)
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
# Notifications
# ============================================================

async def create_notification(
    db: Client, user_id: str, work_id: str,
    notification_type: str, title: str, message: str,
    metadata: dict = None,
):
    """Create an in-app notification for a user."""
    db.table("registry_notifications").insert({
        "user_id": user_id,
        "work_id": work_id,
        "type": notification_type,
        "title": title,
        "message": message,
        "metadata": metadata or {},
    }).execute()


async def get_notifications(db: Client, user_id: str, unread_only: bool = False):
    """Get notifications for a user, optionally filtered to unread."""
    query = db.table("registry_notifications").select("*").eq("user_id", user_id)
    if unread_only:
        query = query.eq("read", False)
    result = query.order("created_at", desc=True).limit(50).execute()
    return result.data


async def mark_notification_read(db: Client, user_id: str, notification_id: str):
    """Mark a single notification as read."""
    db.table("registry_notifications").update({"read": True}).eq("id", notification_id).eq("user_id", user_id).execute()


async def mark_all_notifications_read(db: Client, user_id: str):
    """Mark all notifications as read for a user."""
    db.table("registry_notifications").update({"read": True}).eq("user_id", user_id).eq("read", False).execute()


# ============================================================
# Utility
# ============================================================

def compute_document_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
```

- [ ] **Step 4: Verify imports**

Run: `cd src/backend && python -c "from registry.models import WorkCreate, CollaboratorInvite, DisputeRequest; from registry.service import get_works, invite_collaborator, confirm_stake; print('All OK')"`
Expected: `All OK`

- [ ] **Step 5: Commit**

```bash
git add src/backend/registry/
git commit -m "feat: add registry models and service with collaboration workflow"
```

---

### Task 3: Backend Router with Collaboration Endpoints

**Goal:** FastAPI router with all CRUD + collaboration endpoints, mounted in main.py.

**Files:**
- Create: `src/backend/registry/router.py`
- Modify: `src/backend/main.py:34-48`

**Acceptance Criteria:**
- [ ] Standard CRUD for works, stakes, licenses, agreements
- [ ] `POST /registry/collaborators/invite` — create invitation
- [ ] `POST /registry/collaborators/claim` — claim invitation by token
- [ ] `POST /registry/collaborators/{id}/confirm` — confirm stake
- [ ] `POST /registry/collaborators/{id}/dispute` — dispute with reason
- [ ] `POST /registry/works/{work_id}/submit-for-approval` — transition to pending_approval
- [ ] `GET /registry/works/{work_id}/full` — full work with collaborator status
- [ ] `GET /registry/works/pending-review` — works pending user's review as collaborator
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
    """All works where user is a collaborator (any status — not just pending)."""
    works = await service.get_works_as_collaborator(_get_supabase(), user_id)
    return {"works": works}


@router.get("/works/by-project/{project_id}")
async def list_works_by_project(project_id: str, user_id: str = Query(...)):
    """All works inside a specific project."""
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
    return result


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
    if "start_date" in data and data["start_date"]:
        data["start_date"] = data["start_date"].isoformat()
    if "end_date" in data and data["end_date"]:
        data["end_date"] = data["end_date"].isoformat()
    lic = await service.create_license(_get_supabase(), user_id, data)
    if not lic:
        raise HTTPException(status_code=500, detail="Failed to create license")
    return lic


@router.put("/licenses/{license_id}")
async def update_license(license_id: str, body: LicenseUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "start_date" in data and data["start_date"]:
        data["start_date"] = data["start_date"].isoformat()
    if "end_date" in data and data["end_date"]:
        data["end_date"] = data["end_date"].isoformat()
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
async def list_collaborators(work_id: str = Query(...)):
    collabs = await service.get_collaborators(_get_supabase(), work_id)
    return {"collaborators": collabs}


@router.post("/collaborators/invite")
async def invite_collaborator(body: CollaboratorInvite, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    collab = await service.invite_collaborator(_get_supabase(), user_id, data)
    if not collab:
        raise HTTPException(status_code=500, detail="Failed to invite collaborator")
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

- [ ] **Step 3: Verify**

Run: `cd src/backend && python -c "from registry.router import router; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/backend/registry/router.py src/backend/main.py
git commit -m "feat: add registry router with collaboration endpoints"
```

---

### Task 4: Backend Email Invitations via Resend

**Goal:** Send invitation emails to collaborators using the existing Resend integration.

**Files:**
- Create: `src/backend/registry/emails.py`
- Modify: `src/backend/registry/router.py` (send email on invite)

**Acceptance Criteria:**
- [ ] HTML email with work title, inviter name, collaborator role, claim link
- [ ] Claim link: `{FRONTEND_URL}/tools/registry/invite/{invite_token}`
- [ ] Uses `RESEND_API_KEY` and `RESEND_FROM_EMAIL` env vars (existing pattern)
- [ ] Email ALWAYS sent — whether user exists on platform or not
- [ ] If user exists: also auto-linked + in-app notification (email + notification)
- [ ] If user does NOT exist: email with claim link routes through account creation
- [ ] Invitation expires after 48h (`expires_at` column) — claim returns 410 if expired
- [ ] Email preferences configurable in profile (deferred — placeholder hook for later)

**Verify:** `cd src/backend && python -c "from registry.emails import send_invitation_email; print('OK')"`

**Steps:**

- [ ] **Step 1: Create emails module**

Create `src/backend/registry/emails.py`:

```python
"""Invitation email for the Rights Registry."""

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

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="color: #1a3a2a; font-size: 24px; margin: 0;">Msanii Rights Registry</h1>
      </div>

      <p style="font-size: 16px; color: #333;">Hi {recipient_name},</p>

      <p style="font-size: 15px; color: #555;">
        <strong>{inviter_name}</strong> has listed you as a <strong>{role}</strong> on the work
        <strong>"{work_title}"</strong> and is requesting you confirm your ownership stake.
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
        and confirm or dispute your stake.
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
            "subject": f"{inviter_name} needs you to confirm your stake on \"{work_title}\"",
            "html": html_body,
        })
        return response
    except Exception as e:
        print(f"Warning: Failed to send invitation email: {e}")
        return None
```

- [ ] **Step 2: Integrate email sending into invite endpoint**

In `src/backend/registry/router.py`, update the `invite_collaborator` endpoint. The service layer already checks if the user exists and creates an in-app notification if so. The router only sends email if the user does NOT exist on the platform:

```python
@router.post("/collaborators/invite")
async def invite_collaborator(body: CollaboratorInvite, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)

    # Get work title for notification/email
    work = _get_supabase().table("works_registry").select("title").eq("id", body.work_id).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"

    # Service checks if user exists → auto-links + creates notification if yes
    collab = await service.invite_collaborator(_get_supabase(), user_id, data, work_title=work_title)
    if not collab:
        raise HTTPException(status_code=500, detail="Failed to invite collaborator")

    # Always send email — even if user exists on platform (they get both email + in-app notification)
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
```

- [ ] **Step 3: Verify**

Run: `cd src/backend && python -c "from registry.emails import send_invitation_email; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/backend/registry/emails.py src/backend/registry/router.py
git commit -m "feat: add invitation email with claim link via Resend"
```

---

### Task 5: Backend Proof of Ownership PDF with Approval Status

**Goal:** Generate PDF showing ownership, licensing, agreements, AND approval status per stakeholder.

**Files:**
- Create: `src/backend/registry/pdf_generator.py`
- Modify: `src/backend/registry/router.py` (add export endpoint)

**Acceptance Criteria:**
- [ ] PDF header: work title, ISRC/ISWC/UPC, type, release date
- [ ] Ownership tables show holder + percentage + **approval status** (Confirmed/Pending/Disputed)
- [ ] Licensing and agreement sections
- [ ] Footer: work status, generation timestamp, document hash
- [ ] `GET /registry/works/{work_id}/export` returns PDF

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

    # Build collaborator lookup: stake_id → collaborator status
    collaborators = work_data.get("collaborators", [])
    stake_approval = {}  # stake_id → {status, name}
    email_approval = {}  # holder_email → {status, name}
    for c in collaborators:
        if c.get("stake_id"):
            stake_approval[c["stake_id"]] = {"status": c["status"], "name": c["name"]}
        if c.get("email"):
            email_approval[c["email"].lower()] = {"status": c["status"], "name": c["name"]}

    # --- Header ---
    elements.append(Paragraph("PROOF OF OWNERSHIP", title_style))
    elements.append(Paragraph("Rights & Ownership Registry Certificate", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND, spaceAfter=20))

    # --- Work Status Banner ---
    work_status = (work_data.get("status") or "draft").replace("_", " ").title()
    status_color = {"Registered": GREEN, "Disputed": RED, "Pending Approval": AMBER}.get(work_status, colors.HexColor("#666"))
    elements.append(Paragraph(f"<b>Registry Status: <font color='{status_color}'>{work_status}</font></b>", body_style))
    elements.append(Spacer(1, 8))

    # --- Work Details ---
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

    # --- Ownership with Approval Status ---
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
            # Determine approval status
            approval = "—"
            sid = s.get("id")
            hemail = (s.get("holder_email") or "").lower()
            if sid in stake_approval:
                approval = stake_approval[sid]["status"].title()
            elif hemail and hemail in email_approval:
                approval = email_approval[hemail]["status"].title()

            rows.append([
                s.get("holder_name", ""),
                s.get("holder_role", ""),
                f"{s.get('percentage', 0):.2f}%",
                s.get("publisher_or_label") or "—",
                approval,
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
        # Color approval cells
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

    # --- Licensing Rights ---
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

    # --- Agreement History ---
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
            elements.append(Paragraph(
                f"Effective: {agr.get('effective_date', '—')} | Recorded: {agr.get('created_at', '—')}", small_style
            ))
            elements.append(Paragraph(f"Parties: {party_names}", small_style))
            if agr.get("document_hash"):
                elements.append(Paragraph(f"Hash: {agr['document_hash']}", small_style))
            elements.append(Spacer(1, 6))

    # --- Footer ---
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

- [ ] **Step 2: Add export endpoint to router**

Add to `src/backend/registry/router.py` (at end of Works section, before Stakes):

```python
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
```

- [ ] **Step 3: Verify and commit**

Run: `cd src/backend && python -c "from registry.pdf_generator import generate_proof_of_ownership_pdf; print('OK')"`

```bash
git add src/backend/registry/pdf_generator.py src/backend/registry/router.py
git commit -m "feat: add proof-of-ownership PDF with approval status per stakeholder"
```

---

### Task 6: Frontend Types, Hook & Navigation

**Goal:** TypeScript types for all 5 tables, useRegistry hook with collaboration mutations, routing including invite claim route.

**Files:**
- Modify: `src/integrations/supabase/types.ts:261`
- Create: `src/hooks/useRegistry.ts`
- Modify: `src/App.tsx`
- Modify: `src/pages/Tools.tsx`

**Acceptance Criteria:**
- [ ] Types for works_registry, ownership_stakes, licensing_rights, registry_agreements, registry_collaborators
- [ ] Hook exports: all CRUD + `useMyCollaborations`, `useWorksByProject`, `useInviteCollaborator`, `useClaimInvitation`, `useConfirmStake`, `useDisputeStake`, `useSubmitForApproval`, `useExportProof`
- [ ] Separate `useRegistryNotifications.ts` hook: `useRegistryNotifications(unreadOnly?)`, `useMarkNotificationRead()`, `useMarkAllRead()`, `useUnreadCount()`
- [ ] Routes: `/tools/registry`, `/tools/registry/:workId`, `/tools/registry/invite/:token`
- [ ] Registry card on Tools page

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Add TypeScript types**

In `src/integrations/supabase/types.ts`, add these 5 table definitions inside `Tables: {}` (after the `projects` table, before the closing `}` of `Tables`):

```typescript
      works_registry: {
        Row: {
          id: string
          user_id: string
          artist_id: string
          project_id: string
          title: string
          work_type: string
          isrc: string | null
          iswc: string | null
          upc: string | null
          release_date: string | null
          status: string
          notes: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id?: string
          artist_id: string
          project_id: string
          title: string
          work_type?: string
          isrc?: string | null
          iswc?: string | null
          upc?: string | null
          release_date?: string | null
          status?: string
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          artist_id?: string
          project_id?: string
          title?: string
          work_type?: string
          isrc?: string | null
          iswc?: string | null
          upc?: string | null
          release_date?: string | null
          status?: string
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "works_registry_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "works_registry_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      ownership_stakes: {
        Row: {
          id: string
          work_id: string
          user_id: string
          stake_type: string
          holder_name: string
          holder_role: string
          percentage: number
          holder_email: string | null
          holder_ipi: string | null
          publisher_or_label: string | null
          notes: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          work_id: string
          user_id?: string
          stake_type: string
          holder_name: string
          holder_role: string
          percentage: number
          holder_email?: string | null
          holder_ipi?: string | null
          publisher_or_label?: string | null
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          work_id?: string
          user_id?: string
          stake_type?: string
          holder_name?: string
          holder_role?: string
          percentage?: number
          holder_email?: string | null
          holder_ipi?: string | null
          publisher_or_label?: string | null
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "ownership_stakes_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      licensing_rights: {
        Row: {
          id: string
          work_id: string
          user_id: string
          license_type: string
          licensee_name: string
          licensee_email: string | null
          territory: string
          start_date: string
          end_date: string | null
          terms: string | null
          status: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          work_id: string
          user_id?: string
          license_type: string
          licensee_name: string
          licensee_email?: string | null
          territory?: string
          start_date: string
          end_date?: string | null
          terms?: string | null
          status?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          work_id?: string
          user_id?: string
          license_type?: string
          licensee_name?: string
          licensee_email?: string | null
          territory?: string
          start_date?: string
          end_date?: string | null
          terms?: string | null
          status?: string
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "licensing_rights_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      registry_agreements: {
        Row: {
          id: string
          work_id: string
          user_id: string
          agreement_type: string
          title: string
          description: string | null
          effective_date: string
          parties: Json
          file_id: string | null
          document_hash: string | null
          created_at: string
        }
        Insert: {
          id?: string
          work_id: string
          user_id?: string
          agreement_type: string
          title: string
          description?: string | null
          effective_date: string
          parties?: Json
          file_id?: string | null
          document_hash?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          work_id?: string
          user_id?: string
          agreement_type?: string
          title?: string
          description?: string | null
          effective_date?: string
          parties?: Json
          file_id?: string | null
          document_hash?: string | null
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "registry_agreements_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      registry_collaborators: {
        Row: {
          id: string
          work_id: string
          stake_id: string | null
          invited_by: string
          collaborator_user_id: string | null
          email: string
          name: string
          role: string
          status: string
          invite_token: string
          dispute_reason: string | null
          invited_at: string
          expires_at: string
          responded_at: string | null
        }
        Insert: {
          id?: string
          work_id: string
          stake_id?: string | null
          invited_by?: string
          collaborator_user_id?: string | null
          email: string
          name: string
          role: string
          status?: string
          invite_token?: string
          dispute_reason?: string | null
          invited_at?: string
          expires_at?: string
          responded_at?: string | null
        }
        Update: {
          id?: string
          work_id?: string
          stake_id?: string | null
          invited_by?: string
          collaborator_user_id?: string | null
          email?: string
          name?: string
          role?: string
          status?: string
          invite_token?: string
          dispute_reason?: string | null
          invited_at?: string
          expires_at?: string
          responded_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "registry_collaborators_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
```

- [ ] **Step 2: Create useRegistry hook**

Create `src/hooks/useRegistry.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

// --- Types ---

export interface Work {
  id: string;
  user_id: string;
  artist_id: string;
  project_id: string;
  title: string;
  work_type: string;
  isrc: string | null;
  iswc: string | null;
  upc: string | null;
  release_date: string | null;
  status: string;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface OwnershipStake {
  id: string;
  work_id: string;
  user_id: string;
  stake_type: string;
  holder_name: string;
  holder_role: string;
  percentage: number;
  holder_email: string | null;
  holder_ipi: string | null;
  publisher_or_label: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface LicensingRight {
  id: string;
  work_id: string;
  user_id: string;
  license_type: string;
  licensee_name: string;
  licensee_email: string | null;
  territory: string;
  start_date: string;
  end_date: string | null;
  terms: string | null;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Agreement {
  id: string;
  work_id: string;
  user_id: string;
  agreement_type: string;
  title: string;
  description: string | null;
  effective_date: string;
  parties: Array<{ name: string; role: string; email?: string }>;
  file_id: string | null;
  document_hash: string | null;
  created_at: string;
}

export interface Collaborator {
  id: string;
  work_id: string;
  stake_id: string | null;
  invited_by: string;
  collaborator_user_id: string | null;
  email: string;
  name: string;
  role: string;
  status: string;
  invite_token: string;
  dispute_reason: string | null;
  invited_at: string;
  expires_at: string;
  responded_at: string | null;
}

export interface WorkFull extends Work {
  stakes: OwnershipStake[];
  licenses: LicensingRight[];
  agreements: Agreement[];
  collaborators: Collaborator[];
}

// --- Helper ---

async function apiFetch<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
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
      artist_id: string;
      project_id: string;
      title: string;
      work_type?: string;
      isrc?: string;
      iswc?: string;
      upc?: string;
      release_date?: string;
      notes?: string;
    }) => {
      return apiFetch(`${API_URL}/registry/works?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      toast.success("Work registered");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ workId, ...body }: { workId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/works/${workId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
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
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      toast.success("Work deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Stakes ---

export function useCreateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string;
      stake_type: string;
      holder_name: string;
      holder_role: string;
      percentage: number;
      holder_email?: string;
      holder_ipi?: string;
      publisher_or_label?: string;
      notes?: string;
    }) =>
      apiFetch(`${API_URL}/registry/stakes?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Ownership stake added");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ stakeId, ...body }: { stakeId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Ownership stake updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (stakeId: string) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Ownership stake removed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Licenses ---

export function useCreateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string;
      license_type: string;
      licensee_name: string;
      licensee_email?: string;
      territory?: string;
      start_date: string;
      end_date?: string;
      terms?: string;
    }) =>
      apiFetch(`${API_URL}/registry/licenses?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("License added");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ licenseId, ...body }: { licenseId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("License updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (licenseId: string) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("License removed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Agreements ---

export function useCreateAgreement() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string;
      agreement_type: string;
      title: string;
      description?: string;
      effective_date: string;
      parties: Array<{ name: string; role: string; email?: string }>;
      file_id?: string;
      document_hash?: string;
    }) =>
      apiFetch(`${API_URL}/registry/agreements?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Agreement recorded");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Collaboration ---

export function useInviteCollaborator() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string;
      email: string;
      name: string;
      role: string;
      stake_id?: string;
    }) =>
      apiFetch<Collaborator>(`${API_URL}/registry/collaborators/invite?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Invitation sent");
    },
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
          method: "POST",
          headers: { "Content-Type": "application/json" },
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

// --- Export ---

export function useExportProof() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (workId: string) => {
      const res = await fetch(
        `${API_URL}/registry/works/${workId}/export?user_id=${user!.id}`
      );
      if (!res.ok) throw new Error("Failed to generate proof of ownership");
      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition") || "";
      const match = disposition.match(/filename="?(.+?)"?$/);
      const filename = match ? match[1] : "Proof_of_Ownership.pdf";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    onSuccess: () => toast.success("Proof of ownership downloaded"),
    onError: (e: Error) => toast.error(e.message),
  });
}
```

- [ ] **Step 3: Add routes to App.tsx**

In `src/App.tsx`, add imports after the existing page imports:

```typescript
import Registry from "./pages/Registry";
import WorkDetail from "./pages/WorkDetail";
import InviteClaim from "./pages/InviteClaim";
```

Add 3 routes inside `<Routes>`, before the `NotFound` catch-all:

```tsx
            <Route
              path="/tools/registry"
              element={
                <ProtectedRoute>
                  <Registry />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools/registry/:workId"
              element={
                <ProtectedRoute>
                  <WorkDetail />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools/registry/invite/:token"
              element={
                <ProtectedRoute>
                  <InviteClaim />
                </ProtectedRoute>
              }
            />
```

**Important:** The `/tools/registry/invite/:token` route must appear BEFORE `/tools/registry/:workId` so React Router matches it first. Otherwise `invite` is captured as a `workId`.

- [ ] **Step 4: Add Registry card to Tools page**

In `src/pages/Tools.tsx`, add `Shield` to the lucide-react import:

```typescript
import { Music, Calculator, ArrowRight, ArrowLeft, Bot, FileText, Shield } from "lucide-react";
```

Add the Registry card after the Split Sheet card (inside the grid div):

```tsx
          {/* Rights Registry Tool Card */}
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
                Launch Tool →
              </Button>
            </CardContent>
          </Card>
```

- [ ] **Step 5: Verify and commit**

Run: `npm run build`
Expected: Build succeeds (Registry, WorkDetail, InviteClaim pages are stubs at this point — they'll be built in Tasks 7-9)

```bash
git add src/integrations/supabase/types.ts src/hooks/useRegistry.ts src/App.tsx src/pages/Tools.tsx
git commit -m "feat: add registry types, hooks with collaboration, routing, and tool card"
```

---

### Task 7: Frontend Registry List Page with Pending Reviews

**Goal:** Registry page with "My Works" listing AND "Pending Your Review" section for collaborator works.

**Files:**
- Create: `src/pages/Registry.tsx`

**Acceptance Criteria:**
- [ ] "My Works" section — works the user created, filterable by artist
- [ ] "Pending Your Review" section — works where user is a collaborator with `pending_approval` status
- [ ] "Shared Works" section — ALL works where user is a collaborator (any status), grouped by artist
- [ ] Filter by artist, search by title
- [ ] Status badges: Draft (yellow), Pending Approval (amber), Registered (green), Disputed (red)
- [ ] "Register Work" creation dialog — requires selecting artist AND project (project is required)
- [ ] Click navigates to `/tools/registry/:workId`

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create Registry.tsx**

Create `src/pages/Registry.tsx`:

```tsx
import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useWorks, useMyCollaborations, useCreateWork, type Work } from "@/hooks/useRegistry";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Music, ArrowLeft, Plus, Search, Shield, Loader2, FileText, AlertCircle, Users,
} from "lucide-react";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const WORK_TYPES = [
  { value: "single", label: "Single" },
  { value: "ep_track", label: "EP Track" },
  { value: "album_track", label: "Album Track" },
  { value: "composition", label: "Composition" },
];

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-yellow-100 text-yellow-800",
  pending_approval: "bg-amber-100 text-amber-800",
  registered: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
};

interface Artist { id: string; name: string; }
interface Project { id: string; name: string; artist_id: string; }

const Registry = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedArtistId, setSelectedArtistId] = useState<string>("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Create form state
  const [newTitle, setNewTitle] = useState("");
  const [newArtistId, setNewArtistId] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  const [newWorkType, setNewWorkType] = useState("single");
  const [newIsrc, setNewIsrc] = useState("");
  const [newIswc, setNewIswc] = useState("");

  // Fetch artists
  const artistsQuery = useQuery<Artist[]>({
    queryKey: ["registry-artists", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(`${API_URL}/artists?user_id=${user.id}`);
      if (!res.ok) return [];
      const data = await res.json();
      return (Array.isArray(data) ? data : data.artists || []).map(
        (a: Record<string, unknown>) => ({ id: a.id as string, name: a.name as string })
      );
    },
    enabled: !!user?.id,
  });

  // Fetch projects for selected artist in create form
  const projectsQuery = useQuery<Project[]>({
    queryKey: ["registry-projects", newArtistId],
    queryFn: async () => {
      if (!newArtistId) return [];
      const res = await fetch(`${API_URL}/projects/${newArtistId}`);
      if (!res.ok) return [];
      const data = await res.json();
      return (Array.isArray(data) ? data : data.projects || []).map(
        (p: Record<string, unknown>) => ({
          id: p.id as string, name: p.name as string, artist_id: p.artist_id as string,
        })
      );
    },
    enabled: !!newArtistId,
  });

  const artistFilter = selectedArtistId === "all" ? undefined : selectedArtistId;
  const worksQuery = useWorks(artistFilter);
  const collabQuery = useMyCollaborations();
  const createWork = useCreateWork();

  const filteredWorks = useMemo(() => {
    const works = worksQuery.data || [];
    if (!searchQuery.trim()) return works;
    const q = searchQuery.toLowerCase();
    return works.filter(
      (w) =>
        w.title.toLowerCase().includes(q) ||
        (w.isrc && w.isrc.toLowerCase().includes(q)) ||
        (w.iswc && w.iswc.toLowerCase().includes(q))
    );
  }, [worksQuery.data, searchQuery]);

  const handleCreate = async () => {
    if (!newTitle.trim() || !newArtistId || !newProjectId) return;
    await createWork.mutateAsync({
      artist_id: newArtistId,
      project_id: newProjectId,
      title: newTitle.trim(),
      work_type: newWorkType,
      isrc: newIsrc.trim() || undefined,
      iswc: newIswc.trim() || undefined,
    });
    setShowCreateDialog(false);
    setNewTitle(""); setNewArtistId(""); setNewProjectId("");
    setNewWorkType("single"); setNewIsrc(""); setNewIswc("");
  };

  const getArtistName = (artistId: string) =>
    (artistsQuery.data || []).find((a) => a.id === artistId)?.name || "Unknown";

  const myCollabWorks = collabQuery.data || [];
  // Split into pending (need action) vs confirmed (already responded)
  const pendingWorks = myCollabWorks.filter((w) => w.status === "pending_approval");
  const allCollabWorks = myCollabWorks;

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
          <Button variant="outline" onClick={() => navigate("/tools")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Tools
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-foreground flex items-center gap-3">
              <Shield className="w-8 h-8 text-primary" /> Rights Registry
            </h2>
            <p className="text-muted-foreground mt-1">
              Track ownership, splits, licensing, and generate proof-of-ownership documents
            </p>
          </div>
          <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
            <DialogTrigger asChild>
              <Button><Plus className="w-4 h-4 mr-2" /> Register Work</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Register a New Work</DialogTitle></DialogHeader>
              <div className="space-y-4 pt-2">
                <div>
                  <label className="text-sm font-medium">Title *</label>
                  <Input value={newTitle} onChange={(e) => setNewTitle(e.target.value)}
                    placeholder="Song or composition title" />
                </div>
                <div>
                  <label className="text-sm font-medium">Artist *</label>
                  <Select value={newArtistId} onValueChange={(v) => { setNewArtistId(v); setNewProjectId(""); }}>
                    <SelectTrigger><SelectValue placeholder="Select artist" /></SelectTrigger>
                    <SelectContent>
                      {(artistsQuery.data || []).map((a) => (
                        <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium">Project *</label>
                  <Select value={newProjectId} onValueChange={setNewProjectId} disabled={!newArtistId}>
                    <SelectTrigger><SelectValue placeholder={newArtistId ? "Select project" : "Select artist first"} /></SelectTrigger>
                    <SelectContent>
                      {(projectsQuery.data || []).map((p) => (
                        <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium">Work Type</label>
                  <Select value={newWorkType} onValueChange={setNewWorkType}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {WORK_TYPES.map((t) => (
                        <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">ISRC</label>
                    <Input value={newIsrc} onChange={(e) => setNewIsrc(e.target.value)}
                      placeholder="e.g. USRC17607839" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">ISWC</label>
                    <Input value={newIswc} onChange={(e) => setNewIswc(e.target.value)}
                      placeholder="e.g. T-345246800-1" />
                  </div>
                </div>
                <Button onClick={handleCreate}
                  disabled={!newTitle.trim() || !newArtistId || !newProjectId || createWork.isPending}
                  className="w-full">
                  {createWork.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Plus className="w-4 h-4 mr-2" />}
                  Register Work
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Pending Your Review — works needing action */}
        {pendingWorks.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-amber-500" /> Pending Your Review
            </h3>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {pendingWorks.map((work) => (
                <Card key={work.id}
                  className="min-w-[250px] hover:border-primary/50 transition-colors cursor-pointer border-amber-200 bg-amber-50/30"
                  onClick={() => navigate(`/tools/registry/${work.id}`)}>
                  <CardContent className="p-4">
                    <div className="font-medium text-sm">{work.title}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {work.work_type.replace("_", " ").toUpperCase()}
                    </div>
                    <Badge className="mt-2 bg-amber-100 text-amber-800">Needs Your Confirmation</Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* My Works as Collaborator — all works user is involved in across all artists/projects */}
        {allCollabWorks.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
              <Users className="w-5 h-5 text-primary" /> Shared Works
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {allCollabWorks.map((work) => (
                <Card key={work.id}
                  className="hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => navigate(`/tools/registry/${work.id}`)}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium text-sm">{work.title}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {getArtistName(work.artist_id)} · {work.work_type.replace("_", " ").toUpperCase()}
                        </div>
                      </div>
                      <Badge className={STATUS_COLORS[work.status] || "bg-gray-100 text-gray-800"}>
                        {work.status.replace("_", " ")}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex gap-3 mb-6">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by title, ISRC, or ISWC..." className="pl-9" />
          </div>
          <Select value={selectedArtistId} onValueChange={setSelectedArtistId}>
            <SelectTrigger className="w-48"><SelectValue placeholder="All Artists" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Artists</SelectItem>
              {(artistsQuery.data || []).map((a) => (
                <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Works Grid */}
        {worksQuery.isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : filteredWorks.length === 0 ? (
          <div className="text-center py-20">
            <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-semibold text-foreground mb-2">No works registered</h3>
            <p className="text-muted-foreground mb-4">
              Register your first work to start tracking ownership and rights
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="w-4 h-4 mr-2" /> Register Work
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredWorks.map((work) => (
              <Card key={work.id}
                className="hover:border-primary/50 transition-colors cursor-pointer group"
                onClick={() => navigate(`/tools/registry/${work.id}`)}>
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-base leading-tight">{work.title}</CardTitle>
                    <Badge className={STATUS_COLORS[work.status] || "bg-gray-100 text-gray-800"}>
                      {work.status.replace("_", " ")}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{getArtistName(work.artist_id)}</p>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>{work.work_type.replace("_", " ").toUpperCase()}</span>
                    {work.isrc && <span>ISRC: {work.isrc}</span>}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default Registry;
```

- [ ] **Step 2: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/pages/Registry.tsx
git commit -m "feat: add Registry list page with pending reviews and project-required creation"
```

---

### Task 8: Frontend Work Detail with Approval Workflow

**Goal:** Work detail page with ownership/licensing/agreements tabs plus collaboration status and approval actions.

**Files:**
- Create: `src/pages/WorkDetail.tsx`
- Create: `src/components/registry/OwnershipPanel.tsx`
- Create: `src/components/registry/LicensingPanel.tsx`
- Create: `src/components/registry/AgreementsPanel.tsx`
- Create: `src/components/registry/CollaborationStatus.tsx`

**Acceptance Criteria:**
- [ ] Two view modes: **owner** (full CRUD + submit + invite) vs **collaborator** (read-only + confirm/dispute)
- [ ] `CollaborationStatus` shows progress bar and per-collaborator status
- [ ] `OwnershipPanel` shows approval badge next to each stakeholder
- [ ] "Submit for Approval" only enabled in `draft`/`disputed` status with ≥1 collaborator
- [ ] Dispute requires a reason via dialog
- [ ] Auto-status updates via backend, frontend refetches on mutation success

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create CollaborationStatus.tsx**

Create `src/components/registry/CollaborationStatus.tsx`:

```tsx
import { type Collaborator, useSubmitForApproval } from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Send, Loader2, CheckCircle, AlertTriangle, Clock } from "lucide-react";

const COLLAB_STATUS_ICON: Record<string, typeof CheckCircle> = {
  confirmed: CheckCircle,
  disputed: AlertTriangle,
  invited: Clock,
};

const COLLAB_STATUS_COLOR: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
  invited: "bg-amber-100 text-amber-800",
};

interface Props {
  workId: string;
  workStatus: string;
  collaborators: Collaborator[];
  isOwner: boolean;
}

export default function CollaborationStatus({ workId, workStatus, collaborators, isOwner }: Props) {
  const submitForApproval = useSubmitForApproval();

  const confirmed = collaborators.filter((c) => c.status === "confirmed").length;
  const total = collaborators.length;
  const pct = total > 0 ? (confirmed / total) * 100 : 0;

  const canSubmit = isOwner && (workStatus === "draft" || workStatus === "disputed") && total > 0;

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Collaboration Status</CardTitle>
          {canSubmit && (
            <Button size="sm" onClick={() => submitForApproval.mutate(workId)}
              disabled={submitForApproval.isPending}>
              {submitForApproval.isPending ? (
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              ) : (
                <Send className="w-3 h-3 mr-1" />
              )}
              Submit for Approval
            </Button>
          )}
        </div>
        {total > 0 && (
          <div className="mt-2">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>{confirmed} of {total} confirmed</span>
              <span>{pct.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-green-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {total === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-2">
            No collaborators invited yet. Add ownership stakes and invite stakeholders.
          </p>
        ) : (
          <div className="space-y-2">
            {collaborators.map((c) => {
              const Icon = COLLAB_STATUS_ICON[c.status] || Clock;
              return (
                <div key={c.id} className="flex items-center justify-between p-2 rounded-lg border">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    <div>
                      <span className="text-sm font-medium">{c.name}</span>
                      <span className="text-xs text-muted-foreground ml-2">({c.role})</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{c.email}</span>
                    <Badge className={COLLAB_STATUS_COLOR[c.status] || ""}>{c.status}</Badge>
                  </div>
                </div>
              );
            })}
            {collaborators.some((c) => c.status === "disputed" && c.dispute_reason) && (
              <div className="mt-2 p-3 bg-red-50 rounded-lg border border-red-200">
                <p className="text-sm font-medium text-red-800 mb-1">Dispute Reasons:</p>
                {collaborators.filter((c) => c.status === "disputed" && c.dispute_reason).map((c) => (
                  <p key={c.id} className="text-xs text-red-700">
                    <strong>{c.name}:</strong> {c.dispute_reason}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Create OwnershipPanel.tsx**

Create `src/components/registry/OwnershipPanel.tsx`:

```tsx
import { useState } from "react";
import {
  useCreateStake, useUpdateStake, useDeleteStake,
  type OwnershipStake, type Collaborator,
} from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, Pencil, Trash2, Loader2 } from "lucide-react";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];

const APPROVAL_COLORS: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
  invited: "bg-amber-100 text-amber-800",
};

interface Props {
  workId: string;
  stakes: OwnershipStake[];
  collaborators: Collaborator[];
  isOwner: boolean;
}

function StakeSection({
  label, stakeType, stakes, workId, collaborators, isOwner,
}: {
  label: string; stakeType: string; stakes: OwnershipStake[];
  workId: string; collaborators: Collaborator[]; isOwner: boolean;
}) {
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [pct, setPct] = useState("");
  const [email, setEmail] = useState("");
  const [ipi, setIpi] = useState("");
  const [pubLabel, setPubLabel] = useState("");

  const createStake = useCreateStake();
  const updateStake = useUpdateStake();
  const deleteStake = useDeleteStake();

  const totalPct = stakes.reduce((s, st) => s + st.percentage, 0);
  const barWidth = Math.min(totalPct, 100);

  const resetForm = () => {
    setName(""); setRole(""); setPct(""); setEmail(""); setIpi(""); setPubLabel("");
    setShowAdd(false); setEditingId(null);
  };

  const handleSubmit = async () => {
    const percentage = parseFloat(pct);
    if (!name || !role || isNaN(percentage) || percentage <= 0) return;
    if (editingId) {
      await updateStake.mutateAsync({
        stakeId: editingId, holder_name: name, holder_role: role, percentage,
        holder_email: email || undefined, holder_ipi: ipi || undefined,
        publisher_or_label: pubLabel || undefined,
      });
    } else {
      await createStake.mutateAsync({
        work_id: workId, stake_type: stakeType, holder_name: name,
        holder_role: role, percentage,
        holder_email: email || undefined, holder_ipi: ipi || undefined,
        publisher_or_label: pubLabel || undefined,
      });
    }
    resetForm();
  };

  const startEdit = (stake: OwnershipStake) => {
    setEditingId(stake.id); setName(stake.holder_name); setRole(stake.holder_role);
    setPct(String(stake.percentage)); setEmail(stake.holder_email || "");
    setIpi(stake.holder_ipi || ""); setPubLabel(stake.publisher_or_label || "");
    setShowAdd(true);
  };

  const getApprovalStatus = (stake: OwnershipStake): string | null => {
    const byStake = collaborators.find((c) => c.stake_id === stake.id);
    if (byStake) return byStake.status;
    const byEmail = stake.holder_email
      ? collaborators.find((c) => c.email.toLowerCase() === stake.holder_email!.toLowerCase())
      : null;
    return byEmail?.status || null;
  };

  const isPending = createStake.isPending || updateStake.isPending;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{label} Ownership</CardTitle>
          {isOwner && (
            <Dialog open={showAdd} onOpenChange={(open) => { if (!open) resetForm(); setShowAdd(open); }}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline"><Plus className="w-3 h-3 mr-1" /> Add</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{editingId ? "Edit" : "Add"} {label} Stake</DialogTitle>
                </DialogHeader>
                <div className="space-y-3 pt-2">
                  <div>
                    <label className="text-sm font-medium">Holder Name *</label>
                    <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Role *</label>
                    <Select value={role} onValueChange={setRole}>
                      <SelectTrigger><SelectValue placeholder="Select role" /></SelectTrigger>
                      <SelectContent>
                        {ROLES.map((r) => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Percentage *</label>
                    <Input type="number" min="0.01" max="100" step="0.01" value={pct}
                      onChange={(e) => setPct(e.target.value)} placeholder="e.g. 50.00" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm font-medium">Email</label>
                      <Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">IPI Number</label>
                      <Input value={ipi} onChange={(e) => setIpi(e.target.value)} placeholder="IPI/CAE" />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Publisher / Label</label>
                    <Input value={pubLabel} onChange={(e) => setPubLabel(e.target.value)}
                      placeholder="Publisher or label name" />
                  </div>
                  <Button onClick={handleSubmit} disabled={isPending} className="w-full">
                    {isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {editingId ? "Update" : "Add"} Stake
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          )}
        </div>
        <div className="mt-2">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>{totalPct.toFixed(2)}% allocated</span>
            <span>{(100 - totalPct).toFixed(2)}% unallocated</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all ${totalPct > 100 ? "bg-red-500" : "bg-primary"}`}
              style={{ width: `${barWidth}%` }} />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {stakes.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No {label.toLowerCase()} ownership stakes recorded
          </p>
        ) : (
          <div className="space-y-2">
            {stakes.map((stake) => {
              const approval = getApprovalStatus(stake);
              return (
                <div key={stake.id} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{stake.holder_name}</span>
                      <span className="text-xs text-muted-foreground">({stake.holder_role})</span>
                      {approval && (
                        <Badge className={APPROVAL_COLORS[approval] || "bg-gray-100 text-gray-800"}>
                          {approval}
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {stake.publisher_or_label && <span>{stake.publisher_or_label} · </span>}
                      {stake.holder_ipi && <span>IPI: {stake.holder_ipi}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">{stake.percentage.toFixed(2)}%</span>
                    {isOwner && (
                      <>
                        <Button size="icon" variant="ghost" className="h-7 w-7"
                          onClick={() => startEdit(stake)}>
                          <Pencil className="w-3 h-3" />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive"
                          onClick={() => deleteStake.mutate(stake.id)}>
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function OwnershipPanel({ workId, stakes, collaborators, isOwner }: Props) {
  const masterStakes = stakes.filter((s) => s.stake_type === "master");
  const pubStakes = stakes.filter((s) => s.stake_type === "publishing");
  return (
    <div className="space-y-6">
      <StakeSection label="Master" stakeType="master" stakes={masterStakes}
        workId={workId} collaborators={collaborators} isOwner={isOwner} />
      <StakeSection label="Publishing" stakeType="publishing" stakes={pubStakes}
        workId={workId} collaborators={collaborators} isOwner={isOwner} />
    </div>
  );
}
```

- [ ] **Step 3: Create LicensingPanel.tsx**

Create `src/components/registry/LicensingPanel.tsx` — identical to the code in the first plan revision's Task 7 Step 2 (the complete LicensingPanel with add/edit/delete dialog, license type selector, territory, date range, status badges). Add an `isOwner` prop: when false, hide the Add/Edit/Delete buttons. The component signature is:

```tsx
interface Props {
  workId: string;
  licenses: LicensingRight[];
  isOwner: boolean;
}
```

Wrap all CRUD buttons in `{isOwner && (...)}` guards.

- [ ] **Step 4: Create AgreementsPanel.tsx**

Create `src/components/registry/AgreementsPanel.tsx` — identical to the code in the first plan revision's Task 7 Step 3 (the complete AgreementsPanel with timeline, party inputs, agreement type selector). Add an `isOwner` prop: when false, hide the "Record Agreement" button. The component signature is:

```tsx
interface Props {
  workId: string;
  agreements: Agreement[];
  isOwner: boolean;
}
```

Wrap the "Record Agreement" dialog trigger in `{isOwner && (...)}`.

- [ ] **Step 5: Create WorkDetail.tsx**

Create `src/pages/WorkDetail.tsx`:

```tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  useWorkFull, useUpdateWork, useDeleteWork,
  useConfirmStake, useDisputeStake,
  type Collaborator,
} from "@/hooks/useRegistry";
import CollaborationStatus from "@/components/registry/CollaborationStatus";
import OwnershipPanel from "@/components/registry/OwnershipPanel";
import LicensingPanel from "@/components/registry/LicensingPanel";
import AgreementsPanel from "@/components/registry/AgreementsPanel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Music, ArrowLeft, Loader2, Shield, Pencil, Trash2,
  Users, Scale, FileCheck, CheckCircle, XCircle,
} from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-yellow-100 text-yellow-800",
  pending_approval: "bg-amber-100 text-amber-800",
  registered: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
};

const WorkDetail = () => {
  const { workId } = useParams<{ workId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const workQuery = useWorkFull(workId);
  const updateWork = useUpdateWork();
  const deleteWork = useDeleteWork();
  const confirmStake = useConfirmStake();
  const disputeStake = useDisputeStake();

  const [showEdit, setShowEdit] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editIsrc, setEditIsrc] = useState("");
  const [editIswc, setEditIswc] = useState("");
  const [editUpc, setEditUpc] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [editReleaseDate, setEditReleaseDate] = useState("");
  const [editNotes, setEditNotes] = useState("");

  const [showDisputeDialog, setShowDisputeDialog] = useState(false);
  const [disputeReason, setDisputeReason] = useState("");
  const [disputeCollabId, setDisputeCollabId] = useState("");

  const work = workQuery.data;
  const isOwner = work?.user_id === user?.id;

  // Find this user's collaborator record (if they're a collaborator)
  const myCollab: Collaborator | undefined = work?.collaborators?.find(
    (c) => c.collaborator_user_id === user?.id
  );

  const openEdit = () => {
    if (!work) return;
    setEditTitle(work.title); setEditIsrc(work.isrc || ""); setEditIswc(work.iswc || "");
    setEditUpc(work.upc || ""); setEditStatus(work.status);
    setEditReleaseDate(work.release_date || ""); setEditNotes(work.notes || "");
    setShowEdit(true);
  };

  const handleUpdate = async () => {
    if (!workId || !editTitle.trim()) return;
    await updateWork.mutateAsync({
      workId, title: editTitle.trim(),
      isrc: editIsrc.trim() || undefined, iswc: editIswc.trim() || undefined,
      upc: editUpc.trim() || undefined, status: editStatus,
      release_date: editReleaseDate || undefined, notes: editNotes.trim() || undefined,
    });
    setShowEdit(false);
  };

  const handleDelete = async () => {
    if (!workId) return;
    if (!window.confirm("Delete this work and all its ownership, licensing, and agreement records?")) return;
    await deleteWork.mutateAsync(workId);
    navigate("/tools/registry");
  };

  const handleConfirm = () => {
    if (myCollab) confirmStake.mutate(myCollab.id);
  };

  const openDispute = () => {
    if (myCollab) {
      setDisputeCollabId(myCollab.id);
      setDisputeReason("");
      setShowDisputeDialog(true);
    }
  };

  const handleDispute = async () => {
    if (!disputeCollabId || !disputeReason.trim()) return;
    await disputeStake.mutateAsync({ collaboratorId: disputeCollabId, reason: disputeReason.trim() });
    setShowDisputeDialog(false);
  };

  if (workQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!work) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Work not found</p>
        <Button variant="outline" onClick={() => navigate("/tools/registry")}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Registry
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
          <Button variant="outline" onClick={() => navigate("/tools/registry")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Registry
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Collaborator action banner */}
        {myCollab && !isOwner && myCollab.status === "invited" && (
          <div className="mb-6 p-4 rounded-lg border-2 border-amber-300 bg-amber-50">
            <p className="text-sm font-medium text-amber-900 mb-3">
              You've been listed as <strong>{myCollab.role}</strong> on this work.
              Please review the ownership details and confirm or dispute your stake.
            </p>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleConfirm} disabled={confirmStake.isPending}>
                {confirmStake.isPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <CheckCircle className="w-3 h-3 mr-1" />}
                Confirm My Stake
              </Button>
              <Button size="sm" variant="destructive" onClick={openDispute}>
                <XCircle className="w-3 h-3 mr-1" /> Dispute
              </Button>
            </div>
          </div>
        )}

        {/* Work Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Shield className="w-6 h-6 text-primary" />
              <h2 className="text-2xl font-bold text-foreground">{work.title}</h2>
              <Badge className={STATUS_COLORS[work.status] || ""}>{work.status.replace("_", " ")}</Badge>
            </div>
            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
              <span>{work.work_type.replace("_", " ").toUpperCase()}</span>
              {work.isrc && <span>ISRC: {work.isrc}</span>}
              {work.iswc && <span>ISWC: {work.iswc}</span>}
              {work.upc && <span>UPC: {work.upc}</span>}
              {work.release_date && <span>Released: {work.release_date}</span>}
            </div>
          </div>
          {isOwner && (
            <div className="flex items-center gap-2">
              <Dialog open={showEdit} onOpenChange={setShowEdit}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" onClick={openEdit}>
                    <Pencil className="w-4 h-4 mr-1" /> Edit
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader><DialogTitle>Edit Work</DialogTitle></DialogHeader>
                  <div className="space-y-3 pt-2">
                    <div>
                      <label className="text-sm font-medium">Title *</label>
                      <Input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Status</label>
                      <Select value={editStatus} onValueChange={setEditStatus}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="draft">Draft</SelectItem>
                          <SelectItem value="registered">Registered</SelectItem>
                          <SelectItem value="disputed">Disputed</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div><label className="text-sm font-medium">ISRC</label>
                        <Input value={editIsrc} onChange={(e) => setEditIsrc(e.target.value)} /></div>
                      <div><label className="text-sm font-medium">ISWC</label>
                        <Input value={editIswc} onChange={(e) => setEditIswc(e.target.value)} /></div>
                      <div><label className="text-sm font-medium">UPC</label>
                        <Input value={editUpc} onChange={(e) => setEditUpc(e.target.value)} /></div>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Release Date</label>
                      <Input type="date" value={editReleaseDate}
                        onChange={(e) => setEditReleaseDate(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Notes</label>
                      <Input value={editNotes} onChange={(e) => setEditNotes(e.target.value)} />
                    </div>
                    <Button onClick={handleUpdate} disabled={updateWork.isPending} className="w-full">
                      {updateWork.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      Save Changes
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
              <Button variant="outline" size="sm" className="text-destructive" onClick={handleDelete}>
                <Trash2 className="w-4 h-4 mr-1" /> Delete
              </Button>
            </div>
          )}
        </div>

        {/* Collaboration Status */}
        <CollaborationStatus workId={work.id} workStatus={work.status}
          collaborators={work.collaborators || []} isOwner={isOwner} />

        {/* Tabs */}
        <Tabs defaultValue="ownership">
          <TabsList className="mb-6">
            <TabsTrigger value="ownership" className="gap-1.5">
              <Users className="w-4 h-4" /> Ownership
            </TabsTrigger>
            <TabsTrigger value="licensing" className="gap-1.5">
              <Scale className="w-4 h-4" /> Licensing
            </TabsTrigger>
            <TabsTrigger value="agreements" className="gap-1.5">
              <FileCheck className="w-4 h-4" /> Agreements
            </TabsTrigger>
          </TabsList>
          <TabsContent value="ownership">
            <OwnershipPanel workId={work.id} stakes={work.stakes || []}
              collaborators={work.collaborators || []} isOwner={isOwner} />
          </TabsContent>
          <TabsContent value="licensing">
            <LicensingPanel workId={work.id} licenses={work.licenses || []} isOwner={isOwner} />
          </TabsContent>
          <TabsContent value="agreements">
            <AgreementsPanel workId={work.id} agreements={work.agreements || []} isOwner={isOwner} />
          </TabsContent>
        </Tabs>

        {/* Dispute Dialog */}
        <Dialog open={showDisputeDialog} onOpenChange={setShowDisputeDialog}>
          <DialogContent>
            <DialogHeader><DialogTitle>Dispute Your Stake</DialogTitle></DialogHeader>
            <div className="space-y-3 pt-2">
              <p className="text-sm text-muted-foreground">
                Please explain why you're disputing this ownership stake.
              </p>
              <Textarea value={disputeReason} onChange={(e) => setDisputeReason(e.target.value)}
                placeholder="Reason for dispute..." rows={3} />
              <Button onClick={handleDispute} variant="destructive"
                disabled={!disputeReason.trim() || disputeStake.isPending} className="w-full">
                {disputeStake.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                Submit Dispute
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  );
};

export default WorkDetail;
```

- [ ] **Step 6: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/pages/WorkDetail.tsx src/components/registry/
git commit -m "feat: add Work Detail page with collaboration, approval workflow, and ownership panels"
```

---

### Task 9: Frontend Invitation Flow & Claim Handler

**Goal:** Invite collaborators by email and handle invitation claim links.

**Files:**
- Create: `src/pages/InviteClaim.tsx`
- Create: `src/components/registry/InviteCollaboratorModal.tsx`

**Acceptance Criteria:**
- [ ] `InviteCollaboratorModal`: form with email, name, role fields + optional stake_id selector
- [ ] Calls `useInviteCollaborator` mutation on submit
- [ ] `/tools/registry/invite/:token` claims invitation and redirects to work detail
- [ ] If not logged in, redirects to auth then back to claim
- [ ] Success toast on claim

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create InviteCollaboratorModal.tsx**

Create `src/components/registry/InviteCollaboratorModal.tsx`:

```tsx
import { useState } from "react";
import { useInviteCollaborator, type OwnershipStake } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Loader2, Send } from "lucide-react";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];

interface Props {
  workId: string;
  stakes: OwnershipStake[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function InviteCollaboratorModal({ workId, stakes, open, onOpenChange }: Props) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [stakeId, setStakeId] = useState<string>("");

  const invite = useInviteCollaborator();

  const resetForm = () => {
    setEmail(""); setName(""); setRole(""); setStakeId("");
  };

  const handleSubmit = async () => {
    if (!email.trim() || !name.trim() || !role) return;
    await invite.mutateAsync({
      work_id: workId,
      email: email.trim(),
      name: name.trim(),
      role,
      stake_id: stakeId || undefined,
    });
    resetForm();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) resetForm(); onOpenChange(o); }}>
      <DialogContent>
        <DialogHeader><DialogTitle>Invite Collaborator</DialogTitle></DialogHeader>
        <div className="space-y-3 pt-2">
          <div>
            <label className="text-sm font-medium">Email *</label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="collaborator@example.com" />
          </div>
          <div>
            <label className="text-sm font-medium">Name *</label>
            <Input value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Full name" />
          </div>
          <div>
            <label className="text-sm font-medium">Role *</label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger><SelectValue placeholder="Select role" /></SelectTrigger>
              <SelectContent>
                {ROLES.map((r) => <SelectItem key={r} value={r}>{r}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {stakes.length > 0 && (
            <div>
              <label className="text-sm font-medium">Link to Stake (optional)</label>
              <Select value={stakeId} onValueChange={setStakeId}>
                <SelectTrigger><SelectValue placeholder="Select a stake to link" /></SelectTrigger>
                <SelectContent>
                  {stakes.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.holder_name} — {s.stake_type} {s.percentage}%
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <Button onClick={handleSubmit} disabled={invite.isPending || !email.trim() || !name.trim() || !role}
            className="w-full">
            {invite.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Send className="w-4 h-4 mr-2" />}
            Send Invitation
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 2: Create InviteClaim.tsx**

Create `src/pages/InviteClaim.tsx`:

```tsx
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useClaimInvitation } from "@/hooks/useRegistry";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

const InviteClaim = () => {
  const { token } = useParams<{ token: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const claimInvitation = useClaimInvitation();
  const [claimed, setClaimed] = useState(false);

  useEffect(() => {
    if (!token) return;

    if (!user) {
      navigate(`/auth?redirect=/tools/registry/invite/${token}`);
      return;
    }

    if (!claimed) {
      setClaimed(true);
      claimInvitation.mutate(token, {
        onSuccess: (data) => {
          toast.success("Invitation claimed — review your stake");
          navigate(`/tools/registry/${data.work_id}`);
        },
        onError: () => {
          toast.error("Invalid or expired invitation");
          navigate("/tools/registry");
        },
      });
    }
  }, [token, user, claimed, claimInvitation, navigate]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Claiming your invitation...</p>
      </div>
    </div>
  );
};

export default InviteClaim;
```

Note: If `Auth.tsx` doesn't support the `redirect` query param, add this after successful login in `Auth.tsx`:
```typescript
const searchParams = new URLSearchParams(window.location.search);
const redirect = searchParams.get("redirect");
if (redirect) { navigate(redirect); } else { navigate("/dashboard"); }
```

- [ ] **Step 3: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/pages/InviteClaim.tsx src/components/registry/InviteCollaboratorModal.tsx
git commit -m "feat: add invitation modal and claim handler with auth redirect"
```

---

### Task 10: Frontend Proof of Ownership Export

**Goal:** Export button on work detail page that downloads proof-of-ownership PDF with approval status.

**Files:**
- Create: `src/components/registry/ProofOfOwnership.tsx`
- Modify: `src/pages/WorkDetail.tsx`

**Acceptance Criteria:**
- [ ] "Export Proof" button in work detail header
- [ ] Downloads PDF showing ownership + approval status per stakeholder
- [ ] Loading spinner during PDF generation
- [ ] Success toast on download

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create ProofOfOwnership.tsx**

Create `src/components/registry/ProofOfOwnership.tsx`:

```tsx
import { useExportProof } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Download, Loader2 } from "lucide-react";

interface Props {
  workId: string;
}

export default function ProofOfOwnership({ workId }: Props) {
  const exportProof = useExportProof();
  return (
    <Button
      variant="default" size="sm"
      onClick={() => exportProof.mutate(workId)}
      disabled={exportProof.isPending}
    >
      {exportProof.isPending ? (
        <Loader2 className="w-4 h-4 mr-1 animate-spin" />
      ) : (
        <Download className="w-4 h-4 mr-1" />
      )}
      Export Proof
    </Button>
  );
}
```

- [ ] **Step 2: Add to WorkDetail header**

In `src/pages/WorkDetail.tsx`, add the import at the top:

```typescript
import ProofOfOwnership from "@/components/registry/ProofOfOwnership";
```

In the header actions area, add `ProofOfOwnership` before the Edit/Delete buttons. Update the owner actions block to:

```tsx
          {isOwner && (
            <div className="flex items-center gap-2">
              <ProofOfOwnership workId={work.id} />
              {/* existing Edit and Delete buttons unchanged */}
```

Also add an `InviteCollaboratorModal` trigger button in the header and the modal itself. Add imports:

```typescript
import InviteCollaboratorModal from "@/components/registry/InviteCollaboratorModal";
import { UserPlus } from "lucide-react";
```

Add state:
```typescript
const [showInvite, setShowInvite] = useState(false);
```

Add the button and modal in the owner actions area:
```tsx
              <Button variant="outline" size="sm" onClick={() => setShowInvite(true)}>
                <UserPlus className="w-4 h-4 mr-1" /> Invite
              </Button>
              <InviteCollaboratorModal workId={work.id} stakes={work.stakes || []}
                open={showInvite} onOpenChange={setShowInvite} />
```

- [ ] **Step 3: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/components/registry/ProofOfOwnership.tsx src/pages/WorkDetail.tsx
git commit -m "feat: add proof-of-ownership export and invite button to work detail"
```

---

### Task 11: Workspace Notifications Integration

**Goal:** Wire registry notifications into the Workspace tool's Notifications tab, replacing the placeholder.

**Files:**
- Create: `src/hooks/useRegistryNotifications.ts`
- Create: `src/components/workspace/RegistryNotifications.tsx`
- Modify: `src/pages/Workspace.tsx:167-172`

**Acceptance Criteria:**
- [ ] `useRegistryNotifications` hook with query for notifications and mutations for mark-read
- [ ] Notifications tab shows collaboration events: invitations, confirmations, disputes, status changes
- [ ] Each notification links to the relevant work detail page
- [ ] "Mark all as read" button
- [ ] Unread count badge on the Notifications tab trigger
- [ ] Empty state when no notifications

**Verify:** `npm run build`

**Steps:**

- [ ] **Step 1: Create useRegistryNotifications hook**

Create `src/hooks/useRegistryNotifications.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

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
    refetchInterval: 30000, // poll every 30s for new notifications
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
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-notifications"] });
    },
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
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-notifications"] });
    },
  });
}
```

- [ ] **Step 2: Create RegistryNotifications component**

Create `src/components/workspace/RegistryNotifications.tsx`:

```tsx
import { useNavigate } from "react-router-dom";
import {
  useRegistryNotifications,
  useMarkNotificationRead,
  useMarkAllRead,
  type RegistryNotification,
} from "@/hooks/useRegistryNotifications";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, CheckCheck, Shield, ExternalLink } from "lucide-react";

const TYPE_COLORS: Record<string, string> = {
  invitation: "bg-blue-100 text-blue-800",
  confirmation: "bg-green-100 text-green-800",
  dispute: "bg-red-100 text-red-800",
  status_change: "bg-purple-100 text-purple-800",
};

export function RegistryNotifications() {
  const navigate = useNavigate();
  const { data: notifications, isLoading } = useRegistryNotifications();
  const markRead = useMarkNotificationRead();
  const markAllRead = useMarkAllRead();

  const handleClick = (n: RegistryNotification) => {
    if (!n.read) markRead.mutate(n.id);
    if (n.work_id) navigate(`/tools/registry/${n.work_id}`);
  };

  const unreadCount = (notifications || []).filter((n) => !n.read).length;

  if (isLoading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Bell className="w-8 h-8 mx-auto mb-3 animate-pulse" />
        <p>Loading notifications...</p>
      </div>
    );
  }

  if (!notifications?.length) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <h3 className="text-lg font-semibold mb-2">No notifications yet</h3>
        <p>Collaboration updates from the Rights Registry will appear here</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary" />
          Rights Registry Notifications
          {unreadCount > 0 && (
            <Badge variant="destructive" className="ml-1">{unreadCount}</Badge>
          )}
        </h3>
        {unreadCount > 0 && (
          <Button variant="ghost" size="sm" onClick={() => markAllRead.mutate()}>
            <CheckCheck className="w-4 h-4 mr-1" /> Mark all read
          </Button>
        )}
      </div>
      <div className="space-y-2">
        {notifications.map((n) => (
          <div
            key={n.id}
            className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-muted/50 ${
              !n.read ? "bg-primary/5 border-primary/20" : ""
            }`}
            onClick={() => handleClick(n)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {!n.read && <div className="w-2 h-2 rounded-full bg-primary shrink-0" />}
                  <span className="text-sm font-medium">{n.title}</span>
                  <Badge className={TYPE_COLORS[n.type] || "bg-gray-100 text-gray-800"}>
                    {n.type.replace("_", " ")}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-1 ml-4">{n.message}</p>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
                <span>{new Date(n.created_at).toLocaleDateString()}</span>
                {n.work_id && <ExternalLink className="w-3 h-3" />}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Update Workspace.tsx**

In `src/pages/Workspace.tsx`, add the import:

```typescript
import { RegistryNotifications } from "@/components/workspace/RegistryNotifications";
import { useUnreadCount } from "@/hooks/useRegistryNotifications";
```

Inside the component, add the unread count:

```typescript
const unreadNotifications = useUnreadCount();
```

Update the Notifications tab trigger to show the badge (around line 145):

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

Replace the placeholder notifications tab content (around line 167-172) with:

```tsx
          <TabsContent value="notifications">
            <RegistryNotifications />
          </TabsContent>
```

- [ ] **Step 4: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add src/hooks/useRegistryNotifications.ts src/components/workspace/RegistryNotifications.tsx src/pages/Workspace.tsx
git commit -m "feat: add registry notifications to Workspace tool with unread badge"
```

---

### Task 12: Artist Profile Claiming Flow

**Goal:** Make artist profiles claimable by the real person they represent. When an artist's email matches a platform user, the profile becomes theirs to view and edit.

**Files:**
- Create: `supabase/migrations/20260329100000_add_artist_claim.sql` (already created)
- Modify: `src/backend/registry/service.py` (add `claim_artist`, `auto_claim_on_invite`)
- Modify: `src/backend/registry/router.py` (add `/artists/claim` endpoint)
- Modify: `src/integrations/supabase/types.ts` (add `claimed_by`, `claim_status`, `claim_token`, `claimed_at` to artists type)
- Modify: `src/pages/ArtistProfile.tsx` (show claim status, allow claimed user to edit)
- Modify: `src/components/registry/InviteCollaboratorModal.tsx` (select from artist roster)

**Acceptance Criteria:**
- [ ] Migration adds `claimed_by`, `claim_status`, `claim_token`, `claimed_at` to artists
- [ ] RLS: claimed user can view + edit their artist profile, view their projects/files
- [ ] When creating an artist profile with an email that matches a platform user → auto-set `claimed_by`, status = `claimed`
- [ ] When inviting a collaborator on a work, if their email matches an unclaimed artist profile → auto-claim
- [ ] Artist profile page shows claim status badge (Unclaimed / Pending / Claimed)
- [ ] Claimed user can edit their own bio, links, avatar (but NOT delete the profile)
- [ ] Invite modal lets you select from your artist roster (pulls their email) OR enter a new email

**Verify:** `npm run build` + `supabase db push`

**Steps:**

- [ ] **Step 1: Apply migration**

Migration file already created at `supabase/migrations/20260329100000_add_artist_claim.sql`. It adds:
- `claimed_by uuid` — the user who claimed this profile
- `claim_status text` — unclaimed / pending / claimed
- `claim_token uuid` — for email verification link
- `claimed_at timestamptz` — when claimed
- Updated RLS on `artists`, `projects`, `project_files` to include `claimed_by`

Run: `supabase db push`

- [ ] **Step 2: Add claim service functions**

In `src/backend/registry/service.py`, add:

```python
async def auto_claim_artist(db: Client, email: str):
    """If an artist profile has this email and is unclaimed, check if a user with this email exists and claim it."""
    # Find unclaimed artist profiles with this email
    artists = (
        db.table("artists")
        .select("id, email, claim_status")
        .eq("email", email)
        .eq("claim_status", "unclaimed")
        .execute()
    )
    if not artists.data:
        return

    # Check if a platform user exists with this email
    user_id = await check_user_exists(db, email)
    if not user_id:
        # Mark as pending — will be claimed when they sign up
        for artist in artists.data:
            db.table("artists").update({"claim_status": "pending"}).eq("id", artist["id"]).execute()
        return

    # Auto-claim all matching artist profiles
    from datetime import datetime, timezone
    for artist in artists.data:
        db.table("artists").update({
            "claimed_by": user_id,
            "claim_status": "claimed",
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", artist["id"]).execute()


async def claim_artist_by_token(db: Client, claim_token: str, user_id: str):
    """Claim an artist profile via token (from email link)."""
    from datetime import datetime, timezone
    result = (
        db.table("artists")
        .select("*")
        .eq("claim_token", claim_token)
        .single()
        .execute()
    )
    if not result.data:
        return None, "not_found"

    artist = result.data
    if artist.get("claimed_by") and artist["claimed_by"] != user_id:
        return None, "already_claimed"

    updated = (
        db.table("artists")
        .update({
            "claimed_by": user_id,
            "claim_status": "claimed",
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", artist["id"])
        .execute()
    )
    return (updated.data[0] if updated.data else artist), None
```

- [ ] **Step 3: Update invite_collaborator to trigger auto_claim_artist**

In `src/backend/registry/service.py`, at the end of `invite_collaborator()`, after creating the notification, add:

```python
    # Also try to auto-claim any unclaimed artist profile with this email
    await auto_claim_artist(db, data["email"])
```

- [ ] **Step 4: Add claim endpoint to router**

In `src/backend/registry/router.py`, add:

```python
@router.post("/artists/claim")
async def claim_artist(claim_token: str = Query(...), user_id: str = Query(...)):
    artist, error = await service.claim_artist_by_token(_get_supabase(), claim_token, user_id)
    if error == "already_claimed":
        raise HTTPException(status_code=409, detail="Profile already claimed by another user")
    if error:
        raise HTTPException(status_code=404, detail="Artist profile not found")
    return artist
```

- [ ] **Step 5: Update TypeScript types for artists**

In `src/integrations/supabase/types.ts`, update the `artists` table types. Add to Row:

```typescript
          claimed_by: string | null
          claim_status: string
          claim_token: string | null
          claimed_at: string | null
```

Add to Insert (all optional):
```typescript
          claimed_by?: string | null
          claim_status?: string
          claim_token?: string | null
          claimed_at?: string | null
```

Add to Update (all optional):
```typescript
          claimed_by?: string | null
          claim_status?: string
          claim_token?: string | null
          claimed_at?: string | null
```

- [ ] **Step 6: Update InviteCollaboratorModal to select from roster**

In `src/components/registry/InviteCollaboratorModal.tsx`, add an artist roster selector. The modal gets a new optional `artists` prop:

```tsx
interface Props {
  workId: string;
  stakes: OwnershipStake[];
  artists?: Array<{ id: string; name: string; email: string }>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}
```

Add a "Select from roster" section at the top of the form:

```tsx
          {artists && artists.length > 0 && (
            <div>
              <label className="text-sm font-medium">Select from roster</label>
              <Select value="" onValueChange={(artistId) => {
                const a = artists.find((x) => x.id === artistId);
                if (a) { setEmail(a.email); setName(a.name); }
              }}>
                <SelectTrigger><SelectValue placeholder="Pick an artist..." /></SelectTrigger>
                <SelectContent>
                  {artists.map((a) => (
                    <SelectItem key={a.id} value={a.id}>{a.name} ({a.email})</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">Or enter details manually below</p>
            </div>
          )}
```

- [ ] **Step 7: Update ArtistProfile page with claim status**

In `src/pages/ArtistProfile.tsx`, add a claim status badge in the header area. Show different UI based on who's viewing:

```tsx
// Determine if current user is the claimed artist
const isClaimedByMe = artist?.claimed_by === user?.id;
const isManager = artist?.user_id === user?.id;

// Show claim status badge
{artist.claim_status === "claimed" && (
  <Badge className="bg-green-100 text-green-800">Verified</Badge>
)}
{artist.claim_status === "pending" && (
  <Badge className="bg-amber-100 text-amber-800">Pending Verification</Badge>
)}
{artist.claim_status === "unclaimed" && isManager && (
  <Badge className="bg-gray-100 text-gray-800">Unclaimed</Badge>
)}
```

If `isClaimedByMe` is true, the user can edit bio, social links, DSP links, and avatar. If `isManager` is true, they can manage projects and works. Both can view everything.

- [ ] **Step 8: Verify and commit**

Run: `npm run build`
Expected: Build succeeds

```bash
git add supabase/migrations/20260329100000_add_artist_claim.sql src/backend/registry/ src/integrations/supabase/types.ts src/components/registry/InviteCollaboratorModal.tsx src/pages/ArtistProfile.tsx
git commit -m "feat: add claimable artist profiles with auto-claim on invite"
```
