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
