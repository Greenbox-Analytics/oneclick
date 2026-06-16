-- ============================================================
-- Track Registry
--   - registered_tracks: user-scoped Spotify-sourced track metadata
--   - track_contract_matches: per-work match to a project contract
--   - works_registry: link FK + royalty_source marker
-- Royalty values themselves continue to live in ownership_stakes.
-- ============================================================

-- 1. registered_tracks — user-scoped Spotify metadata, one row per (user, Spotify track id)
create table registered_tracks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  spotify_track_id text not null,
  isrc text,
  title text not null,
  primary_artist_name text not null,
  other_artists jsonb not null default '[]',
  album_name text,
  album_cover_url text,
  release_date date,
  spotify_url text not null,
  contributors jsonb not null default '[]',
  enrichment_source text not null default 'spotify'
    check (enrichment_source in ('spotify', 'spotify+musicbrainz')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, spotify_track_id)
);

create index idx_registered_tracks_user_id
  on registered_tracks(user_id);
create index idx_registered_tracks_isrc
  on registered_tracks(isrc) where isrc is not null;

-- 2. track_contract_matches — confirmed link from a work to a contract file
create table track_contract_matches (
  id uuid primary key default gen_random_uuid(),
  work_id uuid not null references works_registry(id) on delete cascade,
  contract_file_id uuid not null references project_files(id) on delete cascade,
  matched_signal text not null
    check (matched_signal in ('isrc', 'title_exact', 'title_artist', 'artist_only', 'fuzzy', 'manual')),
  confidence text not null
    check (confidence in ('high', 'medium', 'low')),
  confirmed_by uuid references auth.users(id) on delete set null,
  confirmed_at timestamptz,
  created_at timestamptz not null default now(),
  unique (work_id, contract_file_id)
);

create index idx_track_contract_matches_work_id
  on track_contract_matches(work_id);
create index idx_track_contract_matches_contract_file_id
  on track_contract_matches(contract_file_id);

-- 3. works_registry — link to canonical track + provenance for royalty values
alter table works_registry
  add column registered_track_id uuid references registered_tracks(id) on delete set null,
  add column royalty_source text
    check (royalty_source in ('contract', 'manual', 'mixed'));

create index idx_works_registry_registered_track_id
  on works_registry(registered_track_id) where registered_track_id is not null;

-- ============================================================
-- Row Level Security
-- ============================================================

-- registered_tracks: owner-only — mirrors works_registry / ownership_stakes pattern.
alter table registered_tracks enable row level security;

create policy "Owner can read registered tracks"
  on registered_tracks for select
  using (auth.uid() = user_id);

create policy "Owner can insert registered tracks"
  on registered_tracks for insert
  with check (auth.uid() = user_id);

create policy "Owner can update registered tracks"
  on registered_tracks for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Owner can delete registered tracks"
  on registered_tracks for delete
  using (auth.uid() = user_id);

-- track_contract_matches: project members can read; editors+ can write.
-- Confirmed work-only collaborators also see matches for their work.
alter table track_contract_matches enable row level security;

create policy "Members and collaborators can read track matches"
  on track_contract_matches for select
  using (
    work_id in (select id from works_registry where is_project_member(project_id))
    or work_id in (
      select work_id from registry_collaborators
      where collaborator_user_id = auth.uid() and status = 'confirmed'
    )
  );

create policy "Editors can insert track matches"
  on track_contract_matches for insert
  with check (
    work_id in (
      select id from works_registry
      where get_project_role(project_id) in ('owner', 'admin', 'editor')
    )
  );

create policy "Editors can update track matches"
  on track_contract_matches for update
  using (
    work_id in (
      select id from works_registry
      where get_project_role(project_id) in ('owner', 'admin', 'editor')
    )
  );

create policy "Editors can delete track matches"
  on track_contract_matches for delete
  using (
    work_id in (
      select id from works_registry
      where get_project_role(project_id) in ('owner', 'admin', 'editor')
    )
  );

-- ============================================================
-- Updated-at trigger
-- update_updated_at_column() is already defined in an earlier migration.
-- ============================================================
create trigger registered_tracks_updated_at
  before update on registered_tracks
  for each row execute function update_updated_at_column();
