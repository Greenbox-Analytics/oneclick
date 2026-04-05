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
