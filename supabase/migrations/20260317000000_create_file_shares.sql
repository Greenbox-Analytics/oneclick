-- File sharing records
create table file_shares (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  contact_id uuid references contacts(id) on delete set null,
  recipient_email text not null,
  recipient_name text,
  file_name text not null,
  file_source text not null,   -- 'project_file' or 'audio_file'
  file_id text not null,
  message text,
  shared_at timestamptz default now() not null,
  link_expires_at timestamptz not null,
  status text default 'sent' not null
);

-- Indexes
create index idx_file_shares_user_id on file_shares(user_id);
create index idx_file_shares_contact_id on file_shares(contact_id);

-- RLS
alter table file_shares enable row level security;

create policy "Users can view their own file shares"
  on file_shares for select
  using (auth.uid() = user_id);

create policy "Users can insert their own file shares"
  on file_shares for insert
  with check (auth.uid() = user_id);
