-- Create user_onboarding table to track per-tool walkthrough completion
-- One row per user, one boolean column per tool
create table user_onboarding (
  user_id uuid references auth.users on delete cascade not null primary key,
  oneclick_completed boolean not null default false,
  zoe_completed boolean not null default false,
  splitsheet_completed boolean not null default false,
  artists_completed boolean not null default false,
  workspace_completed boolean not null default false,
  portfolio_completed boolean not null default false,
  created_at timestamptz not null default now()
);

-- Enable RLS
alter table user_onboarding enable row level security;

-- Users can only read their own row
create policy "Users can read own onboarding status"
  on user_onboarding for select
  using ( auth.uid() = user_id );

-- Users can insert their own row
create policy "Users can insert own onboarding status"
  on user_onboarding for insert
  with check ( auth.uid() = user_id );

-- Users can update their own row
create policy "Users can update own onboarding status"
  on user_onboarding for update
  using ( auth.uid() = user_id )
  with check ( auth.uid() = user_id );

-- Auto-create a user_onboarding row when a new user signs up
create or replace function public.handle_new_user_onboarding()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.user_onboarding (user_id)
  values (new.id);
  return new;
end;
$$;

create trigger on_auth_user_created_onboarding
  after insert on auth.users
  for each row execute procedure public.handle_new_user_onboarding();
