# Supabase Migrations — Portfolio/Registry Redesign

Run these 8 migration files **in order** for the portfolio/registry redesign (branch: `msaniiV2-tasks-licensing-artistmanagement`):

```
supabase/migrations/20260403000001_create_project_members.sql
supabase/migrations/20260403000002_create_pending_project_invites.sql
supabase/migrations/20260403000003_create_work_files.sql
supabase/migrations/20260403000004_create_work_audio_links.sql
supabase/migrations/20260403000005_add_content_hash_to_project_files.sql
supabase/migrations/20260403000006_update_works_registry.sql
supabase/migrations/20260403000007_simplify_collaborator_status.sql
supabase/migrations/20260403000008_update_rls_policies.sql
```

## What each migration does

| # | File | Summary |
|---|---|---|
| 1 | `_create_project_members` | `project_members` table, auto-owner trigger (SECURITY DEFINER), owner protection triggers, partial unique index, RLS policies |
| 2 | `_create_pending_project_invites` | `pending_project_invites` table for inviting non-existing users, auth.users signup trigger to auto-convert |
| 3 | `_create_work_files` | `work_files` join table (work ↔ file), RLS for project members + confirmed collaborators |
| 4 | `_create_work_audio_links` | `work_audio_links` join table (work ↔ audio), same RLS pattern |
| 5 | `_add_content_hash_to_project_files` | Adds `content_hash` TEXT column + partial index for file deduplication |
| 6 | `_update_works_registry` | Adds `custom_work_type` column, adds 'other' to work_type CHECK, **migrates disputed → draft**, removes 'disputed' from status CHECK |
| 7 | `_simplify_collaborator_status` | **Migrates disputed → declined**, removes 'disputed' from status CHECK, drops `dispute_reason` column |
| 8 | `_update_rls_policies` | Projects SELECT for members, case-insensitive email invite visibility, file/audio access for confirmed collaborators, project member cascading access, editor+ write policies, owner/admin delete policy |

## Important notes

- Migrations 6 and 7 include **data migrations** (`UPDATE ... SET status = 'draft' WHERE status = 'disputed'`) that must run BEFORE the CHECK constraint changes. They are written in the correct order within each file.
- Migration 1 creates a SECURITY DEFINER trigger on `projects` — new projects will auto-create an owner entry in `project_members`.
- Migration 2 creates a SECURITY DEFINER trigger on `auth.users` — new signups will auto-convert pending project invites to memberships.
- All existing migrations (pre-20260403) are untouched.

## How to run

```bash
# Option A: Supabase CLI
supabase db push

# Option B: Manual (via Supabase SQL editor or psql)
# Run each file in order, 1 through 8
```
