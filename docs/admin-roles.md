# Admin Roles

Who can use the admin console (`/admin/users`), how grants are managed, and where the gating is enforced.

Admin access lets a user list every account, grant/revoke Pro, apply tier overrides, manage beta tester grants, view per-user usage, and promote/demote other admins. Operationally it's used by the founding team and ops; it is **not** a tier customers can buy.

---

## Two Grant Paths

| Path | Source | Use case | Revoke by |
|------|--------|----------|-----------|
| **Bootstrap (env)** | `ADMIN_EMAILS` env var, comma-separated, case-insensitive | Seed the first admin(s); recovery path if DB is unreachable | Remove from env and redeploy |
| **DB-managed** | `profiles.is_admin = true` | Day-to-day promotions of additional team members | "Demote" button in `/admin/users` (blocked for env admins and self) |

A user is admin if **either** path is true. The check is unified in one helper — see `is_user_admin(...)` below.

**Bootstrapping a fresh deploy:** Set `ADMIN_EMAILS=founder@example.com` (in GSM for prod) before the first admin signs in. Once they're in, they can promote others via the UI; the env var can stay or be pared back to a recovery seed. With no env admins AND no DB admins, calls to `/admin/me` return **500** so operators get a loud signal instead of a confusing 403.

---

## Backend

### Endpoints (`src/backend/subscriptions/admin_router.py`, prefix `/admin`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/me` | Returns `{ email, isAdmin: true }`. Used by `useIsAdmin` to gate the UI |
| GET | `/admin/users` | Paginated user list with `tier`, `has_override`, `is_admin`, `is_env_admin` |
| GET | `/admin/users/{user_id}` | User detail: identity + admin flags + entitlements + raw override row |
| POST | `/admin/users/{user_id}/grant` | Set subscription tier to `pro` (admin-grant, no Stripe) |
| POST | `/admin/users/{user_id}/revoke` | Set subscription tier to `free` |
| POST | `/admin/users/{user_id}/promote` | `profiles.is_admin = true` |
| POST | `/admin/users/{user_id}/demote` | `profiles.is_admin = false` — see safeguards below |
| POST | `/admin/users/{user_id}/override` | Apply a sparse tier override (caps, features, expiry) |
| DELETE | `/admin/users/{user_id}/override` | Clear the override row |
| GET | `/admin/pro-requests` | List Pro-access requests (from `/pro-requests`) |
| GET | `/admin/tester-grants` | List active beta tester grants |
| POST | `/admin/tester-grants` | Grant tester access (full-Pro override with `reason='tester'`) |
| DELETE | `/admin/tester-grants/{user_id}` | Revoke tester grant |

All endpoints depend on `Depends(require_admin)`.

### Demote safeguards

| Case | Response |
|------|----------|
| `user_id` equals caller's `user_id` | `400 "Cannot demote yourself"` |
| Target's email is in `ADMIN_EMAILS` | `400 "Cannot demote env-managed admin — remove from ADMIN_EMAILS instead"` |
| Auth lookup for target's email fails (deleted user, transient error) | `400 "Could not verify target user — try again"` (fail-closed) |

Self-demote is also covered by the frontend, but the backend block is the authoritative gate.

### Helpers (`src/backend/subscriptions/admin_auth.py`)

| Function | Returns | Notes |
|----------|---------|-------|
| `env_admin_emails()` | `set[str]` of lowercased emails from `ADMIN_EMAILS` | Re-read on every call (env-driven, no caching) |
| `is_env_admin(email)` | `bool` | Pure function, no DB hit |
| `is_db_admin(supabase, user_id)` | `bool` | One PK lookup on `profiles`; swallows exceptions to `False` so a transient DB error doesn't lock out env admins |
| `is_user_admin(supabase, email, user_id)` | `bool` | Single source of truth. Env check first, then DB |
| `require_admin(...)` | `str` (caller email) | FastAPI dep. 200/403/500 per the bootstrap-check logic |

Bypass behavior for the entitlements endpoint: `/me/entitlements` (in `subscriptions/router.py`) passes the admin flag into `EntitlementsService.get_for_user_safe`, which returns Pro-shaped caps/features regardless of subscription tier. Same path applies to env and DB admins.

### Database

- Column: `public.profiles.is_admin BOOLEAN NOT NULL DEFAULT false` (migration `supabase/migrations/20260517000000_add_profiles_is_admin.sql`)
- Partial index: `idx_profiles_is_admin_true ON public.profiles (is_admin) WHERE is_admin = true` — used by the bootstrap "any admin exists?" check
- **No RLS hardening on this column.** The backend service-role client is the only writer (`AdminService.promote_user`/`demote_user`); frontend has no path to write `is_admin`. If a user-scoped write path is ever added, add a `BEFORE UPDATE` trigger that rejects `is_admin` changes when `session_user <> 'service_role'`.

---

## Frontend

### Hooks (`src/hooks/useAdmin.ts`)

| Hook | Returns | Used by |
|------|---------|---------|
| `useIsAdmin()` | `{ isAdmin, loading }` — hits `/admin/me`, degrades to `false` on 403/error | Profile dropdowns, route guard, PostHog tagging |
| `useAdminUsers(search, page)` | Paginated user list with admin flags | `/admin/users` table |
| `useEntitlementsForUser(userId)` (in `useEntitlements.ts`) | `{ user, entitlements, override }` with `user.is_admin`/`is_env_admin` | User-detail sheet |
| `useAdminMutations()` | `{ grantPro, revokePro, applyOverride, clearOverride, promoteAdmin, demoteAdmin }` | Detail-sheet action buttons |
| `useAdminPosthogTag()` (in `useAdminPosthogTag.ts`) | side-effect only | Mounted once in `App.tsx` to re-identify admin users in PostHog after sign-in |

### Components

| File | Role |
|------|------|
| `src/components/AdminProtectedRoute.tsx` | Route guard. Spinner while loading, `<Navigate to="/dashboard" />` if not admin |
| `src/pages/AdminUsers.tsx` | The console: searchable user table + side-sheet with Tier / Role / Override sections + tester grants panel |

### Where the "Admin" link appears

The link is rendered conditionally with `{isAdmin && ...}` in the **Profile dropdown menu** on two surfaces:

| Surface | File | Lines |
|---------|------|-------|
| Landing page (`/`) | `src/pages/Index.tsx` | ~103-108 |
| Dashboard (`/dashboard`) | `src/pages/Dashboard.tsx` | ~231-236 |

There is no persistent sidebar — the app navigates via dropdowns. The link only shows when `/admin/me` returns `isAdmin: true`.

### Role section UI (in the user-detail sheet)

| Target state | What renders |
|--------------|--------------|
| Env admin (`is_env_admin = true`) | `Admin (env)` badge + "Managed via ADMIN_EMAILS — edit env to revoke." |
| DB admin, target ≠ caller | `Admin` badge + "Demote" button |
| DB admin, target = caller | `Admin` badge + "(cannot demote yourself)" text |
| Non-admin | "Promote to admin" button |

Env admin display takes precedence: if a user is both env- and DB-admin, the env label wins (env is immutable from the UI; the DB grant is redundant for them).

---

## Three-Layer Defense

A non-admin who somehow guesses the URL still can't get in:

1. **Link visibility** — `useIsAdmin` returns `false` → menu item not rendered.
2. **Route guard** — `<AdminProtectedRoute>` wraps `/admin/users` in `App.tsx`. Same `useIsAdmin` hook; redirects non-admins to `/dashboard`.
3. **Backend enforcement** — every `/admin/*` endpoint has `Depends(require_admin)`. Returns 403 regardless of how the request was made (curl, leaked token, etc.).

The Task-1 change of unifying `is_user_admin` into one helper is what makes all three layers consistent: a DB admin sees the link, can navigate to the route, and the backend accepts their calls — without any per-layer code change.

---

## PostHog Tagging

PostHog person property `is_admin: true` is set by `useAdminPosthogTag` (mounted via `<AdminPosthogTagger />` in `App.tsx`) once `/admin/me` resolves. The previous env-only `VITE_ADMIN_EMAILS` check in `AuthContext.tsx` was removed so DB admins are also tagged — the source of truth is the backend, not a frontend mirror.

Beta cohort dashboards can filter out internal usage with `is_admin != true`.

---

## Operational Procedures

| Procedure | Steps |
|-----------|-------|
| **Bootstrap a new environment** | Set `ADMIN_EMAILS=founder@example.com` in GSM (prod) or `.env` (local). Founder signs in. Founder promotes additional admins via `/admin/users`. |
| **Promote a teammate** | Founder opens `/admin/users`, finds the user (or asks them to sign up first if not yet listed), clicks **Manage →** in their row, then **Promote to admin** in the Role section. Confirmation via toast. |
| **Demote a teammate** | Same flow, **Demote** button. The PostHog `is_admin` tag will refresh on their next session. |
| **Revoke an env admin** | Edit `ADMIN_EMAILS`, remove the email, redeploy. Their access drops on next request (no token revoke needed — admin check runs per-request, no caching beyond `useIsAdmin`'s 5-min React Query `staleTime`). |
| **Recover from "I demoted everyone"** | Add yourself back to `ADMIN_EMAILS`, redeploy. The env path is the recovery channel. |

---

## Related Files

- Plan: `docs/superpowers/plans/2026-05-16-db-backed-admin-roles.md` (gitignored — local only)
- Migration: `supabase/migrations/20260517000000_add_profiles_is_admin.sql`
- Backend: `src/backend/subscriptions/admin_auth.py`, `admin_router.py`, `admin_service.py`
- Frontend: `src/components/AdminProtectedRoute.tsx`, `src/pages/AdminUsers.tsx`, `src/hooks/useAdmin.ts`, `src/hooks/useAdminPosthogTag.ts`
- Tests: `src/backend/tests/test_admin_router.py`, `test_admin_service.py`
- Brief reference: see "Admin Roles" section in `.claude/CLAUDE.md`
