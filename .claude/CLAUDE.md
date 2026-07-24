# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Msanii is a music industry management platform for artists, managers, and collaborators. It handles artist profiles, project/work management, rights registration, contract analysis, royalty calculations, and collaboration workflows.

## Role
You are a software engineer on the Msanii squad. Users are artists, managers, and collaborators — mostly non-technical — who rely on this app for a wide range of day-to-day work: managing artist profiles and projects, registering rights, analyzing contracts, calculating royalties, collaborating on works, organizing files, and more. The product surface will keep expanding into adjacent music-industry workflows, so design changes to compose with future features rather than locking the app to today's tools.

Optimize for:
1. **Correctness** in anything touching ownership splits, royalty math, contract data, or RLS-protected resources — never approximate, never widen access.
2. **Usability** for non-technical users — clear copy, forgiving flows, sensible defaults, no jargon leaking into the UI.
3. **Smallest correct change** — fix what was asked, don't sprawl into unrelated refactors or speculative abstractions.

**Pushback is encouraged** — and expected — when (a) you see a better approach than what was proposed, (b) a request risks something destructive or hard to reverse, or (c) a change would compromise correctness, security, or UX. Explain the tradeoff and recommend an alternative; don't silently comply with a worse path. Always ask before taking destructive or irreversible actions.

**Use installed Claude Code skills aggressively** — they exist to raise the floor on quality. In particular:
- `/superpowers-extended-cc:brainstorming` before designing any new feature or non-trivial change
- `/superpowers-extended-cc:write-plan` for multi-step work, before touching code
- `/superpowers-extended-cc:test-driven-development` when adding features or fixing bugs
- `/superpowers-extended-cc:systematic-debugging` for any bug, test failure, or unexpected behavior
- `/superpowers-extended-cc:verification-before-completion` before claiming any work is done
- `/vercel-react-best-practices` when writing or refactoring React/Next.js code
- `/frontend-design` when building or restyling UI components or pages
- `/superpowers-extended-cc:requesting-code-review` before merging or shipping major work

If a skill plausibly applies, invoke it — don't rationalize skipping it.

**After every new feature or bugfix, run the full verification suite** (frontend build + backend lint + backend tests — see the Verification section below) and confirm everything passes before declaring the work complete. A green local run is the bar; "should work" is not.

## Tech Stack

**Frontend:** React 18 + TypeScript, Vite, Tailwind CSS, Radix UI / shadcn components, TanStack React Query, React Router DOM, React Hook Form + Zod, BlockNote (rich text editor)

**Backend:** FastAPI (Python 3.11), Uvicorn, deployed via Docker on Cloud Run

**Database:** Supabase (PostgreSQL) with Row Level Security (RLS). Migrations in `supabase/migrations/`.

**AI:** OpenAI API for contract analysis (Zoe) and document processing

**Email:** Resend for transactional emails (invitations, notifications)

**Analytics:** PostHog — `posthog-python` (backend, `src/backend/analytics.py`) and `posthog-js` (frontend, `src/lib/posthog.ts`). See the PostHog section below for events and dashboards.

## Commands

```bash
# Frontend
npm run dev          # Dev server on http://localhost:8080
npm run build        # Production build (outputs to dist/)
npm run build:dev    # Development build
npm run lint         # ESLint
npm run preview      # Preview production build

# Backend (from src/backend/)
poetry install                              # Install Python dependencies
poetry run uvicorn main:app --port 8000     # Local backend server
poetry run pytest -v                        # Run all backend tests
poetry run pytest tests/test_X.py -v        # Run specific test file
poetry run ruff check .                     # Lint Python code
poetry run ruff format .                    # Auto-format Python code
poetry run ruff format --check .            # Check formatting without changes

# Taskfile shortcuts (from repo root)
task test            # Run all backend tests
task lint            # Run all linters (ruff + ESLint)
task lint:backend    # Run ruff lint + format check
task lint:frontend   # Run ESLint
task format          # Auto-fix formatting (ruff + ESLint)
```

Environment variables: copy `.env.example` to `.env` and fill in Supabase URL/keys, backend URL, OpenAI key, integration OAuth credentials.

## Verification — REQUIRED After Every Change

**After completing any feature, bug fix, or edit, always run these checks before considering the work done:**

```bash
# Frontend: build must pass (catches TypeScript errors, missing imports)
npm run build

# Backend: lint + format must pass
cd src/backend && poetry run ruff check . && poetry run ruff format --check .

# Backend: tests must pass
cd src/backend && poetry run pytest -v
```

If any check fails, fix the issue before moving on. Do not skip these steps.

## Project Structure

```
src/
├── backend/                # FastAPI server (separate Python project, own Dockerfile)
│   ├── main.py             # App entry point, mounts all routers, event handler registration
│   ├── auth.py             # JWT authentication via Supabase JWKS
│   ├── pagination.py       # Shared pagination utilities
│   ├── boards/             # Kanban board management
│   ├── integrations/       # Third-party integrations
│   │   ├── oauth.py        # Shared OAuth token management (encryption, refresh, state JWT)
│   │   ├── events.py       # Internal event bus (emit/subscribe for notifications)
│   │   ├── connections_router.py  # GET /integrations/connections (list user's connections)
│   │   ├── google_drive/   # OAuth, file browse, import/export, PDF upload
│   │   └── slack/          # OAuth, channels, Block Kit notifications, webhook (@mentions)
│   ├── oneclick/           # Royalty calculation tool + PDF share to Drive/Slack
│   ├── registry/           # Metadata registry (works, stakes, collaborators, licensing, PDF)
│   ├── splitsheet/         # Split sheet PDF/DOCX generator
│   ├── settings/           # User/workspace settings
│   ├── projects/           # Project management endpoints
│   ├── tests/              # pytest test suite (mock-based, no real DB)
│   └── zoe_chatbot/        # Zoe AI contract chatbot + document helpers
├── components/             # React components
│   ├── ui/                 # shadcn base components
│   ├── oneclick/           # OneClick calculation results, contract/statement selectors
│   ├── project/            # Project detail tabs (works, files, audio, members, settings)
│   │   └── integrations/   # Drive import dialog, project Slack settings
│   ├── profile/            # User profile components
│   ├── registry/           # Metadata registry panels
│   ├── workspace/          # Workspace tabs, integration hub, boards
│   │   ├── boards/         # Kanban board, calendar view, task detail panel
│   │   └── integrations/   # DrivePanel, SlackPanel (workspace-level config)
│   ├── notes/              # BlockNote rich text editor
│   ├── walkthrough/        # Tool onboarding/walkthrough system
│   ├── onboarding/         # Onboarding flow steps
│   └── zoe/                # Zoe AI chat components
├── pages/                  # Route pages (most are lazy-loaded)
├── hooks/                  # Custom React Query hooks (data fetching, mutations)
├── config/                 # App configuration (walkthrough configs)
├── contexts/               # AuthContext (single context, wraps the app)
├── integrations/           # Supabase client + generated types
├── types/                  # TypeScript type definitions
└── lib/                    # Utilities (apiFetch, API_URL)
```

## Architecture

### Path Alias
`@/` maps to `./src/` (configured in tsconfig.json and vite.config.ts). All imports use this alias.

### TypeScript Config
Relaxed strict mode: `strictNullChecks: false`, `noImplicitAny: false`, `noUnusedLocals: false`. Don't add strict null checks to existing code.

### Frontend Data Flow
- Pages are lazy-loaded in `App.tsx` via `React.lazy()` with a `<Suspense>` wrapper
- `AuthProvider` wraps all routes; `ProtectedRoute` guards authenticated pages
- Data fetching uses TanStack React Query hooks in `src/hooks/` (one hook per domain: `useRegistry`, `useBoards`, `usePortfolioData`, etc.)
- API calls to the backend go through `VITE_BACKEND_API_URL`
- Direct Supabase queries use `src/integrations/supabase/client.ts`
- Types generated from Supabase schema in `src/integrations/supabase/types.ts`

### Backend API Route Prefixes
All routers are mounted in `src/backend/main.py`:

| Prefix | Router | Purpose |
|--------|--------|---------|
| `/integrations` | Connections | List user's integration connections |
| `/integrations/google-drive` | Google Drive | OAuth, file browse, import/export, PDF upload |
| `/integrations/slack` | Slack | OAuth, channels, notification settings, webhook |
| `/boards` | Boards | Kanban board CRUD |
| `/settings` | Settings | Workspace settings |
| `/splitsheet` | Split Sheet | PDF/DOCX generation |
| `/registry` | Registry | Works, stakes, collaborators, licensing |
| `/projects` | Projects | Project management |
| `/oneclick` | OneClick Share | PDF generation + share to Drive/Slack |

Additional endpoints (file upload, Zoe chat, OneClick calculation) are defined directly in `main.py`.

### Backend Module Pattern
Each module follows: `router.py` (FastAPI routes) + `service.py` (business logic) + `models.py` (Pydantic schemas). All endpoints accept `user_id` query param for Supabase RLS context.

## Key Routes

| Path | Page | Purpose |
|---|---|---|
| `/portfolio` | Portfolio | Project grid by artist |
| `/projects/{projectId}` | ProjectDetail | Tabbed project view (works, files, audio, members, settings) |
| `/tools/registry` | Registry | Ownership tracking dashboard |
| `/tools/registry/{workId}` | WorkDetail | Per-work ownership, licensing, agreements |
| `/tools/registry/invite/{token}` | InviteClaim | Collaborator invitation claim |
| `/tools/oneclick` | OneClick | Royalty calculation entry point |
| `/tools/zoe` | Zoe | AI contract analysis chat |
| `/tools/split-sheet` | SplitSheet | Split sheet generator |
| `/workspace` | Workspace | Kanban boards, calendar, settings |
| `/artists` | Artists | Artist profile management |

## Core Concepts

### Artists & Projects
- Users create **artist profiles** (private to the creator — personal docs/notes folder)
- Each artist has **projects** (albums, EPs, singles, etc.)
- Projects contain **works** (individual tracks/compositions)

### Metadata Registry
- Works are registered with ownership stakes (master % and publishing %)
- **Collaborators** are invited per-work with splits, roles, and terms
- Collaboration flow: Invited -> Accepted / Declined
- Work statuses: Draft -> Pending -> Registered

### Dual-Layer Access Control
- **Project members** (owner/admin/editor/viewer) — see all works in a project
- **Work-only collaborators** — see only the specific work they're invited to
- RLS policies enforce both layers

### File Management
- Files stored in Supabase Storage (`project-files`, `audio-files` buckets)
- Files linked to projects AND optionally to specific works (via `work_files` join table)
- Audio files are artist-scoped via `audio_folders` (`audio_files.folder_id → audio_folders.artist_id`), and linkable to specific works via `work_audio_links`
- SHA-256 content hash for deduplication on upload

### Tool Integration
- **OneClick** reads contracts from portfolio, works, and artist profiles for royalty analysis. Confirmed calculations feed a gated payment ledger (`src/backend/oneclick/royalties/ledger_sync.py` is the ONLY writer of `royalty_lines`) — identity, gates, credits, and invariants are documented in `src/backend/oneclick/ONECLICK.md` ("Royalty Ledger & Payment Tracking"); read that before touching royalties code
- **Zoe** analyzes contracts tied to works (including shared works where user is a collaborator)
- Both tools are standalone but read from the shared data model

### Credits (behind `CREDITS_ENABLED`)
Metered AI actions (Zoe message, OneClick run, Registry contract parse) draw from a per-user **credit wallet** (`credit_wallets` two buckets: `bundle_balance` expires monthly, `reserve_balance` — admin grants — never expires) backed by an append-only `credit_ledger` and transactional RPCs (`debit_credits`/`grant_credits`/`rollover_wallet`, all `SECURITY DEFINER`, service-role only). Prices live in `credit_prices` (DB, public read). Cache hits and Zoe conversational replies are free. Decision chokepoint: `EntitlementsService.check_credits()`; charge-on-success via `gated_credits()` → `debit_for_action()` at the tool endpoints. Overage is opt-in (`/me/billing-prefs`), billed off the request path via Stripe InvoiceItems (daily `POST /internal/billing-sweep` + `invoice.created` webhook). Tiers are Free / Pro / Pro Max (`tier_entitlements.monthly_credits`). **Flag off = legacy tier gating; the stored feature flags are bypassed in code, never mutated, so it's a true rollback.** Real LLM cost is logged to `ai_usage_log` (via the `TrackedOpenAI` proxy) to calibrate prices. Spec: `docs/superpowers/specs/2026-07-12-credits-billing-design.md`; plan: `docs/superpowers/plans/2026-07-13-credits-backend.md`.

### Admin Roles

Admin access is granted two ways:
1. **Bootstrap:** Emails in the `ADMIN_EMAILS` env var are always admins. Used to seed the first admin(s); store in GSM in prod.
2. **DB-managed:** `profiles.is_admin = true` grants admin access. Toggled by other admins via `/admin/users` → "Promote to admin" / "Demote".

Self-demote and env-admin demote are blocked at the backend. To revoke an env admin, remove them from `ADMIN_EMAILS` and redeploy. The single source of truth for "is this user an admin?" is `is_user_admin(supabase, email, user_id)` in `src/backend/subscriptions/admin_auth.py`.

## Analytics & PostHog

Product analytics live in a single PostHog project (`https://us.posthog.com`, project id `427173`, "Default project"). Dev and prod backends both emit to it — be aware when reading numbers.

### Wrappers

| Layer | File | Purpose |
|---|---|---|
| Backend | `src/backend/analytics.py` | Thin `posthog-python` wrapper. `capture()`/`identify()` are no-ops unless `POSTHOG_ENABLED=true` is set in env. |
| Backend middleware | `src/backend/middleware/analytics_middleware.py` | Fires `request_completed` / `request_failed` for every API call. Excludes `/static`, `/docs`, `/redoc`, `/openapi.json`, `/health`. |
| Frontend | `src/lib/posthog.ts` | `posthog-js` init — autocapture off, session recording off, person profiles `identified_only`. Captures `$pageview` / `$pageleave`. |
| Frontend hook | `src/hooks/useAnalyticsContext.ts` | Pulls `/me/analytics-context` and `identify()`s the user with plan/role/tester/admin properties. Cached in localStorage. |

### Event taxonomy

The canonical "a tool was used" signal is `tool_used` with `properties.tool ∈ {zoe, oneclick, splitsheet}`. Registry events are separate (`work_created`, `registry_work_registered`, `registry_collaborator_invited`).

| Event | Fired in | Notes |
|---|---|---|
| `tool_used` | `main.py` (zoe, oneclick), `splitsheet/router.py` | `properties.tool` distinguishes which tool. |
| `zoe_query_submitted` / `zoe_response_received` / `zoe_query_failed` | `main.py` | Step events for the Zoe funnel. |
| `oneclick_calc_started` / `oneclick_calc_completed` / `oneclick_calc_failed` | `main.py` | Step events for the OneClick funnel. |
| `splitsheet_generated` | `splitsheet/router.py` | Fired after PDF/DOCX is built. |
| `work_created` / `work_submitted_for_registration` / `registry_work_registered` / `registry_collaborator_invited` | `registry/router.py` | Registry lifecycle. |
| `contract_uploaded` | `main.py` | Includes `file_size`. |
| `checkout_started` / `billing_portal_opened` | `subscriptions/billing_router.py` | Stripe entry points. |
| `subscription_activated` / `subscription_canceled` / `payment_failed` | `subscriptions/stripe_events.py` | Stripe webhook outcomes. |
| `request_completed` / `request_failed` | Middleware | Every API request — useful for traffic, noisy for tool counts. |

When adding a new tool or feature, follow the existing pattern: emit a step event when work starts (e.g. `<tool>_started`) and a completion event when it succeeds. `tool_used` may be redundant if you already have a step event — prefer one or the other consistently.

### Dashboards

| Dashboard | Path | Purpose |
|---|---|---|
| Analytics basics (wizard-built) | `/dashboard/1593101` | Subscription funnel, churn, monthly active users, registry activity. |
| Tool Usage — per tool + comparative | `/dashboard/1597175` | Per-tool counts, drop-off funnels, stacked comparative, unique users, weekly line. Built by Claude. |

Admin-facing in-app analytics: `GET /admin/analytics/summary` (`src/backend/admin/analytics_router.py`) returns per-tool opens/completions/last_used via HogQL — used by the admin dashboard.

### Environment variables

| Var | Backend / Frontend | Purpose |
|---|---|---|
| `POSTHOG_ENABLED` | Backend | Must be `"true"` for `capture()` to actually emit. Defaults to off. |
| `POSTHOG_PROJECT_TOKEN` | Backend | `phc_…` ingest token (also the SDK key). |
| `POSTHOG_HOST` | Backend | `https://us.i.posthog.com`. |
| `POSTHOG_PERSONAL_API_KEY` | Backend (scripts only) | `phx_…` for the dashboard setup script and ad-hoc PostHog REST calls. NOT for ingest. |
| `POSTHOG_PROJECT_ID` | Backend (scripts only) | Numeric project id (`427173`) used by the REST API. |
| `VITE_POSTHOG_PROJECT_TOKEN` | Frontend | Same `phc_…` token, exposed to the browser. |
| `VITE_POSTHOG_HOST` | Frontend | `https://us.i.posthog.com`. |
| `VITE_POSTHOG_DASHBOARD_URL` | Frontend | Base URL for "Open in PostHog" links in admin UI. |
| `VITE_APP_ENV` | Frontend | `local` (default) / `dev` / `prod`. Registered as PostHog super-property `environment` on every browser event. Dashboards filter on this. |

### Dashboard setup script

`src/backend/scripts/posthog_setup_dashboard.py` is an idempotent script that creates/maintains cohorts, insights, and dashboards in PostHog. State file is `scripts/.posthog_dashboard_state.{env}.json` — it maps logical names → entity IDs so the UI can rename things without breaking the script. Run with `--adopt` once to seed state from existing entities.

### Dashboard backfill script

`src/backend/scripts/posthog_apply_env_filter.py` is a one-time backfill that walks insights on a given dashboard and PATCHes the env + date filter into each one's filter tree. Idempotent; supports `--dry-run`. Use after deploying the env-tagging change to clean up pre-existing dashboards (e.g. `1593101`, `1597175`) that weren't built via the setup script.

```bash
# Dry-run first — inspects each insight and prints the proposed mutation diff.
# Query-based insights (insight.query non-null) are skipped with a WARNING because
# their filter trees live under query.source.properties, not filters.properties.
poetry run python -m scripts.posthog_apply_env_filter \
    --dashboard-id 1593101 --dashboard-id 1597175 --dry-run

# Apply for real
poetry run python -m scripts.posthog_apply_env_filter \
    --dashboard-id 1593101 --dashboard-id 1597175
```

### Known caveats

- **Single PostHog project across dev + prod.** Both `deploy-backend.yml` and `deploy-backend-dev.yml` set `POSTHOG_ENABLED=true` and share the same `POSTHOG_PROJECT_TOKEN`. Dev traffic is mixed in with prod traffic — separated via the `environment` event property (`local` / `dev` / `prod`). Backend tags via `APP_ENV`, frontend tags via `VITE_APP_ENV`. The load-bearing exclusion is `environment IN ('dev', 'prod')` — applied in HogQL queries (`admin/analytics_router.py`) and as a property filter on insights. The `timestamp >= '2026-05-19'` clause is a hard floor in HogQL but a default-view boundary on insights (dashboard date pickers can override `date_from`). Events before the cutoff are mostly untagged local-dev pollution and are excluded by the env IN-list anyway.
- **Test ingest leakage is partially mitigated.** `src/backend/tests/conftest.py` still does not pin `POSTHOG_ENABLED=false`. If a developer's shell has it enabled, pytest runs will still hit PostHog's ingest endpoint and burn quota. With env tagging in place, those leaked events tag as `environment=local` (assuming `APP_ENV` is unset in test env, which it is by default) and are excluded from every dashboard — so dashboard pollution from tests is now fixed. The remaining cost is ingest spend, not dirty data.

## Database Conventions

- All tables use UUID primary keys (`gen_random_uuid()`)
- Timestamps: `created_at` and `updated_at` with `TIMESTAMPTZ DEFAULT now()`
- RLS enabled on all tables — policies check `auth.uid()`
- Migrations in `supabase/migrations/` — named `YYYYMMDD######_description.sql`
- **Do not run migrations directly.** Create migration files and let the user run them.

## Frontend Conventions

- Components use shadcn/ui (Radix primitives + Tailwind)
- UI components live in `src/components/ui/` — add new shadcn components there
- Data fetching via TanStack React Query hooks in `src/hooks/`
- Protected routes wrap with `ProtectedRoute` component using `AuthContext`

## Backend Conventions

- Backend is a separate Python project in `src/backend/` with its own `Dockerfile`; deps are managed with Poetry (`pyproject.toml` + `poetry.lock`) and installed directly in the container — no `requirements.txt`
- Backend deploys to Cloud Run on port 8080 (Docker), runs locally on port 8000
- All endpoints accept `user_id` query param for RLS context

## Deployment

| Environment | Frontend | Backend | Trigger |
|-------------|----------|---------|---------|
| **Dev** | Vercel (auto-deploy from `main`) | Cloud Run (`msanii-backend-dev`) | Push to `main` |
| **Prod** | Vercel (CLI deploy) | Cloud Run (`msanii-backend`) | Published tag release (`v*`) |

Both environments share the same Supabase database — data is user-scoped.

- Dev backend deploys on push to `main` (only when `src/backend/**` changes)
- Prod deploys on tag push (`v*`) — create via `git tag v1.0.0 && git push origin v1.0.0` or GitHub Releases UI

## Design Spec

Current design spec: `docs/superpowers/specs/2026-04-03-portfolio-registry-redesign.md`

This covers the Portfolio -> Project Detail -> Work Detail page restructure, dual-layer access control, Registry dashboard redesign, and OneClick/Zoe integration points.
