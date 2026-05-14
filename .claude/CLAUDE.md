# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Msanii is a music industry management platform for artists, managers, and collaborators. It handles artist profiles, project/work management, rights registration, contract analysis, royalty calculations, and collaboration workflows.

## Tech Stack

**Frontend:** React 18 + TypeScript, Vite, Tailwind CSS, Radix UI / shadcn components, TanStack React Query, React Router DOM, React Hook Form + Zod, BlockNote (rich text editor)

**Backend:** FastAPI (Python 3.11), Uvicorn, deployed via Docker on Cloud Run

**Database:** Supabase (PostgreSQL) with Row Level Security (RLS). Migrations in `supabase/migrations/`.

**AI:** OpenAI API for contract analysis (Zoe) and document processing

**Email:** Resend for transactional emails (invitations, notifications)

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
│   │   ├── slack/          # OAuth, channels, Block Kit notifications, webhook (@mentions)
│   │   ├── notion/         # OAuth, database listing, bidirectional task sync
│   │   └── monday/         # OAuth, board listing, bidirectional task sync
│   ├── oneclick/           # Royalty calculation tool + PDF share to Drive/Slack
│   ├── registry/           # Rights registry (works, stakes, collaborators, licensing, PDF)
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
│   ├── registry/           # Rights registry panels
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
| `/integrations/notion` | Notion | OAuth, database listing, task sync |
| `/integrations/monday` | Monday.com | OAuth, board listing, task sync |
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

### Rights Registry
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
- Audio files are project-scoped, linkable to works (via `work_audio_links`)
- SHA-256 content hash for deduplication on upload

### Tool Integration
- **OneClick** reads contracts from portfolio, works, and artist profiles for royalty analysis
- **Zoe** analyzes contracts tied to works (including shared works where user is a collaborator)
- Both tools are standalone but read from the shared data model

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

- Backend is a separate Python project in `src/backend/` with its own `Dockerfile` and `requirements.txt`
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
