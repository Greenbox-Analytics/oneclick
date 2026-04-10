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
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000  # Local backend server
```

Environment variables: copy `.env.example` to `.env` and fill in Supabase URL/keys, backend URL, OpenAI key.

## Project Structure

```
src/
├── backend/          # FastAPI server (separate Python project, own Dockerfile)
│   ├── main.py       # App entry point, mounts all routers
│   ├── boards/       # Kanban board management
│   ├── integrations/ # Google Drive, Slack, Notion, Monday.com
│   ├── oneclick/     # Royalty calculation tool
│   ├── registry/     # Rights registry (works, stakes, collaborators, licensing)
│   ├── splitsheet/   # Split sheet generator
│   ├── settings/     # User/workspace settings
│   ├── projects/     # Project management endpoints
│   └── zoe_chatbot/  # Zoe AI contract chatbot + document helpers
├── components/       # React components
│   ├── ui/           # shadcn base components
│   ├── registry/     # Rights registry panels
│   ├── workspace/    # Workspace/board components
│   ├── notes/        # BlockNote rich text editor
│   ├── onboarding/   # Onboarding flow steps
│   └── zoe/          # Zoe AI chat components
├── pages/            # Route pages (most are lazy-loaded)
├── hooks/            # Custom React Query hooks (data fetching, mutations)
├── contexts/         # AuthContext (single context, wraps the app)
├── integrations/     # Supabase client + generated types
├── types/            # TypeScript type definitions
└── lib/              # Utilities
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
| `/integrations/google-drive` | Google Drive | File sync |
| `/integrations/slack` | Slack | Notifications |
| `/integrations/notion` | Notion | Content sync |
| `/integrations/monday` | Monday.com | Project sync |
| `/boards` | Boards | Kanban board CRUD |
| `/settings` | Settings | Workspace settings |
| `/splitsheet` | Split Sheet | PDF/DOCX generation |
| `/registry` | Registry | Works, stakes, collaborators, licensing |
| `/projects` | Projects | Project management |

Additional endpoints (file upload, Zoe chat, OneClick) are defined directly in `main.py`.

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
