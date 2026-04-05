# Msanii — Artist Management Platform

## What This Is

Msanii is a music industry management platform for artists, managers, and collaborators. It handles artist profiles, project/work management, rights registration, contract analysis, royalty calculations, and collaboration workflows.

## Tech Stack

**Frontend:** React 18 + TypeScript, Vite, Tailwind CSS, Radix UI / shadcn components, TanStack React Query, React Router DOM, React Hook Form + Zod, BlockNote (rich text editor)

**Backend:** FastAPI (Python 3.11), Uvicorn, deployed via Docker on Cloud Run

**Database:** Supabase (PostgreSQL) with Row Level Security (RLS). Migrations in `supabase/migrations/`.

**AI:** OpenAI API for contract analysis (Zoe) and document processing

**Email:** Resend for transactional emails (invitations, notifications)

## Running Locally

```bash
npm run dev          # Frontend on http://localhost:5173
# Backend runs on http://localhost:8000 (Docker/Cloud Run)
```

Environment variables: see `.env.example` for required keys (Supabase URL/keys, backend URL, OpenAI key).

## Project Structure

```
src/
├── backend/          # FastAPI server
│   ├── boards/       # Kanban board management
│   ├── integrations/ # Google, Slack, Notion
│   ├── oneclick/     # Royalty calculation tool
│   ├── registry/     # Rights registry (works, stakes, collaborators, licensing)
│   ├── splitsheet/   # Split sheet generator
│   ├── settings/     # User/workspace settings
│   └── vector_search/# Document vector search
├── components/       # React components
│   ├── ui/           # shadcn base components
│   ├── registry/     # Rights registry panels (ownership, licensing, agreements, collaboration)
│   ├── workspace/    # Workspace/board components
│   ├── notes/        # BlockNote rich text editor
│   ├── onboarding/   # Onboarding flow steps
│   └── zoe/          # Zoe AI chat components
├── pages/            # Route pages
├── hooks/            # Custom React hooks (data fetching, mutations)
├── contexts/         # AuthContext
├── integrations/     # Supabase client + generated types
├── types/            # TypeScript type definitions
└── lib/              # Utilities
```

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
- Collaboration flow: Invited → Accepted / Declined
- Work statuses: Draft → Pending → Registered

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
- Data fetching via TanStack React Query hooks in `src/hooks/`
- API calls go through the backend URL (`VITE_BACKEND_API_URL`)
- Supabase client for direct DB queries where appropriate (`src/integrations/supabase/client.ts`)
- Types generated from Supabase schema in `src/integrations/supabase/types.ts`
- Protected routes wrap with `ProtectedRoute` component using `AuthContext`

## Backend Conventions

- FastAPI routers in `src/backend/{module}/router.py`
- Service logic in `src/backend/{module}/service.py`
- Pydantic models in `src/backend/{module}/models.py`
- All endpoints accept `user_id` query param for RLS context
- Backend deployed via Docker (Python 3.11-slim) on Google Cloud Run

## Design Spec

Current design spec: `docs/superpowers/specs/2026-04-03-portfolio-registry-redesign.md`

This covers the Portfolio → Project Detail → Work Detail page restructure, dual-layer access control, Registry dashboard redesign, and OneClick/Zoe integration points.
