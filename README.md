# Msanii

A music industry management platform for artists, managers, and collaborators. Handles artist profiles, project/work management, rights registration, contract analysis, royalty calculations, and collaboration workflows.

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- [Node.js](https://nodejs.org/) 18+ (installed via Conda)
- [Python](https://www.python.org/) 3.11+ (installed via Conda)
- [Poetry](https://python-poetry.org/docs/#installation) (Python dependency management)
- [go-task](https://taskfile.dev/) (task runner — optional but recommended)

### 1. Set Up Environment

```bash
# Create and activate conda environment
conda create -n msanii-ai nodejs python=3.11 -c conda-forge
conda activate msanii-ai

# Install go-task (Taskfile runner)
brew install go-task
```

> **Note:** Do NOT install Homebrew's `task` package — that's Taskwarrior (a todo app), not the Taskfile runner. Use `go-task`.

### 2. Install Dependencies

```bash
# Frontend
npm install

# Backend
cd src/backend
poetry install
cd ../..
```

Or with go-task:

```bash
task install
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- **Supabase (required):**
  - `VITE_SUPABASE_URL` — Project URL from Supabase Dashboard → Settings → API
  - `VITE_SUPABASE_ANON_KEY` — Anon/public key (same page)
  - `VITE_SUPABASE_SECRET_KEY` — Service role key (same page, used by backend for RLS bypass)
  - `DATABASE_PW` — Database password (Settings → Database)
- **Backend:** `VITE_BACKEND_API_URL` (default: `http://localhost:8000`)
- **OpenAI:** `OPENAI_API_KEY` (for Zoe contract analysis)
- **Integrations (optional):** Google Drive and Slack OAuth credentials — see `.env.example` for details

### 4. Run the Project

```bash
# Terminal 1: Frontend
npm run dev
# → http://localhost:8080

# Terminal 2: Backend
cd src/backend
poetry run uvicorn main:app --port 8000
```

## Commands

### Frontend

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server on http://localhost:8080 |
| `npm run build` | Production build (outputs to `dist/`) |
| `npm run lint` | ESLint |
| `npm run preview` | Preview production build |

### Backend (from `src/backend/`)

| Command | Description |
|---------|-------------|
| `poetry install` | Install Python dependencies |
| `poetry run uvicorn main:app --port 8000` | Local backend server |
| `poetry run pytest -v` | Run all tests |
| `poetry run pytest tests/test_X.py -v` | Run specific test file |
| `poetry run ruff check .` | Lint Python code |
| `poetry run ruff format .` | Auto-format Python code |

### Taskfile Shortcuts (from repo root)

| Command | Description |
|---------|-------------|
| `task install` | Install all dependencies (Poetry + npm) |
| `task test` | Run all backend tests |
| `task lint` | Run all linters (ruff + ESLint) |
| `task lint:backend` | Run ruff lint + format check |
| `task lint:frontend` | Run ESLint |
| `task format` | Auto-fix formatting (ruff + ESLint) |
| `task ci` | Full CI pipeline locally (lint + test + build) |

## Authentication

Uses Supabase Auth with Google OAuth. To set up:

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Get project credentials from Settings → API
3. Add to `.env`:
   ```
   VITE_SUPABASE_URL=your-project-url
   VITE_SUPABASE_ANON_KEY=your-anon-key
   VITE_SUPABASE_SECRET_KEY=your-service-role-key
   ```

## Integrations

### Google Drive

Enables importing contracts/files from Drive into projects, and exporting documents (split sheets, royalty reports) back to Drive.

**Setup:**
1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Set authorized redirect URI to: `{BACKEND_URL}/integrations/google-drive/callback`
3. Add to `.env`: `GOOGLE_DRIVE_CLIENT_ID` and `GOOGLE_DRIVE_CLIENT_SECRET`

**OAuth flow:** User clicks "Connect" → redirected to Google consent screen → grants Drive access → redirected back to `/workspace?connected=google_drive` → token encrypted and stored in `integration_connections` table.

### Slack

Enables per-project Slack channel linking, rich Block Kit notifications (task updates, contract uploads, royalty calculations), and inbound @mention notifications.

**Setup:**
1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps)
2. Add OAuth scopes: `channels:read`, `chat:write`, `commands`, `incoming-webhook`
3. Enable Event Subscriptions → subscribe to `app_mention` event → set request URL to `{BACKEND_URL}/integrations/slack/webhook`
4. Set authorized redirect URI to: `{BACKEND_URL}/integrations/slack/callback`
5. Add to `.env`: `SLACK_CLIENT_ID` and `SLACK_CLIENT_SECRET`

**OAuth flow:** Same pattern as Drive — user clicks "Connect" → Slack consent → token stored. The app then sends notifications to linked channels and receives @mention webhooks.

### Shared Integration Config

Both integrations require encryption keys for secure token storage:

```bash
# Generate encryption key (one-time)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate OAuth state secret (one-time)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add both to `.env` as `INTEGRATION_ENCRYPTION_KEY` and `INTEGRATION_OAUTH_STATE_SECRET`.

## Testing

### Backend Tests

Tests use `pytest` with FastAPI's `TestClient` and mocked Supabase (no real database needed):

```bash
cd src/backend
poetry run pytest -v                           # All tests
poetry run pytest tests/test_integrations.py -v  # Integration tests only
poetry run pytest tests/test_boards.py -v        # Board tests only
```

The tests mock the Supabase client and OAuth tokens — they verify endpoint behavior (request/response shapes, status codes, error handling) without calling external APIs. For example, integration tests verify:
- Connection listing returns correct fields and omits encrypted tokens
- OAuth auth endpoints return redirect URLs
- Disconnect endpoints clean up properly
- Slack webhook handles URL verification challenges and `app_mention` events
- OneClick share validates required fields and returns correct responses

### Frontend Build Check

```bash
npm run build  # Catches TypeScript errors, missing imports
```

## Project Structure

```
oneclick/
├── src/
│   ├── backend/               # FastAPI server (Python, Poetry, Docker)
│   │   ├── main.py            # App entry, all routers mounted here
│   │   ├── boards/            # Kanban board management
│   │   ├── integrations/      # Google Drive, Slack, Notion, Monday.com
│   │   ├── oneclick/          # Royalty calculator + PDF share
│   │   ├── registry/          # Rights registry
│   │   ├── splitsheet/        # Split sheet generator
│   │   ├── settings/          # Workspace settings
│   │   ├── projects/          # Project management
│   │   ├── tests/             # pytest test suite
│   │   └── zoe_chatbot/       # Zoe AI contract chatbot
│   ├── components/            # React components
│   │   ├── ui/                # shadcn base components
│   │   ├── project/           # Project detail tabs + integration UIs
│   │   ├── workspace/         # Workspace tabs, integration hub, boards
│   │   ├── oneclick/          # OneClick calculation UI
│   │   ├── registry/          # Rights registry panels
│   │   ├── notes/             # BlockNote rich text editor
│   │   └── zoe/               # Zoe AI chat
│   ├── pages/                 # Route pages (lazy-loaded)
│   ├── hooks/                 # React Query hooks
│   ├── contexts/              # AuthContext
│   ├── integrations/          # Supabase client + types
│   ├── types/                 # TypeScript type definitions
│   └── lib/                   # Utilities
├── supabase/migrations/       # Database migrations
├── Taskfile.yml               # Task runner config
└── .env.example               # Environment variable template
```

## Deployment

| Environment | Frontend | Backend | Trigger |
|-------------|----------|---------|---------|
| **Dev** | Vercel (auto-deploy from `main`) | Cloud Run (`msanii-backend-dev`) | Push to `main` |
| **Prod** | Vercel (CLI deploy) | Cloud Run (`msanii-backend`) | Published tag release (`v*`) |

Both environments share the same Supabase database — data is user-scoped.

### Deploy to Dev

Push or merge to `main`:

```bash
git checkout main
git merge your-feature-branch
git push origin main
```

### Deploy to Prod

Create a tag release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Or create a release through GitHub: Releases → Draft a new release → Choose tag → Publish.

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `GCP_PROJECT_ID` | GCP project for Cloud Run |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | GCP auth (WIF) |
| `GCP_SERVICE_ACCOUNT_EMAIL` | GCP auth (WIF) |
| `DEV_ALLOWED_ORIGINS` | Dev Vercel URL for CORS |
| `PROD_ALLOWED_ORIGINS` | Prod Vercel URL for CORS |
| `VERCEL_PROD_TOKEN` | Vercel API token for prod deploys |
| `VERCEL_ORG_ID` | Vercel org ID |
| `VERCEL_PROJECT_ID` | Vercel prod project ID |

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run `task ci` (or manually: `task lint && task test && npm run build`)
4. Submit a pull request to `main`
5. Once merged, changes auto-deploy to dev
6. When ready for prod, create a tag release

## License

This project is private and proprietary.
