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

## Secrets — NEVER read `.env`

**Do not read, cat, grep, or otherwise load `.env` (or any secrets file) into context.** Not with the Read tool, not via `dotenv_values`/`load_dotenv` in a script you write, not with `grep`/`head`/`sed` — a key that reaches the transcript is a leaked key, even if it is never printed to the user.

Reference secrets **by variable name only**. When a script needs credentials, read them from the process environment and let the shell supply the values:

```python
# Correct — names only, values never enter context.
import os
sb = create_client(os.environ["VITE_SUPABASE_URL"], os.environ["VITE_SUPABASE_SECRET_KEY"])
```

```bash
# Correct — the subshell loads the file; the values are never echoed or read back.
set -a; . .env; set +a; poetry run python my_script.py
```

Never echo, print, or interpolate a secret's value into command output, a log line, a test fixture, or a file. To confirm a variable is set, test for presence (`[ -n "$KEY" ]`), never print it. If a task genuinely cannot proceed without the user seeing a key, ask them to run the command themselves with `!` rather than reading it yourself.

`.env.example` is safe to read — it holds names and placeholders, no values.

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
│   ├── oneclick/           # Royalty calculation tool + PDF share to Drive
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
│   │   └── integrations/   # Drive import dialog
│   ├── profile/            # User profile components
│   ├── registry/           # Metadata registry panels
│   ├── workspace/          # Workspace tabs, integration hub, boards
│   │   ├── boards/         # Kanban board, calendar view, task detail panel
│   │   └── integrations/   # DrivePanel (workspace-level config)
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
| `/boards` | Boards | Kanban board CRUD |
| `/settings` | Settings | Workspace settings |
| `/splitsheet` | Split Sheet | PDF/DOCX generation |
| `/registry` | Registry | Works, stakes, collaborators, licensing |
| `/projects` | Projects | Project management |
| `/oneclick` | OneClick Share | PDF generation + share to Drive |

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
- Users create **artist profiles**. `artists.team_id` decides who owns one: `NULL` = personal (private to the creator), `NOT NULL` = owned by that organization and visible to every ACTIVE member. See `docs/licensing.md`
- Each artist has **projects** (albums, EPs, singles, etc.)
- Projects contain **works** (individual tracks/compositions)
- **`artists.user_id` is the CREATOR, not the owner.** It keeps pointing at whoever made the artist after it is handed to a team. Never write `artists.user_id = auth.uid()` in a policy or query without `AND team_id IS NULL`, and never use it to mean "owner"

### Metadata Registry
- Works are registered with ownership stakes (master % and publishing %)
- **Collaborators** are invited per-work with splits, roles, and terms
- Collaboration flow: Invited -> Accepted / Declined
- Work statuses: Draft -> Pending -> Registered

### Three-Layer Access Control
- **Artist ownership** — the personal owner, or every active member of the owning org. Sees the whole subtree: projects, works, files, audio, credentials
- **Project members** (owner/admin/editor/viewer) — see all works in a project
- **Work-only collaborators** — see only the specific work they're invited to
- RLS policies enforce all three, and they OR together. The artist layer is expressed once as `can_access_artist(artist_id, auth.uid(), require_admin => false)` — **a new artist-scoped table needs exactly one `FOR ALL` policy calling that function**

### File Management
- Files stored in Supabase Storage (`project-files`, `audio-files` buckets)
- Files linked to projects AND optionally to specific works (via `work_files` join table)
- Audio files are artist-scoped via `audio_folders` (`audio_files.folder_id → audio_folders.artist_id`), and linkable to specific works via `work_audio_links`
- SHA-256 content hash for deduplication on upload
- **Storage attribution follows artist ownership, and lives in TRIGGERS.** `trigger_storage_pf_change` / `trigger_storage_af_change` call `_bump_storage(artist_id, delta)`, which routes bytes to `organizations.storage_bytes` for a team artist and to the creator's `usage_counters` otherwise. `recalc_user_storage` / `recalc_team_storage` are repair paths only — changing just those leaves every real upload charging the wrong side
- A file cannot cross an ownership boundary: `lock_asset_owner_move` blocks `UPDATE project_files SET project_id = <other owner>`. `WITH CHECK` can't see `OLD` and permissive policies OR, so no policy can express "the owner must not change"

### Tool Integration
- **OneClick** reads contracts from portfolio, works, and artist profiles for royalty analysis. Confirmed calculations feed a gated payment ledger (`src/backend/oneclick/royalties/ledger_sync.py` is the ONLY writer of `royalty_lines`) — identity, gates, credits, and invariants are documented in `src/backend/oneclick/ONECLICK.md` ("Royalty Ledger & Payment Tracking"); read that before touching royalties code
- **Zoe** analyzes contracts tied to works (including shared works where user is a collaborator)
- Both tools are standalone but read from the shared data model

### Credits (behind `CREDITS_ENABLED`)
Metered AI actions (Zoe message, OneClick run, Registry contract parse, Registry derive-from-contracts) draw from a per-user **credit wallet** (`credit_wallets` two buckets: `bundle_balance` expires monthly, `reserve_balance` — admin grants and purchased packs — never expires) backed by an append-only `credit_ledger` and transactional RPCs (`debit_credits`/`grant_credits`/`rollover_wallet`, all `SECURITY DEFINER`, service-role only).

**Charges are METERED, not flat.** `credit_prices` (DB, public read) is only the pre-flight ESTIMATE the balance check reserves against; the amount actually debited comes from the tokens the request burned. The `TrackedOpenAI` proxy accumulates real OpenAI cost into a per-request contextvar (`utils/llm/tracking.py` — `llm_usage_totals`, read via `credits_for_llm_usage()`), and `credits_for_cost()` (`ai_pricing.py`) converts it: `cost × CREDIT_MARKUP (default 3.0) ÷ CREDIT_OVERAGE_USD`, rounded up, floor 1 credit for any real spend. Two consequences: **an action that makes no LLM call measures 0 and is never charged** (that is how cache hits and Zoe's conversational replies are free — no per-tool special-casing), and a heavy run can exceed its estimate, which `debit_credits` absorbs as bundle drift. Fallback to the flat estimate happens in exactly two cases: no tracked scope active, or a model missing from `MODEL_RATES` (cost unknowable — never under-charge silently). Token counts and `cost_usd` ride on `credit_ledger.metadata` so a charge is explainable after the fact.

**Admins are metered like everyone else** (owner decision, 2026-08-15): `check_credits` has NO admin bypass — `BYPASS_PAYWALLS=true` is the only short-circuit (ops escape hatch). Admin ledger rows are what make the credit system testable by its own operators; admins self-grant when low. Admin-implicit-Pro (unlimited caps/features in `get_for_user`) is unchanged — it's the credit gate that admins don't skip. Also note: `load_dotenv()` runs once at backend startup and `uvicorn --reload` does not watch `.env` — a flag added to `.env` needs a backend restart to take effect.

**The debit MUST be called inside the `set_llm_context` / `iter_with_llm_context` scope** — outside it the accumulator is invisible and the charge silently falls back to the flat estimate. Decision chokepoint: `EntitlementsService.check_credits()`; charge-on-success via `gated_credits()` → `debit_for_action()` at the tool endpoints. Any new endpoint that reaches an LLM needs BOTH a `gated_credits` gate and a tracking scope, or its cost leaks entirely. Overage is opt-in (`/me/billing-prefs`), billed off the request path via Stripe InvoiceItems (daily `POST /internal/billing-sweep` + `invoice.created` webhook). Tiers are Free / Basic / Pro, keyed `free` / `basic` / `pro` (`tier_entitlements.monthly_credits`; renamed from `pro`/`pro_max` in `20260728000001_rename_tier_keys.sql` — pre-rename analytics used `pro` for the $25 plan). Under credits the AI tools are open on every tier including Free (the wallet is the only gate); storage is a hard cap with no pay-per-use. **Flag off = legacy tier gating; the stored feature flags are bypassed in code, never mutated, so it's a true rollback.** Real LLM cost is logged to `ai_usage_log` (via the `TrackedOpenAI` proxy) to calibrate prices; the planning dashboard is `subscriptions/pricing_model/` (`task pricing`).

**Monthly grants (2026-08-16 rescale): free 100 / basic 2,000 / pro 5,000.** Subscribers paid at merge time keep their pre-rescale grants (basic 3,000 / pro 8,000) via `subscriptions.grandfathered_monthly_credits`, honored only while `now() < grandfathered_until` — stamped ONCE at the backfill from the row's `current_period_end` (a monthly subscriber keeps the old grant until this month's renewal, an annual subscriber until their term ends; `now() + 1 month` when there's no Stripe period). The stamp is never extended — not by interval switches, only a TIER CHANGE ends it, and it ends it EARLY by NULLing both columns immediately in the same webhook write (`handle_subscription_updated`, `handle_subscription_deleted`, and the checkout-completed re-grant when the stored tier differs). Precedence — explicit admin override > unexpired grandfather > tier value — lives in exactly ONE helper, `EntitlementsService._resolve_monthly_grant`, called at all five grant sites (`get_for_user`, `check_credits`, `get_credit_usage`, the sweep's rollover, and the checkout-completed webhook) so a mismatch there can't silently hand two different users two different bundles.

### Licensing / organizations (behind `LICENSING_ENABLED`)
**Full guide: `docs/licensing.md`. Read it before touching orgs, team artists, or org billing.**

**Team-owned artists are the ownership edge.** `artists.team_id` NULL = personal, NOT NULL = the org owns the artist and everything that hangs off it (ten tables cascade off one row). One `SECURITY DEFINER` predicate — `can_access_artist(artist_id, user_id, require_admin => false)` — answers visibility for every artist-scoped table, so a new table needs exactly one `FOR ALL` policy calling it. `artists.user_id` stays the CREATOR after a transfer, which is why `20260803000002` re-scopes all 21 pre-existing creator-keyed policies with `AND team_id IS NULL` — without that an offboarded member keeps full read/write on the subtree of any artist they created. `team_id` has one writer: `POST /orgs/{id}/artists/{id}/transfer` (one-way in v1, 409 on re-transfer), enforced by the `artists_lock_team_id` trigger, which refuses any change made under an end-user JWT. Artists are created **client-side** (`NewArtist.tsx` and `NewArtistDialog.tsx` — there is no backend `POST /artists`), so `artists_insert_team`'s `WITH CHECK` is the only thing stopping a caller inserting into someone else's team. Storage follows the same edge inside the storage triggers via `_bump_storage`; a team's cap is `ENTERPRISE_SEAT_STORAGE_BYTES` × active seats (the env value is per-seat). **`org_project_links` was retired in `20260804000001`** — a project's payer comes from its artist; any reference to linking/unlinking a project is stale.

**Dispersal + caps** (`20260730000001_dispersal_and_caps.sql` — a fix-forward over the applied `20260721000001_licensing_core.sql`, which still describes the retired seat-wallet shape). An org negotiates a monthly credit volume — set ONLY by a Msanii admin (`PUT /admin/orgs/{id}/dispersal`), never by the org's own admin: any signed-in user can create an org and is auto-made its admin, and dispersed credits count toward the activation floor, so a customer-writable dial would mint free credits and self-activate. The org admin owns `default_member_cap` (dividing what they paid for) via `PUT /orgs/{id}`. The daily sweep rolls the org's **one** pool wallet (`credit_wallets` `owner_type='org'`) each period — pending orgs included: dispersal counts toward the activation floor and auto-activates the org when met (`maybe_activate_org`, shared with pack purchases) — granting `organizations.monthly_dispersal_credits` into the EXPIRING bundle bucket, so an unspent month can't be banked and burned later, while purchased packs (reserve) never expire. Members hold **no wallet**: they spend straight from the pool, bounded by `org_members.monthly_cap` (falling back to `organizations.default_member_cap`, then uncapped). The cap counter (`cap_used`/`cap_period_end`) moves **inside `debit_credits`** under the pool lock — a service pre-check can't stop a member's two concurrent actions both slipping under the ceiling. An over-cap debit is recorded and flagged (`cap_exceeded`), never rejected: charge-on-success means the work already happened.

**Pool visibility is admin-only.** The pool balance — and the dispersal, activation floor, `cumulative_paid_in` and org `storage_bytes` beside it — are commercial facts about the ORG, so only an ACTIVE org admin sees them. Two matching predicates enforce it at the read, never in the UI: `subscriptions.service._pool_visible_to(role)` for the credits payload (`get_for_user` org branch + `_get_credit_usage_org` → `bundleBalance`/`reserveBalance`/`balance` become **None**) and `orgs.service.redact_org_for_role(org, role)` for the org row (`get_org` + `list_my_orgs`, which both `select("*")`). Both default CLOSED, and "admin" means admin AND active — a suspended admin gets the member view, matching `is_org_admin`. **Redaction is None/absent, never 0**: a 0 reads as "the org is out of credits", which a member would act on. A member sees `memberCap`/`memberCapUsed`; with no cap there is no number at all and the UI says "Pulling from org credits pool" (`POOL_ONLY_LABEL`). Frontend surfaces must go through `creditStanding()` in `src/lib/credits.ts` — it returns null for "no number exists", and hand-rolling the cap→balance fallback chain is how `0` gets rendered for a redacted pool. Relatedly, `/me/credits/usage` `tools` is always the CALLER's own spend (`_aggregate_tool_usage(..., org_member_id=...)`, matching `metadata.org_member_id`); org-wide rollups are admin-only on `GET /orgs/{id}/usage`.

**New members start CAPPED, and "no limit" needs a sentinel.** `20260814000001` backfills and defaults `organizations.default_member_cap` to **2,000**, so the chain `org_members.monthly_cap → organizations.default_member_cap → uncapped` now caps a new member out of the box. That closes NULL's old meaning, so `-1` is the explicit "no limit" (the same `-1 = unlimited` idiom `tier_entitlements` uses) at BOTH levels. Three distinct settings on a member row: `N` = own ceiling, `NULL` = inherit the org default, `-1` = no limit. The sentinel is normalized to None/NULL on every read — `EntitlementsService._member_cap`, `orgs.service.effective_member_cap`, and a `CASE` inside `debit_credits` — so nothing above storage ever sees a negative cap; skip any of the three and a -1 member gets flagged `cap_exceeded` on every debit. **Invites live 48h** (`20260814000002`, `orgs.service.INVITE_TTL` for re-invites — keep the two in step) and gain a terminal `expired` status that exists purely to give `expire_stale_invites` (daily, from the billing sweep) a one-shot edge to notify the inviting admin off; the status UPDATE is filtered on `pending` and IS the claim, so a retry notifies nobody. `get_pending_invites` also filters `expires_at`, since a lapsed row reads `pending` until the sweep catches it.

**An invite notification is a to-do, not a message.** Its Accept/Decline buttons are gated on `notification.read`, so anything that marks it read strands the invite — which is why `registry.service.mark_all_notifications_read` skips actionable invite rows (`_is_actionable_invite`, mirroring `NotificationRow.tsx`'s `isInvite`) and why accept/decline close their own row via `mark_invite_notifications_read`. Never mark an invite row read except by actioning it.

**Self-serve teams (spec `2026-08-15-pricing-tiers-teams`; "teams" = "orgs", same table).** `organizations.kind` is `self_serve` | `enterprise` (existing orgs backfilled `enterprise`). Enterprise keeps today's negotiated model and is now admin-created ONLY — `POST /admin/orgs` + `PUT /admin/orgs/{id}/kind`, the ONLY producers of `kind='enterprise'` rows — because plain `POST /orgs` is now self-serve and slot-gated (both flags on: `kind='self_serve'`, born `active`, no activation floor — the slot IS the activation; `LICENSING_ENABLED` on with `CREDITS_ENABLED` off is a 503, never a silent fall-through to an ungoverned enterprise row). Team slots (teams a tier may COVER) are Basic 1 / Pro 3; team sizes (members EXCLUDING the covering owner) are 3 / 10 — `tier_entitlements.max_teams`/`max_team_members`/`team_storage_bytes`, and a Msanii admin's own dials resolve as Pro regardless of their subscription tier. Joining a team is free on every tier — only owning (covering) one draws on a slot. **Coverage is claim-based, never assigned:** `covered_by`/`covered_at` are set at creation and move only through `POST /orgs/{id}/coverage/claim` (free-slot-gated) and `.../release` (current coverer only — the release write is conditioned on `WHERE covered_by = user_id`, so a rival's claim landing in the read-then-write gap loses the race cleanly instead of getting silently overwritten). **Standing is a daily-sweep PREDICATE, never an event:** `orgs.standing.evaluate_standing` ranks each coverer's orgs `covered_at DESC` up to their slot count; an uncovered org gets `grace_started_at`, then past `ORG_GRACE_DAYS` (env, default 14) flips to `status='lapsed'` — the one new RLS predicate this whole feature adds, `can_access_artist`'s team branch gaining `AND o.status <> 'lapsed'`, denies the entire subtree to everyone (admins included), same posture as `archived_at`. **Archive is reversible** (frees the slot, admin-only, no balance precondition — whatever the pool holds survives). **Dissolve is terminal-soft**: `dissolved_at` is stamped, every team artist reverts to its creator (or the dissolving admin, when the creator no longer holds a seat) via a service-role `UPDATE artists SET team_id = NULL`, the pool's purchased reserve is forfeited through the existing clawback RPC, and the org row/pool wallet/ledger are all RETAINED (archived, never deleted — support still needs to read them). `_require_live_org` guards the mutating lifecycle endpoints (invite/accept, member cap/role, credit-requests, project-member grants, claim-coverage, transfer-credits); reads (`get_org`, dissolve-preview, usage, ledger) and `cancel-topup` are deliberately exempt — the latter is the documented retry path for a Stripe cancel that failed mid-archive/dissolve, so a dead org must still be able to stop its own charge. **Team storage is a fully separate per-OWNER pool from personal storage**, summed across every self-serve org `covered_by` still points at, active + archived but excluding dissolved (`orgs.storage_guard.pool_state` — parked bytes in an archived team still count against the cap, so archive-then-recreate can't dodge it): Basic 10 GiB hard cap, Pro 100 GiB then PAYG at `TEAM_STORAGE_OVERAGE_USD_PER_GB` (env, seeded $0.025/GB/mo, billed monthly as a Stripe InvoiceItem on the covering owner's PERSONAL customer, idempotent per owner-period). Funding is admin-driven, all landing in the pool's reserve (never expires): the reserve-only `transfer_credits` RPC (personal reserve → pool, 409 on insufficient reserve — never a silent clamp), packs, and a recurring top-up riding the purchasing admin's PERSONAL Stripe customer (`kind='org_topup'` metadata — the personal subscription webhook handlers early-return on it so an org top-up can never overwrite the purchaser's own plan; `invoice.paid` grants pool reserve, idempotent on the Stripe invoice id; offboarding the purchasing admin cancels it).

Caps are ceilings, not reservations, so they may deliberately sum to more than the dispersal. Two org walls with different remedies: `capReached` → the member asks an admin to raise it (`credit_requests` is a **cap-raise** request; approving writes `monthly_cap` and moves nothing); a dry pool → only an admin buying credits helps, so the member sees no CTA. No pay-as-you-go on a pool. Offboarding is a soft status transition with nothing to reclaim, and archiving leaves pool credits for support (admin clawback is reserve-only) and leaves `artists.team_id` attached — `can_access_artist` already denies on `archived_at`, so the roster goes inert without being destroyed. Per-member spend for the console comes from the pool ledger grouped by `metadata.org_member_id`. QA: `scripts/qa_licensing_loop.py` (HTTP lifecycle) + `supabase/qa/gates_team_artists.sql` (artist ownership, RLS, storage triggers) + `supabase/qa/launch_gates_credit_rpcs.sql` (the money RPCs) — pytest mocks `sb.rpc()` and never reaches Postgres, so the gate scripts are the SQL layer's ONLY executable coverage.

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
| `team_created` / `team_archived` / `team_unarchived` / `team_dissolved` | `orgs/router.py` | Self-serve team lifecycle (`POST /orgs` is self-serve-only now; enterprise creation via `POST /admin/orgs` is admin tooling and untracked). |
| `team_grace_started` / `team_lapsed` / `team_reactivated` | `orgs/standing.py` (`_notify_standing`, `f"team_{kind}"`) | Daily standing-sweep transitions (`evaluate_standing`); captured to the covering admin, not the affected members. |
| `coverage_claimed` / `coverage_released` | `orgs/router.py` | Team-slot coverage claim/release. |
| `credits_transferred` | `orgs/router.py` | Personal reserve → org pool (`transfer_credits` RPC). |
| `org_topup_started` / `org_topup_renewed` | `subscriptions/stripe_events.py` | Recurring org top-up lifecycle, off Stripe `invoice.paid`. |
| `org_topup_canceled` | `orgs/router.py` (manual cancel) + `orgs/service.py` (offboard, dissolve) | Three call sites by design — `trigger` property (`manual`/`offboard`/`dissolve`) distinguishes them; not a double-fire. |
| `team_storage_overage_billed` | `orgs/storage_guard.py` | Pro team-storage PAYG, billed by the sweep. |
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
- `stripe-python` is pinned to 15.1.0 (`pyproject.toml` range `>=11,<16`) where `StripeObject` stopped being a `dict` subclass: any `x.metadata.get(...)`-style read on a live Stripe payload must go through `_plain()` in `subscriptions/stripe_events.py` instead of dict-style access, and webhook-path tests must build events with `stripe.Event.construct_from(...)`, never a raw dict mock

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

**Superseded in part:** access control is now three layers, not two — artist ownership sits above project members and work-only collaborators. See `docs/licensing.md`.
