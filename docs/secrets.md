# Secrets & Environment Variables

Every env var the app reads, where to get it, and which are required to run.

**Local:** `.env` in the repo root (gitignored). Copy `.env.example` and fill in values.
**Production:** Stored in Google Secret Manager (GSM); injected into Cloud Run + Vercel at deploy time. Frontend `VITE_*` vars are baked into the Vite build, so they must be set at build time (via Vercel project env), not at runtime.

Required = the app crashes or that feature is unusable without it. Optional = there's a sensible default in code.

---

## 🔴 Required to boot the app at all

These are the bare minimum to run `npm run dev` + `poetry run uvicorn main:app` and have a working signed-in user. Without them you'll hit immediate startup or auth errors.

| Var | Used by | Source | Format / notes |
|-----|---------|--------|----------------|
| `VITE_SUPABASE_URL` | FE + BE | Supabase Dashboard → Project Settings → API → Project URL | `https://<ref>.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | FE | Same dashboard → "anon public" key | `eyJ...` JWT |
| `VITE_SUPABASE_SECRET_KEY` | BE | Same dashboard → "service_role" key (REVEAL secret) | `eyJ...` JWT. **Never** expose to frontend |
| `DATABASE_PW` | Scripts / migrations | Supabase Dashboard → Project Settings → Database → Connection string password | One-time; auto-rotation requires re-saving |
| `VITE_BACKEND_API_URL` | FE | Your local backend URL | `http://localhost:8000` for dev |
| `ALLOWED_ORIGINS` | BE (CORS) | Comma-separated frontend URLs | `http://localhost:8080` for dev. Prod set in `deploy-backend.yml` |

---

## 🟠 Required for the Pricing / Billing feature

Without these, `/billing/*` endpoints raise `KeyError` on first call and the Pricing page is non-functional. See [stripe-integration.md](stripe-integration.md) for the full local-testing flow.

| Var | Source | Format |
|-----|--------|--------|
| `STRIPE_SECRET_KEY` | Stripe Dashboard → Developers → API keys → Secret key | `sk_test_...` locally, `sk_live_...` in prod |
| `STRIPE_WEBHOOK_SECRET` | Local: output of `stripe listen --forward-to ...`. Prod: Webhooks → endpoint → Reveal signing secret | `whsec_...`; **different per environment** |
| `STRIPE_PRICE_MONTHLY` | Stripe Dashboard → Products → recurring price ID | `price_...` |
| `STRIPE_PRICE_ANNUAL` | Stripe Dashboard → Products → second price ID | `price_...` |
| `FRONTEND_URL` | Your frontend's base URL | `http://localhost:8080` (dev), `https://app.msanii.com` (prod). Used for Checkout success/cancel redirects + Portal return |

**Test-mode IDs do NOT work with `sk_live_`** — recreate products + prices in live mode for prod.

---

## 🟠 Required for OpenAI features (Zoe + OneClick contract parsing)

| Var | Source | Required? |
|-----|--------|-----------|
| `OPENAI_API_KEY` | platform.openai.com → API keys | Yes — without it, Zoe + contract analysis crash on first call |
| `OPENAI_LLM_MODEL` | Model name override | Optional. Defaults coded in `zoe_chatbot/` and `oneclick/contract_parser.py` |
| `OPENAI_LLM_MODEL_LARGE` | Large-context model name | Optional |
| `OPENAI_BASE_URL` | Alternate API host (OpenAI-compatible proxy, Azure, etc.) | Optional. Empty = use OpenAI default |

`CLAUDE_API_KEY`, `CLAUDE_MODEL`, `CLAUDE_BASE_URL` are in `.env.example` but are **not currently read** by any backend code — leftover scaffolding. Safe to leave blank.

---

## 🟠 Required for emails (invites, welcome, Pro-access notifications)

| Var | Source | Required? |
|-----|--------|-----------|
| `RESEND_API_KEY` | resend.com → API Keys | Yes for transactional email |
| `RESEND_FROM_EMAIL` | An address on a Resend-verified domain | Yes — no fallback. Format: `Display Name <noreply@verified-domain.com>` |
| `OPS_NOTIFICATION_EMAIL` | Where `/pro-requests` submissions get emailed | Optional; defaults to `tech@greenboxanalytics.ca` |

---

## 🟡 Required for OAuth integrations (Google Drive, Slack, Notion)

Each integration is independently optional — the feature is hidden in the UI if its credentials are missing.

| Var | Source |
|-----|--------|
| `GOOGLE_DRIVE_CLIENT_ID`, `GOOGLE_DRIVE_CLIENT_SECRET` | [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) → OAuth 2.0 Client ID |
| `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET` | [api.slack.com/apps](https://api.slack.com/apps) → your app → Basic Information |
| `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` | [notion.so/profile/integrations](https://www.notion.so/profile/integrations) → New integration (public, OAuth) |

OAuth callback URLs to register in each console (replace host as appropriate):
- Google Drive: `{BACKEND_URL}/integrations/google-drive/callback`
- Slack: `{BACKEND_URL}/integrations/slack/callback`
- Notion: `{BACKEND_URL}/integrations/notion/callback`

OAuth tokens are encrypted at rest with the next two keys.

---

## 🟡 OAuth token + credentials encryption keys

Generate once per environment and stash in GSM. **Rotating these invalidates every stored OAuth token / credential** — users have to re-authorize all integrations.

| Var | Generate with | Purpose |
|-----|---------------|---------|
| `INTEGRATION_ENCRYPTION_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` | AES-128-CBC via Fernet for stored OAuth tokens |
| `INTEGRATION_OAUTH_STATE_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(32))"` | Signs OAuth state JWTs (CSRF protection during the OAuth flow) |
| `CREDENTIALS_AES_KEY` | `python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"` | AES-256-GCM 32-byte key for the Artist Credentials Vault (DSP login, etc.) |

---

## 🟢 Admin & paywall (operational config)

| Var | Default | Notes |
|-----|---------|-------|
| `ADMIN_EMAILS` | empty | Comma-separated email allowlist. Bootstraps admin access; additional admins managed via `/admin/users` UI. See [admin-roles.md](admin-roles.md) |
| `BYPASS_PAYWALLS` | `false` in code, `true` in `.env.example` | When `true`, all users get Pro-shaped entitlements. Keep on during private beta; flip off when paywalls go live |

---

## 🟢 PostHog analytics

| Var | Source | Notes |
|-----|--------|-------|
| `POSTHOG_ENABLED` | — | Set to `"true"` only in prod/staging. Dev/test should leave unset/false so events don't pollute the project |
| `POSTHOG_API_KEY` | PostHog → Project Settings → Project API Key | `phc_...`. Used by backend |
| `VITE_POSTHOG_API_KEY` | Same value | Used by frontend (safe to ship to browser — it's the public Project key, not the personal key) |
| `POSTHOG_HOST` | Default `https://us.i.posthog.com` | Set to `https://eu.i.posthog.com` for EU cloud, or your self-hosted URL |
| `VITE_POSTHOG_HOST` | Same | Frontend mirror |
| `VITE_POSTHOG_DASHBOARD_URL` | Default `https://us.posthog.com` | For the "View Analytics" link in `/admin/users` |

---

## 🟢 Vercel deployment (CI only — not needed for local dev)

| Var | Source | Used by |
|-----|--------|---------|
| `VERCEL_PROD_TOKEN` | Vercel → Account Settings → Tokens | GitHub Actions prod deploy workflow |
| `VERCEL_ORG_ID` | Vercel → Project Settings → General → Team ID | Same |
| `VERCEL_PROJECT_ID` | Vercel → Project Settings → General → Project ID | Same |

Stored as GitHub repo secrets, not in `.env`.

---

## ⚪ Test-only

| Var | Default | Notes |
|-----|---------|-------|
| `RUN_INTEGRATION_TESTS` | unset | Set to `1` to enable pytest markers that hit real Supabase. Local-only, not for CI/prod |

---

## Variables read by code but NOT in `.env.example`

These work but aren't documented in `.env.example` — usually because they're optional/defaulted, or used only by tests or specific code paths. Worth being aware of if you're tracing config:

| Var | Read by | Notes |
|-----|---------|-------|
| `VITE_FRONTEND_URL` | Backend (`projects/emails.py`, `users/emails.py`, `registry/emails.py`, `integrations/oauth.py`, `integrations/slack/blocks.py`) | Default `http://localhost:8080`. Distinct from `FRONTEND_URL` — both exist; one was added later and unifying them is a TODO |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Backend tests only (`test_subscription_triggers.py`) — fallback for `VITE_SUPABASE_ANON_KEY` | Safe to ignore unless you're running those tests |
| `VITE_ACCESS_CODE` | Frontend `src/components/AccessGate.tsx` | If set, gates the entire app behind a single shared access code. Used for closed beta. Leave unset to disable the gate |

---

## Where to store secrets

| Environment | Storage | How they reach the app |
|-------------|---------|------------------------|
| **Local dev** | `.env` (gitignored) — copy from `.env.example` | `python-dotenv` (backend) + Vite (frontend) read at startup |
| **Vercel (frontend)** | Vercel Project Settings → Environment Variables | Baked into the Vite build at deploy time. Only `VITE_*` vars are exposed to the browser |
| **Cloud Run (backend)** | Google Secret Manager → mounted as env vars in the Cloud Run service config (see `deploy-backend.yml`) | Runtime env. No code change needed when rotating |
| **GitHub Actions** | Repo Settings → Secrets and variables → Actions | Used by deploy workflows (Vercel tokens, dev-vs-prod allowed origins) |

**Never commit `.env`, `service-account.json`, or any file with credentials.** `.env` is in `.gitignore`. Crypto keys (`INTEGRATION_ENCRYPTION_KEY`, `CREDENTIALS_AES_KEY`, etc.) should be generated per-environment and never reused across dev/staging/prod.

---

## Quick-start: local setup from zero

```bash
# 1. Clone and copy env template
cp .env.example .env

# 2. Fill in the REQUIRED vars (the 🔴 group above):
#    - Supabase URL, anon key, service-role key, DB password
#    - VITE_BACKEND_API_URL=http://localhost:8000

# 3. Generate crypto keys (one-time):
python -c "from cryptography.fernet import Fernet; print('INTEGRATION_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
python -c "import secrets; print('INTEGRATION_OAUTH_STATE_SECRET=' + secrets.token_urlsafe(32))"
python -c "import secrets, base64; print('CREDENTIALS_AES_KEY=' + base64.b64encode(secrets.token_bytes(32)).decode())"

# 4. Add yourself as admin
#    ADMIN_EMAILS=you@example.com

# 5. (Optional) Add OPENAI_API_KEY + RESEND_API_KEY if you want Zoe/emails

# 6. (Optional) Add Stripe test keys + run `stripe listen` if testing billing
#    See docs/stripe-integration.md

# 7. Run
cd src/backend && poetry install && poetry run uvicorn main:app --port 8000
# In another terminal:
npm install && npm run dev
```

You should be able to sign in, create an artist, and see the app render. Pricing/billing, integrations, and email features each require their respective sections above.

---

## Related Docs

- [stripe-integration.md](stripe-integration.md) — full Stripe local-test flow
- [admin-roles.md](admin-roles.md) — `ADMIN_EMAILS` + DB admin grants
- [.env.example](../.env.example) — canonical template with inline comments
- Deployment workflow files: `.github/workflows/deploy-backend.yml`, Vercel project settings
