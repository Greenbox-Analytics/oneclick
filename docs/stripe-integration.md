# Stripe Integration

How subscription billing is wired up — Checkout, Customer Portal, webhooks — and how to run it end-to-end locally.

Stripe powers two Pro plans (monthly + annual). Free is the default tier; Pro unlocks higher caps and full feature access. The backend never trusts client-supplied tier data — Pro status flows from Stripe → webhook → `subscriptions` table → entitlements.

---

## Architecture

```
┌─ Pricing.tsx ─┐                  ┌──────── Stripe ────────┐
│ "Subscribe"   │ ─POST checkout─► │ Checkout Session       │
└───────────────┘                  └──────────┬─────────────┘
                                              │ user pays
                                              ▼
                                   ┌──────────────────────────┐
                                   │ webhook → /billing/webhook│
                                   └──────────┬───────────────┘
                                              │ verify sig + idempotency
                                              ▼
                              ┌─────────────────────────────────┐
                              │ subscriptions table             │
                              │ (tier=pro, status, period_end…) │
                              └─────────────────────────────────┘
                                              │
                                              ▼
                              /me/entitlements reads → frontend gates
```

---

## Backend

### Endpoints (`src/backend/subscriptions/billing_router.py`, prefix `/billing`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/billing/create-checkout-session` | Body `{ plan: "monthly" \| "annual" }`. Creates a Stripe Checkout Session, returns `{ url }` to redirect. Sets `user_id` in both session and subscription metadata so the webhook can match it back. |
| POST | `/billing/create-portal-session` | Creates a Stripe Customer Portal session, returns `{ url }`. Returns 404 if the user has no `stripe_customer_id` (e.g., admin-granted Pro users). |
| POST | `/billing/webhook` | Stripe webhook receiver. Verifies signature, dedupes via `stripe_events` table, dispatches to handler. |

### Webhook event handlers (`src/backend/subscriptions/stripe_events.py`)

| Stripe event | Handler | What it does |
|--------------|---------|--------------|
| `checkout.session.completed` | `handle_checkout_session_completed` | Upserts `subscriptions` row with `tier='pro'`, `stripe_customer_id`, `stripe_subscription_id`, period info |
| `customer.subscription.updated` | `handle_subscription_updated` | Syncs status / period / cancel-at-period-end / price. **Does NOT change tier** — only `.deleted` drops to free |
| `customer.subscription.deleted` | `handle_subscription_deleted` | Sets `tier='free'`, `status='canceled'`; keeps `stripe_customer_id` for re-subscribe convenience |
| `invoice.payment_failed` | `handle_invoice_payment_failed` | Sets `status='past_due'`. Tier stays `pro` during Stripe's automatic retries; if retries exhaust, `.deleted` fires |
| `invoice.payment_succeeded` | **intentionally NOT handled** | The parallel `customer.subscription.updated` event carries the same period info; handling both creates redundant writes |

Unknown event types are acked (200, `{ handled: false }`) so Stripe stops retrying.

### Webhook flow safety

1. **Signature verification** (`stripe.Webhook.construct_event`) — 400 on failure, no DB write.
2. **Idempotency** — `INSERT INTO stripe_events (event_id, ...)` before handling. Duplicate event → 200 with `{ duplicate: true }`. Stripe retries are safe.
3. **Handler failure recovery** — if a handler raises, the idempotency row is deleted (best-effort) so Stripe will retry the event. Returns 500.
4. **Manual idempotency override** — to force-replay a webhook event, delete its row from `stripe_events` and replay from Stripe Dashboard → Developers → Events → "Resend".

### Stripe SDK setup (`src/backend/subscriptions/stripe_client.py`)

- Singleton lazy-init: `stripe.api_key` set from `STRIPE_SECRET_KEY` on first `get_stripe()` call.
- Pinned API version: `2024-06-20`. Don't bump without testing event payloads.
- Library: `stripe = "^11.0"` (Python SDK).

### Database

| Table / column | Purpose |
|----------------|---------|
| `subscriptions` | One row per user. `tier`, `status`, `stripe_customer_id`, `stripe_subscription_id`, `stripe_price_id`, `current_period_start/end`, `cancel_at_period_end`, `canceled_at`. Created in `20260509000001_subscription_foundation.sql`. |
| `stripe_events` | Webhook idempotency. PK is `event_id` (Stripe's `evt_...`). Migration: `20260515_stripe_events.sql`. Old rows are debug-only; can be purged. |

---

## Frontend

### Hooks (`src/hooks/useBilling.ts`)

| Hook | Returns | Use |
|------|---------|-----|
| `useCreateCheckoutSession()` | Mutation: `(plan) => Promise<url>` | "Subscribe" button → call `mutateAsync("monthly")` → `window.location.href = url` |
| `useCreatePortalSession()` | Mutation: `() => Promise<url>` | "Manage billing" button. Throws ApiError(404) when user has no Stripe customer. |

### Pages

| File | Route | Role |
|------|-------|------|
| `src/pages/Pricing.tsx` | `/pricing` | Two-plan comparison; "Subscribe" calls `useCreateCheckoutSession` |
| `src/pages/Subscription.tsx` | `/subscription` | Post-checkout return page; shows current plan, opens Portal |

After Checkout, Stripe redirects to `${FRONTEND_URL}/subscription?stripe_session_id=...&welcome=true`. Cancel redirects to `${FRONTEND_URL}/pricing?canceled=true`.

---

## Local Testing

You need: (1) Stripe test-mode keys, (2) the Stripe CLI for webhook forwarding, (3) local backend + frontend running, (4) a Stripe-linked test card.

### 1. Get Stripe test secrets

In the Stripe Dashboard, toggle "View test data" (top-left). Then:

| Secret | Where | Format |
|--------|-------|--------|
| `STRIPE_SECRET_KEY` | Developers → API keys → "Secret key" (Reveal) | `sk_test_...` |
| `STRIPE_PRICE_MONTHLY` | Products → create a recurring price → copy the price ID | `price_...` |
| `STRIPE_PRICE_ANNUAL` | Same, second product or second price on same product | `price_...` |
| `STRIPE_WEBHOOK_SECRET` | See step 3 below — comes from `stripe listen` | `whsec_...` |

Add them to `.env`. **Test-mode prices and secrets are separate from live-mode** — keep them straight.

### 2. Install the Stripe CLI

```
brew install stripe/stripe-cli/stripe
stripe login
```

`stripe login` opens a browser and pairs the CLI with your Stripe account (uses your dashboard session — no API keys needed for the CLI itself).

### 3. Start webhook forwarding

In a dedicated terminal (leave it running):

```
stripe listen --forward-to http://localhost:8000/billing/webhook
```

It prints a `whsec_...` signing secret on startup. **Copy it into `.env` as `STRIPE_WEBHOOK_SECRET` and restart the backend.** This secret is specific to the CLI session — it changes if you stop and restart `stripe listen`. Production has its own permanent `whsec_...` from the Dashboard webhook endpoint.

### 4. Run the app

```
# Terminal 1 — backend (already in src/backend/)
poetry run uvicorn main:app --port 8000

# Terminal 2 — frontend (already at repo root)
npm run dev

# Terminal 3 — stripe listen (from step 3)
```

### 5. End-to-end test

1. Sign in at `http://localhost:8080`.
2. Go to `/pricing`. Click "Subscribe" on Monthly.
3. Stripe Checkout opens. Use test card:
   - **Success:** `4242 4242 4242 4242`, any future expiry, any CVC, any ZIP
   - **Auth required (3DS):** `4000 0025 0000 3155`
   - **Decline:** `4000 0000 0000 9995`
4. Complete checkout. You're redirected to `/subscription?stripe_session_id=...&welcome=true`.
5. In the `stripe listen` terminal, you should see `--> checkout.session.completed` and `customer.subscription.created`/`.updated` events forwarded.
6. Check the DB: `SELECT * FROM subscriptions WHERE user_id = '<your-uid>'` — `tier` should be `pro`, `stripe_*` fields populated.
7. Reload the app. `/me/entitlements` should now return Pro caps + features.

### 6. Trigger specific events for testing

Useful when you don't want to run a full checkout:

```
# Simulate a failed renewal
stripe trigger invoice.payment_failed

# Simulate a cancellation
stripe trigger customer.subscription.deleted
```

These fire against the most recent test subscription. Useful for exercising the status-transition handlers without running ten checkouts.

### 7. Test the Customer Portal

After the user has a Stripe subscription, hit "Manage billing" on `/subscription`. Stripe opens the Portal. You can:
- Switch plans (fires `customer.subscription.updated`)
- Cancel at period end (fires `customer.subscription.updated` with `cancel_at_period_end=true`, then `.deleted` when the period ends — or use `stripe trigger` to fast-forward)
- Update payment method

Portal access is gated on having a `stripe_customer_id` in `subscriptions`. Admin-granted Pro users (tier overrides) won't have one and get a 404 — that's intentional.

---

## Production

### Webhook endpoint setup (one-time per environment)

1. Stripe Dashboard → toggle to **Live mode**.
2. Developers → Webhooks → "Add endpoint".
3. URL: `https://<your-backend-host>/billing/webhook`
4. Events to send (must match `HANDLERS` in `stripe_events.py`):
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_failed`
5. Save. Reveal the signing secret (`whsec_...`).
6. Store it in GSM as `STRIPE_WEBHOOK_SECRET` for the production environment. **It is different from the dev/test webhook secret.**
7. Replace test-mode `STRIPE_SECRET_KEY` with `sk_live_...` in production GSM.
8. Re-create products + prices in live mode (test-mode IDs do NOT work with `sk_live_`). Update `STRIPE_PRICE_MONTHLY` and `STRIPE_PRICE_ANNUAL` in production GSM.

### Prod cutover checklist

- [ ] Live-mode `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_*` all set in GSM
- [ ] `FRONTEND_URL=https://app.msanii.com` (or the live domain — `localhost` will break Stripe redirects)
- [ ] `BYPASS_PAYWALLS` is `false` (or unset) — setting it to `true` would give every user Pro-shaped entitlements regardless of Stripe status
- [ ] Live webhook endpoint receives a test event from Dashboard ("Send test webhook") and returns 200
- [ ] Run one real `$0.50` Pro signup with your own card, then refund + cancel — sanity check the round-trip

### Refunds / disputes

Handle in Stripe Dashboard. The current handler set does NOT process `charge.refunded` — the user keeps Pro access until their period ends (or you manually downgrade them via `/admin/users` → Revoke Pro). If you need automatic Pro-on-refund handling, add a handler.

---

## Beta-Period Behavior

`BYPASS_PAYWALLS=true` (opt-in via your local `.env`) makes every authenticated user receive Pro-shaped entitlements regardless of their `subscriptions.tier`. Stripe still works — checkout completes, webhooks update the DB — but no UI gates fire because the bypass short-circuits in `EntitlementsService.get_for_user`. Useful for demoing the Pro UX without paying; never set in production. Default is `false` everywhere.

Admin users (env or DB) always get Pro entitlements at `/me/entitlements` even when `BYPASS_PAYWALLS=false`. See [admin-roles.md](admin-roles.md).

---

## Common Gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| Webhook returns 400 "Invalid signature" | `STRIPE_WEBHOOK_SECRET` doesn't match the source (CLI session ≠ Dashboard endpoint) | Re-copy the secret from `stripe listen` output AND restart the backend |
| Checkout completes but `subscriptions` row unchanged | `stripe listen` not running, OR backend can't reach Stripe (firewall) | Check the CLI terminal for `--> checkout.session.completed` events; check backend logs |
| Checkout redirect goes to `localhost:8080` in prod | `FRONTEND_URL` not set in prod GSM | Set it; redeploy |
| "No Stripe subscription on file" 404 on Portal click | User is admin-granted Pro (no Stripe customer) | Expected — admin-granted Pro users manage tier via `/admin/users`, not Portal |
| `stripe trigger` events don't update DB | The triggered event's `metadata.user_id` is empty (CLI defaults) | Use a real Checkout flow OR pass `--add subscription:metadata[user_id]=<uid>` |
| Webhook handler raises, Stripe keeps retrying forever | The idempotency-row cleanup on failure succeeded, so each retry re-attempts | Check backend logs; fix the handler; the next retry will succeed |

---

## Related Files

- Backend: `src/backend/subscriptions/{billing_router.py,stripe_client.py,stripe_events.py}`
- Frontend: `src/hooks/useBilling.ts`, `src/pages/{Pricing.tsx,Subscription.tsx}`
- DB: `supabase/migrations/20260509000001_subscription_foundation.sql`, `20260515_stripe_events.sql`
- Tests: `src/backend/tests/test_billing_router.py`, `test_stripe_client.py`, `test_stripe_events.py`
- Env vars: see [secrets.md](secrets.md) for the full list with sources
