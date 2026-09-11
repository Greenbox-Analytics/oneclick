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
| POST | `/billing/create-checkout-session` | Body `{ plan: "basic_monthly" \| "basic_annual" \| "pro_monthly" \| "pro_annual", cancel_path?, success_path? }` (return paths are whitelisted to relative `/...` paths). Creates a Stripe Checkout Session, returns `{ url }` to redirect. Sets `user_id` in both session and subscription metadata so the webhook can match it back. **One live personal subscription per user** (2026-09-10): if the caller's `subscriptions` row names a `stripe_subscription_id` whose `status` isn't `canceled` (`active`, `trialing` and `past_due` all count), returns **409** with `detail: { code: "subscription_exists", reason, tier }` — plan changes for a subscriber go through the Portal, and the frontend opens it on this error (`useSubscriptionConflict`). Admin-granted paid tiers (no Stripe id) may still buy. A re-subscriber with a stored `stripe_customer_id` is attached to that Customer (`customer=`) instead of Checkout minting a new one per purchase (`customer_email`). |
| POST | `/billing/create-portal-session` | Optional body `{ flow?: "subscription_cancel" }`. Creates a Stripe Customer Portal session, returns `{ url }`. No body = the Portal home, whose "Return to Msanii" link lands on `/profile?portal=return`. `subscription_cancel` (2026-09-10) deep-links into Stripe's cancel confirmation page for the user's live subscription (`flow_data`), and Stripe redirects to `/profile?portal=canceled` the moment they confirm — the only Portal surface that redirects on its own; backing out lands on the home return URL. Returns 404 if the user has no `stripe_customer_id` (e.g., admin-granted Pro users); for the cancel flow, **409** `{ code: "nothing_to_cancel" \| "already_canceling", reason }` when there is no live subscription or it is already set to end — decided against **Stripe, not just the row**: the endpoint reads the subscription first (the row lags its `.updated` webhook, or misses it when the listener is down, and Stripe refuses a second cancel flow for a subscription already set to cancel), and writes a cancel Stripe already holds onto the row (`stripe_events.sync_scheduled_cancel`) so the refetch the 409 triggers flips the card to "Ends"; and **502** `{ code: "portal_flow_unavailable" }` when Stripe refuses the flow (a Portal configuration with cancellations off) or can't read the subscription — the frontend then falls back to the Portal home. |
| POST | `/billing/sync-subscription` | No body. Mirrors the user's live subscription from Stripe onto their `subscriptions` row — `cancel_at_period_end`, `current_period_start/end`, `canceled_at` (`stripe_events.sync_period_from_stripe`; never status, tier or credits, which stay with the webhook handlers) — conditioned on the row still naming that subscription. Called by `usePortalReturn` on every Portal return so the plan card flips within one round trip instead of waiting on the `.updated` webhook (which lands later and writes the same truth). Returns `{ synced: false }` when the row names nothing live or Stripe reports the subscription canceled (the `deleted` handler's job); **502** `{ code: "stripe_unavailable" }` when Stripe can't be read — the caller then waits on the webhook. |
| POST | `/billing/webhook` | Stripe webhook receiver. Verifies signature, dedupes via `stripe_events` table, dispatches to handler. |

### Webhook event handlers (`src/backend/subscriptions/stripe_events.py`)

| Stripe event | Handler | What it does |
|--------------|---------|--------------|
| `checkout.session.completed` | `handle_checkout_session_completed` | Upserts `subscriptions` row with the tier resolved from the price, `stripe_customer_id`, `stripe_subscription_id`, period info. If the row already named a *different* live subscription (a race past the endpoint's 409, or a session created before it shipped), the new one wins — the user paid for it — and the old one is **canceled at Stripe immediately, prorated** (`Subscription.cancel(prorate=True)`), logged at ERROR with both ids and customers. Stripe refusing the cancel (already gone) is logged and acked; any other Stripe error 500s so Stripe retries |
| `customer.subscription.updated` | `handle_subscription_updated` | Syncs status / period / cancel-at-period-end / price — and tier, when `CREDITS_ENABLED` is on (portal plan switches only surface here; an upgrade tops the bundle up). The stored `cancel_at_period_end` is `_scheduled_to_end(sub)`: Stripe's *flexible* billing mode reports a Portal cancel as `cancel_at` alone (classic, the pinned API version, sets the flag), so either shape sets the column the profile's "Ends <date>" keys on. **Ignored when the row names a different live subscription** (id-guard, below) |
| `customer.subscription.deleted` | `handle_subscription_deleted` | Sets `tier='free'`, `status='canceled'`; keeps `stripe_customer_id` for re-subscribe convenience. **Ignored when the row names a different live subscription** — canceling a duplicate must not free the plan the user actually pays for |
| `invoice.payment_failed` | `handle_invoice_payment_failed` | Sets `status='past_due'`. Tier stays paid during Stripe's automatic retries; if retries exhaust, `.deleted` fires. Same id-guard: a failed charge on a duplicate never marks the real plan past due |
| `invoice.payment_succeeded` | **intentionally NOT handled** | The parallel `customer.subscription.updated` event carries the same period info; handling both creates redundant writes |

Unknown event types are acked (200, `{ handled: false }`) so Stripe stops retrying.

**Id-guards (2026-09-10).** Every personal-subscription handler compares the event's subscription id with the row's `stripe_subscription_id` (`_names_other_live_subscription`). A row naming a *different, not-canceled* subscription means the event belongs to a duplicate or an orphan: it is logged and skipped. A row naming nothing, or naming a canceled subscription, stays permissive on purpose, so dashboard-created subscriptions and out-of-order delivery (`.updated` before `checkout.session.completed`) keep working as before.

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
| `useCreateCheckoutSession()` | Mutation: `(plan \| { plan, cancel_path?, success_path? }) => Promise<url>` | "Upgrade" button → call `mutateAsync("basic_monthly")` → `window.location.href = url` |
| `useCheckoutReturn()` | `{ activating }` | Handles the `?welcome=true&stripe_session_id=` success return on `/profile`: strips the URL, polls entitlements until the webhook lands (10s cap), drives the "Activating your subscription…" overlay, toasts either way. |
| `useTopupReturn()` | — | Same for credit purchases (`?topup=success\|canceled`) on `/profile` and `/teams`. |
| `useCreatePortalSession()` | Mutation: `({ flow? }?) => Promise<url>` | The Portal home (no args) or the cancel flow (`{ flow: "subscription_cancel" }`). Throws ApiError(404) when the user has no Stripe customer; 409/502 for the cancel flow (see the endpoint table). |
| `useOpenBillingPortal()` | `{ openPortal, isPending }` | "Manage subscription": opens the Portal home, toasts on 404. |
| `useOpenCancelFlow()` | `{ openCancelFlow, isPending }` | "Cancel plan" on the Billing plan card: opens the Portal's cancel flow. A 409 means the card was stale (refetch + explain, no portal); any other refusal toasts and falls back to the Portal home, where "Cancel plan" still exists. |
| `useSyncSubscription()` | mutation → `{ synced }` | `POST /billing/sync-subscription`: mirror the user's live subscription from Stripe onto their row (cancel flag, period). Called by `usePortalReturn` on every Portal return. |
| `usePortalReturn()` | `{ syncing }` | Handles `?portal=canceled\|return` on `/profile`. Both: strip the URL, call `useSyncSubscription` and refetch entitlements when it resolves — the card is right within one round trip, no waiting on the webhook. `canceled`: drives the "Updating your plan…" overlay until the read says `cancelAtPeriodEnd` (or turns free — a Portal configured to cancel immediately) and toasts the end date; a 1s poll underneath is the fallback if the sync fails (10s cap, then an honest "will show shortly"). `return`: no overlay or toast — nothing is known about what changed on the Portal home; a silent 5s refetch burst as the fallback. Poll ticks pass `cancelRefetch: false` (the `invalidateQueries` default restarts an in-flight fetch, so a fetch slower than the tick could never land while polling; `useCheckoutReturn` does the same). Same latch-once idiom as `useCheckoutReturn`. |
| `useSubscriptionConflict()` | `(err) => Promise<boolean>` | Shared recovery for the endpoint's 409 in every checkout caller (Pricing, Onboarding, the dashboard's resume strip, Billing's plan card): forgets the remembered plan, refetches entitlements, toasts, opens the Portal. `isSubscriptionConflict(err)` is the predicate; `hasLiveSubscription(ent)` in `@/lib/tiers` is the client-side mirror of the server's rule. |

### Pages

| File | Route | Role |
|------|-------|------|
| `src/pages/Pricing.tsx` | `/pricing` | Plan comparison; "Upgrade to Basic/Pro" calls `useCreateCheckoutSession`. Reads `?canceled=true` (toast + strip). A live subscriber sees "Current plan" on their tier and "Switch to …" on the other, which opens the Customer Portal — never a second Checkout. |
| `src/pages/Onboarding.tsx` | `/onboarding` | Plan step during signup; sends `cancel_path=/onboarding?upgrade=cancelled` so a cancelled checkout resumes on the plan step |
| `src/pages/Profile.tsx` | `/profile` | Account & Billing — the post-checkout return page and the Portal return page (`?portal=canceled\|return`). `PlanCard` shows the plan: "Renews <date>", or "Ends <date> · N days left" once a cancel is scheduled (`fmtDaysLeft` in `@/lib/utils`), with "Manage subscription" (Portal home) and "Cancel plan" (Portal cancel flow; gone once the cancel is scheduled — reactivating is "Renew plan" on the Portal home). `/subscription` is a legacy redirect here that keeps the query string. |

After Checkout, Stripe redirects to `${FRONTEND_URL}/profile?stripe_session_id=...&welcome=true`. Cancel redirects to `${FRONTEND_URL}/pricing?canceled=true` by default, or to the caller's `cancel_path`.

**Abandoned checkouts are resumable.** Both entry points remember the chosen plan per user in localStorage for 7 days (`src/lib/pendingPlan.ts`). While entitlements still read Free, the dashboard strip (`UpgradeBanner`) and the Billing `PlanCard` offer "Finish upgrading to …", which just creates a fresh Checkout Session. The memory is cleared the moment the success URL is reached (so a slow webhook can never lead to a second checkout), on an explicit Free choice, on the strip's X, and whenever entitlements read paid. It never grants anything — entitlements always come from `/me/entitlements`.

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
4. Complete checkout. You're redirected to `/profile?stripe_session_id=...&welcome=true`; the URL is cleaned immediately, an "Activating your subscription…" overlay shows until the webhook lands (at most 10s), then a "Welcome to Basic!" toast.
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

After the user has a Stripe subscription, hit "Manage subscription" on `/profile`. Stripe opens the Portal home. You can:
- Switch plans (fires `customer.subscription.updated`)
- Cancel at period end (fires `customer.subscription.updated` with `cancel_at_period_end=true`, then `.deleted` when the period ends — or use `stripe trigger` to fast-forward)
- Update payment method

"Return to Msanii" lands on `/profile?portal=return`, where `usePortalReturn` mirrors the subscription from Stripe onto the row (`POST /billing/sync-subscription`) and refetches entitlements, so a cancel or "Renew plan" made on the Portal home shows on the plan card within a round trip — the webhook lands later and writes the same truth. A short refetch burst underneath covers a sync that fails.

**"Cancel plan" on `/profile` (2026-09-10)** skips the Portal home: `create-portal-session` with `{ flow: "subscription_cancel" }` opens Stripe's cancel confirmation page with the navigation hidden, and Stripe redirects to `/profile?portal=canceled` the moment the user confirms. Only a Portal *flow* can do that — the home always needs a "Return" click; backing out of the flow lands on the home return URL. The profile shows an "Updating your plan…" overlay while `usePortalReturn` mirrors the subscription from Stripe onto the row (`POST /billing/sync-subscription`) and refetches — about a second, no waiting on the `.updated` webhook, which lands later and writes the same truth — then "Ends <date> · N days left" on the plan card, the amber banner, and a toast. If the sync fails the hook polls for the webhook instead (10s cap, then an honest "will show here shortly"); locally that fallback needs `stripe listen` (step 3). The button disappears once the cancel is scheduled; reactivating is "Renew plan" on the Portal home. What the flow does follows the **Portal configuration** (Dashboard → Settings → Billing → Customer portal → *Cancellations*, separate for test and live mode): cancellations must be **enabled**, and "Ends <date>" assumes **"Cancel at end of billing period"** — under "Cancel immediately" the `.deleted` webhook drops the user to Free on the spot (the hook handles that too). Stripe doesn't document what a configuration with cancellations off does to the flow; the endpoint maps any Stripe refusal to a 502 and the button falls back to the Portal home. One refusal is known from the first test run (2026-09-10): a subscription already set to cancel gets `The subscription sub_… is already set to be canceled at period end.` — which is why the endpoint reads the subscription from Stripe before deep-linking, rather than trusting the row's `cancel_at_period_end`.

Portal access is gated on having a `stripe_customer_id` in `subscriptions`. Admin-granted Pro users (tier overrides) won't have one and get a 404 — that's intentional.

`/pricing` sends subscribers here too: "Switch to Pro/Basic" opens the Portal home, where "Update plan" does the switch. So the Portal configuration (Dashboard → Settings → Billing → Customer portal) must list both products' monthly and annual prices under *Subscriptions → Customers can switch plans* — otherwise the button opens a portal that can only cancel. A subscriber can't start a second Checkout at all: `create-checkout-session` refuses with a 409 (see the endpoint table).

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
- [ ] `FRONTEND_URL=https://www.msanii-beta.com` (the live domain — `localhost` will break Stripe redirects). **No trailing slash**: `success_url` is built as `f"{FRONTEND_URL}{success_path}"`, so a slash yields `//profile?...` and React Router won't match the route — the user lands on a blank page after paying
- [ ] `BYPASS_PAYWALLS` is `false` (or unset) — setting it to `true` would give every user Pro-shaped entitlements regardless of Stripe status
- [ ] Live webhook endpoint receives a test event from Dashboard ("Send test webhook") and returns 200
- [ ] Live-mode Customer Portal configuration lists both products' monthly + annual prices under "Customers can switch plans" (the `/pricing` "Switch to …" path lands on the Portal home)
- [ ] Live-mode Customer Portal configuration has Cancellations **enabled** with "Cancel at end of billing period" (the `/profile` "Cancel plan" flow and its "Ends <date>" copy assume it)
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
| Upgrade returns 409 `subscription_exists` | The user already holds a live personal subscription (`active` / `trialing` / `past_due`) | Expected — plan changes go through the Portal ("Manage subscription" on `/profile`, "Switch to …" on `/pricing`). A `canceled` row whose stale id is still set is NOT live and may buy again |
| A user is billed for two personal subscriptions | Checkouts before 2026-09-10 could create a second subscription on a second Customer (`customer_email`), and the webhook overwrote the stored id | Ops: Dashboard → Subscriptions, filter *Active*, sort by customer email — each pre-fix checkout minted a new Customer with the same email. Cancel the subscription that is **not** `subscriptions.stripe_subscription_id` (its `deleted` event hits the id-guard and leaves the row alone) and refund by hand. Canceling the stored one instead frees the row while the other keeps billing |
| "Cancel plan" toasts "opening your billing portal instead" | Stripe refused the `subscription_cancel` flow, or the subscription couldn't be read — most likely Cancellations are off in the Portal configuration for that mode (backend logs `Portal subscription_cancel flow refused` / `could not read`; endpoint 502 `portal_flow_unavailable`). A subscription already set to cancel is NOT this case any more: the endpoint checks Stripe first and 409s | Read the Stripe error in the backend log; enable Cancellations in the Portal configuration (test and live are separate); the Portal home's own "Cancel plan" is the fallback meanwhile |
| "Cancel plan" toasts "already set to end" while the card still said "Renews", then the card flips to "Ends" | The row was behind Stripe (a cancel made on the Portal home or in the Dashboard whose webhook hadn't landed). The click read Stripe and healed the row | Nothing to repair. If it keeps happening, webhooks aren't landing: check `stripe listen` (step 3) / the endpoint secret |
| After a Portal return the card keeps the old "Renews"/"Ends" until a manual reload | `POST /billing/sync-subscription` failed on arrival (backend log `sync-subscription: Stripe could not read`, 502 `stripe_unavailable`) AND the webhook didn't land within the fallback poll (10s after a cancel, 5s after a plain return) | Fix Stripe reachability from the backend; the row shows once the webhook lands (next refetch: 60s `staleTime`, or a reload) |
| Plan card reads "Ends <date>" with the period end, but the Dashboard cancel was "at a custom date" | The row stores only `cancel_at_period_end`; the date shown is `current_period_end` | Expected for now — `cancel_at` isn't stored (follow-up in `docs/plans/2026-09-10-portal-cancel-return.md`) |
| `stripe trigger` events don't update DB | The triggered event's `metadata.user_id` is empty (CLI defaults) | Use a real Checkout flow OR pass `--add subscription:metadata[user_id]=<uid>` |
| Webhook handler raises, Stripe keeps retrying forever | The idempotency-row cleanup on failure succeeded, so each retry re-attempts | Check backend logs; fix the handler; the next retry will succeed |

---

## Related Files

- Backend: `src/backend/subscriptions/{billing_router.py,stripe_client.py,stripe_events.py}`
- Frontend: `src/hooks/{useBilling.ts,useCheckoutReturn.ts,usePortalReturn.ts,useTopupReturn.ts}`, `src/lib/pendingPlan.ts`, `src/pages/{Pricing.tsx,Onboarding.tsx,Profile.tsx}`, `src/components/billing/{PlanCard.tsx,UpgradeBanner.tsx}`
- DB: `supabase/migrations/20260509000001_subscription_foundation.sql`, `20260515_stripe_events.sql`
- Tests: `src/backend/tests/test_billing_router.py`, `test_stripe_client.py`, `test_stripe_events.py`
- Env vars: see [secrets.md](secrets.md) for the full list with sources
