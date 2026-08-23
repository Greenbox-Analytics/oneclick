# Testing Stripe payments locally

How to exercise the credit purchase flow (bundles + custom amounts, personal
wallet and org pools) against Stripe **test mode**, end to end: checkout →
webhook → `grant_credits` → balance visible in the UI.

Everything here is test mode. If your `STRIPE_SECRET_KEY` starts with
`sk_live_`, stop and re-pull secrets — never run local tests with live keys.

---

## 1. One-time setup

### Secrets

```bash
gcloud auth login --update-adc
gcloud config set project msanii-484501
task setup:secrets:dev
```

That merges the Stripe test key, price IDs, and (optionally)
`STRIPE_CREDITS_PRODUCT_ID` from GSM into your `.env`, and turns on
`CREDITS_ENABLED` + `LICENSING_ENABLED` for local testing. It is additive —
anything you already have a live value for is kept.

### Stripe CLI

```bash
brew install stripe/stripe-cli/stripe
stripe login          # browser handshake into the Msanii test account
```

### Database

The credit-packs migrations must be applied (they are, on the shared DB, if
`GET /billing/credit-packs` returns four labelled packs — see §3). If not, run
`supabase/migrations/20260820000001_credit_pack_presentation.sql` and
`20260822000002_credit_packs_round_rates.sql` in order.

---

## 2. The webhook signing secret — read this, it is THE local gotcha

There are **two different signing secrets** and only one works locally:

- The GSM `STRIPE_WEBHOOK_SECRET_DEV` belongs to the **deployed** dev
  endpoint (Stripe dashboard → registered webhook).
- Events forwarded through **your** `stripe listen` are signed with **your
  CLI's own secret**, printed when it starts:
  `> Ready! ... Your webhook signing secret is whsec_...`

For local testing, `STRIPE_WEBHOOK_SECRET` in your `.env` must be **your
CLI's** `whsec_...`. If it isn't, every delivery returns `400 Invalid
signature`, the payment succeeds in Stripe, and **no credits are granted** —
this exact symptom cost us an evening, so check it first.

And the second half of the gotcha: **editing `.env` does nothing to a running
backend.** `load_dotenv()` runs once at startup and `uvicorn --reload` only
watches `.py` files. After any `.env` change, Ctrl+C the backend and start it
again.

---

## 3. Running the loop

Three terminals:

```bash
# 1 — backend
cd src/backend && poetry run uvicorn main:app --port 8000 --reload

# 2 — webhook tunnel (leave running the whole session)
stripe listen --forward-to localhost:8000/billing/webhook

# 3 — frontend
npm run dev        # http://localhost:8080
```

Sanity check before clicking anything:

```bash
curl -s localhost:8000/billing/credit-packs | python3 -m json.tool
```

Expect four packs (Starter/Creator/Studio/Label), a `custom` block, and a
`prices` block. If `packs` is empty the migrations aren't applied; if the
request 404s the backend isn't up.

Then log in at `localhost:8080` and open `/profile` — if the "Credits &
usage" card with an **Add credits** button is missing, `CREDITS_ENABLED`
isn't `true` in the environment the backend actually loaded (see §2 on
restarts).

---

## 4. Test cards

Any future expiry, any CVC, any postcode.

| Card | Behaviour |
|---|---|
| `4242 4242 4242 4242` | succeeds immediately |
| `4000 0025 0000 3155` | forces a 3-D Secure challenge |
| `4000 0000 0000 9995` | declines (insufficient funds) |

## 5. Current catalog (what the numbers should be)

| Pack | Credits | Price | Rate |
|---|---|---|---|
| Starter | 300 | $5.70 | 1.9¢/credit |
| Creator | 1,200 | $20.40 | 1.7¢/credit |
| Studio | 4,000 | $64.00 | 1.6¢/credit |
| Label | 15,000 | $232.50 | 1.55¢/credit |
| Custom | 250–100,000 | 2¢/credit flat | e.g. 1,300 cr = **$26.00 exactly** |

The custom price is computed **server-side** from the credit count — the
browser only ever sends a count. If Stripe Checkout shows any amount other
than `credits × $0.02`, that's a bug, not a rounding quirk.

---

## 6. Test matrix

Watch terminal 2 during all of these — every relevant delivery should log
`[200] POST .../billing/webhook`. A `[400]` means §2. Non-checkout events
(`charge.succeeded`, `payment_intent.*`) returning `200` with
`"handled": false` are normal — we only act on seven event types.

### A. Bundle → personal wallet
`/profile` → Add credits → **Starter** → `4242…`.
✅ Land on `/profile?topup=success`, URL self-cleans, "Credits added" toast,
reserve balance **+300** within a second or two.

### B. Custom amount → personal wallet
Same dialog, drag/type **1,300** → the button reads "Buy 1,300 credits —
$26.00" → pay.
✅ Stripe charges exactly $26.00; balance **+1,300**.

### C. Bundle → org pool
You need to be an **admin** of a live org (`/teams`). Open Buy credits from
the org console → any pack → pay.
✅ Return to `/teams?topup=success`; the **pool** balance rises (your
personal wallet does not). Toast may say "payment received — credits will
appear shortly" instead of the counted version; that's expected for pools.

### D. Custom amount → org pool
Same as C with the custom slider.
✅ Pool +N; ledger metadata carries `{"custom": true, "org_id": ...}`.

### E. Cancel
Start any checkout, click the back arrow in Stripe.
✅ Return to `?topup=canceled`, info toast, **no** ledger row, no charge.

### F. Decline
Buy with `4000 0000 0000 9995`.
✅ Checkout shows the decline; nothing completes, nothing granted.

### G. Idempotency (the money-critical one)
Take an event id from terminal 2 (a `checkout.session.completed`, `evt_...`)
and replay it:

```bash
stripe events resend evt_XXXXXXXXXXXX
```

✅ Terminal 2 shows `[200]`, but the balance does **not** move again and no
second `credit_ledger` row appears. Grants are keyed `topup:{session.id}`
with a unique index on `credit_ledger.request_id`, so redeliveries converge.
Resend twice if you like — same result.

### H. Missed-webhook recovery
Kill `stripe listen`, buy a pack (payment succeeds, no credits arrive —
expected: Stripe can't reach localhost). Restart `stripe listen`, find the
event id (`stripe events list`), and `stripe events resend evt_...`.
✅ Credits arrive. This is also how you fix it if you ever paid with the
tunnel down by accident.

---

## 7. Verifying in the database

Supabase SQL editor (shared DB — filter by **your** email):

```sql
select l.created_at, l.kind, l.delta, l.balance_after, l.request_id, l.metadata
from credit_ledger l
join credit_wallets w on w.id = l.wallet_id
join auth.users u on u.id = w.owner_id
where w.owner_type = 'user' and u.email = 'YOU@greenboxanalytics.ca'
order by l.created_at desc limit 10;
```

Expected per purchase: one row, `kind='purchase'`,
`request_id='topup:cs_test_…'`. Metadata is `{"pack_key": "...",
"price_cents": ...}` for a bundle, `{"custom": true, "credits": N,
"price_cents": ...}` for a custom amount. Purchased credits land in
`reserve_balance` (never expires), not `bundle_balance` (monthly). For org
purchases swap to `w.owner_type = 'org'` and drop the users join.

---

## 8. Pitfalls

- **Don't use `stripe trigger checkout.session.completed`.** It fabricates a
  session with none of our metadata; the handler logs
  `topup: session ... missing metadata` and correctly grants nothing. Only a
  real checkout (or a resend of one) carries the metadata fulfilment reads.
- **Dev and prod share one Supabase database.** Your test purchases create
  real ledger rows on your own account — fine, but do it on your own/test
  user, not someone's real account.
- **The `stripe listen` secret can differ per machine/login.** If webhooks
  400 on a machine where they used to work, re-check §2 before anything else.
- **A price mismatch in the webhook logs an error but still grants** — by
  design (the customer already paid). If you see
  `charged N cents ... expected M — granting anyway` in backend logs outside
  a deliberate rate-change test, flag it.
- **Admins are metered like everyone else** — being an admin doesn't skip
  the credit gate, so your test spends draw down real (test-account) credits.
  Top yourself back up from `/admin/users` if needed.
