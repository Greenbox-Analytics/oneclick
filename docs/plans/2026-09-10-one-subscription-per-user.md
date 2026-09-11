# One live personal subscription per user

Date: 2026-09-10. Backend guard + webhook hardening, small frontend. No
schema or Stripe-dashboard changes (one portal-configuration check, below).

## The hole

`create_checkout_session` never read the `subscriptions` row. A Basic
subscriber clicking "Upgrade to Pro" on `/pricing` (or "Finish upgrading" on
the dashboard) got a fresh Checkout, which always creates a **new** Stripe
subscription — on a **new** Customer, because the session was created with
`customer_email` rather than `customer=`. `checkout.session.completed` then
upserted the row with the new id, and the old subscription kept billing with
nothing in the database naming it. Worse, the three other personal handlers
(`updated`, `deleted`, `payment_failed`) wrote by `user_id` with no
subscription-id check, so canceling the *duplicate* in the portal fired a
`deleted` event that freed the plan the user was actually paying for.

## Four rules

1. **Live** = the row names a `stripe_subscription_id` AND `status != canceled`.
   `active`, `trialing`, `past_due` all count — a past-due user fixes their
   card in the portal, they don't buy a second plan. Admin-granted paid tiers
   have no id, so they read as not live and may buy.
2. Checkout while live → **409** `{ code: "subscription_exists", reason, tier }`.
   Every frontend checkout caller funnels it through `useSubscriptionConflict`:
   forget the remembered plan, refetch entitlements, toast, open the Portal.
   `/pricing` doesn't even offer a second Checkout to a subscriber: their tier
   reads "Current plan", the other "Switch to …" (Portal).
3. Every personal-subscription webhook write is conditioned on the row naming
   this subscription, or naming none (`_names_other_live_subscription`).
   Names a *different, not-canceled* one → log + skip. **Permissive on null
   and on canceled** on purpose: dashboard-created subscriptions and
   out-of-order delivery (`updated` before `checkout.session.completed`)
   keep working, and a lost `deleted` webhook can't lock a row forever.
4. `checkout.session.completed` that replaces a different live subscription
   (a race past the 409, or a session created before deploy): the new one is
   the plan — the user just paid for it — and the old one is canceled at
   Stripe immediately with proration, AFTER the upsert so its `deleted` event
   hits rule 3. Logged at ERROR with both ids and customers. Cancel-not-refuse
   because refusing would leave the user paying for a plan they don't get;
   `InvalidRequestError` (already gone) is the goal state and is acked, any
   other Stripe error 500s so Stripe retries the idempotent handler.

`customer=` reuse rides along: a re-subscriber lands on the Customer that
carries their invoices, which is also where rule 4's proration credit can be
used. Pre-fix duplicates sit on two Customers, so the credit is stranded —
the ERROR line carries both customer ids for a manual refund.

## Test-double note

The id-guard's `isinstance` checks are the live-data contract (PostgREST
returns str/None) and also what keeps `test_stripe_events.py`'s bare
`MagicMock()` supabase fixtures on today's path: a MagicMock row is not a
dict and can never read as a live subscription.

## Ops: existing duplicates

Stripe Dashboard → Subscriptions, filter *Active*, sort by customer email
(each pre-fix checkout minted a new Customer with the same email). Cancel the
subscription that is **not** `subscriptions.stripe_subscription_id` — its
`deleted` event now hits the id-guard — and refund by hand. Canceling the
stored one instead frees the row while the other keeps billing.

Check once per environment: the Customer Portal configuration must list both
products' monthly and annual prices under "Customers can switch plans", or
"Switch to …" opens a portal that can only cancel.

## Follow-ups (out of scope)

- `planPeriod` (`models.py`) is derived by substring `"annual" in
  stripe_price_id`; real price ids are opaque, so every real subscriber reads
  "monthly" (PlanCard's "/ year" label, `checkout_completed` analytics).
  Compare against `STRIPE_PRICE_ANNUAL` / `STRIPE_PRICE_PRO_MAX_ANNUAL` the
  way `_tier_for_price` does.
- ~~Portal `return_url` is the legacy `/subscription` redirect~~ — closed
  later the same day (`/profile?portal=return`, see
  `2026-09-10-portal-cancel-return.md`). Still open: a
  `flow_data.subscription_update` deep-link would land "Switch to …" on the
  plan picker instead of the Portal home.
- A late `customer.subscription.updated` delivered after `.deleted` for the
  *same* id can re-elevate a freed row (pre-existing, rare).
- Still open from 2026-09-09: `/auth?redirect=/pricing&plan=…` drops `plan`;
  server-side pending plan; duplicate `checkout_started` analytics.
