# Portal cancel: back on `/profile`, reading "Ends <date> · N days left"

Date: 2026-09-10. Small backend change (one endpoint, one webhook flag), small
frontend. No schema change. One Stripe-dashboard requirement (below).

## The ask, and the Stripe constraint

Cancel on the Stripe page → automatically back on the profile page → the
plan card reads "Ends <date>" with the days remaining, instead of "Renews".

The Customer Portal *home* never redirects on its own: after a cancel there
the user sits on Stripe's own "your plan will be canceled on …" banner until
they click "Return to Msanii". An automatic redirect exists only for a Portal
**flow** (`flow_data.type = subscription_cancel`): Stripe's cancel
confirmation page with the navigation hidden, which with
`after_completion.type = redirect` sends the customer to our URL the moment
they confirm. Backing out of the flow goes to the session's `return_url`.
(Stripe docs: "The top level `return_url` is a link back to your website that
the customer can click at any time (if they decide not to cancel, for
example). The `flow_data[after_completion][redirect][return_url]` is a link
back to your website after a customer successfully cancels.")

So `/profile` gained a **"Cancel plan"** button that opens that flow. It puts
cancelling one click closer than before; the Portal home's own "Cancel plan"
was always there (cancellations are on by default in the Portal
configuration), so this moves *where* the confirmation lives, not *whether*
it exists. Reaching `?portal=canceled` is therefore proof the cancel went
through — only the webhook can still be in flight.

## Two signals, one hook

`create-portal-session` takes an optional `{ flow: "subscription_cancel" }`.
Both sessions return to `/profile`, and both returns do the same first thing
(added the same evening, after the first live run — see "Why the row is
mirrored" under Guards): `POST /billing/sync-subscription` reads the
subscription from Stripe and writes its period and cancel fields onto the
row, then entitlements are refetched. The card is right within one round
trip; the `.updated` webhook lands later and writes the same truth.

- `?portal=canceled` — the flow's `after_completion` redirect. `usePortalReturn`
  strips the URL and shows "Updating your plan…" until the read says
  `cancelAtPeriodEnd` (toast: "Your Basic plan is set to end on Oct 10,
  2026. You keep full access until then."), or turns free — a Portal
  configured to cancel *immediately* — or 10s pass (toast: the cancellation
  went through, the date will show shortly). A 1s poll underneath is the
  fallback for a failed sync: it observes the webhook. A missing or
  `degraded` read never settles: without a trustworthy subscription it would
  look like an immediate cancel before the first fetch resolves.
- `?portal=return` — the Portal home's "Return to Msanii" link (also where an
  abandoned cancel flow lands, and where Pricing's "Switch to …" and the org
  panel's portal button now return). Nothing is known about what changed
  there, so: the sync, then a silent five-tick refetch burst as the fallback,
  no overlay, no toast.

Poll ticks pass `cancelRefetch: false`. `invalidateQueries` otherwise cancels
an in-flight fetch and starts over (`query.fetch` in query-core: data present
+ `cancelRefetch` → `cancel({ silent: true })`), so a fetch slower than the 1s
tick — a local backend six Supabase round trips away sits right at that
edge — could never land while polling; the poll would starve itself and the
page would look frozen until the ticks stopped. `useCheckoutReturn` got the
same fix.

Same latch-once idiom as `useCheckoutReturn` (see
`2026-09-09-checkout-return-and-resume.md`): the signal is frozen in
`useState`, `syncing` goes true in exactly one place immediately before a
timeout is armed, settle is idempotent through a never-reset ref, and the
sync fires once per return through another never-reset ref (StrictMode runs
effects twice).

## The row

`PlanCard`: "Renews <date>" → "Ends <date> · 12 days left" once
`cancelAtPeriodEnd` is set (`fmtDaysLeft` counts *calendar* days in the
viewer's zone, like the date beside it; "today" on the last day, nothing once
it has passed). "Cancel plan" hides once the cancel is scheduled — reactivating
is "Renew plan" on the Portal home, behind "Manage subscription". The profile's
amber banner says the same thing in a sentence.

## Guards

- Backend 409 `nothing_to_cancel` unless the row names a live subscription
  (same rule as `create-checkout-session`: an id AND `status != canceled`);
  409 `already_canceling` when `cancel_at_period_end` is already set. The
  frontend treats a 409 as a stale card: refetch, explain, no portal.
- **Stripe decides, not the row** (added after the first test run, same day).
  The row's flag lags its `.updated` webhook — or never gets it when the
  listener is down, which is how the first click failed: the cancel had gone
  through at Stripe, the row still said "renews", and Stripe refused a second
  flow with `The subscription sub_… is already set to be canceled at period
  end.` — a 502 and the fallback toast, with the card still offering "Cancel
  plan". So before deep-linking the endpoint retrieves the subscription:
  Stripe says canceled → 409 `nothing_to_cancel` (freeing the plan stays the
  `deleted` handler's job; a warning names the row); Stripe says scheduled →
  `sync_scheduled_cancel` writes the cancel fields the `.updated` handler
  would (`cancel_at_period_end`, `current_period_end`, `canceled_at`),
  conditioned on the row still naming that subscription, then 409
  `already_canceling` — so the refetch the 409 triggers flips the card to
  "Ends". Unreadable → 502. One Stripe read per click, on a rare action.
- **Why the row is mirrored on every return too.** The second live run, with
  the listener up, still needed a manual reload to see a cancel or a "Renew
  plan": the return waited on the webhook inside a short fixed window (10s /
  5s), and its 1s poll could cancel its own fetches (above). So the mirror
  moved into `POST /billing/sync-subscription` (`sync_period_from_stripe`,
  the same helper the cancel deep-link uses), which every return calls
  before anything else. It writes `cancel_at_period_end`,
  `current_period_start/end`, `canceled_at` — never status, tier or credits,
  which stay with the webhook handlers — conditioned on the row still naming
  that subscription, so it converges with the webhook whichever lands first.
  Stripe reporting the subscription canceled is left to the `deleted`
  handler (`{ synced: false }`); Stripe unreadable is a 502 and the hook
  falls back to polling for the webhook.
- Stripe doesn't document what a Portal configuration with cancellations
  *off* does to the flow. Any other Stripe refusal is a 502
  `portal_flow_unavailable`, logged, and the button falls back to the Portal
  home — the feature degrades, it never dead-ends.
- Locally, `stripe listen` still matters for everything the mirror doesn't
  cover (status, tier, credits, the `deleted` event) and as the fallback
  when the sync fails — the documented dev setup
  (`docs/stripe-integration.md` §3).
- `handle_subscription_updated` stores `_scheduled_to_end(sub)`: Stripe's
  *flexible* billing mode reports a Portal cancel as `cancel_at` alone with
  `cancel_at_period_end=false`. The pinned API version (`2024-06-20`) is
  classic, so this is insurance against a bump — the whole feature keys on
  that one column.

## Check once per environment

Dashboard → Settings → Billing → Customer portal → Cancellations: **enabled**,
**"Cancel at end of billing period"**. Test and live configurations are
separate. Added to the prod cutover checklist in `docs/stripe-integration.md`.

## Follow-ups (out of scope)

- ~~A backend "sync my subscription from Stripe" step on arrival~~ — done the
  same evening (`POST /billing/sync-subscription`, above). `useCheckoutReturn`
  still waits on `checkout.session.completed`: there is no row to mirror
  before it lands.
- Store `cancel_at`: a Dashboard cancel "at a custom date" shows
  `current_period_end` as the end date today.
- A `subscription_update` deep-link for `/pricing`'s "Switch to …" (same
  `flow_data` mechanism; lands on the plan picker instead of the Portal home).
- A per-caller portal `return_path` whitelist (the org panel's portal button
  returns to `/profile`, as it always has).
- Still open from 2026-09-09/10: `planPeriod` substring derivation, a late
  `.updated` after `.deleted`, `/auth?redirect=…&plan=` dropping `plan`,
  server-side pending plan, duplicate `checkout_started`.
