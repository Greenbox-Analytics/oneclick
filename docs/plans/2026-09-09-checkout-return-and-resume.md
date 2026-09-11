# Checkout return flows: no stuck overlay, resumable abandoned checkouts

Date: 2026-09-09. Frontend only; no backend, schema, or Stripe changes.

## The two bugs

**"Activating your subscription…" never cleared.** `Profile.tsx` polled
entitlements in an effect keyed on `[welcome, stripeSessionId, isPaid]`. The
only `setIsPolling(false)` lived in the 10s timeout. When the webhook landed,
`isPaid` flipped, the effect's cleanup cleared that timeout, the effect re-ran
into the paid branch, toasted "Welcome to Basic!" and returned with the
overlay still up. Stripping the URL then re-keyed the effect once more. The
faster the webhook, the more reliably it hung.

`useTopupReturn` (credit purchases) had the same defect in a quieter form: its
effect was keyed on the live `?topup=` param and stripped that param first,
so React Router's `useSearchParams` (memoized on `location.search`) re-keyed
the effect and its cleanup killed the poll it had just armed. The "Credits
added" toast could never fire.

**Abandoned checkout = dashboard as Free, plan step unreachable.** Onboarding
saves `onboarding_completed = true` before redirecting to Stripe (on purpose:
a cancel must not strand the user in the wizard). The cancel URL
`/onboarding?upgrade=cancelled` therefore landed on an "already onboarded"
profile and `loadProfile` bounced to `/dashboard` over the cancel handler.
`/pricing?canceled=true`, the backend default, was read by nothing. A closed
tab hit neither URL, and nothing anywhere remembered the chosen plan.

## The idiom: latch the return signal once

```ts
const [sessionId] = useState(() =>
  searchParams.get("welcome") === "true" ? searchParams.get("stripe_session_id") : null,
);
useEffect(() => { if (!sessionId) return; /* strip params, then poll */ }, [sessionId]);
```

`useSearchParams` re-memoizes on `location.search`; deriving the signal
during render lets the effect's own URL cleanup tear it down. Freezing it at
mount makes the deps stable. Used by `useCheckoutReturn`, `useTopupReturn`,
the onboarding cancel resume, and the pricing cancel toast. Toasts carry
stable sonner `id`s so a StrictMode double run updates instead of stacking.

`useCheckoutReturn`'s invariant: `activating` goes true in exactly one place,
immediately before a timeout is armed whose handler settles; `settle` is
idempotent and takes the overlay down before any side effect. The only other
thing that clears the timers is unmount, which takes the overlay with it.

## The fallback: remember intent, never entitlement

`src/lib/pendingPlan.ts` — per-user localStorage (`msanii_pending_plan.<uid>`),
`{ plan, startedAt }`, 7-day TTL, plan validated against the four
`CheckoutPlan` literals. localStorage (unlike `pendingInvite`'s
sessionStorage) because a closed Stripe tab is the case it exists for.

- Written by both checkout entry points once the session exists, right
  before the redirect. Resume buttons re-stash (fresh window).
- **Cleared on ARRIVAL at the success URL**, not on "paid". Reaching
  `/profile?welcome=true&…` means Checkout completed; if the webhook is late,
  a still-present "Finish upgrading" button would start a second paid
  checkout. Worst case now is "no resume nudge after a lost webhook".
- Also cleared on an explicit Free choice (onboarding, pricing), on the
  strip's X, and by any reader that sees a paid tier.
- Readers (`UpgradeBanner` on the dashboard, `PlanCard` on Billing) gate on
  `useEntitlements()` — server truth — and say nothing while it is loading
  or `degraded`. The pending strip ignores the generic banner's permanent
  dismiss flag (that flag means "stop selling me Pro"; this is "finish what
  you started", newer and self-expiring, with its own X).

Nothing reads the stash to decide what a user may do; its only side effect
is calling the existing authenticated `create-checkout-session` endpoint.

## Onboarding resume details

`resumeAtPlan` is latched at mount and seeds `currentStep` to 3. `loadProfile`
still backfills the onboarded cache but skips the `/dashboard` bounce when
resuming, and now loads `role` too — "Continue with Free" re-runs the
profile upsert, which would otherwise have blanked it.

## Follow-ups (out of scope)

- ~~Backend: `create_checkout_session` does not refuse when the user already
  holds an active Stripe subscription.~~ Closed 2026-09-10, see
  `2026-09-10-one-subscription-per-user.md` (409 guard, webhook id-guards,
  `/pricing` switches through the Portal).
- `/auth?redirect=/pricing&plan=…` drops `plan`: email signup hardcodes
  `/onboarding`, and `StepPlan` only sells Basic.
- Server-side pending plan (a `checkout.session.expired` webhook or a column)
  for cross-device resume, mirroring the invite plan's deferred step 3.
- The frontend `captureCheckoutStarted` on `/pricing` duplicates the
  backend's `checkout_started` with an incompatible `plan` value.
