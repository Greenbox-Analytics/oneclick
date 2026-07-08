<wizard-report>
# PostHog post-wizard report

The wizard has completed a deep integration of PostHog analytics into the Msanii FastAPI backend. The project already had a foundational `analytics.py` wrapper and `AnalyticsMiddleware` in place; this integration layered in 10 targeted business-critical events across 4 files, set up environment variables, and built a monitoring dashboard.

## Environment setup

Three environment variables were written to `.env`:
- `POSTHOG_PROJECT_TOKEN` — PostHog project token
- `POSTHOG_HOST` — `https://us.i.posthog.com`
- `POSTHOG_ENABLED` — `true`

## Events added

| Event | Description | File |
|---|---|---|
| `checkout_started` | User initiated a Stripe Checkout session (plan captured as property) | `src/backend/subscriptions/billing_router.py` |
| `billing_portal_opened` | User opened the Stripe Customer Portal | `src/backend/subscriptions/billing_router.py` |
| `subscription_activated` | Stripe webhook fired — user upgraded to Pro (price ID + status captured) | `src/backend/subscriptions/stripe_events.py` |
| `subscription_canceled` | Stripe webhook fired — user downgraded back to Free | `src/backend/subscriptions/stripe_events.py` |
| `payment_failed` | Stripe webhook fired — renewal payment failed, subscription past_due | `src/backend/subscriptions/stripe_events.py` |
| `tool_used` (tool=zoe) | User submitted a Zoe AI query | `src/backend/main.py` |
| `tool_used` (tool=oneclick) | User ran an OneClick royalty calculation | `src/backend/main.py` |
| `contract_uploaded` | User uploaded a contract PDF to a project (file_size captured) | `src/backend/main.py` |
| `work_created` | User created a new work in the Metadata Registry (work_type captured) | `src/backend/registry/router.py` |
| `work_submitted_for_registration` | User submitted a work for approval/registration | `src/backend/registry/router.py` |

**Pre-existing events (not duplicated):**
- `tool_used` (tool=splitsheet) — already in `src/backend/splitsheet/router.py`
- `request_completed` — middleware, every successful API response
- `request_failed` — middleware, every exception-level API failure

## Next steps

We've built some insights and a dashboard for you to keep an eye on user behavior, based on the events we just instrumented:

- [Analytics basics dashboard](/dashboard/1593101)
- [Subscription Upgrade Funnel](/insights/xLD3nxVV) — checkout_started → subscription_activated conversion rate
- [Tool Usage by Tool Type](/insights/H9gwKx41) — daily tool_used events broken down by tool (zoe/oneclick/splitsheet)
- [Churn & Payment Failure Events](/insights/0j4qGh0B) — subscription_canceled and payment_failed over time
- [Monthly Unique Tool Users](/insights/uf7pUYfP) — weekly active users of any Msanii tool
- [Work Registry Activity](/insights/6GzpQ5o0) — works created and submitted for registration per day

### Agent skill

We've left an agent skill folder in your project at `.claude/skills/integration-fastapi/`. You can use this context for further agent development when using Claude Code. This will help ensure the model provides the most up-to-date approaches for integrating PostHog.

</wizard-report>
