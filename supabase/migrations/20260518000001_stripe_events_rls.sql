-- ============================================================================
-- Enable RLS on public.stripe_events
--
-- The table is a backend-only idempotency + audit log for Stripe webhooks.
-- It contains no user data per se, but stores raw Stripe event payloads
-- (customer IDs, amounts, subscription details). Without RLS, any
-- authenticated client could query it via PostgREST.
--
-- Strategy: enable RLS with NO policies. PostgREST callers (anon, authed)
-- get zero access. The backend writes via the Supabase service-role key,
-- which bypasses RLS — so the webhook handler's INSERT/SELECT continues to
-- work unchanged.
--
-- Supabase advisor flagged this as "RLS Disabled in Public — Critical".
-- ============================================================================

ALTER TABLE public.stripe_events ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.stripe_events IS
  'Stripe webhook idempotency log. Service-role-only access. RLS is enabled '
  'with no policies — PostgREST callers cannot read or write. Backend uses '
  'service-role key which bypasses RLS. Do NOT add permissive policies; the '
  'payload column may contain customer IDs, subscription amounts, etc.';
