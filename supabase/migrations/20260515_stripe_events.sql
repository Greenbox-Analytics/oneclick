-- ============================================================================
-- Sub-project 4: Stripe webhook idempotency
-- Spec: docs/superpowers/specs/2026-05-12-stripe-integration-design.md
-- ============================================================================

-- Idempotency table for Stripe webhook events.
-- The webhook handler INSERTs event_id before processing; on duplicate, the
-- event is skipped. On handler failure, the row is deleted so Stripe can
-- retry. The optional `payload` column stores the raw event JSON for
-- debuggability — purge old rows when the table grows.
CREATE TABLE stripe_events (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  processed_at TIMESTAMPTZ DEFAULT now(),
  payload JSONB  -- nullable; for debugging webhook deliveries
);

CREATE INDEX idx_stripe_events_type_time
  ON stripe_events (event_type, processed_at DESC);
