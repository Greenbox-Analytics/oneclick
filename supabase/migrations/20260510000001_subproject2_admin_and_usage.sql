-- ============================================================================
-- Sub-project 2: per-AI-tool counters + pro_requests waitlist + atomic counter RPC
-- Spec: docs/superpowers/specs/2026-05-09-pricing-tiers-sp2-design.md
-- ============================================================================

-- Extend usage_counters with per-AI-tool counters that reset on period rollover.
ALTER TABLE usage_counters
  ADD COLUMN IF NOT EXISTS zoe_queries_this_period INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS oneclick_runs_this_period INTEGER NOT NULL DEFAULT 0;

-- ----------------------------------------------------------------------------
-- pro_requests: upgrade waitlist (logged-in or logged-out submissions)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pro_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL,
  message TEXT,
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  status TEXT NOT NULL CHECK (status IN ('new', 'contacted', 'converted', 'declined')) DEFAULT 'new',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pro_requests_status ON pro_requests(status);
CREATE INDEX IF NOT EXISTS idx_pro_requests_email ON pro_requests(email);

ALTER TABLE pro_requests ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "No public read on pro_requests" ON pro_requests;
CREATE POLICY "No public read on pro_requests" ON pro_requests FOR SELECT USING (false);
-- Insert is intentionally NOT exposed via RLS — backend POST /pro-requests uses
-- service role for both insert and read, gated by application-level auth.

-- ----------------------------------------------------------------------------
-- Atomic counter increment helper used by EntitlementsService.increment_usage()
--
-- Self-creates the usage_counters row if missing — protects against users
-- predating SP1's signup trigger or whose row was deleted. Without this,
-- UPDATE WHERE user_id=? matches 0 rows for missing users and the increment
-- is silently lost.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION increment_usage_counter(p_user_id UUID, p_counter_name TEXT)
RETURNS VOID AS $$
BEGIN
  -- Ensure the row exists (sane defaults for period; existing rows untouched)
  INSERT INTO usage_counters (user_id, period_start, period_end)
  VALUES (p_user_id, now(), now() + INTERVAL '1 month')
  ON CONFLICT (user_id) DO NOTHING;

  IF p_counter_name = 'zoe_queries_this_period' THEN
    UPDATE usage_counters SET zoe_queries_this_period = zoe_queries_this_period + 1, updated_at = now()
      WHERE user_id = p_user_id;
  ELSIF p_counter_name = 'oneclick_runs_this_period' THEN
    UPDATE usage_counters SET oneclick_runs_this_period = oneclick_runs_this_period + 1, updated_at = now()
      WHERE user_id = p_user_id;
  ELSIF p_counter_name = 'split_sheets_this_period' THEN
    UPDATE usage_counters SET split_sheets_this_period = split_sheets_this_period + 1, updated_at = now()
      WHERE user_id = p_user_id;
  ELSE
    RAISE EXCEPTION 'Unknown counter name: %', p_counter_name;
  END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
