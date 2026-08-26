-- supabase/migrations/20260713000001_ai_usage_log.sql
-- ============================================================================
-- Credits Phase A: ai_usage_log — every LLM call's real tokens + cost.
-- Spec: docs/superpowers/specs/2026-07-12-credits-billing-design.md §6
-- Populated best-effort by utils/llm/tracking.py. Calibrates credit prices
-- against the OpenAI invoice. No user-facing behavior change.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_usage_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,               -- no FK to auth.users: log rows deliberately survive user deletion
  tool TEXT NOT NULL,                  -- 'zoe' | 'oneclick' | 'registry'; no CHECK: new tools are added in code; keep the log write-path migration-free
  model TEXT NOT NULL,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cached_tokens INTEGER,
  cost_usd NUMERIC(10, 6),             -- NULL when model missing from rate table
  credits_charged INTEGER NOT NULL DEFAULT 0,
  cache_hit BOOLEAN NOT NULL DEFAULT false,
  success BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_usage_log_user_created
  ON ai_usage_log (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_usage_log_tool_created
  ON ai_usage_log (tool, created_at DESC);

-- Service-role writes only; no client access.
ALTER TABLE ai_usage_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "No public access on ai_usage_log" ON ai_usage_log;
CREATE POLICY "No public access on ai_usage_log" ON ai_usage_log
  FOR ALL USING (false);
