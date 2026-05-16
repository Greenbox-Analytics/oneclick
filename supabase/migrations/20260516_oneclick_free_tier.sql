-- ============================================================================
-- 2026-05-16: OneClick moves to Free tier with 1 use/month cap
-- ============================================================================

-- Add the per-period cap column. Default -1 (unlimited) so existing Pro
-- behavior is preserved; we explicitly seed Free=1 below.
ALTER TABLE tier_entitlements
  ADD COLUMN max_oneclick_runs_per_month INTEGER NOT NULL DEFAULT -1;

-- Free tier: enable OneClick, cap at 1 use per usage_counters period
UPDATE tier_entitlements
SET max_oneclick_runs_per_month = 1,
    oneclick_enabled = true
WHERE tier = 'free';

-- Pro tier: unlimited (explicit for clarity even though it's the default)
UPDATE tier_entitlements
SET max_oneclick_runs_per_month = -1
WHERE tier = 'pro';
