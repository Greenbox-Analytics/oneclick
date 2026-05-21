-- ============================================================================
-- 2026-05-18: Add max_oneclick_runs_per_month to tier_overrides
-- ============================================================================
-- Migration 20260516_oneclick_free_tier.sql added this column to
-- tier_entitlements but missed tier_overrides. As a result, granting a tester
-- override via admin_service.create_tester_grant() (which writes -1 for this
-- column) failed with a 500 because the column did not exist on tier_overrides.
--
-- Nullable (no NOT NULL, no default) to match the rest of tier_overrides: a
-- NULL override means "fall through to tier_entitlements". Setting -1
-- explicitly means "unlimited for this user", which is what tester grants do.

ALTER TABLE tier_overrides
  ADD COLUMN IF NOT EXISTS max_oneclick_runs_per_month INTEGER;
