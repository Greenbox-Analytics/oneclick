-- supabase/migrations/20260816000001_tier_team_dials_and_grandfather.sql
-- ============================================================================
-- Spec 2026-08-15 (rev 2) §1/§2/§10.1: team dials on tiers, new grant values,
-- and grandfathering for subscribers active at merge time.
--
-- ONE transaction on purpose: the grandfather backfill must be visible in the
-- same instant the grant values change, or a concurrently-rolling wallet could
-- grant a pre-merge subscriber the NEW (smaller) bundle.
--
-- Grandfathering lives on subscriptions, NOT tier_overrides: the override row
-- is per-user, tier-blind, wins over tier defaults across upgrades/cancels,
-- and shares its PK with tester/admin override rows (review finding 4).
--
-- Grandfathering is NOT indefinite (owner policy clarification, spec §1
-- amended): it expires with the ALREADY-PAID billing period — monthly
-- subscribers keep the old grant until their current period ends, annual
-- until their term ends; the next renewal lands on the new tier grant.
-- `grandfathered_until` is stamped ONCE by this backfill and never extended
-- afterward — a subscriber who never triggers a tier-changing webhook simply
-- ages out at their own period end, same as everyone else.
-- ============================================================================
BEGIN;

ALTER TABLE tier_entitlements
  ADD COLUMN IF NOT EXISTS max_teams INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_team_members INTEGER NOT NULL DEFAULT 0, -- seats EXCLUDING the covering owner
  ADD COLUMN IF NOT EXISTS team_storage_bytes BIGINT NOT NULL DEFAULT 0; -- per-OWNER pool, all owned teams

ALTER TABLE subscriptions
  ADD COLUMN IF NOT EXISTS grandfathered_monthly_credits INTEGER,
  -- Expires-with-paid-period policy: the grandfathered grant reads as expired
  -- once now() passes this timestamp. Stamped ONCE by the backfill below,
  -- never extended by any webhook — see header.
  ADD COLUMN IF NOT EXISTS grandfathered_until TIMESTAMPTZ,
  -- Task 13 (storage PAYG) bookkeeping: the last owner-period a team-storage
  -- InvoiceItem was raised for. Added here so Migration B stays org-scoped.
  ADD COLUMN IF NOT EXISTS last_team_storage_invoiced_period TIMESTAMPTZ;

-- Grandfather FIRST (values read the CURRENT tier grants implicitly — they are
-- hardcoded to today's live numbers, asserted below). `grandfathered_until`
-- defaults to the subscriber's own current_period_end (the paid-through date
-- Stripe already gave us); a NULL current_period_end (never seen a
-- subscription.updated webhook yet) falls back to a 1-month floor so nobody
-- grandfathers forever on a data gap.
UPDATE subscriptions SET grandfathered_monthly_credits = 3000,
  grandfathered_until = COALESCE(current_period_end, now() + interval '1 month')
 WHERE tier = 'basic' AND status IN ('active', 'trialing', 'past_due')
   AND grandfathered_monthly_credits IS NULL;
UPDATE subscriptions SET grandfathered_monthly_credits = 8000,
  grandfathered_until = COALESCE(current_period_end, now() + interval '1 month')
 WHERE tier = 'pro' AND status IN ('active', 'trialing', 'past_due')
   AND grandfathered_monthly_credits IS NULL;

-- New grants + team dials.
UPDATE tier_entitlements SET monthly_credits = 100 WHERE tier = 'free';
UPDATE tier_entitlements SET monthly_credits = 2000,
  max_teams = 1, max_team_members = 3, team_storage_bytes = 10737418240      -- 10 GiB
 WHERE tier = 'basic';
UPDATE tier_entitlements SET monthly_credits = 5000,
  max_teams = 3, max_team_members = 10, team_storage_bytes = 107374182400    -- 100 GiB
 WHERE tier = 'pro';

COMMIT;
