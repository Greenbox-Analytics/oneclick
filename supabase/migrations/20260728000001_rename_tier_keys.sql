-- supabase/migrations/20260728000001_rename_tier_keys.sql
-- ============================================================================
-- Rename the tier KEYS so they match what users are shown:
--     'pro'     ($25 plan, labeled "Basic") -> 'basic'
--     'pro_max' ($50 plan, labeled "Pro")   -> 'pro'
--
-- WHY NOW: the keys were declared permanent, which meant every Stripe report,
-- HogQL query and support conversation would carry a key->label translation
-- table forever. Paid volume is ~0 and CREDITS_ENABLED is still off, so this is
-- the cheapest it will ever be.
--
-- MUST run AFTER 20260713000002_credits_schema.sql (which introduces 'pro_max').
--
-- ORDER IS LOAD-BEARING: 'pro' -> 'basic' FIRST, then 'pro_max' -> 'pro'. The
-- reverse order would collide on tier_entitlements' primary key (and would
-- silently merge the two tiers' subscribers).
--
-- ANALYTICS CAVEAT: events emitted before this migration used 'pro' for the $25
-- plan; events after use 'pro' for the $50 plan. Historical PostHog series that
-- filter on plan == 'pro' straddle the change — floor them at this date, or read
-- them as "any paid" (see scripts/posthog_setup_dashboard.py, whose paid cohort
-- now matches both keys).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- ONE transaction, ONE guarded block. Both are load-bearing:
--
--   * Transaction: a half-applied rename leaves tier values the CHECK forbids
--     and the backend can't read. All or nothing.
--
--   * Guard: this rename is NOT naturally idempotent and a second run is
--     DESTRUCTIVE. On an already-renamed database `WHERE tier = 'pro'` matches
--     the TOP tier (the ex-pro_max), so re-running would quietly map $50
--     subscribers down to 'basic'. tier_entitlements would fail on its primary
--     key, but `subscriptions` has no such protection — under `psql -f` without
--     ON_ERROR_STOP, execution continues past the failure and corrupts paid
--     rows. So the whole body early-returns when no 'pro_max' rows remain.
-- ---------------------------------------------------------------------------
BEGIN;

DO $$
DECLARE
  c RECORD;
  stale INTEGER;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM tier_entitlements WHERE tier = 'pro_max')
     AND NOT EXISTS (SELECT 1 FROM subscriptions WHERE tier = 'pro_max') THEN
    RAISE NOTICE 'tier keys already renamed (no pro_max rows found) — skipping';
    RETURN;
  END IF;

  -- 1. Drop the tier CHECKs (catalog-driven — a restored/edited DB may carry a
  --    different generated name, and dropping by guessed name would silently
  --    no-op, leaving a CHECK that rejects 'basic'). Same idiom as 20260713000002.
  FOR c IN
    SELECT conrelid::regclass AS tbl, conname FROM pg_constraint
    WHERE conrelid IN ('public.subscriptions'::regclass, 'public.tier_entitlements'::regclass)
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%tier%'
  LOOP
    EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', c.tbl, c.conname);
  END LOOP;

  -- 2. Rename the values, in the only safe order.
  UPDATE tier_entitlements SET tier = 'basic' WHERE tier = 'pro';
  UPDATE tier_entitlements SET tier = 'pro'   WHERE tier = 'pro_max';

  UPDATE subscriptions SET tier = 'basic' WHERE tier = 'pro';
  UPDATE subscriptions SET tier = 'pro'   WHERE tier = 'pro_max';

  -- 3. Re-add the CHECKs under known names, with the new key set.
  EXECUTE 'ALTER TABLE subscriptions ADD CONSTRAINT subscriptions_tier_check '
          'CHECK (tier IN (''free'', ''basic'', ''pro''))';
  EXECUTE 'ALTER TABLE tier_entitlements ADD CONSTRAINT tier_entitlements_tier_check '
          'CHECK (tier IN (''free'', ''basic'', ''pro''))';

  -- 4. Fail loudly if anything still carries an old key (e.g. a row written by a
  --    backend instance that had not been redeployed yet) — rolls the whole
  --    transaction back rather than leaving a mixed vocabulary behind.
  SELECT count(*) INTO stale FROM (
    SELECT tier FROM subscriptions
    UNION ALL
    SELECT tier FROM tier_entitlements
  ) t WHERE tier = 'pro_max';
  IF stale > 0 THEN
    RAISE EXCEPTION 'tier rename incomplete: % row(s) still on pro_max', stale;
  END IF;
END $$;

COMMIT;
