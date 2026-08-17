-- Credit base rates (spec 2026-08-17-credit-base-rates-design.md).
--
-- No schema change. credit_prices.credits already held the per-action flat
-- number that check_credits reserves against; this migration promotes it from
-- "pre-flight estimate" to "BASE RATE". The charge became max(base, metered)
-- in subscriptions/service.py::debit_for_action.
--
-- ⚠️ APPLY THIS BEFORE DEPLOYING THE BACKEND. check_credits treats an action
-- with no seeded price as a config error and DENIES it on every tier, so the
-- split_sheet code shipping first would 402 every sheet for everyone. The
-- reverse order is harmless (old prices just mean smaller charges).
-- This is the OPPOSITE of 20260818000001_boards_to_orgs.sql, which was
-- forward-breaking and had to follow its backend. Do not carry that habit over.

BEGIN;

UPDATE credit_prices SET credits = 5,  updated_at = now() WHERE action = 'zoe_message';
UPDATE credit_prices SET credits = 30, updated_at = now() WHERE action = 'oneclick_run';
UPDATE credit_prices SET credits = 30, updated_at = now() WHERE action = 'registry_parse';

-- Split sheets become a metered action. They keep their monthly cap
-- (tier_entitlements.max_split_sheets_per_month) — cap first, then credits.
INSERT INTO credit_prices (action, credits) VALUES ('split_sheet', 20)
  ON CONFLICT (action) DO UPDATE SET credits = EXCLUDED.credits, updated_at = now();

-- Free grant 100 -> 150. 20260816000001 set it to 100 two days ago and the
-- grandfather stamps key off that migration, so this must be a NEW file.
UPDATE tier_entitlements SET monthly_credits = 150 WHERE tier = 'free';

DO $$
DECLARE
  v_bad TEXT := '';
  v_free INTEGER;
BEGIN
  SELECT string_agg(action || '=' || credits, ', ')
    INTO v_bad
    FROM credit_prices
   WHERE (action = 'zoe_message'    AND credits <> 5)
      OR (action = 'oneclick_run'   AND credits <> 30)
      OR (action = 'registry_parse' AND credits <> 30)
      OR (action = 'split_sheet'    AND credits <> 20);
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'credit_prices wrong after migration: %', v_bad;
  END IF;

  IF (SELECT count(*) FROM credit_prices
       WHERE action IN ('zoe_message','oneclick_run','registry_parse','split_sheet')) <> 4 THEN
    RAISE EXCEPTION 'credit_prices is missing one of the four metered actions';
  END IF;

  SELECT monthly_credits INTO v_free FROM tier_entitlements WHERE tier = 'free';
  IF v_free IS DISTINCT FROM 150 THEN
    RAISE EXCEPTION 'free monthly_credits is %, expected 150', v_free;
  END IF;
END $$;

COMMIT;
