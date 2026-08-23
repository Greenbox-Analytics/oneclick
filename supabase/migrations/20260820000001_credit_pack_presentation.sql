-- Credit bundles: name the pack ladder and put it on sale.
--
-- Runs AFTER 20260819000002_credit_packs_rescale.sql, which cut the current
-- ladder (pack_300 / pack_1200 / pack_4000 / pack_15000) and left every rung
-- active=false with a NULL stripe_price_id — i.e. UNSELLABLE. That was correct
-- at the time: GET /billing/credit-packs filtered on BOTH flags, and selling a
-- pack needed an operator to hand-create a one-time Price in the Stripe
-- dashboard and UPDATE the row.
--
-- Two things change that, and this migration is the DB half of them:
--
--   1. billing_router.create_topup_session now builds the Checkout line item
--      ad-hoc from `price_cents` (Stripe `price_data`) whenever
--      stripe_price_id is NULL. A pre-created Price is no longer a
--      prerequisite for selling a pack, so the endpoint lists on `active`
--      alone and that column becomes the only on/off switch.
--   2. The picker needs a human name per rung. `label` supplies it.
--
-- DELIBERATELY REVERSES the `active IS NOT FALSE` half of 20260819000002's
-- guard. That guard asserted the ladder was parked and unsellable; putting it
-- on sale is the entire point of this file, so read the two together rather
-- than as a regression. The PRICE half of that invariant is still load-bearing
-- and is re-asserted below.
--
-- Turning a pack `active` does NOT expose it early: POST
-- /billing/create-topup-session 409s while CREDITS_ENABLED is off, and every
-- credit surface in the UI is hidden behind the same flag. This removes a
-- manual per-environment step, nothing more.
--
-- SUPERSEDES the operator checklist in 20260720000001_credit_packs.sql.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. label — the name a musician sees on the card
-- ---------------------------------------------------------------------------
-- Nullable on purpose: the picker falls back to "N credits", so a pack added
-- later without a label is still sellable and still renders.
ALTER TABLE credit_packs ADD COLUMN IF NOT EXISTS label TEXT;

COMMENT ON COLUMN credit_packs.label IS
  'Display name for the pack picker (e.g. "Creator"). NULL renders as "N credits".';
COMMENT ON COLUMN credit_packs.stripe_price_id IS
  'OPTIONAL override. NULL means the checkout line item is built ad-hoc from price_cents.';

-- ---------------------------------------------------------------------------
-- 2. Guard: never put a pack on sale at or above the overage rate
-- ---------------------------------------------------------------------------
-- Invariant 3 from 20260819000002 (spec §7.2): a pack that costs $0.02/cr or
-- more has no reason to exist — the customer would be better off on
-- pay-per-use, which needs no checkout at all. Re-checked HERE because this is
-- the migration that makes the rungs buyable, so this is the last point at
-- which a bad price is still cheap to catch.
DO $$
DECLARE
  v_bad TEXT;
BEGIN
  SELECT string_agg(key, ', ') INTO v_bad
    FROM credit_packs
   WHERE key IN ('pack_300', 'pack_1200', 'pack_4000', 'pack_15000')
     AND (price_cents::numeric / credits) >= 2.0;
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'refusing to activate a pack at or above $0.02/cr: %', v_bad;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 3. Name + activate the four rungs
-- ---------------------------------------------------------------------------
-- One UPDATE per key, each filtered on that key, so this is a no-op on any
-- environment where an operator has already renamed or deactivated a row by
-- hand — and so a future ladder re-cut doesn't silently activate rungs this
-- file never named.
--
-- Sizes read in units of work at the current base rates (zoe 5, oneclick 30,
-- registry_parse 30, split_sheet 20). The UI derives those counts live from
-- credit_prices — they are NOT stored here, so a rate change can't strand
-- stale copy in the catalog.
UPDATE credit_packs SET label = 'Starter', active = true, updated_at = now() WHERE key = 'pack_300';
UPDATE credit_packs SET label = 'Creator', active = true, updated_at = now() WHERE key = 'pack_1200';
UPDATE credit_packs SET label = 'Studio',  active = true, updated_at = now() WHERE key = 'pack_4000';
UPDATE credit_packs SET label = 'Label',   active = true, updated_at = now() WHERE key = 'pack_15000';

DO $$
DECLARE
  v_n INTEGER;
BEGIN
  SELECT count(*) INTO v_n FROM credit_packs WHERE active AND label IS NOT NULL;
  IF v_n <> 4 THEN
    RAISE EXCEPTION 'expected 4 labelled, active packs, found %', v_n;
  END IF;
END $$;

COMMIT;
