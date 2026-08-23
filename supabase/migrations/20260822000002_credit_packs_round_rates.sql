-- Re-price the credit pack ladder to round per-credit rates (owner decision,
-- 2026-08-22): 1.9¢ / 1.7¢ / 1.6¢ / 1.55¢. The 20260819000002 ladder priced
-- the rungs at 1.83 / 1.75 / 1.65 / 1.6 ¢ — rates that read like floats on
-- the purchase card. Rather than rounding the DISPLAY (which would make the
-- shown rate × credits disagree with the actual charge), the price itself
-- moves so the arithmetic is exact end to end.
--
-- Credit counts are UNCHANGED — they are sized in units of work
-- (~10/40/133/500 OneClick runs) and still don't collide with any tier's
-- monthly grant. Only price_cents moves:
--
--   pack_300    Starter    300 cr   $5.50 -> $5.70    1.9¢/cr
--   pack_1200   Creator  1,200 cr  $21.00 -> $20.40   1.7¢/cr
--   pack_4000   Studio   4,000 cr  $66.00 -> $64.00   1.6¢/cr
--   pack_15000  Label   15,000 cr $240.00 -> $232.50  1.55¢/cr
--
-- Spec §7.2 invariants all hold: every rate is strictly ABOVE Basic's
-- 1.5¢/cr (packs never undercut a subscription) and strictly BELOW the 2¢
-- overage rate (packs always beat pay-as-you-go), re-asserted below.
--
-- Safe to apply with the ladder LIVE: checkout bills price_cents ad-hoc at
-- session-creation time (no Stripe Price mirrors these rows), so the new
-- price applies to the next checkout and any session already open completes
-- at the price it was quoted. Filtered on stripe_price_id IS NULL so an
-- operator-overridden pack (real Stripe Price) is never silently repriced
-- out from under its Price.

BEGIN;

UPDATE credit_packs SET price_cents =   570, updated_at = now()
 WHERE key = 'pack_300'   AND stripe_price_id IS NULL;
UPDATE credit_packs SET price_cents =  2040, updated_at = now()
 WHERE key = 'pack_1200'  AND stripe_price_id IS NULL;
UPDATE credit_packs SET price_cents =  6400, updated_at = now()
 WHERE key = 'pack_4000'  AND stripe_price_id IS NULL;
UPDATE credit_packs SET price_cents = 23250, updated_at = now()
 WHERE key = 'pack_15000' AND stripe_price_id IS NULL;

DO $$
DECLARE
  v_bad TEXT;
BEGIN
  -- Both price fences at once: above every subscription rate (Basic 1.5¢),
  -- below overage (2¢). A rung outside the band has no reason to exist.
  SELECT string_agg(key || ' @ ' || round(price_cents::numeric / credits, 4) || 'c', ', ')
    INTO v_bad
    FROM credit_packs
   WHERE key IN ('pack_300', 'pack_1200', 'pack_4000', 'pack_15000')
     AND (price_cents::numeric / credits <= 1.5 OR price_cents::numeric / credits >= 2.0);
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'pack rate outside the (1.5c, 2.0c) band: %', v_bad;
  END IF;
END $$;

COMMIT;
