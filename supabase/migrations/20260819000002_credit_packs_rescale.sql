-- Re-cut the credit pack ladder for base rates (spec §7).
--
-- Safe to replace outright: every seeded pack is still active=false with a
-- NULL stripe_price_id, i.e. UNSELLABLE (GET /billing/credit-packs filters
-- them out), so no customer and no Stripe Price depends on the old keys.
--
-- The old ladder had a visible defect: pack_2000 sold 2,000 credits for $36
-- while a Basic subscription sells 2,000 credits/month for $30 AND includes a
-- team slot, unlimited split sheets and 100 GB. Side by side that reads as a
-- rip-off. pack_500 also priced at exactly the overage rate ($0.020/cr), so it
-- offered no discount for the friction of a checkout.
--
-- Invariants every rung satisfies (spec §7.2):
--   1. No pack sits at a credit count a subscription tier also sells (2,000 / 5,000).
--   2. Every pack is pricier per credit than every subscription (Basic $0.0150,
--      Pro $0.0100, Pro annual $0.0083) — packs must not cannibalise recurring revenue.
--   3. Every pack is cheaper per credit than overage ($0.0200) — else it has no reason to exist.
--   4. Sizes read in units of work: an OneClick run or registry parse is 30 credits,
--      a split sheet 20, a Zoe message 5.

BEGIN;

DELETE FROM credit_packs
 WHERE key IN ('pack_500', 'pack_2000', 'pack_10000', 'pack_50000')
   AND stripe_price_id IS NULL;

INSERT INTO credit_packs (key, credits, price_cents, sort_order) VALUES
  ('pack_300',     300,    550, 1),   -- $0.0183/cr  ~10 runs
  ('pack_1200',   1200,   2100, 2),   -- $0.0175/cr  ~40 runs
  ('pack_4000',   4000,   6600, 3),   -- $0.0165/cr  ~133 runs
  ('pack_15000', 15000,  24000, 4)    -- $0.0160/cr  ~500 runs
ON CONFLICT (key) DO UPDATE
  SET credits = EXCLUDED.credits,
      price_cents = EXCLUDED.price_cents,
      sort_order = EXCLUDED.sort_order,
      updated_at = now();

DO $$
DECLARE
  v_n INTEGER;
  v_bad TEXT;
BEGIN
  SELECT count(*) INTO v_n FROM credit_packs
   WHERE key IN ('pack_300','pack_1200','pack_4000','pack_15000');
  IF v_n <> 4 THEN
    RAISE EXCEPTION 'expected 4 new packs, found %', v_n;
  END IF;

  -- Invariant 3: strictly cheaper per credit than overage ($0.02).
  SELECT string_agg(key, ', ') INTO v_bad FROM credit_packs
   WHERE active IS NOT FALSE OR (price_cents::numeric / credits) >= 2.0;
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'pack must be inactive and under $0.02/cr: %', v_bad;
  END IF;

  -- Invariant 1: no pack collides with a tier's monthly grant.
  SELECT string_agg(p.key, ', ') INTO v_bad
    FROM credit_packs p
    JOIN tier_entitlements t ON t.monthly_credits = p.credits;
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'pack credit count collides with a tier grant: %', v_bad;
  END IF;
END $$;

COMMIT;
