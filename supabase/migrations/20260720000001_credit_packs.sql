-- supabase/migrations/20260720000001_credit_packs.sql
-- ============================================================================
-- Licensing Phase A: one-time credit top-up packs.
-- Spec: docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §3
--
-- OPERATOR CHECKLIST (do this after running this migration):
--   1. In the Stripe dashboard, create FOUR one-time Prices (mode=payment),
--      one per pack: pack_500 ($10.00), pack_2000 ($36.00),
--      pack_10000 ($150.00), pack_50000 ($600.00).
--   2. For each pack, run:
--        UPDATE credit_packs SET stripe_price_id = 'price_...', active = true
--        WHERE key = '<pack_key>';
--   3. Once stripe_price_id is set AND active=true, the pack becomes
--      purchasable via GET /billing/credit-packs + POST /billing/create-topup-session.
--      Until then it stays hidden/unsellable (never a checkout 500).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. credit_packs — public-read catalog of one-time top-up packs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_packs (
  key TEXT PRIMARY KEY,               -- 'pack_500' | 'pack_2000' | 'pack_10000' | 'pack_50000'
  credits INTEGER NOT NULL CHECK (credits > 0),
  price_cents INTEGER NOT NULL CHECK (price_cents > 0),
  -- Filled by the operator after creating the one-time Price in Stripe.
  -- Nullable + active=false by default so an unconfigured pack is
  -- UNSELLABLE (endpoint filters it out) instead of a checkout 500.
  stripe_price_id TEXT,
  active BOOLEAN NOT NULL DEFAULT false,
  sort_order INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO credit_packs (key, credits, price_cents, sort_order) VALUES
  ('pack_500',     500,   1000, 1),   -- $0.020/cr  (all CALIBRATE)
  ('pack_2000',   2000,   3600, 2),   -- $0.018/cr
  ('pack_10000', 10000,  15000, 3),   -- $0.015/cr
  ('pack_50000', 50000,  60000, 4)    -- $0.012/cr
ON CONFLICT (key) DO NOTHING;

ALTER TABLE credit_packs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Public read on credit_packs" ON credit_packs;
CREATE POLICY "Public read on credit_packs" ON credit_packs FOR SELECT USING (true);

-- ---------------------------------------------------------------------------
-- 2. credit_ledger.kind gains 'purchase'. Catalog-driven drop: the CHECK was
-- inline/unnamed, so a live DB may carry a different generated name (e.g.
-- after a restore or an out-of-band edit). Dropping by guessed name would
-- silently no-op and the stale CHECK would keep rejecting 'purchase' —
-- surfacing only at the first real top-up. Drop ALL kind CHECKs via the
-- catalog instead, then re-add one with a known name.
-- ---------------------------------------------------------------------------
DO $$
DECLARE c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.credit_ledger'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%kind%'
  LOOP
    EXECUTE format('ALTER TABLE public.credit_ledger DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;
ALTER TABLE credit_ledger ADD CONSTRAINT credit_ledger_kind_check
  CHECK (kind IN ('monthly_grant', 'debit', 'overage_debit', 'admin_grant',
                  'refund', 'expiry', 'storage_bill', 'purchase', 'clawback'));

-- ---------------------------------------------------------------------------
-- 3. grant_credits learns kind='purchase'. Body copied EXACTLY from
-- 20260713000002 (lines 307-375: same idempotency, same guarded
-- subtransaction, same grants posture) with ONE functional line changed —
-- the kind whitelist below now includes 'purchase'.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.grant_credits(
  p_wallet_id UUID,
  p_amount INTEGER,
  p_kind TEXT,
  p_bucket TEXT DEFAULT 'reserve',
  p_metadata JSONB DEFAULT '{}',
  p_request_id TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_existing RECORD;
  v_bundle INTEGER;
  v_reserve INTEGER;
  v_balance_after INTEGER;
BEGIN
  IF p_amount < 0 THEN RAISE EXCEPTION 'grant amount must be >= 0'; END IF;
  IF p_bucket NOT IN ('bundle', 'reserve') THEN RAISE EXCEPTION 'invalid bucket %', p_bucket; END IF;
  -- Kind whitelist symmetric with debit_credits. storage_bill rows are
  -- inserted directly by the sweep, never via grant_credits — that's why
  -- it's absent here. 'purchase' added for one-time credit top-up packs
  -- (Licensing Phase A).
  IF p_kind NOT IN ('monthly_grant', 'admin_grant', 'refund', 'purchase') THEN
    RAISE EXCEPTION 'invalid grant kind %', p_kind;
  END IF;

  -- Optional idempotency (e.g. Stripe retries a webhook whose handler failed
  -- AFTER granting): same pre-check / re-check / converge as debit_credits.
  IF p_request_id IS NOT NULL THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    IF FOUND THEN
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END IF;
  END IF;

  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve
    FROM credit_wallets WHERE id = p_wallet_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', p_wallet_id; END IF;

  -- Re-check idempotency under the lock (racer committed while we waited).
  IF p_request_id IS NOT NULL THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    IF FOUND THEN
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END IF;
  END IF;

  IF p_bucket = 'bundle' THEN v_bundle := v_bundle + p_amount;
  ELSE v_reserve := v_reserve + p_amount;
  END IF;
  v_balance_after := v_bundle + v_reserve;

  -- UPDATE + INSERT in one guarded subtransaction so a request_id collision
  -- rolls back both (see debit_credits for the failure mode).
  BEGIN
    UPDATE credit_wallets SET bundle_balance = v_bundle, reserve_balance = v_reserve, updated_at = now()
      WHERE id = p_wallet_id;
    INSERT INTO credit_ledger (wallet_id, delta, kind, request_id, balance_after, metadata)
      VALUES (p_wallet_id, p_amount, p_kind, p_request_id, v_balance_after,
              p_metadata || jsonb_build_object('bucket', p_bucket));
  EXCEPTION WHEN unique_violation THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END;

  RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after);
END;
$$;

-- Service-role only: this mutates money state and bypasses RLS (SECURITY
-- DEFINER). Supabase grants EXECUTE on public functions to anon/authenticated
-- by default and exposes them at /rest/v1/rpc/* — without this REVOKE, any
-- signed-in user could mint credits into their own wallet. Restated (not
-- newly introduced) so the minting-hole posture from 20260713000002 does not
-- regress across this CREATE OR REPLACE.
REVOKE EXECUTE ON FUNCTION public.grant_credits(UUID, INTEGER, TEXT, TEXT, JSONB, TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.grant_credits(UUID, INTEGER, TEXT, TEXT, JSONB, TEXT) TO service_role;

-- ---------------------------------------------------------------------------
-- debit_credits learns kind='clawback' (admin adjust for pack refunds and
-- chargebacks — spec 2026-07-19 §3 refund policy; endpoint lands with this
-- migration's Phase A). Body copied EXACTLY from 20260713000002 with the
-- whitelist as the only functional change. Grants posture restated below.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.debit_credits(
  p_wallet_id UUID,
  p_amount INTEGER,
  p_action TEXT,
  p_request_id TEXT,
  p_kind TEXT DEFAULT 'debit',
  p_metadata JSONB DEFAULT '{}'
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_existing RECORD;
  v_bundle INTEGER;
  v_reserve INTEGER;
  v_from_reserve INTEGER;
  v_balance_after INTEGER;
BEGIN
  IF p_amount < 0 THEN RAISE EXCEPTION 'debit amount must be >= 0'; END IF;
  -- 'clawback' added for the Phase A admin adjust endpoint (pack refunds /
  -- chargebacks). Distinct from the GRANT kind 'refund' (credits given TO a
  -- user) so ledger rows are direction-unambiguous. Never used by tool call
  -- sites; has its own reserve-only clamped branch below.
  IF p_kind NOT IN ('debit', 'overage_debit', 'clawback') THEN
    RAISE EXCEPTION 'invalid debit kind %', p_kind;
  END IF;

  -- Fast-path idempotency check (no lock taken yet).
  IF p_request_id IS NOT NULL THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    IF FOUND THEN
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END IF;
  END IF;

  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve
    FROM credit_wallets WHERE id = p_wallet_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', p_wallet_id; END IF;

  -- Re-check idempotency under the lock: a racer that won while we waited has
  -- already committed its ledger row (it held this lock until commit).
  IF p_request_id IS NOT NULL THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    IF FOUND THEN
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END IF;
  END IF;

  IF p_kind = 'overage_debit' THEN
    -- Overage does NOT drain buckets (a partial balance must not be eaten on
    -- top of the full Stripe charge — that double-charges and violates
    -- reserve persistence). delta=0 keeps sum(delta)==balance reconciliation;
    -- the billable amount rides metadata.credits_billed for the sweep.
    -- UPDATE + INSERT share the guarded subtransaction, same invariant as the
    -- plain-debit path below: if the UPDATE sat outside it, a unique_violation
    -- on the INSERT would leave overage_this_period inflated on every raced
    -- duplicate — the EXCEPTION handler only rolls back statements INSIDE it.
    v_balance_after := v_bundle + v_reserve;
    BEGIN
      UPDATE credit_wallets SET
        overage_this_period = overage_this_period + p_amount,
        updated_at = now()
      WHERE id = p_wallet_id;

      INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
      VALUES (p_wallet_id, 0, p_kind, p_action, p_request_id, v_balance_after,
              p_metadata || jsonb_build_object('credits_billed', p_amount));
    EXCEPTION WHEN unique_violation THEN
      -- Concurrent duplicate with the same request_id won the race: the
      -- subtransaction rolled back BOTH statements — converge on the winner.
      SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END;
    RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after);
  END IF;

  IF p_kind = 'clawback' THEN
    -- Admin clawback (pack refunds / chargebacks — spec 2026-07-19 §3).
    -- Purchased credits land in RESERVE, so a clawback drains reserve ONLY,
    -- clamped to what remains. The generic bundle-first drain below would be
    -- a NO-OP clawback: it removes monthly-grant credits that rollover_wallet
    -- restores wholesale at the next period, and any negative landed on the
    -- bundle is forgiven there as "drift". Clamping means no negative is ever
    -- created — the shortfall is a written-off cost, returned to the caller
    -- so support can see the refund exceeded what was recoverable.
    v_from_reserve := LEAST(GREATEST(v_reserve, 0), p_amount);
    v_reserve := v_reserve - v_from_reserve;
    v_balance_after := v_bundle + v_reserve;
    BEGIN
      UPDATE credit_wallets SET
        reserve_balance = v_reserve,
        updated_at = now()
      WHERE id = p_wallet_id;

      INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
      VALUES (p_wallet_id, -v_from_reserve, p_kind, p_action, p_request_id, v_balance_after,
              p_metadata || jsonb_build_object('credits_requested', p_amount,
                                               'credits_removed', v_from_reserve,
                                               'shortfall', p_amount - v_from_reserve));
    EXCEPTION WHEN unique_violation THEN
      SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END;
    RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after,
                              'removed', v_from_reserve, 'shortfall', p_amount - v_from_reserve);
  END IF;

  -- Drain bundle first; reserve second; any remainder (accepted concurrency
  -- drift) lands on the bundle, which may go negative.
  v_from_reserve := LEAST(GREATEST(v_reserve, 0), GREATEST(p_amount - GREATEST(v_bundle, 0), 0));
  v_bundle := v_bundle - (p_amount - v_from_reserve);
  v_reserve := v_reserve - v_from_reserve;
  v_balance_after := v_bundle + v_reserve;

  -- Wallet UPDATE and ledger INSERT share ONE guarded subtransaction: a
  -- plpgsql EXCEPTION block only rolls back statements INSIDE it, so if the
  -- UPDATE sat outside and the INSERT hit unique_violation, the debit would
  -- survive with no ledger row — a silent double-debit on retry.
  BEGIN
    UPDATE credit_wallets SET
      bundle_balance = v_bundle,
      reserve_balance = v_reserve,
      updated_at = now()
    WHERE id = p_wallet_id;

    INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
    VALUES (p_wallet_id, -p_amount, p_kind, p_action, p_request_id, v_balance_after,
            p_metadata || jsonb_build_object('from_reserve', v_from_reserve));
  EXCEPTION WHEN unique_violation THEN
    -- Concurrent duplicate with the same request_id won the race: the
    -- subtransaction rolled back BOTH the wallet update and the insert —
    -- converge on the winner's result.
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END;

  RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after);
END;
$$;

REVOKE EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB) TO service_role;
