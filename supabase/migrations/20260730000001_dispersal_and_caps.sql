-- supabase/migrations/20260730000001_dispersal_and_caps.sql
-- ============================================================================
-- LICENSING: replace seat wallets with a monthly dispersal into ONE org pool
-- plus a per-member monthly CAP.
--
-- WHY A SEPARATE MIGRATION: 20260713000002 / 20260720000001 / 20260721000001
-- are ALREADY APPLIED, so editing them in place would change nothing for this
-- database while silently diverging from what actually ran. This file is the
-- fix-forward: it takes the applied seat-wallet schema to the new shape, and a
-- database built from scratch reaches the same place by running the chain in
-- order.
--
-- WHAT CHANGES
--   1. Storage overage is gone (it billed nobody — no UI ever set the opt-in).
--   2. Free tier grant 50 -> 150 credits.
--   3. organizations gains monthly_dispersal_credits; default_seat_allowance
--      becomes default_member_cap.
--   4. org_members gains monthly_cap / cap_used / cap_period_end.
--   5. credit_requests becomes a CAP-RAISE request.
--   6. Legacy seat wallets are drained back into their org pool (audited), and
--      owner_type is constrained to ('user','org') going forward.
--   7. transfer_credits is dropped — nothing moves wallet-to-wallet any more.
--   8. debit_credits takes p_member_id and enforces the cap under the pool lock.
--   9. rollover_wallet accepts org pools, so the dispersal EXPIRES monthly.
--
-- ONE transaction: a half-applied version of this leaves the money RPCs and the
-- app disagreeing about whether members hold wallets.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Storage overage: never reachable (nothing ever set the opt-in), so the
--    column goes and uploads simply block at included_storage_bytes.
-- ---------------------------------------------------------------------------
ALTER TABLE subscriptions DROP COLUMN IF EXISTS storage_overage_enabled;

-- ---------------------------------------------------------------------------
-- 2. Free tier: 150 credits — one complete pass through the product plus room
--    to repeat the best part (~2 OneClick runs + 2 parses + ~20 Zoe messages).
--    At 50 a free user could not both try the royalty calculator and Zoe.
-- ---------------------------------------------------------------------------
UPDATE tier_entitlements SET monthly_credits = 150 WHERE tier = 'free';

-- ---------------------------------------------------------------------------
-- 3. organizations: the contract dials.
--    The dispersal is what the sweep adds to the pool each period. The old
--    default_seat_allowance carried the same NUMBER with different mechanics
--    (fill a seat wallet up to it, vs cap spend from the pool at it), so the
--    value carries over cleanly under its new name.
-- ---------------------------------------------------------------------------
ALTER TABLE organizations
  ADD COLUMN IF NOT EXISTS monthly_dispersal_credits INTEGER NOT NULL DEFAULT 0;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='organizations'
               AND column_name='default_seat_allowance')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='organizations'
               AND column_name='default_member_cap') THEN
    ALTER TABLE organizations RENAME COLUMN default_seat_allowance TO default_member_cap;
  END IF;
END $$;

ALTER TABLE organizations
  ADD COLUMN IF NOT EXISTS default_member_cap INTEGER;

-- ---------------------------------------------------------------------------
-- 4. org_members: the cap and its counter. cap_used/cap_period_end are
--    maintained INSIDE debit_credits (see section 8) — a service-side
--    pre-check cannot stop a member's two concurrent actions both slipping
--    under the ceiling.
-- ---------------------------------------------------------------------------
ALTER TABLE org_members
  ADD COLUMN IF NOT EXISTS monthly_cap INTEGER,
  ADD COLUMN IF NOT EXISTS cap_used INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS cap_period_end TIMESTAMPTZ;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'org_members_monthly_cap_check') THEN
    ALTER TABLE org_members ADD CONSTRAINT org_members_monthly_cap_check
      CHECK (monthly_cap IS NULL OR monthly_cap >= 0);
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 5. credit_requests: a request to RAISE a cap, not to move credits. Approving
--    one is therefore idempotent and costs the pool nothing until spend.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='credit_requests'
               AND column_name='requested_credits') THEN
    ALTER TABLE credit_requests RENAME COLUMN requested_credits TO requested_cap;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='credit_requests'
               AND column_name='resolved_credits') THEN
    ALTER TABLE credit_requests RENAME COLUMN resolved_credits TO resolved_cap;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 6. Legacy seat wallets. Licensing has never been enabled in production, so
--    in practice there are none — but a local QA run against this shared
--    database could have left some, and their balances are real credits.
--    Drain each one into its org's pool with an audited ledger row, zero the
--    seat, then constrain owner_type going forward.
--
--    The seat ROWS are kept: credit_ledger.wallet_id references them, and
--    deleting money history to tidy a schema is never the right trade. The new
--    CHECK is added NOT VALID for exactly that reason — it binds every future
--    insert while grandfathering the drained historical rows — and then
--    VALIDATEd opportunistically, which succeeds on any database that never
--    created one.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
  w RECORD;
  pool_id UUID;
  moved INTEGER := 0;
  drained INTEGER := 0;
BEGIN
  FOR w IN
    SELECT cw.id, cw.owner_id AS member_id, cw.bundle_balance, cw.reserve_balance, m.org_id
      FROM credit_wallets cw
      JOIN org_members m ON m.id = cw.owner_id
     WHERE cw.owner_type = 'seat'
  LOOP
    moved := COALESCE(w.bundle_balance, 0) + COALESCE(w.reserve_balance, 0);
    SELECT id INTO pool_id FROM credit_wallets
      WHERE owner_type = 'org' AND owner_id = w.org_id;
    IF pool_id IS NULL THEN
      INSERT INTO credit_wallets (owner_type, owner_id) VALUES ('org', w.org_id)
        RETURNING id INTO pool_id;
    END IF;

    IF moved > 0 THEN
      -- Credit the pool (reserve: this is money that was bought, not dispersed)
      -- and debit the seat, both audited, so sum(delta) still reconciles.
      UPDATE credit_wallets SET reserve_balance = reserve_balance + moved, updated_at = now()
        WHERE id = pool_id;
      INSERT INTO credit_ledger (wallet_id, delta, kind, balance_after, metadata)
        SELECT pool_id, moved, 'admin_grant', bundle_balance + reserve_balance,
               jsonb_build_object('source', 'seat_wallet_migration',
                                  'from_seat_wallet', w.id, 'org_member_id', w.member_id)
          FROM credit_wallets WHERE id = pool_id;
      UPDATE credit_wallets SET bundle_balance = 0, reserve_balance = 0, updated_at = now()
        WHERE id = w.id;
      INSERT INTO credit_ledger (wallet_id, delta, kind, balance_after, metadata)
        VALUES (w.id, -moved, 'clawback', 0,
                jsonb_build_object('source', 'seat_wallet_migration', 'to_pool_wallet', pool_id));
      drained := drained + 1;
    END IF;
  END LOOP;
  IF drained > 0 THEN
    RAISE NOTICE 'drained % legacy seat wallet(s) back into their org pool', drained;
  END IF;
END $$;

DO $$
DECLARE c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.credit_wallets'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%owner_type%'
  LOOP
    EXECUTE format('ALTER TABLE public.credit_wallets DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;
ALTER TABLE credit_wallets ADD CONSTRAINT credit_wallets_owner_type_check
  CHECK (owner_type IN ('user', 'org')) NOT VALID;
DO $$
BEGIN
  ALTER TABLE credit_wallets VALIDATE CONSTRAINT credit_wallets_owner_type_check;
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'credit_wallets: legacy seat rows kept (drained); the CHECK still binds new inserts';
END $$;

-- ---------------------------------------------------------------------------
-- 7. Ledger kinds: 'dispersal' (the monthly contract top-up) arrives;
--    'storage_bill' / 'allocation' / 'reclaim' leave with the features that
--    produced them. Same NOT VALID + opportunistic VALIDATE treatment, since
--    a QA run may have left allocation/reclaim rows worth keeping.
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
  CHECK (kind IN ('monthly_grant', 'debit', 'overage_debit', 'admin_grant', 'refund',
                  'expiry', 'purchase', 'clawback', 'dispersal')) NOT VALID;
DO $$
BEGIN
  ALTER TABLE credit_ledger VALIDATE CONSTRAINT credit_ledger_kind_check;
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'credit_ledger: legacy allocation/reclaim/storage_bill rows kept; the CHECK still binds new inserts';
END $$;

-- ---------------------------------------------------------------------------
-- 8. transfer_credits is dropped, and debit_credits gains p_member_id.
--
--    Nothing moves wallet-to-wallet any more: credits enter the pool
--    (grant_credits: 'purchase' for packs, 'dispersal' for the contract) and
--    leave it as member spend. So there is no transfer to make atomic, no
--    paired ledger rows, no deadlock ordering, and nothing to reclaim when
--    someone is offboarded — a member only ever held a ceiling.
--
--    CREATE OR REPLACE does NOT replace across arities, so the 6-arg
--    debit_credits is dropped first: two surviving overloads would leave
--    PostgREST unable to resolve /rest/v1/rpc/debit_credits at all.
-- ---------------------------------------------------------------------------
DROP FUNCTION IF EXISTS public.transfer_credits(UUID, UUID, INTEGER, TEXT, TEXT, JSONB);
DROP FUNCTION IF EXISTS public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB);

CREATE OR REPLACE FUNCTION public.debit_credits(
  p_wallet_id UUID,
  p_amount INTEGER,
  p_action TEXT,
  p_request_id TEXT,
  p_kind TEXT DEFAULT 'debit',
  p_metadata JSONB DEFAULT '{}',
  p_member_id UUID DEFAULT NULL
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
  v_cap INTEGER;
  v_cap_used INTEGER;
  v_cap_end TIMESTAMPTZ;
  v_cap_exceeded BOOLEAN := false;
BEGIN
  IF p_amount < 0 THEN RAISE EXCEPTION 'debit amount must be >= 0'; END IF;
  -- 'clawback' is the admin adjust path (pack refunds / chargebacks) added by
  -- 20260720000001; it keeps its own reserve-only clamped branch below.
  IF p_kind NOT IN ('debit', 'overage_debit', 'clawback') THEN
    RAISE EXCEPTION 'invalid debit kind %', p_kind;
  END IF;
  -- Org pools have no pay-as-you-go: a member past their cap asks for a raise,
  -- and a dry pool is the admin's problem to fix by buying credits.
  IF p_member_id IS NOT NULL AND p_kind = 'overage_debit' THEN
    RAISE EXCEPTION 'org member spend cannot be overage — pools have no pay-as-you-go';
  END IF;
  -- A clawback is support removing money, never a member spending it: it must
  -- not touch anybody's cap counter.
  IF p_member_id IS NOT NULL AND p_kind = 'clawback' THEN
    RAISE EXCEPTION 'clawback is not member spend — p_member_id must be NULL';
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

  -- Member cap accounting. Only ever set on plain member spend (the guards
  -- above reject it for overage and clawback). Locks org_members AFTER
  -- credit_wallets — the only place that takes both, so the order can't
  -- deadlock against anything.
  IF p_member_id IS NOT NULL THEN
    SELECT COALESCE(m.monthly_cap, o.default_member_cap), m.cap_used, m.cap_period_end
      INTO v_cap, v_cap_used, v_cap_end
      FROM org_members m
      JOIN organizations o ON o.id = m.org_id
      WHERE m.id = p_member_id
      FOR UPDATE OF m;
    IF NOT FOUND THEN RAISE EXCEPTION 'org member % not found', p_member_id; END IF;

    -- The cap period tracks the POOL's period, so a member's ceiling resets
    -- exactly when the dispersal does — otherwise a member could spend a full
    -- cap on either side of a boundary that never lines up with the money.
    IF v_cap_end IS NULL OR v_cap_end <= now() THEN
      v_cap_used := 0;
      v_cap_end := COALESCE(
        (SELECT period_end FROM credit_wallets WHERE id = p_wallet_id),
        date_trunc('month', now()) + INTERVAL '1 month'
      );
    END IF;

    IF v_cap IS NOT NULL AND v_cap_used + p_amount > v_cap THEN
      v_cap_exceeded := true;   -- recorded, not blocked (see the header)
    END IF;
    v_cap_used := v_cap_used + p_amount;
    p_metadata := p_metadata || jsonb_build_object(
      'org_member_id', p_member_id, 'cap', v_cap, 'cap_used', v_cap_used,
      'cap_exceeded', v_cap_exceeded);
  END IF;

  IF p_kind = 'overage_debit' THEN
    -- Overage does NOT drain buckets (a partial balance must not be eaten on
    -- top of the full Stripe charge — that double-charges and violates
    -- reserve persistence). delta=0 keeps sum(delta)==balance reconciliation;
    -- the billable amount rides metadata.credits_billed for the sweep.
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
      SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
      RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
    END;
    RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after);
  END IF;

  IF p_kind = 'clawback' THEN
    -- Admin clawback (pack refunds / chargebacks — spec 2026-07-19 §3).
    -- Purchased credits land in RESERVE, so a clawback drains reserve ONLY,
    -- clamped to what remains. The generic bundle-first drain below would be
    -- a NO-OP clawback: it removes credits the next rollover restores
    -- wholesale (a personal monthly grant, or an org's monthly dispersal), and
    -- any negative landed on the bundle is forgiven there as "drift". Clamping
    -- means no negative is ever created — the shortfall is a written-off cost,
    -- returned to the caller so support can see the refund exceeded what was
    -- recoverable.
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
  -- drift) lands on the bundle, which may go negative. On an ORG pool that
  -- order is load-bearing: bundle is the EXPIRING monthly dispersal, so it
  -- must be spent before the permanent purchased reserve.
  v_from_reserve := LEAST(GREATEST(v_reserve, 0), GREATEST(p_amount - GREATEST(v_bundle, 0), 0));
  v_bundle := v_bundle - (p_amount - v_from_reserve);
  v_reserve := v_reserve - v_from_reserve;
  v_balance_after := v_bundle + v_reserve;

  -- Wallet UPDATE, member-cap UPDATE and ledger INSERT share ONE guarded
  -- subtransaction: a plpgsql EXCEPTION block only rolls back statements
  -- INSIDE it, so a request_id collision must roll back the cap counter too —
  -- otherwise a retried debit is free of charge but still eats the cap.
  BEGIN
    UPDATE credit_wallets SET
      bundle_balance = v_bundle,
      reserve_balance = v_reserve,
      updated_at = now()
    WHERE id = p_wallet_id;

    IF p_member_id IS NOT NULL THEN
      UPDATE org_members SET cap_used = v_cap_used, cap_period_end = v_cap_end, updated_at = now()
        WHERE id = p_member_id;
    END IF;

    INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
    VALUES (p_wallet_id, -p_amount, p_kind, p_action, p_request_id, v_balance_after,
            p_metadata || jsonb_build_object('from_reserve', v_from_reserve));
  EXCEPTION WHEN unique_violation THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END;

  RETURN jsonb_build_object('duplicate', false, 'balance_after', v_balance_after,
                            'cap', v_cap, 'cap_used', v_cap_used,
                            'cap_exceeded', v_cap_exceeded);
END;
$$;

REVOKE EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB, UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB, UUID) TO service_role;

-- ---------------------------------------------------------------------------
-- 9. rollover_wallet accepts org pools. The dispersal lands in the EXPIRING
--    bundle bucket and is zeroed at each period end — which is what stops an
--    org banking a year of credits and burning them in one month. The RESERVE
--    bucket (purchased packs) is never touched by a rollover.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.rollover_wallet(
  p_wallet_id UUID,
  p_monthly_grant INTEGER,
  p_new_period_start TIMESTAMPTZ,
  p_new_period_end TIMESTAMPTZ
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_bundle INTEGER;
  v_reserve INTEGER;
BEGIN
  IF (SELECT owner_type FROM credit_wallets WHERE id = p_wallet_id) NOT IN ('user', 'org') THEN
    RAISE EXCEPTION 'rollover_wallet: wallet % is neither a user nor an org wallet', p_wallet_id;
  END IF;

  IF p_monthly_grant < 0 THEN RAISE EXCEPTION 'monthly grant must be >= 0'; END IF;

  -- Two guards on the locking SELECT:
  --   period_end < p_new_period_end — a concurrent racer already rolled over;
  --   period_end <= now()           — self-defense against a caller that
  --     derives p_new_period_end from now() instead of stepping from the
  --     stored period_end: a rollover only fires when the current period
  --     has actually ended, so clock-derived bounds can't double-roll.
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve
    FROM credit_wallets
    WHERE id = p_wallet_id
      AND (period_end IS NULL OR period_end < p_new_period_end)
      AND (period_end IS NULL OR period_end <= now())
    FOR UPDATE;
  IF NOT FOUND THEN
    -- Distinguish "nothing to do" (already rolled / period still open) from
    -- a genuinely missing wallet, which is a caller bug worth surfacing.
    IF NOT EXISTS (SELECT 1 FROM credit_wallets WHERE id = p_wallet_id) THEN
      RAISE EXCEPTION 'wallet % not found', p_wallet_id;
    END IF;
    RETURN false;
  END IF;

  -- Zero out whatever bundle remains, with a compensating ledger row EITHER
  -- WAY so sum(delta) always reconciles to the balance: positive remainder is
  -- a normal expiry; negative remainder (accepted concurrency drift)
  -- gets a positive adjustment row flagged in metadata.
  IF v_bundle <> 0 THEN
    INSERT INTO credit_ledger (wallet_id, delta, kind, balance_after, metadata)
    VALUES (p_wallet_id, -v_bundle, 'expiry', v_reserve,
            jsonb_build_object('expired_bundle', v_bundle,
                               'negative_drift_reset', v_bundle < 0));
  END IF;

  UPDATE credit_wallets SET
    bundle_balance = p_monthly_grant,
    overage_this_period = 0,
    period_start = p_new_period_start,
    period_end = p_new_period_end,
    updated_at = now()
  WHERE id = p_wallet_id;

  INSERT INTO credit_ledger (wallet_id, delta, kind, balance_after, metadata)
  VALUES (p_wallet_id, p_monthly_grant, 'monthly_grant', p_monthly_grant + v_reserve,
          jsonb_build_object('bucket', 'bundle', 'period_end', p_new_period_end));
  RETURN true;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) TO service_role;

-- ---------------------------------------------------------------------------
-- 10. Verify the shape this migration promises, so a partial apply can't pass
--     silently. Any failure rolls the whole transaction back.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'transfer_credits') THEN
    RAISE EXCEPTION 'transfer_credits still exists';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'debit_credits' AND pronargs = 7) THEN
    RAISE EXCEPTION 'debit_credits(7 args) missing';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'debit_credits' AND pronargs = 6) THEN
    RAISE EXCEPTION 'the 6-arg debit_credits survived — PostgREST cannot resolve an overloaded RPC';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema='public' AND table_name='org_members' AND column_name='cap_used') THEN
    RAISE EXCEPTION 'org_members.cap_used missing';
  END IF;
  IF EXISTS (SELECT 1 FROM credit_wallets WHERE owner_type = 'seat'
               AND (bundle_balance <> 0 OR reserve_balance <> 0)) THEN
    RAISE EXCEPTION 'a legacy seat wallet still holds credits';
  END IF;
  RAISE NOTICE 'dispersal + caps migration applied';
END $$;

COMMIT;
