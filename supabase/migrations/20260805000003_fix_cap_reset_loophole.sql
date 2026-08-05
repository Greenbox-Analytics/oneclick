-- supabase/migrations/20260805000003_fix_cap_reset_loophole.sql
-- ============================================================================
-- Close the cap-reset loophole in debit_credits.
--
-- 20260730000001 copies a lapsed member cap window straight from the pool
-- wallet's period_end. But if the POOL itself has not been rolled yet (the
-- daily sweep runs once a day), that value is ALSO in the past — so the
-- "lapsed" branch re-enters on EVERY debit, cap_used resets to 0 each time,
-- and a member can loop actions straight past their cap until the sweep runs.
--
-- Fix: after resolving the pool's period_end, step a lapsed bound forward by
-- whole months until it is in the future — the same way the personal lazy
-- rollover advances periods (subscriptions/service.py _maybe_rollover_wallet:
-- `while new_period_end <= now: += 1 month`). The window keeps the pool's
-- day-of-month anchor, so it lands on the bound the sweep will stamp when it
-- rolls the pool, and the counter carries across that roll instead of
-- resetting per debit.
--
-- The function body below is otherwise copied VERBATIM from 20260730000001 —
-- the only change is the WHILE loop (and its comment) in the cap-period
-- block. Same REVOKE/GRANT posture re-asserted after the replace.
-- ============================================================================

BEGIN;

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
    --
    -- Floored into the FUTURE: if the pool has not been rolled yet (the daily
    -- sweep has not run), its period_end is also in the past, and copying it
    -- verbatim would re-enter this branch on every debit — cap_used resets to
    -- 0 each time and the cap never binds until the sweep. Stepping a lapsed
    -- bound forward by whole months mirrors the personal lazy rollover
    -- (_maybe_rollover_wallet), keeping the pool's day-of-month anchor.
    IF v_cap_end IS NULL OR v_cap_end <= now() THEN
      v_cap_used := 0;
      v_cap_end := COALESCE(
        (SELECT period_end FROM credit_wallets WHERE id = p_wallet_id),
        date_trunc('month', now()) + INTERVAL '1 month'
      );
      WHILE v_cap_end <= now() LOOP
        v_cap_end := v_cap_end + INTERVAL '1 month';
      END LOOP;
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

-- CREATE OR REPLACE preserves ACLs, but the service-role-only posture is
-- load-bearing on a money RPC — re-asserted, matching 20260730000001.
REVOKE EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB, UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB, UUID) TO service_role;

-- ---------------------------------------------------------------------------
-- Verify the shape, so a partial apply can't pass silently.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc p
                 JOIN pg_namespace n ON n.oid = p.pronamespace
                 WHERE n.nspname = 'public' AND p.proname = 'debit_credits' AND p.pronargs = 7) THEN
    RAISE EXCEPTION 'debit_credits(7 args) missing after replace';
  END IF;
  IF has_function_privilege('authenticated', 'public.debit_credits(uuid, integer, text, text, text, jsonb, uuid)', 'EXECUTE') THEN
    RAISE EXCEPTION 'debit_credits is executable by authenticated — REVOKE did not stick';
  END IF;
  RAISE NOTICE 'debit_credits cap-reset fix applied';
END $$;

COMMIT;
