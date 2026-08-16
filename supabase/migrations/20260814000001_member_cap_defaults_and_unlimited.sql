-- supabase/migrations/20260814000001_member_cap_defaults_and_unlimited.sql
-- ============================================================================
-- New members start CAPPED (2,000 credits/month) instead of unlimited, and an
-- admin gains an explicit "no limit" that survives a non-null org default.
--
-- WHY BOTH: the cap fallback chain is
--     org_members.monthly_cap -> organizations.default_member_cap -> uncapped
-- so "a new member defaults to 2,000" is expressed by giving the org a
-- default_member_cap of 2000. But once an org HAS a default, NULL on the member
-- row stops meaning uncapped and starts meaning "inherit 2000" — closing the
-- only way to say "this member has no limit". Hence the -1 sentinel, the same
-- "-1 = unlimited" idiom tier_entitlements already uses for caps.
--
-- NEW ORGS ONLY — no backfill. Existing orgs that never set a default keep a
-- NULL one, so their members stay uncapped exactly as they are today. Capping
-- them retroactively would cut off people mid-workflow on a limit nobody chose.
-- An existing org opts in by setting the default in org settings.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. New orgs start their members capped.
-- ---------------------------------------------------------------------------
-- A column DEFAULT and nothing else: it applies to rows inserted from here on
-- and leaves every existing row untouched. orgs.service.create_org does not
-- name the column, so this IS the seed for a new org.
ALTER TABLE organizations ALTER COLUMN default_member_cap SET DEFAULT 2000;

-- ---------------------------------------------------------------------------
-- 2. Allow the -1 sentinel on both levels of the chain.
-- ---------------------------------------------------------------------------
-- The original CHECK is `monthly_cap IS NULL OR monthly_cap >= 0`, which would
-- reject -1. Dropped by its known name (20260730000001 created it explicitly).
ALTER TABLE org_members DROP CONSTRAINT IF EXISTS org_members_monthly_cap_check;
ALTER TABLE org_members ADD CONSTRAINT org_members_monthly_cap_check
  CHECK (monthly_cap IS NULL OR monthly_cap >= -1);

-- Same latitude on the org default, and nothing more negative than the
-- sentinel on either — -7 is a typo, not a meaning.
ALTER TABLE organizations DROP CONSTRAINT IF EXISTS organizations_default_member_cap_check;
ALTER TABLE organizations ADD CONSTRAINT organizations_default_member_cap_check
  CHECK (default_member_cap IS NULL OR default_member_cap >= -1);

-- ---------------------------------------------------------------------------
-- 3. debit_credits: a negative cap is unlimited, not a ceiling of -1.
-- ---------------------------------------------------------------------------
-- Body copied VERBATIM from 20260805000003 (itself a verbatim copy of
-- 20260730000001 plus the cap-period WHILE loop). The ONLY change is the
-- CASE in the cap SELECT, marked in place below. Without it a -1 cap would
-- satisfy `v_cap_used + p_amount > v_cap` on every debit and flag every one
-- of them cap_exceeded — spending would still work (over-cap is recorded,
-- never blocked) but the ledger would be lying.
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
    -- NEGATIVE = EXPLICITLY UNLIMITED (-1 sentinel, the repo's existing
    -- "-1 means unlimited" idiom from tier_entitlements). Normalized to NULL
    -- right here so every line below — the ceiling test, the metadata, the
    -- cap_exceeded flag — keeps treating NULL as "no ceiling" and needs no
    -- change. Without this, a -1 cap would mark EVERY debit cap_exceeded.
    SELECT CASE WHEN COALESCE(m.monthly_cap, o.default_member_cap) < 0 THEN NULL
                ELSE COALESCE(m.monthly_cap, o.default_member_cap) END,
           m.cap_used, m.cap_period_end
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

COMMIT;

DO $$
BEGIN
  -- The column default is the whole point of step 1 — assert it stuck. NOT
  -- asserting "no NULL defaults remain": existing orgs keep theirs by design.
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'organizations'
      AND column_name = 'default_member_cap' AND column_default = '2000'
  ) THEN
    RAISE EXCEPTION 'organizations.default_member_cap default is not 2000';
  END IF;
  IF has_function_privilege('authenticated', 'public.debit_credits(uuid, integer, text, text, text, jsonb, uuid)', 'EXECUTE') THEN
    RAISE EXCEPTION 'debit_credits is executable by authenticated — REVOKE did not stick';
  END IF;
  RAISE NOTICE 'member cap defaults + unlimited sentinel applied';
END $$;
