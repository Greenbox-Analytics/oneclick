-- supabase/migrations/20260713000002_credits_schema.sql
-- ============================================================================
-- Credits Phase B: wallets, ledger, prices, RPCs, pro_max tier.
-- Spec: docs/superpowers/specs/2026-07-12-credits-billing-design.md §2, §6
-- IMPORTANT: existing tier_entitlements flag/cap VALUES are not mutated —
-- flag retirement is code-level under CREDITS_ENABLED (spec §9 rollback).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Tier CHECK constraints gain 'pro_max'
-- ---------------------------------------------------------------------------
-- The original CHECKs were inline and unnamed, so Postgres auto-named them —
-- but a live DB may carry a different generated name (e.g. after a restore or
-- an out-of-band edit). Dropping by guessed name would silently no-op and the
-- stale CHECK would keep rejecting 'pro_max' — surfacing only at the first
-- real pro_max checkout. Drop ALL tier CHECKs via the catalog instead, then
-- re-add one with a known name.
DO $$
DECLARE c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.subscriptions'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%tier%'
  LOOP
    EXECUTE format('ALTER TABLE public.subscriptions DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;
ALTER TABLE subscriptions ADD CONSTRAINT subscriptions_tier_check
  CHECK (tier IN ('free', 'pro', 'pro_max'));

DO $$
DECLARE c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.tier_entitlements'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%tier%'
  LOOP
    EXECUTE format('ALTER TABLE public.tier_entitlements DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;
ALTER TABLE tier_entitlements ADD CONSTRAINT tier_entitlements_tier_check
  CHECK (tier IN ('free', 'pro', 'pro_max'));

-- ---------------------------------------------------------------------------
-- 2. New tier columns (CALIBRATE values from spec §13) + pro_max row
-- ---------------------------------------------------------------------------
ALTER TABLE tier_entitlements
  ADD COLUMN IF NOT EXISTS monthly_credits INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_works INTEGER NOT NULL DEFAULT -1,
  ADD COLUMN IF NOT EXISTS included_storage_bytes BIGINT NOT NULL DEFAULT -1;

UPDATE tier_entitlements SET monthly_credits = 50,   max_works = 10,
  included_storage_bytes = 1073741824       WHERE tier = 'free';   -- 1 GB
UPDATE tier_entitlements SET monthly_credits = 3000, max_works = -1,
  included_storage_bytes = 107374182400     WHERE tier = 'pro';    -- 100 GB

-- pro_max: clone pro, then set its own dials (insert only if missing).
INSERT INTO tier_entitlements
  (tier, max_artists, max_projects, max_boards, max_tasks, max_storage_bytes,
   max_split_sheets_per_month, max_oneclick_runs_per_month,
   zoe_enabled, oneclick_enabled, registry_enabled, integrations_allowed,
   monthly_credits, max_works, included_storage_bytes)
SELECT 'pro_max', max_artists, max_projects, max_boards, max_tasks, max_storage_bytes,
       max_split_sheets_per_month, max_oneclick_runs_per_month,
       zoe_enabled, oneclick_enabled, registry_enabled, integrations_allowed,
       8000, -1, 268435456000                                      -- 250 GB
FROM tier_entitlements WHERE tier = 'pro'
ON CONFLICT (tier) DO NOTHING;

ALTER TABLE tier_overrides
  ADD COLUMN IF NOT EXISTS monthly_credits INTEGER,
  ADD COLUMN IF NOT EXISTS max_works INTEGER;

-- ---------------------------------------------------------------------------
-- 3. Billing prefs (opt-in overage) on subscriptions
-- ---------------------------------------------------------------------------
ALTER TABLE subscriptions
  ADD COLUMN IF NOT EXISTS overage_enabled BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS overage_cap_credits INTEGER,
  ADD COLUMN IF NOT EXISTS storage_overage_enabled BOOLEAN NOT NULL DEFAULT false;

-- ---------------------------------------------------------------------------
-- 4. credit_prices — published per-action prices (public read, like tiers)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_prices (
  action TEXT PRIMARY KEY,
  credits INTEGER NOT NULL CHECK (credits >= 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO credit_prices (action, credits) VALUES
  ('zoe_message', 3), ('oneclick_run', 21), ('registry_parse', 12)
ON CONFLICT (action) DO NOTHING;

ALTER TABLE credit_prices ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Public read on credit_prices" ON credit_prices;
CREATE POLICY "Public read on credit_prices" ON credit_prices FOR SELECT USING (true);

-- ---------------------------------------------------------------------------
-- 5. credit_wallets — two buckets, org-ready
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_wallets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_type TEXT NOT NULL DEFAULT 'user' CHECK (owner_type IN ('user', 'org')),
  owner_id UUID NOT NULL,
  bundle_balance INTEGER NOT NULL DEFAULT 0,   -- monthly grant; expires at rollover
  reserve_balance INTEGER NOT NULL DEFAULT 0,  -- admin grants/refunds; never expires
  overage_this_period INTEGER NOT NULL DEFAULT 0,
  period_start TIMESTAMPTZ,
  period_end TIMESTAMPTZ,
  -- Tracks the last annual standalone-invoice time so the daily sweep bills
  -- monthly regardless of which path (lazy get_for_user vs sweep) advanced the
  -- period. Without this the sweep could only invoice users whose wallet IT
  -- rolled, but the lazy path rolls active users first — so annual overage
  -- would otherwise sit unbilled until the ~12-month Stripe renewal.
  last_standalone_invoice_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (owner_type, owner_id)
);
ALTER TABLE credit_wallets ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users read own wallet" ON credit_wallets;
CREATE POLICY "Users read own wallet" ON credit_wallets
  FOR SELECT USING (owner_type = 'user' AND owner_id = auth.uid());

-- ---------------------------------------------------------------------------
-- 6. credit_ledger — append-only source of truth
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_ledger (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  wallet_id UUID NOT NULL REFERENCES credit_wallets(id),
  delta INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN
    ('monthly_grant', 'debit', 'overage_debit', 'admin_grant', 'refund', 'expiry', 'storage_bill')),
  action TEXT,
  request_id TEXT,
  balance_after INTEGER NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_credit_ledger_wallet_created
  ON credit_ledger (wallet_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_ledger_request_id
  ON credit_ledger (request_id) WHERE request_id IS NOT NULL;
ALTER TABLE credit_ledger ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users read own ledger" ON credit_ledger;
CREATE POLICY "Users read own ledger" ON credit_ledger
  FOR SELECT USING (
    wallet_id IN (SELECT id FROM credit_wallets WHERE owner_type = 'user' AND owner_id = auth.uid())
  );

-- ---------------------------------------------------------------------------
-- 7. Wallet auto-create on signup + backfill for existing users
-- ---------------------------------------------------------------------------
-- Wallets are created with period_end = now() and bundle 0 ON PURPOSE: the
-- first entitlements read triggers the lazy rollover, which grants the
-- tier's monthly credits immediately. This gives existing users their
-- initial grant the moment CREDITS_ENABLED flips on (spec §9 Phase C) and
-- new signups their free grant on first use — no separate seeding step.
--
-- SECURITY DEFINER does NOT switch search_path — the caller's is inherited.
-- GoTrue fires auth.users triggers as supabase_auth_admin, whose search_path
-- is 'auth', so an unqualified table name here would fail to resolve and
-- 500 every signup (lesson: 20260517010000_fix_signup_trigger_search_path.sql).
-- SET search_path AND schema-qualify, for defense in depth.
CREATE OR REPLACE FUNCTION public.create_credit_wallet_for_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  INSERT INTO public.credit_wallets (owner_type, owner_id, period_start, period_end)
  VALUES ('user', NEW.id, now() - INTERVAL '1 month', now())
  ON CONFLICT (owner_type, owner_id) DO NOTHING;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created_credit_wallet ON auth.users;
CREATE TRIGGER on_auth_user_created_credit_wallet
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION create_credit_wallet_for_user();

INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
SELECT 'user', id, now() - INTERVAL '1 month', now() FROM auth.users
ON CONFLICT (owner_type, owner_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 8. RPCs — the transactional piece today's counters lack
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
  IF p_kind NOT IN ('debit', 'overage_debit') THEN
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

-- Defensive: CREATE OR REPLACE does not replace across arities. If an earlier
-- 5-param draft of grant_credits was ever applied, it would survive as an
-- overload with default EXECUTE grants (re-opening the minting hole) and
-- make PostgREST overload resolution ambiguous. Free if it never existed.
DROP FUNCTION IF EXISTS public.grant_credits(UUID, INTEGER, TEXT, TEXT, JSONB);

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
  -- it's absent here.
  IF p_kind NOT IN ('monthly_grant', 'admin_grant', 'refund') THEN
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

-- Service-role only: these mutate money state and bypass RLS (SECURITY DEFINER).
-- Supabase grants EXECUTE on public functions to anon/authenticated by default
-- and exposes them at /rest/v1/rpc/* — without this REVOKE, any signed-in user
-- could mint credits into their own wallet. Same posture as
-- admin_search_users_by_email (20260520000000).
REVOKE EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.grant_credits(UUID, INTEGER, TEXT, TEXT, JSONB, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.debit_credits(UUID, INTEGER, TEXT, TEXT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.grant_credits(UUID, INTEGER, TEXT, TEXT, JSONB, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) TO service_role;
