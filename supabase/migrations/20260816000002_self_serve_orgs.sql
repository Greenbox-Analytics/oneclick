-- supabase/migrations/20260816000002_self_serve_orgs.sql
-- ============================================================================
-- Self-serve orgs: kind/coverage/lifecycle columns on organizations, a
-- 'lapsed' status wired into can_access_artist, transfer_credits (personal
-- reserve -> org pool, reserve-only, idempotent), and recurring-capable packs.
-- Spec 2026-08-15 §2, §4.
--
-- Four independent pieces, one transaction because Step 4's in-transaction
-- self-checks assert on what Steps 1-3 created -- if any of them didn't
-- actually land, the RPC and its grants roll back with everything else
-- instead of leaving a half-migrated DB that looks applied.
--
--  1. organizations: kind ('self_serve' | 'enterprise', backfilled
--     'enterprise' -- every existing org is enterprise, this migration is
--     what INTRODUCES self-serve), coverage (who's paying the topup
--     subscription and since when), lifecycle (grace_started_at,
--     dissolved_at), and status CHECK gains 'lapsed'.
--  2. can_access_artist (supabase/migrations/20260803000001_team_owned_artists.sql,
--     the only definition -- confirmed via
--     grep -rn "CREATE OR REPLACE FUNCTION public.can_access_artist" supabase/migrations/,
--     20260803000002 only CALLS it, never redefines it) is re-created
--     verbatim except the team branch also requires o.status <> 'lapsed'.
--     'pending'/'suspended' stay unchecked, same as before this migration.
--  3. credit_ledger's kind CHECK gains transfer_out/transfer_in (list is the
--     LIVE one from 20260730000001_dispersal_and_caps.sql, verified against
--     that migration's own catalog-drop block). credit_packs gets
--     recurring_stripe_price_id so a pack becomes monthly-buyable the moment
--     an operator sets a recurring Stripe Price on it -- no second catalog
--     table mirroring credit_packs field-for-field (ponytail).
--  4. transfer_credits(p_from_wallet, p_to_wallet, p_amount, p_request_id,
--     p_metadata): a member moving their OWN reserve credits into an org pool
--     they belong to. Reserve-only source (bundle expires monthly; moving it
--     would silently end that expiry for anyone who joins a team), personal
--     locked before pool (the only two-wallet RPC -- debit_credits takes
--     wallet->org_members, not wallet->wallet, so no lock-order cycle with
--     it), RAISE on insufficient reserve rather than clamping (the service
--     maps the message to HTTP 409 -- a silent partial transfer would be
--     worse than a rejected one), two ledger rows sharing one idempotency
--     key pair (request_id / request_id || ':in') inside one guarded
--     subtransaction so a retry after a partial failure can't double-move
--     credits.
--
--     Same name, unrelated shape, to a function 20260730000001 already
--     DROPPED (old signature: UUID, UUID, INTEGER, TEXT, TEXT, JSONB --
--     6 args, paired ':from'/':to' ledger keys, seat-wallet source). Step 6
--     audit (grep -rn "transfer_credits" supabase/ src/backend/scripts/):
--     the only remaining hits are in 20260721000001 (applied history, kept
--     verbatim, already documents its own removal) and 20260730000001 (the
--     removal itself, plus its own self-check that the function was gone).
--     No QA script or backend code calls the old 6-arg form -- nothing stale
--     to worry about; CREATE OR REPLACE here targets a distinct 5-arg
--     signature regardless.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- Step 1: organizations DDL + status CHECK. The original status CHECK
-- (20260721000001) is inline on the column, so it carries Postgres's
-- auto-generated name; re-add it under the same name with 'lapsed'.
-- ---------------------------------------------------------------------------
ALTER TABLE organizations
  ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'enterprise'
    CHECK (kind IN ('self_serve', 'enterprise')),
  ADD COLUMN IF NOT EXISTS covered_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS covered_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS grace_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS dissolved_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS topup_stripe_subscription_id TEXT,
  ADD COLUMN IF NOT EXISTS topup_admin_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE organizations DROP CONSTRAINT IF EXISTS organizations_status_check;
ALTER TABLE organizations ADD CONSTRAINT organizations_status_check
  CHECK (status IN ('pending', 'active', 'suspended', 'lapsed'));

-- ---------------------------------------------------------------------------
-- Step 2: can_access_artist -- the ONE definition of artist authority. Every
-- policy in 20260803000002 calls this instead of repeating the join, so
-- access can never drift between the ten tables that hang off an artist.
--
-- p_require_admin is a third argument rather than a second function because
-- the admin variant is this body plus one join clause; two near-identical
-- SECURITY DEFINER functions are two things to keep in step.
--
-- SECURITY DEFINER because it reads org_members, whose own RLS would
-- otherwise recurse when this is called from inside an artists policy.
--
-- Org status is MOSTLY not checked, only archived_at: a 'pending' org
-- (created, not yet paid up) is a team that is still being set up, and its
-- members should be able to build the roster they are about to pay for.
-- Billing has its own status='active' gate in resolve_billing_org_for_project
-- -- access and payment are different questions.
--
-- 'lapsed' is the one status this function now DOES check: a self-serve org
-- whose grace period ran out goes inert for its whole roster, admins
-- included -- same one-rule semantics as archived_at, just keyed off a
-- different column. 'pending'/'suspended' deliberately stay unchecked, as
-- above.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.can_access_artist(
  p_artist_id UUID,
  p_user_id UUID,
  p_require_admin BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1
      FROM artists a
      LEFT JOIN organizations o ON o.id = a.team_id
      LEFT JOIN org_members m
             ON m.org_id = a.team_id
            AND m.user_id = p_user_id
            AND m.status = 'active'
            AND (NOT p_require_admin OR m.role = 'admin')
     WHERE a.id = p_artist_id
       AND (
         -- personal: the creator, and only while it is not team-owned
         (a.team_id IS NULL AND a.user_id = p_user_id)
         -- team: an active member (admin, when required) of a live org
         -- Spec 2026-08-15 §2 (review finding 1): a lapsed self-serve org's
         -- roster is inert for EVERYONE, admins included -- same one-rule
         -- semantics as archived. 'pending'/'suspended' deliberately still
         -- unchecked here (pre-existing, documented in 20260803000001).
         OR (a.team_id IS NOT NULL AND m.id IS NOT NULL
             AND o.archived_at IS NULL AND o.status <> 'lapsed')
       )
  );
$$;

REVOKE EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.can_access_artist(UUID, UUID, BOOLEAN) TO authenticated, service_role;

-- ---------------------------------------------------------------------------
-- Step 3: ledger kinds + recurring-capable packs.
-- ---------------------------------------------------------------------------
ALTER TABLE credit_ledger DROP CONSTRAINT IF EXISTS credit_ledger_kind_check;
-- The live list (20260730000001) PLUS the two transfer kinds. NOT VALID +
-- guarded VALIDATE, same as that migration: a plain ADD CONSTRAINT scans all
-- existing rows and would fail on any DB holding legacy
-- storage_bill/allocation/reclaim rows.
ALTER TABLE credit_ledger ADD CONSTRAINT credit_ledger_kind_check
  CHECK (kind IN ('monthly_grant', 'debit', 'overage_debit', 'admin_grant', 'refund',
                  'expiry', 'purchase', 'clawback', 'dispersal',
                  'transfer_out', 'transfer_in')) NOT VALID;
DO $$
BEGIN
  ALTER TABLE credit_ledger VALIDATE CONSTRAINT credit_ledger_kind_check;
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'credit_ledger: legacy rows outside the kind list kept; the CHECK still binds new inserts';
END $$;

-- Recurring top-ups reuse the pack catalog (ponytail): a pack with this
-- price id set is buyable as a monthly subscription at the same credits/price.
-- Operator fills it per pack (Stripe recurring Price), like stripe_price_id.
ALTER TABLE credit_packs
  ADD COLUMN IF NOT EXISTS recurring_stripe_price_id TEXT;

-- ---------------------------------------------------------------------------
-- Step 4: transfer_credits -- a member moving their own reserve credits into
-- an org pool. See the header for the invariants; comments inline below.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.transfer_credits(
  p_from_wallet UUID,      -- personal wallet (owner_type='user')
  p_to_wallet UUID,        -- org pool (owner_type='org')
  p_amount INTEGER,
  p_request_id TEXT,
  p_metadata JSONB DEFAULT '{}'
) RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public'
AS $$
DECLARE
  v_existing RECORD;
  v_from RECORD; v_to RECORD;
BEGIN
  IF p_amount <= 0 THEN RAISE EXCEPTION 'transfer amount must be > 0'; END IF;
  IF p_request_id IS NULL THEN RAISE EXCEPTION 'transfer requires a request id'; END IF;

  -- Fast-path idempotency (transfer_out row carries p_request_id).
  SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
  IF FOUND THEN
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END IF;

  -- Lock order: PERSONAL first, POOL second — documented and fixed. The only
  -- two-wallet RPC; debit_credits takes wallet->org_members, so no cycle.
  SELECT id, owner_type, reserve_balance, bundle_balance INTO v_from
    FROM credit_wallets WHERE id = p_from_wallet FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', p_from_wallet; END IF;
  IF v_from.owner_type <> 'user' THEN RAISE EXCEPTION 'transfer source must be a personal wallet'; END IF;

  -- Re-check idempotency under the lock (racer committed while we waited).
  SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
  IF FOUND THEN
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END IF;

  -- RESERVE-ONLY source (spec §4): bundle credits expire monthly; pool reserve
  -- never does. Moving bundle would end expiry for anyone with a team.
  -- 409-not-clamp (review): the service maps this message to HTTP 409.
  IF v_from.reserve_balance < p_amount THEN
    RAISE EXCEPTION 'insufficient reserve: have %, need %', v_from.reserve_balance, p_amount;
  END IF;

  SELECT id, owner_type, reserve_balance, bundle_balance INTO v_to
    FROM credit_wallets WHERE id = p_to_wallet FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', p_to_wallet; END IF;
  IF v_to.owner_type <> 'org' THEN RAISE EXCEPTION 'transfer target must be an org pool'; END IF;

  -- Both UPDATEs and both INSERTs share ONE guarded subtransaction: a
  -- request_id collision must roll back all four or a retry double-moves.
  BEGIN
    UPDATE credit_wallets SET reserve_balance = reserve_balance - p_amount, updated_at = now()
      WHERE id = p_from_wallet;
    UPDATE credit_wallets SET reserve_balance = reserve_balance + p_amount, updated_at = now()
      WHERE id = p_to_wallet;
    INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
      VALUES (p_from_wallet, -p_amount, 'transfer_out', 'org_transfer', p_request_id,
              v_from.bundle_balance + v_from.reserve_balance - p_amount,
              p_metadata || jsonb_build_object('to_wallet', p_to_wallet));
    INSERT INTO credit_ledger (wallet_id, delta, kind, action, request_id, balance_after, metadata)
      VALUES (p_to_wallet, p_amount, 'transfer_in', 'org_transfer', p_request_id || ':in',
              v_to.bundle_balance + v_to.reserve_balance + p_amount,
              p_metadata || jsonb_build_object('from_wallet', p_from_wallet));
  EXCEPTION WHEN unique_violation THEN
    SELECT balance_after INTO v_existing FROM credit_ledger WHERE request_id = p_request_id;
    RETURN jsonb_build_object('duplicate', true, 'balance_after', v_existing.balance_after);
  END;

  RETURN jsonb_build_object('duplicate', false,
    'balance_after', v_from.bundle_balance + v_from.reserve_balance - p_amount);
END;
$$;

REVOKE EXECUTE ON FUNCTION public.transfer_credits(UUID, UUID, INTEGER, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.transfer_credits(UUID, UUID, INTEGER, TEXT, JSONB) TO service_role;

COMMIT;
