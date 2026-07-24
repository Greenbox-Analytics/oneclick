-- supabase/migrations/20260721000001_licensing_core.sql
-- ============================================================================
-- Licensing Phase B — organizations, seats, org credit pool, transfer_credits.
--
-- Spec: docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §4
-- Plan: docs/superpowers/plans/2026-07-20-licensing-phase-b-core.md Task 1
--
-- Load-bearing rules restated (full numbered list lives at the top of the
-- plan; only the ones this migration is directly responsible for):
--  1. Seat/org wallets: period_start/period_end NULL FOREVER, ALL money in
--     reserve_balance, created only by the dedicated helpers (backend Task 4)
--     — never the user-wallet seeding trigger below. rollover_wallet gains a
--     DB-level RAISE on non-user wallets so "seat money never expires" is a
--     database guarantee, not just an application convention.
--  2. transfer_credits writes PAIRED ledger rows keyed request_id||':from' /
--     ':to' — idx_credit_ledger_request_id is UNIQUE GLOBALLY, so one shared
--     key would make every transfer a phantom duplicate on the second insert.
--     Fails on insufficient source (transfers never overdraw); touches
--     reserve_balance on BOTH sides only.
--  4. Orgs get NO last-admin auto-promote (inheriting a funded pool is
--     privilege escalation, unlike inheriting a team board). The guard
--     auto-archives the org on the account-deletion/cascade escape and
--     RAISEs on in-app removal instead of promoting a successor.
--
-- OPERATOR NOTE: apply this AFTER 20260720000000_protect_profiles_is_admin.sql
-- and 20260720000001_credit_packs.sql (both precede this file and are assumed
-- already applied). LICENSING_ENABLED stays false until the backend module
-- (Tasks 2-10) ships and QA runs a real org lifecycle end to end (spec §10).
-- This migration is WRITTEN ONLY — never run it from this task.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Tables (spec §4) — clone the teams patterns (roles, expiring email
-- invites, last-admin guard, SECURITY DEFINER membership helpers, RLS).
-- ---------------------------------------------------------------------------

CREATE TABLE organizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  -- Enterprise minimum = CUMULATIVE credits floor across pool purchases
  -- (CALIBRATE), not a seat-count floor (harder to game with ghost seats;
  -- natural in a credits-denominated model) and not single-purchase (which
  -- would quantize the floor to pack sizes and strand below-floor money in
  -- a pending org). NULL = platform default from env.
  min_initial_purchase_credits INTEGER,
  -- §4 allowance: sweep tops each active seat up to this monthly, pool
  -- permitting. NULL/0 = manual-only allocation.
  default_seat_allowance INTEGER,
  -- 'pending' until cumulative pool purchases >= the minimum (§ lifecycle). Seats
  -- confer enterprise entitlements ONLY while status='active' AND
  -- archived_at IS NULL — otherwise self-serve org creation would hand out
  -- unlimited caps (incl. storage) for $0.
  status TEXT NOT NULL CHECK (status IN ('pending','active','suspended')) DEFAULT 'pending',
  archived_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE org_members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('admin','member')),
  -- 'removed' is a SOFT state (round 5): the row survives removal as (a) the
  -- seat wallet's owner reference / audit chain and (b) the marker that
  -- exempts org-accrued storage from personal overage billing. Re-invite of a
  -- removed member reactivates this row (UNIQUE(org_id,user_id) holds).
  status TEXT NOT NULL CHECK (status IN ('active','suspended','removed')) DEFAULT 'active',
  -- Written ONCE at each active→suspended/removed transition (cleared on
  -- reactivation). The offboard reclaim's request_id uses its epoch: stable
  -- across retries of ONE offboard (replay converges), distinct across
  -- suspension cycles (a second offboard reclaims again). now()-at-reclaim
  -- would be unique per call = no replay protection at all.
  revoked_at TIMESTAMPTZ,
  invited_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(org_id, user_id)
);

CREATE TABLE pending_org_invites (      -- mirrors pending_team_invites exactly
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin','member')) DEFAULT 'member',
  token UUID NOT NULL DEFAULT gen_random_uuid(),
  status TEXT NOT NULL CHECK (status IN ('pending','accepted','declined')) DEFAULT 'pending',
  invited_by UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days')
);

CREATE TABLE credit_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  org_member_id UUID NOT NULL REFERENCES org_members(id) ON DELETE CASCADE,
  requested_credits INTEGER CHECK (requested_credits > 0),  -- NULL = "more, admin decides"
  note TEXT,
  status TEXT NOT NULL CHECK (status IN ('pending','approved','denied')) DEFAULT 'pending',
  resolved_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,   -- repo precedent: 20260518 fix_user_delete_cascades
  resolved_credits INTEGER,           -- what the admin actually granted
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ
);
-- Anti-spam: one open request per seat (each request emails every admin).
CREATE UNIQUE INDEX uq_credit_requests_pending
  ON credit_requests (org_member_id) WHERE status = 'pending';

CREATE INDEX idx_org_members_org_id ON org_members(org_id);
CREATE INDEX idx_org_members_user_id ON org_members(user_id);
-- Case-insensitive uniqueness so the re-invite dedup and the LOWER(email)
-- lookup/RLS agree (same idiom as uq_pending_team_invites_team_email).
CREATE UNIQUE INDEX uq_pending_org_invites_org_email ON pending_org_invites (org_id, LOWER(email));
CREATE INDEX idx_pending_org_invites_email ON pending_org_invites (LOWER(email));
CREATE INDEX idx_credit_requests_org_id ON credit_requests(org_id);
CREATE INDEX idx_credit_requests_org_member_id ON credit_requests(org_member_id);

-- updated_at triggers (reuse the repo-canonical function; pending_org_invites
-- and credit_requests have no updated_at column, same as their teams/pending
-- analogues).
CREATE TRIGGER organizations_updated_at
  BEFORE UPDATE ON organizations
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER org_members_updated_at
  BEFORE UPDATE ON org_members
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- ---------------------------------------------------------------------------
-- 2. profiles.billing_context_org_id — the billing-context switcher (spec §5).
-- Deliberately user-writable: it lives on `profiles`, which any signed-in
-- user can PATCH via the generic PostgREST client. That is a SECURITY
-- PROPERTY, not an oversight — EntitlementsService resolution (backend Task
-- 5) confers NOTHING unless the value matches the caller's own ACTIVE seat in
-- an ACTIVE, non-archived org; any foreign, stale, or forged value falls
-- closed to personal billing. The profiles.is_admin self-escalation guard
-- added in 20260720000000_protect_profiles_is_admin.sql is NOT touched and
-- must NOT be extended to cover this column — is_admin is privileged,
-- billing_context_org_id is a harmless preference by design.
-- ---------------------------------------------------------------------------
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS billing_context_org_id UUID;

-- ---------------------------------------------------------------------------
-- 3. credit_wallets.owner_type CHECK widens to include 'seat'. Catalog-driven
-- drop (same rationale as the tier CHECK widening in 20260713000002 — the
-- original CHECK is inline/unnamed, so a live DB may carry a different
-- generated name; dropping by guessed name would silently no-op).
-- ---------------------------------------------------------------------------
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
  CHECK (owner_type IN ('user', 'org', 'seat'));

-- credit_wallets.owner_id / credit_ledger.wallet_id intentionally carry NO FK
-- to organizations/org_members: owner_type is a polymorphic discriminator
-- (user/org/seat) so a single FK target is impossible without a
-- discriminated-union constraint, AND an ON DELETE CASCADE from org_members
-- would silently destroy money history the moment a seat row disappeared.
-- Orphan prevention here is a SERVICE responsibility, not a schema one:
-- offboarding is reclaim-then-transition (backend Task 3) and seat removal is
-- a SOFT status transition, never a bare DELETE — so no cascade can orphan a
-- wallet or ledger row silently.

-- ---------------------------------------------------------------------------
-- 4. Auto-create-admin trigger on organizations. AFTER INSERT, keyed on
-- NEW.created_by (NOT auth.uid(), which is NULL under the service role —
-- same rationale as auto_create_team_admin in 20260630000001). Atomic with
-- the org insert, so an org can never exist without an admin.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION auto_create_org_admin()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  IF NEW.created_by IS NOT NULL THEN
    INSERT INTO org_members (org_id, user_id, role, status, invited_by)
    VALUES (NEW.id, NEW.created_by, 'admin', 'active', NEW.created_by)
    ON CONFLICT (org_id, user_id) DO NOTHING;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER auto_create_org_admin_trigger
  AFTER INSERT ON organizations
  FOR EACH ROW EXECUTE FUNCTION auto_create_org_admin();

-- ---------------------------------------------------------------------------
-- 5. Last-admin guard trigger on org_members. BEFORE UPDATE OR DELETE.
-- Clones the teams v2 guard (20260703000000_fix_admin_guard_team_teardown_v2)
-- with ONE structural delta: orgs get NO auto-promote branch AT ALL —
-- inheriting a funded credit pool is privilege escalation (a board is
-- harmless to inherit; purchase/allocate/reclaim rights over real money are
-- not). Where teams would promote the longest-tenured member, orgs instead
-- auto-archive (cascade case) or RAISE (in-app case).
--
-- "Losing the last admin" here means: role demotion away from 'admin',
-- status leaving 'active' (suspend/remove), or a DELETE — evaluated only
-- against a row that WAS an active admin (OLD.role='admin' AND
-- OLD.status='active'), since is_org_admin() only counts active admins.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION org_members_admin_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  losing_admin BOOLEAN;
  is_cascade   BOOLEAN;
  other_admins INT;
BEGIN
  IF TG_OP = 'UPDATE' THEN
    losing_admin := (OLD.role = 'admin' AND OLD.status = 'active')
                    AND (NEW.role <> 'admin' OR NEW.status <> 'active');
  ELSE  -- DELETE
    losing_admin := (OLD.role = 'admin' AND OLD.status = 'active');
  END IF;

  IF NOT losing_admin THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  -- Whole-org teardown: the parent organizations row is already gone
  -- (deleted earlier in this command; the cascade fires after). Nothing to
  -- preserve — allow without archiving again.
  IF NOT EXISTS (SELECT 1 FROM organizations WHERE id = OLD.org_id) THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  is_cascade := pg_trigger_depth() > 1
                OR NOT EXISTS (SELECT 1 FROM auth.users WHERE id = OLD.user_id);

  SELECT count(*) INTO other_admins
  FROM org_members
  WHERE org_id = OLD.org_id AND role = 'admin' AND status = 'active' AND id <> OLD.id;

  IF other_admins > 0 THEN
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  -- Losing the ONLY admin. NO auto-promote — ever (spec §4: inheriting a
  -- funded pool is privilege escalation). The account-deletion cascade (or
  -- any other trigger-depth>1 cascade) must not raise — auto-archive the org
  -- and allow; in-app removal/demotion of the last admin is blocked outright.
  IF is_cascade THEN
    UPDATE organizations SET archived_at = now()
      WHERE id = OLD.org_id AND archived_at IS NULL;
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
  END IF;

  RAISE EXCEPTION 'You are the only admin of this organization — promote another member first';
END;
$$;

CREATE TRIGGER org_members_admin_guard_trigger
  BEFORE UPDATE OR DELETE ON org_members
  FOR EACH ROW EXECUTE FUNCTION org_members_admin_guard();

-- ---------------------------------------------------------------------------
-- 6. Membership-check helpers (SECURITY DEFINER to avoid recursive RLS).
-- Arg order (p_user_id, p_org_id) matches teams' is_team_member/is_team_admin
-- and the Python helpers in orgs/authz.py. Both require status='active' —
-- a suspended/removed seat confers no membership or admin rights.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION is_org_member(p_user_id UUID, p_org_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1 FROM org_members
    WHERE org_id = p_org_id AND user_id = p_user_id AND status = 'active'
  );
$$;

CREATE OR REPLACE FUNCTION is_org_admin(p_user_id UUID, p_org_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path TO 'public'
AS $$
  SELECT EXISTS (
    SELECT 1 FROM org_members
    WHERE org_id = p_org_id AND user_id = p_user_id AND role = 'admin' AND status = 'active'
  );
$$;

-- ---------------------------------------------------------------------------
-- 7. Row Level Security. SELECT-only everywhere — every write to these four
-- tables goes through the backend's service-role client, which bypasses RLS
-- entirely; per-endpoint ownership/role checks in orgs/authz.py ARE the
-- authz (repo convention — see reference_backend_authz_model). Adding client
-- write policies here would let an admin (or anyone, for INSERT) mutate
-- org/seat state directly via PostgREST, skipping the reclaim-then-transition
-- and transfer_credits invariants the service layer enforces.
-- ---------------------------------------------------------------------------
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE pending_org_invites ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "organizations_select_members" ON organizations
  FOR SELECT USING (is_org_member(auth.uid(), id));

CREATE POLICY "org_members_select_members" ON org_members
  FOR SELECT USING (is_org_member(auth.uid(), org_id));

-- pending_org_invites: org admins manage via the backend; an invitee can see
-- their own invite by email (clones teams' idiom exactly — LOWER(email)
-- against auth.jwt() ->> 'email', COALESCE-guarded against a missing claim).
CREATE POLICY "pending_org_invites_select_admins" ON pending_org_invites
  FOR SELECT USING (is_org_admin(auth.uid(), org_id));
CREATE POLICY "pending_org_invites_select_own_email" ON pending_org_invites
  FOR SELECT USING (LOWER(email) = LOWER(COALESCE(auth.jwt() ->> 'email', '')));

-- credit_requests: the requesting member reads their own request; org admins
-- read all requests for their org.
CREATE POLICY "credit_requests_select_member_or_admin" ON credit_requests
  FOR SELECT USING (
    is_org_admin(auth.uid(), org_id)
    OR EXISTS (
      SELECT 1 FROM org_members m
      WHERE m.id = credit_requests.org_member_id AND m.user_id = auth.uid()
    )
  );

-- ---------------------------------------------------------------------------
-- transfer_credits — THE org-money primitive (spec §4). Reserve-only on BOTH
-- sides; fails on insufficient source (transfers never overdraw — an admin
-- cannot allocate credits the pool doesn't hold); paired ledger rows carry
-- request_id || ':from' / ':to' because idx_credit_ledger_request_id is
-- UNIQUE GLOBALLY — one shared key would abort the second insert and make
-- every transfer converge as a phantom duplicate.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.transfer_credits(
  p_from_wallet UUID,
  p_to_wallet UUID,
  p_amount INTEGER,
  p_kind TEXT,
  p_request_id TEXT,
  p_metadata JSONB DEFAULT '{}'
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  v_existing RECORD;
  v_first UUID; v_second UUID;
  v_first_bundle INTEGER; v_first_reserve INTEGER;
  v_second_bundle INTEGER; v_second_reserve INTEGER;
  v_from_bundle INTEGER; v_from_reserve INTEGER;
  v_to_bundle INTEGER; v_to_reserve INTEGER;
BEGIN
  IF p_amount <= 0 THEN RAISE EXCEPTION 'transfer amount must be > 0'; END IF;
  IF p_kind NOT IN ('allocation', 'reclaim') THEN
    RAISE EXCEPTION 'invalid transfer kind %', p_kind;
  END IF;
  IF p_request_id IS NULL OR p_request_id = '' THEN
    RAISE EXCEPTION 'transfer requires a request_id';
  END IF;
  IF p_from_wallet = p_to_wallet THEN RAISE EXCEPTION 'cannot transfer to self'; END IF;

  -- Fast-path idempotency on the :from key (no lock yet).
  SELECT balance_after INTO v_existing FROM credit_ledger
    WHERE request_id = p_request_id || ':from';
  IF FOUND THEN RETURN jsonb_build_object('duplicate', true); END IF;

  -- Lock BOTH wallets in id order — every concurrent transfer locks in the
  -- same order regardless of direction, so allocate/reclaim can't deadlock.
  v_first := LEAST(p_from_wallet, p_to_wallet);
  v_second := GREATEST(p_from_wallet, p_to_wallet);
  SELECT bundle_balance, reserve_balance INTO v_first_bundle, v_first_reserve
    FROM credit_wallets WHERE id = v_first FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', v_first; END IF;
  SELECT bundle_balance, reserve_balance INTO v_second_bundle, v_second_reserve
    FROM credit_wallets WHERE id = v_second FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'wallet % not found', v_second; END IF;

  IF p_from_wallet = v_first THEN
    v_from_bundle := v_first_bundle;  v_from_reserve := v_first_reserve;
    v_to_bundle   := v_second_bundle; v_to_reserve   := v_second_reserve;
  ELSE
    v_from_bundle := v_second_bundle; v_from_reserve := v_second_reserve;
    v_to_bundle   := v_first_bundle;  v_to_reserve   := v_first_reserve;
  END IF;

  -- Re-check idempotency under the lock (racer committed while we waited).
  SELECT balance_after INTO v_existing FROM credit_ledger
    WHERE request_id = p_request_id || ':from';
  IF FOUND THEN RETURN jsonb_build_object('duplicate', true); END IF;

  -- Transfers never overdraw: RESERVE-only on both sides (seat/org money
  -- lives in reserve; bundle on these wallets is only accepted debit drift).
  IF v_from_reserve < p_amount THEN
    RAISE EXCEPTION 'insufficient balance on source wallet (have %, need %)',
      v_from_reserve, p_amount;
  END IF;

  v_from_reserve := v_from_reserve - p_amount;
  v_to_reserve := v_to_reserve + p_amount;

  -- Both UPDATEs + both INSERTs in ONE guarded subtransaction: a request_id
  -- collision on either insert rolls back the whole pair atomically.
  BEGIN
    UPDATE credit_wallets SET reserve_balance = v_from_reserve, updated_at = now()
      WHERE id = p_from_wallet;
    UPDATE credit_wallets SET reserve_balance = v_to_reserve, updated_at = now()
      WHERE id = p_to_wallet;
    INSERT INTO credit_ledger (wallet_id, delta, kind, request_id, balance_after, metadata)
      VALUES (p_from_wallet, -p_amount, p_kind, p_request_id || ':from',
              v_from_bundle + v_from_reserve,
              p_metadata || jsonb_build_object('other_wallet', p_to_wallet, 'direction', 'out'));
    INSERT INTO credit_ledger (wallet_id, delta, kind, request_id, balance_after, metadata)
      VALUES (p_to_wallet, p_amount, p_kind, p_request_id || ':to',
              v_to_bundle + v_to_reserve,
              p_metadata || jsonb_build_object('other_wallet', p_from_wallet, 'direction', 'in'));
  EXCEPTION WHEN unique_violation THEN
    RETURN jsonb_build_object('duplicate', true);
  END;

  RETURN jsonb_build_object('duplicate', false,
    'from_balance', v_from_bundle + v_from_reserve,
    'to_balance', v_to_bundle + v_to_reserve);
END;
$$;

-- Ledger kind CHECK gains 'allocation' + 'reclaim' — WRITTEN OUT, not prose
-- (Phase A's plan went stale in exactly this spot):
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
                  'expiry', 'storage_bill', 'purchase', 'clawback',
                  'allocation', 'reclaim'));

-- ---------------------------------------------------------------------------
-- rollover_wallet — re-created. Body is a byte-copy of the CURRENT definition
-- from 20260713000002_credits_schema.sql (lines 377-438), with ONE addition:
-- a guard as the FIRST statements of the body that RAISEs on any wallet whose
-- owner_type is not 'user'. "Seat/org money never expires" was previously an
-- application-level convention (_maybe_rollover_wallet's owner_type guard,
-- the sweep's owner_type='user' filter) — both of those defenses live in
-- code that could be bypassed by a future caller (repair script, new sweep
-- step) calling this RPC directly, since the RPC itself treats NULL
-- period_end as ROLLABLE on both of its locking-SELECT guards. This migration
-- makes the invariant a DATABASE guarantee instead.
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
  IF (SELECT owner_type FROM credit_wallets WHERE id = p_wallet_id) <> 'user' THEN
    RAISE EXCEPTION 'rollover_wallet: wallet % is not a user wallet — seat/org money never expires', p_wallet_id;
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

-- Service-role only: both RPCs mutate money state and bypass RLS (SECURITY
-- DEFINER). Supabase grants EXECUTE on public functions to anon/authenticated
-- by default and exposes them at /rest/v1/rpc/* — without this REVOKE, any
-- signed-in user could move credits between arbitrary wallets or force a
-- rollover. Same posture as every other money RPC (debit_credits,
-- grant_credits, 20260713000002 / 20260720000001).
REVOKE EXECUTE ON FUNCTION public.transfer_credits(UUID, UUID, INTEGER, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.transfer_credits(UUID, UUID, INTEGER, TEXT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) TO service_role;

COMMIT;
