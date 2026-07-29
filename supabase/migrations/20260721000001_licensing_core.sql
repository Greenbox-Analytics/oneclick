-- supabase/migrations/20260721000001_licensing_core.sql
-- ============================================================================
-- Licensing Phase B — organizations, members, ONE org credit pool, per-member caps.
--
-- Spec: docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §4
-- Plan: docs/superpowers/plans/2026-07-20-licensing-phase-b-core.md Task 1
--
-- Load-bearing rules restated (full numbered list lives at the top of the
-- plan; only the ones this migration is directly responsible for):
--  1. ONE wallet per org (owner_type='org'), created by the dedicated helper —
--     never the user-wallet seeding trigger below. Members hold NO wallet:
--     they spend from the org pool against a monthly cap enforced inside
--     debit_credits, under the same lock as the debit itself.
--  2. Two buckets, two lifetimes: the monthly contract dispersal lands in
--     bundle_balance and EXPIRES at each period end (so an org can't bank a
--     year of credits and burn them in one month); purchased packs land in
--     reserve_balance and never expire. rollover_wallet is what expires the
--     bundle, and carries a DB-level RAISE so it can only ever be pointed at
--     a user or org wallet.
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
  -- The negotiated contract volume: credits added to the POOL each month by
  -- the sweep. Lands in the pool's EXPIRING bucket, so an unspent month does
  -- not bank — otherwise the org could hoard a year of credits and burn them
  -- in one, which removes the COGS ceiling the dispersal model exists to give.
  -- 0 = no contract (pack purchases only).
  monthly_dispersal_credits INTEGER NOT NULL DEFAULT 0,
  -- Default per-member monthly CAP, applied when org_members.monthly_cap is
  -- NULL. A cap is a CEILING on the shared pool, never a reservation, so caps
  -- may deliberately sum to more than the dispersal (most members never reach
  -- theirs). NULL = uncapped members (the pool is the only limit).
  default_member_cap INTEGER,
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
  -- 'removed' is a SOFT state: the row survives removal as (a) the audit chain
  -- for everything this member spent from the pool and (b) the marker that
  -- exempts org-accrued storage from personal billing. Re-invite of a removed
  -- member reactivates this row (UNIQUE(org_id,user_id) holds). Nothing has to
  -- be reclaimed on the way out — a member never held credits, only a ceiling.
  status TEXT NOT NULL CHECK (status IN ('active','suspended','removed')) DEFAULT 'active',
  -- Written ONCE at each active→suspended/removed transition (cleared on
  -- reactivation). Kept as the offboarding audit timestamp.
  revoked_at TIMESTAMPTZ,
  -- This member's monthly ceiling on POOL spend. NULL = fall through to
  -- organizations.default_member_cap; NULL there too = uncapped.
  monthly_cap INTEGER CHECK (monthly_cap IS NULL OR monthly_cap >= 0),
  -- Spend counter for the current cap period, maintained INSIDE debit_credits
  -- under the wallet lock (see the RPC below) so two concurrent actions cannot
  -- both slip under the cap. Rolls when cap_period_end passes.
  cap_used INTEGER NOT NULL DEFAULT 0,
  cap_period_end TIMESTAMPTZ,
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
  -- A request to RAISE this member's monthly cap, not to move credits: nothing
  -- is allocated, so approving one is idempotent and costs the pool nothing
  -- until the member actually spends. NULL = "raise it, admin decides".
  requested_cap INTEGER CHECK (requested_cap > 0),
  note TEXT,
  status TEXT NOT NULL CHECK (status IN ('pending','approved','denied')) DEFAULT 'pending',
  resolved_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,   -- repo precedent: 20260518 fix_user_delete_cascades
  resolved_cap INTEGER,               -- the cap the admin actually set
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ
);
-- Anti-spam: one open request per member (each request emails every admin).
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
-- 3. credit_wallets.owner_type CHECK — re-asserted as ('user','org'). There is
-- no 'seat' owner: members spend from the ORG POOL against a monthly cap, so a
-- member never holds a wallet. Catalog-driven
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
  CHECK (owner_type IN ('user', 'org'));

-- credit_wallets.owner_id / credit_ledger.wallet_id intentionally carry NO FK
-- to organizations: owner_type is a polymorphic discriminator (user/org) so a
-- single FK target is impossible without a discriminated-union constraint, AND
-- an ON DELETE CASCADE would silently destroy money history. Orphan prevention
-- is a SERVICE responsibility: member removal is a SOFT status transition,
-- never a bare DELETE, and the pool outlives every membership row anyway.

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
-- inheriting a funded credit pool is privilege escalation (a board is harmless
-- to inherit; the right to buy credits and raise everyone's caps is not).
-- Where teams would promote the longest-tenured member, orgs instead
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
-- write policies here would let an admin (or anyone, for INSERT) mutate org or
-- membership state directly via PostgREST — including their OWN monthly_cap,
-- which is the whole enforcement mechanism.
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
-- No transfer primitive. In the dispersal model credits only ever enter the
-- ORG POOL (grant_credits: 'purchase' for packs, 'dispersal' for the monthly
-- contract volume) and leave it as member spend (debit_credits). Nothing moves
-- wallet-to-wallet, so there is no transfer to make atomic, no paired ledger
-- rows, no deadlock ordering to get right, and nothing to reclaim when someone
-- is offboarded — a member only ever held a ceiling, never a balance.
-- ---------------------------------------------------------------------------

-- Ledger kind CHECK gains 'dispersal' (the monthly contract top-up into the org
-- pool) — WRITTEN OUT, not prose (Phase A's plan went stale in exactly this
-- spot). 'allocation'/'reclaim' are deliberately absent: nothing transfers.
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
                  'expiry', 'purchase', 'clawback', 'dispersal'));

-- ---------------------------------------------------------------------------
-- debit_credits — re-created with a 7th parameter, p_member_id. Body is the
-- CURRENT definition (20260720000001_credit_packs.sql, which added the
-- reserve-only clamped 'clawback' branch) plus the member-cap block. Rebasing on
-- the credits_schema version instead would silently drop clawback support.
--
-- WHY THE CAP LIVES HERE AND NOT IN THE SERVICE: a cap is a promise about a
-- SHARED pool. A service-side pre-check cannot keep two of a member's own
-- concurrent actions from both passing it, so the counter has to move inside
-- the same transaction and the same wallet lock as the debit.
--
-- WHY IT RECORDS RATHER THAN REJECTS: debits are charge-on-success — by the
-- time this runs, the LLM call has already happened and been paid for. A
-- rejection here would hand out free work. So an over-cap debit is written
-- anyway, `cap_exceeded` is flagged in both the return value and the ledger
-- metadata, and the wall that actually stops members is the pre-check in
-- EntitlementsService.check_credits(). The only way past that pre-check is a
-- genuine race, and a race that overshoots by one action is visible in the
-- ledger rather than silent.
--
-- CREATE OR REPLACE does NOT replace across arities, so the 6-arg version is
-- dropped first — otherwise both overloads survive and PostgREST cannot
-- resolve /rest/v1/rpc/debit_credits at all.
-- ---------------------------------------------------------------------------
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
-- rollover_wallet — re-created. Body is a byte-copy of the CURRENT definition
-- from 20260713000002_credits_schema.sql, with ONE addition: a guard as the
-- FIRST statement that RAISEs on any wallet that is neither 'user' nor 'org'.
--
-- ORG pools DO roll: the monthly dispersal lands in the expiring bundle bucket
-- and is zeroed at each period end, which is what stops an org banking a year
-- of credits and burning them in one month. What never expires is the RESERVE
-- bucket — purchased packs — and rollover only touches the bundle.
--
-- The guard exists because the RPC treats a NULL period_end as ROLLABLE on
-- both of its locking-SELECT guards, so a future caller (repair script, new
-- sweep step) pointing it at the wrong wallet would silently expire money.
-- This makes the invariant a DATABASE guarantee rather than a convention.
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

-- Service-role only: both RPCs mutate money state and bypass RLS (SECURITY
-- DEFINER). Supabase grants EXECUTE on public functions to anon/authenticated
-- by default and exposes them at /rest/v1/rpc/* — without this REVOKE, any
-- signed-in user could move credits between arbitrary wallets or force a
-- rollover. Same posture as every other money RPC (debit_credits,
-- grant_credits, 20260713000002 / 20260720000001).
REVOKE EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.rollover_wallet(UUID, INTEGER, TIMESTAMPTZ, TIMESTAMPTZ) TO service_role;

COMMIT;
