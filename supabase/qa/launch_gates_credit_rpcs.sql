-- ============================================================================
-- LAUNCH GATES: credits/licensing money RPCs against the REAL database.
-- Spec: docs/superpowers/specs/2026-07-12-credits-billing-design.md §10
--       docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §10
--
-- WHY THIS EXISTS: the pytest suite mocks sb.rpc(), so grant_credits /
-- debit_credits / rollover_wallet have ZERO executable coverage until they
-- run against a real Postgres. This script is that run.
--
-- HOW TO RUN: paste the whole file into the Supabase SQL editor and execute.
-- It creates synthetic wallets (random owner ids — no FK, no real users
-- touched), runs every gate, then RAISES ON PURPOSE so the single enclosing
-- transaction rolls back and no test data survives. The raised error message
-- IS the report: read the [PASS]/[FAIL] lines in it. "All gates passed" still
-- ends in an exception — that is the rollback mechanism, not a failure.
--
-- Every variable is v_-prefixed. PL/pgSQL's variable_conflict defaults to
-- `error`, so a variable named `org_id` in `WHERE org_id = org_id` aborts the
-- whole block with "column reference org_id is ambiguous".
--
-- NOT a migration. Lives in supabase/qa/ so nothing auto-applies it.
-- ============================================================================

DO $$
DECLARE
  v_wallet_a UUID;  -- bundle-first debit ordering + negative drift
  v_wallet_b UUID;  -- clawback clamp
  v_wallet_c UUID;  -- rollover happy path + double-roll guard
  v_wallet_d UUID;  -- self-serve transfer_credits: personal wallet moving reserve into the org pool
  v_pool     UUID;  -- org pool: member spend, cap enforcement, dispersal rollover
  v_org_id   UUID;  -- organizations row backing the pool
  v_mem_id   UUID;  -- org_members row carrying the cap being enforced
  v_usr_id   UUID;  -- auth user behind that membership
  v_res    JSONB;
  v_ok     BOOLEAN;
  v_bundle INTEGER; v_reserve INTEGER;
  v_reserve_before INTEGER;
  v_n      INTEGER;
  v_txt    TEXT;
  v_pass   INTEGER := 0;
  v_fail   INTEGER := 0;
  v_report TEXT := '';

BEGIN
  -- ---------------------------------------------------------------- setup --
  -- owner_id carries no FK (polymorphic discriminator) — random ids are safe.
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO v_wallet_a;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO v_wallet_b;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO v_wallet_c;
  -- v_wallet_d: reserve 100 (admin_grant) + bundle 50 (monthly_grant) — the
  -- bundle is there so 12a-12d can prove the transfer leaves it untouched
  -- (transfer_credits is reserve-only; moving bundle would silently end its
  -- monthly expiry for anyone who joins a team).
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO v_wallet_d;
  PERFORM grant_credits(v_wallet_d, 100, 'admin_grant', 'reserve', '{}', NULL);
  PERFORM grant_credits(v_wallet_d, 50, 'monthly_grant', 'bundle', '{}', NULL);
  -- The org pool needs REAL organizations/org_members rows: debit_credits joins
  -- them to resolve the effective cap, so a random owner_id would not exercise
  -- the cap path at all.
  SELECT id INTO v_usr_id FROM auth.users LIMIT 1;
  INSERT INTO organizations (name, status, monthly_dispersal_credits, default_member_cap)
    VALUES ('GATE org', 'active', 500, 40)
    RETURNING id INTO v_org_id;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('org', v_org_id, now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO v_pool;
  -- auto_create_org_admin already made an admin row if a user exists; take it,
  -- else insert one so the cap gates have a member to bind to.
  SELECT id INTO v_mem_id FROM org_members WHERE org_id = v_org_id LIMIT 1;
  IF v_mem_id IS NULL THEN
    INSERT INTO org_members (org_id, user_id, role, status, monthly_cap)
      VALUES (v_org_id, v_usr_id, 'admin', 'active', 10)
      RETURNING id INTO v_mem_id;
  ELSE
    UPDATE org_members SET monthly_cap = 10, cap_used = 0, cap_period_end = NULL WHERE id = v_mem_id;
  END IF;

  -- ------------------------------------------------ 1. grant_credits happy --
  v_res := grant_credits(v_pool, 100, 'purchase', 'reserve', '{}', 'gate-grant-1');
  IF (v_res->>'duplicate')::boolean = false AND (v_res->>'balance_after')::int = 100 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 1. grant_credits purchase->reserve: balance 100';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 1. grant_credits happy — got ' || v_res::text;
  END IF;

  -- ------------------------------------- 2. grant_credits idempotent replay --
  v_res := grant_credits(v_pool, 100, 'purchase', 'reserve', '{}', 'gate-grant-1');
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = v_pool;
  IF (v_res->>'duplicate')::boolean = true AND v_reserve = 100 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 2. grant replay: duplicate=true, no double-grant';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 2. grant replay — got ' || v_res::text || ', reserve=' || v_reserve;
  END IF;

  -- --------------------------------- 3. debit_credits bundle-first ordering --
  PERFORM grant_credits(v_wallet_a, 50, 'monthly_grant', 'bundle', '{}', NULL);
  PERFORM grant_credits(v_wallet_a, 100, 'admin_grant', 'reserve', '{}', NULL);
  v_res := debit_credits(v_wallet_a, 120, 'gate_probe', 'gate-debit-1', 'debit', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = v_wallet_a;
  SELECT metadata->>'from_reserve' INTO v_txt FROM credit_ledger WHERE request_id = 'gate-debit-1';
  IF (v_res->>'balance_after')::int = 30 AND v_bundle = 0 AND v_reserve = 30 AND v_txt = '70' THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 3. debit 120 of (50 bundle + 100 reserve): bundle drained first, from_reserve=70';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 3. debit ordering — balance_after=' || (v_res->>'balance_after') || ' bundle=' || v_bundle || ' reserve=' || v_reserve || ' from_reserve=' || COALESCE(v_txt, 'NULL');
  END IF;

  -- ------------------------- 4. debit negative drift lands on bundle (design) --
  v_res := debit_credits(v_wallet_a, 50, 'gate_probe', 'gate-debit-2', 'debit', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = v_wallet_a;
  IF (v_res->>'balance_after')::int = -20 AND v_bundle = -20 AND v_reserve = 0 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 4. overdraft drift: -20 on bundle (forgiven at rollover), reserve never negative';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 4. drift — balance_after=' || (v_res->>'balance_after') || ' bundle=' || v_bundle || ' reserve=' || v_reserve;
  END IF;

  -- ------------------------------------ 5. clawback: reserve-only + clamped --
  PERFORM grant_credits(v_wallet_b, 30, 'admin_grant', 'reserve', '{}', NULL);
  v_res := debit_credits(v_wallet_b, 50, 'admin_adjust', 'gate-claw-1', 'clawback', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = v_wallet_b;
  IF (v_res->>'removed')::int = 30 AND (v_res->>'shortfall')::int = 20
     AND v_reserve = 0 AND v_bundle = 0 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 5. clawback 50 of 30 reserve: removed=30, shortfall=20, clamped at zero';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 5. clawback clamp — got ' || v_res::text || ' bundle=' || v_bundle || ' reserve=' || v_reserve;
  END IF;

  -- ------------------------------- 6. member spend draws the POOL + cap counter --
  -- The pool starts at 100 reserve (gate 1's grant) with the member capped at 10.
  -- The pool's period_end is deliberately in the PAST (sweep not run yet): the
  -- stamped cap window must still land in the future (20260805000003), or gate
  -- 7 would see the counter reset back to 0.
  v_res := debit_credits(v_pool, 3, 'zoe_message', 'gate-cap-1', 'debit', '{}', v_mem_id);
  SELECT cap_used, cap_period_end INTO v_n, v_txt FROM org_members WHERE id = v_mem_id;
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = v_pool;
  IF (v_res->>'cap_exceeded')::boolean = false AND (v_res->>'cap_used')::int = 3
     AND v_n = 3 AND v_reserve = 97 AND v_txt IS NOT NULL AND v_txt::timestamptz > now() THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 6. member debit 3: pool 100->97, cap_used=3 in the SAME txn, FUTURE period stamped';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 6. member debit — got ' || v_res::text || ' cap_used=' || v_n || ' reserve=' || v_reserve || ' period=' || COALESCE(v_txt, 'NULL');
  END IF;

  -- ------------------------------- 7. cap overshoot is RECORDED, not rejected --
  -- Charge-on-success means the work already happened; rejecting here would hand
  -- out free work. The debit lands, flagged, and the counter keeps moving.
  -- cap_used = 23 also proves the counter did NOT reset between debits while
  -- the pool sits un-rolled (the FIX 20260805000003 loophole).
  v_res := debit_credits(v_pool, 20, 'oneclick_run', 'gate-cap-2', 'debit', '{}', v_mem_id);
  SELECT cap_used INTO v_n FROM org_members WHERE id = v_mem_id;
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = v_pool;
  IF (v_res->>'cap_exceeded')::boolean = true AND v_n = 23 AND v_reserve = 77 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 7. over-cap debit: cap_exceeded=true, debit still recorded (pool 77, cap_used 23)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 7. over-cap debit — got ' || v_res::text || ' cap_used=' || v_n || ' reserve=' || v_reserve;
  END IF;

  -- ------------------------- 8. replay rolls back the cap counter too --
  -- The counter UPDATE shares the guarded subtransaction with the ledger INSERT,
  -- so a duplicate request_id must not eat the cap a second time.
  v_res := debit_credits(v_pool, 3, 'zoe_message', 'gate-cap-1', 'debit', '{}', v_mem_id);
  SELECT cap_used INTO v_n FROM org_members WHERE id = v_mem_id;
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = v_pool;
  IF (v_res->>'duplicate')::boolean = true AND v_n = 23 AND v_reserve = 77 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 8. member debit replay: duplicate=true, cap_used and pool unchanged';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 8. member debit replay — got ' || v_res::text || ' cap_used=' || v_n || ' reserve=' || v_reserve;
  END IF;

  -- --------------------- 9. the member id is stamped on the ledger row --
  SELECT COUNT(*) INTO v_n FROM credit_ledger
    WHERE request_id = 'gate-cap-1' AND (metadata->>'org_member_id')::uuid = v_mem_id;
  IF v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9. ledger row attributes the spend to the member (per-member reporting works)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9. ledger row missing metadata.org_member_id';
  END IF;

  -- -------------------- 9b. NULL member cap falls through to the org default --
  UPDATE org_members SET monthly_cap = NULL, cap_used = 0, cap_period_end = NULL WHERE id = v_mem_id;
  v_res := debit_credits(v_pool, 1, 'zoe_message', 'gate-cap-3', 'debit', '{}', v_mem_id);
  IF (v_res->>'cap')::int = 40 AND (v_res->>'cap_exceeded')::boolean = false THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9b. NULL member cap resolves to organizations.default_member_cap (40)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9b. cap fallthrough — got ' || v_res::text;
  END IF;

  -- ------- 9b-i. the -1 sentinel is NO LIMIT, not a ceiling of minus one --
  -- Every org now carries a default_member_cap (2,000 — 20260814000001), so
  -- NULL on the member row means "inherit", and -1 is the only way to say
  -- "no limit". Without the CASE that normalizes it, `cap_used + amount > -1`
  -- is true for every debit and EVERY ONE would be flagged cap_exceeded.
  UPDATE org_members SET monthly_cap = -1, cap_used = 0, cap_period_end = NULL WHERE id = v_mem_id;
  v_res := debit_credits(v_pool, 50, 'oneclick_run', 'gate-cap-unlim-1', 'debit', '{}', v_mem_id);
  IF (v_res->>'cap_exceeded')::boolean = false AND v_res->>'cap' IS NULL THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9b-i. member cap -1 = unlimited: 50 > org default 40, still not flagged, cap reported NULL';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9b-i. -1 member cap not treated as unlimited — got ' || v_res::text;
  END IF;

  -- ------- 9b-ii. an org-wide -1 default uncaps members who inherit it --
  UPDATE organizations SET default_member_cap = -1 WHERE id = v_org_id;
  UPDATE org_members SET monthly_cap = NULL, cap_used = 0, cap_period_end = NULL WHERE id = v_mem_id;
  v_res := debit_credits(v_pool, 1, 'zoe_message', 'gate-cap-unlim-2', 'debit', '{}', v_mem_id);
  IF (v_res->>'cap_exceeded')::boolean = false AND v_res->>'cap' IS NULL THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9b-ii. org default -1 uncaps an inheriting member';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9b-ii. org default -1 not treated as unlimited — got ' || v_res::text;
  END IF;
  -- Restore the fixture for the gates below.
  UPDATE organizations SET default_member_cap = 40 WHERE id = v_org_id;
  UPDATE org_members SET monthly_cap = NULL, cap_used = 0, cap_period_end = NULL WHERE id = v_mem_id;

  -- ------------------- 9c. clawback and overage reject a member id --
  v_ok := false;
  BEGIN
    PERFORM debit_credits(v_pool, 1, 'admin_adjust', 'gate-cap-4', 'clawback', '{}', v_mem_id);
  EXCEPTION WHEN OTHERS THEN
    v_ok := SQLERRM LIKE '%not member spend%';
  END;
  IF v_ok THEN
    v_ok := false;
    BEGIN
      PERFORM debit_credits(v_pool, 1, 'zoe_message', 'gate-cap-5', 'overage_debit', '{}', v_mem_id);
    EXCEPTION WHEN OTHERS THEN
      v_ok := SQLERRM LIKE '%pools have no pay-as-you-go%';
    END;
  END IF;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9c. clawback + overage both reject p_member_id (a cap counter must only track spend)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9c. clawback/overage accepted a member id';
  END IF;

  -- (Gate 10, seat-wallet allocation, was retired with seat wallets in
  -- 20260730000001 — the numbering gap is deliberate.)

  -- --------- 11. the dispersal rollover expires the pool BUNDLE, keeps reserve --
  -- THE licensing gate: an org must not be able to bank a year of monthly
  -- credits and burn them in one month, and must not lose credits it BOUGHT.
  PERFORM grant_credits(v_pool, 30, 'monthly_grant', 'bundle', '{}', NULL);
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve_before FROM credit_wallets WHERE id = v_pool;
  v_ok := rollover_wallet(v_pool, 500, now() - interval '1 day', now() + interval '30 days');
  SELECT bundle_balance, reserve_balance INTO v_n, v_reserve FROM credit_wallets WHERE id = v_pool;
  IF v_ok AND v_n = 500 AND v_reserve = v_reserve_before THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11. org dispersal rollover: unspent bundle (' || v_bundle || ') expired, new 500 granted, purchased reserve kept (' || v_reserve || ')';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 11. org dispersal rollover — rolled=' || v_ok || ' bundle=' || v_n || ' reserve=' || v_reserve || ' expected reserve=' || v_reserve_before;
  END IF;

  -- ------------- 11b. rollover still refuses anything that is not user/org --
  v_ok := false;
  BEGIN
    -- A wallet id that exists but with no owner_type in ('user','org') can't be
    -- constructed (the CHECK forbids it), so assert the guard via a missing id:
    -- the RPC must raise rather than silently no-op on an unknown wallet.
    PERFORM rollover_wallet(gen_random_uuid(), 0, now(), now() + interval '30 days');
  EXCEPTION WHEN OTHERS THEN
    v_ok := SQLERRM LIKE '%not found%' OR SQLERRM LIKE '%neither a user nor an org%';
    v_txt := SQLERRM;
  END;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11b. rollover_wallet raises on an unknown wallet instead of no-oping';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 11b. rollover unknown-wallet guard — ' || COALESCE(v_txt, 'DID NOT RAISE');
  END IF;

  -- --------------------------- 12. rollover happy: expire bundle, keep reserve --
  PERFORM grant_credits(v_wallet_c, 10, 'monthly_grant', 'bundle', '{}', NULL);
  PERFORM grant_credits(v_wallet_c, 25, 'admin_grant', 'reserve', '{}', NULL);
  v_ok := rollover_wallet(v_wallet_c, 3000, now(), now() + interval '30 days');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = v_wallet_c;
  SELECT COUNT(*) INTO v_n FROM credit_ledger
    WHERE wallet_id = v_wallet_c AND kind = 'expiry' AND delta = -10;
  IF v_ok AND v_bundle = 3000 AND v_reserve = 25 AND v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12. rollover: leftover bundle 10 expired (ledger row), fresh 3000 granted, reserve 25 SURVIVES';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12. rollover happy — returned=' || v_ok || ' bundle=' || v_bundle || ' reserve=' || v_reserve || ' expiry rows=' || v_n;
  END IF;

  -- ----------------------------------------- 13. rollover double-roll guard --
  v_ok := rollover_wallet(v_wallet_c, 3000, now(), now() + interval '30 days');
  SELECT bundle_balance INTO v_bundle FROM credit_wallets WHERE id = v_wallet_c;
  IF v_ok = false AND v_bundle = 3000 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 13. immediate re-roll returns false: no double monthly grant';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 13. double-roll guard — returned=' || v_ok || ' bundle=' || v_bundle;
  END IF;

  -- --------------------------------------------------- 14. catalog checks --
  SELECT COUNT(*) INTO v_n FROM pg_trigger
    WHERE tgrelid = 'public.profiles'::regclass
      AND tgname = 'protect_profiles_privileged_columns' AND NOT tgisinternal;
  IF v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14a. profiles is_admin protection trigger installed';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14a. profiles trigger MISSING — is_admin self-escalation is LIVE';
  END IF;

  SELECT COUNT(*) INTO v_n FROM pg_indexes
    WHERE schemaname = 'public' AND indexname = 'idx_credit_ledger_request_id';
  IF v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14b. ledger request_id unique index present (idempotency backbone)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14b. idx_credit_ledger_request_id MISSING';
  END IF;

  -- Was: org_project_links UNIQUE(project_id) (one org per project). That table
  -- was retired in 20260804000001 — a project belongs to an org because its
  -- ARTIST does, and an artist carries at most one team_id, so "one org per
  -- project" is now a column, not a constraint. What is worth asserting is that
  -- the FK is RESTRICT: deleting an org must never take a roster with it.
  SELECT COUNT(*) INTO v_n
    FROM pg_constraint
   WHERE conrelid = 'artists'::regclass
     AND contype = 'f'
     AND confrelid = 'organizations'::regclass
     AND confdeltype = 'r';
  IF v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14c. artists.team_id -> organizations is ON DELETE RESTRICT (one org per artist, roster protected)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14c. artists.team_id FK missing or not ON DELETE RESTRICT';
  END IF;

  -- Money RPCs must NOT be callable by signed-in users (minting hole).
  IF NOT has_function_privilege('authenticated', 'public.grant_credits(uuid, integer, text, text, jsonb, text)', 'EXECUTE')
     AND NOT has_function_privilege('authenticated', 'public.debit_credits(uuid, integer, text, text, text, jsonb, uuid)', 'EXECUTE')
     AND NOT has_function_privilege('authenticated', 'public.rollover_wallet(uuid, integer, timestamptz, timestamptz)', 'EXECUTE')
     AND NOT has_function_privilege('anon', 'public.debit_credits(uuid, integer, text, text, text, jsonb, uuid)', 'EXECUTE') THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14d. all three money RPCs revoked from authenticated/anon (service-role only)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14d. a money RPC is EXECUTABLE by authenticated/anon — signed-in users can mint/move credits';
  END IF;

  -- --------------------------------------------- 12a-12d. transfer_credits --
  -- Self-serve orgs (20260816000002): a member moves their OWN reserve
  -- credits into an org pool they belong to. Reserve-only source (bundle
  -- expires monthly and moving it would silently end that expiry), personal
  -- wallet locked before the pool, RAISE — not clamp — on insufficient
  -- reserve (the service maps that message to HTTP 409), two ledger rows
  -- sharing one idempotency key pair (request_id / request_id || ':in').
  -- Placed after the existing gates (not renumbered into them): nothing
  -- past gate 9c asserts an absolute value for v_pool's reserve_balance
  -- (gate 11 only compares the rollover's reserve AFTER to a v_reserve_before
  -- it captures itself), so 12a landing 40 more reserve on the pool here
  -- cannot break any later balance assertion.

  -- 12a. transfer happy path: 40 reserve -> pool; ledger rows on both sides
  v_res := transfer_credits(v_wallet_d, v_pool, 40, 'gate-xfer-1', '{}');
  SELECT reserve_balance, bundle_balance INTO v_reserve, v_bundle FROM credit_wallets WHERE id = v_wallet_d;
  SELECT COUNT(*) INTO v_n FROM credit_ledger
   WHERE request_id IN ('gate-xfer-1', 'gate-xfer-1:in')
     AND kind IN ('transfer_out', 'transfer_in');
  IF (v_res->>'duplicate')::boolean = false AND v_reserve = 60 AND v_bundle = 50 AND v_n = 2 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12a. transfer 40: reserve 100->60, bundle untouched, two ledger rows';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12a. transfer — ' || v_res::text || ' reserve=' || v_reserve || ' bundle=' || v_bundle || ' rows=' || v_n;
  END IF;

  -- 12b. replay: duplicate=true, nothing moves
  v_res := transfer_credits(v_wallet_d, v_pool, 40, 'gate-xfer-1', '{}');
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = v_wallet_d;
  IF (v_res->>'duplicate')::boolean = true AND v_reserve = 60 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12b. transfer replay is a no-op';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12b. replay — ' || v_res::text || ' reserve=' || v_reserve;
  END IF;

  -- 12c. reserve-only: 61 > 60 reserve MUST raise even though bundle holds 50
  v_ok := false;
  BEGIN
    PERFORM transfer_credits(v_wallet_d, v_pool, 61, 'gate-xfer-2', '{}');
  EXCEPTION WHEN OTHERS THEN v_ok := SQLERRM LIKE '%insufficient reserve%';
  END;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12c. bundle can never fund a transfer (insufficient reserve raised)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12c. over-reserve transfer did not raise';
  END IF;

  -- 12d. target must be an org pool
  v_ok := false;
  BEGIN
    PERFORM transfer_credits(v_wallet_d, v_wallet_a, 1, 'gate-xfer-3', '{}');
  EXCEPTION WHEN OTHERS THEN v_ok := SQLERRM LIKE '%must be an org pool%';
  END;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12d. user->user transfer refused';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12d. user->user transfer allowed';
  END IF;

  -- ------------------------------------------------- report + forced rollback --
  RAISE EXCEPTION E'\n=== LAUNCH GATES: % passed, % failed. ALL TEST DATA ROLLED BACK (this exception is the rollback mechanism, not a failure). ===%\n',
    v_pass, v_fail, v_report;
END $$;
