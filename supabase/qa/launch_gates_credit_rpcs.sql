-- ============================================================================
-- LAUNCH GATES: credits/licensing money RPCs against the REAL database.
-- Spec: docs/superpowers/specs/2026-07-12-credits-billing-design.md §10
--       docs/superpowers/specs/2026-07-19-enterprise-licensing-credits-design.md §10
--
-- WHY THIS EXISTS: the pytest suite mocks sb.rpc(), so grant_credits /
-- debit_credits / transfer_credits / rollover_wallet have ZERO executable
-- coverage until they run against a real Postgres. This script is that run.
--
-- HOW TO RUN: paste the whole file into the Supabase SQL editor and execute.
-- It creates synthetic wallets (random owner ids — no FK, no real users
-- touched), runs every gate, then RAISES ON PURPOSE so the single enclosing
-- transaction rolls back and no test data survives. The raised error message
-- IS the report: read the [PASS]/[FAIL] lines in it. "All gates passed" still
-- ends in an exception — that is the rollback mechanism, not a failure.
--
-- NOT a migration. Lives in supabase/qa/ so nothing auto-applies it.
-- ============================================================================

DO $$
DECLARE
  w_user_a UUID;   -- bundle-first debit ordering + negative drift
  w_user_b UUID;   -- clawback clamp
  w_user_c UUID;   -- rollover happy path + double-roll guard
  w_org    UUID;   -- transfer source + rollover non-user rejection
  w_seat   UUID;   -- transfer destination
  v_res    JSONB;
  v_ok     BOOLEAN;
  v_bundle INTEGER; v_reserve INTEGER;
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
    RETURNING id INTO w_user_a;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO w_user_b;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('user', gen_random_uuid(), now() - interval '31 days', now() - interval '1 day')
    RETURNING id INTO w_user_c;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('org', gen_random_uuid(), NULL, NULL)
    RETURNING id INTO w_org;
  INSERT INTO credit_wallets (owner_type, owner_id, period_start, period_end)
    VALUES ('seat', gen_random_uuid(), NULL, NULL)
    RETURNING id INTO w_seat;

  -- ------------------------------------------------ 1. grant_credits happy --
  v_res := grant_credits(w_org, 100, 'purchase', 'reserve', '{}', 'gate-grant-1');
  IF (v_res->>'duplicate')::boolean = false AND (v_res->>'balance_after')::int = 100 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 1. grant_credits purchase->reserve: balance 100';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 1. grant_credits happy — got ' || v_res::text;
  END IF;

  -- ------------------------------------- 2. grant_credits idempotent replay --
  v_res := grant_credits(w_org, 100, 'purchase', 'reserve', '{}', 'gate-grant-1');
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = w_org;
  IF (v_res->>'duplicate')::boolean = true AND v_reserve = 100 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 2. grant replay: duplicate=true, no double-grant';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 2. grant replay — got ' || v_res::text || ', reserve=' || v_reserve;
  END IF;

  -- --------------------------------- 3. debit_credits bundle-first ordering --
  PERFORM grant_credits(w_user_a, 50, 'monthly_grant', 'bundle', '{}', NULL);
  PERFORM grant_credits(w_user_a, 100, 'admin_grant', 'reserve', '{}', NULL);
  v_res := debit_credits(w_user_a, 120, 'gate_probe', 'gate-debit-1', 'debit', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = w_user_a;
  SELECT metadata->>'from_reserve' INTO v_txt FROM credit_ledger WHERE request_id = 'gate-debit-1';
  IF (v_res->>'balance_after')::int = 30 AND v_bundle = 0 AND v_reserve = 30 AND v_txt = '70' THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 3. debit 120 of (50 bundle + 100 reserve): bundle drained first, from_reserve=70';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 3. debit ordering — balance_after=' || (v_res->>'balance_after') || ' bundle=' || v_bundle || ' reserve=' || v_reserve || ' from_reserve=' || COALESCE(v_txt, 'NULL');
  END IF;

  -- ------------------------- 4. debit negative drift lands on bundle (design) --
  v_res := debit_credits(w_user_a, 50, 'gate_probe', 'gate-debit-2', 'debit', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = w_user_a;
  IF (v_res->>'balance_after')::int = -20 AND v_bundle = -20 AND v_reserve = 0 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 4. overdraft drift: -20 on bundle (forgiven at rollover), reserve never negative';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 4. drift — balance_after=' || (v_res->>'balance_after') || ' bundle=' || v_bundle || ' reserve=' || v_reserve;
  END IF;

  -- ------------------------------------ 5. clawback: reserve-only + clamped --
  PERFORM grant_credits(w_user_b, 30, 'admin_grant', 'reserve', '{}', NULL);
  v_res := debit_credits(w_user_b, 50, 'admin_adjust', 'gate-claw-1', 'clawback', '{}');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = w_user_b;
  IF (v_res->>'removed')::int = 30 AND (v_res->>'shortfall')::int = 20
     AND v_reserve = 0 AND v_bundle = 0 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 5. clawback 50 of 30 reserve: removed=30, shortfall=20, clamped at zero';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 5. clawback clamp — got ' || v_res::text || ' bundle=' || v_bundle || ' reserve=' || v_reserve;
  END IF;

  -- --------------------------------------- 6. transfer_credits: allocation --
  v_res := transfer_credits(w_org, w_seat, 60, 'allocation', 'gate-xfer-1', '{}');
  SELECT COUNT(*) INTO v_n FROM credit_ledger
    WHERE request_id IN ('gate-xfer-1:from', 'gate-xfer-1:to');
  IF (v_res->>'from_balance')::int = 40 AND (v_res->>'to_balance')::int = 60 AND v_n = 2 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 6. allocation org->seat 60: balances 40/60, pair of :from/:to ledger rows';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 6. allocation — got ' || v_res::text || ', ledger pair count=' || v_n;
  END IF;

  -- ------------------------------------- 7. transfer idempotent replay --
  v_res := transfer_credits(w_org, w_seat, 60, 'allocation', 'gate-xfer-1', '{}');
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = w_seat;
  IF (v_res->>'duplicate')::boolean = true AND v_reserve = 60 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 7. transfer replay: duplicate=true, no double-move';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 7. transfer replay — got ' || v_res::text || ', seat reserve=' || v_reserve;
  END IF;

  -- ------------------------- 8. transfer insufficient source: raise, no writes --
  v_ok := false;
  BEGIN
    PERFORM transfer_credits(w_org, w_seat, 500, 'allocation', 'gate-xfer-2', '{}');
  EXCEPTION WHEN OTHERS THEN
    v_ok := SQLERRM LIKE '%insufficient balance on source wallet%';
    v_txt := SQLERRM;
  END;
  SELECT COUNT(*) INTO v_n FROM credit_ledger WHERE request_id LIKE 'gate-xfer-2%';
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = w_org;
  IF v_ok AND v_n = 0 AND v_reserve = 40 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 8. insufficient transfer: raised, zero ledger rows, source untouched';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 8. insufficient transfer — raised_ok=' || v_ok || ' (' || COALESCE(v_txt, 'no error') || ') rows=' || v_n || ' org reserve=' || v_reserve;
  END IF;

  -- ------------------------------------------ 9. transfer reclaim direction --
  v_res := transfer_credits(w_seat, w_org, 60, 'reclaim', 'gate-xfer-3', '{}');
  SELECT reserve_balance INTO v_reserve FROM credit_wallets WHERE id = w_org;
  IF (v_res->>'from_balance')::int = 0 AND (v_res->>'to_balance')::int = 100 AND v_reserve = 100 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9. reclaim seat->org 60: pool restored to 100';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9. reclaim — got ' || v_res::text || ', org reserve=' || v_reserve;
  END IF;

  -- ------------------------------------------- 10. transfer to self rejected --
  v_ok := false;
  BEGIN
    PERFORM transfer_credits(w_org, w_org, 1, 'allocation', 'gate-xfer-4', '{}');
  EXCEPTION WHEN OTHERS THEN
    v_ok := SQLERRM LIKE '%cannot transfer to self%';
  END;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 10. self-transfer rejected';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 10. self-transfer was not rejected';
  END IF;

  -- ----------------- 11. rollover_wallet REJECTS non-user wallets (DB guarantee) --
  -- THE licensing gate: "seat/org money never expires" must hold at the DB
  -- even if application code regresses (rule 11's backstop).
  v_ok := false;
  BEGIN
    PERFORM rollover_wallet(w_org, 0, now(), now() + interval '30 days');
  EXCEPTION WHEN OTHERS THEN
    v_ok := SQLERRM LIKE '%not a user wallet%';
    v_txt := SQLERRM;
  END;
  IF v_ok THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11. rollover_wallet(org wallet) raised: seat/org money never expires';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 11. rollover non-user guard — ' || COALESCE(v_txt, 'DID NOT RAISE');
  END IF;

  -- --------------------------- 12. rollover happy: expire bundle, keep reserve --
  PERFORM grant_credits(w_user_c, 10, 'monthly_grant', 'bundle', '{}', NULL);
  PERFORM grant_credits(w_user_c, 25, 'admin_grant', 'reserve', '{}', NULL);
  v_ok := rollover_wallet(w_user_c, 3000, now(), now() + interval '30 days');
  SELECT bundle_balance, reserve_balance INTO v_bundle, v_reserve FROM credit_wallets WHERE id = w_user_c;
  SELECT COUNT(*) INTO v_n FROM credit_ledger
    WHERE wallet_id = w_user_c AND kind = 'expiry' AND delta = -10;
  IF v_ok AND v_bundle = 3000 AND v_reserve = 25 AND v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12. rollover: leftover bundle 10 expired (ledger row), fresh 3000 granted, reserve 25 SURVIVES';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12. rollover happy — returned=' || v_ok || ' bundle=' || v_bundle || ' reserve=' || v_reserve || ' expiry rows=' || v_n;
  END IF;

  -- ----------------------------------------- 13. rollover double-roll guard --
  v_ok := rollover_wallet(w_user_c, 3000, now(), now() + interval '30 days');
  SELECT bundle_balance INTO v_bundle FROM credit_wallets WHERE id = w_user_c;
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

  SELECT COUNT(*) INTO v_n FROM pg_constraint WHERE conname = 'org_project_links_project_id_key';
  IF v_n = 1 THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14c. org_project_links UNIQUE(project_id) present (one org per project)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14c. org_project_links_project_id_key MISSING';
  END IF;

  -- Money RPCs must NOT be callable by signed-in users (minting hole).
  IF NOT has_function_privilege('authenticated', 'public.grant_credits(uuid, integer, text, text, jsonb, text)', 'EXECUTE')
     AND NOT has_function_privilege('authenticated', 'public.debit_credits(uuid, integer, text, text, text, jsonb)', 'EXECUTE')
     AND NOT has_function_privilege('authenticated', 'public.transfer_credits(uuid, uuid, integer, text, text, jsonb)', 'EXECUTE')
     AND NOT has_function_privilege('authenticated', 'public.rollover_wallet(uuid, integer, timestamptz, timestamptz)', 'EXECUTE')
     AND NOT has_function_privilege('anon', 'public.transfer_credits(uuid, uuid, integer, text, text, jsonb)', 'EXECUTE') THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14d. all four money RPCs revoked from authenticated/anon (service-role only)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14d. a money RPC is EXECUTABLE by authenticated/anon — signed-in users can mint/move credits';
  END IF;

  -- ------------------------------------------------- report + forced rollback --
  RAISE EXCEPTION E'\n=== LAUNCH GATES: % passed, % failed. ALL TEST DATA ROLLED BACK (this exception is the rollback mechanism, not a failure). ===%\n',
    v_pass, v_fail, v_report;
END $$;
