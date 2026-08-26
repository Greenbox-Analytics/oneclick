-- Boards on Teams — can_access_board() truth table.
-- Run in the Supabase SQL editor AFTER 20260818000001_boards_to_orgs.sql.
--
-- Reports the way its two siblings do (gates_team_artists.sql,
-- launch_gates_credit_rpcs.sql): the pass/fail tally is accumulated into
-- v_report and delivered by a RAISE EXCEPTION at the end. That is deliberate
-- and it is NOT a failure — the SQL editor does not surface RAISE NOTICE, so
-- the error message IS the report, and the raise is what rolls every row of
-- test data back (including the credit wallets the auth.users signup triggers
-- create, whose owner_id is polymorphic with no FK and so would otherwise
-- survive a cascade).
--
-- This file is the ONLY executable coverage of the SQL predicate: pytest mocks
-- sb.rpc()/table() and never reaches Postgres, and MockQueryBuilder discards
-- filter arguments, so seat status and per-user listing can only be proven here.
DO $$
DECLARE
  u_owner  UUID := gen_random_uuid();  -- org admin, creates both team boards
  u_member UUID := gen_random_uuid();  -- plain active member
  u_listed UUID := gen_random_uuid();  -- plain member, listed on the restricted board
  u_susp   UUID := gen_random_uuid();  -- suspended seat
  u_out    UUID := gen_random_uuid();  -- not in the org at all
  v_org UUID; b_open UUID; b_restricted UUID; b_personal UUID;
  v_pass INTEGER := 0;
  v_fail INTEGER := 0;
  v_report TEXT := '';
BEGIN
  INSERT INTO auth.users (id, email) VALUES
    (u_owner,  'qa-bot-owner@example.com'),
    (u_member, 'qa-bot-member@example.com'),
    (u_listed, 'qa-bot-listed@example.com'),
    (u_susp,   'qa-bot-susp@example.com'),
    (u_out,    'qa-bot-out@example.com');

  INSERT INTO organizations (name, kind, status, covered_by, covered_at)
    VALUES ('QA Boards Org', 'self_serve', 'active', u_owner, now())
    RETURNING id INTO v_org;

  INSERT INTO org_members (org_id, user_id, role, status) VALUES
    (v_org, u_owner,  'admin',  'active'),
    (v_org, u_member, 'member', 'active'),
    (v_org, u_listed, 'member', 'active'),
    (v_org, u_susp,   'member', 'suspended');

  INSERT INTO boards (team_id, owner_id, name) VALUES (v_org, u_owner, 'open')
    RETURNING id INTO b_open;
  INSERT INTO boards (team_id, owner_id, name, restricted) VALUES (v_org, u_owner, 'restricted', true)
    RETURNING id INTO b_restricted;
  INSERT INTO board_members (board_id, user_id) VALUES (b_restricted, u_listed);
  INSERT INTO boards (owner_id, name) VALUES (u_member, 'personal')
    RETURNING id INTO b_personal;

  -- 1. personal board: only its owner
  IF can_access_board(b_personal, u_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 1. personal: owner sees it';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 1. personal: owner CANNOT see it';
  END IF;

  IF NOT can_access_board(b_personal, u_owner) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 2. personal: non-owner denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 2. personal: non-owner GRANTED (leak)';
  END IF;

  -- 2. open team board: every live active seat, nobody else
  IF can_access_board(b_open, u_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 3. open: active member sees it';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 3. open: active member CANNOT see it';
  END IF;

  IF NOT can_access_board(b_open, u_out) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 4. open: outsider denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 4. open: outsider GRANTED (cross-tenant leak)';
  END IF;

  IF NOT can_access_board(b_open, u_susp) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 5. open: suspended seat denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 5. open: suspended seat GRANTED';
  END IF;

  -- 3. restricted board: owner, org admin, and listed members only
  IF can_access_board(b_restricted, u_owner) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 6. restricted: owner/admin sees it';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 6. restricted: owner/admin CANNOT see it';
  END IF;

  IF can_access_board(b_restricted, u_listed) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 7. restricted: listed member sees it';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 7. restricted: listed member CANNOT see it';
  END IF;

  IF NOT can_access_board(b_restricted, u_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 8. restricted: unlisted member denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 8. restricted: unlisted member GRANTED (narrowing broken)';
  END IF;

  -- 4. org lifecycle denies the whole subtree, admins included
  UPDATE organizations SET archived_at = now() WHERE id = v_org;
  IF NOT can_access_board(b_open, u_owner) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9. archived org: even the admin is denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9. archived org: admin still GRANTED';
  END IF;

  UPDATE organizations SET archived_at = NULL, status = 'lapsed' WHERE id = v_org;
  IF NOT can_access_board(b_open, u_owner) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 10. lapsed org: even the admin is denied';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 10. lapsed org: admin still GRANTED';
  END IF;

  UPDATE organizations SET status = 'active' WHERE id = v_org;
  IF can_access_board(b_open, u_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11. reactivated org: access restored';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 11. reactivated org: access NOT restored';
  END IF;

  -- 5. the trigger that RLS cannot express (WITH CHECK cannot see OLD).
  --    Under a service-role/superuser session auth.uid() is NULL, so the move
  --    is allowed here — this pins that the backend path stays open. The
  --    end-user-JWT refusal is covered by scripts/qa_boards_on_teams.py case 7.
  BEGIN
    UPDATE boards SET team_id = NULL WHERE id = b_open;
    UPDATE boards SET team_id = v_org WHERE id = b_open;
    v_pass := v_pass + 1;
    v_report := v_report || E'\n[PASS] 12. boards_lock_team_id_trg allows a service-role move (auth.uid() IS NULL)';
  EXCEPTION WHEN insufficient_privilege THEN
    v_fail := v_fail + 1;
    v_report := v_report || E'\n[FAIL] 12. boards_lock_team_id_trg blocks the BACKEND too (auth.uid() should be NULL here)';
  END;

  RAISE EXCEPTION E'\n=== BOARD ACCESS GATES: % passed, % failed. ALL TEST DATA ROLLED BACK. ===%\n',
    v_pass, v_fail, v_report;
END $$;
