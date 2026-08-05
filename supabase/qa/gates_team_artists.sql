-- supabase/qa/gates_team_artists.sql
-- ============================================================================
-- Executable coverage for the team-owned-artists SQL layer. pytest mocks the
-- Supabase client and never reaches Postgres, so the predicates, the RLS
-- policies and the storage triggers have no other test.
--
-- Paste into the Supabase SQL editor. Raises on purpose at the end: the error
-- message IS the report, and the raise is what rolls the test data back.
--
-- Every variable is v_-prefixed. PL/pgSQL's variable_conflict defaults to
-- `error`, so a variable named `org_id` in `WHERE org_id = org_id` aborts the
-- whole block with "column reference org_id is ambiguous".
--
-- Gates 1-6 cover 20260803000001, 7-14 cover 20260803000002, 15-16 cover
-- 20260803000003.
--
-- Run the whole file after each of those migrations, and read the count in
-- context: after 20260803000001 alone, gates 1-6 pass and 7-11 fail because the
-- policies they exercise do not exist yet. Expected totals are 6 after
-- 20260803000001, 14 after 20260803000002, and 16 after 20260803000003.
-- ============================================================================
DO $$
DECLARE
  v_owner UUID; v_member UUID; v_stranger UUID;
  v_org UUID; v_personal UUID; v_team UUID;
  v_proj_team UUID; v_proj_personal UUID;
  v_pass INT := 0; v_fail INT := 0; v_report TEXT := '';
BEGIN
  SELECT id INTO v_owner    FROM auth.users ORDER BY created_at LIMIT 1;
  SELECT id INTO v_member   FROM auth.users ORDER BY created_at OFFSET 1 LIMIT 1;
  SELECT id INTO v_stranger FROM auth.users ORDER BY created_at OFFSET 2 LIMIT 1;
  IF v_member IS NULL OR v_stranger IS NULL THEN
    RAISE EXCEPTION 'need at least 3 auth.users rows to run these gates';
  END IF;

  INSERT INTO organizations (name, status) VALUES ('GATE artists', 'active') RETURNING id INTO v_org;
  DELETE FROM org_members WHERE org_id = v_org;  -- drop the auto-created admin
  INSERT INTO org_members (org_id, user_id, role, status) VALUES (v_org, v_owner, 'admin', 'active');
  INSERT INTO org_members (org_id, user_id, role, status) VALUES (v_org, v_member, 'member', 'active');

  INSERT INTO artists (name, email, user_id) VALUES ('GATE personal', 'g@x.test', v_owner)
    RETURNING id INTO v_personal;
  INSERT INTO artists (name, email, user_id, team_id) VALUES ('GATE team', 'g2@x.test', v_owner, v_org)
    RETURNING id INTO v_team;

  -- ---------------------------------------------------------------- 1..5 --
  IF can_access_artist(v_personal, v_owner) AND NOT can_access_artist(v_personal, v_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 1. personal artist: creator yes, teammate no';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 1. personal artist visibility';
  END IF;

  IF can_access_artist(v_team, v_member) AND NOT can_access_artist(v_team, v_stranger) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 2. team artist: active member yes, stranger no';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 2. team artist visibility';
  END IF;

  UPDATE org_members SET status = 'suspended' WHERE org_id = v_org AND user_id = v_member;
  IF NOT can_access_artist(v_team, v_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 3. suspended member loses access';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 3. suspended member still sees the team artist';
  END IF;
  UPDATE org_members SET status = 'active' WHERE org_id = v_org AND user_id = v_member;

  UPDATE organizations SET archived_at = now() WHERE id = v_org;
  IF NOT can_access_artist(v_team, v_member) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 4. archived org revokes access';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 4. archived org still grants access';
  END IF;
  UPDATE organizations SET archived_at = NULL WHERE id = v_org;

  IF can_access_artist(v_team, v_owner, TRUE) AND NOT can_access_artist(v_team, v_member, TRUE) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 5. admin variant: org admin yes, member no';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 5. admin variant role gate';
  END IF;

  -- ------------------------------------------------------------------- 6 --
  -- The lock trigger. Run as an end-user JWT: the CREATOR is the dangerous
  -- case, because their user_id still matches the row.
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_owner, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;
  BEGIN
    UPDATE artists SET team_id = NULL WHERE id = v_team;
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 6. creator DETACHED a team artist from its team';
  EXCEPTION WHEN insufficient_privilege THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 6. team_id cannot be changed under a user JWT';
  END;

  -- ---------------------------------------------------------------- 7..8 --
  -- Still the CREATOR, still under the authenticated role.
  IF EXISTS (SELECT 1 FROM artists WHERE id = v_team) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 7. creator still reads the team artist (as a member)';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 7. creator lost read on the team artist';
  END IF;

  -- v_owner is BOTH the creator and an org admin in this fixture, so a delete
  -- is legitimately allowed here. What gate 6 proved is that they lost their
  -- personal-OWNER rights; gate 10 proves a non-admin member cannot delete.
  -- Demote v_owner to 'member' before this block if you want the
  -- creator-but-not-admin case explicitly.
  BEGIN
    DELETE FROM artists WHERE id = v_team;
    IF EXISTS (SELECT 1 FROM artists WHERE id = v_team) THEN
      v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 8. delete silently affected no rows';
    ELSE
      v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 8. org admin can delete a team artist';
      -- put it back for the remaining gates
      INSERT INTO artists (id, name, email, user_id, team_id)
        VALUES (v_team, 'GATE team', 'g2@x.test', v_owner, v_org);
    END IF;
  EXCEPTION WHEN insufficient_privilege THEN
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 8. org admin was refused the delete';
  END;
  RESET ROLE;

  -- --------------------------------------------------------------- 9..10 --
  -- Now as the ordinary MEMBER: no admin role, so no delete, and no forging.
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_member, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;

  IF EXISTS (SELECT 1 FROM artists WHERE id = v_team)
     AND NOT EXISTS (SELECT 1 FROM artists WHERE id = v_personal) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 9. member sees the team artist, not the personal one';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 9. artists RLS';
  END IF;

  BEGIN
    INSERT INTO artists (name, email, user_id, team_id)
      VALUES ('GATE forged', 'f@x.test', v_member, gen_random_uuid());
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 10. insert into a foreign team was ALLOWED';
  EXCEPTION WHEN insufficient_privilege OR check_violation OR foreign_key_violation THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 10. cannot create an artist in a team you are not in';
  END;

  BEGIN
    DELETE FROM artists WHERE id = v_team;
    IF EXISTS (SELECT 1 FROM artists WHERE id = v_team) THEN
      v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11. ordinary member''s delete affected no rows';
    ELSE
      v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 11. ordinary member DELETED a team artist';
    END IF;
  EXCEPTION WHEN insufficient_privilege THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 11. ordinary member cannot delete a team artist';
  END;

  RESET ROLE;
  PERFORM set_config('request.jwt.claims', NULL, true);

  -- -------------------------------------------------------------- 12..14 --
  -- Policy gates that need real projects/files (20260803000002).
  INSERT INTO projects (name, artist_id) VALUES ('GATE p-team', v_team) RETURNING id INTO v_proj_team;
  INSERT INTO projects (name, artist_id) VALUES ('GATE p-pers', v_personal) RETURNING id INTO v_proj_personal;
  INSERT INTO project_files (project_id, file_name, file_url, file_size)
    VALUES (v_proj_team, 'c.wav', 'https://x.test/c.wav', 500);

  -- The creator backdoor. v_owner CREATED the team artist, so artists.user_id
  -- still points at them. Offboard them: every child-table policy must now
  -- deny, because each was re-scoped with `AND a.team_id IS NULL`.
  UPDATE org_members SET status = 'suspended' WHERE org_id = v_org AND user_id = v_owner;
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_owner, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;
  IF NOT EXISTS (SELECT 1 FROM projects WHERE id = v_proj_team)
     AND NOT EXISTS (SELECT 1 FROM project_files WHERE project_id = v_proj_team) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 12. offboarded CREATOR loses the team artist''s projects and files';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 12. offboarded creator still reads the team subtree (creator backdoor)';
  END IF;
  RESET ROLE;
  UPDATE org_members SET status = 'active' WHERE org_id = v_org AND user_id = v_owner;

  -- The one-statement walk-out: move a team file into a personal project.
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_member, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;
  BEGIN
    UPDATE project_files SET project_id = v_proj_personal WHERE project_id = v_proj_team;
    IF EXISTS (SELECT 1 FROM project_files WHERE project_id = v_proj_personal AND file_name = 'c.wav') THEN
      v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 13. member MOVED a team file into a personal project';
    ELSE
      v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 13. cross-owner file move affected no rows';
    END IF;
  EXCEPTION WHEN insufficient_privilege THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 13. cross-owner file move refused by the lock trigger';
  END;
  RESET ROLE;
  PERFORM set_config('request.jwt.claims', NULL, true);

  -- Credentials are admin-only, read included.
  INSERT INTO artist_credentials (artist_id, user_id, platform_name, login_identifier, password_ciphertext)
    VALUES (v_team, v_owner, 'GATE DSP', 'gate@x.test', 'ciphertext');
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_member, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;
  IF NOT EXISTS (SELECT 1 FROM artist_credentials WHERE artist_id = v_team) THEN
    v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 14. ordinary member cannot read a team artist''s DSP credentials';
  ELSE
    v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 14. ordinary member CAN read team credentials';
  END IF;
  RESET ROLE;
  PERFORM set_config('request.jwt.claims', NULL, true);

  -- -------------------------------------------------------------- 15..16 --
  -- Storage attribution (20260803000003). The only executable coverage the
  -- rewritten triggers have.
  DECLARE
    v_org_before BIGINT; v_user_before BIGINT;
  BEGIN
    SELECT COALESCE(storage_bytes, 0) INTO v_org_before FROM organizations WHERE id = v_org;
    SELECT COALESCE(total_storage_bytes, 0) INTO v_user_before FROM usage_counters WHERE user_id = v_owner;

    INSERT INTO project_files (project_id, file_name, file_url, file_size)
      VALUES (v_proj_team, 'a.wav', 'https://x.test/a.wav', 1000);
    INSERT INTO project_files (project_id, file_name, file_url, file_size)
      VALUES (v_proj_personal, 'b.wav', 'https://x.test/b.wav', 7);

    IF (SELECT storage_bytes FROM organizations WHERE id = v_org) = v_org_before + 1000
       AND COALESCE((SELECT total_storage_bytes FROM usage_counters WHERE user_id = v_owner), 0)
           = v_user_before + 7 THEN
      v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 15. team bytes hit the org, personal bytes hit the user';
    ELSE
      v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 15. storage attribution';
    END IF;

    DELETE FROM project_files WHERE project_id = v_proj_team AND file_name = 'a.wav';
    IF (SELECT storage_bytes FROM organizations WHERE id = v_org) = v_org_before THEN
      v_pass := v_pass + 1; v_report := v_report || E'\n[PASS] 16. deleting a team file gives the bytes back';
    ELSE
      v_fail := v_fail + 1; v_report := v_report || E'\n[FAIL] 16. team storage did not decrement';
    END IF;
  END;

  RAISE EXCEPTION E'\n=== TEAM ARTIST GATES: % passed, % failed. ALL TEST DATA ROLLED BACK. ===%\n',
    v_pass, v_fail, v_report;
END $$;
