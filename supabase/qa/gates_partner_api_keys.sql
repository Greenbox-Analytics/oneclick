-- Paste into the Supabase SQL editor AFTER running 20260826000001.
-- Proves partner_api_keys is both invisible AND unwritable under an end-user
-- JWT (deny-all RLS): SELECT and INSERT are probed separately, because a later
-- permissive FOR INSERT policy would leave a read-only gate passing.
-- Seeds a real row first — an EMPTY table would also count 0 and fake a pass.
DO $$
DECLARE
  v_uid UUID;
  v_org UUID;
  v_count INT;
BEGIN
  SELECT id INTO v_uid FROM auth.users LIMIT 1;
  IF v_uid IS NULL THEN
    RAISE EXCEPTION 'SETUP FAIL: need at least 1 row in auth.users';
  END IF;

  -- Seed as superuser: throwaway org + one key row (the terminal RAISE rolls both back).
  INSERT INTO organizations (name, status) VALUES ('GATE partner keys', 'active') RETURNING id INTO v_org;
  INSERT INTO partner_api_keys (org_id, label, key_hash, key_prefix)
  VALUES (v_org, 'gate root', 'gate-hash-not-a-real-key', 'mk_live_gate');

  SELECT count(*) INTO v_count FROM partner_api_keys WHERE org_id = v_org;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'SETUP FAIL: expected 1 seeded key, found %', v_count;
  END IF;

  -- Impersonate an authenticated end user (same idiom as gates_team_artists.sql:96-98).
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', v_uid, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;

  SELECT count(*) INTO v_count FROM partner_api_keys;
  IF v_count <> 0 THEN
    RAISE EXCEPTION 'FAIL: end-user JWT can read % partner_api_keys rows', v_count;
  END IF;

  -- Writes: a refused INSERT raises 42501 (RLS check, or a missing GRANT —
  -- both are the refusal this gate wants). Reaching the RAISE means it landed.
  BEGIN
    INSERT INTO partner_api_keys (org_id, label, key_hash, key_prefix)
    VALUES (v_org, 'gate injected', 'gate-hash-injected', 'mk_live_bad');
    RAISE EXCEPTION 'FAIL: end-user JWT can INSERT into partner_api_keys';
  EXCEPTION
    WHEN insufficient_privilege THEN NULL;
  END;

  RESET ROLE;
  RAISE EXCEPTION 'PASS (2/2): partner_api_keys deny-all RLS holds for SELECT and INSERT — rollback on purpose';
END $$;
