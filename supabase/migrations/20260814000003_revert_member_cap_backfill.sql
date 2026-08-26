-- supabase/migrations/20260814000003_revert_member_cap_backfill.sql
-- ============================================================================
-- REPAIR: undo the default_member_cap backfill that 20260814000001 shipped.
--
-- That migration set `default_member_cap = 2000` on every org that had NULL,
-- which capped the members of EXISTING orgs who were previously uncapped. The
-- 2,000 default is only meant to apply to orgs created from here on — the
-- column DEFAULT does that, and it is left in place. This reverts the row
-- writes and nothing else.
--
-- NOT reverted (all correct, all forward-only): the column DEFAULT, the -1
-- sentinel CHECKs, and the debit_credits replace. org_members.monthly_cap was
-- never touched by the backfill.
--
-- ---------------------------------------------------------------------------
-- HOW IT IDENTIFIES THE BACKFILLED ROWS
-- ---------------------------------------------------------------------------
-- `default_member_cap = 2000` on its own is NOT proof: the org settings UI
-- pre-fills 2000, so an admin who flipped the toggle on and saved without
-- editing wrote exactly that value deliberately. Clearing theirs would uncap
-- their members — the same class of mistake this migration exists to fix.
--
-- Two things separate the two cases:
--
--   1. TIME. The backfill ran on 2026-08-14, so anything set to 2000 before
--      that day is an admin's own choice and is left alone (CUTOFF below).
--
--   2. NEVER-UPDATED ROWS ARE EXCLUDED. An org created AFTER 20260814000001
--      picks up 2000 from the new column DEFAULT and must KEEP it — that is
--      the feature working. An INSERTed row has created_at = updated_at (both
--      `DEFAULT now()`, same transaction); the BEFORE UPDATE trigger only
--      fires on a real update. So `updated_at > created_at` admits exactly the
--      rows something wrote to after creation, which is what a backfill is.
--
--   3. A SHARED TIMESTAMP. `organizations_updated_at` fires
--      update_updated_at_column() -> `NEW.updated_at = now()`, and now() is
--      TRANSACTION-START time in Postgres — stable for every row in a
--      statement. So the backfill stamped one identical updated_at across
--      every row it touched, to the microsecond. An admin's separate save
--      lands on its own timestamp.
--
-- If the candidate rows do NOT all share one timestamp, something other than
-- the backfill also wrote 2000 in the window, this migration cannot tell which
-- is which, and it ABORTS rather than guessing. Resolve those by hand:
--
--   SELECT id, name, default_member_cap, updated_at FROM organizations
--    WHERE default_member_cap = 2000 ORDER BY updated_at DESC;
--   UPDATE organizations SET default_member_cap = NULL WHERE id IN (...);
--
-- On an abort, the statements after it report "current transaction is aborted".
-- That is the abort working — nothing was changed. Read the RAISE above it.
--
-- IDEMPOTENT: reverted rows become NULL and no longer match `= 2000`, so a
-- second run finds nothing.
-- ============================================================================

BEGIN;

DO $$
DECLARE
  -- The day 20260814000001 was applied. Orgs set to 2000 BEFORE this are an
  -- admin's own choice and are never touched. Widen only if the backfill ran
  -- earlier than you think; narrowing it further is always safe.
  v_cutoff CONSTANT TIMESTAMPTZ := '2026-08-14 00:00:00+00';
  v_stamps INTEGER;
  v_rows   INTEGER;
  v_when   TIMESTAMPTZ;
  v_names  TEXT;
BEGIN
  SELECT COUNT(DISTINCT updated_at), COUNT(*), MIN(updated_at)
    INTO v_stamps, v_rows, v_when
    FROM organizations
   WHERE default_member_cap = 2000
     AND updated_at >= v_cutoff
     AND updated_at > created_at;

  IF v_rows = 0 THEN
    RAISE NOTICE 'Nothing to revert: no org was UPDATED to default_member_cap = 2000 on or after %.', v_cutoff;
    RETURN;
  END IF;

  IF v_stamps > 1 THEN
    SELECT string_agg(format('  %s (%s) updated_at=%s', name, id, updated_at), E'\n' ORDER BY updated_at)
      INTO v_names
      FROM organizations
     WHERE default_member_cap = 2000
       AND updated_at >= v_cutoff
       AND updated_at > created_at;
    RAISE EXCEPTION E'Refusing to guess: % candidate orgs span % distinct updated_at values, so at least one 2000 was set by an admin rather than the backfill.\n%\nA single bulk UPDATE stamps ONE timestamp — pick the backfilled ids by hand (see this file''s header) and clear those.',
      v_rows, v_stamps, v_names;
  END IF;

  -- One timestamp across every candidate: the signature of the single bulk
  -- UPDATE the backfill ran. Safe to revert wholesale.
  SELECT string_agg(format('  %s (%s)', name, id), E'\n' ORDER BY name)
    INTO v_names
    FROM organizations
   WHERE default_member_cap = 2000
     AND updated_at >= v_cutoff
     AND updated_at > created_at;

  UPDATE organizations
     SET default_member_cap = NULL
   WHERE default_member_cap = 2000
     AND updated_at >= v_cutoff
     AND updated_at > created_at;

  RAISE NOTICE E'Reverted % org(s) backfilled at % back to NULL (members uncapped again):\n%',
    v_rows, v_when, v_names;
END $$;

COMMIT;

DO $$
BEGIN
  -- The forward-only half of 20260814000001 must survive this repair: new orgs
  -- still start their members capped at 2,000.
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'organizations'
      AND column_name = 'default_member_cap' AND column_default = '2000'
  ) THEN
    RAISE EXCEPTION 'organizations.default_member_cap DEFAULT is no longer 2000 — new orgs would start uncapped';
  END IF;
  RAISE NOTICE 'backfill revert complete; new-org default of 2000 intact';
END $$;
