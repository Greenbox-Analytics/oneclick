-- supabase/migrations/20260814000002_org_invite_48h_expiry.sql
-- ============================================================================
-- Org invites live 48 hours (was 7 days) and gain a terminal 'expired' state
-- so the sweep can tell the inviting admin the invite lapsed.
--
-- Acceptance was ALREADY blocked past expires_at (orgs.service.accept_invite
-- raises InviteInvalidError), so this adds no enforcement — it adds a state
-- worth NOTIFYING on. 'pending' past its expiry is indistinguishable from
-- 'pending and still live' when you only look at status, which is why the
-- sweep needs somewhere to write the transition exactly once.
--
-- OUTSTANDING INVITES ARE NOT SHORTENED: the new window applies to invites
-- created from here on. Retroactively cutting someone's 7-day invite to 48h
-- (possibly to a moment already past) would silently kill live invitations.
-- ============================================================================

BEGIN;

-- 48h from creation. Rows already out there keep the expires_at they were
-- stamped with; only new inserts pick this up.
ALTER TABLE pending_org_invites
  ALTER COLUMN expires_at SET DEFAULT (now() + interval '48 hours');

-- The original CHECK is inline and unnamed, so Postgres auto-named it — and a
-- restored or hand-edited DB may carry a different generated name. Guessing
-- the name would silently no-op and leave a CHECK that rejects 'expired',
-- surfacing only at the first sweep. Drop every status CHECK via the catalog,
-- then re-add one with a name we control (same approach as
-- 20260713000002's tier CHECK).
DO $$
DECLARE c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.pending_org_invites'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%status%'
  LOOP
    EXECUTE format('ALTER TABLE public.pending_org_invites DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;

ALTER TABLE pending_org_invites ADD CONSTRAINT pending_org_invites_status_check
  CHECK (status IN ('pending', 'accepted', 'declined', 'expired'));

-- The sweep's only query: pending rows past their expiry. Partial on status so
-- it stays small — accepted/declined/expired rows are the bulk over time.
CREATE INDEX IF NOT EXISTS idx_pending_org_invites_expiring
  ON pending_org_invites (expires_at) WHERE status = 'pending';

COMMIT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'pending_org_invites_status_check'
      AND pg_get_constraintdef(oid) ILIKE '%expired%'
  ) THEN
    RAISE EXCEPTION 'pending_org_invites status CHECK does not allow expired';
  END IF;
  RAISE NOTICE 'org invite 48h expiry applied';
END $$;
