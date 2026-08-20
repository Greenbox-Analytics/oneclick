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

-- The original CHECK (20260721000001) is inline on the status column, so it
-- carries Postgres's auto-generated name. Re-add it under the same name.
ALTER TABLE pending_org_invites DROP CONSTRAINT IF EXISTS pending_org_invites_status_check;

ALTER TABLE pending_org_invites ADD CONSTRAINT pending_org_invites_status_check
  CHECK (status IN ('pending', 'accepted', 'declined', 'expired'));

-- The sweep's only query: pending rows past their expiry. Partial on status so
-- it stays small — accepted/declined/expired rows are the bulk over time.
CREATE INDEX IF NOT EXISTS idx_pending_org_invites_expiring
  ON pending_org_invites (expires_at) WHERE status = 'pending';

COMMIT;
