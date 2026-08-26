-- Pre-signup tester designation (spec: 2026-08-08-admin-credits-testers).
-- An admin designates an email before any account exists; /me/bootstrap-tester
-- converts the row into a live tier_overrides tester grant on the user's first
-- VERIFIED sign-in and stamps claimed_at (conditional update = the claim mutex).
--
-- grant_duration_days is a DURATION applied at claim time (tier override
-- expires_at = claim + days) — deliberately not an absolute date, so a 30-day
-- grant claimed on day 29 still yields 30 days of access. There is no claim
-- deadline and no stale-row cleanup (YAGNI — admins revoke pending rows).
CREATE TABLE IF NOT EXISTS pending_tester_grants (
  email               TEXT PRIMARY KEY CHECK (email = lower(btrim(email))),  -- stored lowercased + trimmed
  reason              TEXT NOT NULL DEFAULT 'tester',
  grant_duration_days INTEGER CHECK (grant_duration_days > 0),
  credits             INTEGER CHECK (credits > 0),  -- initial allocation override; NULL = default
  created_by          TEXT NOT NULL,           -- admin email
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  claimed_at          TIMESTAMPTZ,
  claimed_user_id     UUID                     -- no FK on purpose: claim history must survive user deletion
);

-- Service-role only: RLS on, no policies. Matches other admin-plane tables.
ALTER TABLE pending_tester_grants ENABLE ROW LEVEL SECURITY;
