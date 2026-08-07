-- External calendar subscriptions (import side of the workspace calendar).
--
-- The mirror of the .ics feed Msanii publishes: a user pastes the secret iCal URL
-- from Google/Apple/Outlook and those events render read-only in the workspace
-- calendar. Events are NOT stored — they're fetched and parsed on demand and held
-- in a short-lived in-process cache — so there is no sync state to reconcile and
-- nothing to clean up when an event is deleted upstream.
--
-- Deliberately NOT touching board_tasks: imported events never become tasks, so
-- they can't pollute the Kanban board or count against task limits.

CREATE TABLE IF NOT EXISTS calendar_subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  name TEXT NOT NULL,
  -- Last fetch outcome, surfaced in the UI so a dead/rotated URL is visible
  -- rather than silently showing an empty calendar.
  last_fetched_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, url)
);

CREATE INDEX IF NOT EXISTS idx_calendar_subscriptions_user
  ON calendar_subscriptions(user_id);

ALTER TABLE calendar_subscriptions ENABLE ROW LEVEL SECURITY;

-- A subscription URL is a bearer credential for someone's private calendar —
-- only its owner may ever read it.
CREATE POLICY calendar_subscriptions_owner ON calendar_subscriptions
  FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
