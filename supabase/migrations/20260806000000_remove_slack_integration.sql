-- Remove Slack integration.
--
-- Drops orphaned rows for provider='slack' in all integration-keyed tables,
-- then rewrites the provider CHECK constraint on integration_connections to
-- exclude 'slack'. Finally strips slack from tier_entitlements.integrations_allowed.
--
-- Mirrors 20260516120000_remove_monday_integration.sql and
-- 20260707000000_remove_notion_integration.sql. Google Drive is now the only provider.
--
-- DELIBERATELY LEFT IN PLACE (dormant, nothing reads them once the code is gone):
--   * slack_notifications          — the inbound @mention inbox (20260411000001)
--   * projects.slack_channel_id    — per-project channel link (20260411000001)
--   * project_notification_settings — provider-agnostic, reusable by a future integration
-- Drop them in a follow-up if the schema tidiness is worth the irreversible data loss.
--
-- Idempotent: safe to re-run.

-- 1. Delete orphaned data
DELETE FROM integration_connections WHERE provider = 'slack';
DELETE FROM notification_settings WHERE provider = 'slack';
DELETE FROM sync_log WHERE provider = 'slack';
DELETE FROM board_tasks WHERE external_provider = 'slack';

-- 2. Rewrite the CHECK constraint on integration_connections.provider.
-- Postgres assigns the original constraint a generated name; drop by deduced
-- name then re-add. If a future provider list is added, edit this CHECK.
ALTER TABLE integration_connections
  DROP CONSTRAINT IF EXISTS integration_connections_provider_check;
ALTER TABLE integration_connections
  ADD CONSTRAINT integration_connections_provider_check
  CHECK (provider IN ('google_drive'));

-- 3. Remove 'slack' from tier_entitlements.integrations_allowed on EVERY tier.
-- The monday/notion migrations targeted tier='pro' only; since then the tier keys
-- were renamed (20260728000001) to free/basic/pro, and slack was granted on more
-- than one tier — so this is unscoped on purpose.
UPDATE tier_entitlements
SET integrations_allowed = integrations_allowed - 'slack'
WHERE integrations_allowed ? 'slack';

-- 4. Same for any per-user override that pinned slack.
UPDATE tier_overrides
SET integrations_allowed = integrations_allowed - 'slack'
WHERE integrations_allowed ? 'slack';
