-- Remove Notion integration.
--
-- Drops orphaned rows for provider='notion' in all integration-keyed tables,
-- then rewrites the provider CHECK constraint on integration_connections to
-- exclude 'notion'. Finally updates the 'pro' tier entitlement to drop notion
-- from integrations_allowed.
--
-- Mirrors 20260516120000_remove_monday_integration.sql.
-- Idempotent: safe to re-run.

-- 1. Delete orphaned data
DELETE FROM integration_connections WHERE provider = 'notion';
DELETE FROM notification_settings WHERE provider = 'notion';
DELETE FROM sync_log WHERE provider = 'notion';
DELETE FROM board_tasks WHERE external_provider = 'notion';

-- 2. Rewrite the CHECK constraint on integration_connections.provider.
-- Postgres assigns the original constraint a generated name; drop by deduced
-- name then re-add. If a future provider list is added, edit this CHECK.
ALTER TABLE integration_connections
  DROP CONSTRAINT IF EXISTS integration_connections_provider_check;
ALTER TABLE integration_connections
  ADD CONSTRAINT integration_connections_provider_check
  CHECK (provider IN ('google_drive', 'slack'));

-- 3. Remove 'notion' from tier_entitlements.integrations_allowed for 'pro'.
-- Uses jsonb '-' operator (remove element from array).
UPDATE tier_entitlements
SET integrations_allowed = integrations_allowed - 'notion'
WHERE tier = 'pro';
