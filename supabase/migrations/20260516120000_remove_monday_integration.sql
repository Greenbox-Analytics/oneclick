-- Remove Monday.com integration.
--
-- Drops orphaned rows for provider='monday' in all integration-keyed tables,
-- then rewrites the provider CHECK constraint on integration_connections to
-- exclude 'monday'. Finally updates the 'pro' tier entitlement to drop monday
-- from integrations_allowed.
--
-- Idempotent: safe to re-run.

-- 1. Delete orphaned data
DELETE FROM integration_connections WHERE provider = 'monday';
DELETE FROM notification_settings WHERE provider = 'monday';
DELETE FROM sync_log WHERE provider = 'monday';
DELETE FROM board_tasks WHERE external_provider = 'monday';

-- 2. Rewrite the CHECK constraint on integration_connections.provider.
-- Postgres assigns the original constraint a generated name; drop by deduced
-- name then re-add. If a future provider list is added, edit this CHECK.
ALTER TABLE integration_connections
  DROP CONSTRAINT IF EXISTS integration_connections_provider_check;
ALTER TABLE integration_connections
  ADD CONSTRAINT integration_connections_provider_check
  CHECK (provider IN ('google_drive', 'slack', 'notion'));

-- 3. Remove 'monday' from tier_entitlements.integrations_allowed for 'pro'.
-- Uses jsonb '-' operator (remove element from array).
UPDATE tier_entitlements
SET integrations_allowed = integrations_allowed - 'monday'
WHERE tier = 'pro';
