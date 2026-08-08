-- Dropbox integration: allow the provider and extend the provenance table.

-- 1. integration_connections.provider: allow 'dropbox'.
--    Same drop/re-add pattern as 20260707000000_remove_notion_integration.sql.
ALTER TABLE integration_connections
  DROP CONSTRAINT IF EXISTS integration_connections_provider_check;
ALTER TABLE integration_connections
  ADD CONSTRAINT integration_connections_provider_check
  CHECK (provider IN ('google_drive', 'dropbox'));

-- 2. drive_sync_mappings doubles as the provenance/dedup table for all storage
--    providers. Existing rows are Drive rows (backfilled by the DEFAULT).
--    Dropbox rows store the Dropbox file id ("id:...") in drive_file_id and
--    reuse the existing sync_direction CHECK values, read provider-neutrally:
--    'from_drive' = imported from the remote, 'to_drive' = exported to it.
--    share_url stores the Dropbox shared link for exported files.
ALTER TABLE drive_sync_mappings
  ADD COLUMN IF NOT EXISTS provider TEXT NOT NULL DEFAULT 'google_drive',
  ADD COLUMN IF NOT EXISTS share_url TEXT;

-- 3. Grant Dropbox on every tier (free/basic/pro) — it is not a paid gate,
--    same as google_drive. Without this, GET /me/entitlements reports
--    integrationsAllowed without dropbox while the integration works.
UPDATE tier_entitlements
SET integrations_allowed = integrations_allowed || '["dropbox"]'::jsonb
WHERE NOT (integrations_allowed ? 'dropbox');

-- 4. One Dropbox export copy per (user, project file). Makes the double-click
--    race the database's problem instead of the UI's. Partial index: Dropbox
--    export rows only, so Drive rows and import provenance are unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS drive_sync_mappings_dropbox_export_uniq
  ON drive_sync_mappings (user_id, project_file_id)
  WHERE provider = 'dropbox' AND sync_direction = 'to_drive';
