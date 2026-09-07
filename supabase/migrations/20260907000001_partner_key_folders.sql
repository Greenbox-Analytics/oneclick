-- Key folders + inactive-key timestamp (spec: owner decisions 2026-09-06).
--
-- An inactive key (revoked or expired) is HIDDEN from the console after 30
-- days, never deleted: its spend still counts in the org's totals, series and
-- its folder's total. `revoked_at` is the clock that hiding needs — `status`
-- alone says nothing about when.
ALTER TABLE partner_api_keys ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ NULL;
UPDATE partner_api_keys SET revoked_at = now() WHERE status = 'revoked' AND revoked_at IS NULL;

-- Folders are the org's own grouping of keys by use case / project. Spend is
-- attributed by a key's CURRENT folder, so moving a key moves its history.
CREATE TABLE IF NOT EXISTS partner_key_folders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  name TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 80),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (org_id, name)
);

-- Deny-all: RLS on, zero policies — the same idiom partner_api_keys uses.
-- Only the service-role client touches this table.
ALTER TABLE partner_key_folders ENABLE ROW LEVEL SECURITY;

-- ON DELETE SET NULL: dropping a folder unfiles its keys, it never deletes
-- them (a key is a live credential; a folder is a label).
ALTER TABLE partner_api_keys ADD COLUMN IF NOT EXISTS folder_id UUID NULL REFERENCES partner_key_folders(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_partner_api_keys_folder ON partner_api_keys(folder_id);
