-- Partner API: ONE key type (2026-09-04, owner decision).
--
-- The backend/license hierarchy (parent_key_id + user_ref, machine-minted
-- per-customer keys) is gone. A key is simply an org credential: it resolves
-- to its org, spends that org's pool, and can do everything the API offers.
-- Nothing in the app reads or writes these two columns any more, and no key
-- rows existed when this shipped, so dropping is safe. Attribution per key
-- stays on credit_ledger.metadata.partner_key_id.

ALTER TABLE partner_api_keys DROP CONSTRAINT IF EXISTS partner_keys_child_has_user_ref;
DROP INDEX IF EXISTS idx_partner_api_keys_parent;
ALTER TABLE partner_api_keys
  DROP COLUMN IF EXISTS parent_key_id,
  DROP COLUMN IF EXISTS user_ref;
