-- Partner API alias keys + org capability flag.
--
-- Root keys (parent_key_id IS NULL) are minted by a MSANII admin on the
-- product backend (/admin/orgs/{id}/partner-keys); per-user keys are
-- machine-minted under a root key and MUST carry the partner's opaque
-- user_ref (it lands in ledger metadata).
-- Deny-all RLS: only the service-role client touches this table — an
-- end-user JWT must never read a key hash.

CREATE TABLE IF NOT EXISTS partner_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  parent_key_id UUID NULL REFERENCES partner_api_keys(id) ON DELETE CASCADE,
  user_ref TEXT NULL,
  label TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  key_prefix TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
  expires_at TIMESTAMPTZ NULL,
  created_by UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at TIMESTAMPTZ NULL,
  -- A per-user (child) key must identify the partner's end user.
  CONSTRAINT partner_keys_child_has_user_ref CHECK (
    parent_key_id IS NULL OR user_ref IS NOT NULL
  )
);

CREATE INDEX IF NOT EXISTS idx_partner_api_keys_org ON partner_api_keys (org_id);
CREATE INDEX IF NOT EXISTS idx_partner_api_keys_parent ON partner_api_keys (parent_key_id);

-- Deny-all: RLS on, zero policies. Service-role bypasses RLS by design.
ALTER TABLE partner_api_keys ENABLE ROW LEVEL SECURITY;

-- Capability flag: set ONLY by a Msanii admin (see admin_router). Any signed-in
-- user can create an org, so a customer-writable dial would hand the partner
-- surface to anyone.
ALTER TABLE organizations
  ADD COLUMN IF NOT EXISTS partner_api_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- The API deliverable gets its OWN base rate. It is raw JSON with no UI, no
-- storage and no confirm step, so it will be priced apart from the product's
-- oneclick_run. Seeded equal to today's oneclick_run base; the row — not the
-- number — is the point. Public-read like every credit_prices row; NOT added
-- to Entitlements.to_dict()'s hand-built prices block (no client renders it).
INSERT INTO credit_prices (action, credits) VALUES ('partner_oneclick_run', 30)
ON CONFLICT (action) DO NOTHING;
