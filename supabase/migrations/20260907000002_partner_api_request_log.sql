-- Per-request log for the partner API: the rate-limit counter, and the only
-- record of WHERE a key is being used from.
--
-- One row per auth attempt, written at the single chokepoint every partner
-- route shares (partner_api.router.get_partner_context). `key_id` is NULL on a
-- failed auth — there is no key to attribute it to — and `key_prefix` records
-- what was presented, but ONLY when it starts with mk_live_: a 12-char prefix
-- of our own scheme is not a credential, whereas a fragment of some other
-- system's secret pasted into the header by mistake would be.
--
-- A rolling window, not history. The daily billing sweep purges rows past
-- PARTNER_LOG_RETENTION_DAYS (default 30). Per-key SPEND history is unaffected
-- — that lives on credit_ledger.metadata.partner_key_id and is never purged.

CREATE TABLE IF NOT EXISTS partner_api_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key_id UUID NULL REFERENCES partner_api_keys(id) ON DELETE CASCADE,
  org_id UUID NULL REFERENCES organizations(id) ON DELETE CASCADE,
  key_prefix TEXT NULL,
  client_ip TEXT NULL,
  user_agent TEXT NULL,
  path TEXT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'invalid_key', 'rate_limited')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The rate-limit count: one key's rows inside a 60s window. On the hot path of
-- every partner request, so it leads with key_id.
CREATE INDEX IF NOT EXISTS idx_partner_api_requests_key_time
  ON partner_api_requests (key_id, created_at DESC);
-- The daily purge, and the "who is hammering us with dead keys" scan.
CREATE INDEX IF NOT EXISTS idx_partner_api_requests_time
  ON partner_api_requests (created_at DESC);

-- Deny-all: RLS on, zero policies — the same idiom partner_api_keys uses. Only
-- the service-role client touches this table. Client IPs are operator data.
ALTER TABLE partner_api_requests ENABLE ROW LEVEL SECURITY;
