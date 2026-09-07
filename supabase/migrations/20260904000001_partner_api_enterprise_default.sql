-- Partner API: on by default for enterprise orgs (2026-09-04).
--
-- Enterprise orgs are created ONLY by a Msanii admin (POST /admin/orgs, or
-- PUT /admin/orgs/{id}/kind flipping to enterprise), which is exactly the
-- vetting `partner_api_enabled` was guarding against self-serve orgs — any
-- signed-in user can create one of those. So an enterprise org is born with
-- the bit on (admin_service.create_enterprise_org / set_org_kind); this
-- backfills the rows that predate that. Self-serve orgs stay off unless a
-- Msanii admin flips them explicitly. The bit remains admin-revocable.
--
-- Dissolved orgs are skipped: they are retained for support only, and the
-- surface would 409 on them anyway (_require_live_org).

UPDATE organizations
   SET partner_api_enabled = true
 WHERE kind = 'enterprise'
   AND dissolved_at IS NULL
   AND partner_api_enabled = false;
