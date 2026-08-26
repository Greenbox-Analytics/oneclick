-- Team storage pools now match the tier's personal storage (owner decision,
-- 2026-08-16): Basic 100 GiB (was 10), Pro 250 GiB then PAYG (was 100).
-- The per-OWNER pool semantics (orgs/storage_guard.pool_state) are unchanged;
-- only the dials move. storage_guard.is_pro_like is keyed on the resolved
-- tier, not pool size, so Basic's bigger pool stays a HARD cap.
UPDATE tier_entitlements SET team_storage_bytes = 107374182400 WHERE tier = 'basic'; -- 100 GiB
UPDATE tier_entitlements SET team_storage_bytes = 268435456000 WHERE tier = 'pro';   -- 250 GiB

