-- Drop two write-only columns from calendar_subscriptions.
--
-- last_fetched_at was written on every calendar load and read by nothing; updated_at
-- had no writer at all. Fix-forward over the applied 20260806000001 rather than an
-- edit to it, since that migration has already run.
--
-- last_error stays: it IS rendered, under each calendar's name in the import panel.
--
-- Idempotent: safe to re-run.

ALTER TABLE calendar_subscriptions DROP COLUMN IF EXISTS last_fetched_at;
ALTER TABLE calendar_subscriptions DROP COLUMN IF EXISTS updated_at;
