-- ============================================================
-- Project-aware payout coverage.
--
-- royalty_payout_coverage previously had PK (payout_id, royalty_statement_id),
-- which physically allowed only ONE coverage row per (payout, statement) — fine
-- when a statement maps to a single project, but a royalty statement can carry
-- lines for multiple projects/artists. The settlement bucket is now
-- (payee, statement, project), so a payout covering a multi-project statement
-- must be able to carry one coverage row per project.
--
-- Widen the PK to include project_id (already a column). Safe on existing data:
-- today's coverage is single-project per statement, so every existing row is
-- still unique under the wider key (the prefix was already unique).
-- ============================================================

alter table public.royalty_payout_coverage
  drop constraint if exists royalty_payout_coverage_pkey;

alter table public.royalty_payout_coverage
  add constraint royalty_payout_coverage_pkey
  primary key (payout_id, royalty_statement_id, project_id);
