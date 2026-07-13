-- Allow SoundExchange (US non-interactive digital performance) stakes.
-- These are tracked separately from master ownership: SoundExchange pays
-- performers/rights owners directly, so its percentages must never be
-- counted toward the master split total.
-- The validate_stake_total trigger is per-(work_id, stake_type), so the
-- new type gets its own <=100% enforcement without changes.

alter table ownership_stakes
  drop constraint ownership_stakes_stake_type_check;

alter table ownership_stakes
  add constraint ownership_stakes_stake_type_check
  check (stake_type in ('master', 'publishing', 'soundexchange'));
