-- Relax the authoritative stake-total constraint from a strict 100% to 100% + a
-- small 0.5% margin. This absorbs tiny rounding from AI-parsed fractional splits
-- and keeps the DB in sync with the app-layer STAKE_TOTAL_MARGIN
-- (src/backend/registry/service.py) and the frontend SPLIT_TOTAL_MARGIN
-- (src/components/registry/splitsShared.ts). Original definition:
-- 20260329000000_create_rights_registry.sql (validate_stake_total).

create or replace function validate_stake_total()
returns trigger as $$
declare
  total numeric;
begin
  select coalesce(sum(percentage), 0) into total
  from ownership_stakes
  where work_id = new.work_id
    and stake_type = new.stake_type
    and id != coalesce(new.id, '00000000-0000-0000-0000-000000000000'::uuid);

  if (total + new.percentage) > 100.5 then
    raise exception 'Total %% for % would exceed 100%% (current: %%%, adding: %%%, allowed margin: 0.5%%)',
      new.stake_type, total, new.percentage;
  end if;
  return new;
end;
$$ language plpgsql;
