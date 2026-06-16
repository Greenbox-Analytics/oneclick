-- Add explicit is_released boolean so the works dashboard reflects the user's
-- choice from the Add Work wizard. Previously "released" was inferred from the
-- presence of release_date, which silently flipped the badge as soon as a date
-- was added later, even if the user originally chose "Unreleased."

alter table works_registry
  add column is_released boolean not null default true;

-- Backfill existing rows. We default to true so historical works keep showing
-- as released (matching what the dashboard was rendering before). Users can
-- flip individual works to unreleased via the Work editor.
update works_registry set is_released = true;
