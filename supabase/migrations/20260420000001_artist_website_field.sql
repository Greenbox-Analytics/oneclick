-- Replace Electronic Press Kit (EPK) field with Artist Website on the artists table.
ALTER TABLE artists ADD COLUMN IF NOT EXISTS additional_website TEXT;
ALTER TABLE artists DROP COLUMN IF EXISTS additional_epk;
