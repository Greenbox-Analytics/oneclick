ALTER TABLE works_registry ADD COLUMN custom_work_type TEXT;

-- Data migration BEFORE constraint changes
UPDATE works_registry SET status = 'draft' WHERE status = 'disputed';

ALTER TABLE works_registry DROP CONSTRAINT IF EXISTS works_registry_work_type_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_work_type_check
  CHECK (work_type IN ('single', 'ep_track', 'album_track', 'composition', 'other'));

ALTER TABLE works_registry DROP CONSTRAINT IF EXISTS works_registry_status_check;
ALTER TABLE works_registry ADD CONSTRAINT works_registry_status_check
  CHECK (status IN ('draft', 'pending_approval', 'registered'));
