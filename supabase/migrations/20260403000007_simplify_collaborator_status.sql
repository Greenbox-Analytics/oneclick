-- Data migration BEFORE constraint changes
UPDATE registry_collaborators SET status = 'declined' WHERE status = 'disputed';

ALTER TABLE registry_collaborators DROP CONSTRAINT IF EXISTS registry_collaborators_status_check;
ALTER TABLE registry_collaborators ADD CONSTRAINT registry_collaborators_status_check
  CHECK (status IN ('invited', 'confirmed', 'declined', 'revoked'));

ALTER TABLE registry_collaborators DROP COLUMN IF EXISTS dispute_reason;
