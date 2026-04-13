-- Fix: allow owner member row to be deleted when the entire project is being deleted.
-- The previous trigger blocked ALL owner deletions, including cascade from project delete.
-- New approach: only block owner deletion if the project still exists.

CREATE OR REPLACE FUNCTION prevent_owner_deletion()
RETURNS TRIGGER AS $$
BEGIN
  -- Allow cascade: if the project is being deleted, the row won't exist
  IF OLD.role = 'owner' AND EXISTS (
    SELECT 1 FROM projects WHERE id = OLD.project_id
  ) THEN
    RAISE EXCEPTION 'Cannot remove the project owner';
  END IF;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;
