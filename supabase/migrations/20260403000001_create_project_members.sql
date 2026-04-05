-- project_members: dual-layer access control (project-level roles)
CREATE TABLE project_members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
  invited_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(project_id, user_id)
);

-- Ensure at most one owner per project
CREATE UNIQUE INDEX one_owner_per_project ON project_members (project_id) WHERE role = 'owner';

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_project_members_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER project_members_updated_at
  BEFORE UPDATE ON project_members
  FOR EACH ROW EXECUTE FUNCTION update_project_members_updated_at();

-- Prevent owner deletion
CREATE OR REPLACE FUNCTION prevent_owner_deletion()
RETURNS TRIGGER AS $$
BEGIN
  IF OLD.role = 'owner' THEN
    RAISE EXCEPTION 'Cannot remove the project owner';
  END IF;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER prevent_owner_deletion_trigger
  BEFORE DELETE ON project_members
  FOR EACH ROW EXECUTE FUNCTION prevent_owner_deletion();

-- Prevent owner role changes and promotion to owner
CREATE OR REPLACE FUNCTION protect_owner_role()
RETURNS TRIGGER AS $$
BEGIN
  IF OLD.role = 'owner' AND NEW.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot change the project owner role';
  END IF;
  IF NEW.role = 'owner' AND OLD.role != 'owner' THEN
    RAISE EXCEPTION 'Cannot promote to owner';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER protect_owner_role_trigger
  BEFORE UPDATE ON project_members
  FOR EACH ROW EXECUTE FUNCTION protect_owner_role();

-- Auto-create owner when project is created (SECURITY DEFINER to bypass RLS)
CREATE OR REPLACE FUNCTION auto_create_project_owner()
RETURNS TRIGGER SECURITY DEFINER AS $$
BEGIN
  IF auth.uid() IS NOT NULL THEN
    INSERT INTO project_members (project_id, user_id, role)
    VALUES (NEW.id, auth.uid(), 'owner');
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_create_project_owner_trigger
  AFTER INSERT ON projects
  FOR EACH ROW EXECUTE FUNCTION auto_create_project_owner();

-- RLS
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY "project_members_select_members" ON project_members
  FOR SELECT USING (
    project_id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );

CREATE POLICY "project_members_insert_admins" ON project_members
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_update_admins" ON project_members
  FOR UPDATE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "project_members_delete_admins" ON project_members
  FOR DELETE USING (
    (project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    ))
    OR
    (user_id = auth.uid() AND role != 'owner')
  );

CREATE INDEX idx_project_members_project_id ON project_members(project_id);
CREATE INDEX idx_project_members_user_id ON project_members(user_id);
