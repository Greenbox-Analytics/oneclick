CREATE TABLE pending_project_invites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
  invited_by UUID NOT NULL REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ DEFAULT now() + interval '7 days',
  UNIQUE(project_id, email)
);

ALTER TABLE pending_project_invites ENABLE ROW LEVEL SECURITY;

CREATE POLICY "pending_invites_select" ON pending_project_invites
  FOR SELECT USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "pending_invites_insert" ON pending_project_invites
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "pending_invites_delete" ON pending_project_invites
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
    )
  );

-- On signup: convert pending invites to project_members
CREATE OR REPLACE FUNCTION process_pending_project_invites()
RETURNS TRIGGER SECURITY DEFINER SET search_path = public AS $$
BEGIN
  INSERT INTO project_members (project_id, user_id, role, invited_by)
  SELECT pi.project_id, NEW.id, pi.role, pi.invited_by
  FROM pending_project_invites pi
  WHERE LOWER(pi.email) = LOWER(NEW.email)
    AND pi.expires_at > now();

  DELETE FROM pending_project_invites
  WHERE LOWER(email) = LOWER(NEW.email);

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER process_pending_invites_on_signup
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION process_pending_project_invites();
