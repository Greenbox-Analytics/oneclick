-- project_expenses: per-project expense tracking for net-vs-gross royalty calculations.
-- An expense may optionally be linked to one or more works (tracks) via project_expense_works.
-- Untagged expenses are treated as project-wide and allocated proportionally at calc time.

CREATE TABLE project_expenses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  created_by UUID NOT NULL REFERENCES auth.users(id),
  description TEXT NOT NULL,
  amount NUMERIC(12, 2) NOT NULL CHECK (amount >= 0),
  category TEXT CHECK (
    category IN (
      'studio', 'mixing_mastering', 'marketing', 'travel',
      'equipment', 'distribution', 'other'
    )
  ),
  incurred_on DATE,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Join table for optional per-track tagging (an expense may apply to 0..N works)
CREATE TABLE project_expense_works (
  expense_id UUID NOT NULL REFERENCES project_expenses(id) ON DELETE CASCADE,
  work_id UUID NOT NULL REFERENCES works_registry(id) ON DELETE CASCADE,
  PRIMARY KEY (expense_id, work_id)
);

CREATE INDEX idx_project_expenses_project_id ON project_expenses(project_id);
CREATE INDEX idx_project_expense_works_expense_id ON project_expense_works(expense_id);
CREATE INDEX idx_project_expense_works_work_id ON project_expense_works(work_id);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_project_expenses_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER project_expenses_updated_at
  BEFORE UPDATE ON project_expenses
  FOR EACH ROW EXECUTE FUNCTION update_project_expenses_updated_at();

-- RLS: members can read; owner/admin/editor can write.
ALTER TABLE project_expenses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "project_expenses_select_members" ON project_expenses
  FOR SELECT USING (
    project_id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
  );

CREATE POLICY "project_expenses_insert_editors" ON project_expenses
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

CREATE POLICY "project_expenses_update_editors" ON project_expenses
  FOR UPDATE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

CREATE POLICY "project_expenses_delete_editors" ON project_expenses
  FOR DELETE USING (
    project_id IN (
      SELECT project_id FROM project_members
      WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
    )
  );

-- RLS for join table: gated through the parent expense's project membership.
ALTER TABLE project_expense_works ENABLE ROW LEVEL SECURITY;

CREATE POLICY "project_expense_works_select_members" ON project_expense_works
  FOR SELECT USING (
    expense_id IN (
      SELECT id FROM project_expenses
      WHERE project_id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid())
    )
  );

CREATE POLICY "project_expense_works_insert_editors" ON project_expense_works
  FOR INSERT WITH CHECK (
    expense_id IN (
      SELECT id FROM project_expenses
      WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );

CREATE POLICY "project_expense_works_delete_editors" ON project_expense_works
  FOR DELETE USING (
    expense_id IN (
      SELECT id FROM project_expenses
      WHERE project_id IN (
        SELECT project_id FROM project_members
        WHERE user_id = auth.uid() AND role IN ('owner', 'admin', 'editor')
      )
    )
  );
