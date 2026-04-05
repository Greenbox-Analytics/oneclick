-- Enhanced board tasks: rich fields, junction tables, comments

-- 1. New columns on board_tasks
ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS start_date DATE;
ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS color TEXT;
ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS parent_task_id UUID REFERENCES board_tasks(id) ON DELETE SET NULL;
ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS is_parent BOOLEAN DEFAULT false;

-- Make column_id nullable (parent tasks don't belong to a column)
ALTER TABLE board_tasks ALTER COLUMN column_id DROP NOT NULL;

-- 2. Junction table: Task <-> Artists (many-to-many)
CREATE TABLE board_task_artists (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_id, artist_id)
);

ALTER TABLE board_task_artists ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own task artists"
  ON board_task_artists
  USING (EXISTS (
    SELECT 1 FROM board_tasks
    WHERE board_tasks.id = board_task_artists.task_id
    AND board_tasks.user_id = auth.uid()
  ));

-- 3. Junction table: Task <-> Projects (many-to-many)
CREATE TABLE board_task_projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_id, project_id)
);

ALTER TABLE board_task_projects ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own task projects"
  ON board_task_projects
  USING (EXISTS (
    SELECT 1 FROM board_tasks
    WHERE board_tasks.id = board_task_projects.task_id
    AND board_tasks.user_id = auth.uid()
  ));

-- 4. Junction table: Task <-> Contracts/project_files (many-to-many)
CREATE TABLE board_task_contracts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  project_file_id UUID NOT NULL REFERENCES project_files(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_id, project_file_id)
);

ALTER TABLE board_task_contracts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own task contracts"
  ON board_task_contracts
  USING (EXISTS (
    SELECT 1 FROM board_tasks
    WHERE board_tasks.id = board_task_contracts.task_id
    AND board_tasks.user_id = auth.uid()
  ));

-- 5. Comments on tasks
CREATE TABLE board_task_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id UUID NOT NULL REFERENCES board_tasks(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE board_task_comments ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own comments"
  ON board_task_comments
  USING (auth.uid() = user_id);

-- 6. Indexes
CREATE INDEX idx_board_task_artists_task ON board_task_artists(task_id);
CREATE INDEX idx_board_task_projects_task ON board_task_projects(task_id);
CREATE INDEX idx_board_task_contracts_task ON board_task_contracts(task_id);
CREATE INDEX idx_board_task_comments_task ON board_task_comments(task_id);
CREATE INDEX idx_board_tasks_due_date ON board_tasks(user_id, due_date);
CREATE INDEX idx_board_tasks_start_date ON board_tasks(user_id, start_date);
CREATE INDEX idx_board_tasks_parent ON board_tasks(parent_task_id);
CREATE INDEX idx_board_tasks_is_parent ON board_tasks(user_id, is_parent);

-- 7. Data migration: copy existing single FK values into junction tables
INSERT INTO board_task_artists (task_id, artist_id)
SELECT id, artist_id FROM board_tasks
WHERE artist_id IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO board_task_projects (task_id, project_id)
SELECT id, project_id FROM board_tasks
WHERE project_id IS NOT NULL
ON CONFLICT DO NOTHING;
