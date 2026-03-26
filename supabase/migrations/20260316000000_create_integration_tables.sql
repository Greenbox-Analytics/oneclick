-- Integration tables for Google Drive, Slack, Notion, Monday.com, and Project Boards

-- 1. OAuth token storage per user per provider
CREATE TABLE integration_connections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL CHECK (provider IN ('google_drive', 'slack', 'notion', 'monday')),
  access_token_encrypted TEXT NOT NULL,
  refresh_token_encrypted TEXT,
  token_expires_at TIMESTAMPTZ,
  provider_user_id TEXT,
  provider_workspace_id TEXT,
  provider_metadata JSONB DEFAULT '{}',
  scopes TEXT[],
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'revoked')),
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, provider)
);

ALTER TABLE integration_connections ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own connections"
  ON integration_connections USING (auth.uid() = user_id);

-- 2. Kanban board columns
CREATE TABLE board_columns (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  artist_id UUID REFERENCES artists(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  position INTEGER NOT NULL DEFAULT 0,
  color TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE board_columns ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own columns"
  ON board_columns USING (auth.uid() = user_id);

-- 3. Kanban board tasks
CREATE TABLE board_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  column_id UUID NOT NULL REFERENCES board_columns(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  position INTEGER NOT NULL DEFAULT 0,
  priority TEXT CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
  due_date DATE,
  artist_id UUID REFERENCES artists(id) ON DELETE SET NULL,
  project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
  assignee_name TEXT,
  labels TEXT[],
  external_id TEXT,
  external_provider TEXT,
  external_url TEXT,
  last_synced_at TIMESTAMPTZ,
  sync_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE board_tasks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own tasks"
  ON board_tasks USING (auth.uid() = user_id);

-- 4. Google Drive sync mappings
CREATE TABLE drive_sync_mappings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  project_file_id UUID REFERENCES project_files(id) ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  drive_file_id TEXT NOT NULL,
  drive_folder_id TEXT,
  sync_direction TEXT NOT NULL CHECK (sync_direction IN ('to_drive', 'from_drive', 'bidirectional')),
  last_synced_at TIMESTAMPTZ,
  drive_modified_at TIMESTAMPTZ,
  local_modified_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE drive_sync_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own drive mappings"
  ON drive_sync_mappings USING (auth.uid() = user_id);

-- 5. Per-user notification preferences per provider
CREATE TABLE notification_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  event_type TEXT NOT NULL,
  enabled BOOLEAN DEFAULT true,
  channel_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, provider, event_type)
);

ALTER TABLE notification_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own notification settings"
  ON notification_settings USING (auth.uid() = user_id);

-- 6. Sync audit log
CREATE TABLE sync_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('push', 'pull', 'bidirectional')),
  entity_type TEXT NOT NULL,
  entity_id UUID,
  external_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('success', 'conflict', 'error')),
  error_message TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE sync_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users view own sync logs"
  ON sync_log USING (auth.uid() = user_id);

-- Indexes for common queries
CREATE INDEX idx_integration_connections_user_provider ON integration_connections(user_id, provider);
CREATE INDEX idx_board_columns_user ON board_columns(user_id);
CREATE INDEX idx_board_tasks_column ON board_tasks(column_id);
CREATE INDEX idx_board_tasks_user ON board_tasks(user_id);
CREATE INDEX idx_drive_sync_user ON drive_sync_mappings(user_id);
CREATE INDEX idx_sync_log_user ON sync_log(user_id, created_at DESC);
