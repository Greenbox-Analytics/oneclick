-- Add integration fields to projects table
ALTER TABLE projects ADD COLUMN IF NOT EXISTS slack_channel_id TEXT;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS drive_folder_id TEXT;

-- Per-project notification preferences (which events post to Slack)
CREATE TABLE IF NOT EXISTS project_notification_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  enabled BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(project_id, event_type)
);

ALTER TABLE project_notification_settings ENABLE ROW LEVEL SECURITY;

-- RLS: project owner and admins can manage settings
CREATE POLICY "Project members manage notification settings"
  ON project_notification_settings
  USING (
    EXISTS (
      SELECT 1 FROM project_members pm
      WHERE pm.project_id = project_notification_settings.project_id
        AND pm.user_id = auth.uid()
        AND pm.role IN ('owner', 'admin')
    )
  );

-- Inbound Slack @mention notifications
CREATE TABLE IF NOT EXISTS slack_notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  channel_id TEXT NOT NULL,
  sender_name TEXT NOT NULL,
  sender_avatar_url TEXT,
  message_text TEXT NOT NULL,
  slack_ts TEXT NOT NULL,
  is_read BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE slack_notifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users view own slack notifications"
  ON slack_notifications USING (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_slack_notifications_user_recent
  ON slack_notifications(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_project_notification_settings_project
  ON project_notification_settings(project_id);

-- Retention: auto-delete Slack notifications older than 90 days (requires pg_cron extension)
-- Run manually if pg_cron is enabled:
-- SELECT cron.schedule('cleanup-old-slack-notifications', '0 3 * * *',
--   $$DELETE FROM slack_notifications WHERE created_at < now() - interval '90 days'$$);
