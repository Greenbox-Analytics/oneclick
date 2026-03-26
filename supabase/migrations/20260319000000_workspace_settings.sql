CREATE TABLE workspace_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  board_period TEXT NOT NULL DEFAULT 'monthly',
  custom_period_days INTEGER DEFAULT NULL,
  accent_color TEXT DEFAULT NULL,
  board_grouping TEXT DEFAULT 'column',
  use_24h_time BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id)
);

-- RLS: users can only read/write their own row
ALTER TABLE workspace_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY ws_settings_user ON workspace_settings FOR ALL USING (auth.uid() = user_id);
