export type IntegrationProvider = "google_drive" | "slack" | "notion" | "monday";

export type ConnectionStatus = "active" | "expired" | "revoked" | "disconnected";

export interface IntegrationConnection {
  id: string;
  user_id: string;
  provider: IntegrationProvider;
  status: ConnectionStatus;
  provider_workspace_id?: string;
  provider_user_id?: string;
  scopes?: string[];
  created_at: string;
  updated_at: string;
}

export interface IntegrationInfo {
  provider: IntegrationProvider;
  name: string;
  description: string;
  icon: string;
  color: string;
  connection?: IntegrationConnection;
}

// Board types
export interface BoardColumn {
  id: string;
  user_id: string;
  artist_id?: string;
  title: string;
  position: number;
  color?: string;
  created_at: string;
  updated_at: string;
}

export interface BoardTask {
  id: string;
  column_id?: string;
  user_id: string;
  title: string;
  description?: string;
  position: number;
  priority?: "low" | "medium" | "high" | "urgent";
  start_date?: string;
  due_date?: string;
  color?: string;
  parent_task_id?: string;
  is_parent?: boolean;
  parent_title?: string;
  column_title?: string;
  artist_id?: string;
  project_id?: string;
  artist_ids?: string[];
  project_ids?: string[];
  contract_ids?: string[];
  artists?: { id: string; name: string }[];
  assignee_name?: string;
  labels?: string[];
  external_id?: string;
  external_provider?: string;
  external_url?: string;
  last_synced_at?: string;
  sync_hash?: string;
  created_at: string;
  updated_at: string;
}

export interface BoardTaskDetail extends BoardTask {
  artists: { id: string; name: string; avatar?: string }[];
  projects: { id: string; name: string }[];
  contracts: { id: string; file_name: string }[];
  comments: TaskComment[];
  children?: BoardTask[];
  parent?: { id: string; title: string } | null;
}

export interface ParentTaskWithChildren extends BoardTask {
  children: BoardTask[];
  child_count: number;
}

export interface TaskComment {
  id: string;
  task_id: string;
  user_id: string;
  content: string;
  created_at: string;
  updated_at: string;
}

// Google Drive types
export interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  modifiedTime?: string;
  size?: string;
  iconLink?: string;
  webViewLink?: string;
}

// Slack types
export interface SlackChannel {
  id: string;
  name: string;
  is_private: boolean;
}

// Notion types
export interface NotionDatabase {
  id: string;
  title: string;
  url: string;
}

// Monday types
export interface MondayBoard {
  id: string;
  name: string;
  state: string;
  kind: string;
  columns: { id: string; title: string; type: string }[];
}

export interface NotificationSetting {
  id: string;
  user_id: string;
  provider: string;
  event_type: string;
  enabled: boolean;
  channel_id?: string;
}
