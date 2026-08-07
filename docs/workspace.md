# Workspace

The Workspace (`/workspace`) is the operational hub — integrations, Kanban boards, calendar, notifications, and settings. This doc covers everything accessible from the Workspace page tabs.

---

# Integrations

Msanii connects to external services through a shared OAuth layer. Each integration stores encrypted tokens in Supabase, fires named events through an internal event bus, and exposes FastAPI routers under `/integrations/<provider>`. Google Drive is the only implemented integration (Slack was removed in `20260806000000_remove_slack_integration.sql`).

---

## Integrations — Backend Endpoints

All routers are mounted in `src/backend/main.py`. Authentication is enforced via the `Authorization: Bearer <supabase_jwt>` header on every request; the `get_current_user_id` dependency extracts the user ID from the token.

### Connections

Mounted at `/integrations` (`src/backend/integrations/connections_router.py`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/integrations/connections` | List all active integration connections for the current user. Returns id, provider, status, provider_user_id, provider_workspace_id, scopes, timestamps. Tokens are never included in the response. |

### Google Drive

Mounted at `/integrations/google-drive` (`src/backend/integrations/google_drive/router.py`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/integrations/google-drive/auth` | Start the OAuth flow. Returns `{ auth_url }` for the browser to redirect to. |
| GET | `/integrations/google-drive/callback` | OAuth callback. Receives `code` and `state` query params, exchanges them for tokens, stores the connection, then redirects to `/workspace?connected=google_drive`. |
| DELETE | `/integrations/google-drive/disconnect` | Remove the user's Google Drive connection from `integration_connections`. |
| GET | `/integrations/google-drive/browse` | List files and folders in Google Drive. Query param: `folder_id` (default `root`). Returns `{ files }`. |
| POST | `/integrations/google-drive/import` | Import a Drive file into a project. Body: `DriveImportRequest` (`drive_file_id`, `project_id`, optional `file_type`). Returns `{ file, source }`. |
| POST | `/integrations/google-drive/export` | Export a project file to Drive. Body: `DriveExportRequest` (`project_file_id`, optional `drive_folder_id`). Returns `{ drive_file, source }`. |
| POST | `/integrations/google-drive/sync/setup` | Configure sync for a project folder. Body: `DriveSyncSetup` (`project_id`, `drive_folder_id`, `sync_direction`). |
| GET | `/integrations/google-drive/sync/status` | Return all `drive_sync_mappings` rows for the current user. |

### OneClick Share

Mounted at `/oneclick` (`src/backend/oneclick/share.py`).

| Method | Path | Description |
|--------|------|-------------|
| POST | `/oneclick/share` | Generate a PDF royalty report and deliver it to Drive. Body: `ShareRequest` (`target`: `"drive"`, `artist_name`, `payments`, `total_payments`, optional `folder_id`). |

## OAuth Flow

The same flow applies to all four providers. State is a short-lived signed JWT (10-minute expiry) used for CSRF protection.

1. **Frontend requests the auth URL.** The hook calls `GET /integrations/<provider>/auth`. The backend generates a signed JWT state containing `user_id`, `provider`, and `exp`, then builds the full provider authorization URL.
2. **Browser redirect.** The frontend sets `window.location.href` to the auth URL, sending the user to the OAuth consent screen.
3. **Provider redirects back.** After consent the provider calls `GET /integrations/<provider>/callback?code=...&state=...`.
4. **State verification.** The backend decodes and validates the JWT. Invalid or expired state returns HTTP 400.
5. **Token exchange.** Authorization code is exchanged for access/refresh tokens.
6. **Token encryption and storage.** Tokens encrypted with AES-256 via Fernet, upserted into `integration_connections`.
7. **Redirect to frontend.** User sent to `/workspace?connected=<provider>`.
8. **Subsequent requests.** `get_valid_token` decrypts the stored token on every API call. Auto-refreshes Google Drive tokens if expiring within 5 minutes.

## Event Bus

`src/backend/integrations/events.py` provides a lightweight publish/subscribe bus.

**Named constants:**

| Constant | Value |
|----------|-------|
| `CONTRACT_UPLOADED` | `"contract_uploaded"` |
| `CONTRACT_DELETED` | `"contract_deleted"` |
| `ROYALTY_CALCULATED` | `"royalty_calculated"` |
| `ARTIST_CREATED` | `"artist_created"` |
| `TASK_CREATED` | `"task_created"` |
| `TASK_UPDATED` | `"task_updated"` |
| `TASK_COMPLETED` | `"task_completed"` |

**Subscribers: none.** Slack was the only consumer and was removed, so `emit()` currently runs zero handlers — the calls are cheap no-ops kept in place for the next integration. Register a handler in `main.py` to revive it.

## Integrations — Frontend

### Hooks

| Hook | File | Returns |
|------|------|---------|
| `useIntegrations` | `useIntegrations.ts` | `{ connections, connect(provider), disconnect(provider), isConnecting, isLoading }` |
| `useDriveBrowse(folderId?, enabled?)` | `useGoogleDrive.ts` | `QueryResult<DriveFile[]>` |
| `useDriveImport` | `useGoogleDrive.ts` | Mutation: `{ drive_file_id, project_id }` |
| `useDriveExport` | `useGoogleDrive.ts` | Mutation: `{ project_file_id }` |
| `useProjectNotificationSettings(projectId?)` | `useProjectIntegrations.ts` | `{ isEventEnabled, toggleEvent }` |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `IntegrationHub` | `src/components/workspace/IntegrationHub.tsx` | Card grid of all integrations + panel toggle |
| `IntegrationCard` | `src/components/workspace/IntegrationCard.tsx` | Single card (connect/disconnect/configure) |
| `DrivePanel` | `src/components/workspace/integrations/DrivePanel.tsx` | Workspace-level Drive file browser |
| `DriveImportDialog` | `src/components/project/integrations/DriveImportDialog.tsx` | Project-level Drive import dialog |

## Integrations — Database Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `integration_connections` | `user_id, provider, status, access_token_encrypted` | OAuth token storage (encrypted) |
| `notification_settings` | `user_id, provider, event_type, enabled, channel_id` | Workspace-level notification prefs |
| `project_notification_settings` | `project_id, event_type, enabled` | Per-project event toggles |
| ~~`slack_notifications`~~ | — | Dormant: kept by `20260806000000_remove_slack_integration.sql` but no longer read or written |
| `drive_sync_mappings` | `user_id, project_id, drive_file_id, sync_direction` | Drive file sync tracking |
| `sync_log` | `user_id, provider, direction, status, metadata` | Audit trail for all sync ops |

## Integrations — Local Testing

```bash
TOKEN="your-supabase-jwt-here"
BASE="http://localhost:8000"

# List connections
curl -H "Authorization: Bearer $TOKEN" "$BASE/integrations/connections"
```

```bash
cd src/backend && poetry run pytest tests/test_integrations.py -v
```

---

# Boards (Kanban)

Users create columns, move tasks via drag-and-drop, attach tasks to artists/projects/contracts, set dates, add comments, and group tasks under parent "epics". A calendar view queries tasks by date range.

---

## Boards — Backend Endpoints

### Columns

| Method | Path | Description |
|--------|------|-------------|
| GET | `/boards/columns` | List columns (optional `?artist_id=` filter) |
| POST | `/boards/columns` | Create column `{ title, color?, artist_id? }` |
| PUT | `/boards/columns/{column_id}` | Update column |
| DELETE | `/boards/columns/{column_id}` | Delete column + cascade tasks |
| POST | `/boards/columns/defaults` | Seed 5 default columns |

### Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/boards/tasks` | List all tasks (paginated) |
| GET | `/boards/tasks/period` | Tasks by `?period_start=&period_end=&is_current=` |
| POST | `/boards/tasks` | Create task. Emits `task_created` |
| GET | `/boards/tasks/{task_id}/detail` | Full task with artists, projects, contracts, comments |
| PUT | `/boards/tasks/{task_id}` | Update task. Moving to "Done" sets `completed_at`. Emits `task_updated` |
| DELETE | `/boards/tasks/{task_id}` | Delete task |
| PUT | `/boards/tasks/reorder` | Batch drag-and-drop reorder |

### Parent Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/boards/parents` | List parent tasks with nested children |
| POST | `/boards/parents` | Create parent task (epic) |

### Calendar

| Method | Path | Description |
|--------|------|-------------|
| GET | `/boards/calendar?start=&end=` | Tasks within date range |

### Comments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/boards/tasks/{task_id}/comments` | Add comment |
| DELETE | `/boards/comments/{comment_id}` | Delete comment (author only) |

## Boards — Frontend

### Hooks

| Hook | File | Returns |
|------|------|---------|
| `useBoards(options?)` | `useBoards.ts` | `{ columns, tasks, createColumn, createTask, updateTask, deleteTask, reorderTasks }` |
| `useCalendarTasks(start, end)` | `useCalendarTasks.ts` | `{ tasks }` |
| `useParentTasks(search?, artistId?)` | `useParentTasks.ts` | `{ parents, ungrouped, createParent }` |
| `useTaskDetail(taskId)` | `useTaskDetail.ts` | `{ task, addComment, deleteComment }` |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `KanbanBoard` | `src/components/workspace/boards/KanbanBoard.tsx` | Main board with drag-and-drop |
| `TaskDetailPanel` | `src/components/workspace/boards/TaskDetailPanel.tsx` | Slide-out task editor |
| `CalendarView` | `src/components/workspace/boards/CalendarView.tsx` | Calendar layout |

## Boards — Database Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `board_columns` | `user_id, title, position, color` | Kanban columns |
| `board_tasks` | `column_id, user_id, title, priority, due_date, position, completed_at` | Task records |
| `board_task_artists` | `task_id, artist_id` | Task ↔ Artist junction |
| `board_task_projects` | `task_id, project_id` | Task ↔ Project junction |
| `board_task_contracts` | `task_id, project_file_id` | Task ↔ File junction |
| `board_task_comments` | `task_id, user_id, content` | Task comments |

## Boards — Local Testing

```bash
# List columns
curl -H "Authorization: Bearer $TOKEN" "$BASE/boards/columns"

# Create a task
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  "$BASE/boards/tasks" \
  -d '{"title": "Fix mix levels", "column_id": "COL_ID", "priority": "high"}'

# Get task detail
curl -H "Authorization: Bearer $TOKEN" "$BASE/boards/tasks/TASK_ID/detail"
```

```bash
cd src/backend && poetry run pytest tests/test_boards.py -v
```

---

# Settings

## Settings — Backend Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/settings` | Get workspace settings (timezone, time format, board period) |
| PUT | `/settings` | Update workspace settings |

## Settings — Frontend

| Hook | File | Returns |
|------|------|---------|
| `useWorkspaceSettings()` | `useWorkspaceSettings.ts` | `{ settings, updateSettings }` |

| Component | File | Purpose |
|-----------|------|---------|
| `WorkspaceSettings` | `src/components/workspace/WorkspaceSettings.tsx` | Timezone, time format, board period config |

## Settings — Database Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `workspace_settings` | `user_id, board_period, timezone, use_24h_time` | User preferences |

```bash
cd src/backend && poetry run pytest tests/test_settings.py -v
```
