# Slack & Google Drive Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the Slack and Google Drive integrations end-to-end — the backend APIs exist but the frontend is stubbed. Deliver a working OAuth connect/disconnect flow, a Google Drive file browser with import/export, Slack channel picker with notification settings, and event-driven Slack notifications.

**Architecture:** The backend is already built (OAuth, services, routers). This plan focuses on: (1) a new backend endpoint to list a user's connections, (2) unstubbing `useIntegrations` to call real backend endpoints, (3) building provider-specific UIs (Drive file browser, Slack notification settings), (4) wiring the event bus to dispatch Slack notifications, and (5) expanding the IntegrationHub with post-connect configuration panels.

**Tech Stack:** React 18, TypeScript, TanStack React Query, Tailwind CSS, shadcn/ui, FastAPI (Python), Supabase

---

## File Structure

### New Files
- `src/backend/integrations/connections_router.py` — GET `/integrations/connections` endpoint (returns user's active connections)
- `src/hooks/useGoogleDrive.ts` — React Query hook for Drive browse/import/export
- `src/hooks/useSlackSettings.ts` — React Query hook for Slack channels + notification settings
- `src/components/workspace/integrations/DrivePanel.tsx` — Google Drive file browser + import/export UI
- `src/components/workspace/integrations/SlackPanel.tsx` — Slack channel picker + notification settings UI

### Modified Files
- `src/backend/main.py` — Mount `connections_router`, register Slack event handlers at startup
- `src/hooks/useIntegrations.ts` — Replace stubs with real API calls
- `src/components/workspace/IntegrationHub.tsx` — Show provider panels when connected
- `src/components/workspace/IntegrationCard.tsx` — Add "Configure" button for connected integrations
- `src/backend/integrations/events.py` — No changes needed (already complete)
- `src/backend/integrations/slack/service.py` — No changes needed (notify_for_event already works)

---

### Task 0: Backend — Connections List Endpoint

**Goal:** Add a GET endpoint that returns all integration connections for the authenticated user, so the frontend can show connection status.

**Files:**
- Create: `src/backend/integrations/connections_router.py`
- Modify: `src/backend/main.py:36-54` (add import + mount)

**Acceptance Criteria:**
- [ ] `GET /integrations/connections` returns `{ connections: [...] }` with provider, status, created_at, updated_at for each
- [ ] Endpoint uses JWT auth via `get_current_user_id`
- [ ] Encrypted tokens are NOT returned in the response

**Verify:** `curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/integrations/connections` → `{"connections": [...]}`

**Steps:**

- [ ] **Step 1: Create the connections router**

```python
# src/backend/integrations/connections_router.py
"""Endpoint to list a user's integration connections (no secrets exposed)."""

from fastapi import APIRouter, Depends
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("/connections")
async def list_connections(user_id: str = Depends(get_current_user_id)):
    """Return all integration connections for the current user (tokens omitted)."""
    result = (
        _get_supabase()
        .table("integration_connections")
        .select("id, user_id, provider, status, provider_user_id, provider_workspace_id, scopes, created_at, updated_at")
        .eq("user_id", user_id)
        .execute()
    )
    return {"connections": result.data or []}
```

- [ ] **Step 2: Mount the router in main.py**

In `src/backend/main.py`, add after the existing integration router imports (around line 39):

```python
from integrations.connections_router import router as connections_router
```

And add after the existing `app.include_router` calls (around line 54):

```python
app.include_router(connections_router, prefix="/integrations", tags=["Integrations"])
```

- [ ] **Step 3: Verify the endpoint works**

Run: `uvicorn main:app --host 0.0.0.0 --port 8000` (from `src/backend/`)
Test: `curl http://localhost:8000/integrations/connections` with a valid Bearer token
Expected: `{"connections": []}` (empty for new user) or list of connections

- [ ] **Step 4: Commit**

```bash
git add src/backend/integrations/connections_router.py src/backend/main.py
git commit -m "feat: add GET /integrations/connections endpoint"
```

---

### Task 1: Frontend — Unstub useIntegrations Hook

**Goal:** Replace the stubbed `useIntegrations` hook with real API calls to fetch connections, initiate OAuth, and disconnect.

**Files:**
- Modify: `src/hooks/useIntegrations.ts`

**Acceptance Criteria:**
- [ ] `connectionsQuery` fetches from `GET /integrations/connections`
- [ ] `connectMutation` calls `GET /integrations/{provider}/auth`, gets `auth_url`, and redirects the browser
- [ ] `disconnectMutation` calls `DELETE /integrations/{provider}/disconnect` and invalidates the connections query
- [ ] Loading and error states are handled

**Verify:** Open `/workspace` → Integrations tab → click "Connect" on Google Drive → browser redirects to Google OAuth URL

**Steps:**

- [ ] **Step 1: Rewrite the hook with real API calls**

```typescript
// src/hooks/useIntegrations.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { IntegrationConnection, IntegrationProvider } from "@/types/integrations";

const PROVIDER_NAMES: Record<IntegrationProvider, string> = {
  google_drive: "Google Drive",
  slack: "Slack",
  notion: "Notion",
  monday: "Monday.com",
};

// Map provider key to backend URL segment
const PROVIDER_URL_SEGMENT: Record<IntegrationProvider, string> = {
  google_drive: "google-drive",
  slack: "slack",
  notion: "notion",
  monday: "monday",
};

export function useIntegrations() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const connectionsQuery = useQuery<IntegrationConnection[]>({
    queryKey: ["integrations", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ connections: IntegrationConnection[] }>(
        `${API_URL}/integrations/connections`
      );
      return data.connections;
    },
    enabled: !!user?.id,
  });

  const connectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      const segment = PROVIDER_URL_SEGMENT[provider];
      const data = await apiFetch<{ auth_url: string }>(
        `${API_URL}/integrations/${segment}/auth`
      );
      // Redirect browser to OAuth provider
      window.location.href = data.auth_url;
    },
    onError: (error: Error, provider) => {
      toast.error(`Failed to connect ${PROVIDER_NAMES[provider]}: ${error.message}`);
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      const segment = PROVIDER_URL_SEGMENT[provider];
      await apiFetch(`${API_URL}/integrations/${segment}/disconnect`, {
        method: "DELETE",
      });
    },
    onSuccess: (_data, provider) => {
      toast.success(`${PROVIDER_NAMES[provider]} disconnected`);
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
    },
    onError: (error: Error, provider) => {
      toast.error(`Failed to disconnect ${PROVIDER_NAMES[provider]}: ${error.message}`);
    },
  });

  return {
    connections: connectionsQuery.data || [],
    isLoading: connectionsQuery.isLoading,
    connect: connectMutation.mutate,
    disconnect: disconnectMutation.mutate,
    isConnecting: connectMutation.isPending,
  };
}
```

- [ ] **Step 2: Verify OAuth flow end-to-end**

1. Run frontend (`npm run dev`) and backend (`uvicorn main:app`)
2. Go to `/workspace` → Integrations tab
3. Click "Connect" on Google Drive
4. Confirm browser redirects to Google OAuth consent screen
5. After consent, confirm redirect back to `/workspace?connected=google_drive`
6. Confirm the toast "Google Drive connected successfully!" appears
7. Confirm the card now shows "Connected" badge

- [ ] **Step 3: Verify disconnect flow**

1. Click "Disconnect" on a connected integration
2. Confirm toast "Google Drive disconnected" appears
3. Confirm card returns to "Connect" button state

- [ ] **Step 4: Commit**

```bash
git add src/hooks/useIntegrations.ts
git commit -m "feat: wire useIntegrations hook to real backend API"
```

---

### Task 2: Google Drive — File Browser Hook

**Goal:** Create a React Query hook for Google Drive operations: browse files, import to project, export to Drive.

**Files:**
- Create: `src/hooks/useGoogleDrive.ts`

**Acceptance Criteria:**
- [ ] `useDriveBrowse(folderId)` returns files in a Drive folder
- [ ] `useDriveImport()` returns a mutation to import a Drive file into a project
- [ ] `useDriveExport()` returns a mutation to export a project file to Drive
- [ ] Hook is disabled when Drive is not connected

**Verify:** Import the hook in a test component, call `useDriveBrowse("root")` → returns list of files from connected Drive account

**Steps:**

- [ ] **Step 1: Create the Google Drive hook**

```typescript
// src/hooks/useGoogleDrive.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { DriveFile } from "@/types/integrations";

export function useDriveBrowse(folderId: string = "root", enabled: boolean = true) {
  const { user } = useAuth();

  return useQuery<DriveFile[]>({
    queryKey: ["drive-files", user?.id, folderId],
    queryFn: async () => {
      const params = new URLSearchParams({ folder_id: folderId });
      const data = await apiFetch<{ files: DriveFile[] }>(
        `${API_URL}/integrations/google-drive/browse?${params}`
      );
      return data.files;
    },
    enabled: !!user?.id && enabled,
  });
}

export function useDriveImport() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { drive_file_id: string; project_id: string; file_type?: string }) => {
      return apiFetch<{ file: Record<string, unknown>; source: string }>(
        `${API_URL}/integrations/google-drive/import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }
      );
    },
    onSuccess: () => {
      toast.success("File imported from Google Drive");
      queryClient.invalidateQueries({ queryKey: ["project-files"] });
    },
    onError: (error: Error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });
}

export function useDriveExport() {
  return useMutation({
    mutationFn: async (params: { project_file_id: string; drive_folder_id?: string }) => {
      return apiFetch<{ drive_file: Record<string, unknown>; source: string }>(
        `${API_URL}/integrations/google-drive/export`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }
      );
    },
    onSuccess: () => {
      toast.success("File exported to Google Drive");
    },
    onError: (error: Error) => {
      toast.error(`Export failed: ${error.message}`);
    },
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add src/hooks/useGoogleDrive.ts
git commit -m "feat: add useGoogleDrive hook for browse/import/export"
```

---

### Task 3: Google Drive — File Browser Panel

**Goal:** Build a Drive file browser component that lets users navigate folders, preview files, and import them into a project.

**Files:**
- Create: `src/components/workspace/integrations/DrivePanel.tsx`

**Acceptance Criteria:**
- [ ] Shows a file/folder list from the user's Google Drive root
- [ ] Clicking a folder navigates into it; breadcrumb trail for navigation back
- [ ] Each file row shows name, type icon, modified date, and size
- [ ] "Import" button on each file opens a project picker and imports the file
- [ ] Loading skeleton while fetching files

**Verify:** Connect Google Drive → panel appears → browse folders → import a file → confirm it appears in the project's files

**Steps:**

- [ ] **Step 1: Create the DrivePanel component**

```tsx
// src/components/workspace/integrations/DrivePanel.tsx
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Folder,
  FileText,
  FileAudio,
  FileImage,
  File,
  ChevronRight,
  Download,
  ArrowLeft,
} from "lucide-react";
import { useDriveBrowse, useDriveImport } from "@/hooks/useGoogleDrive";
import { useProjectsList } from "@/hooks/useProjectsList";
import type { DriveFile } from "@/types/integrations";

function getFileIcon(mimeType: string) {
  if (mimeType === "application/vnd.google-apps.folder") return <Folder className="w-4 h-4 text-blue-500" />;
  if (mimeType.startsWith("audio/")) return <FileAudio className="w-4 h-4 text-purple-500" />;
  if (mimeType.startsWith("image/")) return <FileImage className="w-4 h-4 text-green-500" />;
  if (mimeType.includes("pdf") || mimeType.includes("document")) return <FileText className="w-4 h-4 text-red-500" />;
  return <File className="w-4 h-4 text-muted-foreground" />;
}

function formatFileSize(bytes?: string) {
  if (!bytes) return "";
  const size = parseInt(bytes, 10);
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

interface DrivePanelProps {
  onClose?: () => void;
}

export function DrivePanel({ onClose }: DrivePanelProps) {
  const [folderStack, setFolderStack] = useState<{ id: string; name: string }[]>([
    { id: "root", name: "My Drive" },
  ]);
  const [importingFileId, setImportingFileId] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");

  const currentFolder = folderStack[folderStack.length - 1];
  const { data: files, isLoading } = useDriveBrowse(currentFolder.id);
  const importMutation = useDriveImport();
  const { projects } = useProjectsList();

  const navigateToFolder = (file: DriveFile) => {
    setFolderStack((prev) => [...prev, { id: file.id, name: file.name }]);
  };

  const navigateBack = () => {
    if (folderStack.length > 1) {
      setFolderStack((prev) => prev.slice(0, -1));
    }
  };

  const handleImport = (file: DriveFile) => {
    if (!selectedProjectId) return;
    importMutation.mutate(
      { drive_file_id: file.id, project_id: selectedProjectId },
      { onSuccess: () => setImportingFileId(null) }
    );
  };

  const isFolder = (mimeType: string) =>
    mimeType === "application/vnd.google-apps.folder";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">Google Drive</CardTitle>
          <div className="flex items-center text-sm text-muted-foreground">
            {folderStack.map((folder, idx) => (
              <span key={folder.id} className="flex items-center">
                {idx > 0 && <ChevronRight className="w-3 h-3 mx-1" />}
                <button
                  className="hover:text-foreground hover:underline"
                  onClick={() => setFolderStack(folderStack.slice(0, idx + 1))}
                >
                  {folder.name}
                </button>
              </span>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {folderStack.length > 1 && (
            <Button variant="ghost" size="sm" onClick={navigateBack}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
          )}
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {/* Project selector for imports */}
        <div className="mb-4">
          <Select value={selectedProjectId} onValueChange={setSelectedProjectId}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a project to import into..." />
            </SelectTrigger>
            <SelectContent>
              {(projects || []).map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* File list */}
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : (
          <div className="divide-y divide-border rounded-md border">
            {files?.length === 0 && (
              <p className="p-4 text-sm text-muted-foreground text-center">
                This folder is empty
              </p>
            )}
            {files?.map((file) => (
              <div
                key={file.id}
                className={`flex items-center justify-between px-3 py-2 text-sm ${
                  isFolder(file.mimeType)
                    ? "cursor-pointer hover:bg-muted/50"
                    : ""
                }`}
                onClick={() => isFolder(file.mimeType) && navigateToFolder(file)}
              >
                <div className="flex items-center gap-3 min-w-0">
                  {getFileIcon(file.mimeType)}
                  <span className="truncate">{file.name}</span>
                </div>
                <div className="flex items-center gap-4 shrink-0">
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                  </span>
                  {file.modifiedTime && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(file.modifiedTime).toLocaleDateString()}
                    </span>
                  )}
                  {!isFolder(file.mimeType) && (
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!selectedProjectId || importMutation.isPending}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleImport(file);
                      }}
                    >
                      <Download className="w-4 h-4 mr-1" />
                      Import
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/components/workspace/integrations/DrivePanel.tsx
git commit -m "feat: add Google Drive file browser panel"
```

---

### Task 4: Slack — Settings Hook & Notification Panel

**Goal:** Create a React Query hook for Slack channels and notification settings, then build the Slack configuration panel.

**Files:**
- Create: `src/hooks/useSlackSettings.ts`
- Create: `src/components/workspace/integrations/SlackPanel.tsx`

**Acceptance Criteria:**
- [ ] `useSlackChannels()` fetches available Slack channels
- [ ] `useSlackSettings()` fetches + updates notification settings per event type
- [ ] SlackPanel shows a channel picker per event type (task_created, task_updated, contract_uploaded, royalty_calculated)
- [ ] Toggling an event on/off updates the backend immediately

**Verify:** Connect Slack → panel appears → select a channel for "Task Created" → toggle it on → confirm setting saved in DB

**Steps:**

- [ ] **Step 1: Create the Slack settings hook**

```typescript
// src/hooks/useSlackSettings.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { SlackChannel, NotificationSetting } from "@/types/integrations";

export function useSlackChannels(enabled: boolean = true) {
  const { user } = useAuth();

  return useQuery<SlackChannel[]>({
    queryKey: ["slack-channels", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ channels: SlackChannel[] }>(
        `${API_URL}/integrations/slack/channels`
      );
      return data.channels;
    },
    enabled: !!user?.id && enabled,
  });
}

export function useSlackSettings(enabled: boolean = true) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const settingsQuery = useQuery<NotificationSetting[]>({
    queryKey: ["slack-settings", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ settings: NotificationSetting[] }>(
        `${API_URL}/integrations/slack/settings`
      );
      return data.settings;
    },
    enabled: !!user?.id && enabled,
  });

  const updateMutation = useMutation({
    mutationFn: async (params: {
      event_type: string;
      enabled: boolean;
      channel_id?: string;
    }) => {
      return apiFetch(`${API_URL}/integrations/slack/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slack-settings"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to update setting: ${error.message}`);
    },
  });

  return {
    settings: settingsQuery.data || [],
    isLoading: settingsQuery.isLoading,
    updateSetting: updateMutation.mutate,
    isUpdating: updateMutation.isPending,
  };
}
```

- [ ] **Step 2: Create the SlackPanel component**

```tsx
// src/components/workspace/integrations/SlackPanel.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Hash, Lock } from "lucide-react";
import { useSlackChannels, useSlackSettings } from "@/hooks/useSlackSettings";

const EVENT_TYPES = [
  {
    key: "task_created",
    label: "Task Created",
    description: "When a new task is created on the board",
  },
  {
    key: "task_updated",
    label: "Task Updated",
    description: "When a task is modified or moved",
  },
  {
    key: "task_completed",
    label: "Task Completed",
    description: "When a task is marked as done",
  },
  {
    key: "contract_uploaded",
    label: "Contract Uploaded",
    description: "When a new contract is uploaded to a project",
  },
  {
    key: "royalty_calculated",
    label: "Royalty Calculated",
    description: "When a royalty calculation is completed",
  },
];

interface SlackPanelProps {
  onClose?: () => void;
}

export function SlackPanel({ onClose }: SlackPanelProps) {
  const { data: channels, isLoading: channelsLoading } = useSlackChannels();
  const { settings, isLoading: settingsLoading, updateSetting } = useSlackSettings();

  const getSettingForEvent = (eventType: string) =>
    settings.find((s) => s.event_type === eventType);

  const handleToggle = (eventType: string, enabled: boolean) => {
    const existing = getSettingForEvent(eventType);
    updateSetting({
      event_type: eventType,
      enabled,
      channel_id: existing?.channel_id || undefined,
    });
  };

  const handleChannelChange = (eventType: string, channelId: string) => {
    const existing = getSettingForEvent(eventType);
    updateSetting({
      event_type: eventType,
      enabled: existing?.enabled ?? true,
      channel_id: channelId,
    });
  };

  const isLoading = channelsLoading || settingsLoading;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div>
          <CardTitle className="text-base">Slack Notifications</CardTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Choose which events send notifications and to which channels
          </p>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {EVENT_TYPES.map((event) => {
              const setting = getSettingForEvent(event.key);
              const isEnabled = setting?.enabled ?? false;

              return (
                <div
                  key={event.key}
                  className="flex items-center justify-between gap-4 rounded-lg border p-3"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{event.label}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {event.description}
                    </p>
                    {isEnabled && (
                      <div className="mt-2">
                        <Select
                          value={setting?.channel_id || ""}
                          onValueChange={(val) =>
                            handleChannelChange(event.key, val)
                          }
                        >
                          <SelectTrigger className="h-8 text-xs w-56">
                            <SelectValue placeholder="Select channel..." />
                          </SelectTrigger>
                          <SelectContent>
                            {(channels || []).map((ch) => (
                              <SelectItem key={ch.id} value={ch.id}>
                                <div className="flex items-center gap-1.5">
                                  {ch.is_private ? (
                                    <Lock className="w-3 h-3" />
                                  ) : (
                                    <Hash className="w-3 h-3" />
                                  )}
                                  {ch.name}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </div>
                  <Switch
                    checked={isEnabled}
                    onCheckedChange={(checked) =>
                      handleToggle(event.key, checked)
                    }
                  />
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useSlackSettings.ts src/components/workspace/integrations/SlackPanel.tsx
git commit -m "feat: add Slack notification settings hook and panel"
```

---

### Task 5: IntegrationHub — Show Provider Panels When Connected

**Goal:** Update the IntegrationHub to show provider-specific configuration panels (DrivePanel, SlackPanel) when the integration is connected, and add a "Configure" action to connected cards.

**Files:**
- Modify: `src/components/workspace/IntegrationHub.tsx`
- Modify: `src/components/workspace/IntegrationCard.tsx`

**Acceptance Criteria:**
- [ ] Connected Google Drive card has a "Configure" button that opens the DrivePanel
- [ ] Connected Slack card has a "Configure" button that opens the SlackPanel
- [ ] Only one panel is visible at a time
- [ ] Panel can be closed, returning to the card grid view

**Verify:** Connect an integration → see "Configure" button → click it → panel appears below the cards → close panel → back to card grid

**Steps:**

- [ ] **Step 1: Add a Configure button to IntegrationCard**

In `src/components/workspace/IntegrationCard.tsx`, add an `onConfigure` optional prop and show the button next to "Disconnect" when connected:

```tsx
// src/components/workspace/IntegrationCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Check, Loader2, Unplug, Settings } from "lucide-react";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";

interface IntegrationCardProps {
  provider: IntegrationProvider;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  status: ConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  onConfigure?: () => void;
  isConnecting?: boolean;
}

export function IntegrationCard({
  name,
  description,
  icon,
  color,
  status,
  onConnect,
  onDisconnect,
  onConfigure,
  isConnecting,
}: IntegrationCardProps) {
  const isConnected = status === "active";

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center gap-4 pb-3">
        <div
          className="w-12 h-12 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: `${color}15` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">{name}</CardTitle>
            {isConnected && (
              <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                <Check className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            )}
            {status === "expired" && (
              <Badge variant="destructive" className="text-xs">
                Expired
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {isConnected ? (
          <div className="flex items-center gap-2">
            {onConfigure && (
              <Button variant="outline" size="sm" onClick={onConfigure}>
                <Settings className="w-4 h-4 mr-2" />
                Configure
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={onDisconnect}
              className="text-destructive hover:text-destructive"
            >
              <Unplug className="w-4 h-4 mr-2" />
              Disconnect
            </Button>
          </div>
        ) : (
          <Button size="sm" onClick={onConnect} disabled={isConnecting}>
            {isConnecting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Connecting...
              </>
            ) : (
              "Connect"
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Update IntegrationHub to manage panel visibility**

```tsx
// src/components/workspace/IntegrationHub.tsx
import { useState } from "react";
import { IntegrationCard } from "./IntegrationCard";
import { useIntegrations } from "@/hooks/useIntegrations";
import { DrivePanel } from "./integrations/DrivePanel";
import { SlackPanel } from "./integrations/SlackPanel";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";

const INTEGRATIONS = [
  {
    provider: "google_drive" as IntegrationProvider,
    name: "Google Drive",
    description: "Sync contracts and royalty statements with Google Drive",
    icon: <img src="/drive.webp" alt="Google Drive" className="w-6 h-6 object-contain" />,
    color: "#4285F4",
  },
  {
    provider: "slack" as IntegrationProvider,
    name: "Slack",
    description: "Get notifications and sync updates to Slack channels",
    icon: <img src="/slack.png" alt="Slack" className="w-6 h-6 object-contain" />,
    color: "#4A154B",
  },
  {
    provider: "notion" as IntegrationProvider,
    name: "Notion",
    description: "Sync project boards and tasks with Notion databases",
    icon: <img src="/Notion_app_logo.png" alt="Notion" className="w-6 h-6 object-contain" />,
    color: "#000000",
  },
  {
    provider: "monday" as IntegrationProvider,
    name: "Monday.com",
    description: "Sync project boards and tasks with Monday.com boards",
    icon: <img src="/mondaycom.png" alt="Monday.com" className="w-6 h-6 object-contain" />,
    color: "#FF3D57",
  },
];

const CONFIGURABLE_PROVIDERS: IntegrationProvider[] = ["google_drive", "slack"];

export function IntegrationHub() {
  const { connections, connect, disconnect, isConnecting } = useIntegrations();
  const [activePanel, setActivePanel] = useState<IntegrationProvider | null>(null);

  const getStatus = (provider: IntegrationProvider): ConnectionStatus => {
    const conn = connections.find((c) => c.provider === provider);
    return conn?.status || "disconnected";
  };

  const isConnected = (provider: IntegrationProvider) =>
    getStatus(provider) === "active";

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Connected Services</h3>
        <p className="text-sm text-muted-foreground">
          Connect your favorite tools to sync data and receive notifications
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {INTEGRATIONS.map((integration) => (
          <IntegrationCard
            key={integration.provider}
            {...integration}
            status={getStatus(integration.provider)}
            onConnect={() => connect(integration.provider)}
            onDisconnect={() => {
              disconnect(integration.provider);
              if (activePanel === integration.provider) setActivePanel(null);
            }}
            onConfigure={
              CONFIGURABLE_PROVIDERS.includes(integration.provider) &&
              isConnected(integration.provider)
                ? () =>
                    setActivePanel(
                      activePanel === integration.provider
                        ? null
                        : integration.provider
                    )
                : undefined
            }
            isConnecting={isConnecting}
          />
        ))}
      </div>

      {/* Provider-specific configuration panels */}
      {activePanel === "google_drive" && (
        <DrivePanel onClose={() => setActivePanel(null)} />
      )}
      {activePanel === "slack" && (
        <SlackPanel onClose={() => setActivePanel(null)} />
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify the full UI flow**

1. Open `/workspace` → Integrations tab
2. Confirm cards render with correct status (connected/disconnected)
3. For a connected integration, confirm "Configure" button appears
4. Click "Configure" → panel opens below the cards
5. Click "Configure" again or "Close" → panel closes
6. Click "Disconnect" → panel closes, card returns to disconnected state

- [ ] **Step 4: Commit**

```bash
git add src/components/workspace/IntegrationCard.tsx src/components/workspace/IntegrationHub.tsx
git commit -m "feat: add configuration panels to integration hub"
```

---

### Task 6: Backend — Wire Slack Event Handlers at Startup

**Goal:** Register the Slack `notify_for_event` function as a handler for all relevant events in the event bus, so Slack notifications fire automatically when tasks are created, contracts uploaded, etc.

**Files:**
- Modify: `src/backend/main.py` (add startup event registration)

**Acceptance Criteria:**
- [ ] On app startup, Slack notification handler is registered for: `task_created`, `task_updated`, `task_completed`, `contract_uploaded`, `royalty_calculated`
- [ ] When a task is created via the boards API, a Slack notification is sent if the user has it configured
- [ ] Handler failures don't crash the board operations (event bus already isolates errors)

**Verify:** Create a task via POST `/boards/tasks` → check Slack channel → notification message appears

**Steps:**

- [ ] **Step 1: Add event handler registration in main.py**

Find the app initialization section in `src/backend/main.py` (after the router mounts, around line 55). Add the following:

```python
# --- Register Slack notification handlers on events ---
from integrations import events
from integrations.slack.service import notify_for_event as slack_notify


async def _slack_event_handler(event_name: str, payload: dict):
    """Bridge between event bus and Slack notification service."""
    user_id = payload.get("user_id")
    if not user_id:
        return
    await slack_notify(get_supabase_client(), user_id, event_name, payload)


# Register for all notifiable events
for _event in [
    events.TASK_CREATED,
    events.TASK_UPDATED,
    events.TASK_COMPLETED,
    events.CONTRACT_UPLOADED,
    events.CONTRACT_DELETED,
    events.ROYALTY_CALCULATED,
]:
    events.on(_event, _slack_event_handler)
```

- [ ] **Step 2: Verify end-to-end notification flow**

1. Connect Slack integration (OAuth)
2. Configure a notification: enable `task_created` → select a channel
3. Create a task via the Kanban board
4. Confirm the Slack channel receives a message: "New task created: {title}"

- [ ] **Step 3: Commit**

```bash
git add src/backend/main.py
git commit -m "feat: register Slack notification handlers on event bus"
```

---

### Task 7: Integration Testing & Polish

**Goal:** End-to-end manual test checklist and UI polish for both integrations.

**Files:**
- Modify: `src/components/workspace/integrations/DrivePanel.tsx` (if fixes needed)
- Modify: `src/components/workspace/integrations/SlackPanel.tsx` (if fixes needed)
- Modify: `src/hooks/useIntegrations.ts` (if fixes needed)

**Acceptance Criteria:**
- [ ] Google Drive: connect → browse → navigate folders → import file → file appears in project
- [ ] Google Drive: disconnect → card shows "Connect" again → panel closes
- [ ] Slack: connect → configure notifications → create task → notification arrives in Slack
- [ ] Slack: disable a notification → create task → no notification sent
- [ ] Slack: disconnect → settings cleaned up → card shows "Connect"
- [ ] OAuth redirect back to `/workspace` shows success toast and correct status
- [ ] Expired token on Drive browse → user sees clear error (401 handling)
- [ ] All loading states show skeletons/spinners (no blank flashes)

**Verify:** Complete the full manual test checklist above for both providers

**Steps:**

- [ ] **Step 1: Run the full Google Drive flow**

1. Start backend: `cd src/backend && uvicorn main:app --host 0.0.0.0 --port 8000`
2. Start frontend: `npm run dev`
3. Go to `/workspace` → Integrations
4. Click "Connect" on Google Drive → complete OAuth → confirm redirect + toast
5. Click "Configure" → browse files → navigate folders
6. Select a project → import a file → confirm toast + file in project
7. Click "Disconnect" → confirm cleanup

- [ ] **Step 2: Run the full Slack flow**

1. Click "Connect" on Slack → complete OAuth → confirm redirect + toast
2. Click "Configure" → see event types
3. Toggle "Task Created" on → select a channel
4. Go to "Project Boards" tab → create a task
5. Check Slack channel → confirm notification arrived
6. Toggle "Task Created" off → create another task → confirm no notification
7. Click "Disconnect" → confirm cleanup

- [ ] **Step 3: Fix any issues found during testing**

Address bugs, missing error states, or UI polish items discovered during the test runs.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: integration testing polish and bug fixes"
```
