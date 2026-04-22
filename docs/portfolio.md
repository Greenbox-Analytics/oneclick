# Portfolio & Projects

The Portfolio (`/portfolio`) and Project Detail (`/projects/:id`) pages manage the core data: artist profiles, projects (albums/EPs/singles), file storage, audio, project membership, and notes.

---

## Backend Endpoints

### Artists (defined in `main.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/artists` | List user's artists (paginated) |
| GET | `/artists/{artist_id}` | Get single artist |
| POST | `/artists` | Create artist profile |
| PUT | `/artists/{artist_id}` | Update artist |
| DELETE | `/artists/{artist_id}` | Delete artist + cascade |

### Projects (defined in `main.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/projects` | List all user's projects |
| GET | `/projects/{artist_id}` | List projects for an artist |
| POST | `/projects` | Create project `{ name, artist_id }` |

### Files (defined in `main.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/files/{project_id}` | List project files (paginated) |
| GET | `/files/artist/{artist_id}/category/{category}` | Files by artist + category |
| POST | `/upload` | Upload file to project (with SHA-256 dedup) |
| POST | `/contracts/upload` | Upload contract (triggers PDF-to-markdown background task) |
| DELETE | `/contracts/{contract_id}` | Delete file from storage + DB |
| GET | `/contracts/{contract_id}/markdown` | Get contract markdown (for Zoe/OneClick) |

### Members (`projects/router.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/projects/{project_id}/members` | List project members |
| POST | `/projects/{project_id}/members` | Add member (invite by email) |
| PUT | `/projects/{project_id}/members/{member_id}` | Update role (owner/admin/editor/viewer) |
| DELETE | `/projects/{project_id}/members/{member_id}` | Remove member |
| GET | `/projects/{project_id}/pending-invites` | List pending invitations |
| DELETE | `/projects/{project_id}/pending-invites/{invite_id}` | Cancel invite |

---

## Frontend

### Hooks

| Hook | File | Returns |
|------|------|---------|
| `usePortfolioData(filters)` | `usePortfolioData.ts` | `{ artists, ownedProjects, sharedProjects, isLoading }` |
| `useArtistsList()` | `useArtistsList.ts` | `{ artists: { id, name }[] }` |
| `useProjectsList(artistIds?, projectIds?)` | `useProjectsList.ts` | `{ projects, contracts }` |
| `useProjectMembers(projectId)` | `useProjectMembers.ts` | `{ members, addMember, updateRole, removeMember }` |
| `useMyRole(projectId)` | `useProjectMembers.ts` | `"owner" | "admin" | "editor" | "viewer" | undefined` |
| `useNotes(scope)` | `useNotes.ts` | Notes and folders with CRUD mutations |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `FilesTab` | `src/components/project/FilesTab.tsx` | File list with upload, Drive import/export |
| `AudioTab` | `src/components/project/AudioTab.tsx` | Audio file management |
| `MembersTab` | `src/components/project/MembersTab.tsx` | Member list + invite |
| `SettingsTab` | `src/components/project/SettingsTab.tsx` | Project settings + Slack config |
| `WorksTab` | `src/components/project/WorksTab.tsx` | Works list within project |
| `AddWorkDialog` | `src/components/project/AddWorkDialog.tsx` | Create new work dialog |
| `NotesView` | `src/components/notes/NotesView.tsx` | BlockNote rich text editor |

### Pages

| Page | File | Route |
|------|------|-------|
| Portfolio | `src/pages/Portfolio.tsx` | `/portfolio` |
| ProjectDetail | `src/pages/ProjectDetail.tsx` | `/projects/:projectId` |
| Artists | `src/pages/Artists.tsx` | `/artists` |

---

## Database Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `artists` | `user_id, name` | Artist profiles |
| `projects` | `artist_id, name, slack_channel_id, drive_folder_id` | Albums, EPs, singles |
| `project_members` | `project_id, user_id, role` | Access control |
| `pending_project_invites` | `project_id, email, role` | Pending membership invites |
| `project_files` | `project_id, file_name, file_path, folder_category, content_hash` | Uploaded files |
| `audio_files` | `folder_id, file_name, file_path` | Audio uploads |
| `project_audio_links` | `project_id, audio_file_id` | Audio ↔ Project junction |
| `notes` | `entity_id, entity_type, title, content` | Rich text notes (BlockNote JSON) |
| `note_folders` | `entity_id, entity_type, name` | Note organization |

---

## Local Testing

```bash
TOKEN="your-supabase-jwt-here"
BASE="http://localhost:8000"

# List artists
curl -H "Authorization: Bearer $TOKEN" "$BASE/artists"

# Create a project
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  "$BASE/projects" -d '{"artist_id": "AID", "name": "My EP"}'

# Upload a contract
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -F "file=@contract.pdf" -F "project_id=PID" \
  "$BASE/contracts/upload"

# List project members
curl -H "Authorization: Bearer $TOKEN" "$BASE/projects/PROJECT_ID/members"
```

```bash
cd src/backend
poetry run pytest tests/test_artists.py tests/test_projects.py tests/test_files.py tests/test_project_members.py tests/test_notes.py -v
```
