# Rights Registry

The Rights Registry is the largest domain in Msanii. It provides a complete ownership and licensing ledger for musical works. Each work tracks ownership stakes (master and publishing percentages), licensing rights, immutable agreement records, and collaborator invitations. Works can be linked to project files and audio files, and the full ownership chain can be exported as a signed PDF. A TeamCard system lets collaborators control what contact information they share. An in-app notification layer surfaces invitation and status-change events in real time.

---

## Backend Endpoints

All endpoints are mounted under the `/registry` prefix in `src/backend/registry/router.py`.  
Authentication is resolved via the `Authorization: Bearer <jwt>` header; user identity is extracted by `get_current_user_id`.

### Works

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/works` | List all works owned by the authenticated user. Accepts `artist_id` (filter), `page` (≥1), `page_size` (1–100, default 50). Returns `{ works: Work[] }`. |
| `GET` | `/registry/works/my-collaborations` | List works where the user is a collaborator (not owner). Accepts same pagination params. Returns `{ works: Work[] }`. |
| `GET` | `/registry/works/by-project/{project_id}` | List all works scoped to a project. Returns `{ works: Work[] }`. |
| `GET` | `/registry/works/{work_id}` | Fetch a single work. 404 if not found. |
| `GET` | `/registry/works/{work_id}/full` | Fetch a work with all related data embedded: `stakes`, `licenses`, `agreements`, `collaborators`. 404 if not found. |
| `POST` | `/registry/works` | Create a work. Body: `WorkCreate`. Returns the created work. |
| `PUT` | `/registry/works/{work_id}` | Update a work. Body: `WorkUpdate`. Returns updated work; 404 if not found. |
| `DELETE` | `/registry/works/{work_id}` | Delete a work. Returns `{ ok: true }`. |
| `POST` | `/registry/works/{work_id}/submit-for-approval` | Transition work status from `draft` → `pending`. Re-sends invitation emails to any pending collaborators. Returns the updated work; 400 on validation error. |
| `GET` | `/registry/works/{work_id}/export` | Stream a PDF "Proof of Ownership" for the work. Response header: `Content-Disposition: attachment; filename="Proof_of_Ownership_<title>.pdf"`. 404 if not found. |

**WorkCreate fields:**

| Field | Type | Required |
|-------|------|----------|
| `artist_id` | string (UUID) | yes |
| `project_id` | string (UUID) | yes |
| `title` | string | yes |
| `work_type` | string | no (default `"single"`) |
| `custom_work_type` | string | no |
| `isrc` | string | no |
| `iswc` | string | no |
| `upc` | string | no |
| `release_date` | date (ISO 8601) | no |
| `notes` | string | no |

**WorkUpdate fields:** all optional — `title`, `work_type`, `project_id`, `isrc`, `iswc`, `upc`, `release_date`, `status`, `notes`.

Work `status` values: `draft` → `pending` → `registered`.

---

### Stakes

Ownership stakes represent percentage shares of master or publishing rights for a work. Total stakes per `stake_type` across a work must not exceed 100%.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/stakes?work_id=<uuid>` | List all stakes for a work. Returns `{ stakes: OwnershipStake[] }`. |
| `POST` | `/registry/stakes` | Create a stake. Validates that adding `percentage` would not exceed 100% for the given `stake_type`. Body: `StakeCreate`. |
| `PUT` | `/registry/stakes/{stake_id}` | Update a stake. Re-validates 100% cap if `percentage` is changed. Body: `StakeUpdate`. |
| `DELETE` | `/registry/stakes/{stake_id}` | Delete a stake. Returns `{ ok: true }`. |

**StakeCreate fields:**

| Field | Type | Required |
|-------|------|----------|
| `work_id` | string (UUID) | yes |
| `stake_type` | `"master"` or `"publishing"` | yes |
| `holder_name` | string | yes |
| `holder_role` | string | yes |
| `percentage` | float (0–100) | yes |
| `holder_email` | string | no |
| `holder_ipi` | string | no |
| `publisher_or_label` | string | no |
| `notes` | string | no |

**StakeUpdate fields:** all optional — same columns as above except `work_id`.

---

### Licensing

License records describe who has been granted rights to a work, in what territory, and for what duration.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/licenses?work_id=<uuid>` | List licenses for a work. Returns `{ licenses: LicensingRight[] }`. |
| `POST` | `/registry/licenses` | Create a license. Body: `LicenseCreate`. |
| `PUT` | `/registry/licenses/{license_id}` | Update a license. Body: `LicenseUpdate`. 404 if not found. |
| `DELETE` | `/registry/licenses/{license_id}` | Delete a license. Returns `{ ok: true }`. |

**LicenseCreate fields:**

| Field | Type | Required |
|-------|------|----------|
| `work_id` | string (UUID) | yes |
| `license_type` | string | yes |
| `licensee_name` | string | yes |
| `licensee_email` | string | no |
| `territory` | string | no (default `"worldwide"`) |
| `start_date` | date (ISO 8601) | yes |
| `end_date` | date (ISO 8601) | no |
| `terms` | string | no |

**LicenseUpdate fields:** all optional — `license_type`, `licensee_name`, `licensee_email`, `territory`, `start_date`, `end_date`, `terms`, `status`.

---

### Agreements

Agreements are immutable records (no update or delete endpoint). They capture signed deals, splits, and other formal documents tied to a work.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/agreements?work_id=<uuid>` | List agreements for a work. Returns `{ agreements: Agreement[] }`. |
| `POST` | `/registry/agreements` | Create an agreement record. Body: `AgreementCreate`. |

**AgreementCreate fields:**

| Field | Type | Required |
|-------|------|----------|
| `work_id` | string (UUID) | yes |
| `agreement_type` | string | yes |
| `title` | string | yes |
| `description` | string | no |
| `effective_date` | date (ISO 8601) | yes |
| `parties` | `Array<{ name, role, email? }>` | yes |
| `file_id` | string (UUID, reference to uploaded file) | no |
| `document_hash` | string (SHA-256 of file) | no |

---

### Collaboration

Collaboration endpoints manage the invite/claim/accept lifecycle for work-level collaborators. Only the work creator or an existing non-revoked collaborator can list collaborators.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/collaborators?work_id=<uuid>` | List collaborators for a work. 403 if caller has no access. Returns `{ collaborators: Collaborator[] }`. |
| `POST` | `/registry/collaborators/invite` | Invite a collaborator by email. Sends a Resend invitation email. Body: `CollaboratorInvite`. |
| `POST` | `/registry/collaborators/invite-with-stakes` | Invite a collaborator and atomically propose stake splits. Sends a rich invitation email. Body: `CollaboratorInviteWithStakes`. 403 on permission failure. |
| `POST` | `/registry/collaborators/claim?invite_token=<token>` | Authenticated user claims their invitation by token. 410 if expired; 404 if token not found or already used. |
| `GET` | `/registry/collaborators/my-invites` | List all pending invitations addressed to the current user's email. Returns `{ invites: DashboardInvite[] }`. |
| `POST` | `/registry/collaborators/{collaborator_id}/confirm` | Work owner confirms a collaborator's stake after the collaborator accepts. |
| `POST` | `/registry/collaborators/{collaborator_id}/revoke` | Work owner revokes an invitation. Only the inviter can call this. |
| `POST` | `/registry/collaborators/{collaborator_id}/resend` | Resend an expired or pending invitation with a fresh token (48-hour expiry). Sends a new email. 404 if not found. |
| `POST` | `/registry/collaborators/{collaborator_id}/decline` | Collaborator declines their invitation. 403 if not authorized; 400 on invalid state. |
| `POST` | `/registry/collaborators/{collaborator_id}/accept-from-dashboard` | Collaborator accepts an invitation directly from the dashboard (no token required). 403 if not authorized. |

**CollaboratorInvite fields:**

| Field | Type | Required |
|-------|------|----------|
| `work_id` | string (UUID) | yes |
| `email` | email string | yes |
| `name` | string | yes |
| `role` | string | yes |
| `stake_id` | string (UUID) | no |

**CollaboratorInviteWithStakes additional fields:**

| Field | Type | Required |
|-------|------|----------|
| `stakes` | `Array<{ stake_type, percentage }>` | no (default `[]`) |
| `notes` | string | no |

Collaborator `status` values: `invited` → `accepted` / `declined` → `confirmed` / `revoked`.

---

### File / Audio Links

These endpoints attach project files and audio files to a specific work. Files live in Supabase Storage and are referenced by UUID.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/works/{work_id}/files` | List file links for a work. Returns `{ files: WorkFileLink[] }`. Each entry embeds the `project_files` row. |
| `POST` | `/registry/works/{work_id}/files?file_id=<uuid>` | Link a project file to a work. Returns `{ link: WorkFileLink }`. |
| `DELETE` | `/registry/works/{work_id}/files/{link_id}` | Remove a file link from a work. |
| `GET` | `/registry/works/{work_id}/audio` | List audio links for a work. Returns `{ audio: WorkAudioLink[] }`. Each entry embeds the `audio_files` row. |
| `POST` | `/registry/works/{work_id}/audio?audio_file_id=<uuid>` | Link an audio file to a work. Returns `{ link: WorkAudioLink }`. |
| `DELETE` | `/registry/works/{work_id}/audio/{link_id}` | Remove an audio link from a work. |

---

### TeamCard

TeamCards are per-user contact cards. A user controls which fields are visible to collaborators via `visible_fields`. Viewing another user's card requires a non-revoked collaboration relationship.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/teamcard` | Fetch the authenticated user's own TeamCard. 404 if onboarding is incomplete. |
| `PUT` | `/registry/teamcard` | Update the authenticated user's TeamCard. Body: `TeamCardUpdate`. Returns updated card. |
| `GET` | `/registry/teamcard/{collaborator_user_id}` | Fetch a collaborator's visible TeamCard fields. 403 if no collaboration link exists. |
| `GET` | `/registry/artists/{artist_id}/with-teamcard` | Fetch an artist record with TeamCard data merged in. 404 if artist not found. |
| `GET` | `/registry/artists/with-teamcards` | Batch endpoint: return all of the user's artists with TeamCard overlays. Returns `{ artists: [...] }`. |

**TeamCardUpdate fields:** all optional — `display_name`, `first_name`, `last_name`, `avatar_url`, `bio`, `phone`, `website`, `company`, `role`, `social_links` (object), `dsp_links` (object), `custom_links` (array of `{ label, url }`), `visible_fields` (string array).

---

### Notes

Notes are BlockNote-format rich-text documents scoped to an artist, project, or folder.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/notes` | List notes. Accepts query params: `artist_id`, `project_id`, `folder_id`. Returns `{ notes: Note[] }`. |
| `GET` | `/registry/notes/{note_id}` | Fetch a single note. 404 if not found. |
| `POST` | `/registry/notes` | Create a note. Body: `NoteCreate`. |
| `PUT` | `/registry/notes/{note_id}` | Update a note. Body: `NoteUpdate`. 404 if not found. |
| `DELETE` | `/registry/notes/{note_id}` | Delete a note. Returns `{ ok: true }`. |
| `GET` | `/registry/folders` | List folders. Accepts `artist_id`, `project_id`. Returns `{ folders: Folder[] }`. |
| `POST` | `/registry/folders` | Create a folder. Body: `FolderCreate`. |
| `PUT` | `/registry/folders/{folder_id}` | Update a folder. Body: `FolderUpdate`. 404 if not found. |
| `DELETE` | `/registry/folders/{folder_id}` | Delete a folder. Returns `{ ok: true }`. |
| `GET` | `/registry/projects/{project_id}/about` | Fetch BlockNote content for the project "About" section. Access restricted to project owners and collaborators. Returns `{ about_content: [...] }`. |
| `PUT` | `/registry/projects/{project_id}/about` | Update the project "About" content. Body: `{ about_content: [...] }`. Returns `{ ok: true }`. |

**NoteCreate fields:**

| Field | Type | Required |
|-------|------|----------|
| `title` | string | no (default `"Untitled"`) |
| `content` | array (BlockNote JSON) | no (default `[]`) |
| `artist_id` | string (UUID) | no |
| `project_id` | string (UUID) | no |
| `folder_id` | string (UUID) | no |
| `pinned` | boolean | no (default `false`) |

**FolderCreate fields:** `name` (required), `artist_id`, `project_id`, `parent_folder_id`, `sort_order` (default 0).

---

### Notifications

Registry notifications are generated server-side when collaboration events occur (invitations, confirmations, status changes). The hook polls every 30 seconds.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/notifications` | List notifications for the user. Accepts `unread_only=true/false` (default `false`). Returns `{ notifications: RegistryNotification[] }`. |
| `POST` | `/registry/notifications/{notification_id}/read` | Mark a single notification as read. Returns `{ ok: true }`. |
| `POST` | `/registry/notifications/read-all` | Mark all of the user's notifications as read. Returns `{ ok: true }`. |

Notification `type` values observed in the UI: `invitation`, `confirmation`, `dispute`, `status_change`.

---

## Collaboration Flow

The Rights Registry uses a multi-step invite-and-confirm flow to ensure all parties explicitly agree to ownership splits before a work is registered.

```
Owner creates work (status: draft)
        |
        v
Owner adds ownership stakes (master / publishing %)
        |
        v
Owner invites collaborator(s)
  - POST /registry/collaborators/invite          (basic, no stakes)
  - POST /registry/collaborators/invite-with-stakes  (rich, proposes splits)
        |
        | Invitation email sent via Resend (invite_token, 48-hour expiry)
        v
Collaborator receives email → clicks link → lands on /tools/registry/invite/{token}
        |
        v
Collaborator claims the token (must be authenticated)
  POST /registry/collaborators/claim?invite_token=<token>
  - status: invited → accepted
  - 410 if token expired (owner must resend)
        |
        v
Collaborator optionally accepts/declines from dashboard (no token needed)
  POST /registry/collaborators/{id}/accept-from-dashboard
  POST /registry/collaborators/{id}/decline
        |
        v
Owner confirms collaborator's stake
  POST /registry/collaborators/{collaborator_id}/confirm
  - status: accepted → confirmed
        |
        v
Owner submits work for approval (all collaborators must be confirmed)
  POST /registry/works/{work_id}/submit-for-approval
  - status: draft → pending
  - Re-sends invitation emails to any still-pending collaborators
        |
        v
Work is eventually marked: status → registered
```

**Edge cases:**

- If a collaborator's invitation expires, the owner calls `POST /registry/collaborators/{id}/resend` to issue a fresh token and resend the email.
- The owner can call `POST /registry/collaborators/{id}/revoke` at any time to remove a collaborator. Revoked collaborators are excluded from work views and cannot be re-listed.
- The `GET /registry/collaborators/my-invites` endpoint lets a collaborator see all pending invitations addressed to their email from the in-app dashboard without needing the email link.

---

## Frontend Hooks

All hooks live in `src/hooks/`. They wrap TanStack React Query and call the backend via `apiFetch` (which attaches the Supabase JWT automatically).

### useRegistry.ts

| Hook | Returns | Description |
|------|---------|-------------|
| `useWorks(artistId?)` | `Work[]` | List works owned by current user, optionally filtered by artist. Query key: `["registry-works", userId, artistId]`. |
| `useMyCollaborations()` | `Work[]` | List works where current user is a collaborator. |
| `useWorksByProject(projectId?)` | `Work[]` | List works scoped to a project. |
| `useWorkFull(workId?)` | `WorkFull \| null` | Fetch a single work with stakes, licenses, agreements, collaborators embedded. |
| `useCreateWork()` | `UseMutationResult` | Create a work. Uses optimistic update to show work immediately in the project list. |
| `useUpdateWork()` | `UseMutationResult` | Update a work. Uses optimistic update on the `registry-work-full` cache. |
| `useDeleteWork()` | `UseMutationResult` | Delete a work. |
| `useCreateStake()` | `UseMutationResult` | Add an ownership stake to a work. |
| `useUpdateStake()` | `UseMutationResult` | Update an existing stake. |
| `useDeleteStake()` | `UseMutationResult` | Remove a stake. |
| `useCreateLicense()` | `UseMutationResult` | Add a license record. |
| `useUpdateLicense()` | `UseMutationResult` | Update a license. |
| `useDeleteLicense()` | `UseMutationResult` | Remove a license. |
| `useCreateAgreement()` | `UseMutationResult` | Record an immutable agreement. |
| `useInviteCollaborator()` | `UseMutationResult` | Send a basic collaborator invitation. |
| `useClaimInvitation()` | `UseMutationResult` | Claim an invitation by token. |
| `useConfirmStake()` | `UseMutationResult` | Owner confirms collaborator's stake. |
| `useSubmitForApproval()` | `UseMutationResult` | Submit a work for approval, transitioning status to `pending`. |
| `useRevokeCollaborator()` | `UseMutationResult` | Revoke a collaborator. |
| `useResendInvitation()` | `UseMutationResult` | Resend a fresh invitation email to an existing collaborator record. |
| `useMyInvites()` | `DashboardInvite[]` | List the current user's pending invitations (dashboard view). Query key: `["registry-my-invites", userId]`. |
| `useAcceptFromDashboard()` | `UseMutationResult` | Accept an invitation from the dashboard. |
| `useDeclineInvitation()` | `UseMutationResult` | Decline an invitation. |
| `useExportProof()` | `UseMutationResult` | Download the Proof of Ownership PDF for a work (triggers a browser file download). |

### useWorkFiles.ts

| Hook | Returns | Description |
|------|---------|-------------|
| `useWorkFiles(workId?)` | `WorkFileLink[]` | List project files linked to a work. Query key: `["work-files", workId]`. |
| `useLinkFileToWork()` | `UseMutationResult` | Link a project file (`file_id`) to a work. |
| `useUnlinkFileFromWork()` | `UseMutationResult` | Remove a file link by `link_id`. |

### useWorkAudio.ts

| Hook | Returns | Description |
|------|---------|-------------|
| `useWorkAudio(workId?)` | `WorkAudioLink[]` | List audio files linked to a work. Query key: `["work-audio", workId]`. |
| `useLinkAudioToWork()` | `UseMutationResult` | Link an audio file (`audio_file_id`) to a work. |
| `useUnlinkAudioFromWork()` | `UseMutationResult` | Remove an audio link by `link_id`. |

### useRegistryNotifications.ts

| Hook | Returns | Description |
|------|---------|-------------|
| `useRegistryNotifications(unreadOnly?)` | `RegistryNotification[]` | Fetch registry notifications. Polls every 30 seconds. Query key: `["registry-notifications", userId, unreadOnly]`. |
| `useUnreadCount()` | `number` | Combined unread count: registry notifications + Slack unread badge. |
| `useMarkNotificationRead()` | `UseMutationResult` | Mark a single notification as read. |
| `useMarkAllRead()` | `UseMutationResult` | Mark all notifications as read. |

### useTeamCard.ts

| Hook | Returns | Description |
|------|---------|-------------|
| `useMyTeamCard()` | `TeamCard \| null` | Fetch the current user's own TeamCard. Query key: `["team-card", userId]`. |
| `useUpdateTeamCard()` | `UseMutationResult` | Update own TeamCard fields (all optional, email/id excluded). |
| `useCollaboratorTeamCard(collaboratorUserId?)` | `TeamCard \| null` | Fetch a collaborator's visible TeamCard. Requires an existing non-revoked collaboration. Query key: `["collaborator-team-card", collaboratorUserId]`. |

---

## Frontend Components

| Component | File | Purpose |
|-----------|------|---------|
| `OwnershipPanel` | `src/components/registry/OwnershipPanel.tsx` | CRUD UI for ownership stakes. Displays master/publishing splits with per-row collaborator confirmation status badges. |
| `LicensingPanel` | `src/components/registry/LicensingPanel.tsx` | CRUD UI for license records tied to a work. |
| `AgreementsPanel` | `src/components/registry/AgreementsPanel.tsx` | Display and creation of immutable agreement records. |
| `CollaborationStatus` | `src/components/registry/CollaborationStatus.tsx` | Collaboration workflow panel: shows collaborator list with `invited`/`accepted`/`confirmed` status, provides submit-for-approval, resend, revoke, and decline actions. |
| `InviteCollaboratorModal` | `src/components/registry/InviteCollaboratorModal.tsx` | Modal form for inviting a new collaborator (email, name, role, optional stake). |
| `ProofOfOwnership` | `src/components/registry/ProofOfOwnership.tsx` | Button/trigger that calls `useExportProof` to download the PDF ownership certificate. |
| `RegistryNotifications` | `src/components/workspace/RegistryNotifications.tsx` | Notification feed shown in the Workspace panel. Renders registry notifications with type-colored badges (`invitation`, `confirmation`, `dispute`, `status_change`) and a "mark all read" action. Clicking a notification navigates to `/tools/registry/{work_id}`. Also renders `SlackMentions` above the registry list. |

---

## Database Tables

All tables use UUID primary keys (`gen_random_uuid()`) and `created_at`/`updated_at` timestamps. RLS is enabled; policies check `auth.uid()`.

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `works_registry` | `id`, `user_id`, `artist_id`, `project_id`, `title`, `work_type`, `isrc`, `iswc`, `upc`, `release_date`, `status` | Core work records. `status`: `draft` → `pending` → `registered`. |
| `ownership_stakes` | `id`, `work_id`, `user_id`, `stake_type`, `holder_name`, `holder_role`, `percentage`, `holder_email`, `holder_ipi`, `publisher_or_label` | Master and publishing ownership splits. Total per `(work_id, stake_type)` must not exceed 100. |
| `licensing_rights` | `id`, `work_id`, `user_id`, `license_type`, `licensee_name`, `licensee_email`, `territory`, `start_date`, `end_date`, `terms`, `status` | Licensing grants for a work. |
| `work_agreements` | `id`, `work_id`, `user_id`, `agreement_type`, `title`, `description`, `effective_date`, `parties` (jsonb), `file_id`, `document_hash` | Immutable agreement records. No update/delete endpoints. |
| `registry_collaborators` | `id`, `work_id`, `stake_id`, `invited_by`, `collaborator_user_id`, `email`, `name`, `role`, `status`, `invite_token`, `expires_at`, `invited_at`, `responded_at` | Invitation and acceptance state per collaborator per work. `status`: `invited` → `accepted`/`declined`/`confirmed`/`revoked`. |
| `work_files` | `id`, `work_id`, `file_id`, `created_at` | Join table linking `works_registry` to `project_files`. |
| `work_audio_links` | `id`, `work_id`, `audio_file_id`, `created_at` | Join table linking `works_registry` to `audio_files`. |
| `team_cards` | `id`, `user_id`, `display_name`, `first_name`, `last_name`, `email`, `avatar_url`, `bio`, `phone`, `website`, `company`, `role`, `social_links` (jsonb), `dsp_links` (jsonb), `custom_links` (jsonb), `visible_fields` (text[]) | Per-user contact cards surfaced to collaborators. |
| `registry_notes` | `id`, `user_id`, `artist_id`, `project_id`, `folder_id`, `title`, `content` (jsonb, BlockNote), `pinned` | Rich-text notes scoped to artist or project. |
| `registry_folders` | `id`, `user_id`, `artist_id`, `project_id`, `parent_folder_id`, `name`, `sort_order` | Folder hierarchy for notes. |
| `projects` | `id`, `artist_id`, `name`, `about_content` (jsonb, BlockNote) | Project records; `about_content` is managed via `/registry/projects/{id}/about`. |
| `registry_notifications` | `id`, `user_id`, `work_id`, `type`, `title`, `message`, `read`, `metadata` (jsonb) | In-app notification events for collaboration activity. |

---

## Local Testing

Start the backend locally before running these:

```bash
cd src/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Obtain a JWT from Supabase (or copy one from browser DevTools → Network → any request's `Authorization` header) and export it:

```bash
export JWT="eyJ..."
export BASE="http://localhost:8000"
```

### Create a work

```bash
curl -s -X POST "$BASE/registry/works" \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "artist_id": "<artist-uuid>",
    "project_id": "<project-uuid>",
    "title": "Test Track",
    "work_type": "single",
    "isrc": "USRC12345678"
  }' | jq .
```

### List works (paginated)

```bash
curl -s "$BASE/registry/works?page=1&page_size=10" \
  -H "Authorization: Bearer $JWT" | jq .
```

### Add an ownership stake

```bash
curl -s -X POST "$BASE/registry/stakes" \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "work_id": "<work-uuid>",
    "stake_type": "master",
    "holder_name": "Jane Doe",
    "holder_role": "Artist",
    "percentage": 50
  }' | jq .
```

### Invite a collaborator

```bash
curl -s -X POST "$BASE/registry/collaborators/invite" \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "work_id": "<work-uuid>",
    "email": "collaborator@example.com",
    "name": "Collaborator Name",
    "role": "Producer"
  }' | jq .
```

### Claim an invitation (as the collaborator)

```bash
curl -s -X POST "$BASE/registry/collaborators/claim?invite_token=<token>" \
  -H "Authorization: Bearer $COLLABORATOR_JWT" | jq .
```

### Submit work for approval

```bash
curl -s -X POST "$BASE/registry/works/<work-uuid>/submit-for-approval" \
  -H "Authorization: Bearer $JWT" | jq .
```

### Download Proof of Ownership PDF

```bash
curl -s -o "proof.pdf" \
  -H "Authorization: Bearer $JWT" \
  "$BASE/registry/works/<work-uuid>/export"
```

### List unread notifications

```bash
curl -s "$BASE/registry/notifications?unread_only=true" \
  -H "Authorization: Bearer $JWT" | jq .
```

### Automated tests

No `tests/test_registry_*.py` files exist in `src/backend/tests/` at this time. When test files are added, run them with:

```bash
cd src/backend
pytest tests/test_registry_*.py -v
```
