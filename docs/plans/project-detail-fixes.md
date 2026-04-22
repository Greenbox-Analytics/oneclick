# Project Detail page — fix plan

## Context

This plan bundles (a) the bugs surfaced by auditing the Project Detail page and (b) a set of UI cleanups and feature asks from the user. Items are grouped by priority so we can ship them in waves without blocking on the larger pieces.

The page in scope is [src/pages/ProjectDetail.tsx](../../src/pages/ProjectDetail.tsx) and its tab components in [src/components/project/](../../src/components/project/).

---

## Priority 0 — Quick UI fixes (do first, minutes of work)

### 1. Disable the "Works" tab

Change [src/pages/ProjectDetail.tsx:213-215](../../src/pages/ProjectDetail.tsx#L213-L215) — add `disabled` to the `<TabsTrigger value="works">`. Also: if `activeTab === "works"` is the default, switch the default to `"files"` so new visitors don't land on a disabled tab. Remove or gate the `<TabsContent value="works">` at line 233 so it won't render even if some stale URL still points at it.

### 2. Remove the "Add Work" button at the top-right

Delete the block at [src/pages/ProjectDetail.tsx:199-203](../../src/pages/ProjectDetail.tsx#L199-L203) (the `canEdit(userRole) && <Button … Add Work>` group). Also remove the now-unused `setAddWorkOpen` state and the `AddWorkDialog` mounted from this page (search for `addWorkOpen`). The add-work path remains available inside the Works tab itself (once/if re-enabled).

### 3. Remove the duplicate "Upload Audio" button in AudioTab

[src/components/project/AudioTab.tsx](../../src/components/project/AudioTab.tsx) has two:
- **Top-right**, default variant (green) — [lines 215-222](../../src/components/project/AudioTab.tsx#L215-L222).
- **Empty-state outline**, not green — [lines 239-246](../../src/components/project/AudioTab.tsx#L239-L246).

The user wants to keep the **non-green** (outline) one. Two options, pick one:

- **Literal**: just delete the top-right button. Upload is then only visible when there are zero audio files. ❗ After the first upload, there's no way to upload again. Not recommended.
- **Recommended**: delete the top-right button AND move the outline button out of the empty-state branch so it's always visible — same non-green styling, always available.

Go with the recommended variant.

---

## Priority 1 — Fix audio upload (blocking)

Users cannot upload audio. Two root causes stack:

### 4. Storage bucket `audio-files` doesn't exist

[AudioTab.tsx:147-149](../../src/components/project/AudioTab.tsx#L147-L149) calls `supabase.storage.from("audio-files").upload(...)`, but no migration creates that bucket. The only bucket in [supabase/migrations/20251121152900_fix_artist_files_schema.sql:17-40](../../supabase/migrations/20251121152900_fix_artist_files_schema.sql#L17-L40) is `project-files`.

**Fix**: new migration `supabase/migrations/YYYYMMDD######_create_audio_files_bucket.sql` that:
- Inserts a row into `storage.buckets` with `id='audio-files'`, `public=false` (we'll use signed URLs — see Priority 2 item #13).
- Creates the four RLS policies on `storage.objects` for `bucket_id = 'audio-files'`: insert / select / update / delete, each gated by membership of the project that owns the folder. Mirror the structure of the `project-files` policies but scope by artist ownership (`audio_folders.artist_id` → `artists.user_id = auth.uid()`).

### 5. Non-atomic upload leaves orphans

[AudioTab.tsx:173-177](../../src/components/project/AudioTab.tsx#L173-L177) — if `project_audio_links` insert fails, the storage object AND the `audio_files` row are already committed. Extend the cleanup at line 169 to also fire on `linkError`: delete the storage object AND the `audio_files` row before re-throwing.

### 6. Uses public URL in a private bucket

[AudioTab.tsx:152, 318](../../src/components/project/AudioTab.tsx#L152) — replace `getPublicUrl(filePath)` with `createSignedUrl(filePath, 60 * 60)` for the "Open" action. For playback later (if we add an inline player), request the signed URL on demand. Don't store a URL on the `audio_files` row — store only the `file_path` and generate signed URLs at access time.

### 7. Add AudioTab delete action

Users have no delete affordance — add a trash `Button` next to "Open" ([around line 313-322](../../src/components/project/AudioTab.tsx#L313-L322)) that:
- Confirms via an `AlertDialog`.
- Deletes `project_audio_links` row, `work_audio_links` rows, `audio_files` row, and the storage object. Run them sequentially in the order that leaves orphans impossible (links → row → storage).
- Invalidates `["project-audio-tab", projectId]`.

### 8. No backend MIME / size validation

Add server-side validation on any new endpoint that handles audio (and on the storage path if we move to a backend-proxied upload). For now, at minimum: add a client-side size cap (e.g. 50 MB) and a MIME-type whitelist in [AudioTab.tsx handleUpload](../../src/components/project/AudioTab.tsx#L118) as a defense-in-depth line, with toast on rejection.

### 9. SHA-256 deduplication

Add a `content_hash TEXT` column to `audio_files` via migration, compute SHA-256 client-side before upload (mirror [FilesTab.tsx lines 122-148 pattern](../../src/components/project/FilesTab.tsx#L122-L148)), and check if a row already exists for this `(artist_id, content_hash)` tuple — if yes, just link the existing `audio_files` row to the project instead of re-uploading.

---

## Priority 1 — Fix invites / Resend

### 10. `profiles.email` lookup is broken

[src/backend/projects/service.py:32](../../src/backend/projects/service.py#L32) queries `profiles.email`, but [the profiles schema](../../supabase/migrations/20260122000000_create_profiles_table.sql) has no `email` column. Result: every invite falls into the "pending" branch, even for existing users.

**Fix**: look up the email in `auth.users` via the admin/service-role client. Options:
- Cleanest: add a backend service function `get_user_id_by_email(email)` that uses the service-role Supabase client to query `auth.users`.
- Alternative: add a `profiles.email` column and backfill it via a trigger on `auth.users` insert / update (plus backfill existing rows).

Pick the admin-client lookup — it's one query, no schema change.

### 11. Silent Resend failures

[src/backend/projects/router.py:52-53](../../src/backend/projects/router.py#L52-L53) and [projects/emails.py:59-61](../../src/backend/projects/emails.py#L59-L61) swallow exceptions and only `print`. Do:
- In `emails.py`, **raise** on failure instead of returning `None` silently (keep one `try/except` wrapper at the caller to decide what to do).
- In the background task at `router.py:38-53`, on failure: update the `pending_project_invites` row to set a `last_email_error TEXT` + `last_email_attempt_at TIMESTAMPTZ` (new columns via migration) so the UI can show "delivery failed — retry".
- Expose a `POST /projects/{project_id}/pending-invites/{invite_id}/resend` endpoint.

### 12. Duplicate-invite → 500

[service.py add_member](../../src/backend/projects/service.py#L23) doesn't catch the `UNIQUE(project_id, email)` violation. Wrap the `pending_project_invites` insert in a try/except that maps the specific Postgres unique-violation code (`23505`) to an HTTP 409 with a friendly message.

### 13. Pending-invites query not invalidated

[src/hooks/useProjectMembers.ts:57-58](../../src/hooks/useProjectMembers.ts#L57-L58) — invalidate `["project-pending-invites", projectId]` in addition to `["project-members", projectId]` on `onSuccess`.

---

## Priority 2 — New feature: send files/audio via email (Resend)

### 14. "Share via email" action for files and audio

Scope for v1:
- Add a backend endpoint: `POST /projects/{project_id}/share-email`
  - Body: `{ recipient_email: str, subject: str, message: str, file_ids: list[UUID], audio_file_ids: list[UUID] }`
  - Permission: caller must be project member (editor+).
  - For each id: fetch `file_path`, download from the appropriate bucket via service-role Supabase client, attach to the Resend email. Attachment limit: Resend caps at **40 MB total** per email — enforce that server-side and return 413 if exceeded.
  - Use `resend.Emails.send(...)` with `attachments=[{"filename": name, "content": base64_bytes}]`.
- New email template in a new file `src/backend/projects/share_email.py` (distinct from `emails.py` to keep invite logic separate).
- Frontend: add a "Share via email" menu item per file/audio row (dropdown on the row) and a bulk action when multiple are selected. New dialog with email, subject, optional message, preview of attachment list, size total, send button. Use TanStack mutation; toast success/failure.

Edge cases to handle explicitly:
- File in storage is missing → return 404 and list which files failed.
- Exceeds 40 MB → return 413 with a friendly message suggesting the user share fewer files or use a signed link instead. (v2 idea: fall back to sending signed URLs in the email body when over the cap.)

---

## Priority 2 — UI polish

### 15. Disable destructive actions during mutations

[SettingsTab.tsx](../../src/components/project/SettingsTab.tsx) — delete-project button: add `disabled={deleteProject.isPending}` and a spinner. Do the same for any unprotected destructive button in other tabs.

### 16. `linkingInProgress` race in AudioTab

[AudioTab.tsx:102-115](../../src/components/project/AudioTab.tsx#L102-L115) — disable the `Select`/Link button while the mutation is pending (the flag is already there; wire it into the button's `disabled`).

### 17. Member name loading flash

[MembersTab.tsx](../../src/components/project/MembersTab.tsx) — defer rendering the list until the profile / team-card lookups resolve, or show a skeleton row instead of the "Member" fallback.

---

## Deferred / larger scope

### Invite accept/reject flow

Currently invites are auto-claimed by a DB trigger on `auth.users` insert. There's no way for an invitee to decline. This is a larger UX change (token-based invite links, decline endpoint, dedicated claim page). Not scoped in this plan — open a separate spec.

---

## Rollout order (suggested)

1. **Wave 1 (one PR)** — items #1, #2, #3 (pure UI, low risk).
2. **Wave 2 (one PR, + one migration)** — items #4, #5, #6, #7 (unblock audio upload end-to-end). Requires running the new bucket migration.
3. **Wave 3 (one PR)** — items #10, #11, #12, #13 (invite / Resend correctness).
4. **Wave 4 (one PR, + one migration)** — items #8, #9 (audio hardening).
5. **Wave 5 (feature PR)** — item #14 (share via email).
6. **Wave 6 (polish PR)** — items #15, #16, #17.

## Verification per wave

After each wave:

```bash
npm run build
cd src/backend && poetry run ruff check . && poetry run ruff format --check .
cd src/backend && poetry run pytest -v
```

End-to-end manual checks specific to each wave:
- Wave 1: Works tab is disabled, no top-right Add Work, only one non-green Upload Audio button visible.
- Wave 2: Upload an audio file, it appears in the tab, "Open" plays via signed URL, delete removes storage + DB + links.
- Wave 3: Invite an existing user → they're auto-added (no pending row). Invite a new user → they get the email. Invite the same email twice → 409. Invite shows up in UI without refresh.
- Wave 4: Upload oversized / non-audio file → rejected. Re-upload same file → dedup, no duplicate row.
- Wave 5: Select a file + audio, share via email, attachment arrives. Try 50 MB → 413.
- Wave 6: Spam-click delete-project → single request fires.

## Critical files touched

| File | What changes |
|---|---|
| [src/pages/ProjectDetail.tsx](../../src/pages/ProjectDetail.tsx) | Disable Works tab, remove Add Work button/dialog |
| [src/components/project/AudioTab.tsx](../../src/components/project/AudioTab.tsx) | Remove duplicate button, atomic upload, signed URLs, delete action, validation, dedup |
| [src/components/project/SettingsTab.tsx](../../src/components/project/SettingsTab.tsx) | Disable delete during pending |
| [src/components/project/MembersTab.tsx](../../src/components/project/MembersTab.tsx) | Skeleton while profiles load |
| [src/hooks/useProjectMembers.ts](../../src/hooks/useProjectMembers.ts) | Invalidate pending-invites key |
| [src/backend/projects/service.py](../../src/backend/projects/service.py) | auth.users email lookup, duplicate-invite 409 |
| [src/backend/projects/router.py](../../src/backend/projects/router.py) | Resend error persistence, resend endpoint, share endpoint |
| [src/backend/projects/emails.py](../../src/backend/projects/emails.py) | Raise on failure instead of swallow |
| `src/backend/projects/share_email.py` *(new)* | Share-via-email implementation |
| `supabase/migrations/YYYYMMDD_create_audio_files_bucket.sql` *(new)* | Bucket + RLS policies |
| `supabase/migrations/YYYYMMDD_add_audio_content_hash.sql` *(new)* | `content_hash` column |
| `supabase/migrations/YYYYMMDD_add_invite_email_status.sql` *(new)* | `last_email_error`, `last_email_attempt_at` |
