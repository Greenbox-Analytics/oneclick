"""Shared storage-cap gate for provider imports (Google Drive, Dropbox).

Both import paths write into the same project-files bucket and project_files
table and must be cleaned up the same way on a rejected insert, and mapped to
the same typed error — this module owns that sequence once so the two
providers can't drift out of sync.
"""

import time

from supabase import Client

from subscriptions.enforcement import gated_upload


class StorageCapExceededError(Exception):
    """The project_files INSERT trigger rejected the row for storage cap —
    either the pre-check below was skipped (caller-is-owner unknown) or a
    race beat it. Callers' routers map this to 402, same as the direct-upload
    endpoint in main.py."""


def store_imported_file(
    supabase: Client,
    user_id: str,
    project_id: str,
    file_name: str,
    content: bytes,
    *,
    mime: str,
    folder_category: str,
    file_size: int | None = None,
    owner_user_id: str | None = None,
) -> dict:
    """Gate against the storage cap, write to the project-files bucket, and insert
    the project_files row — removing the uploaded object if the insert is rejected.

    Shared by the Google Drive and Dropbox import paths.

    `owner_user_id` must be passed ONLY when the caller is known to be the
    project's storage-counter owner (the storage triggers route by ARTIST
    ownership — the org for a team artist, else the creator — which a plain
    project-member role check cannot establish). When it is None, the
    pre-check is skipped entirely and the DB trigger is the sole enforcement;
    StorageCapExceededError is that backstop, not merely a race-loser branch.
    Gating a non-owner's wallet would produce false 402s for editors
    importing into someone else's project, so never guess this value.

    Returns the inserted project_files row, or {} if the insert echoed no data
    (pre-existing shape both callers already handle).
    """
    if owner_user_id is not None:
        gated_upload(
            owner_user_id,
            size=file_size if file_size is not None else len(content),
            host_user_id=owner_user_id,
            resource_project_id=project_id,
        )

    timestamp = int(time.time())
    storage_path = f"{user_id}/{project_id}/{timestamp}_{file_name}"
    supabase.storage.from_("project-files").upload(
        storage_path,
        content,
        file_options={"content-type": mime},
    )

    file_url = supabase.storage.from_("project-files").get_public_url(storage_path)
    file_record = {
        "project_id": project_id,
        "file_name": file_name,
        "folder_category": folder_category,
        "file_path": storage_path,
        "file_url": file_url,
        "file_size": file_size,
        "file_type": mime,
    }
    try:
        result = supabase.table("project_files").insert(file_record).execute()
    except Exception as db_error:
        # The blob is already in Storage; the row failed. Clean up the orphan
        # rather than leaving an object nothing references and no counter counts.
        try:
            supabase.storage.from_("project-files").remove([storage_path])
        except Exception as cleanup_error:
            print(f"Failed to cleanup uploaded file after DB error: {cleanup_error}")

        error_message = str(db_error)
        if "Storage cap exceeded" in error_message or "23514" in error_message:
            raise StorageCapExceededError(
                "This import would exceed the storage limit for this project. Free up space or upgrade to Pro."
            ) from db_error
        raise

    return result.data[0] if result.data else {}
