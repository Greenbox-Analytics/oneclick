"""Integration test for full account deletion against real Supabase (+ optional Stripe).

Gated by RUN_INTEGRATION_TESTS=1 so it never runs in CI / normal test runs.
Requires VITE_SUPABASE_URL and VITE_SUPABASE_SECRET_KEY in the environment.

Exercises the Task 1 cascade migration: creates two registry_collaborators rows
(one as invited_by, one as collaborator_user_id) and asserts both behave per
the migration spec — CASCADE on invited_by, SET NULL on collaborator_user_id.
Without the migration this test would fail with a FK violation when
delete_user_account runs auth.admin.delete_user.
"""

import os
import uuid

import pytest

RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION_TESTS=1 to run")


def _sb():
    from supabase import create_client

    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SECRET_KEY")
    if not (url and key):
        pytest.skip("supabase env not set")
    return create_client(url, key)


@pytest.fixture
def temp_user_with_data():
    """Create a user with: artist, project, project_file + storage object,
    a registry_collaborators row where user is invited_by (CASCADE target),
    and a registry_collaborators row where user is collaborator_user_id
    (SET NULL target). The two registry rows ensure the integration test
    actually exercises the Task 1 cascade migration — without them, the
    test passes even if the FK fix is missing.
    """
    sb = _sb()
    email = f"acct-del-{uuid.uuid4()}@test.local"
    created = sb.auth.admin.create_user({"email": email, "password": "test-pw-1234567890!"})
    user_id = created.user.id

    # Second user we keep around so the SET NULL row has an inviter that stays.
    other_email = f"acct-del-other-{uuid.uuid4()}@test.local"
    other_user = sb.auth.admin.create_user({"email": other_email, "password": "test-pw-1234567890!"}).user
    other_user_id = other_user.id

    artist_id = sb.table("artists").insert({"user_id": user_id, "name": f"A {uuid.uuid4()}"}).execute().data[0]["id"]
    project_id = (
        sb.table("projects").insert({"artist_id": artist_id, "name": f"P {uuid.uuid4()}"}).execute().data[0]["id"]
    )

    file_path = f"{artist_id}/{project_id}/test-{uuid.uuid4()}.txt"
    sb.storage.from_("project-files").upload(file_path, b"hello", {"content-type": "text/plain"})
    sb.table("project_files").insert(
        {
            "project_id": project_id,
            "file_path": file_path,
            "filename": "test.txt",
            "uploaded_by": user_id,
        }
    ).execute()

    # works_registry requires user_id, artist_id, project_id, title — all NOT NULL.
    # Build a second artist+project owned by other_user_id so the SET NULL row
    # is NOT cascade-deleted via the artist FK chain when user_id is deleted.
    other_artist_id = (
        sb.table("artists").insert({"user_id": other_user_id, "name": f"OA {uuid.uuid4()}"}).execute().data[0]["id"]
    )
    other_project_id = (
        sb.table("projects")
        .insert({"artist_id": other_artist_id, "name": f"OP {uuid.uuid4()}"})
        .execute()
        .data[0]["id"]
    )

    work_id = (
        sb.table("works_registry")
        .insert(
            {
                "user_id": user_id,
                "artist_id": artist_id,
                "project_id": project_id,
                "title": f"W {uuid.uuid4()}",
            }
        )
        .execute()
        .data[0]["id"]
    )
    other_work_id = (
        sb.table("works_registry")
        .insert(
            {
                "user_id": other_user_id,
                "artist_id": other_artist_id,
                "project_id": other_project_id,
                "title": f"OW {uuid.uuid4()}",
            }
        )
        .execute()
        .data[0]["id"]
    )

    # Row 1: user is the inviter → CASCADE.
    invited_by_row_id = (
        sb.table("registry_collaborators")
        .insert(
            {
                "work_id": work_id,
                "invited_by": user_id,
                "collaborator_user_id": other_user_id,
                "email": other_email,
                "name": "Test Collaborator",
                "role": "writer",
                "status": "invited",
            }
        )
        .execute()
        .data[0]["id"]
    )

    # Row 2: user is the invitee on OTHER user's work → SET NULL.
    collaborator_row_id = (
        sb.table("registry_collaborators")
        .insert(
            {
                "work_id": other_work_id,
                "invited_by": other_user_id,
                "collaborator_user_id": user_id,
                "email": email,
                "name": "Test Invitee",
                "role": "writer",
                "status": "invited",
            }
        )
        .execute()
        .data[0]["id"]
    )

    yield {
        "user_id": user_id,
        "email": email,
        "other_user_id": other_user_id,
        "artist_id": artist_id,
        "project_id": project_id,
        "file_path": file_path,
        "invited_by_row_id": invited_by_row_id,
        "collaborator_row_id": collaborator_row_id,
        "other_work_id": other_work_id,
    }

    # Cleanup — delete both users if they survived.
    for uid in (user_id, other_user_id):
        try:
            sb.auth.admin.delete_user(uid)
        except Exception:
            pass


def test_delete_user_account_full_cascade(temp_user_with_data):
    from users.account_deletion_service import delete_user_account

    sb = _sb()
    d = temp_user_with_data

    delete_user_account(sb, d["user_id"], d["email"])

    # Auth user gone
    got_user = True
    try:
        sb.auth.admin.get_user_by_id(d["user_id"])
    except Exception:
        got_user = False
    assert not got_user, "auth user should be deleted"

    # Cascaded rows
    assert sb.table("artists").select("id").eq("id", d["artist_id"]).execute().data == []
    assert sb.table("projects").select("id").eq("id", d["project_id"]).execute().data == []
    assert sb.table("project_files").select("id").eq("project_id", d["project_id"]).execute().data == []

    # registry_collaborators — the rows that the Task 1 migration is about.
    # Without that migration, delete_user_account above would have raised a
    # FK violation BEFORE reaching this assertion.
    invited_by_rows = sb.table("registry_collaborators").select("id").eq("id", d["invited_by_row_id"]).execute().data
    assert invited_by_rows == [], "invited_by row should cascade-delete"

    collaborator_rows = (
        sb.table("registry_collaborators")
        .select("id, collaborator_user_id")
        .eq("id", d["collaborator_row_id"])
        .execute()
        .data
    )
    assert len(collaborator_rows) == 1, "collaborator_user_id row should remain (SET NULL, not CASCADE)"
    assert collaborator_rows[0]["collaborator_user_id"] is None, "collaborator_user_id should be NULL after delete"

    # Storage cleaned
    listing = sb.storage.from_("project-files").list(f"{d['artist_id']}/{d['project_id']}")
    file_name = d["file_path"].split("/")[-1]
    assert not any(item.get("name", "").endswith(file_name) for item in (listing or []))


def test_delete_user_account_blocks_sole_db_admin(temp_user_with_data):
    from users.account_deletion_service import LastAdminError, delete_user_account

    sb = _sb()
    d = temp_user_with_data

    sb.table("profiles").upsert({"id": d["user_id"], "is_admin": True}).execute()

    other_db_admins = [
        r for r in sb.table("profiles").select("id").eq("is_admin", True).execute().data if r["id"] != d["user_id"]
    ]
    if other_db_admins:
        pytest.skip("other db admins present; cannot test sole-admin case in this env")

    with pytest.raises(LastAdminError):
        delete_user_account(sb, d["user_id"], d["email"])
