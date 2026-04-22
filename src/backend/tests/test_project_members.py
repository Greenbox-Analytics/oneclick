"""Tests for project member management endpoints.

Acceptance criteria:
1. List project members
2. Add member (existing user → direct add; new email → pending invite + email mocked)
3. Update member role
4. Remove member
5. List pending invites
6. Cancel pending invite
"""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

PROJECT_ID = "proj-00000000-0000-0000-0000-000000000001"
MEMBER_ID = "memb-00000000-0000-0000-0000-000000000001"
OTHER_USER_ID = "00000000-0000-0000-0000-000000000002"
INVITE_ID = "invt-00000000-0000-0000-0000-000000000001"

OWNER_MEMBER = {
    "id": MEMBER_ID,
    "project_id": PROJECT_ID,
    "user_id": TEST_USER_ID,
    "role": "owner",
    "invited_by": None,
    "created_at": "2025-01-01T00:00:00+00:00",
}

EDITOR_MEMBER = {
    "id": "memb-00000000-0000-0000-0000-000000000002",
    "project_id": PROJECT_ID,
    "user_id": OTHER_USER_ID,
    "role": "editor",
    "invited_by": TEST_USER_ID,
    "created_at": "2025-01-02T00:00:00+00:00",
}

ADMIN_MEMBER = {
    **OWNER_MEMBER,
    "role": "admin",
}

PENDING_INVITE = {
    "id": INVITE_ID,
    "project_id": PROJECT_ID,
    "email": "newuser@example.com",
    "role": "viewer",
    "invited_by": TEST_USER_ID,
    "created_at": "2025-01-03T00:00:00+00:00",
}

EXISTING_PROFILE = {
    "id": OTHER_USER_ID,
    "email": "existing@example.com",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder(data):
    """Return a MockQueryBuilder pre-loaded with the given data.

    ``data`` may be a list (normal SELECT) or a dict (maybe_single / single
    responses), matching how supabase-py returns results.
    """
    b = MockQueryBuilder()
    count = len(data) if isinstance(data, list) else (1 if data else 0)
    b.execute.return_value = MagicMock(data=data, count=count)
    return b


def _seq_side_effect(sequences: list):
    """Return a side_effect that pops from *sequences* on each .table() call.

    Each element of *sequences* is the ``data`` value for that call — either a
    list (normal query) or a dict (maybe_single / single query).
    Calls beyond the sequence length return an empty list.
    """
    idx = [0]

    def _side_effect(name):
        data = sequences[idx[0]] if idx[0] < len(sequences) else []
        idx[0] += 1
        return _builder(data)

    return _side_effect


def _schema_side_effect(user_id: str | None):
    """Stub ``db.schema('auth').from_('users')…`` used by ``_find_user_id_by_email``.

    Returns a side_effect for ``mock_supabase.schema`` whose downstream
    ``.from_('users').select(...).ilike(...).limit(...).execute()`` chain
    yields ``[{'id': user_id}]`` when ``user_id`` is given, or ``[]`` to
    simulate "no account found".
    """
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(
        data=[{"id": user_id}] if user_id else [],
        count=1 if user_id else 0,
    )
    schema_mock = MagicMock()
    schema_mock.from_.return_value = b
    return lambda name: schema_mock


# ===========================================================================
# GET /projects/{project_id}/members
# ===========================================================================


class TestListMembers:
    """GET /projects/{project_id}/members returns all project members."""

    def test_returns_members_list(self, client, mock_supabase):
        """Returns a list of members wrapped in {members: [...]}."""
        mock_supabase.table.side_effect = lambda name: _builder([OWNER_MEMBER, EDITOR_MEMBER])

        response = client.get(f"/projects/{PROJECT_ID}/members")

        assert response.status_code == 200
        data = response.json()
        assert "members" in data
        assert len(data["members"]) == 2

    def test_returns_empty_list_when_no_members(self, client, mock_supabase):
        """Returns empty members list when project has no members."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.get(f"/projects/{PROJECT_ID}/members")

        assert response.status_code == 200
        data = response.json()
        assert data["members"] == []

    def test_member_record_contains_expected_fields(self, client, mock_supabase):
        """Each member record has id, project_id, user_id, and role."""
        mock_supabase.table.side_effect = lambda name: _builder([OWNER_MEMBER])

        response = client.get(f"/projects/{PROJECT_ID}/members")

        member = response.json()["members"][0]
        assert member["id"] == MEMBER_ID
        assert member["project_id"] == PROJECT_ID
        assert member["user_id"] == TEST_USER_ID
        assert member["role"] == "owner"

    def test_returns_multiple_roles(self, client, mock_supabase):
        """Returns members with different roles correctly."""
        viewer = {**EDITOR_MEMBER, "role": "viewer", "id": "memb-003"}
        mock_supabase.table.side_effect = lambda name: _builder([OWNER_MEMBER, EDITOR_MEMBER, viewer])

        response = client.get(f"/projects/{PROJECT_ID}/members")

        assert response.status_code == 200
        assert len(response.json()["members"]) == 3


# ===========================================================================
# POST /projects/{project_id}/members  — add existing user
# ===========================================================================


class TestAddMemberExistingUser:
    """POST /projects/{project_id}/members — target email maps to existing profile."""

    def test_adds_existing_user_directly(self, client, mock_supabase):
        """Returns type=added when auth.users has a matching account."""
        new_member_record = {
            "id": "memb-new-001",
            "project_id": PROJECT_ID,
            "user_id": OTHER_USER_ID,
            "role": "editor",
            "invited_by": TEST_USER_ID,
        }

        # _find_user_id_by_email queries db.schema('auth').from_('users')
        mock_supabase.schema.side_effect = _schema_side_effect(OTHER_USER_ID)
        # db.table() calls in order: get_user_role, duplicate-membership check, insert
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                None,  # duplicate-membership check (maybe_single) — not a member
                [new_member_record],  # project_members INSERT → data[0]
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "existing@example.com", "role": "editor"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "added"
        assert data["member"]["id"] == "memb-new-001"

    def test_add_member_with_viewer_role(self, client, mock_supabase):
        """Returns type=added for viewer role as well."""
        viewer_record = {
            "id": "memb-new-002",
            "project_id": PROJECT_ID,
            "user_id": OTHER_USER_ID,
            "role": "viewer",
            "invited_by": TEST_USER_ID,
        }

        mock_supabase.schema.side_effect = _schema_side_effect(OTHER_USER_ID)
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                None,  # duplicate-membership check (maybe_single) — not a member
                [viewer_record],  # insert → data[0]
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "existing@example.com", "role": "viewer"},
        )

        assert response.status_code == 200
        assert response.json()["type"] == "added"

    def test_returns_403_when_caller_is_not_admin(self, client, mock_supabase):
        """Returns 403 when caller has viewer role (not owner/admin)."""
        # get_user_role returns viewer role dict → service raises PermissionError
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "viewer"},  # get_user_role (maybe_single)
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "someone@example.com", "role": "editor"},
        )

        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    def test_returns_403_when_caller_is_not_a_member(self, client, mock_supabase):
        """Returns 403 when caller has no role on the project."""
        # get_user_role maybe_single returns None/falsy → role is None
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                None,  # get_user_role (maybe_single) → no membership
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "intruder@example.com", "role": "editor"},
        )

        assert response.status_code == 403

    def test_returns_400_for_invalid_role(self, client, mock_supabase):
        """Returns 400 when role is not admin/editor/viewer."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "existing@example.com", "role": "owner"},
        )

        assert response.status_code == 400

    def test_returns_422_for_invalid_email(self, client, mock_supabase):
        """Returns 422 when email field is malformed."""
        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "not-an-email", "role": "editor"},
        )

        assert response.status_code == 422

    def test_returns_422_when_role_missing(self, client, mock_supabase):
        """Returns 422 when required role field is absent."""
        response = client.post(
            f"/projects/{PROJECT_ID}/members",
            json={"email": "valid@example.com"},
        )

        assert response.status_code == 422


# ===========================================================================
# POST /projects/{project_id}/members  — pending invite (new email)
# ===========================================================================


class TestAddMemberPendingInvite:
    """POST /projects/{project_id}/members — target email has no existing account."""

    def test_creates_pending_invite_for_unknown_email(self, client, mock_supabase):
        """Returns type=pending and invite record when email has no auth.users account."""
        # _find_user_id_by_email returns None → skip existing-user branch
        mock_supabase.schema.side_effect = _schema_side_effect(None)
        # db.table() calls in order: get_user_role, pending_project_invites insert
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                [PENDING_INVITE],  # pending_project_invites INSERT → data[0]
            ]
        )

        # Patch the background task email sender so no real email is sent
        with patch("projects.emails.send_project_invite_email"):
            response = client.post(
                f"/projects/{PROJECT_ID}/members",
                json={"email": "newuser@example.com", "role": "viewer"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "pending"
        assert data["invite"]["id"] == INVITE_ID

    def test_pending_invite_stores_lowercase_email(self, client, mock_supabase):
        """The pending invite stores the email in lowercase."""
        invite_lower = {**PENDING_INVITE, "email": "newuser@example.com"}

        mock_supabase.schema.side_effect = _schema_side_effect(None)
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                [invite_lower],  # pending_project_invites INSERT → data[0]
            ]
        )

        with patch("projects.emails.send_project_invite_email"):
            response = client.post(
                f"/projects/{PROJECT_ID}/members",
                json={"email": "NEWUSER@EXAMPLE.COM", "role": "viewer"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "pending"
        # The stored email should be lowercase
        assert data["invite"]["email"] == "newuser@example.com"


# ===========================================================================
# PUT /projects/{project_id}/members/{member_id}
# ===========================================================================


class TestUpdateMemberRole:
    """PUT /projects/{project_id}/members/{member_id} updates a member's role."""

    def test_updates_role_to_editor(self, client, mock_supabase):
        """Returns updated member when caller is owner and role is valid."""
        updated = {**EDITOR_MEMBER, "role": "editor"}

        # update_member_role: get_user_role (maybe_single) → dict, then UPDATE → list
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                [updated],  # project_members UPDATE → data[0]
            ]
        )

        response = client.put(
            f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}",
            json={"role": "editor"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "member" in data
        assert data["member"]["role"] == "editor"

    def test_updates_role_to_admin(self, client, mock_supabase):
        """Returns updated member with admin role."""
        updated = {**EDITOR_MEMBER, "role": "admin"}

        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                [updated],  # project_members UPDATE → data[0]
            ]
        )

        response = client.put(
            f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}",
            json={"role": "admin"},
        )

        assert response.status_code == 200
        assert response.json()["member"]["role"] == "admin"

    def test_returns_403_when_caller_is_viewer(self, client, mock_supabase):
        """Returns 403 when caller has viewer role (cannot change roles)."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "viewer"},  # get_user_role (maybe_single)
            ]
        )

        response = client.put(
            f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}",
            json={"role": "editor"},
        )

        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    def test_returns_400_for_owner_role(self, client, mock_supabase):
        """Returns 400 when attempting to set role to 'owner'."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role — caller is owner but role 'owner' is invalid
            ]
        )

        response = client.put(
            f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}",
            json={"role": "owner"},
        )

        assert response.status_code == 400

    def test_returns_422_when_role_missing(self, client, mock_supabase):
        """Returns 422 when role field is absent from request body."""
        response = client.put(
            f"/projects/{PROJECT_ID}/members/{MEMBER_ID}",
            json={},
        )

        assert response.status_code == 422


# ===========================================================================
# DELETE /projects/{project_id}/members/{member_id}
# ===========================================================================


class TestRemoveMember:
    """DELETE /projects/{project_id}/members/{member_id} removes a project member."""

    def test_admin_removes_other_member(self, client, mock_supabase):
        """Owner/admin can remove another member; returns {deleted: member_id}."""
        # remove_member calls:
        #   1. project_members.select().single() → target.data dict
        #   2. get_user_role → maybe_single dict
        #   3. project_members.delete() — no data read
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                EDITOR_MEMBER,  # single() → dict (target member)
                {"role": "owner"},  # get_user_role (maybe_single) → caller is owner
                [],  # DELETE — result not read
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == EDITOR_MEMBER["id"]

    def test_member_removes_themselves(self, client, mock_supabase):
        """A member can remove themselves (leave the project)."""
        # The caller is TEST_USER_ID; target member is also TEST_USER_ID
        self_member = {**EDITOR_MEMBER, "user_id": TEST_USER_ID, "role": "editor"}

        mock_supabase.table.side_effect = _seq_side_effect(
            [
                self_member,  # single() → dict (target member; is_self=True)
                {"role": "editor"},  # get_user_role (maybe_single) — checked but is_self bypasses admin check
                [],  # DELETE
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/members/{self_member['id']}")

        assert response.status_code == 200
        assert "deleted" in response.json()

    def test_returns_403_when_removing_owner(self, client, mock_supabase):
        """Returns 403 when attempting to remove the project owner."""
        owner_target = {**OWNER_MEMBER, "role": "owner"}

        mock_supabase.table.side_effect = _seq_side_effect(
            [
                owner_target,  # single() → dict (target is owner)
                {"role": "owner"},  # get_user_role (maybe_single) — caller is owner
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/members/{MEMBER_ID}")

        assert response.status_code == 403
        assert "owner" in response.json()["detail"].lower()

    def test_returns_403_when_viewer_removes_others(self, client, mock_supabase):
        """Returns 403 when a viewer tries to remove another member."""
        # Target is a different user so is_self=False; caller has viewer role
        other_member = {**EDITOR_MEMBER, "user_id": "some-other-user-id", "role": "editor"}

        mock_supabase.table.side_effect = _seq_side_effect(
            [
                other_member,  # single() → dict (target member)
                {"role": "viewer"},  # get_user_role (maybe_single) — caller is viewer
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/members/{EDITOR_MEMBER['id']}")

        assert response.status_code == 403

    def test_returns_400_when_member_not_found(self, client, mock_supabase):
        """Returns 400 when target member does not exist on the project."""
        # single() returns no data → target.data is falsy → ValueError raised
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                None,  # single() → no member found
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/members/nonexistent-member")

        assert response.status_code == 400


# ===========================================================================
# GET /projects/{project_id}/pending-invites
# ===========================================================================


class TestListPendingInvites:
    """GET /projects/{project_id}/pending-invites lists pending email invites."""

    def test_returns_invites_list(self, client, mock_supabase):
        """Returns list of pending invites wrapped in {invites: [...]}."""
        mock_supabase.table.side_effect = lambda name: _builder([PENDING_INVITE])

        response = client.get(f"/projects/{PROJECT_ID}/pending-invites")

        assert response.status_code == 200
        data = response.json()
        assert "invites" in data
        assert len(data["invites"]) == 1

    def test_returns_empty_list_when_no_pending_invites(self, client, mock_supabase):
        """Returns empty invites list when no pending invites exist."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.get(f"/projects/{PROJECT_ID}/pending-invites")

        assert response.status_code == 200
        assert response.json()["invites"] == []

    def test_invite_record_contains_expected_fields(self, client, mock_supabase):
        """Each invite record has id, project_id, email, and role."""
        mock_supabase.table.side_effect = lambda name: _builder([PENDING_INVITE])

        response = client.get(f"/projects/{PROJECT_ID}/pending-invites")

        invite = response.json()["invites"][0]
        assert invite["id"] == INVITE_ID
        assert invite["project_id"] == PROJECT_ID
        assert invite["email"] == "newuser@example.com"
        assert invite["role"] == "viewer"

    def test_returns_multiple_pending_invites(self, client, mock_supabase):
        """Returns all pending invites when multiple exist."""
        invite2 = {**PENDING_INVITE, "id": "invt-002", "email": "another@example.com", "role": "editor"}
        mock_supabase.table.side_effect = lambda name: _builder([PENDING_INVITE, invite2])

        response = client.get(f"/projects/{PROJECT_ID}/pending-invites")

        assert response.status_code == 200
        assert len(response.json()["invites"]) == 2


# ===========================================================================
# DELETE /projects/{project_id}/pending-invites/{invite_id}
# ===========================================================================


class TestCancelPendingInvite:
    """DELETE /projects/{project_id}/pending-invites/{invite_id} cancels a pending invite."""

    def test_admin_cancels_invite(self, client, mock_supabase):
        """Owner/admin can cancel a pending invite; returns {deleted: invite_id}."""
        # delete_pending_invite: get_user_role (maybe_single) → dict, then DELETE
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role (maybe_single)
                [],  # DELETE — result not read
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/pending-invites/{INVITE_ID}")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == INVITE_ID

    def test_admin_member_can_also_cancel(self, client, mock_supabase):
        """An admin (not just owner) can cancel a pending invite."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "admin"},  # get_user_role (maybe_single)
                [],  # DELETE
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/pending-invites/{INVITE_ID}")

        assert response.status_code == 200
        assert response.json()["deleted"] == INVITE_ID

    def test_returns_403_when_caller_is_viewer(self, client, mock_supabase):
        """Returns 403 when caller is a viewer (cannot cancel invites)."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "viewer"},  # get_user_role (maybe_single)
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/pending-invites/{INVITE_ID}")

        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()

    def test_returns_403_when_caller_is_editor(self, client, mock_supabase):
        """Returns 403 when caller is an editor (cannot cancel invites)."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "editor"},  # get_user_role (maybe_single)
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/pending-invites/{INVITE_ID}")

        assert response.status_code == 403

    def test_returns_403_when_caller_not_a_member(self, client, mock_supabase):
        """Returns 403 when caller has no membership on the project."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                None,  # get_user_role (maybe_single) → no membership → role is None
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/pending-invites/{INVITE_ID}")

        assert response.status_code == 403
