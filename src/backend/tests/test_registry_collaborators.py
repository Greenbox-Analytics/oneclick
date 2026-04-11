"""Tests for Registry Collaboration & Invitation endpoints.

Acceptance criteria:
1. List collaborators for a work (creator and collaborator access)
2. Invite collaborator (email sending mocked)
3. Accept from dashboard / decline
4. Revoke collaborator
5. List my invites
"""

from datetime import UTC
from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder

WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
COLLAB_ID = "bbbbbbbb-0000-0000-0000-000000000002"
INVITE_TOKEN = "test-invite-token-abc123"

# ---------------------------------------------------------------------------
# Helper: build a minimal collaborator row
# ---------------------------------------------------------------------------


def _collab_row(**overrides):
    base = {
        "id": COLLAB_ID,
        "work_id": WORK_ID,
        "invited_by": TEST_USER_ID,
        "email": "collab@example.com",
        "name": "Alice Collab",
        "role": "producer",
        "status": "invited",
        "invite_token": INVITE_TOKEN,
        "expires_at": None,
        "collaborator_user_id": None,
        "stake_id": None,
        "responded_at": None,
        "invited_at": "2026-04-10T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ============================================================
# 1. List collaborators
# ============================================================


class TestListCollaborators:
    """GET /registry/collaborators?work_id=..."""

    def test_creator_can_list_collaborators(self, client, mock_supabase):
        """Work creator gets collaborators list."""
        work_builder = MockQueryBuilder()
        work_builder.execute.return_value = MagicMock(data={"user_id": TEST_USER_ID}, count=1)

        collab_check_builder = MockQueryBuilder()
        collab_check_builder.execute.return_value = MagicMock(data=[], count=0)

        collaborators_builder = MockQueryBuilder()
        collaborators_builder.execute.return_value = MagicMock(data=[_collab_row()], count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                # works_registry ownership check
                return work_builder
            elif n == 2:
                # registry_collaborators existence check for current user
                return collab_check_builder
            else:
                # get_collaborators query
                return collaborators_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/collaborators?work_id={WORK_ID}")
        assert response.status_code == 200
        body = response.json()
        assert "collaborators" in body
        assert len(body["collaborators"]) == 1
        assert body["collaborators"][0]["id"] == COLLAB_ID

    def test_non_creator_non_collab_gets_403(self, client, mock_supabase):
        """Someone unrelated to the work gets 403."""
        work_builder = MockQueryBuilder()
        # Different owner
        work_builder.execute.return_value = MagicMock(data={"user_id": "other-user-id"}, count=1)

        collab_check_builder = MockQueryBuilder()
        collab_check_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return work_builder
            else:
                return collab_check_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/collaborators?work_id={WORK_ID}")
        assert response.status_code == 403

    def test_work_not_found_returns_404(self, client, mock_supabase):
        """Missing work returns 404."""
        work_builder = MockQueryBuilder()
        work_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: work_builder

        response = client.get(f"/registry/collaborators?work_id={WORK_ID}")
        assert response.status_code == 404

    def test_existing_collaborator_can_list(self, client, mock_supabase):
        """An existing collaborator on the work can also view the list."""
        work_builder = MockQueryBuilder()
        # Different owner
        work_builder.execute.return_value = MagicMock(data={"user_id": "other-owner-id"}, count=1)

        collab_check_builder = MockQueryBuilder()
        # Current user IS a collaborator
        collab_check_builder.execute.return_value = MagicMock(data=[{"id": COLLAB_ID}], count=1)

        collaborators_builder = MockQueryBuilder()
        collaborators_builder.execute.return_value = MagicMock(data=[_collab_row()], count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return work_builder
            elif n == 2:
                return collab_check_builder
            else:
                return collaborators_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/collaborators?work_id={WORK_ID}")
        assert response.status_code == 200
        assert "collaborators" in response.json()


# ============================================================
# 2. Invite collaborator
# ============================================================


class TestInviteCollaborator:
    """POST /registry/collaborators/invite"""

    def test_invite_sends_email_and_returns_collab(self, client, mock_supabase):
        """Successful invite creates the record and sends an email."""
        # works_registry lookup for work title
        work_title_builder = MockQueryBuilder()
        work_title_builder.execute.return_value = MagicMock(data={"title": "My Test Track"}, count=1)

        # rpc call to check_user_exists
        mock_supabase.rpc.return_value = MockQueryBuilder()
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=None)

        # registry_collaborators insert
        insert_builder = MockQueryBuilder()
        insert_builder.execute.return_value = MagicMock(data=[_collab_row()], count=1)

        # profiles lookup for inviter name
        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"full_name": "Test User"}, count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                # router: works_registry title lookup
                return work_title_builder
            elif n == 2:
                # service: registry_collaborators insert
                return insert_builder
            else:
                # router: profiles inviter name
                return profile_builder

        mock_supabase.table.side_effect = table_side_effect

        with patch("registry.emails.send_invitation_email") as mock_email:
            response = client.post(
                "/registry/collaborators/invite",
                json={
                    "work_id": WORK_ID,
                    "email": "collab@example.com",
                    "name": "Alice Collab",
                    "role": "producer",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == COLLAB_ID
        assert body["email"] == "collab@example.com"
        mock_email.assert_called_once()

    def test_invite_failure_returns_500(self, client, mock_supabase):
        """If insert returns no data, endpoint returns 500."""
        work_title_builder = MockQueryBuilder()
        work_title_builder.execute.return_value = MagicMock(data={"title": "Test Track"}, count=1)

        mock_supabase.rpc.return_value = MockQueryBuilder()
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=None)

        insert_builder = MockQueryBuilder()
        insert_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return work_title_builder
            else:
                return insert_builder

        mock_supabase.table.side_effect = table_side_effect

        with patch("registry.emails.send_invitation_email"):
            response = client.post(
                "/registry/collaborators/invite",
                json={
                    "work_id": WORK_ID,
                    "email": "collab@example.com",
                    "name": "Alice Collab",
                    "role": "producer",
                },
            )

        assert response.status_code == 500


# ============================================================
# 3. Claim invitation
# ============================================================


class TestClaimInvitation:
    """POST /registry/collaborators/claim?invite_token=..."""

    def test_claim_valid_token(self, client, mock_supabase):
        """Valid non-expired token returns the updated collaborator."""
        # The claim lookup
        claim_builder = MockQueryBuilder()
        claim_builder.execute.return_value = MagicMock(
            data=_collab_row(collaborator_user_id=None, expires_at=None),
            count=1,
        )

        # Update collaborator_user_id
        update_builder = MockQueryBuilder()
        update_builder.execute.return_value = MagicMock(data=[_collab_row(collaborator_user_id=TEST_USER_ID)], count=1)

        # auto_verify_artist: artists lookup
        artists_builder = MockQueryBuilder()
        artists_builder.execute.return_value = MagicMock(data=[], count=0)

        # notification: works_registry creator
        works_builder = MockQueryBuilder()
        works_builder.execute.return_value = MagicMock(data={"user_id": "owner-id", "title": "My Track"}, count=1)

        # notification insert
        notif_builder = MockQueryBuilder()
        notif_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return claim_builder
            elif n == 2:
                return update_builder
            elif n == 3:
                return artists_builder
            elif n == 4:
                return works_builder
            else:
                return notif_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/claim?invite_token={INVITE_TOKEN}")
        assert response.status_code == 200

    def test_claim_expired_token_returns_410(self, client, mock_supabase):
        """Expired invitation token returns 410."""
        from datetime import datetime, timedelta

        expired_time = (datetime.now(UTC) - timedelta(hours=49)).isoformat()

        claim_builder = MockQueryBuilder()
        claim_builder.execute.return_value = MagicMock(
            data=_collab_row(expires_at=expired_time, collaborator_user_id=None),
            count=1,
        )

        mock_supabase.table.side_effect = lambda name: claim_builder

        response = client.post(f"/registry/collaborators/claim?invite_token={INVITE_TOKEN}")
        assert response.status_code == 410

    def test_claim_not_found_token_returns_404(self, client, mock_supabase):
        """Unknown token returns 404."""
        not_found_builder = MockQueryBuilder()
        not_found_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: not_found_builder

        response = client.post("/registry/collaborators/claim?invite_token=nonexistent-token")
        assert response.status_code == 404


# ============================================================
# 4a. Accept from dashboard
# ============================================================


class TestAcceptFromDashboard:
    """POST /registry/collaborators/{collaborator_id}/accept-from-dashboard"""

    def test_accept_matching_email_returns_accepted(self, client, mock_supabase):
        """Collaborator whose email matches can accept from dashboard."""
        # collab row lookup
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(email="collab@example.com"), count=1)

        # profiles email lookup
        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "collab@example.com"}, count=1)

        # update collaborator status
        update_builder = MockQueryBuilder()
        update_builder.execute.return_value = MagicMock(data=[], count=0)

        # _check_auto_register: registry_collaborators statuses
        check_reg_builder = MockQueryBuilder()
        check_reg_builder.execute.return_value = MagicMock(data=[{"status": "confirmed"}], count=1)

        # _check_auto_register: works_registry update
        works_reg_update_builder = MockQueryBuilder()
        works_reg_update_builder.execute.return_value = MagicMock(data=[], count=0)

        # notification: works_registry creator
        works_notif_builder = MockQueryBuilder()
        works_notif_builder.execute.return_value = MagicMock(
            data={"user_id": "owner-id", "title": "Test Work"}, count=1
        )

        # notification insert
        notif_builder = MockQueryBuilder()
        notif_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            elif n == 2:
                return profile_builder
            elif n == 3:
                return update_builder
            elif n == 4:
                return check_reg_builder
            elif n == 5:
                return works_reg_update_builder
            elif n == 6:
                return works_notif_builder
            else:
                return notif_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/accept-from-dashboard")
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] == COLLAB_ID

    def test_accept_mismatched_email_returns_403(self, client, mock_supabase):
        """User with different email gets 403."""
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(email="collab@example.com"), count=1)

        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "different@example.com"}, count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            else:
                return profile_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/accept-from-dashboard")
        assert response.status_code == 403

    def test_accept_not_found_collab_returns_400(self, client, mock_supabase):
        """Missing collaborator record raises ValueError -> 400."""
        not_found_builder = MockQueryBuilder()
        not_found_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: not_found_builder

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/accept-from-dashboard")
        assert response.status_code == 400


# ============================================================
# 4b. Decline invitation
# ============================================================


class TestDeclineInvitation:
    """POST /registry/collaborators/{collaborator_id}/decline"""

    def test_decline_matching_email_succeeds(self, client, mock_supabase):
        """User whose email matches can decline an invitation."""
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(email="collab@example.com"), count=1)

        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "collab@example.com"}, count=1)

        update_builder = MockQueryBuilder()
        update_builder.execute.return_value = MagicMock(data=[], count=0)

        # works_registry for notification
        works_builder = MockQueryBuilder()
        works_builder.execute.return_value = MagicMock(data={"user_id": "owner-id", "title": "My Track"}, count=1)

        # notification insert
        notif_builder = MockQueryBuilder()
        notif_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            elif n == 2:
                return profile_builder
            elif n == 3:
                return update_builder
            elif n == 4:
                return works_builder
            else:
                return notif_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/decline")
        assert response.status_code == 200
        body = response.json()
        assert body["declined"] == COLLAB_ID

    def test_decline_mismatched_email_returns_403(self, client, mock_supabase):
        """User with non-matching email gets 403."""
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(email="collab@example.com"), count=1)

        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "wrong@example.com"}, count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            else:
                return profile_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/decline")
        assert response.status_code == 403

    def test_decline_not_found_collab_returns_400(self, client, mock_supabase):
        """Missing collaborator raises ValueError -> 400."""
        not_found_builder = MockQueryBuilder()
        not_found_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: not_found_builder

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/decline")
        assert response.status_code == 400


# ============================================================
# 5. Revoke collaborator
# ============================================================


class TestRevokeCollaborator:
    """POST /registry/collaborators/{collaborator_id}/revoke"""

    def test_revoke_by_inviter_succeeds(self, client, mock_supabase):
        """Work inviter can revoke a collaborator."""
        # collab lookup (invited_by matches user_id)
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(status="invited", stake_id=None), count=1)

        # update status to revoked
        revoke_update_builder = MockQueryBuilder()
        revoke_update_builder.execute.return_value = MagicMock(data=[], count=0)

        # works_registry status check (no registered revert needed)
        works_status_builder = MockQueryBuilder()
        works_status_builder.execute.return_value = MagicMock(data={"status": "draft"}, count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            elif n == 2:
                return revoke_update_builder
            else:
                return works_status_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/revoke")
        assert response.status_code == 200
        body = response.json()
        # revoke_collaborator returns collab.data which is our _collab_row dict
        assert body["id"] == COLLAB_ID

    def test_revoke_also_reverts_registered_work_to_draft(self, client, mock_supabase):
        """If work was registered, revoking a collab reverts it to draft."""
        collab_builder = MockQueryBuilder()
        collab_builder.execute.return_value = MagicMock(data=_collab_row(status="confirmed", stake_id=None), count=1)

        revoke_update_builder = MockQueryBuilder()
        revoke_update_builder.execute.return_value = MagicMock(data=[], count=0)

        # work is registered
        works_status_builder = MockQueryBuilder()
        works_status_builder.execute.return_value = MagicMock(data={"status": "registered"}, count=1)

        # works_registry update to draft
        works_revert_builder = MockQueryBuilder()
        works_revert_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_builder
            elif n == 2:
                return revoke_update_builder
            elif n == 3:
                return works_status_builder
            else:
                return works_revert_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/revoke")
        assert response.status_code == 200

    def test_revoke_not_found_returns_404(self, client, mock_supabase):
        """Non-inviter or missing record returns 404."""
        not_found_builder = MockQueryBuilder()
        not_found_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: not_found_builder

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/revoke")
        assert response.status_code == 404


# ============================================================
# 6. List my invites
# ============================================================


class TestGetMyInvites:
    """GET /registry/collaborators/my-invites"""

    def test_returns_pending_invites_for_user(self, client, mock_supabase):
        """Returns invites where the user's email matches and status is 'invited'."""
        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "collab@example.com"}, count=1)

        invites_builder = MockQueryBuilder()
        invites_builder.execute.return_value = MagicMock(data=[_collab_row()], count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return profile_builder
            else:
                return invites_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get("/registry/collaborators/my-invites")
        assert response.status_code == 200
        body = response.json()
        assert "invites" in body
        assert len(body["invites"]) == 1
        assert body["invites"][0]["id"] == COLLAB_ID

    def test_returns_empty_list_when_no_profile_email(self, client, mock_supabase):
        """If profile has no email, returns empty invites list."""
        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: profile_builder

        response = client.get("/registry/collaborators/my-invites")
        assert response.status_code == 200
        body = response.json()
        assert body["invites"] == []

    def test_returns_empty_list_when_no_pending_invites(self, client, mock_supabase):
        """Returns empty list when user has no pending invites."""
        profile_builder = MockQueryBuilder()
        profile_builder.execute.return_value = MagicMock(data={"email": "collab@example.com"}, count=1)

        invites_builder = MockQueryBuilder()
        invites_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return profile_builder
            else:
                return invites_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get("/registry/collaborators/my-invites")
        assert response.status_code == 200
        body = response.json()
        assert body["invites"] == []


# ============================================================
# Extra: Confirm stake
# ============================================================


class TestConfirmStake:
    """POST /registry/collaborators/{collaborator_id}/confirm"""

    def test_confirm_stake_succeeds(self, client, mock_supabase):
        """Collaborator can confirm their stake."""
        # collab ownership check
        collab_check_builder = MockQueryBuilder()
        collab_check_builder.execute.return_value = MagicMock(data={"work_id": WORK_ID, "status": "invited"}, count=1)

        # work status check
        work_status_builder = MockQueryBuilder()
        work_status_builder.execute.return_value = MagicMock(data={"status": "pending_approval"}, count=1)

        # update to confirmed
        confirmed_row = _collab_row(status="confirmed")
        update_builder = MockQueryBuilder()
        update_builder.execute.return_value = MagicMock(data=[confirmed_row], count=1)

        # _check_auto_register: all collaborators
        all_collabs_builder = MockQueryBuilder()
        all_collabs_builder.execute.return_value = MagicMock(data=[{"status": "confirmed"}], count=1)

        # _check_auto_register: works update to registered
        works_reg_builder = MockQueryBuilder()
        works_reg_builder.execute.return_value = MagicMock(data=[], count=0)

        # notification: work creator lookup
        work_notif_builder = MockQueryBuilder()
        work_notif_builder.execute.return_value = MagicMock(data={"user_id": "owner-id", "title": "Test Work"}, count=1)

        # notification insert
        notif_builder = MockQueryBuilder()
        notif_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return collab_check_builder
            elif n == 2:
                return work_status_builder
            elif n == 3:
                return update_builder
            elif n == 4:
                return all_collabs_builder
            elif n == 5:
                return works_reg_builder
            elif n == 6:
                return work_notif_builder
            else:
                return notif_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/confirm")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "confirmed"

    def test_confirm_stake_not_found_returns_404(self, client, mock_supabase):
        """Confirm with bad ID returns 404."""
        not_found_builder = MockQueryBuilder()
        not_found_builder.execute.return_value = MagicMock(data=None, count=0)

        mock_supabase.table.side_effect = lambda name: not_found_builder

        response = client.post(f"/registry/collaborators/{COLLAB_ID}/confirm")
        assert response.status_code == 404
