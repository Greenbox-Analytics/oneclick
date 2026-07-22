from datetime import datetime
from unittest.mock import ANY, MagicMock, patch

import pytest

from users.account_deletion_service import (
    LastAdminError,
    _archive_sole_admin_orgs,
    _reclaim_own_seats,
    _reclaim_seat_to_pool,
    _teardown_archived_org_grants,
    cancel_user_stripe,
    delete_user_account,
    list_user_storage_paths,
    would_be_last_admin,
)


def test_list_user_storage_paths_empty_user():
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    paths = list_user_storage_paths(sb, "user-1")
    assert paths == []


def test_list_user_storage_paths_collects_project_and_audio_files():
    """project_files attach via project_id; audio_files attach via folder_id
    (audio_folders.artist_id). Schema mismatch was the source of the
    `column audio_files.project_id does not exist` 502 in prod.
    """
    sb = MagicMock()
    tables = {
        "artists": MagicMock(),
        "projects": MagicMock(),
        "project_files": MagicMock(),
        "audio_folders": MagicMock(),
        "audio_files": MagicMock(),
    }
    tables["artists"].select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
    tables["projects"].select.return_value.in_.return_value.execute.return_value.data = [{"id": "p1"}]
    tables["project_files"].select.return_value.in_.return_value.execute.return_value.data = [
        {"file_path": "a1/p1/contracts/foo.pdf"},
        {"file_path": "a1/p1/notes/bar.md"},
    ]
    tables["audio_folders"].select.return_value.in_.return_value.execute.return_value.data = [{"id": "f1"}]
    tables["audio_files"].select.return_value.in_.return_value.execute.return_value.data = [
        {"file_path": "a1/f1/song.wav"},
    ]
    sb.table.side_effect = lambda name: tables[name]

    paths = list_user_storage_paths(sb, "user-1")

    assert ("project-files", "a1/p1/contracts/foo.pdf") in paths
    assert ("project-files", "a1/p1/notes/bar.md") in paths
    assert ("audio-files", "a1/f1/song.wav") in paths
    assert len(paths) == 3

    # Schema assertions: audio path must go via audio_folders.artist_id → audio_files.folder_id.
    tables["audio_folders"].select.return_value.in_.assert_called_with("artist_id", ["a1"])
    tables["audio_files"].select.return_value.in_.assert_called_with("folder_id", ["f1"])


def test_list_user_storage_paths_user_with_no_projects_still_walks_audio():
    """Audio is artist-scoped, not project-scoped — absence of projects must
    not short-circuit the audio walk.
    """
    sb = MagicMock()
    tables = {
        "artists": MagicMock(),
        "projects": MagicMock(),
        "audio_folders": MagicMock(),
        "audio_files": MagicMock(),
    }
    tables["artists"].select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
    tables["projects"].select.return_value.in_.return_value.execute.return_value.data = []
    tables["audio_folders"].select.return_value.in_.return_value.execute.return_value.data = [{"id": "f1"}]
    tables["audio_files"].select.return_value.in_.return_value.execute.return_value.data = [
        {"file_path": "a1/f1/song.wav"},
    ]
    sb.table.side_effect = lambda name: tables[name]

    paths = list_user_storage_paths(sb, "user-1")
    assert paths == [("audio-files", "a1/f1/song.wav")]


def test_list_user_storage_paths_user_with_no_audio_folders():
    """Project files exist but artist has no audio folders → audio_files is not queried."""
    sb = MagicMock()
    tables = {
        "artists": MagicMock(),
        "projects": MagicMock(),
        "project_files": MagicMock(),
        "audio_folders": MagicMock(),
        "audio_files": MagicMock(),
    }
    tables["artists"].select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
    tables["projects"].select.return_value.in_.return_value.execute.return_value.data = [{"id": "p1"}]
    tables["project_files"].select.return_value.in_.return_value.execute.return_value.data = [
        {"file_path": "a1/p1/foo.pdf"},
    ]
    tables["audio_folders"].select.return_value.in_.return_value.execute.return_value.data = []
    sb.table.side_effect = lambda name: tables[name]

    paths = list_user_storage_paths(sb, "user-1")
    assert paths == [("project-files", "a1/p1/foo.pdf")]
    tables["audio_files"].select.assert_not_called()


def test_list_user_storage_paths_user_with_no_projects_or_audio():
    sb = MagicMock()
    tables = {
        "artists": MagicMock(),
        "projects": MagicMock(),
        "audio_folders": MagicMock(),
    }
    tables["artists"].select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
    tables["projects"].select.return_value.in_.return_value.execute.return_value.data = []
    tables["audio_folders"].select.return_value.in_.return_value.execute.return_value.data = []
    sb.table.side_effect = lambda name: tables[name]

    paths = list_user_storage_paths(sb, "user-1")
    assert paths == []


def _sub_row(sb, row):
    sb.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = row


def test_cancel_user_stripe_no_subscription_row():
    sb = MagicMock()
    _sub_row(sb, None)
    cancel_user_stripe(sb, "user-1")  # should not raise


def test_cancel_user_stripe_all_ids_null():
    sb = MagicMock()
    _sub_row(sb, {"stripe_subscription_id": None, "stripe_customer_id": None})
    cancel_user_stripe(sb, "user-1")  # no-op


def test_cancel_user_stripe_calls_subscription_and_customer_delete():
    sb = MagicMock()
    _sub_row(sb, {"stripe_subscription_id": "sub_abc", "stripe_customer_id": "cus_xyz"})
    fake_stripe = MagicMock()
    with patch("users.account_deletion_service.get_stripe", return_value=fake_stripe):
        cancel_user_stripe(sb, "user-1")
    fake_stripe.Subscription.delete.assert_called_once_with("sub_abc")
    fake_stripe.Customer.delete.assert_called_once_with("cus_xyz")


def test_cancel_user_stripe_customer_only():
    """Free-tier user with a customer record but no subscription."""
    sb = MagicMock()
    _sub_row(sb, {"stripe_subscription_id": None, "stripe_customer_id": "cus_xyz"})
    fake_stripe = MagicMock()
    with patch("users.account_deletion_service.get_stripe", return_value=fake_stripe):
        cancel_user_stripe(sb, "user-1")
    fake_stripe.Subscription.delete.assert_not_called()
    fake_stripe.Customer.delete.assert_called_once_with("cus_xyz")


def test_cancel_user_stripe_already_canceled_is_ok():
    sb = MagicMock()
    _sub_row(sb, {"stripe_subscription_id": "sub_abc", "stripe_customer_id": "cus_xyz"})
    fake_stripe = MagicMock()

    class _StripeErr(Exception):
        pass

    fake_stripe.InvalidRequestError = _StripeErr
    fake_stripe.Subscription.delete.side_effect = _StripeErr("No such subscription: sub_abc")
    with patch("users.account_deletion_service.get_stripe", return_value=fake_stripe):
        cancel_user_stripe(sb, "user-1")
    # Customer.delete still runs and succeeds
    fake_stripe.Customer.delete.assert_called_once_with("cus_xyz")


def test_cancel_user_stripe_other_error_reraises():
    sb = MagicMock()
    _sub_row(sb, {"stripe_subscription_id": "sub_abc", "stripe_customer_id": "cus_xyz"})
    fake_stripe = MagicMock()

    class _StripeErr(Exception):
        pass

    fake_stripe.InvalidRequestError = _StripeErr
    fake_stripe.Subscription.delete.side_effect = RuntimeError("API down")
    with (
        patch("users.account_deletion_service.get_stripe", return_value=fake_stripe),
        pytest.raises(RuntimeError),
    ):
        cancel_user_stripe(sb, "user-1")
    fake_stripe.Customer.delete.assert_not_called()


def test_would_be_last_admin_non_admin_returns_false():
    sb = MagicMock()
    with patch("users.account_deletion_service.is_user_admin", return_value=False):
        assert would_be_last_admin(sb, "user-1", "u1@test.com") is False


def test_would_be_last_admin_only_env_admin_returns_true():
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    with (
        patch("users.account_deletion_service.is_user_admin", return_value=True),
        patch("users.account_deletion_service.env_admin_emails", return_value={"u1@test.com"}),
    ):
        assert would_be_last_admin(sb, "user-1", "u1@test.com") is True


def test_would_be_last_admin_db_admin_with_other_admins_returns_false():
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {"id": "user-1"},
        {"id": "user-2"},
    ]
    with (
        patch("users.account_deletion_service.is_user_admin", return_value=True),
        patch("users.account_deletion_service.env_admin_emails", return_value=set()),
    ):
        assert would_be_last_admin(sb, "user-1", "u1@test.com") is False


def test_would_be_last_admin_env_plus_other_admins_returns_false():
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    with (
        patch("users.account_deletion_service.is_user_admin", return_value=True),
        patch("users.account_deletion_service.env_admin_emails", return_value={"u1@test.com", "other@test.com"}),
    ):
        assert would_be_last_admin(sb, "user-1", "u1@test.com") is False


def test_delete_user_account_blocks_last_admin():
    sb = MagicMock()
    with (
        patch("users.account_deletion_service.would_be_last_admin", return_value=True),
        patch("users.account_deletion_service.analytics_capture") as analytics,
        pytest.raises(LastAdminError),
    ):
        delete_user_account(sb, "user-1", "u1@test.com")
    sb.auth.admin.delete_user.assert_not_called()
    event_names = [c.args[1] for c in analytics.call_args_list]
    assert event_names == ["account_delete_blocked"]
    assert analytics.call_args.args[2]["reason"] == "last_admin"


def test_delete_user_account_happy_path():
    sb = MagicMock()
    with (
        patch("users.account_deletion_service.would_be_last_admin", return_value=False),
        patch("users.account_deletion_service.cancel_user_stripe") as cancel,
        patch(
            "users.account_deletion_service.list_user_storage_paths",
            return_value=[
                ("project-files", "a/p/x.pdf"),
                ("audio-files", "a/p/song.wav"),
            ],
        ),
        patch("users.account_deletion_service.analytics_capture") as analytics,
    ):
        delete_user_account(sb, "user-1", "u1@test.com")

    cancel.assert_called_once_with(sb, "user-1")
    sb.storage.from_.assert_any_call("project-files")
    sb.storage.from_.assert_any_call("audio-files")
    sb.auth.admin.delete_user.assert_called_once_with("user-1")
    event_names = [c.args[1] for c in analytics.call_args_list]
    assert event_names == ["account_delete_started", "account_deleted"]


def test_delete_user_account_stripe_failure_aborts():
    sb = MagicMock()
    with (
        patch("users.account_deletion_service.would_be_last_admin", return_value=False),
        patch("users.account_deletion_service.cancel_user_stripe", side_effect=RuntimeError("stripe down")),
        patch("users.account_deletion_service.analytics_capture") as analytics,
        pytest.raises(RuntimeError),
    ):
        delete_user_account(sb, "user-1", "u1@test.com")
    sb.auth.admin.delete_user.assert_not_called()
    event_names = [c.args[1] for c in analytics.call_args_list]
    assert "account_delete_started" in event_names
    assert "account_deleted" not in event_names


def test_delete_user_account_storage_failure_aborts():
    sb = MagicMock()
    sb.storage.from_.return_value.remove.side_effect = RuntimeError("S3 down")
    with (
        patch("users.account_deletion_service.would_be_last_admin", return_value=False),
        patch("users.account_deletion_service.cancel_user_stripe"),
        patch(
            "users.account_deletion_service.list_user_storage_paths",
            return_value=[
                ("project-files", "a/p/x.pdf"),
            ],
        ),
        patch("users.account_deletion_service.analytics_capture"),
        pytest.raises(RuntimeError),
    ):
        delete_user_account(sb, "user-1", "u1@test.com")
    sb.auth.admin.delete_user.assert_not_called()


def test_delete_user_account_no_storage_objects():
    sb = MagicMock()
    with (
        patch("users.account_deletion_service.would_be_last_admin", return_value=False),
        patch("users.account_deletion_service.cancel_user_stripe"),
        patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
        patch("users.account_deletion_service.analytics_capture"),
    ):
        delete_user_account(sb, "user-1", "u1@test.com")
    sb.storage.from_.assert_not_called()
    sb.auth.admin.delete_user.assert_called_once_with("user-1")


# ---------------------------------------------------------------------------
# Licensing Phase B, Task 10 — org seat reclaim / sole-admin org teardown
# BEFORE account deletion. Reuses orgs.service._offboard's request_id
# grammar (offboard:{member_id}:{epoch(revoked_at)}, orgteardown:
# {other_member_id}:{epoch(archived_at)}) but reimplemented locally since
# _offboard is gated on authz.require_admin — inapplicable to a
# system-initiated cleanup on the deleting user's own seats / a sole
# admin's org.
# ---------------------------------------------------------------------------


def _seat_wallet_row(sb, data):
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = data


class TestReclaimSeatToPool:
    def test_missing_wallet_is_noop(self):
        sb = MagicMock()
        _seat_wallet_row(sb, [])
        _reclaim_seat_to_pool(sb, member_id="m1", org_id="org1", request_id="offboard:m1:100", reason="x")
        sb.rpc.assert_not_called()

    def test_zero_balance_is_noop(self):
        sb = MagicMock()
        _seat_wallet_row(sb, [{"id": "seat-1", "bundle_balance": 0, "reserve_balance": 0}])
        _reclaim_seat_to_pool(sb, member_id="m1", org_id="org1", request_id="offboard:m1:100", reason="x")
        sb.rpc.assert_not_called()

    def test_negative_net_balance_is_noop(self):
        """bundle can go negative from accepted debit drift; net <= 0 -> no RPC."""
        sb = MagicMock()
        _seat_wallet_row(sb, [{"id": "seat-1", "bundle_balance": -10, "reserve_balance": 5}])
        _reclaim_seat_to_pool(sb, member_id="m1", org_id="org1", request_id="offboard:m1:100", reason="x")
        sb.rpc.assert_not_called()

    def test_transfers_net_balance_to_pool(self):
        sb = MagicMock()
        _seat_wallet_row(sb, [{"id": "seat-1", "bundle_balance": 0, "reserve_balance": 40}])
        with patch(
            "users.account_deletion_service.org_wallets.read_or_create_org_wallet",
            return_value={"id": "pool-1"},
        ) as mock_pool:
            _reclaim_seat_to_pool(
                sb, member_id="m1", org_id="org1", request_id="offboard:m1:100", reason="account_deleted"
            )

        mock_pool.assert_called_once_with(sb, "org1")
        sb.rpc.assert_called_once_with(
            "transfer_credits",
            {
                "p_from_wallet": "seat-1",
                "p_to_wallet": "pool-1",
                "p_amount": 40,
                "p_kind": "reclaim",
                "p_request_id": "offboard:m1:100",
                "p_metadata": {"org_id": "org1", "reason": "account_deleted"},
            },
        )

    def test_rpc_failure_is_logged_never_raised(self):
        sb = MagicMock()
        _seat_wallet_row(sb, [{"id": "seat-1", "bundle_balance": 0, "reserve_balance": 40}])
        sb.rpc.return_value.execute.side_effect = RuntimeError("db down")
        with patch(
            "users.account_deletion_service.org_wallets.read_or_create_org_wallet",
            return_value={"id": "pool-1"},
        ):
            _reclaim_seat_to_pool(sb, member_id="m1", org_id="org1", request_id="offboard:m1:100", reason="x")
        # No exception propagated == pass.


class TestReclaimOwnSeats:
    """_reclaim_own_seats now takes pre-fetched `own_rows` + the
    `archived_org_ids` set returned by `_archive_sole_admin_orgs` (which the
    caller — `delete_user_account` — runs FIRST; see finding 1 in the Phase
    B review). It no longer queries org_members itself."""

    def test_transitions_status_and_reclaims_with_offboard_key(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        own_rows = [
            {"id": "m1", "org_id": "org1", "user_id": "u1", "status": "active", "role": "member", "revoked_at": None}
        ]
        om.update.return_value.eq.return_value.execute.return_value.data = [{"id": "m1"}]
        om.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "revoked_at": "2026-07-20T00:00:00+00:00"
        }

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", own_rows, set())

        om.update.assert_called_once_with({"status": "removed", "revoked_at": ANY})
        mock_reclaim.assert_called_once()
        kwargs = mock_reclaim.call_args.kwargs
        assert kwargs["member_id"] == "m1"
        assert kwargs["org_id"] == "org1"
        assert kwargs["reason"] == "account_deleted"
        expected_epoch = int(datetime.fromisoformat("2026-07-20T00:00:00+00:00").timestamp())
        assert kwargs["request_id"] == f"offboard:m1:{expected_epoch}"

    def test_retry_reuses_stored_revoked_at_without_restamping(self):
        """A row already 'removed' with revoked_at set is a retry of a prior
        attempt whose reclaim failed after the status write landed — reuse
        the stored timestamp (no new UPDATE), so the derived request_id is
        IDENTICAL across retries and transfer_credits' own idempotency
        converges rather than minting a fresh key."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        own_rows = [
            {
                "id": "m1",
                "org_id": "org1",
                "user_id": "u1",
                "status": "removed",
                "role": "member",
                "revoked_at": "2026-07-01T00:00:00+00:00",
            }
        ]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", own_rows, set())

        om.update.assert_not_called()
        expected_epoch = int(datetime.fromisoformat("2026-07-01T00:00:00+00:00").timestamp())
        assert mock_reclaim.call_args.kwargs["request_id"] == f"offboard:m1:{expected_epoch}"

    def test_status_update_failure_is_logged_and_skipped(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        own_rows = [
            {"id": "m1", "org_id": "org1", "user_id": "u1", "status": "active", "role": "member", "revoked_at": None}
        ]
        om.update.return_value.eq.return_value.execute.side_effect = RuntimeError("db down")

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", own_rows, set())  # must not raise

        mock_reclaim.assert_not_called()

    def test_no_org_membership_is_a_noop(self):
        sb = MagicMock()
        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", [], set())
        mock_reclaim.assert_not_called()

    def test_skips_membership_whose_org_is_already_archived(self):
        """Core of Phase B review finding 1: a membership whose org_id is in
        the archived set must be skipped ENTIRELY — no status-flip UPDATE
        (which would trip org_members_admin_guard for a sole active admin,
        per supabase/migrations/20260721000001_licensing_core.sql) and no
        reclaim call (the seat was already reclaimed by
        `_archive_sole_admin_orgs`)."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        own_rows = [
            {"id": "m1", "org_id": "org1", "user_id": "u1", "status": "active", "role": "admin", "revoked_at": None}
        ]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", own_rows, {"org1"})

        om.update.assert_not_called()
        mock_reclaim.assert_not_called()

    def test_archived_org_skipped_while_other_org_still_processed(self):
        """Mixed case: one membership belongs to an already-archived
        sole-admin org (skip, no status flip) and the other is a normal
        non-sole-admin membership (soft-remove + offboard-key reclaim,
        unchanged behavior)."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        own_rows = [
            {
                "id": "m_admin",
                "org_id": "org1",
                "user_id": "u1",
                "status": "active",
                "role": "admin",
                "revoked_at": None,
            },
            {
                "id": "m_member",
                "org_id": "org2",
                "user_id": "u1",
                "status": "active",
                "role": "member",
                "revoked_at": None,
            },
        ]
        om.update.return_value.eq.return_value.execute.return_value.data = [{"id": "m_member"}]
        om.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "revoked_at": "2026-07-20T00:00:00+00:00"
        }

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            _reclaim_own_seats(sb, "u1", own_rows, {"org1"})

        om.update.assert_called_once_with({"status": "removed", "revoked_at": ANY})
        mock_reclaim.assert_called_once()
        assert mock_reclaim.call_args.kwargs["member_id"] == "m_member"
        assert mock_reclaim.call_args.kwargs["org_id"] == "org2"
        expected_epoch = int(datetime.fromisoformat("2026-07-20T00:00:00+00:00").timestamp())
        assert mock_reclaim.call_args.kwargs["request_id"] == f"offboard:m_member:{expected_epoch}"


class TestArchiveSoleAdminOrgs:
    """_archive_sole_admin_orgs now runs BEFORE _reclaim_own_seats, returns
    the set of org_ids it archived, reclaims EVERY member's seat balance
    (including the deleting admin's own — no more `.neq("user_id", ...)`
    restriction), and never writes to org_members at all (finding 1)."""

    def test_sole_admin_archives_org_and_reclaims_every_member_including_own(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        # other_admins query (role=admin, status=active, user_id != caller) -> none.
        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = []

        # organizations.archived_at: not yet archived, then the post-UPDATE reread.
        orgs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = [
            MagicMock(data=None),
            MagicMock(data={"archived_at": "2026-07-20T12:00:00+00:00"}),
        ]

        # all_members query (org_id eq ONLY — no .neq("user_id", ...) anymore)
        # -> every seat in the org, INCLUDING the deleting admin's own ("m1").
        om.select.return_value.eq.return_value.execute.return_value.data = [
            {"id": "m1"},
            {"id": "m2"},
            {"id": "m3"},
        ]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == {"org1"}
        orgs_tbl.update.assert_called_once()
        assert "archived_at" in orgs_tbl.update.call_args[0][0]
        om.update.assert_not_called()  # org_members is NEVER written by this function

        expected_epoch = int(datetime.fromisoformat("2026-07-20T12:00:00+00:00").timestamp())
        calls = mock_reclaim.call_args_list
        assert len(calls) == 3
        member_ids = {c.kwargs["member_id"] for c in calls}
        assert member_ids == {"m1", "m2", "m3"}  # includes the sole admin's own seat
        for c in calls:
            assert c.kwargs["org_id"] == "org1"
            assert c.kwargs["reason"] == "org_teardown"
            assert c.kwargs["request_id"] == f"orgteardown:{c.kwargs['member_id']}:{expected_epoch}"

    def test_skips_when_another_active_admin_exists(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = [
            {"id": "m_other_admin"}
        ]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == set()
        orgs_tbl.update.assert_not_called()
        mock_reclaim.assert_not_called()

    def test_non_admin_own_rows_are_never_considered(self):
        """own_rows entries that aren't an active admin seat must not even
        trigger the other-admins lookup — the set of admin_org_ids is empty."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        own_rows = [
            {"org_id": "org1", "role": "member", "status": "active"},
            {"org_id": "org2", "role": "admin", "status": "removed"},
        ]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == set()
        om.select.assert_not_called()
        orgs_tbl.update.assert_not_called()
        mock_reclaim.assert_not_called()

    def test_retry_reuses_existing_archived_at_without_restamping(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = []
        orgs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "archived_at": "2026-07-01T00:00:00+00:00"
        }
        # sole member of the org is the admin themselves.
        om.select.return_value.eq.return_value.execute.return_value.data = [{"id": "m1"}]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim:
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == {"org1"}
        orgs_tbl.update.assert_not_called()
        expected_epoch = int(datetime.fromisoformat("2026-07-01T00:00:00+00:00").timestamp())
        mock_reclaim.assert_called_once()
        assert mock_reclaim.call_args.kwargs["request_id"] == f"orgteardown:m1:{expected_epoch}"

    def test_fresh_archive_calls_teardown_for_the_archived_org(self):
        """Licensing Phase C, Task 4 (rule 12): a FRESH archive (this call
        actually flips archived_at) must also run
        `_teardown_archived_org_grants` for that org, so a sole-admin
        deletion never leaves a live grant or a tombstone link behind."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = []
        orgs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = [
            MagicMock(data=None),
            MagicMock(data={"archived_at": "2026-07-20T12:00:00+00:00"}),
        ]
        om.select.return_value.eq.return_value.execute.return_value.data = [{"id": "m1"}]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with (
            patch("users.account_deletion_service._reclaim_seat_to_pool"),
            patch("users.account_deletion_service._teardown_archived_org_grants") as mock_teardown,
        ):
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == {"org1"}
        mock_teardown.assert_called_once_with(sb, "org1")

    def test_retry_already_archived_org_still_calls_teardown(self):
        """A retry-detected already-archived org (the archive itself is a
        no-op, reusing the stored archived_at) must ALSO re-run the
        teardown — a prior attempt may have archived the org but crashed
        before the grant/link cleanup ran; re-running it is idempotent."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = []
        orgs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "archived_at": "2026-07-01T00:00:00+00:00"
        }
        om.select.return_value.eq.return_value.execute.return_value.data = [{"id": "m1"}]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with (
            patch("users.account_deletion_service._reclaim_seat_to_pool"),
            patch("users.account_deletion_service._teardown_archived_org_grants") as mock_teardown,
        ):
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == {"org1"}
        orgs_tbl.update.assert_not_called()
        mock_teardown.assert_called_once_with(sb, "org1")

    def test_skipped_org_never_calls_teardown(self):
        """Not the sole admin -> no archive, no teardown either."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())

        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = [
            {"id": "m_other_admin"}
        ]

        own_rows = [{"org_id": "org1", "role": "admin", "status": "active"}]

        with (
            patch("users.account_deletion_service._reclaim_seat_to_pool"),
            patch("users.account_deletion_service._teardown_archived_org_grants") as mock_teardown,
        ):
            archived = _archive_sole_admin_orgs(sb, "u1", own_rows)

        assert archived == set()
        mock_teardown.assert_not_called()


class TestTeardownArchivedOrgGrants:
    """Standalone coverage of `_teardown_archived_org_grants` (Licensing
    Phase C, Task 4, rule 12) — mirrors `orgs.service._teardown_archived_org_
    grants`'s own test coverage in tests/test_orgs_service.py. Both cleanup
    steps (revoke org-granted memberships, delete org_project_links rows)
    are independently best-effort: a failure in one must not prevent the
    other from running, and neither may ever raise out of this function
    (account deletion must never be blocked by this cleanup)."""

    def test_calls_revoke_org_scoped_and_deletes_links(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        links_tbl = tables.setdefault("org_project_links", MagicMock())

        with patch("orgs.projects.revoke_org_granted_memberships") as mock_revoke:
            _teardown_archived_org_grants(sb, "org1")

        mock_revoke.assert_called_once_with(sb, "org1")
        links_tbl.delete.return_value.eq.assert_called_once_with("org_id", "org1")

    def test_revoke_failure_is_logged_and_link_delete_still_runs(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        links_tbl = tables.setdefault("org_project_links", MagicMock())

        with patch("orgs.projects.revoke_org_granted_memberships", side_effect=RuntimeError("boom")):
            _teardown_archived_org_grants(sb, "org1")  # must not raise

        links_tbl.delete.return_value.eq.assert_called_once_with("org_id", "org1")

    def test_link_delete_failure_is_logged_and_does_not_raise(self):
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        links_tbl = tables.setdefault("org_project_links", MagicMock())
        links_tbl.delete.side_effect = RuntimeError("link delete boom")

        with patch("orgs.projects.revoke_org_granted_memberships") as mock_revoke:
            _teardown_archived_org_grants(sb, "org1")  # must not raise

        mock_revoke.assert_called_once_with(sb, "org1")

    def test_organic_and_other_org_rows_survive_through_real_revoke_helper(self):
        """End-to-end through the REAL orgs.projects.revoke_org_granted_
        memberships (not mocked): the delete is filtered on org_id only,
        never project_id/user_id, matching archive's "every grant this org
        ever made" scope while still never touching organic or other-org
        rows (the org_id filter is the entire safety mechanism)."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        pm_tbl = tables.setdefault("project_members", MagicMock())
        pm_tbl.delete.return_value.eq.return_value.execute.return_value.data = [
            {"id": "pm1", "org_id": "org1"},
            {"id": "pm2", "org_id": "org1"},
        ]

        _teardown_archived_org_grants(sb, "org1")

        pm_tbl.delete.return_value.eq.assert_called_once_with("org_id", "org1")


class TestDeleteUserAccountOrgIntegration:
    """Integration-level: verifies delete_user_account wires the two helpers
    above in the right order (Phase B review finding 1 requires
    archive-then-reclaim, NOT the reverse), passes data through correctly,
    and never lets an org-reclaim failure block the underlying account
    deletion."""

    def test_archive_runs_before_reclaim_and_auth_delete(self):
        sb = MagicMock()
        call_order: list[str] = []
        with (
            patch("users.account_deletion_service.would_be_last_admin", return_value=False),
            patch("users.account_deletion_service.cancel_user_stripe"),
            patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
            patch("users.account_deletion_service.analytics_capture"),
            patch(
                "users.account_deletion_service._archive_sole_admin_orgs",
                side_effect=lambda *a, **k: (call_order.append("archive"), set())[1],
            ),
            patch(
                "users.account_deletion_service._reclaim_own_seats",
                side_effect=lambda *a, **k: call_order.append("reclaim"),
            ),
        ):
            sb.auth.admin.delete_user.side_effect = lambda *a, **k: call_order.append("delete_user")
            delete_user_account(sb, "user-1", "u1@test.com")

        assert call_order == ["archive", "reclaim", "delete_user"]

    def test_own_rows_and_archived_ids_flow_into_reclaim_step(self):
        """own_rows is fetched ONCE by delete_user_account and handed to
        BOTH helpers; the archived-id set _archive_sole_admin_orgs returns
        flows straight into _reclaim_own_seats' 4th argument (the
        "archived-set plumbing" that lets it skip those orgs)."""
        sb = MagicMock()
        own_rows_from_db = [{"org_id": "org1", "role": "admin", "status": "active", "id": "m1", "user_id": "user-1"}]
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = own_rows_from_db
        archived_ids = {"org1"}
        with (
            patch("users.account_deletion_service.would_be_last_admin", return_value=False),
            patch("users.account_deletion_service.cancel_user_stripe"),
            patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
            patch("users.account_deletion_service.analytics_capture"),
            patch("users.account_deletion_service._archive_sole_admin_orgs", return_value=archived_ids) as mock_archive,
            patch("users.account_deletion_service._reclaim_own_seats") as mock_reclaim,
        ):
            delete_user_account(sb, "user-1", "u1@test.com")

        mock_archive.assert_called_once_with(sb, "user-1", own_rows_from_db)
        mock_reclaim.assert_called_once_with(sb, "user-1", own_rows_from_db, archived_ids)
        sb.auth.admin.delete_user.assert_called_once_with("user-1")

    def test_reclaim_failure_is_logged_and_never_blocks_deletion(self):
        sb = MagicMock()
        with (
            patch("users.account_deletion_service.would_be_last_admin", return_value=False),
            patch("users.account_deletion_service.cancel_user_stripe"),
            patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
            patch("users.account_deletion_service.analytics_capture") as analytics,
            patch("users.account_deletion_service._reclaim_own_seats", side_effect=RuntimeError("boom")),
        ):
            delete_user_account(sb, "user-1", "u1@test.com")

        sb.auth.admin.delete_user.assert_called_once_with("user-1")
        event_names = [c.args[1] for c in analytics.call_args_list]
        assert event_names == ["account_delete_started", "account_deleted"]

    def test_archive_failure_is_logged_and_never_blocks_deletion(self):
        sb = MagicMock()
        with (
            patch("users.account_deletion_service.would_be_last_admin", return_value=False),
            patch("users.account_deletion_service.cancel_user_stripe"),
            patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
            patch("users.account_deletion_service.analytics_capture") as analytics,
            patch("users.account_deletion_service._reclaim_own_seats", return_value=[]),
            patch("users.account_deletion_service._archive_sole_admin_orgs", side_effect=RuntimeError("boom")),
        ):
            delete_user_account(sb, "user-1", "u1@test.com")

        sb.auth.admin.delete_user.assert_called_once_with("user-1")
        event_names = [c.args[1] for c in analytics.call_args_list]
        assert event_names == ["account_delete_started", "account_deleted"]

    def test_sole_admin_deletion_reclaims_own_seat_without_status_flip(self):
        """End-to-end reproduction of Phase B review finding 1, exercising
        the REAL _archive_sole_admin_orgs + _reclaim_own_seats (neither is
        mocked away): a sole admin deleting their account gets their own
        seat balance reclaimed via the orgteardown key, the org archived,
        and NO org_members UPDATE is ever attempted for that membership —
        the direct status flip that would trip org_members_admin_guard
        (supabase/migrations/20260721000001_licensing_core.sql) for a sole
        active admin never happens."""
        sb = MagicMock()
        tables: dict = {}
        sb.table.side_effect = lambda name: tables.setdefault(name, MagicMock())
        om = tables.setdefault("org_members", MagicMock())
        orgs_tbl = tables.setdefault("organizations", MagicMock())

        # own_rows fetch (delete_user_account's own query) AND the
        # all_members fetch inside _archive_sole_admin_orgs share the same
        # single-eq query shape; both should surface just this one row (the
        # sole admin's own membership is the org's only seat here).
        own_row = {
            "id": "m_self",
            "org_id": "org1",
            "user_id": "u1",
            "status": "active",
            "role": "admin",
            "revoked_at": None,
        }
        om.select.return_value.eq.return_value.execute.return_value.data = [own_row]
        # other_admins query -> none (sole admin).
        om.select.return_value.eq.return_value.eq.return_value.eq.return_value.neq.return_value.execute.return_value.data = []
        # organizations.archived_at: not yet archived, then the post-UPDATE reread.
        orgs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = [
            MagicMock(data=None),
            MagicMock(data={"archived_at": "2026-07-20T12:00:00+00:00"}),
        ]

        with (
            patch("users.account_deletion_service.would_be_last_admin", return_value=False),
            patch("users.account_deletion_service.cancel_user_stripe"),
            patch("users.account_deletion_service.list_user_storage_paths", return_value=[]),
            patch("users.account_deletion_service.analytics_capture"),
            patch("users.account_deletion_service._reclaim_seat_to_pool") as mock_reclaim,
        ):
            delete_user_account(sb, "u1", "u1@test.com")

        # Org archived.
        orgs_tbl.update.assert_called_once()
        assert "archived_at" in orgs_tbl.update.call_args[0][0]

        # Own seat reclaimed exactly once, via the orgteardown key (not offboard).
        mock_reclaim.assert_called_once()
        kwargs = mock_reclaim.call_args.kwargs
        assert kwargs["member_id"] == "m_self"
        assert kwargs["org_id"] == "org1"
        assert kwargs["reason"] == "org_teardown"
        expected_epoch = int(datetime.fromisoformat("2026-07-20T12:00:00+00:00").timestamp())
        assert kwargs["request_id"] == f"orgteardown:m_self:{expected_epoch}"

        # No status-flip UPDATE was ever attempted on org_members.
        om.update.assert_not_called()

        sb.auth.admin.delete_user.assert_called_once_with("u1")
