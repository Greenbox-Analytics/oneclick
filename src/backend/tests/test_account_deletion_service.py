from unittest.mock import MagicMock, patch

import pytest

from users.account_deletion_service import (
    LastAdminError,
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
    sb = MagicMock()

    def table_side_effect(name):
        m = MagicMock()
        if name == "artists":
            m.select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
        elif name == "projects":
            m.select.return_value.in_.return_value.execute.return_value.data = [{"id": "p1"}]
        elif name == "project_files":
            m.select.return_value.in_.return_value.execute.return_value.data = [
                {"file_path": "a1/p1/contracts/foo.pdf"},
                {"file_path": "a1/p1/notes/bar.md"},
            ]
        elif name == "audio_files":
            m.select.return_value.in_.return_value.execute.return_value.data = [
                {"file_path": "a1/p1/audio/song.wav"},
            ]
        return m

    sb.table.side_effect = table_side_effect
    paths = list_user_storage_paths(sb, "user-1")
    assert ("project-files", "a1/p1/contracts/foo.pdf") in paths
    assert ("project-files", "a1/p1/notes/bar.md") in paths
    assert ("audio-files", "a1/p1/audio/song.wav") in paths
    assert len(paths) == 3


def test_list_user_storage_paths_user_with_no_projects():
    sb = MagicMock()

    def table_side_effect(name):
        m = MagicMock()
        if name == "artists":
            m.select.return_value.eq.return_value.execute.return_value.data = [{"id": "a1"}]
        elif name == "projects":
            m.select.return_value.in_.return_value.execute.return_value.data = []
        return m

    sb.table.side_effect = table_side_effect
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
