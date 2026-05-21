"""Tests for the DELETE /users/account endpoint."""

from unittest.mock import patch

from tests.conftest import TEST_USER_EMAIL


def test_delete_account_email_mismatch_returns_400(client):
    r = client.request("DELETE", "/users/account", json={"confirmation_email": "wrong@test.com"})
    assert r.status_code == 400
    assert "match" in r.json()["detail"].lower()


def test_delete_account_missing_body_returns_422(client):
    r = client.request("DELETE", "/users/account", json={})
    assert r.status_code in (400, 422)


def test_delete_account_blank_email_returns_422(client):
    r = client.request("DELETE", "/users/account", json={"confirmation_email": ""})
    assert r.status_code in (400, 422)


def test_delete_account_last_admin_returns_409(client):
    from users.account_deletion_service import LastAdminError

    with patch("users.router.delete_user_account", side_effect=LastAdminError("only admin")):
        r = client.request("DELETE", "/users/account", json={"confirmation_email": TEST_USER_EMAIL})
    assert r.status_code == 409
    assert r.json()["detail"] == "last_admin"


def test_delete_account_success_returns_204_case_insensitive(client):
    with patch("users.router.delete_user_account") as fn:
        r = client.request(
            "DELETE",
            "/users/account",
            json={"confirmation_email": TEST_USER_EMAIL.upper()},
        )
    assert r.status_code == 204
    fn.assert_called_once()


def test_delete_account_generic_failure_returns_502(client):
    with patch("users.router.delete_user_account", side_effect=RuntimeError("stripe down")):
        r = client.request("DELETE", "/users/account", json={"confirmation_email": TEST_USER_EMAIL})
    assert r.status_code == 502
    assert r.json()["detail"] == "account_delete_failed"
