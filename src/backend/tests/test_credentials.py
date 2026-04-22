"""Tests for the artist credentials vault.

Covers:
- AES-256-GCM encrypt/decrypt roundtrip with a configured key
- POST /credentials creates a row and does not echo the password back
- POST /credentials/{id}/reveal returns 401 on bad Msanii password
- POST /credentials/{id}/reveal returns plaintext on good Msanii password
"""

import base64
import os
import secrets
from unittest.mock import MagicMock


def _ensure_key():
    if not os.getenv("CREDENTIALS_AES_KEY"):
        os.environ["CREDENTIALS_AES_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode()


def test_aes_roundtrip():
    _ensure_key()
    # Import after env is set so module-level key cache uses a fresh value.
    from credentials import service

    service._key_cache = None  # reset cached key between tests
    ciphertext = service.encrypt_password("hunter2")
    assert ciphertext != "hunter2"
    assert service.decrypt_password(ciphertext) == "hunter2"


def test_aes_two_encrypts_produce_different_ciphertext():
    _ensure_key()
    from credentials import service

    service._key_cache = None
    a = service.encrypt_password("same-password")
    b = service.encrypt_password("same-password")
    assert a != b  # fresh nonce each time


def test_create_credential_does_not_return_ciphertext(client, mock_supabase):
    _ensure_key()
    from credentials import service as svc

    svc._key_cache = None

    inserted_row = {
        "id": "11111111-1111-1111-1111-111111111111",
        "artist_id": "22222222-2222-2222-2222-222222222222",
        "user_id": "00000000-0000-0000-0000-000000000001",
        "platform_name": "DistroKid",
        "login_identifier": "artist@example.com",
        "password_ciphertext": "ZmFrZQ==",
        "url": None,
        "notes": None,
        "created_at": "2026-04-20T00:00:00Z",
        "updated_at": "2026-04-20T00:00:00Z",
    }

    def _table(_name):
        qb = MagicMock()
        qb.insert.return_value = qb
        qb.select.return_value = qb
        qb.update.return_value = qb
        qb.delete.return_value = qb
        qb.eq.return_value = qb
        qb.order.return_value = qb
        qb.maybe_single.return_value = qb
        qb.execute.return_value = MagicMock(data=[inserted_row])
        return qb

    mock_supabase.table.side_effect = _table

    resp = client.post(
        "/credentials",
        json={
            "artist_id": inserted_row["artist_id"],
            "platform_name": "DistroKid",
            "login_identifier": "artist@example.com",
            "password": "super-secret",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "password" not in body
    assert "password_ciphertext" not in body
    assert body["platform_name"] == "DistroKid"


def test_reveal_returns_401_on_bad_msanii_password(client, mock_supabase):
    _ensure_key()
    from credentials import service as svc

    svc._key_cache = None
    svc._failed_reveals.clear()

    user_obj = MagicMock()
    user_obj.user.email = "artist@example.com"
    mock_supabase.auth.admin.get_user_by_id.return_value = user_obj
    mock_supabase.auth.sign_in_with_password.side_effect = Exception("invalid credentials")

    resp = client.post(
        "/credentials/abcd-1234/reveal",
        json={"msanii_password": "wrong"},
    )
    assert resp.status_code == 401


def test_reveal_returns_plaintext_on_good_msanii_password(client, mock_supabase):
    _ensure_key()
    from credentials import service as svc

    svc._key_cache = None
    svc._failed_reveals.clear()

    # Encrypt a known value to place in the mocked DB row
    ciphertext = svc.encrypt_password("my-real-distrokid-password")

    user_obj = MagicMock()
    user_obj.user.email = "artist@example.com"
    mock_supabase.auth.admin.get_user_by_id.return_value = user_obj
    mock_supabase.auth.sign_in_with_password.return_value = MagicMock()  # truthy success

    def _table(_name):
        qb = MagicMock()
        qb.select.return_value = qb
        qb.eq.return_value = qb
        qb.maybe_single.return_value = qb
        qb.execute.return_value = MagicMock(data={"password_ciphertext": ciphertext})
        return qb

    mock_supabase.table.side_effect = _table

    resp = client.post(
        "/credentials/abcd-1234/reveal",
        json={"msanii_password": "correct-horse-battery-staple"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"password": "my-real-distrokid-password"}
