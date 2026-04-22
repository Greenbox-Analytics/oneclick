"""Artist credentials vault — service layer.

Stores external-platform logins/passwords encrypted at rest. Passwords are
AES-256-GCM encrypted using a backend-held key (``CREDENTIALS_AES_KEY``) and
only decrypted on a reveal request that passes Msanii-password re-auth.

Nonce layout (base64-decoded): [12-byte nonce | ciphertext || 16-byte tag].
The ``AESGCM`` primitive appends the tag to the ciphertext automatically.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

_NONCE_BYTES = 12

# Rate limiter — bounded failed reveal attempts per user across the process.
# Fine for a single Cloud Run instance; promote to a DB-backed counter if we
# ever scale horizontally.
_RATE_WINDOW_SECS = 15 * 60
_RATE_MAX_FAILURES = 5
_failed_reveals: dict[str, list[float]] = {}


@dataclass
class _Key:
    raw: bytes


_key_cache: _Key | None = None


def _get_key() -> bytes:
    global _key_cache
    if _key_cache is not None:
        return _key_cache.raw
    encoded = os.getenv("CREDENTIALS_AES_KEY")
    if not encoded:
        raise RuntimeError(
            "CREDENTIALS_AES_KEY is not configured. Generate with: "
            'python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"'
        )
    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise RuntimeError(f"CREDENTIALS_AES_KEY is not valid base64: {exc}")
    if len(raw) != 32:
        raise RuntimeError(f"CREDENTIALS_AES_KEY must decode to 32 bytes (got {len(raw)})")
    _key_cache = _Key(raw=raw)
    return raw


def encrypt_password(plaintext: str) -> str:
    aes = AESGCM(_get_key())
    nonce = os.urandom(_NONCE_BYTES)
    ct_with_tag = aes.encrypt(nonce, plaintext.encode("utf-8"), associated_data=None)
    return base64.b64encode(nonce + ct_with_tag).decode("ascii")


def decrypt_password(ciphertext_b64: str) -> str:
    aes = AESGCM(_get_key())
    blob = base64.b64decode(ciphertext_b64)
    nonce, ct_with_tag = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    plaintext = aes.decrypt(nonce, ct_with_tag, associated_data=None)
    return plaintext.decode("utf-8")


def _record_failure(user_id: str) -> None:
    now = time.monotonic()
    bucket = _failed_reveals.setdefault(user_id, [])
    bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW_SECS]
    bucket.append(now)


def _failures_in_window(user_id: str) -> int:
    now = time.monotonic()
    bucket = _failed_reveals.get(user_id, [])
    fresh = [t for t in bucket if now - t < _RATE_WINDOW_SECS]
    _failed_reveals[user_id] = fresh
    return len(fresh)


def _clear_failures(user_id: str) -> None:
    _failed_reveals.pop(user_id, None)


async def list_credentials(supabase, user_id: str, artist_id: str) -> list[dict]:
    res = (
        supabase.table("artist_credentials")
        .select("id, artist_id, platform_name, login_identifier, url, notes, created_at, updated_at")
        .eq("user_id", user_id)
        .eq("artist_id", artist_id)
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


async def create_credential(
    supabase,
    user_id: str,
    artist_id: str,
    platform_name: str,
    login_identifier: str,
    password: str,
    url: str | None,
    notes: str | None,
) -> dict:
    row = {
        "artist_id": artist_id,
        "user_id": user_id,
        "platform_name": platform_name,
        "login_identifier": login_identifier,
        "password_ciphertext": encrypt_password(password),
        "url": url,
        "notes": notes,
    }
    res = supabase.table("artist_credentials").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to insert credential")
    inserted = res.data[0]
    inserted.pop("password_ciphertext", None)
    return inserted


async def update_credential(
    supabase,
    user_id: str,
    credential_id: str,
    changes: dict,
) -> dict:
    patch: dict = {}
    for key in ("platform_name", "login_identifier", "url", "notes"):
        if key in changes and changes[key] is not None:
            patch[key] = changes[key]
    if "password" in changes and changes["password"] is not None:
        patch["password_ciphertext"] = encrypt_password(changes["password"])
    if not patch:
        raise HTTPException(status_code=400, detail="No updatable fields provided")
    res = supabase.table("artist_credentials").update(patch).eq("id", credential_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Credential not found")
    updated = res.data[0]
    updated.pop("password_ciphertext", None)
    return updated


async def delete_credential(supabase, user_id: str, credential_id: str) -> None:
    res = supabase.table("artist_credentials").delete().eq("id", credential_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Credential not found")


async def reveal_credential(
    supabase,
    user_id: str,
    credential_id: str,
    msanii_password: str,
) -> str:
    if _failures_in_window(user_id) >= _RATE_MAX_FAILURES:
        raise HTTPException(status_code=429, detail="Too many failed reveal attempts. Try again later.")

    # Look up the user's email via the service-role admin API so we can re-verify their password.
    try:
        user_resp = supabase.auth.admin.get_user_by_id(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user: {exc}")

    email = None
    if user_resp and getattr(user_resp, "user", None) is not None:
        email = user_resp.user.email
    if not email:
        raise HTTPException(status_code=400, detail="User email not available for re-auth")

    try:
        supabase.auth.sign_in_with_password({"email": email, "password": msanii_password})
    except Exception:
        _record_failure(user_id)
        raise HTTPException(status_code=401, detail="Invalid Msanii password")

    _clear_failures(user_id)

    row = (
        supabase.table("artist_credentials")
        .select("password_ciphertext")
        .eq("id", credential_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not row or not row.data:
        raise HTTPException(status_code=404, detail="Credential not found")
    return decrypt_password(row.data["password_ciphertext"])
