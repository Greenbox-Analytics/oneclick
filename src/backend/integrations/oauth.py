"""
Shared OAuth token management for all integrations.
Handles token encryption/decryption, refresh, and OAuth state JWT generation.
"""

import os
import time
from typing import Optional
from cryptography.fernet import Fernet
import jwt
import httpx
from supabase import Client


# Encryption key for storing OAuth tokens at rest (AES-256 via Fernet)
ENCRYPTION_KEY = os.getenv("INTEGRATION_ENCRYPTION_KEY")
# Secret for signing OAuth state JWTs (CSRF protection)
OAUTH_STATE_SECRET = os.getenv("INTEGRATION_OAUTH_STATE_SECRET")

FRONTEND_URL = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")
BACKEND_URL = os.getenv("VITE_BACKEND_API_URL", "http://localhost:8000")

# Provider OAuth configs loaded from environment
PROVIDER_CONFIGS = {
    "google_drive": {
        "client_id": lambda: os.getenv("GOOGLE_DRIVE_CLIENT_ID"),
        "client_secret": lambda: os.getenv("GOOGLE_DRIVE_CLIENT_SECRET"),
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    },
    "slack": {
        "client_id": lambda: os.getenv("SLACK_CLIENT_ID"),
        "client_secret": lambda: os.getenv("SLACK_CLIENT_SECRET"),
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "scopes": ["channels:read", "chat:write", "commands", "incoming-webhook"],
    },
    "notion": {
        "client_id": lambda: os.getenv("NOTION_CLIENT_ID"),
        "client_secret": lambda: os.getenv("NOTION_CLIENT_SECRET"),
        "auth_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "scopes": [],  # Notion scopes are set in the integration config
    },
    "monday": {
        "client_id": lambda: os.getenv("MONDAY_CLIENT_ID"),
        "client_secret": lambda: os.getenv("MONDAY_CLIENT_SECRET"),
        "auth_url": "https://auth.monday.com/oauth2/authorize",
        "token_url": "https://auth.monday.com/oauth2/token",
        "scopes": ["boards:read", "boards:write", "updates:read", "updates:write"],
    },
}


def _get_fernet() -> Fernet:
    """Get Fernet cipher for token encryption."""
    if not ENCRYPTION_KEY:
        raise RuntimeError(
            "INTEGRATION_ENCRYPTION_KEY not set. Generate one with: "
            "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    return Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def encrypt_token(token: str) -> str:
    """Encrypt an OAuth token for database storage."""
    f = _get_fernet()
    return f.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an OAuth token from database storage."""
    f = _get_fernet()
    return f.decrypt(encrypted_token.encode()).decode()


def generate_oauth_state(user_id: str, provider: str) -> str:
    """Generate a signed JWT state parameter for OAuth CSRF protection."""
    if not OAUTH_STATE_SECRET:
        raise RuntimeError("INTEGRATION_OAUTH_STATE_SECRET not set.")
    payload = {
        "user_id": user_id,
        "provider": provider,
        "exp": int(time.time()) + 600,  # 10 minute expiry
        "iat": int(time.time()),
    }
    return jwt.encode(payload, OAUTH_STATE_SECRET, algorithm="HS256")


def verify_oauth_state(state: str) -> dict:
    """Verify and decode an OAuth state JWT. Returns payload dict."""
    if not OAUTH_STATE_SECRET:
        raise RuntimeError("INTEGRATION_OAUTH_STATE_SECRET not set.")
    return jwt.decode(state, OAUTH_STATE_SECRET, algorithms=["HS256"])


def get_oauth_redirect_url(provider: str) -> str:
    """Get the OAuth callback URL for a provider."""
    return f"{BACKEND_URL}/integrations/{provider.replace('_', '-')}/callback"


def build_auth_url(provider: str, user_id: str) -> str:
    """Build the full OAuth authorization URL for a provider."""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    state = generate_oauth_state(user_id, provider)
    params = {
        "client_id": config["client_id"](),
        "redirect_uri": get_oauth_redirect_url(provider),
        "state": state,
        "response_type": "code",
    }

    if provider == "notion":
        params["owner"] = "user"
    elif provider == "slack":
        params["scope"] = ",".join(config["scopes"])
    else:
        params["scope"] = " ".join(config["scopes"])
        if provider == "google_drive":
            params["access_type"] = "offline"
            params["prompt"] = "consent"

    query = "&".join(f"{k}={v}" for k, v in params.items() if v)
    return f"{config['auth_url']}?{query}"


async def exchange_code_for_tokens(provider: str, code: str) -> dict:
    """Exchange an authorization code for access/refresh tokens."""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    payload = {
        "client_id": config["client_id"](),
        "client_secret": config["client_secret"](),
        "code": code,
        "redirect_uri": get_oauth_redirect_url(provider),
        "grant_type": "authorization_code",
    }

    headers = {}
    if provider == "notion":
        # Notion uses Basic auth for token exchange
        import base64
        credentials = base64.b64encode(
            f"{config['client_id']()}:{config['client_secret']()}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials}"
        payload = {"code": code, "grant_type": "authorization_code", "redirect_uri": get_oauth_redirect_url(provider)}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["token_url"],
            data=payload if provider != "notion" else None,
            json=payload if provider == "notion" else None,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def refresh_access_token(provider: str, refresh_token: str) -> dict:
    """Refresh an expired access token using the refresh token."""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    # Slack and Notion don't use refresh tokens the same way
    if provider in ("slack", "notion"):
        return {}

    payload = {
        "client_id": config["client_id"](),
        "client_secret": config["client_secret"](),
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(config["token_url"], data=payload)
        response.raise_for_status()
        return response.json()


async def get_valid_token(supabase_client: Client, user_id: str, provider: str) -> Optional[str]:
    """
    Get a valid access token for a user+provider, refreshing if expired.
    Returns None if no connection exists or refresh fails.
    """
    result = (
        supabase_client.table("integration_connections")
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", provider)
        .eq("status", "active")
        .single()
        .execute()
    )

    if not result.data:
        return None

    connection = result.data
    access_token = decrypt_token(connection["access_token_encrypted"])

    # Check if token is expired (with 5 minute buffer)
    if connection.get("token_expires_at"):
        from datetime import datetime, timezone
        expires_at = datetime.fromisoformat(connection["token_expires_at"].replace("Z", "+00:00"))
        if expires_at.timestamp() - time.time() < 300:
            # Token expired or expiring soon, try refresh
            refresh_encrypted = connection.get("refresh_token_encrypted")
            if not refresh_encrypted:
                # Mark as expired
                supabase_client.table("integration_connections").update(
                    {"status": "expired"}
                ).eq("id", connection["id"]).execute()
                return None

            try:
                refresh_token = decrypt_token(refresh_encrypted)
                new_tokens = await refresh_access_token(provider, refresh_token)
                if new_tokens.get("access_token"):
                    # Update stored tokens
                    update_data = {
                        "access_token_encrypted": encrypt_token(new_tokens["access_token"]),
                        "status": "active",
                    }
                    if new_tokens.get("expires_in"):
                        from datetime import timedelta
                        new_expiry = datetime.now(timezone.utc) + timedelta(seconds=new_tokens["expires_in"])
                        update_data["token_expires_at"] = new_expiry.isoformat()
                    if new_tokens.get("refresh_token"):
                        update_data["refresh_token_encrypted"] = encrypt_token(new_tokens["refresh_token"])

                    supabase_client.table("integration_connections").update(
                        update_data
                    ).eq("id", connection["id"]).execute()

                    return new_tokens["access_token"]
            except Exception:
                supabase_client.table("integration_connections").update(
                    {"status": "expired"}
                ).eq("id", connection["id"]).execute()
                return None

    return access_token


async def store_connection(
    supabase_client: Client,
    user_id: str,
    provider: str,
    tokens: dict,
) -> dict:
    """Store or update an OAuth connection in the database."""
    from datetime import datetime, timezone, timedelta

    data = {
        "user_id": user_id,
        "provider": provider,
        "access_token_encrypted": encrypt_token(tokens["access_token"]),
        "status": "active",
    }

    if tokens.get("refresh_token"):
        data["refresh_token_encrypted"] = encrypt_token(tokens["refresh_token"])

    if tokens.get("expires_in"):
        expiry = datetime.now(timezone.utc) + timedelta(seconds=tokens["expires_in"])
        data["token_expires_at"] = expiry.isoformat()

    # Provider-specific metadata
    if provider == "slack":
        data["provider_workspace_id"] = tokens.get("team", {}).get("id")
        data["provider_user_id"] = tokens.get("authed_user", {}).get("id")
    elif provider == "notion":
        data["provider_workspace_id"] = tokens.get("workspace_id")
        data["provider_user_id"] = tokens.get("owner", {}).get("user", {}).get("id")

    if tokens.get("scope"):
        data["scopes"] = tokens["scope"].split(",") if isinstance(tokens["scope"], str) else tokens["scope"]

    # Upsert: update if connection exists, create if not
    existing = (
        supabase_client.table("integration_connections")
        .select("id")
        .eq("user_id", user_id)
        .eq("provider", provider)
        .execute()
    )

    if existing.data:
        result = (
            supabase_client.table("integration_connections")
            .update(data)
            .eq("id", existing.data[0]["id"])
            .execute()
        )
    else:
        result = (
            supabase_client.table("integration_connections")
            .insert(data)
            .execute()
        )

    return result.data[0] if result.data else {}
