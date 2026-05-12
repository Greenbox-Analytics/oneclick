"""JWT authentication dependency for FastAPI endpoints.

Uses Supabase's JWKS endpoint to verify tokens with asymmetric keys (RS256/ES256).
No shared secret needed — public keys are fetched and cached automatically.
"""

import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientError

_bearer_scheme = HTTPBearer(auto_error=False)
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Lazy-initialized JWKS client that fetches public keys from Supabase."""
    global _jwks_client
    if _jwks_client is None:
        supabase_url = os.getenv("VITE_SUPABASE_URL")
        if not supabase_url:
            raise HTTPException(
                status_code=500,
                detail="VITE_SUPABASE_URL not configured",
            )
        _jwks_client = PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json")
    return _jwks_client


async def _verify_token(credentials: HTTPAuthorizationCredentials) -> dict:
    """Verify the JWT using JWKS and return the decoded payload. Shared by all
    auth dependencies. Raises HTTPException on any verification failure."""
    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
    except jwt.DecodeError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    alg = header.get("alg", "")
    try:
        if alg in ("RS256", "ES256"):
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],
                audience="authenticated",
            )
        raise HTTPException(
            status_code=401,
            detail="Unsupported token algorithm — only ES256/RS256 signing keys are accepted",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.InvalidTokenError, PyJWKClientError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Extract and verify user_id from Supabase JWT Bearer token (RS256/ES256 via JWKS)."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = await _verify_token(credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")
    return user_id


async def get_current_user_email(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Extract and verify email from Supabase JWT. Same JWKS verification as get_current_user_id."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = await _verify_token(credentials)
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token: missing email")
    return email


async def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str | None:
    """Like get_current_user_id but returns None for missing OR invalid tokens.

    Used by endpoints that allow both authenticated and unauthenticated requests
    (e.g., POST /pro-requests). A logged-out user with a stale/expired token in
    browser storage should be treated as anonymous, not blocked with 401."""
    if not credentials:
        return None
    try:
        payload = await _verify_token(credentials)
    except HTTPException:
        return None
    return payload.get("sub")
