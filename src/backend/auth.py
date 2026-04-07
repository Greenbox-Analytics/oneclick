"""JWT authentication dependency for FastAPI endpoints.

Uses Supabase's JWKS endpoint to verify tokens with asymmetric keys (RS256/ES256).
No shared secret needed — public keys are fetched and cached automatically.
"""

import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from jwt import PyJWKClient, PyJWKClientError

_bearer_scheme = HTTPBearer(auto_error=False)
_jwks_client: Optional[PyJWKClient] = None


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
        _jwks_client = PyJWKClient(
            f"{supabase_url}/auth/v1/.well-known/jwks.json"
        )
    return _jwks_client


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """Extract and verify user_id from Supabase JWT Bearer token (RS256/ES256 via JWKS)."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
    except jwt.DecodeError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    alg = header.get("alg", "")

    try:
        if alg in ("RS256", "ES256"):
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],
                audience="authenticated",
            )
        else:
            raise HTTPException(
                status_code=401,
                detail="Unsupported token algorithm — only ES256/RS256 signing keys are accepted",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.InvalidTokenError, PyJWKClientError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")
    return user_id
