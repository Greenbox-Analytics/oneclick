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
    """Extract and verify user_id from Supabase JWT Bearer token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(
            credentials.credentials
        )
        payload = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="Invalid token: missing subject"
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.InvalidTokenError, PyJWKClientError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
