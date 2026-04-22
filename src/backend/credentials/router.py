"""FastAPI router for the artist credentials vault."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from credentials import service
from credentials.models import (
    CredentialCreate,
    CredentialListItem,
    CredentialUpdate,
    RevealRequest,
    RevealResponse,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


@router.get("", response_model=list[CredentialListItem])
async def list_credentials(
    artist_id: str = Query(...),
    user_id: str = Depends(get_current_user_id),
):
    rows = await service.list_credentials(_get_supabase(), user_id, artist_id)
    return rows


@router.post("", response_model=CredentialListItem)
async def create_credential(
    body: CredentialCreate,
    user_id: str = Depends(get_current_user_id),
):
    return await service.create_credential(
        _get_supabase(),
        user_id,
        artist_id=body.artist_id,
        platform_name=body.platform_name,
        login_identifier=body.login_identifier,
        password=body.password,
        url=body.url,
        notes=body.notes,
    )


@router.patch("/{credential_id}", response_model=CredentialListItem)
async def update_credential(
    credential_id: str,
    body: CredentialUpdate,
    user_id: str = Depends(get_current_user_id),
):
    return await service.update_credential(
        _get_supabase(),
        user_id,
        credential_id,
        body.model_dump(exclude_unset=True),
    )


@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: str,
    user_id: str = Depends(get_current_user_id),
):
    await service.delete_credential(_get_supabase(), user_id, credential_id)
    return {"ok": True}


@router.post("/{credential_id}/reveal", response_model=RevealResponse)
async def reveal_credential(
    credential_id: str,
    body: RevealRequest,
    user_id: str = Depends(get_current_user_id),
):
    plaintext = await service.reveal_credential(
        _get_supabase(),
        user_id,
        credential_id,
        body.msanii_password,
    )
    return RevealResponse(password=plaintext)
