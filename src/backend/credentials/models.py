"""Pydantic models for the artist credentials vault."""

from pydantic import BaseModel


class CredentialCreate(BaseModel):
    artist_id: str
    platform_name: str
    login_identifier: str
    password: str
    url: str | None = None
    notes: str | None = None


class CredentialUpdate(BaseModel):
    platform_name: str | None = None
    login_identifier: str | None = None
    password: str | None = None
    url: str | None = None
    notes: str | None = None


class CredentialListItem(BaseModel):
    id: str
    artist_id: str
    platform_name: str
    login_identifier: str
    url: str | None = None
    notes: str | None = None
    created_at: str
    updated_at: str


class RevealRequest(BaseModel):
    msanii_password: str


class RevealResponse(BaseModel):
    password: str
