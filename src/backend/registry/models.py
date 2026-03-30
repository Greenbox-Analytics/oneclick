"""Pydantic models for the Rights & Ownership Registry, TeamCards, and Notes."""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date


# --- Works ---

class WorkCreate(BaseModel):
    artist_id: str
    project_id: str
    title: str
    work_type: str = "single"
    isrc: Optional[str] = None
    iswc: Optional[str] = None
    upc: Optional[str] = None
    release_date: Optional[date] = None
    notes: Optional[str] = None


class WorkUpdate(BaseModel):
    title: Optional[str] = None
    work_type: Optional[str] = None
    project_id: Optional[str] = None
    isrc: Optional[str] = None
    iswc: Optional[str] = None
    upc: Optional[str] = None
    release_date: Optional[date] = None
    status: Optional[str] = None
    notes: Optional[str] = None


# --- Ownership Stakes ---

class StakeCreate(BaseModel):
    work_id: str
    stake_type: str
    holder_name: str
    holder_role: str
    percentage: float
    holder_email: Optional[str] = None
    holder_ipi: Optional[str] = None
    publisher_or_label: Optional[str] = None
    notes: Optional[str] = None


class StakeUpdate(BaseModel):
    stake_type: Optional[str] = None
    holder_name: Optional[str] = None
    holder_role: Optional[str] = None
    percentage: Optional[float] = None
    holder_email: Optional[str] = None
    holder_ipi: Optional[str] = None
    publisher_or_label: Optional[str] = None
    notes: Optional[str] = None


# --- Licensing Rights ---

class LicenseCreate(BaseModel):
    work_id: str
    license_type: str
    licensee_name: str
    licensee_email: Optional[str] = None
    territory: str = "worldwide"
    start_date: date
    end_date: Optional[date] = None
    terms: Optional[str] = None


class LicenseUpdate(BaseModel):
    license_type: Optional[str] = None
    licensee_name: Optional[str] = None
    licensee_email: Optional[str] = None
    territory: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    terms: Optional[str] = None
    status: Optional[str] = None


# --- Agreements ---

class PartyInput(BaseModel):
    name: str
    role: str
    email: Optional[str] = None


class AgreementCreate(BaseModel):
    work_id: str
    agreement_type: str
    title: str
    description: Optional[str] = None
    effective_date: date
    parties: List[PartyInput]
    file_id: Optional[str] = None
    document_hash: Optional[str] = None


# --- Collaboration ---

class CollaboratorInvite(BaseModel):
    work_id: str
    stake_id: Optional[str] = None
    email: EmailStr
    name: str
    role: str


class DisputeRequest(BaseModel):
    reason: str


# --- TeamCard ---

class TeamCardUpdate(BaseModel):
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    social_links: Optional[dict] = None
    dsp_links: Optional[dict] = None
    custom_links: Optional[list] = None
    visible_fields: Optional[list] = None


# --- Notes ---

class NoteCreate(BaseModel):
    title: str = "Untitled"
    content: list = []
    artist_id: Optional[str] = None
    project_id: Optional[str] = None
    folder_id: Optional[str] = None
    pinned: bool = False


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[list] = None
    folder_id: Optional[str] = None
    pinned: Optional[bool] = None


class FolderCreate(BaseModel):
    name: str
    artist_id: Optional[str] = None
    project_id: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: int = 0


class FolderUpdate(BaseModel):
    name: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: Optional[int] = None


# --- Project About ---

class ProjectAboutUpdate(BaseModel):
    about_content: list = []
