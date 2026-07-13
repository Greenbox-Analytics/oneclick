"""Pydantic models for the Rights & Ownership Registry, TeamCards, and Notes."""

from datetime import date

from pydantic import BaseModel, EmailStr

# --- Works ---


class WorkCreate(BaseModel):
    artist_id: str
    project_id: str
    title: str
    work_type: str = "single"
    custom_work_type: str | None = None
    isrc: str | None = None
    iswc: str | None = None
    upc: str | None = None
    release_date: date | None = None
    is_released: bool | None = None
    notes: str | None = None
    genre: str | None = None
    label: str | None = None
    featured_artists: list | None = None


class WorkUpdate(BaseModel):
    title: str | None = None
    work_type: str | None = None
    project_id: str | None = None
    isrc: str | None = None
    iswc: str | None = None
    upc: str | None = None
    release_date: date | None = None
    is_released: bool | None = None
    status: str | None = None
    notes: str | None = None
    genre: str | None = None
    label: str | None = None
    featured_artists: list | None = None


# --- Ownership Stakes ---


class StakeCreate(BaseModel):
    work_id: str
    stake_type: str
    holder_name: str
    holder_role: str
    percentage: float
    holder_email: str | None = None
    holder_ipi: str | None = None
    publisher_or_label: str | None = None
    notes: str | None = None
    is_owner_stake: bool = False


class StakeUpdate(BaseModel):
    stake_type: str | None = None
    holder_name: str | None = None
    holder_role: str | None = None
    percentage: float | None = None
    holder_email: str | None = None
    holder_ipi: str | None = None
    publisher_or_label: str | None = None
    notes: str | None = None


# --- Licensing Rights ---


class LicenseCreate(BaseModel):
    work_id: str
    license_type: str
    licensee_name: str
    licensee_email: str | None = None
    territory: str = "worldwide"
    start_date: date
    end_date: date | None = None
    terms: str | None = None


class LicenseUpdate(BaseModel):
    license_type: str | None = None
    licensee_name: str | None = None
    licensee_email: str | None = None
    territory: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    terms: str | None = None
    status: str | None = None


# --- Agreements ---


class PartyInput(BaseModel):
    name: str
    role: str
    email: str | None = None


class AgreementCreate(BaseModel):
    work_id: str
    agreement_type: str
    title: str
    description: str | None = None
    effective_date: date
    parties: list[PartyInput]
    file_id: str | None = None
    document_hash: str | None = None


# --- Collaboration ---


class CollaboratorInvite(BaseModel):
    work_id: str
    stake_id: str | None = None
    email: EmailStr
    name: str
    role: str


# --- TeamCard ---


class TeamCardUpdate(BaseModel):
    display_name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    avatar_url: str | None = None
    bio: str | None = None
    phone: str | None = None
    website: str | None = None
    company: str | None = None
    role: str | None = None
    social_links: dict | None = None
    dsp_links: dict | None = None
    custom_links: list | None = None
    visible_fields: list | None = None


# --- Notes ---


class NoteCreate(BaseModel):
    title: str = "Untitled"
    content: list = []
    artist_id: str | None = None
    project_id: str | None = None
    folder_id: str | None = None
    pinned: bool = False


class NoteUpdate(BaseModel):
    title: str | None = None
    content: list | None = None
    folder_id: str | None = None
    pinned: bool | None = None


class FolderCreate(BaseModel):
    name: str
    artist_id: str | None = None
    project_id: str | None = None
    parent_folder_id: str | None = None
    sort_order: int = 0


class FolderUpdate(BaseModel):
    name: str | None = None
    parent_folder_id: str | None = None
    sort_order: int | None = None


# --- Project About ---


class ProjectAboutUpdate(BaseModel):
    about_content: list = []


# --- Enhanced Collaboration ---


class StakeInput(BaseModel):
    stake_type: str  # master, publishing, soundexchange
    percentage: float
    # When set, link/update this pre-existing unlinked stake on the work
    # instead of inserting a new row (invite of someone already in Royalty Splits).
    existing_stake_id: str | None = None


# --- Per-collaborator visibility grants & access level ---


class GrantItem(BaseModel):
    resource_type: str  # project_file | audio_file | license | agreement
    resource_id: str


class GrantsBody(BaseModel):
    grants: list[GrantItem] = []
    ownership_breakdown: bool | None = None


class AccessLevelUpdate(BaseModel):
    access_level: str  # viewer | admin


class WorkRoleUpdate(BaseModel):
    role: str


class CollaboratorInviteWithStakes(BaseModel):
    work_id: str
    email: EmailStr
    name: str
    role: str
    stakes: list[StakeInput] = []
    notes: str | None = None
    access_level: str = "viewer"
    initial_grants: list[GrantItem] = []
    ownership_breakdown: bool = False
    terms: list = []


# --- Collaborator-scoped contract derivation ---


class DeriveFromContractsBody(BaseModel):
    work_id: str
    name: str
    email: str | None = None
    contract_file_ids: list[str] | None = None  # None => all linked contracts
