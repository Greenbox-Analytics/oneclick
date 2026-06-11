import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import resend
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import Client, create_client

# Add the backend directory to Python path for module resolution
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from analytics import init_analytics
from auth import get_current_user_email, get_current_user_id
from middleware.analytics_middleware import AnalyticsMiddleware
from pagination import PaginatedResponse, paginate_query
from subscriptions.admin_auth import is_active_tester_row, is_user_admin
from zoe_chatbot.contract_chatbot import ContractChatbot
from zoe_chatbot.helpers import calculate_royalty_payments

# Load environment variables
load_dotenv()

app = FastAPI()
init_analytics()

# --- Mount Integration & Board Routers ---
from admin.analytics_router import router as admin_analytics_router
from boards.router import router as boards_router
from credentials.router import router as credentials_router
from integrations.connections_router import router as connections_router
from integrations.google_drive.router import router as google_drive_router
from integrations.notion.router import router as notion_router
from integrations.slack.router import router as slack_router
from oneclick.share import router as oneclick_share_router
from projects.router import router as projects_router
from projects.share_email import router as projects_share_email_router
from registry.router import router as registry_router
from settings.router import router as settings_router
from splitsheet.router import router as splitsheet_router
from subscriptions.admin_router import router as subscriptions_admin_router
from subscriptions.billing_router import router as billing_router
from subscriptions.pro_requests_router import router as pro_requests_router
from subscriptions.router import router as subscriptions_router
from users.router import router as users_router

app.include_router(google_drive_router, prefix="/integrations/google-drive", tags=["Google Drive"])
app.include_router(slack_router, prefix="/integrations/slack", tags=["Slack"])
app.include_router(notion_router, prefix="/integrations/notion", tags=["Notion"])
app.include_router(connections_router, prefix="/integrations", tags=["Integrations"])
app.include_router(boards_router, prefix="/boards", tags=["Project Boards"])
app.include_router(settings_router, prefix="/settings", tags=["Workspace Settings"])
app.include_router(splitsheet_router, prefix="/splitsheet", tags=["Split Sheet"])
app.include_router(registry_router, prefix="/registry", tags=["Rights Registry"])
app.include_router(projects_router, prefix="/projects", tags=["Projects"])
app.include_router(projects_share_email_router, prefix="/projects", tags=["Projects"])
app.include_router(oneclick_share_router, prefix="/oneclick", tags=["OneClick"])
app.include_router(credentials_router, prefix="/credentials", tags=["Credentials Vault"])
app.include_router(users_router, prefix="/users", tags=["Users"])
app.include_router(subscriptions_router, tags=["Entitlements"])
app.include_router(subscriptions_admin_router, tags=["Admin"])
app.include_router(pro_requests_router, tags=["Pro Requests"])
app.include_router(billing_router)
app.include_router(admin_analytics_router, prefix="/admin/analytics", tags=["admin-analytics"])

# --- Register Slack notification handlers on events ---
from integrations import events
from integrations.slack.service import notify_for_event as slack_notify


async def _slack_event_handler(event_name: str, payload: dict):
    """Bridge between event bus and Slack notification service."""
    user_id = payload.get("user_id")
    if not user_id:
        return
    await slack_notify(get_supabase_client(), user_id, event_name, payload)


# Register for all notifiable events
for _event in [
    events.TASK_CREATED,
    events.TASK_UPDATED,
    events.TASK_COMPLETED,
    events.CONTRACT_UPLOADED,
    events.CONTRACT_DELETED,
    events.ROYALTY_CALCULATED,
]:
    events.on(_event, _slack_event_handler)


def _convert_pdf_background(
    db_url: str,
    db_key: str,
    file_id: str,
    file_path: str,
):
    """Background task: download PDF from storage, convert to markdown, cache in DB."""
    import tempfile

    from supabase import create_client

    from utils.ingestion.pdf_markdown import pdf_to_markdown

    try:
        db = create_client(db_url, db_key)

        # Download the PDF from storage
        file_data = db.storage.from_("project-files").download(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            markdown_text = pdf_to_markdown(tmp_path)
            # Cache the markdown in the DB
            db.table("project_files").update({"contract_markdown": markdown_text}).eq("id", file_id).execute()
            print(f"Background: PDF conversion complete for file {file_id}")
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Background: PDF conversion failed for file {file_id}: {e}")


# Configure CORS - support multiple origins from environment variable
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/") for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")
]

# Middleware order matters: Starlette's add_middleware inserts at the front of
# the chain, so the LAST add_middleware call becomes the OUTERMOST. CORS must be
# outermost among user middleware so that 4xx/known-exception responses get
# Access-Control-Allow-Origin headers attached.
app.add_middleware(AnalyticsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CORS-on-500 fallback. Starlette's ServerErrorMiddleware sits OUTSIDE user
# middleware, so an unhandled exception escapes the CORS layer and produces
# a 500 without Access-Control-Allow-Origin — the browser then shows a CORS
# error instead of the real 500 in DevTools, masking the actual bug. Adding
# an Exception handler keeps the response inside user middleware so CORS
# headers get attached. Pair this with backend logging (already in place via
# AnalyticsMiddleware's request_failed event) so the real traceback is still
# captured server-side.
import logging as _logging
import traceback as _traceback

from fastapi import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse

_uncaught_logger = _logging.getLogger("uvicorn.error")
logger = _logging.getLogger(__name__)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: _Request, exc: Exception):
    _uncaught_logger.error(
        "Unhandled %s on %s %s: %s\n%s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc,
        _traceback.format_exc(),
    )
    origin = request.headers.get("origin")
    cors_headers: dict[str, str] = {}
    if origin and (origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS):
        cors_headers["Access-Control-Allow-Origin"] = origin
        cors_headers["Access-Control-Allow-Credentials"] = "true"
        cors_headers["Vary"] = "Origin"
    return _JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_type": type(exc).__name__},
        headers=cors_headers,
    )


# Initialize Supabase Client (lazy initialization to avoid startup failures)
supabase: Client = None


def get_supabase_client() -> Client:
    """Get or create Supabase client instance"""
    global supabase
    if supabase is None:
        url: str = os.getenv("VITE_SUPABASE_URL")
        key: str = os.getenv("VITE_SUPABASE_SECRET_KEY")

        if not url or not key:
            raise RuntimeError(
                "Missing required environment variables: VITE_SUPABASE_URL and/or VITE_SUPABASE_SECRET_KEY"
            )

        supabase = create_client(url, key)
    return supabase


def normalize_file_name(file_name: str) -> str:
    return file_name.strip().lower()


# SP3: singleton + accessor moved to subscriptions/deps.py to break the import
# cycle with subscriptions/enforcement.py. Re-export here so SP1/SP2 callers
# in this file (Zoe/OneClick increments, etc.) keep working unchanged.
from subscriptions.deps import _get_entitlements_service  # noqa: I001
from subscriptions.enforcement import gated_create, gated_feature, gated_upload
from subscriptions.models import Action


# --- Ownership verification helpers ---


def verify_user_owns_artist(user_id: str, artist_id: str) -> bool:
    """Verify the artist belongs to the user."""
    res = get_supabase_client().table("artists").select("id").eq("id", artist_id).eq("user_id", user_id).execute()
    return bool(res.data)


def get_user_artist_ids(user_id: str) -> list:
    """Get all artist IDs belonging to a user."""
    res = get_supabase_client().table("artists").select("id").eq("user_id", user_id).execute()
    return [a["id"] for a in (res.data or [])]


def verify_user_owns_project(user_id: str, project_id: str) -> bool:
    """Verify the project belongs to one of the user's artists."""
    artist_ids = get_user_artist_ids(user_id)
    if not artist_ids:
        return False
    res = (
        get_supabase_client().table("projects").select("id").eq("id", project_id).in_("artist_id", artist_ids).execute()
    )
    return bool(res.data)


def verify_user_owns_contract(user_id: str, contract_id: str) -> bool:
    """Verify the contract belongs to one of the user's projects/artists."""
    contract_res = get_supabase_client().table("project_files").select("project_id").eq("id", contract_id).execute()
    if not contract_res.data:
        return False
    return verify_user_owns_project(user_id, contract_res.data[0]["project_id"])


def file_name_exists_in_project(project_id: str, file_name: str) -> bool:
    existing_files = (
        get_supabase_client().table("project_files").select("file_name").eq("project_id", project_id).execute()
    )
    target_name = normalize_file_name(file_name)
    return any(normalize_file_name(existing["file_name"]) == target_name for existing in (existing_files.data or []))


def _name_lookup_ids(contract_ids: list[str] | None, contract_markdowns: dict | None) -> list[str]:
    """Every contract we need a filename for: the selection plus everything in the working-set
    markdown payload (the frontend sends full text for selected + recently-used contracts)."""
    return list({*(contract_ids or []), *((contract_markdowns or {}).keys())})


# --- Data Models ---
class RoyaltyBreakdown(BaseModel):
    songName: str
    contributorName: str
    role: str
    royaltyPercentage: float
    amount: float


class RoyaltyResults(BaseModel):
    songTitle: str
    totalContributors: int
    totalRevenue: float
    breakdown: list[RoyaltyBreakdown]


# Conversation Context Models
class RoyaltySplitData(BaseModel):
    party: str
    percentage: float


class RoyaltySplitsByType(BaseModel):
    streaming: list[RoyaltySplitData] | None = None
    publishing: list[RoyaltySplitData] | None = None
    mechanical: list[RoyaltySplitData] | None = None
    sync: list[RoyaltySplitData] | None = None
    master: list[RoyaltySplitData] | None = None
    performance: list[RoyaltySplitData] | None = None
    general: list[RoyaltySplitData] | None = None  # For unspecified royalty types


class ExtractedContractData(BaseModel):
    royalty_splits: RoyaltySplitsByType | None = None
    payment_terms: str | None = None
    parties: list[str] | None = None
    advances: str | None = None
    term_length: str | None = None


class ArtistDataExtracted(BaseModel):
    bio: str | None = None
    social_media: dict[str, str] | None = None
    streaming_links: dict[str, str] | None = None
    genres: list[str] | None = None
    email: str | None = None


class ArtistDiscussed(BaseModel):
    id: str
    name: str
    data_extracted: ArtistDataExtracted | None = None


class ContractDiscussed(BaseModel):
    id: str
    name: str
    data_extracted: ExtractedContractData | None = None


class ContextSwitch(BaseModel):
    timestamp: str
    type: str  # 'artist' | 'project' | 'contract'
    from_value: str | None = None  # Using from_value since 'from' is reserved
    to: str

    class Config:
        # Allow 'from' as an alias in incoming JSON
        populate_by_name = True

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        # Handle 'from' field from frontend
        if isinstance(obj, dict) and "from" in obj:
            obj = {**obj, "from_value": obj.pop("from")}
        return super().model_validate(obj, *args, **kwargs)


class ConversationContext(BaseModel):
    session_id: str
    artist: dict[str, str] | None = None
    artists_discussed: list[ArtistDiscussed] = []  # NEW: Track all discussed artists
    project: dict[str, str] | None = None
    contracts_discussed: list[ContractDiscussed] = []
    context_switches: list[ContextSwitch] = []


# Zoe Chatbot Models
class ZoeAskRequest(BaseModel):
    query: str
    project_id: str | None = None  # Optional - not needed for artist-only queries
    contract_ids: list[str] | None = None  # Changed to support multiple contracts
    session_id: str | None = None  # Session ID for conversation memory
    artist_id: str | None = None  # Artist ID for artist-related queries
    context: ConversationContext | None = None  # Conversation context for structured tracking
    source_preference: Literal["artist_profile", "contract_context", "conversation_history"] | None = None
    contract_markdowns: dict[str, str] | None = None  # Full contract markdown text keyed by contract_id
    host_user_id: str | None = None  # SP3: project owner for host-wins resolution


class ZoeQuickAction(BaseModel):
    id: str
    label: str
    query: str | None = None
    source_preference: Literal["artist_profile", "contract_context", "conversation_history"] | None = None


class ZoeSource(BaseModel):
    contract_file: str
    score: float
    project_name: str

    @classmethod
    def from_dict(cls, source: dict):
        """
        Create ZoeSource from dictionary with validation

        Args:
            source: Dictionary with source data

        Returns:
            ZoeSource instance or None if required fields are missing
        """
        # Validate required fields are not None
        if source.get("contract_file") is None:
            return None

        return cls(
            contract_file=source["contract_file"],
            score=source.get("score", 0.0),
            project_name=source.get("project_name", "Unknown"),
        )


# Models for pinned facts and assumptions in API response
class PinnedFactResponse(BaseModel):
    id: str
    fact_type: str
    description: str
    value: Any
    confidence: float
    source_type: str
    source_reference: str
    scope: dict[str, str | None]
    extracted_at: str
    last_verified: str


class AssumptionResponse(BaseModel):
    id: str
    statement: str
    context: str
    scope: dict[str, str | None]
    introduced_at: str
    verified: bool
    invalidated: bool


class ZoeAskResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: list[ZoeSource]
    search_results_count: int
    highest_score: float | None = None
    session_id: str | None = None  # Return session ID for frontend tracking
    show_quick_actions: bool | None = None  # Show quick action buttons in response
    answered_from: str | None = None  # Indicates if answered from context vs document search
    extracted_data: dict[str, Any] | None = None  # Server-side extracted structured data (contract or artist)
    pending_suggestion: str | None = None  # Follow-up suggestion that can be answered from context
    context_cleared: bool | None = None  # True if context was cleared and user needs to refresh
    needs_source_selection: bool | None = None
    quick_actions: list[ZoeQuickAction] | None = None
    # New fields for enhanced conversation management
    extracted_facts: list[PinnedFactResponse] | None = None  # Facts extracted from this response
    active_assumptions: list[AssumptionResponse] | None = None  # Currently active assumptions
    invalidated_assumptions: list[str] | None = None  # IDs of assumptions invalidated this turn
    confidence_score: float | None = None  # Numeric confidence (0.0-1.0)
    needs_clarification: bool | None = None  # True if query is ambiguous
    clarification_options: list[str] | None = None  # Options for disambiguation


# Initialize Zoe chatbot (singleton)
zoe_chatbot = None


def get_zoe_chatbot():
    """Get or create Zoe chatbot instance"""
    global zoe_chatbot
    if zoe_chatbot is None:
        zoe_chatbot = ContractChatbot()
    return zoe_chatbot


# --- Endpoints ---


@app.get("/")
def read_root():
    return {"message": "Msanii AI Backend is running"}


@app.get("/health")
def health_check():
    """
    Health check endpoint for Cloud Run and monitoring.
    Returns 200 OK if the service is healthy.
    """
    return {"status": "healthy", "service": "msanii-backend", "version": "1.0.0"}


class AnalyticsContext(BaseModel):
    is_tester: bool
    is_admin: bool
    plan: Literal["free", "pro"]
    role: str | None
    email: str | None
    signed_up_at: str | None
    tester_granted_at: str | None
    tester_expires_at: str | None


def _read_profile_role(sb, user_id: str) -> str | None:
    """Return profiles.role for the user, or None if the column doesn't exist
    in this environment.

    Reason: the `industry → role` rename migration is wrapped in IF EXISTS
    (see 20260420000000_rename_industry_to_role.sql) — environments that
    never had `industry` added via the Supabase dashboard also don't have
    `role`. Tolerate that gracefully instead of 500-ing the whole endpoint.
    """
    try:
        res = sb.table("profiles").select("role").eq("id", user_id).maybe_single().execute()
        return (res.data or {}).get("role") if res else None
    except Exception:
        return None


def _read_auth_signed_up_at(sb, user_id: str) -> str | None:
    """Return auth.users.created_at for the user via the Supabase admin client.

    profiles.created_at doesn't exist in some environments; the canonical
    signup timestamp lives on auth.users.created_at and is always present.
    """
    try:
        res = sb.auth.admin.get_user_by_id(user_id)
        created_at = getattr(res.user, "created_at", None) if res and res.user else None
        # auth.admin returns a datetime; serialize for the JSON response.
        return created_at.isoformat() if hasattr(created_at, "isoformat") else created_at
    except Exception:
        return None


@app.get("/me/analytics-context", response_model=AnalyticsContext)
async def get_analytics_context(
    user_id: str = Depends(get_current_user_id),
    user_email: str = Depends(get_current_user_email),
):
    """Pure read — returns identity properties for client-side PostHog identify().

    Does NOT call analytics.identify(): GETs stay idempotent. The client owns
    identify() and server-side identify only fires from explicit mutation handlers.

    Source-of-truth per field (profiles is intentionally minimal in this schema):
      email         <- JWT (auth.users)
      signed_up_at  <- auth.users.created_at (profiles doesn't have created_at)
      role          <- profiles.role IF the column exists in this env (else None)
    """
    sb = get_supabase_client()

    overrides = (
        sb.table("tier_overrides")
        .select("reason, granted_at, expires_at")
        .eq("user_id", user_id)
        .like("reason", "tester%")
        .execute()
    )
    active_tester_row = next((r for r in (overrides.data or []) if is_active_tester_row(r)), None)

    sub_res = sb.table("subscriptions").select("tier").eq("user_id", user_id).execute()
    sub_rows = sub_res.data or []
    plan = sub_rows[0]["tier"] if sub_rows else "free"

    return AnalyticsContext(
        is_tester=active_tester_row is not None,
        is_admin=is_user_admin(sb, user_email, user_id),
        plan=plan,
        role=_read_profile_role(sb, user_id),
        email=user_email,
        signed_up_at=_read_auth_signed_up_at(sb, user_id),
        tester_granted_at=active_tester_row.get("granted_at") if active_tester_row else None,
        tester_expires_at=active_tester_row.get("expires_at") if active_tester_row else None,
    )


@app.post("/me/bootstrap-tester")
async def bootstrap_tester(
    user_id: str = Depends(get_current_user_id),
    user_email: str = Depends(get_current_user_email),
):
    """Auto-grant tester `tier_overrides` row if caller's email is in TESTER_EMAILS.

    Called on every SIGNED_IN event from the frontend (see AuthContext). Idempotent:
    - skips when email not in TESTER_EMAILS (returns reason=not_in_allowlist)
    - skips when user already has any active tester row (returns reason=already_tester)
    - grants reason='tester_env' on first call to distinguish from manual admin grants

    Mirrors AdminService.create_tester_grant's payload exactly so feature gating is identical.
    """
    from datetime import UTC, datetime

    from analytics import identify as analytics_identify
    from subscriptions.admin_auth import is_active_tester_row, is_env_tester

    if not is_env_tester(user_email):
        return {"granted": False, "reason": "not_in_allowlist"}

    sb = get_supabase_client()
    existing = (
        sb.table("tier_overrides")
        .select("reason, expires_at, granted_at")
        .eq("user_id", user_id)
        .like("reason", "tester%")
        .execute()
    )
    rows = existing.data or []
    if any(r.get("reason") == "tester_revoked" for r in rows):
        return {"granted": False, "reason": "revoked"}
    if any(is_active_tester_row(r) for r in rows):
        return {"granted": False, "reason": "already_tester"}

    granted_at = datetime.now(UTC).isoformat()
    payload = {
        "user_id": user_id,
        "max_artists": -1,
        "max_projects": -1,
        "max_tasks": -1,
        "max_storage_bytes": -1,
        "max_split_sheets_per_month": -1,
        "max_oneclick_runs_per_month": -1,
        "zoe_enabled": True,
        "oneclick_enabled": True,
        "registry_enabled": True,
        "integrations_allowed": ["google_drive", "slack", "notion"],
        "reason": "tester_env",
        "expires_at": None,
        "granted_at": granted_at,
    }
    sb.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()

    try:
        analytics_identify(
            user_id,
            {
                "is_tester": True,
                "tester_granted_at": granted_at,
                "tester_expires_at": None,
            },
        )
    except Exception as exc:
        logger.warning("analytics identify on bootstrap-tester failed: %s", exc)

    return {"granted": True, "source": "env"}


@app.get("/artists")
async def get_artists(
    user_id: str = Depends(get_current_user_id),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """
    Fetch artists belonging to the authenticated user.
    """
    try:
        query = get_supabase_client().table("artists").select("*", count="exact").eq("user_id", user_id)
        return paginate_query(query, page, page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artists/{artist_id}")
async def get_artist_by_id(artist_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch a single artist by ID, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = get_supabase_client().table("artists").select("*").eq("id", artist_id).single().execute()
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{artist_id}")
async def get_projects(artist_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch projects for a specific artist, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = get_supabase_client().table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProjectCreateRequest(BaseModel):
    artist_id: str
    name: str
    description: str | None = None


@app.post("/projects")
async def create_project(project: ProjectCreateRequest, user_id: str = Depends(get_current_user_id)):
    """
    Create a new project for an artist — the single source of truth for project
    creation (Portfolio, Zoe, and OneClick all call this).

    The DB trigger `auto_create_project_owner` only fires for user-client inserts
    (it reads auth.uid()), which is NULL under the service-role client used here. So
    we MUST insert the owner project_members row explicitly — otherwise the project
    has zero members and the creator can't upload files (the UI gates the upload
    button on project_members role).
    """
    try:
        if not verify_user_owns_artist(user_id, project.artist_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Gate: count user's existing projects across all their artists
        artist_ids = get_user_artist_ids(user_id)
        if artist_ids:
            count_res = (
                get_supabase_client()
                .table("projects")
                .select("id", count="exact")
                .in_("artist_id", artist_ids)
                .execute()
            )
            project_count = count_res.count or 0
        else:
            project_count = 0
        gated_create(user_id, "project", current_count=project_count)

        # Duplicate-name guard (case-insensitive, per artist) — moved server-side so every
        # entry point enforces it, not just the Portfolio page. We fetch the artist's project
        # names and compare normalized in Python (mirroring file_name_exists_in_project) rather
        # than using ilike, because ilike treats % and _ in the name as SQL wildcards (e.g.
        # "50% Off" would spuriously match "50ABC Off"). Best-effort: this is a check-then-
        # insert with no DB unique constraint, so a genuine concurrent double-submit of the
        # same name could still slip through (documented TOCTOU limitation).
        name = project.name.strip()
        target = name.lower()
        existing = get_supabase_client().table("projects").select("name").eq("artist_id", project.artist_id).execute()
        if any((row.get("name") or "").strip().lower() == target for row in (existing.data or [])):
            raise HTTPException(
                status_code=409,
                detail=f'A project named "{name}" already exists for this artist.',
            )

        res = (
            get_supabase_client()
            .table("projects")
            .insert({"artist_id": project.artist_id, "name": name, "description": project.description})
            .execute()
        )
        new_project = res.data[0]

        # Explicitly add the creator as project owner. The service-role client bypasses the
        # auth.uid()-based trigger, so without this the project would have zero members.
        try:
            get_supabase_client().table("project_members").insert(
                {"project_id": new_project["id"], "user_id": user_id, "role": "owner"}
            ).execute()
        except Exception as member_err:
            # Roll back the just-created project so we don't persist an owner-less (unusable)
            # one. The DELETE cascades to project_members, and prevent_owner_deletion permits
            # that cascade (it only blocks a *top-level* owner-row delete; it returns OLD when
            # pg_trigger_depth() > 1 — migration 20260518020000). So even in the rare edge where
            # the owner row actually committed and only the response read failed, the rollback
            # cleanly removes the project and its owner row — no orphan left behind. The
            # try/except guards any other rollback failure so we still surface a clean 500.
            try:
                get_supabase_client().table("projects").delete().eq("id", new_project["id"]).execute()
            except Exception:
                logger.exception(
                    "Rollback of project %s after owner-insert failure did not complete",
                    new_project["id"],
                )
            raise HTTPException(
                status_code=500,
                detail="Failed to finalize project creation. Please try again.",
            ) from member_err

        return new_project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{project_id}")
async def get_project_files(
    project_id: str,
    user_id: str = Depends(get_current_user_id),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """
    Fetch files associated with a specific project, verifying user ownership.
    """
    try:
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")
        query = get_supabase_client().table("project_files").select("*", count="exact").eq("project_id", project_id)
        return paginate_query(query, page, page_size)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/artist/{artist_id}/category/{category}")
async def get_artist_files_by_category(artist_id: str, category: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch all files for an artist filtered by category, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        # 1. Get all project IDs for the artist
        projects_res = get_supabase_client().table("projects").select("id").eq("artist_id", artist_id).execute()
        project_ids = [p["id"] for p in projects_res.data]

        if not project_ids:
            return []

        # 2. Get files in these projects with the specified category
        # category map: frontend 'contract' -> DB 'contracts', 'royalty' -> 'royalty_statements'
        db_category = category
        if category == "contract":
            db_category = "contract"
        if category == "royalty_statement":
            db_category = "royalty_statement"
        if category == "split_Sheet":
            db_category = "split_sheet"
        if category == "other":
            db_category = "other"

        files_res = (
            get_supabase_client()
            .table("project_files")
            .select("*")
            .in_("project_id", project_ids)
            .eq("folder_category", db_category)
            .execute()
        )
        return files_res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    artist_id: str = Form(...),
    category: str = Form(...),  # 'contract' or 'royalty_statement'
    project_id: str | None = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    """
    Uploads a file to Supabase Storage and creates a record in project_files.
    Verifies user owns the artist before uploading.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        if not project_id or project_id == "none" or project_id == "":
            # Check if a "General" project exists for this artist, if not create one
            res = (
                get_supabase_client()
                .table("projects")
                .select("id")
                .eq("artist_id", artist_id)
                .eq("name", "General Uploads")
                .execute()
            )
            if res.data:
                project_id = res.data[0]["id"]
            else:
                # Create "General Uploads" project
                new_proj = (
                    get_supabase_client()
                    .table("projects")
                    .insert(
                        {
                            "artist_id": artist_id,
                            "name": "General Uploads",
                            "description": "Container for files not assigned to a specific release",
                        }
                    )
                    .execute()
                )
                project_id = new_proj.data[0]["id"]

        if file_name_exists_in_project(project_id, file.filename):
            raise HTTPException(
                status_code=409, detail=f'A file named "{file.filename}" already exists in this project.'
            )

        file_content = await file.read()

        # ---- SP3: gate BEFORE Storage write ----
        gated_upload(user_id, size=len(file_content), host_user_id=user_id)

        # Clean filename
        import re

        safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "", file.filename)

        # Generate unique path: artist_id/project_id/category/timestamp_filename
        timestamp = int(time.time())

        # Map frontend category to DB/Folder category
        folder_category = "other"
        if category == "contract":
            folder_category = "contract"
        elif category == "royalty_statement" or category == "royalty":
            folder_category = "royalty_statement"
        elif category == "split_sheet":
            folder_category = "split_sheet"
        elif category == "other":
            folder_category = "other"

        file_path = f"{artist_id}/{project_id}/{folder_category}/{timestamp}_{safe_filename}"

        # 1. Upload to Storage — explicit content-type so signed URLs serve the
        # real MIME type (the client defaults to text/plain, which breaks inline PDF viewing)
        get_supabase_client().storage.from_("project-files").upload(
            file_path,
            file_content,
            file_options={"content-type": file.content_type or "application/octet-stream"},
        )

        # Get public URL
        file_url = get_supabase_client().storage.from_("project-files").get_public_url(file_path)

        # 2. Insert into Database
        db_record = {
            "project_id": project_id,
            "folder_category": folder_category,
            "file_name": file.filename,
            "file_url": file_url,
            "file_path": file_path,
            "file_size": file.size,
            "file_type": file.content_type,
        }

        try:
            db_res = get_supabase_client().table("project_files").insert(db_record).execute()
        except Exception as db_error:
            try:
                get_supabase_client().storage.from_("project-files").remove([file_path])
            except Exception as cleanup_error:
                print(f"Failed to cleanup uploaded file after DB error: {cleanup_error}")

            error_message = str(db_error)
            is_trigger_reject = "Storage cap exceeded" in error_message or "23514" in error_message
            if is_trigger_reject:
                raise HTTPException(
                    status_code=402,
                    detail="Upload would exceed your storage cap (concurrent upload race). Try again or upgrade to Pro.",
                )
            if "duplicate" in error_message.lower() and "project_files" in error_message.lower():
                raise HTTPException(
                    status_code=409, detail=f'A file named "{file.filename}" already exists in this project.'
                )
            raise db_error

        # Emit event for integrations (Slack notifications) — never break upload on failure
        try:
            from integrations import events

            project_name = ""
            try:
                project_row = (
                    get_supabase_client().table("projects").select("name").eq("id", project_id).single().execute()
                )
                pdata = project_row.data
                if isinstance(pdata, dict):
                    project_name = pdata.get("name", "")
                elif isinstance(pdata, list) and pdata:
                    project_name = pdata[0].get("name", "")
            except Exception:
                pass

            await events.emit(
                events.CONTRACT_UPLOADED,
                {
                    "user_id": user_id,
                    "file_name": file.filename,
                    "project_id": project_id,
                    "project_name": project_name,
                    "file_id": db_res.data[0]["id"] if db_res.data else "",
                },
            )
        except Exception:
            pass

        return {"status": "success", "file": db_res.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Zoe AI Chatbot Endpoints ---


@app.get("/projects")
async def get_all_projects(
    user_id: str = Depends(get_current_user_id),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """
    Fetch projects belonging to the authenticated user's artists.
    """
    try:
        artist_ids = get_user_artist_ids(user_id)
        if not artist_ids:
            if page is not None:
                return PaginatedResponse(data=[], total=0, page=page, page_size=page_size)
            return []
        query = get_supabase_client().table("projects").select("*", count="exact").in_("artist_id", artist_ids)
        return paginate_query(query, page, page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artists/{artist_id}/projects")
async def get_artist_projects(artist_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch projects for a specific artist, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = get_supabase_client().table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/contracts")
async def get_project_contracts(project_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch contracts (PDF files) for a specific project, verifying user ownership.
    """
    try:
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = (
            get_supabase_client()
            .table("project_files")
            .select("*")
            .eq("project_id", project_id)
            .eq("folder_category", "contract")
            .execute()
        )
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/documents")
async def get_project_documents(project_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetch contracts and split sheets for a project (used by Zoe).
    """
    try:
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = (
            get_supabase_client()
            .table("project_files")
            .select("*")
            .eq("project_id", project_id)
            .in_("folder_category", ["contract", "split_sheet"])
            .execute()
        )
        # Derive page_count from the [[PAGE n]] markers in the cached markdown (null until converted),
        # and drop the heavy markdown from the response — the selector only needs the count.
        rows = response.data or []
        for f in rows:
            md = f.pop("contract_markdown", None)
            f["page_count"] = md.count("[[PAGE ") if md else None
        return rows
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/zoe/context-tree")
async def get_zoe_context_tree(user_id: str = Depends(get_current_user_id)):
    """Aggregated artists → projects (with project/document counts) for Zoe's comparison-context
    selector. Lightweight — counts only; documents (with page counts) are fetched per checked project."""
    try:
        db = get_supabase_client()
        artists = db.table("artists").select("id, name").eq("user_id", user_id).execute().data or []
        artist_ids = [a["id"] for a in artists]
        projects = []
        doc_count_by_project: dict[str, int] = {}
        if artist_ids:
            projects = (
                db.table("projects").select("id, name, artist_id").in_("artist_id", artist_ids).execute().data or []
            )
            project_ids = [p["id"] for p in projects]
            if project_ids:
                files = (
                    db.table("project_files")
                    .select("project_id")
                    .in_("project_id", project_ids)
                    .in_("folder_category", ["contract", "split_sheet"])
                    .execute()
                    .data
                    or []
                )
                for f in files:
                    pid = f["project_id"]
                    doc_count_by_project[pid] = doc_count_by_project.get(pid, 0) + 1
        proj_count_by_artist: dict[str, int] = {}
        for p in projects:
            proj_count_by_artist[p["artist_id"]] = proj_count_by_artist.get(p["artist_id"], 0) + 1
        return {
            "artists": [
                {"id": a["id"], "name": a["name"], "project_count": proj_count_by_artist.get(a["id"], 0)}
                for a in artists
            ],
            "projects": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "artist_id": p["artist_id"],
                    "doc_count": doc_count_by_project.get(p["id"], 0),
                }
                for p in projects
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Contract Upload/Deletion Endpoints ---


class ContractUploadRequest(BaseModel):
    project_id: str


class ContractUploadResponse(BaseModel):
    status: str
    contract_id: str
    contract_filename: str
    total_chunks: int
    message: str


class ContractDeleteRequest(BaseModel):
    contract_id: str


class ContractDeleteResponse(BaseModel):
    status: str
    contract_id: str
    message: str


async def _upload_contract_impl(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    file_content: bytes,
    project_id: str,
    user_id: str,
) -> ContractUploadResponse:
    """Core contract upload logic. Reusable by single + multi upload endpoints.

    Note: file_content is pre-read by the caller. The caller is also responsible
    for running gated_upload(...) before calling this helper — the multi-upload
    sums sizes across the batch and runs one gate for all files.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Verify user owns the project
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Get project details
        project_res = get_supabase_client().table("projects").select("name").eq("id", project_id).execute()
        if not project_res.data:
            raise HTTPException(status_code=404, detail="Project not found")

        if file_name_exists_in_project(project_id, file.filename):
            raise HTTPException(
                status_code=409, detail=f'A file named "{file.filename}" already exists in this project.'
            )

        _project_name = project_res.data[0]["name"]

        # Clean filename
        import re

        safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "", file.filename)

        # Generate unique path for storage
        timestamp = int(time.time())
        file_path = f"{user_id}/{project_id}/contract/{timestamp}_{safe_filename}"

        # 1. Upload to Supabase Storage — endpoint accepts PDFs only, so pin the
        # MIME type (the client defaults to text/plain, which breaks inline PDF viewing)
        get_supabase_client().storage.from_("project-files").upload(
            file_path,
            file_content,
            file_options={"content-type": "application/pdf"},
        )

        # Get public URL
        file_url = get_supabase_client().storage.from_("project-files").get_public_url(file_path)

        # 2. Insert into Database
        db_record = {
            "project_id": project_id,
            "folder_category": "contract",
            "file_name": file.filename,
            "file_url": file_url,
            "file_path": file_path,
            "file_size": len(file_content),
            "file_type": file.content_type,
        }

        try:
            db_res = get_supabase_client().table("project_files").insert(db_record).execute()
        except Exception as db_error:
            try:
                get_supabase_client().storage.from_("project-files").remove([file_path])
            except Exception as cleanup_error:
                print(f"Failed to cleanup uploaded contract after DB error: {cleanup_error}")

            error_message = str(db_error)
            is_trigger_reject = "Storage cap exceeded" in error_message or "23514" in error_message
            if is_trigger_reject:
                raise HTTPException(
                    status_code=402,
                    detail="Upload would exceed your storage cap (concurrent upload race). Try again or upgrade to Pro.",
                )
            if "duplicate" in error_message.lower() and "project_files" in error_message.lower():
                raise HTTPException(
                    status_code=409, detail=f'A file named "{file.filename}" already exists in this project.'
                )
            raise db_error

        contract_id = db_res.data[0]["id"]

        # 3. Queue background PDF-to-markdown conversion
        # contract_markdown starts as NULL; background task populates it.
        # GET /contracts/{id}/markdown serves as lazy fallback if task hasn't completed.
        background_tasks.add_task(
            _convert_pdf_background,
            db_url=os.getenv("VITE_SUPABASE_URL"),
            db_key=os.getenv("VITE_SUPABASE_SECRET_KEY"),
            file_id=contract_id,
            file_path=file_path,
        )

        # Emit event for integrations (Slack notifications) — never break upload on failure
        try:
            from integrations import events

            project_name = ""
            try:
                project_row = (
                    get_supabase_client().table("projects").select("name").eq("id", project_id).single().execute()
                )
                pdata = project_row.data
                if isinstance(pdata, dict):
                    project_name = pdata.get("name", "")
                elif isinstance(pdata, list) and pdata:
                    project_name = pdata[0].get("name", "")
            except Exception:
                pass

            await events.emit(
                events.CONTRACT_UPLOADED,
                {
                    "user_id": user_id,
                    "file_name": file.filename,
                    "project_id": project_id,
                    "project_name": project_name,
                    "file_id": contract_id,
                },
            )
        except Exception:
            pass

        analytics_capture(user_id, "contract_uploaded", {"file_size": len(file_content)})
        return ContractUploadResponse(
            status="success",
            contract_id=contract_id,
            contract_filename=file.filename,
            total_chunks=0,
            message="Contract uploaded successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload contract: {str(e)}")


@app.post("/contracts/upload", response_model=ContractUploadResponse)
async def upload_contract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload and process a contract PDF:
    1. Save to Supabase Storage
    2. Extract text and chunk into 300-600 tokens
    3. Generate deterministic chunk IDs (SHA256 of content + metadata)
    4. Embed and upsert to user's namespace in Pinecone

    Args:
        file: PDF file to upload
        project_id: UUID of the project
        user_id: UUID of the authenticated user

    Returns:
        Upload statistics and confirmation
    """
    file_content = await file.read()
    gated_upload(user_id, size=len(file_content), host_user_id=user_id)
    return await _upload_contract_impl(background_tasks, file, file_content, project_id, user_id)


@app.post("/contracts/upload-multiple")
async def upload_multiple_contracts(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    project_id: str = Form(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload and process multiple contract PDFs in one request

    Args:
        files: List of PDF files to upload
        project_id: UUID of the project
        user_id: UUID of the authenticated user

    Returns:
        List of upload results for each file
    """
    # ---- SP3: read all files upfront, gate with TOTAL batch size ----
    file_contents: list[tuple[UploadFile, bytes]] = []
    for file in files:
        contents = await file.read()
        file_contents.append((file, contents))

    total_size = sum(len(c) for _, c in file_contents)
    gated_upload(user_id, size=total_size, host_user_id=user_id)

    # ---- Process each file individually using the shared helper ----
    results = []

    for original_file, contents in file_contents:
        try:
            result = await _upload_contract_impl(
                background_tasks,
                original_file,
                contents,
                project_id,
                user_id,
            )
            results.append(
                {
                    "filename": original_file.filename,
                    "status": "success",
                    "contract_id": result.contract_id,
                    "total_chunks": result.total_chunks,
                }
            )
        except Exception as e:
            results.append({"filename": original_file.filename, "status": "error", "error": str(e)})

    return {
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results,
    }


@app.get("/contracts/{contract_id}/markdown")
async def get_contract_markdown(contract_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Get the full markdown text of a contract for full-document context.
    If markdown is not cached, lazily converts the PDF and caches the result.
    """
    try:
        # Verify user owns the contract
        if not verify_user_owns_contract(user_id, contract_id):
            raise HTTPException(status_code=403, detail="Access denied")

        res = (
            get_supabase_client()
            .table("project_files")
            .select("id, file_name, file_path, contract_markdown")
            .eq("id", contract_id)
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Contract not found")

        record = res.data[0]
        markdown = record.get("contract_markdown")

        # Re-convert when empty OR cached before page markers existed (Tier 1 page-jump).
        from utils.ingestion.pdf_markdown import markdown_has_page_markers, pdf_to_markdown

        if not markdown or not markdown_has_page_markers(markdown):
            file_data = get_supabase_client().storage.from_("project-files").download(record["file_path"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            try:
                markdown = pdf_to_markdown(tmp_path)
                get_supabase_client().table("project_files").update({"contract_markdown": markdown}).eq(
                    "id", contract_id
                ).execute()
            finally:
                os.unlink(tmp_path)

        return {"contract_id": contract_id, "contract_file": record.get("file_name", ""), "markdown": markdown}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get contract markdown: {str(e)}")


@app.get("/contracts/{contract_id}/pdf-url")
async def get_contract_pdf_url(contract_id: str, user_id: str = Depends(get_current_user_id)):
    """Return a short-lived signed URL to the contract's original PDF for inline (iframe) viewing."""
    try:
        if not verify_user_owns_contract(user_id, contract_id):
            raise HTTPException(status_code=403, detail="Access denied")

        res = (
            get_supabase_client()
            .table("project_files")
            .select("id, file_name, file_path")
            .eq("id", contract_id)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Contract not found")

        file_path = res.data[0].get("file_path")
        if not file_path:
            raise HTTPException(status_code=404, detail="Contract has no stored file")

        signed = get_supabase_client().storage.from_("project-files").create_signed_url(file_path, 3600)
        url = signed.get("signedURL") or signed.get("signedUrl") or ""
        if not url:
            raise HTTPException(status_code=500, detail="Could not create file URL")

        return {"contract_id": contract_id, "file_name": res.data[0].get("file_name", ""), "url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get contract PDF URL: {str(e)}")


@app.delete("/contracts/{contract_id}", response_model=ContractDeleteResponse)
async def delete_contract(contract_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Delete a contract and all its vector embeddings

    Args:
        contract_id: UUID of the contract to delete
        user_id: UUID of the authenticated user

    Returns:
        Deletion confirmation
    """
    try:
        # Verify user owns the contract
        if not verify_user_owns_contract(user_id, contract_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # 1. Get contract details from database
        contract_res = get_supabase_client().table("project_files").select("*").eq("id", contract_id).execute()

        if not contract_res.data:
            raise HTTPException(status_code=404, detail="Contract not found")

        contract = contract_res.data[0]

        # 2. (Pinecone deletion removed — vectors are no longer created)

        # 3. Delete from Supabase Storage
        if contract.get("file_path"):
            try:
                get_supabase_client().storage.from_("project-files").remove([contract["file_path"]])
            except Exception as e:
                print(f"Warning: Failed to delete file from storage: {e}")

        # 4. Delete from Database
        get_supabase_client().table("project_files").delete().eq("id", contract_id).execute()

        # Emit event for integrations (Slack notifications) — never break delete on failure
        try:
            from integrations import events

            project_name = ""
            try:
                project_row = (
                    get_supabase_client()
                    .table("projects")
                    .select("name")
                    .eq("id", contract["project_id"])
                    .single()
                    .execute()
                )
                pdata = project_row.data
                if isinstance(pdata, dict):
                    project_name = pdata.get("name", "")
                elif isinstance(pdata, list) and pdata:
                    project_name = pdata[0].get("name", "")
            except Exception:
                pass

            await events.emit(
                events.CONTRACT_DELETED,
                {
                    "user_id": user_id,
                    "file_name": contract.get("file_name", ""),
                    "project_id": contract.get("project_id", ""),
                    "project_name": project_name,
                    "file_id": contract_id,
                },
            )
        except Exception:
            pass

        return ContractDeleteResponse(
            status="success", contract_id=contract_id, message="Contract and all associated data deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete contract: {str(e)}")


@app.post("/zoe/ask-stream")
async def zoe_ask_stream(request: ZoeAskRequest, user_id: str = Depends(get_current_user_id)):
    """
    Zoe AI Chatbot streaming endpoint — returns Server-Sent Events (SSE).

    Uses the same routing logic as /zoe/ask but streams LLM tokens in
    real-time instead of waiting for the full answer.

    SSE event types:
      start    — session metadata
      sources  — search results (before answer)
      token    — streamed answer chunk
      data     — extracted data + confidence (after answer)
      done     — stream complete
      complete — instant non-streamed response (tiers 1/2)
      error    — error during generation
    """
    # SP3: gate Zoe feature; raises 402 for Free users without a Pro host.
    gated_feature(user_id, Action.USE_ZOE, host_user_id=request.host_user_id)

    # Step event: fire BEFORE any work begins so we capture even early failures.
    import time as _time

    _zoe_started_at = _time.perf_counter()
    _query_text = request.query or ""
    # ZoeAskRequest has no attachment field today; reserve for future and default False.
    _has_attachment = bool(getattr(request, "attachment_id", None) or getattr(request, "file_id", None))
    analytics_capture(
        user_id,
        "zoe_query_submitted",
        {
            "tool": "zoe",
            "query_length": len(_query_text),
            "has_attachment": _has_attachment,
            "mode": "contract" if request.contract_ids else "general",
        },
    )

    try:
        # SP2: track per-period Zoe usage; best-effort, never blocks.
        # NOTE: increment fires on connection-open, not completion. A client that
        # opens-then-closes mid-stream still counts as a query. This is acceptable
        # for a usage-display counter; if the counter ever becomes billing-relevant,
        # move this call to the end of the streaming handler (after final yield).
        _get_entitlements_service().increment_usage(user_id, "zoe_queries_this_period")
        analytics_capture(
            user_id, "tool_used", {"tool": "zoe", "mode": "contract" if request.contract_ids else "general"}
        )
        chatbot = get_zoe_chatbot()
        session_id = request.session_id or str(uuid.uuid4())

        # Fetch artist data if artist_id is provided
        artist_data = None
        if request.artist_id:
            try:
                artist_response = (
                    get_supabase_client().table("artists").select("*").eq("id", request.artist_id).single().execute()
                )
                artist_data = artist_response.data
            except Exception as e:
                print(f"Warning: Could not fetch artist data: {e}")

        # Convert context to dict
        context_dict = None
        if request.context:
            context_dict = request.context.model_dump()

        # Look up contract filenames for labeling in LLM context
        contract_names = {}
        name_ids = _name_lookup_ids(request.contract_ids, request.contract_markdowns)
        if name_ids:
            try:
                res = get_supabase_client().table("project_files").select("id, file_name").in_("id", name_ids).execute()
                contract_names = {r["id"]: r["file_name"] for r in (res.data or [])}
            except Exception as e:
                print(f"Warning: Could not fetch contract names: {e}")

        def generate():
            import json as _json

            source_count = 0
            try:
                for event in chatbot.ask_stream(
                    query=request.query,
                    user_id=user_id,
                    project_id=request.project_id,
                    contract_ids=request.contract_ids,
                    top_k=8,
                    session_id=session_id,
                    artist_data=artist_data,
                    context=context_dict,
                    source_preference=request.source_preference,
                    contract_markdowns=request.contract_markdowns,
                    contract_names=contract_names,
                ):
                    # Count sources events for the response_received step event.
                    # Real chatbot yields SSE-formatted strings; tests may yield dicts.
                    try:
                        if isinstance(event, dict):
                            payload = event
                        elif isinstance(event, str) and event.startswith("data: "):
                            payload = _json.loads(event[6:].strip())
                        else:
                            payload = None
                        if isinstance(payload, dict) and payload.get("type") == "sources":
                            source_count += len(payload.get("sources") or []) + len(
                                payload.get("reference_sources") or []
                            )
                    except (ValueError, TypeError):
                        # Best-effort parsing; never block the stream on counter errors.
                        pass
                    yield event

                analytics_capture(
                    user_id,
                    "zoe_response_received",
                    {
                        "tool": "zoe",
                        "duration_ms": int((_time.perf_counter() - _zoe_started_at) * 1000),
                        "source_count": source_count,
                        "mode": "contract" if request.contract_ids else "general",
                    },
                )
            except Exception as e:
                print(f"[Stream] Error in Zoe streaming: {e}")
                analytics_capture(
                    user_id,
                    "zoe_query_failed",
                    {
                        "tool": "zoe",
                        "error_code": type(e).__name__,
                        "mode": "contract" if request.contract_ids else "general",
                    },
                )
                yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error in Zoe streaming chatbot: {str(e)}")
        analytics_capture(
            user_id,
            "zoe_query_failed",
            {"tool": "zoe", "error_code": type(e).__name__, "mode": "contract" if request.contract_ids else "general"},
        )
        raise HTTPException(status_code=500, detail=f"Zoe encountered an error: {str(e)}")


@app.delete("/zoe/session/{session_id}")
async def zoe_clear_session(session_id: str):
    """
    Clear conversation history for a session.
    Use this when starting a new conversation or switching projects.
    """
    try:
        chatbot = get_zoe_chatbot()
        chatbot.clear_session(session_id)
        return {"message": "Session cleared successfully", "session_id": session_id}
    except Exception as e:
        print(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/zoe/session/{session_id}/history")
async def zoe_get_session_history(session_id: str):
    """
    Get conversation history for a session.
    Useful for restoring chat state or debugging.
    """
    try:
        chatbot = get_zoe_chatbot()
        history = chatbot.get_session_history(session_id)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        print(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")


# --- OneClick Royalty Calculation Endpoints ---


class OneClickRoyaltyRequest(BaseModel):
    contract_id: str | None = None
    contract_ids: list[str] | None = None
    project_id: str
    royalty_statement_file_id: str
    host_user_id: str | None = None  # SP3: project owner for host-wins resolution


class RoyaltyPaymentResponse(BaseModel):
    song_title: str
    party_name: str
    role: str
    royalty_type: str
    percentage: float
    total_royalty: float
    amount_to_pay: float
    terms: str | None = None


class OneClickRoyaltyResponse(BaseModel):
    status: str
    total_payments: int
    payments: list[RoyaltyPaymentResponse]
    excel_file_url: str | None = None
    message: str
    is_cached: bool | None = False


class ConfirmCalculationRequest(BaseModel):
    contract_ids: list[str]
    royalty_statement_id: str
    project_id: str
    results: dict[str, Any]


@app.post("/oneclick/confirm")
async def confirm_calculation(request: ConfirmCalculationRequest, user_id: str = Depends(get_current_user_id)):
    """
    Save confirmed calculation results to the database.
    """
    try:
        # 0. Delete old cached calculation for same statement + contracts (if any)
        existing = (
            get_supabase_client()
            .table("royalty_calculations")
            .select("id")
            .eq("royalty_statement_id", request.royalty_statement_id)
            .execute()
        )

        if existing.data:
            calc_ids = [calc["id"] for calc in existing.data]
            all_contracts_res = (
                get_supabase_client()
                .table("royalty_calculation_contracts")
                .select("calculation_id, contract_id")
                .in_("calculation_id", calc_ids)
                .execute()
            )

            contract_map = {}
            for row in all_contracts_res.data or []:
                contract_map.setdefault(row["calculation_id"], set()).add(row["contract_id"])

            for calc in existing.data:
                cached_ids = contract_map.get(calc["id"], set())
                if cached_ids == set(request.contract_ids):
                    get_supabase_client().table("royalty_calculation_contracts").delete().eq(
                        "calculation_id", calc["id"]
                    ).execute()
                    get_supabase_client().table("royalty_calculations").delete().eq("id", calc["id"]).execute()
                    break

        # 1. Insert into royalty_calculations
        calc_res = (
            get_supabase_client()
            .table("royalty_calculations")
            .insert(
                {
                    "royalty_statement_id": request.royalty_statement_id,
                    "project_id": request.project_id,
                    "user_id": user_id,
                    "results": request.results,
                }
            )
            .execute()
        )

        if not calc_res.data:
            raise HTTPException(status_code=500, detail="Failed to save calculation")

        calculation_id = calc_res.data[0]["id"]

        # 2. Insert into junction table for each contract
        junction_rows = [{"calculation_id": calculation_id, "contract_id": cid} for cid in request.contract_ids]

        get_supabase_client().table("royalty_calculation_contracts").insert(junction_rows).execute()

        return {"status": "success", "message": "Calculation saved successfully", "id": calculation_id}

    except Exception as e:
        print(f"Error saving calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save calculation: {str(e)}")


@app.get("/oneclick/calculate-royalties-stream")
async def oneclick_calculate_royalties_stream(
    project_id: str,
    royalty_statement_file_id: str,
    user_id: str = Depends(get_current_user_id),
    contract_id: str | None = None,
    contract_ids: list[str] | None = Query(None),
    force_recalculate: bool = False,
    host_user_id: str | None = Query(None),
):
    """
    OneClick Royalty Calculation with Server-Sent Events (SSE) for real-time progress updates.

    This endpoint streams progress updates to the client as the calculation proceeds:
    - Checks cache for existing confirmed results
    - Downloading files
    - Extracting contract data (parties, works, royalties, summary)
    - Processing royalty statement
    - Calculating payments

    Args:
        contract_id: Single contract ID (optional)
        contract_ids: List of contract IDs (optional)
        user_id: User ID
        project_id: Project ID
        royalty_statement_file_id: Royalty Statement File ID
        force_recalculate: If True, bypass cache
        host_user_id: Project owner user ID for host-wins resolution (SP3)

    Returns:
        SSE stream with progress updates and final results
    """
    # SP3: gate OneClick feature; raises 402 for Free users without a Pro host.
    gated_feature(user_id, Action.USE_ONECLICK, host_user_id=host_user_id)
    # SP2: track per-period OneClick usage; best-effort, never blocks.
    _get_entitlements_service().increment_usage(user_id, "oneclick_runs_this_period")
    analytics_capture(user_id, "tool_used", {"tool": "oneclick"})

    # Step-event instrumentation (Task 7 of PostHog dashboard plan).
    # `_started`/`_completed` are paired so funnel volumes match (cache-hit
    # path fires both together with cached=True, duration_ms=0).
    import time as _time

    _calc_started_at = _time.perf_counter()
    _contract_count = len(contract_ids) if contract_ids else (1 if contract_id else 0)

    async def generate_progress():
        # Initialize paths to None for safe cleanup
        contract_path = None
        statement_path = None

        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting OneClick calculation...', 'progress': 0, 'stage': 'starting'})}\n\n"
            await asyncio.sleep(0.1)

            # Determine contracts to process
            target_contract_ids = []
            if contract_ids:
                target_contract_ids = contract_ids
            elif contract_id:
                target_contract_ids = [contract_id]

            if not target_contract_ids:
                analytics_capture(
                    user_id,
                    "oneclick_calc_failed",
                    {"tool": "oneclick", "error_code": "ValidationError", "stage": "validation"},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'No contracts specified'})}\n\n"
                return

            # --- CACHE CHECK ---
            if not force_recalculate:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Checking cache...', 'progress': 5, 'stage': 'starting'})}\n\n"

                    # Find calculations that match the statement ID
                    # We need to find a calculation that has EXACTLY the same set of contracts

                    # 1. Get all calculations for this statement
                    calcs_res = (
                        get_supabase_client()
                        .table("royalty_calculations")
                        .select("id, results")
                        .eq("royalty_statement_id", royalty_statement_file_id)
                        .execute()
                    )

                    if calcs_res.data:
                        calc_ids = [calc["id"] for calc in calcs_res.data]

                        # Single batch query for ALL calculation-contract associations
                        all_contracts_res = (
                            get_supabase_client()
                            .table("royalty_calculation_contracts")
                            .select("calculation_id, contract_id")
                            .in_("calculation_id", calc_ids)
                            .execute()
                        )

                        # Build lookup map: calculation_id -> set of contract_ids
                        contract_map = {}
                        for row in all_contracts_res.data or []:
                            contract_map.setdefault(row["calculation_id"], set()).add(row["contract_id"])

                        # Check each cached calculation against target contracts
                        for calc in calcs_res.data:
                            cached_ids = contract_map.get(calc["id"], set())
                            if cached_ids == set(target_contract_ids):
                                # CACHE HIT!
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Found cached results!', 'progress': 100, 'stage': 'complete'})}\n\n"

                                result = calc["results"]
                                # Ensure is_cached flag is set
                                result["is_cached"] = True
                                # Add type field so frontend SSE handler recognizes it
                                result["type"] = "complete"

                                # Cache hit: fire paired _started + _completed so
                                # the funnel sees the same volume on each step.
                                analytics_capture(
                                    user_id,
                                    "oneclick_calc_started",
                                    {"tool": "oneclick", "contract_count": _contract_count, "cached": True},
                                )
                                analytics_capture(
                                    user_id,
                                    "oneclick_calc_completed",
                                    {
                                        "tool": "oneclick",
                                        "duration_ms": 0,
                                        "contract_count": _contract_count,
                                        "cached": True,
                                    },
                                )

                                yield f"data: {json.dumps(result)}\n\n"
                                return

                except Exception as e:
                    print(f"Cache check failed (continuing to calculate): {e}")

            # Cache miss (or force_recalculate) — fire `_started` AFTER the
            # cache check so cache hits don't double-fire.
            analytics_capture(
                user_id,
                "oneclick_calc_started",
                {"tool": "oneclick", "contract_count": _contract_count, "cached": False},
            )

            # Step 1: Download royalty statement
            yield f"data: {json.dumps({'type': 'status', 'message': 'Downloading royalty statement...', 'progress': 10, 'stage': 'downloading'})}\n\n"

            statement_res = (
                get_supabase_client().table("project_files").select("*").eq("id", royalty_statement_file_id).execute()
            )
            if not statement_res.data:
                analytics_capture(
                    user_id,
                    "oneclick_calc_failed",
                    {"tool": "oneclick", "error_code": "ValidationError", "stage": "validation"},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'Royalty statement file not found'})}\n\n"
                return

            statement_file = statement_res.data[0]
            file_data = get_supabase_client().storage.from_("project-files").download(statement_file["file_path"])

            file_extension = Path(statement_file["file_name"]).suffix.lower()
            if not file_extension or file_extension not in [".csv", ".xlsx", ".xls"]:
                file_extension = ".xlsx"

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_statement:
                tmp_statement.write(file_data)
                statement_path = tmp_statement.name

            yield f"data: {json.dumps({'type': 'status', 'message': 'Royalty statement downloaded', 'progress': 20, 'stage': 'downloading'})}\n\n"
            await asyncio.sleep(0.1)

            # Step 2: Fetch full contract markdown for each contract
            yield f"data: {json.dumps({'type': 'status', 'message': 'Loading contract text...', 'progress': 25, 'stage': 'downloading'})}\n\n"

            contract_markdowns = {}
            for cid in target_contract_ids:
                try:
                    c_res = (
                        get_supabase_client()
                        .table("project_files")
                        .select("id, file_path, contract_markdown")
                        .eq("id", cid)
                        .execute()
                    )
                    if c_res.data:
                        md = c_res.data[0].get("contract_markdown")
                        if not md:
                            # Lazy migration: convert PDF to markdown
                            from utils.ingestion.pdf_markdown import pdf_to_markdown

                            pdf_data = (
                                get_supabase_client()
                                .storage.from_("project-files")
                                .download(c_res.data[0]["file_path"])
                            )
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                                tmp_pdf.write(pdf_data)
                                tmp_pdf_path = tmp_pdf.name
                            try:
                                md = pdf_to_markdown(tmp_pdf_path)
                                get_supabase_client().table("project_files").update({"contract_markdown": md}).eq(
                                    "id", cid
                                ).execute()
                            finally:
                                os.unlink(tmp_pdf_path)
                        if md:
                            from utils.ingestion.pdf_markdown import strip_page_markers

                            contract_markdowns[cid] = strip_page_markers(md)
                except Exception as e:
                    print(f"Warning: Could not fetch markdown for contract {cid}: {e}")

            # Legacy: still download PDF for single contract if markdown unavailable
            if len(target_contract_ids) == 1 and contract_id and not contract_markdowns:
                contract_res = (
                    get_supabase_client().table("project_files").select("*").eq("id", target_contract_ids[0]).execute()
                )
                if contract_res.data:
                    contract_file = contract_res.data[0]
                    cd = get_supabase_client().storage.from_("project-files").download(contract_file["file_path"])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_contract:
                        tmp_contract.write(cd)
                        contract_path = tmp_contract.name

            yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {len(contract_markdowns)} contract(s)', 'progress': 30, 'stage': 'downloading'})}\n\n"

            await asyncio.sleep(0.1)

            # Step 3: Extract contract data with progress updates
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting parties from contracts...', 'progress': 35, 'stage': 'extracting_parties'})}\n\n"
            await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting works from contracts...', 'progress': 50, 'stage': 'extracting_works'})}\n\n"
            await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting royalty splits...', 'progress': 65, 'stage': 'extracting_royalty'})}\n\n"
            await asyncio.sleep(0.5)

            if len(target_contract_ids) > 1:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Merging contract data...', 'progress': 75, 'stage': 'extracting_summary'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating contract summary...', 'progress': 75, 'stage': 'extracting_summary'})}\n\n"

            await asyncio.sleep(0.3)

            # Calculate payments
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing royalty statement...', 'progress': 80, 'stage': 'processing'})}\n\n"

            payments = calculate_royalty_payments(
                contract_path=contract_path,
                statement_path=statement_path,
                user_id=user_id,
                contract_id=target_contract_ids[0] if len(target_contract_ids) == 1 else None,
                contract_ids=target_contract_ids if len(target_contract_ids) > 1 else None,
                contract_markdowns=contract_markdowns if contract_markdowns else None,
            )

            yield f"data: {json.dumps({'type': 'status', 'message': 'Calculating payments...', 'progress': 90, 'stage': 'calculating'})}\n\n"
            await asyncio.sleep(0.3)

            if not payments or len(payments) == 0:
                analytics_capture(
                    user_id,
                    "oneclick_calc_failed",
                    {"tool": "oneclick", "error_code": "NoPaymentsCalculated", "stage": "calc"},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'No payments could be calculated. Please verify the contract and royalty statement contain matching songs.'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'message': 'Finalizing results...', 'progress': 95, 'stage': 'calculating'})}\n\n"
            await asyncio.sleep(0.2)

            # Send final results
            payment_responses = []
            for payment in payments:
                payment_responses.append(
                    {
                        "song_title": payment["song_title"],
                        "party_name": payment["party_name"],
                        "role": payment["role"],
                        "royalty_type": payment["royalty_type"],
                        "percentage": payment["percentage"],
                        "total_royalty": payment["total_royalty"],
                        "amount_to_pay": payment["amount_to_pay"],
                        "terms": payment.get("terms"),
                    }
                )

            result = {
                "type": "complete",
                "status": "success",
                "total_payments": len(payments),
                "payments": payment_responses,
                "message": f"Successfully calculated {len(payments)} royalty payments",
                "progress": 100,
                "stage": "complete",
            }

            yield f"data: {json.dumps(result)}\n\n"

            _duration_ms = int((_time.perf_counter() - _calc_started_at) * 1000)
            analytics_capture(
                user_id,
                "oneclick_calc_completed",
                {
                    "tool": "oneclick",
                    "duration_ms": _duration_ms,
                    "contract_count": _contract_count,
                    "cached": False,
                },
            )

        except Exception as e:
            import traceback

            error_detail = str(e)
            traceback.print_exc()
            analytics_capture(
                user_id,
                "oneclick_calc_failed",
                {"tool": "oneclick", "error_code": type(e).__name__, "stage": "calc"},
            )
            yield f"data: {json.dumps({'type': 'error', 'message': error_detail})}\n\n"

        finally:
            # Clean up temporary files safely
            if contract_path and os.path.exists(contract_path):
                try:
                    os.unlink(contract_path)
                except Exception:
                    pass

            if statement_path and os.path.exists(statement_path):
                try:
                    os.unlink(statement_path)
                except Exception:
                    pass

    return StreamingResponse(generate_progress(), media_type="text/event-stream")


@app.post("/oneclick/calculate-royalties", response_model=OneClickRoyaltyResponse)
async def oneclick_calculate_royalties(request: OneClickRoyaltyRequest, user_id: str = Depends(get_current_user_id)):
    """
    OneClick Royalty Calculation:
    1. Retrieve streamingroyalty splits from selected contract(s) using vector search
    2. Download royalty statement from Supabase
    3. Calculate payments using royalty_calculator.py methods
    4. Save results to Excel and upload to Supabase
    5. Return payment breakdown

    Args:
        request: Contains contract_id(s), user_id, project_id, and royalty_statement_file_id

    Returns:
        Payment breakdown and Excel file URL
    """
    # Step-event instrumentation (Task 7).
    import time as _time

    _calc_started_at = _time.perf_counter()
    _contract_count = len(request.contract_ids) if request.contract_ids else (1 if request.contract_id else 0)

    try:
        # SP3: gate OneClick feature; raises 402 for Free users without a Pro host.
        gated_feature(user_id, Action.USE_ONECLICK, host_user_id=request.host_user_id)
        # SP2: track per-period OneClick usage; best-effort, never blocks.
        _get_entitlements_service().increment_usage(user_id, "oneclick_runs_this_period")
        analytics_capture(
            user_id,
            "oneclick_calc_started",
            {"tool": "oneclick", "contract_count": _contract_count, "cached": False},
        )
        print(f"\n{'=' * 80}")
        print("ONECLICK ROYALTY CALCULATION")
        print(f"{'=' * 80}")
        print(f"Contract ID: {request.contract_id}")
        print(f"Contract IDs: {request.contract_ids}")
        print(f"User ID: {user_id}")
        print(f"Project ID: {request.project_id}")
        print(f"Royalty Statement File ID: {request.royalty_statement_file_id}")

        # Determine contracts to process
        target_contract_ids = []
        if request.contract_ids:
            target_contract_ids = request.contract_ids
        elif request.contract_id:
            target_contract_ids = [request.contract_id]

        if not target_contract_ids:
            raise HTTPException(status_code=400, detail="No contracts specified")

        # Step 2: Download royalty statement from Supabase
        print("\n--- Step 1: Downloading Royalty Statement ---")
        statement_res = (
            get_supabase_client()
            .table("project_files")
            .select("*")
            .eq("id", request.royalty_statement_file_id)
            .execute()
        )

        if not statement_res.data:
            raise HTTPException(status_code=404, detail="Royalty statement file not found")

        statement_file = statement_res.data[0]
        file_path = statement_file["file_path"]

        # Download file from Supabase storage
        file_data = get_supabase_client().storage.from_("project-files").download(file_path)

        # Detect file extension from original filename
        original_filename = statement_file["file_name"]
        file_extension = Path(original_filename).suffix.lower()

        # Default to .xlsx if no extension found
        if not file_extension or file_extension not in [".csv", ".xlsx", ".xls"]:
            file_extension = ".xlsx"

        # Save to temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_statement:
            tmp_statement.write(file_data)
            statement_path = tmp_statement.name

        print(f"Downloaded royalty statement: {statement_file['file_name']} (detected as {file_extension})")

        # Step 3: Get contract file for parsing (Legacy/Single mode only)
        contract_path = None
        if len(target_contract_ids) == 1 and request.contract_id:
            print("\n--- Step 2: Downloading Contract File ---")
            contract_res = supabase.table("project_files").select("*").eq("id", request.contract_id).execute()

            if not contract_res.data:
                raise HTTPException(status_code=404, detail="Contract file not found")

            contract_file = contract_res.data[0]
            contract_file_path = contract_file["file_path"]

            # Download contract from Supabase storage
            contract_data = supabase.storage.from_("project-files").download(contract_file_path)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_contract:
                tmp_contract.write(contract_data)
                contract_path = tmp_contract.name

            print(f"Downloaded contract: {contract_file['file_name']}")
        else:
            print(f"\n--- Step 2: Preparing {len(target_contract_ids)} contracts (Pinecone) ---")

        try:
            # Step 4: Calculate payments using helper function
            print("\n--- Step 3: Calculating Royalty Payments ---")

            # Use helper function from helpers.py
            payments = calculate_royalty_payments(
                contract_path=contract_path,
                statement_path=statement_path,
                user_id=user_id,
                contract_id=target_contract_ids[0] if len(target_contract_ids) == 1 else None,
                contract_ids=target_contract_ids if len(target_contract_ids) > 1 else None,
            )

            if not payments or len(payments) == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No payments could be calculated. Please verify the contract and royalty statement contain matching songs.",
                )

            print(f"Calculated {len(payments)} payments")

            # Step 5: Format response (payments are already dictionaries from helper)
            payment_responses = []
            for payment in payments:
                payment_responses.append(
                    RoyaltyPaymentResponse(
                        song_title=payment["song_title"],
                        party_name=payment["party_name"],
                        role=payment["role"],
                        royalty_type=payment["royalty_type"],
                        percentage=payment["percentage"],
                        total_royalty=payment["total_royalty"],
                        amount_to_pay=payment["amount_to_pay"],
                        terms=payment.get("terms"),
                    )
                )

            print(f"\n{'=' * 80}")
            print("CALCULATION COMPLETE")
            print(f"{'=' * 80}\n")

            _duration_ms = int((_time.perf_counter() - _calc_started_at) * 1000)
            analytics_capture(
                user_id,
                "oneclick_calc_completed",
                {
                    "tool": "oneclick",
                    "duration_ms": _duration_ms,
                    "contract_count": _contract_count,
                    "cached": False,
                },
            )

            return OneClickRoyaltyResponse(
                status="success",
                total_payments=len(payments),
                payments=payment_responses,
                excel_file_url=None,
                message=f"Successfully calculated {len(payments)} royalty payments",
            )

        finally:
            # Clean up temporary files
            if contract_path and os.path.exists(contract_path):
                os.unlink(contract_path)
            if os.path.exists(statement_path):
                os.unlink(statement_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in OneClick royalty calculation: {str(e)}")
        import traceback

        traceback.print_exc()
        analytics_capture(
            user_id,
            "oneclick_calc_failed",
            {"tool": "oneclick", "error_code": type(e).__name__, "stage": "calc"},
        )
        raise HTTPException(status_code=500, detail=f"Failed to calculate royalties: {str(e)}")


# ─── File Sharing via Resend ────────────────────────────────────────────────


class ShareFileItem(BaseModel):
    file_name: str
    file_path: str
    file_source: Literal["project_file", "audio_file"]
    file_id: str


class ShareFilesRequest(BaseModel):
    contact_id: str | None = None
    recipient_email: str
    recipient_name: str | None = None
    files: list[ShareFileItem]
    message: str | None = None


@app.post("/share/files")
async def share_files(req: ShareFilesRequest, user_id: str = Depends(get_current_user_id)):
    """Generate signed download URLs and send them via Resend email."""
    try:
        if not req.files:
            raise HTTPException(status_code=400, detail="No files to share")

        api_key = os.getenv("RESEND_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Email service not configured")

        resend.api_key = api_key
        sb = get_supabase_client()

        # All files are stored in the project-files bucket
        def bucket_for(source: str) -> str:
            return "project-files"

        # Generate signed URLs (7 days = 604800 seconds) with download option
        expiry_seconds = 7 * 24 * 60 * 60
        link_expires_at = datetime.now(UTC) + timedelta(seconds=expiry_seconds)
        file_links: list[dict] = []

        for f in req.files:
            bucket = bucket_for(f.file_source)
            print(f"DEBUG share: bucket={bucket}, file_path='{f.file_path}', file_name='{f.file_name}'")
            if not f.file_path:
                print(f"Warning: Empty file_path for {f.file_name}, skipping")
                continue
            try:
                signed = sb.storage.from_(bucket).create_signed_url(
                    f.file_path, expiry_seconds, options={"download": True}
                )
                signed_url = signed.get("signedURL") or signed.get("signedUrl") or ""
                if not signed_url:
                    print(f"Warning: Empty signed URL for {f.file_name}, response: {signed}")
                    continue
            except Exception as e:
                print(f"Warning: Failed to create signed URL for {f.file_name}: {e}")
                continue

            file_links.append(
                {
                    "name": f.file_name,
                    "url": signed_url,
                    "source": f.file_source,
                    "file_id": f.file_id,
                }
            )

        if not file_links:
            file_names = [f.file_name for f in req.files]
            file_paths = [f.file_path for f in req.files]
            print(f"ERROR: No signed URLs generated. Files: {file_names}, Paths: {file_paths}")
            raise HTTPException(
                status_code=400,
                detail="Could not generate download links for any files. Check that file paths are valid.",
            )

        # Fetch sender profile name
        sender_name = "Someone"
        try:
            profile = sb.table("profiles").select("full_name").eq("id", user_id).single().execute()
            if profile.data and profile.data.get("full_name"):
                sender_name = profile.data["full_name"]
        except Exception:
            pass

        # Build email HTML with auto-download links
        greeting = f"Hi{' ' + req.recipient_name if req.recipient_name else ''},"
        file_rows = ""
        for fl in file_links:
            download_url = fl["url"]
            file_rows += f"""
            <tr>
              <td style="padding: 10px 16px; border-bottom: 1px solid #eee;">
                <a href="{download_url}" style="color: #16a34a; text-decoration: none; font-weight: 500;">
                  {fl["name"]}
                </a>
              </td>
              <td style="padding: 10px 16px; border-bottom: 1px solid #eee; text-align: right;">
                <a href="{download_url}"
                   style="display: inline-block; padding: 6px 14px; background: #16a34a; color: white;
                          border-radius: 6px; text-decoration: none; font-size: 13px;">
                  Download
                </a>
              </td>
            </tr>"""

        message_block = ""
        if req.message:
            message_block = f"""
            <div style="background: #f9fafb; border-left: 3px solid #16a34a; padding: 12px 16px;
                        margin: 16px 0; border-radius: 4px; font-style: italic; color: #555;">
              {req.message}
            </div>"""

        html_body = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 560px; margin: 0 auto; padding: 24px;">
          <div style="text-align: center; margin-bottom: 24px;">
            <h2 style="color: #111; margin: 0;">Msanii</h2>
            <p style="color: #888; font-size: 13px; margin: 4px 0 0;">Music Portfolio</p>
          </div>
          <p style="color: #333; line-height: 1.6;">{greeting}</p>
          <p style="color: #333; line-height: 1.6;">
            <strong>{sender_name}</strong> has shared {len(file_links)} file{"s" if len(file_links) > 1 else ""} with you.
            Click below to download.
          </p>
          {message_block}
          <table style="width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff;
                        border: 1px solid #eee; border-radius: 8px; overflow: hidden;">
            <thead>
              <tr style="background: #f8f8f8;">
                <th style="padding: 10px 16px; text-align: left; font-size: 13px; color: #666;">File</th>
                <th style="padding: 10px 16px; text-align: right; font-size: 13px; color: #666;"></th>
              </tr>
            </thead>
            <tbody>{file_rows}</tbody>
          </table>
          <p style="color: #999; font-size: 12px; margin-top: 24px;">
            These links will expire in 7 days.
          </p>
        </div>
        """

        # Send email via Resend
        from_address = os.getenv("RESEND_FROM_EMAIL")
        if not from_address:
            raise HTTPException(
                status_code=500,
                detail="Email service not configured: RESEND_FROM_EMAIL not set",
            )
        try:
            resend.Emails.send(
                {
                    "from": from_address,
                    "to": [req.recipient_email],
                    "subject": f"{sender_name} shared files with you — Msanii",
                    "html": html_body,
                }
            )
        except Exception as e:
            print(f"Resend error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

        # Record shares in DB
        for fl in file_links:
            try:
                sb.table("file_shares").insert(
                    {
                        "user_id": user_id,
                        "contact_id": req.contact_id,
                        "recipient_email": req.recipient_email,
                        "recipient_name": req.recipient_name,
                        "file_name": fl["name"],
                        "file_source": fl["source"],
                        "file_id": fl["file_id"],
                        "message": req.message,
                        "link_expires_at": link_expires_at.isoformat(),
                        "status": "sent",
                    }
                ).execute()
            except Exception as e:
                print(f"Warning: Failed to record share in DB: {e}")

        return {"status": "ok", "shared": len(file_links)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in share_files: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to share files: {str(e)}")
