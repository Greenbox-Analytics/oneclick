from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import os
import io
import uuid
import time
import tempfile
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import sys
from pathlib import Path
import json
import asyncio
import resend

# Add the backend directory to Python path for module resolution
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vector_search.contract_chatbot import ContractChatbot
from vector_search.contract_ingestion import ContractIngestion
from vector_search.contract_search import ContractSearch
from vector_search.helpers import calculate_royalty_payments

# Load environment variables
load_dotenv()

app = FastAPI()

# --- Mount Integration & Board Routers ---
from integrations.google_drive.router import router as google_drive_router
from integrations.slack.router import router as slack_router
from integrations.notion.router import router as notion_router
from integrations.monday.router import router as monday_router
from boards.router import router as boards_router
from settings.router import router as settings_router
from splitsheet.router import router as splitsheet_router

app.include_router(google_drive_router, prefix="/integrations/google-drive", tags=["Google Drive"])
app.include_router(slack_router, prefix="/integrations/slack", tags=["Slack"])
app.include_router(notion_router, prefix="/integrations/notion", tags=["Notion"])
app.include_router(monday_router, prefix="/integrations/monday", tags=["Monday.com"])
app.include_router(boards_router, prefix="/boards", tags=["Project Boards"])
app.include_router(settings_router, prefix="/settings", tags=["Workspace Settings"])
app.include_router(splitsheet_router, prefix="/splitsheet", tags=["Split Sheet"])

# Configure CORS - support multiple origins from environment variable
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    res = get_supabase_client().table("projects").select("id").eq("id", project_id).in_("artist_id", artist_ids).execute()
    return bool(res.data)

def verify_user_owns_contract(user_id: str, contract_id: str) -> bool:
    """Verify the contract belongs to one of the user's projects/artists."""
    contract_res = get_supabase_client().table("project_files").select("project_id").eq("id", contract_id).execute()
    if not contract_res.data:
        return False
    return verify_user_owns_project(user_id, contract_res.data[0]["project_id"])

def file_name_exists_in_project(project_id: str, file_name: str) -> bool:
    existing_files = get_supabase_client().table("project_files").select("file_name").eq("project_id", project_id).execute()
    target_name = normalize_file_name(file_name)
    return any(normalize_file_name(existing["file_name"]) == target_name for existing in (existing_files.data or []))

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
    breakdown: List[RoyaltyBreakdown]

# Conversation Context Models
class RoyaltySplitData(BaseModel):
    party: str
    percentage: float

class RoyaltySplitsByType(BaseModel):
    streaming: Optional[List[RoyaltySplitData]] = None
    publishing: Optional[List[RoyaltySplitData]] = None
    mechanical: Optional[List[RoyaltySplitData]] = None
    sync: Optional[List[RoyaltySplitData]] = None
    master: Optional[List[RoyaltySplitData]] = None
    performance: Optional[List[RoyaltySplitData]] = None
    general: Optional[List[RoyaltySplitData]] = None  # For unspecified royalty types

class ExtractedContractData(BaseModel):
    royalty_splits: Optional[RoyaltySplitsByType] = None
    payment_terms: Optional[str] = None
    parties: Optional[List[str]] = None
    advances: Optional[str] = None
    term_length: Optional[str] = None

class ArtistDataExtracted(BaseModel):
    bio: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None
    streaming_links: Optional[Dict[str, str]] = None
    genres: Optional[List[str]] = None
    email: Optional[str] = None

class ArtistDiscussed(BaseModel):
    id: str
    name: str
    data_extracted: Optional[ArtistDataExtracted] = None

class ContractDiscussed(BaseModel):
    id: str
    name: str
    data_extracted: Optional[ExtractedContractData] = None

class ContextSwitch(BaseModel):
    timestamp: str
    type: str  # 'artist' | 'project' | 'contract'
    from_value: Optional[str] = None  # Using from_value since 'from' is reserved
    to: str

    class Config:
        # Allow 'from' as an alias in incoming JSON
        populate_by_name = True

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        # Handle 'from' field from frontend
        if isinstance(obj, dict) and 'from' in obj:
            obj = {**obj, 'from_value': obj.pop('from')}
        return super().model_validate(obj, *args, **kwargs)

class ConversationContext(BaseModel):
    session_id: str
    artist: Optional[Dict[str, str]] = None
    artists_discussed: List[ArtistDiscussed] = []  # NEW: Track all discussed artists
    project: Optional[Dict[str, str]] = None
    contracts_discussed: List[ContractDiscussed] = []
    context_switches: List[ContextSwitch] = []

# Zoe Chatbot Models
class ZoeAskRequest(BaseModel):
    query: str
    project_id: Optional[str] = None  # Optional - not needed for artist-only queries
    contract_ids: Optional[List[str]] = None  # Changed to support multiple contracts
    user_id: str
    session_id: Optional[str] = None  # Session ID for conversation memory
    artist_id: Optional[str] = None  # Artist ID for artist-related queries
    context: Optional[ConversationContext] = None  # Conversation context for structured tracking
    source_preference: Optional[Literal["artist_profile", "contract_context", "conversation_history"]] = None
    contract_markdowns: Optional[Dict[str, str]] = None  # Full contract markdown text keyed by contract_id


class ZoeQuickAction(BaseModel):
    id: str
    label: str
    query: Optional[str] = None
    source_preference: Optional[Literal["artist_profile", "contract_context", "conversation_history"]] = None

class ZoeSource(BaseModel):
    contract_file: str
    score: float
    project_name: str
    
    @classmethod
    def from_dict(cls, source: Dict):
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
            project_name=source.get("project_name", "Unknown")
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
    scope: Dict[str, Optional[str]]
    extracted_at: str
    last_verified: str

class AssumptionResponse(BaseModel):
    id: str
    statement: str
    context: str
    scope: Dict[str, Optional[str]]
    introduced_at: str
    verified: bool
    invalidated: bool

class ZoeAskResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[ZoeSource]
    search_results_count: int
    highest_score: Optional[float] = None
    session_id: Optional[str] = None  # Return session ID for frontend tracking
    show_quick_actions: Optional[bool] = None  # Show quick action buttons in response
    answered_from: Optional[str] = None  # Indicates if answered from context vs document search
    extracted_data: Optional[Dict[str, Any]] = None  # Server-side extracted structured data (contract or artist)
    pending_suggestion: Optional[str] = None  # Follow-up suggestion that can be answered from context
    context_cleared: Optional[bool] = None  # True if context was cleared and user needs to refresh
    needs_source_selection: Optional[bool] = None
    quick_actions: Optional[List[ZoeQuickAction]] = None
    # New fields for enhanced conversation management
    extracted_facts: Optional[List[PinnedFactResponse]] = None  # Facts extracted from this response
    active_assumptions: Optional[List[AssumptionResponse]] = None  # Currently active assumptions
    invalidated_assumptions: Optional[List[str]] = None  # IDs of assumptions invalidated this turn
    confidence_score: Optional[float] = None  # Numeric confidence (0.0-1.0)
    needs_clarification: Optional[bool] = None  # True if query is ambiguous
    clarification_options: Optional[List[str]] = None  # Options for disambiguation

# Initialize Zoe chatbot and contract ingestion (singletons)
zoe_chatbot = None
contract_ingestion = None

def get_zoe_chatbot():
    """Get or create Zoe chatbot instance"""
    global zoe_chatbot
    if zoe_chatbot is None:
        zoe_chatbot = ContractChatbot()
    return zoe_chatbot

def get_contract_ingestion():
    """Get or create contract ingestion instance"""
    global contract_ingestion
    if contract_ingestion is None:
        contract_ingestion = ContractIngestion()
    return contract_ingestion

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
    return {
        "status": "healthy",
        "service": "msanii-backend",
        "version": "1.0.0"
    }

@app.get("/artists")
async def get_artists(user_id: str):
    """
    Fetch artists belonging to the authenticated user.
    """
    try:
        response = get_supabase_client().table("artists").select("*").eq("user_id", user_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artists/{artist_id}")
async def get_artist_by_id(artist_id: str, user_id: str):
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
async def get_projects(artist_id: str, user_id: str):
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
    description: Optional[str] = None
    user_id: str

@app.post("/projects")
async def create_project(project: ProjectCreateRequest):
    """
    Create a new project for an artist, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(project.user_id, project.artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        res = get_supabase_client().table("projects").insert({
            "artist_id": project.artist_id,
            "name": project.name,
            "description": project.description
        }).execute()
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{project_id}")
async def get_project_files(project_id: str, user_id: str):
    """
    Fetch files associated with a specific project, verifying user ownership.
    """
    try:
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = get_supabase_client().table("project_files").select("*").eq("project_id", project_id).execute()
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/artist/{artist_id}/category/{category}")
async def get_artist_files_by_category(artist_id: str, category: str, user_id: str):
    """
    Fetch all files for an artist filtered by category, verifying user ownership.
    """
    try:
        if not verify_user_owns_artist(user_id, artist_id):
            raise HTTPException(status_code=403, detail="Access denied")
        # 1. Get all project IDs for the artist
        projects_res = get_supabase_client().table("projects").select("id").eq("artist_id", artist_id).execute()
        project_ids = [p['id'] for p in projects_res.data]
        
        if not project_ids:
            return []

        # 2. Get files in these projects with the specified category
        # category map: frontend 'contract' -> DB 'contracts', 'royalty' -> 'royalty_statements'
        db_category = category
        if category == 'contract': db_category = 'contract'
        if category == 'royalty_statement': db_category = 'royalty_statement'
        if category == 'split_Sheet': db_category = 'split_sheet'
        if category == 'other': db_category = 'other'


        files_res = get_supabase_client().table("project_files").select("*").in_("project_id", project_ids).eq("folder_category", db_category).execute()
        return files_res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    artist_id: str = Form(...),
    category: str = Form(...), # 'contract' or 'royalty_statement'
    project_id: Optional[str] = Form(None),
    user_id: str = Form(...)
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
            res = get_supabase_client().table("projects").select("id").eq("artist_id", artist_id).eq("name", "General Uploads").execute()
            if res.data:
                project_id = res.data[0]['id']
            else:
                # Create "General Uploads" project
                new_proj = get_supabase_client().table("projects").insert({
                    "artist_id": artist_id,
                    "name": "General Uploads",
                    "description": "Container for files not assigned to a specific release"
                }).execute()
                project_id = new_proj.data[0]['id']

        if file_name_exists_in_project(project_id, file.filename):
            raise HTTPException(
                status_code=409,
                detail=f'A file named "{file.filename}" already exists in this project.'
            )

        file_content = await file.read()
        
        # Clean filename
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', file.filename)
        
        # Generate unique path: artist_id/project_id/category/timestamp_filename
        timestamp = int(time.time())
        
        # Map frontend category to DB/Folder category
        folder_category = 'other'
        if category == 'contract': folder_category = 'contract'
        elif category == 'royalty_statement' or category == 'royalty': folder_category = 'royalty_statement'
        elif category == 'split_sheet': folder_category = 'split_sheet'
        elif category == 'other': folder_category = 'other'
        
        file_path = f"{artist_id}/{project_id}/{folder_category}/{timestamp}_{safe_filename}"
        
        # 1. Upload to Storage
        storage_res = get_supabase_client().storage.from_("project-files").upload(file_path, file_content)
        
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
            "file_type": file.content_type
        }
        
        try:
            db_res = get_supabase_client().table("project_files").insert(db_record).execute()
        except Exception as db_error:
            try:
                get_supabase_client().storage.from_("project-files").remove([file_path])
            except Exception as cleanup_error:
                print(f"Failed to cleanup uploaded file after DB error: {cleanup_error}")

            error_message = str(db_error).lower()
            if "duplicate" in error_message and "project_files" in error_message:
                raise HTTPException(
                    status_code=409,
                    detail=f'A file named "{file.filename}" already exists in this project.'
                )
            raise db_error
        
        return {
            "status": "success", 
            "file": db_res.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Zoe AI Chatbot Endpoints ---

@app.get("/projects")
async def get_all_projects(user_id: str):
    """
    Fetch projects belonging to the authenticated user's artists.
    """
    try:
        artist_ids = get_user_artist_ids(user_id)
        if not artist_ids:
            return []
        response = get_supabase_client().table("projects").select("*").in_("artist_id", artist_ids).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artists/{artist_id}/projects")
async def get_artist_projects(artist_id: str, user_id: str):
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
async def get_project_contracts(project_id: str, user_id: str):
    """
    Fetch contracts (PDF files) for a specific project, verifying user ownership.
    """
    try:
        if not verify_user_owns_project(user_id, project_id):
            raise HTTPException(status_code=403, detail="Access denied")
        response = get_supabase_client().table("project_files").select("*").eq("project_id", project_id).eq("folder_category", "contract").execute()
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/documents")
async def get_project_documents(project_id: str, user_id: str):
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
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Contract Upload/Deletion Endpoints ---

class ContractUploadRequest(BaseModel):
    project_id: str
    user_id: str

class ContractUploadResponse(BaseModel):
    status: str
    contract_id: str
    contract_filename: str
    total_chunks: int
    message: str

class ContractDeleteRequest(BaseModel):
    contract_id: str
    user_id: str

class ContractDeleteResponse(BaseModel):
    status: str
    contract_id: str
    message: str

@app.post("/contracts/upload", response_model=ContractUploadResponse)
async def upload_contract(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    user_id: str = Form(...)
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
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
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
                status_code=409,
                detail=f'A file named "{file.filename}" already exists in this project.'
            )
        
        project_name = project_res.data[0]["name"]
        
        # Read file content
        file_content = await file.read()
        
        # Clean filename
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', file.filename)
        
        # Generate unique path for storage
        timestamp = int(time.time())
        file_path = f"{user_id}/{project_id}/contract/{timestamp}_{safe_filename}"
        
        # 1. Upload to Supabase Storage
        storage_res = get_supabase_client().storage.from_("project-files").upload(file_path, file_content)
        
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
            "file_type": file.content_type
        }
        
        try:
            db_res = get_supabase_client().table("project_files").insert(db_record).execute()
        except Exception as db_error:
            try:
                get_supabase_client().storage.from_("project-files").remove([file_path])
            except Exception as cleanup_error:
                print(f"Failed to cleanup uploaded contract after DB error: {cleanup_error}")

            error_message = str(db_error).lower()
            if "duplicate" in error_message and "project_files" in error_message:
                raise HTTPException(
                    status_code=409,
                    detail=f'A file named "{file.filename}" already exists in this project.'
                )
            raise db_error

        contract_id = db_res.data[0]["id"]
        
        # 3. Save PDF temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # 4. Process and ingest to Pinecone
            ingestion = get_contract_ingestion()
            stats = ingestion.ingest_contract(
                pdf_path=tmp_path,
                user_id=user_id,
                project_id=project_id,
                project_name=project_name,
                contract_id=contract_id,
                contract_filename=file.filename
            )
            
            # Store full markdown text for full-document context
            if stats.get("markdown_text"):
                try:
                    get_supabase_client().table("project_files").update(
                        {"contract_markdown": stats["markdown_text"]}
                    ).eq("id", contract_id).execute()
                except Exception as md_err:
                    print(f"Warning: Failed to store contract markdown: {md_err}")

            return ContractUploadResponse(
                status="success",
                contract_id=contract_id,
                contract_filename=file.filename,
                total_chunks=stats["total_chunks"],
                message=f"Contract uploaded and processed successfully. {stats['total_chunks']} chunks created."
            )
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload contract: {str(e)}")

@app.post("/contracts/upload-multiple")
async def upload_multiple_contracts(
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
    user_id: str = Form(...)
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
    results = []
    
    for file in files:
        try:
            # Process each file individually
            result = await upload_contract(file, project_id, user_id)
            results.append({
                "filename": file.filename,
                "status": "success",
                "contract_id": result.contract_id,
                "total_chunks": result.total_chunks
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }

@app.get("/contracts/{contract_id}/markdown")
async def get_contract_markdown(contract_id: str, user_id: str):
    """
    Get the full markdown text of a contract for full-document context.
    If markdown is not cached, lazily converts the PDF and caches the result.
    """
    try:
        # Verify user owns the contract
        if not verify_user_owns_contract(user_id, contract_id):
            raise HTTPException(status_code=403, detail="Access denied")

        res = get_supabase_client().table("project_files").select(
            "id, file_name, file_path, contract_markdown"
        ).eq("id", contract_id).execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="Contract not found")

        record = res.data[0]
        markdown = record.get("contract_markdown")

        if not markdown:
            # Lazy migration: convert PDF to markdown and cache it
            from vector_search.helpers import pdf_to_markdown
            file_data = get_supabase_client().storage.from_("project-files").download(record["file_path"])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            try:
                markdown = pdf_to_markdown(tmp_path)
                # Cache for future requests
                get_supabase_client().table("project_files").update(
                    {"contract_markdown": markdown}
                ).eq("id", contract_id).execute()
            finally:
                os.unlink(tmp_path)

        return {
            "contract_id": contract_id,
            "contract_file": record.get("file_name", ""),
            "markdown": markdown
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get contract markdown: {str(e)}")


@app.delete("/contracts/{contract_id}", response_model=ContractDeleteResponse)
async def delete_contract(contract_id: str, user_id: str = Form(...)):
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

        # 2. Delete from Pinecone
        ingestion = get_contract_ingestion()
        delete_result = ingestion.delete_contract(user_id=user_id, contract_id=contract_id)
        
        if delete_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {delete_result.get('error')}")
        
        # 3. Delete from Supabase Storage
        if contract.get("file_path"):
            try:
                get_supabase_client().storage.from_("project-files").remove([contract["file_path"]])
            except Exception as e:
                print(f"Warning: Failed to delete file from storage: {e}")
        
        # 4. Delete from Database
        get_supabase_client().table("project_files").delete().eq("id", contract_id).execute()
        
        return ContractDeleteResponse(
            status="success",
            contract_id=contract_id,
            message="Contract and all associated data deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete contract: {str(e)}")

@app.post("/zoe/ask", response_model=ZoeAskResponse)
async def zoe_ask_question(request: ZoeAskRequest):
    """
    Zoe AI Chatbot endpoint - Ask questions about contracts and artists.
    
    Rules:
    1. Always filters by user_id and project_id
    2. If contract_ids are provided, filters by specific contracts and uses top_k=8
    3. If no contract_ids, searches all contracts in project with top_k=8
    4. Only answers if highest similarity ≥ 0.75
    5. Returns answer with source citations
    6. Maintains conversation history per session for context-aware responses
    7. If artist_id is provided, can also answer questions about the artist
    8. If no project_id but artist_id is provided, can answer artist-related questions only
    """
    try:
        # Get Zoe chatbot instance
        chatbot = get_zoe_chatbot()
        
        # Use top_k=8 for both project-wide and multi-contract searches
        top_k = 8
        
        # Generate session_id if not provided (for new conversations)
        session_id = request.session_id or str(uuid.uuid4())
        
        # Fetch artist data if artist_id is provided
        artist_data = None
        if request.artist_id:
            try:
                artist_response = get_supabase_client().table("artists").select("*").eq("id", request.artist_id).single().execute()
                artist_data = artist_response.data
            except Exception as e:
                print(f"Warning: Could not fetch artist data: {e}")
        
        # Convert context to dict for chatbot methods
        context_dict = None
        if request.context:
            context_dict = request.context.model_dump()
            print(f"[Context] Received context with {len(context_dict.get('contracts_discussed', []))} contracts discussed")
            for c in context_dict.get('contracts_discussed', []):
                print(f"[Context]   - {c.get('name')}: {c.get('data_extracted', {})}")
        
        # Handle case where no project is selected (artist-only queries)
        if not request.project_id:
            result = chatbot.ask_without_project(
                query=request.query,
                user_id=request.user_id,
                session_id=session_id,
                artist_data=artist_data,
                context=context_dict,
                source_preference=request.source_preference,
                contract_markdowns=request.contract_markdowns
            )
        elif request.contract_ids and len(request.contract_ids) > 0:
            # Multiple contracts selected - search across all of them
            result = chatbot.ask_multiple_contracts(
                query=request.query,
                user_id=request.user_id,
                project_id=request.project_id,
                contract_ids=request.contract_ids,
                top_k=top_k,
                session_id=session_id,
                artist_data=artist_data,
                context=context_dict,
                source_preference=request.source_preference,
                contract_markdowns=request.contract_markdowns
            )
        else:
            # Project-wide question
            result = chatbot.ask_project(
                query=request.query,
                user_id=request.user_id,
                project_id=request.project_id,
                top_k=top_k,
                session_id=session_id,
                artist_data=artist_data,
                context=context_dict,
                source_preference=request.source_preference,
                contract_markdowns=request.contract_markdowns
            )
        
        # Format response - filter out sources with missing required fields
        valid_sources = []
        for source in result.get("sources", []):
            zoe_source = ZoeSource.from_dict(source)
            if zoe_source is not None:
                valid_sources.append(zoe_source)
        
        # Pass extracted_data through as raw dict (supports both contract and artist payloads)
        extracted_data = result.get("extracted_data")

        quick_actions = None
        if result.get("quick_actions"):
            quick_actions = [ZoeQuickAction(**action) for action in result.get("quick_actions", [])]
        
        return ZoeAskResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=valid_sources,
            search_results_count=result.get("search_results_count", 0),
            highest_score=result.get("highest_score"),
            session_id=session_id,
            show_quick_actions=result.get("show_quick_actions", False),
            answered_from=result.get("answered_from"),
            extracted_data=extracted_data,
            pending_suggestion=result.get("pending_suggestion"),
            context_cleared=result.get("context_cleared"),
            needs_source_selection=result.get("needs_source_selection"),
            quick_actions=quick_actions
        )
        
    except Exception as e:
        print(f"Error in Zoe chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Zoe encountered an error: {str(e)}")


@app.post("/zoe/ask-stream")
async def zoe_ask_stream(request: ZoeAskRequest):
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
    try:
        chatbot = get_zoe_chatbot()
        session_id = request.session_id or str(uuid.uuid4())

        # Fetch artist data if artist_id is provided
        artist_data = None
        if request.artist_id:
            try:
                artist_response = get_supabase_client().table("artists").select("*").eq("id", request.artist_id).single().execute()
                artist_data = artist_response.data
            except Exception as e:
                print(f"Warning: Could not fetch artist data: {e}")

        # Convert context to dict
        context_dict = None
        if request.context:
            context_dict = request.context.model_dump()

        # Look up contract filenames for labeling in LLM context
        contract_names = {}
        if request.contract_ids:
            try:
                res = get_supabase_client().table("project_files").select("id, file_name").in_("id", request.contract_ids).execute()
                contract_names = {r["id"]: r["file_name"] for r in (res.data or [])}
            except Exception as e:
                print(f"Warning: Could not fetch contract names: {e}")

        def generate():
            try:
                for event in chatbot.ask_stream(
                    query=request.query,
                    user_id=request.user_id,
                    project_id=request.project_id,
                    contract_ids=request.contract_ids,
                    top_k=8,
                    session_id=session_id,
                    artist_data=artist_data,
                    context=context_dict,
                    source_preference=request.source_preference,
                    contract_markdowns=request.contract_markdowns,
                    contract_names=contract_names
                ):
                    yield event
            except Exception as e:
                print(f"[Stream] Error in Zoe streaming: {e}")
                import json as _json
                yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error in Zoe streaming chatbot: {str(e)}")
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
    contract_id: Optional[str] = None
    contract_ids: Optional[List[str]] = None
    user_id: str
    project_id: str
    royalty_statement_file_id: str

class RoyaltyPaymentResponse(BaseModel):
    song_title: str
    party_name: str
    role: str
    royalty_type: str
    percentage: float
    total_royalty: float
    amount_to_pay: float
    terms: Optional[str] = None

class OneClickRoyaltyResponse(BaseModel):
    status: str
    total_payments: int
    payments: List[RoyaltyPaymentResponse]
    excel_file_url: Optional[str] = None
    message: str
    is_cached: Optional[bool] = False

class ConfirmCalculationRequest(BaseModel):
    contract_ids: List[str]
    royalty_statement_id: str
    project_id: str
    user_id: str
    results: Dict[str, Any]

@app.post("/oneclick/confirm")
async def confirm_calculation(request: ConfirmCalculationRequest):
    """
    Save confirmed calculation results to the database.
    """
    try:
        # 0. Delete old cached calculation for same statement + contracts (if any)
        existing = get_supabase_client().table("royalty_calculations")\
            .select("id")\
            .eq("royalty_statement_id", request.royalty_statement_id)\
            .execute()

        for calc in existing.data:
            contracts_res = get_supabase_client().table("royalty_calculation_contracts")\
                .select("contract_id")\
                .eq("calculation_id", calc['id'])\
                .execute()
            cached_ids = set(c['contract_id'] for c in contracts_res.data)
            if cached_ids == set(request.contract_ids):
                get_supabase_client().table("royalty_calculation_contracts")\
                    .delete().eq("calculation_id", calc['id']).execute()
                get_supabase_client().table("royalty_calculations")\
                    .delete().eq("id", calc['id']).execute()
                break

        # 1. Insert into royalty_calculations
        calc_res = get_supabase_client().table("royalty_calculations").insert({
            "royalty_statement_id": request.royalty_statement_id,
            "project_id": request.project_id,
            "user_id": request.user_id,
            "results": request.results
        }).execute()
        
        if not calc_res.data:
            raise HTTPException(status_code=500, detail="Failed to save calculation")
            
        calculation_id = calc_res.data[0]['id']
        
        # 2. Insert into junction table for each contract
        junction_rows = [
            {"calculation_id": calculation_id, "contract_id": cid}
            for cid in request.contract_ids
        ]
        
        get_supabase_client().table("royalty_calculation_contracts").insert(junction_rows).execute()
        
        return {"status": "success", "message": "Calculation saved successfully", "id": calculation_id}
        
    except Exception as e:
        print(f"Error saving calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save calculation: {str(e)}")

@app.get("/oneclick/calculate-royalties-stream")
async def oneclick_calculate_royalties_stream(
    user_id: str,
    project_id: str,
    royalty_statement_file_id: str,
    contract_id: Optional[str] = None,
    contract_ids: Optional[List[str]] = Query(None),
    force_recalculate: bool = False
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
        
    Returns:
        SSE stream with progress updates and final results
    """
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
                yield f"data: {json.dumps({'type': 'error', 'message': 'No contracts specified'})}\n\n"
                return

            # --- CACHE CHECK ---
            if not force_recalculate:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Checking cache...', 'progress': 5, 'stage': 'starting'})}\n\n"
                    
                    # Find calculations that match the statement ID
                    # We need to find a calculation that has EXACTLY the same set of contracts
                    
                    # 1. Get all calculations for this statement
                    calcs_res = get_supabase_client().table("royalty_calculations")\
                        .select("id, results")\
                        .eq("royalty_statement_id", royalty_statement_file_id)\
                        .execute()
                    
                    for calc in calcs_res.data:
                        # 2. For each candidate, get its contracts
                        contracts_res = get_supabase_client().table("royalty_calculation_contracts")\
                            .select("contract_id")\
                            .eq("calculation_id", calc['id'])\
                            .execute()
                        
                        cached_contract_ids = [c['contract_id'] for c in contracts_res.data]
                        
                        # 3. Compare sets (order doesn't matter)
                        if set(cached_contract_ids) == set(target_contract_ids):
                            # CACHE HIT!
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Found cached results!', 'progress': 100, 'stage': 'complete'})}\n\n"
                            
                            result = calc['results']
                            # Ensure is_cached flag is set
                            result['is_cached'] = True
                            # Add type field so frontend SSE handler recognizes it
                            result['type'] = 'complete'
                            
                            yield f"data: {json.dumps(result)}\n\n"
                            return

                except Exception as e:
                    print(f"Cache check failed (continuing to calculate): {e}")
            
            # Step 1: Download royalty statement
            yield f"data: {json.dumps({'type': 'status', 'message': 'Downloading royalty statement...', 'progress': 10, 'stage': 'downloading'})}\n\n"
            
            statement_res = get_supabase_client().table("project_files").select("*").eq("id", royalty_statement_file_id).execute()
            if not statement_res.data:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Royalty statement file not found'})}\n\n"
                return
            
            statement_file = statement_res.data[0]
            file_data = get_supabase_client().storage.from_("project-files").download(statement_file["file_path"])
            
            file_extension = Path(statement_file['file_name']).suffix.lower()
            if not file_extension or file_extension not in ['.csv', '.xlsx', '.xls']:
                file_extension = '.xlsx'
            
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
                    c_res = get_supabase_client().table("project_files").select(
                        "id, file_path, contract_markdown"
                    ).eq("id", cid).execute()
                    if c_res.data:
                        md = c_res.data[0].get("contract_markdown")
                        if not md:
                            # Lazy migration: convert PDF to markdown
                            from vector_search.helpers import pdf_to_markdown
                            pdf_data = get_supabase_client().storage.from_("project-files").download(c_res.data[0]["file_path"])
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                tmp_pdf.write(pdf_data)
                                tmp_pdf_path = tmp_pdf.name
                            try:
                                md = pdf_to_markdown(tmp_pdf_path)
                                get_supabase_client().table("project_files").update(
                                    {"contract_markdown": md}
                                ).eq("id", cid).execute()
                            finally:
                                os.unlink(tmp_pdf_path)
                        if md:
                            contract_markdowns[cid] = md
                except Exception as e:
                    print(f"Warning: Could not fetch markdown for contract {cid}: {e}")

            # Legacy: still download PDF for single contract if markdown unavailable
            if len(target_contract_ids) == 1 and contract_id and not contract_markdowns:
                contract_res = get_supabase_client().table("project_files").select("*").eq("id", target_contract_ids[0]).execute()
                if contract_res.data:
                    contract_file = contract_res.data[0]
                    cd = get_supabase_client().storage.from_("project-files").download(contract_file["file_path"])
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_contract:
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
                contract_markdowns=contract_markdowns if contract_markdowns else None
            )
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Calculating payments...', 'progress': 90, 'stage': 'calculating'})}\n\n"
            await asyncio.sleep(0.3)
            
            if not payments or len(payments) == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No payments could be calculated. Please verify the contract and royalty statement contain matching songs.'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Finalizing results...', 'progress': 95, 'stage': 'calculating'})}\n\n"
            await asyncio.sleep(0.2)
            
            # Send final results
            payment_responses = []
            for payment in payments:
                payment_responses.append({
                    'song_title': payment['song_title'],
                    'party_name': payment['party_name'],
                    'role': payment['role'],
                    'royalty_type': payment['royalty_type'],
                    'percentage': payment['percentage'],
                    'total_royalty': payment['total_royalty'],
                    'amount_to_pay': payment['amount_to_pay'],
                    'terms': payment.get('terms')
                })
            
            result = {
                'type': 'complete',
                'status': 'success',
                'total_payments': len(payments),
                'payments': payment_responses,
                'message': f'Successfully calculated {len(payments)} royalty payments',
                'progress': 100,
                'stage': 'complete'
            }
            
            yield f"data: {json.dumps(result)}\n\n"
                    
        except Exception as e:
            import traceback
            error_detail = str(e)
            traceback.print_exc()
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
async def oneclick_calculate_royalties(request: OneClickRoyaltyRequest):
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
    try:
        print(f"\n{'='*80}")
        print("ONECLICK ROYALTY CALCULATION")
        print(f"{'='*80}")
        print(f"Contract ID: {request.contract_id}")
        print(f"Contract IDs: {request.contract_ids}")
        print(f"User ID: {request.user_id}")
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
        statement_res = get_supabase_client().table("project_files").select("*").eq("id", request.royalty_statement_file_id).execute()
        
        if not statement_res.data:
            raise HTTPException(status_code=404, detail="Royalty statement file not found")
        
        statement_file = statement_res.data[0]
        file_path = statement_file["file_path"]
        
        # Download file from Supabase storage
        file_data = get_supabase_client().storage.from_("project-files").download(file_path)
        
        # Detect file extension from original filename
        original_filename = statement_file['file_name']
        file_extension = Path(original_filename).suffix.lower()
        
        # Default to .xlsx if no extension found
        if not file_extension or file_extension not in ['.csv', '.xlsx', '.xls']:
            file_extension = '.xlsx'
        
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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_contract:
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
                user_id=request.user_id,
                contract_id=target_contract_ids[0] if len(target_contract_ids) == 1 else None,
                contract_ids=target_contract_ids if len(target_contract_ids) > 1 else None
            )
            
            if not payments or len(payments) == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No payments could be calculated. Please verify the contract and royalty statement contain matching songs."
                )
            
            print(f"Calculated {len(payments)} payments")
            
            # Step 5: Format response (payments are already dictionaries from helper)
            payment_responses = []
            for payment in payments:
                payment_responses.append(RoyaltyPaymentResponse(
                    song_title=payment['song_title'],
                    party_name=payment['party_name'],
                    role=payment['role'],
                    royalty_type=payment['royalty_type'],
                    percentage=payment['percentage'],
                    total_royalty=payment['total_royalty'],
                    amount_to_pay=payment['amount_to_pay'],
                    terms=payment.get('terms')
                ))
            
            print(f"\n{'='*80}")
            print("CALCULATION COMPLETE")
            print(f"{'='*80}\n")
            
            return OneClickRoyaltyResponse(
                status="success",
                total_payments=len(payments),
                payments=payment_responses,
                excel_file_url=None,
                message=f"Successfully calculated {len(payments)} royalty payments"
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
        raise HTTPException(status_code=500, detail=f"Failed to calculate royalties: {str(e)}")


# ─── File Sharing via Resend ────────────────────────────────────────────────

class ShareFileItem(BaseModel):
    file_name: str
    file_path: str
    file_source: Literal["project_file", "audio_file"]
    file_id: str

class ShareFilesRequest(BaseModel):
    user_id: str
    contact_id: Optional[str] = None
    recipient_email: str
    recipient_name: Optional[str] = None
    files: List[ShareFileItem]
    message: Optional[str] = None

@app.post("/share/files")
async def share_files(req: ShareFilesRequest):
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
        link_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
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

            file_links.append({
                "name": f.file_name,
                "url": signed_url,
                "source": f.file_source,
                "file_id": f.file_id,
            })

        if not file_links:
            file_names = [f.file_name for f in req.files]
            file_paths = [f.file_path for f in req.files]
            print(f"ERROR: No signed URLs generated. Files: {file_names}, Paths: {file_paths}")
            raise HTTPException(status_code=400, detail=f"Could not generate download links for any files. Check that file paths are valid.")

        # Fetch sender profile name
        sender_name = "Someone"
        try:
            profile = sb.table("profiles").select("full_name").eq("id", req.user_id).single().execute()
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
                  {fl['name']}
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
        # Use onboarding@resend.dev as from address until a custom domain is verified
        from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")
        try:
            email_response = resend.Emails.send({
                "from": from_address,
                "to": [req.recipient_email],
                "subject": f"{sender_name} shared files with you — Msanii",
                "html": html_body,
            })
        except Exception as e:
            print(f"Resend error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

        # Record shares in DB
        for fl in file_links:
            try:
                sb.table("file_shares").insert({
                    "user_id": req.user_id,
                    "contact_id": req.contact_id,
                    "recipient_email": req.recipient_email,
                    "recipient_name": req.recipient_name,
                    "file_name": fl["name"],
                    "file_source": fl["source"],
                    "file_id": fl["file_id"],
                    "message": req.message,
                    "link_expires_at": link_expires_at.isoformat(),
                    "status": "sent",
                }).execute()
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
