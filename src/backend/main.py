from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import io
import uuid
import time
import tempfile
from dotenv import load_dotenv
import sys
from pathlib import Path
import json
import asyncio

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
class ExtractedContractData(BaseModel):
    royalty_splits: Optional[List[Dict]] = None
    payment_terms: Optional[str] = None
    parties: Optional[List[str]] = None
    advances: Optional[str] = None
    term_length: Optional[str] = None

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
async def get_artists(user_id: Optional[str] = None):
    """
    Fetch artists. If user_id is provided, filter by that user's artists only.
    """
    try:
        query = get_supabase_client().table("artists").select("*")
        
        # Filter by user_id if provided
        if user_id:
            query = query.eq("user_id", user_id)
        
        response = query.execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artists/{artist_id}")
async def get_artist_by_id(artist_id: str):
    """
    Fetch a single artist by ID with all their details.
    """
    try:
        response = get_supabase_client().table("artists").select("*").eq("id", artist_id).single().execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{artist_id}")
async def get_projects(artist_id: str):
    """
    Fetch projects for a specific artist.
    """
    try:
        # artist_id is UUID in DB, but passed as string here
        response = get_supabase_client().table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProjectCreateRequest(BaseModel):
    artist_id: str
    name: str
    description: Optional[str] = None

@app.post("/projects")
async def create_project(project: ProjectCreateRequest):
    """
    Create a new project for an artist.
    """
    try:
        res = supabase.table("projects").insert({
            "artist_id": project.artist_id,
            "name": project.name,
            "description": project.description
        }).execute()
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{project_id}")
async def get_project_files(project_id: str):
    """
    Fetch files associated with a specific project.
    """
    try:
        response = get_supabase_client().table("project_files").select("*").eq("project_id", project_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/artist/{artist_id}/category/{category}")
async def get_artist_files_by_category(artist_id: str, category: str):
    """
    Fetch all files for an artist filtered by category (across all projects or independent).
    Note: Since files are linked to projects, we might need to join tables or filter differently.
    This implementation finds all projects for the artist, then finds files in those projects with the category.
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    artist_id: str = Form(...),
    category: str = Form(...), # 'contract' or 'royalty_statement'
    project_id: Optional[str] = Form(None)
):
    """
    Uploads a file to Supabase Storage and creates a record in project_files.
    If project_id is not provided, it requires logic to handle 'orphaned' files or create a default project.
    For this implementation, we'll enforce project_id or create a 'General' project if missing.
    """
    try:
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
        
        db_res = get_supabase_client().table("project_files").insert(db_record).execute()
        
        return {
            "status": "success", 
            "file": db_res.data[0]
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Zoe AI Chatbot Endpoints ---

@app.get("/projects")
async def get_all_projects():
    """
    Fetch all projects (for Zoe project selection).
    """
    try:
        response = get_supabase_client().table("projects").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artists/{artist_id}/projects")
async def get_artist_projects(artist_id: str):
    """
    Fetch projects for a specific artist (for Zoe artist-based filtering).
    """
    try:
        response = get_supabase_client().table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/contracts")
async def get_project_contracts(project_id: str):
    """
    Fetch contracts (PDF files) for a specific project.
    """
    try:
        response = get_supabase_client().table("project_files").select("*").eq("project_id", project_id).eq("folder_category", "contract").execute()
        return response.data
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
        
        # Get project details
        project_res = get_supabase_client().table("projects").select("name").eq("id", project_id).execute()
        if not project_res.data:
            raise HTTPException(status_code=404, detail="Project not found")
        
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
        
        db_res = get_supabase_client().table("project_files").insert(db_record).execute()
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
                context=context_dict
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
                context=context_dict
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
                context=context_dict
            )
        
        # Format response - filter out sources with missing required fields
        valid_sources = []
        for source in result.get("sources", []):
            zoe_source = ZoeSource.from_dict(source)
            if zoe_source is not None:
                valid_sources.append(zoe_source)
        
        return ZoeAskResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=valid_sources,
            search_results_count=result.get("search_results_count", 0),
            highest_score=result.get("highest_score"),
            session_id=session_id,
            show_quick_actions=result.get("show_quick_actions", False),
            answered_from=result.get("answered_from")
        )
        
    except Exception as e:
        print(f"Error in Zoe chatbot: {str(e)}")
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
    contract_id: str
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

@app.get("/oneclick/calculate-royalties-stream")
async def oneclick_calculate_royalties_stream(
    contract_id: str,
    user_id: str,
    project_id: str,
    royalty_statement_file_id: str
):
    """
    OneClick Royalty Calculation with Server-Sent Events (SSE) for real-time progress updates.
    
    This endpoint streams progress updates to the client as the calculation proceeds:
    - Downloading files
    - Extracting contract data (parties, works, royalties, summary)
    - Processing royalty statement
    - Calculating payments
    
    Args:
        request: Contains contract_id, user_id, project_id, and royalty_statement_file_id
        
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
            
            # Step 2: Download contract
            yield f"data: {json.dumps({'type': 'status', 'message': 'Downloading contract...', 'progress': 25, 'stage': 'downloading'})}\n\n"
            
            contract_res = get_supabase_client().table("project_files").select("*").eq("id", contract_id).execute()
            if not contract_res.data:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Contract file not found'})}\n\n"
                return
            
            contract_file = contract_res.data[0]
            contract_data = get_supabase_client().storage.from_("project-files").download(contract_file["file_path"])
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_contract:
                tmp_contract.write(contract_data)
                contract_path = tmp_contract.name
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Contract downloaded', 'progress': 30, 'stage': 'downloading'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 3: Extract contract data with progress updates
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting parties from contract...', 'progress': 35, 'stage': 'extracting_parties'})}\n\n"
            await asyncio.sleep(0.5)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting works from contract...', 'progress': 50, 'stage': 'extracting_works'})}\n\n"
            await asyncio.sleep(0.5)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting royalty splits...', 'progress': 65, 'stage': 'extracting_royalty'})}\n\n"
            await asyncio.sleep(0.5)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating contract summary...', 'progress': 75, 'stage': 'extracting_summary'})}\n\n"
            await asyncio.sleep(0.3)
            
            # Calculate payments
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing royalty statement...', 'progress': 80, 'stage': 'processing'})}\n\n"
            
            payments = calculate_royalty_payments(
                contract_path=contract_path,
                statement_path=statement_path,
                user_id=user_id,
                contract_id=contract_id
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
    1. Retrieve streamingroyalty splits from selected contract using vector search
    2. Download royalty statement from Supabase
    3. Calculate payments using royalty_calculator.py methods
    4. Save results to Excel and upload to Supabase
    5. Return payment breakdown
    
    Args:
        request: Contains contract_id, user_id, project_id, and royalty_statement_file_id
        
    Returns:
        Payment breakdown and Excel file URL
    """
    try:
        print(f"\n{'='*80}")
        print("ONECLICK ROYALTY CALCULATION")
        print(f"{'='*80}")
        print(f"Contract ID: {request.contract_id}")
        print(f"User ID: {request.user_id}")
        print(f"Project ID: {request.project_id}")
        print(f"Royalty Statement File ID: {request.royalty_statement_file_id}")
        
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
        
        # Step 3: Get contract file for parsing
        print("\n--- Step 2: Downloading Contract File ---")
        contract_res = get_supabase_client().table("project_files").select("*").eq("id", request.contract_id).execute()
        
        if not contract_res.data:
            raise HTTPException(status_code=404, detail="Contract file not found")
        
        contract_file = contract_res.data[0]
        contract_file_path = contract_file["file_path"]
        
        # Download contract from Supabase storage
        contract_data = get_supabase_client().storage.from_("project-files").download(contract_file_path)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_contract:
            tmp_contract.write(contract_data)
            contract_path = tmp_contract.name
        
        print(f"Downloaded contract: {contract_file['file_name']}")
        
        try:
            # Step 4: Calculate payments using helper function
            print("\n--- Step 3: Calculating Royalty Payments ---")
            
            # Use helper function from helpers.py
            payments = calculate_royalty_payments(
                contract_path=contract_path,
                statement_path=statement_path,
                user_id=request.user_id,
                contract_id=request.contract_id
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
            if os.path.exists(contract_path):
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
