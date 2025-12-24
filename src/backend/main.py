from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import pandas as pd
import io
import uuid
import time
import tempfile
from dotenv import load_dotenv
import sys
from pathlib import Path
from vector_search.contract_chatbot import ContractChatbot
from vector_search.contract_ingestion import ContractIngestion

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Adjust this to match your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase Client
url: str = os.getenv("VITE_SUPABASE_URL")
key: str = os.getenv("VITE_SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError(
        "Missing required environment variables: VITE_SUPABASE_URL and/or VITE_SUPABASE_SECRET_KEY. "
        "Please create a .env file in the backend directory with these variables."
    )

supabase: Client = create_client(url, key)

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

# Zoe Chatbot Models
class ZoeAskRequest(BaseModel):
    query: str
    project_id: str
    contract_ids: Optional[List[str]] = None  # Changed to support multiple contracts
    user_id: str

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

# Initialize Zoe chatbot and contract ingestion (singletons)
zoe_chatbot = None
contract_ingestion = None

def get_zoe_chatbot():
    """Get or create Zoe chatbot instance"""
    global zoe_chatbot
    if zoe_chatbot is None:
        zoe_chatbot = ContractChatbot(region="US")
    return zoe_chatbot

def get_contract_ingestion():
    """Get or create contract ingestion instance"""
    global contract_ingestion
    if contract_ingestion is None:
        contract_ingestion = ContractIngestion(region="US")
    return contract_ingestion

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Msanii AI Backend is running"}

@app.get("/artists")
async def get_artists():
    """
    Fetch all artists.
    """
    try:
        response = supabase.table("artists").select("*").execute()
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
        response = supabase.table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{project_id}")
async def get_project_files(project_id: str):
    """
    Fetch files associated with a specific project.
    """
    try:
        response = supabase.table("project_files").select("*").eq("project_id", project_id).execute()
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
        projects_res = supabase.table("projects").select("id").eq("artist_id", artist_id).execute()
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


        files_res = supabase.table("project_files").select("*").in_("project_id", project_ids).eq("folder_category", db_category).execute()
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
            res = supabase.table("projects").select("id").eq("artist_id", artist_id).eq("name", "General Uploads").execute()
            if res.data:
                project_id = res.data[0]['id']
            else:
                # Create "General Uploads" project
                new_proj = supabase.table("projects").insert({
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
        storage_res = supabase.storage.from_("project-files").upload(file_path, file_content)
        
        # Get public URL
        file_url = supabase.storage.from_("project-files").get_public_url(file_path)
        
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
        
        db_res = supabase.table("project_files").insert(db_record).execute()
        
        return {
            "status": "success", 
            "file": db_res.data[0]
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate", response_model=RoyaltyResults)
async def calculate_royalties(
    contract_files: List[str] = Form(...), # List of file paths or IDs
    royalty_files: List[str] = Form(...)   # List of file paths or IDs
):
    """
    Mock calculation endpoint. 
    In the future, this will:
    1. Download the files from Supabase.
    2. Parse the Excel/CSV royalty statement.
    3. Apply contract logic.
    4. Return real results.
    """
    
    # Mock processing delay
    time.sleep(1) 
    
    # Returning the dummy data structure
    return {
        "songTitle": "Midnight Dreams",
        "totalContributors": 4,
        "totalRevenue": 125000.00,
        "breakdown": [
            {
                "songName": "Midnight Dreams",
                "contributorName": "Luna Rivers",
                "role": "Artist",
                "royaltyPercentage": 45.0,
                "amount": 56250.00
            },
            {
                "songName": "Midnight Dreams",
                "contributorName": "Alex Martinez",
                "role": "Producer",
                "royaltyPercentage": 30.0,
                "amount": 37500.00
            },
            {
                "songName": "Midnight Dreams",
                "contributorName": "Sarah Chen",
                "role": "Songwriter",
                "royaltyPercentage": 20.0,
                "amount": 25000.00
            },
            {
                "songName": "Midnight Dreams",
                "contributorName": "Mike Johnson",
                "role": "Featured Artist",
                "royaltyPercentage": 5.0,
                "amount": 6250.00
            }
        ]
    }

# --- Zoe AI Chatbot Endpoints ---

@app.get("/projects")
async def get_all_projects():
    """
    Fetch all projects (for Zoe project selection).
    """
    try:
        response = supabase.table("projects").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artists/{artist_id}/projects")
async def get_artist_projects(artist_id: str):
    """
    Fetch projects for a specific artist (for Zoe artist-based filtering).
    """
    try:
        response = supabase.table("projects").select("*").eq("artist_id", artist_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/contracts")
async def get_project_contracts(project_id: str):
    """
    Fetch contracts (PDF files) for a specific project.
    """
    try:
        response = supabase.table("project_files").select("*").eq("project_id", project_id).eq("folder_category", "contract").execute()
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
        project_res = supabase.table("projects").select("name").eq("id", project_id).execute()
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
        storage_res = supabase.storage.from_("project-files").upload(file_path, file_content)
        
        # Get public URL
        file_url = supabase.storage.from_("project-files").get_public_url(file_path)
        
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
        
        db_res = supabase.table("project_files").insert(db_record).execute()
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
        contract_res = supabase.table("project_files").select("*").eq("id", contract_id).execute()
        
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
                supabase.storage.from_("project-files").remove([contract["file_path"]])
            except Exception as e:
                print(f"Warning: Failed to delete file from storage: {e}")
        
        # 4. Delete from Database
        supabase.table("project_files").delete().eq("id", contract_id).execute()
        
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
    Zoe AI Chatbot endpoint - Ask questions about contracts.
    
    Rules:
    1. Always filters by user_id and project_id
    2. If contract_ids are provided, filters by specific contracts and uses top_k=3
    3. If no contract_ids, searches all contracts in project with top_k=8
    4. Only answers if highest similarity ≥ 0.75
    5. Returns answer with source citations
    """
    try:
        # Get Zoe chatbot instance
        chatbot = get_zoe_chatbot()
        
        # Determine top_k based on whether specific contracts are selected
        # If specific contracts selected: top_k=3 (focused search)
        # If project-wide search: top_k=8 (broader search)
        top_k = 3 if request.contract_ids and len(request.contract_ids) > 0 else 8
        
        # Ask the question
        if request.contract_ids and len(request.contract_ids) > 0:
            # Contract-specific question(s)
            # For multiple contracts, we'll query each and combine results
            # For now, use the first contract_id (can be enhanced to handle multiple)
            result = chatbot.ask_contract(
                query=request.query,
                user_id=request.user_id,
                project_id=request.project_id,
                contract_id=request.contract_ids[0],  # Use first contract for now
                top_k=top_k
            )
        else:
            # Project-wide question
            result = chatbot.ask_project(
                query=request.query,
                user_id=request.user_id,
                project_id=request.project_id,
                top_k=top_k
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
            highest_score=result.get("highest_score")
        )
        
    except Exception as e:
        print(f"Error in Zoe chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Zoe encountered an error: {str(e)}")
