from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import io
import uuid
import time
from dotenv import load_dotenv
import sys
from pathlib import Path
from vector_search.contract_chatbot import ContractChatbot

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
    page_number: int
    score: float
    project_name: str

class ZoeAskResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[ZoeSource]
    search_results_count: int
    highest_score: Optional[float] = None

# Initialize Zoe chatbot (singleton)
zoe_chatbot = None

def get_zoe_chatbot():
    """Get or create Zoe chatbot instance"""
    global zoe_chatbot
    if zoe_chatbot is None:
        zoe_chatbot = ContractChatbot(region="US")
    return zoe_chatbot

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
        
        # Format response
        return ZoeAskResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=[
                ZoeSource(
                    contract_file=source["contract_file"],
                    page_number=source["page_number"],
                    score=source["score"],
                    project_name=source["project_name"]
                )
                for source in result.get("sources", [])
            ],
            search_results_count=result.get("search_results_count", 0),
            highest_score=result.get("highest_score")
        )
        
    except Exception as e:
        print(f"Error in Zoe chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Zoe encountered an error: {str(e)}")
