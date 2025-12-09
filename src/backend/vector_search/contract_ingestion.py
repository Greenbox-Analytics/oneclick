"""
Contract Ingestion Module
Handles PDF contract uploads, chunking, embedding, and upserting to Pinecone.

Features:
- Extracts text from PDF contracts
- Chunks text into 300-600 token segments
- Generates deterministic vector IDs using SHA256
- Creates rich metadata for filtering
- Upserts to regional Pinecone indexes
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1024  # Character-based chunk size
CHUNK_OVERLAP = 154  # Character overlap
BATCH_SIZE = 20

# Regional index mapping
REGIONAL_INDEXES = {
    "US": "test-3-small-index",
    "EU": "test-3-small-index",
    "UK": "test-3-small-index"
}


class ContractIngestion:
    """Handles contract PDF ingestion and vector storage"""
    
    def __init__(self, region: str = "US"):
        """
        Initialize the contract ingestion handler
        
        Args:
            region: Region code (US, EU, UK) - determines which index to use
        """
        if region not in REGIONAL_INDEXES:
            raise ValueError(f"Invalid region: {region}. Must be one of {list(REGIONAL_INDEXES.keys())}")
        
        self.region = region
        self.index_name = REGIONAL_INDEXES[region]
        self.index = pc.Index(self.index_name)
        
        # Initialize RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Full text content from PDF
        """
        print(f"Extracting text from: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        page_count = len(doc)
        
        for page in doc:
            text += page.get_text() + "\n"
        
        doc.close()
        print(f"Extracted text from {page_count} pages")
        return text
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk text using RecursiveCharacterTextSplitter
        
        Args:
            text: Full text content from PDF
            
        Returns:
            List of chunk dicts with text
        """
        # Use LangChain's RecursiveCharacterTextSplitter
        chunks = self.text_splitter.split_text(text)
        
        print(f"Created {len(chunks)} chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")
        
        # Convert to dict format
        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                "text": chunk
            })
        
        return chunk_dicts
    
    def generate_deterministic_id(self, chunk_text: str, metadata: Dict) -> str:
        """
        Generate deterministic vector ID using SHA256 of content + metadata
        
        This ensures uniqueness across:
        - Different users (via user_id in metadata)
        - Different projects (via project_id in metadata)
        - Different contracts (via contract_id in metadata)
        - Different versions (content or metadata changes = new hash)
        
        Args:
            chunk_text: The text content of the chunk (page_content)
            metadata: Full metadata dict with all identifying information
            
        Returns:
            SHA256 hash as hex string
        """
        # Create canonical JSON of metadata (sorted keys for consistency)
        canonical_metadata = json.dumps(metadata, sort_keys=True)
        
        # Combine page content + canonical metadata
        combined_string = chunk_text + canonical_metadata
        
        # Generate SHA256 hash
        return hashlib.sha256(combined_string.encode()).hexdigest()
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using OpenAI text-embedding-3-small
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"Creating embeddings for {len(texts)} chunks...")
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def ingest_contract(self, 
                       pdf_path: str,
                       user_id: str,
                       project_id: str,
                       project_name: str,
                       contract_id: str,
                       contract_filename: str) -> Dict:
        """
        Complete ingestion pipeline for a contract PDF
        
        Args:
            pdf_path: Path to the PDF file
            user_id: UUID of the user
            project_id: UUID of the project
            project_name: Name of the project
            contract_id: UUID of the contract (from project_files table)
            contract_filename: Original filename of the contract
            
        Returns:
            Dict with ingestion statistics
        """
        print("\n" + "=" * 80)
        print(f"INGESTING CONTRACT: {contract_filename}")
        print("=" * 80)
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError("No text content found in PDF")
        
        # Step 2: Chunk text using RecursiveCharacterTextSplitter
        chunks = self.chunk_text(text)
        
        if not chunks:
            raise ValueError("No chunks created from PDF")
        
        # Step 3: Prepare vectors with metadata
        vectors_to_upsert = []
        texts_for_embedding = [chunk["text"] for chunk in chunks]
        
        # Create embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts_for_embedding), BATCH_SIZE):
            batch_texts = texts_for_embedding[i:i + BATCH_SIZE]
            batch_embeddings = self.create_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Step 4: Create vectors with metadata and deterministic IDs
        uploaded_at = datetime.utcnow().isoformat()
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "user_id": user_id,
                "project_id": project_id,
                "project_name": project_name,
                "contract_id": contract_id,
                "contract_file": contract_filename,  # Use contract_file for consistency
                "uploaded_at": uploaded_at
            }
            
            # Generate deterministic ID from content + metadata
            vector_id = self.generate_deterministic_id(
                chunk_text=chunk["text"],
                metadata=metadata
            )
            
            # Add chunk_text to metadata for retrieval (not used in ID generation order)
            metadata["chunk_text"] = chunk["text"]
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": all_embeddings[i],
                "metadata": metadata
            })
        
        # Step 5: Upsert to Pinecone (using user_id-namespace format)
        namespace = f"{user_id}-namespace"
        
        print(f"\nUpserting {len(vectors_to_upsert)} vectors to index '{self.index_name}'")
        print(f"Namespace: {namespace}")
        
        # Upsert in batches
        for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
            batch = vectors_to_upsert[i:i + BATCH_SIZE]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"Uploaded batch {i // BATCH_SIZE + 1}/{(len(vectors_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        stats = {
            "contract_id": contract_id,
            "contract_filename": contract_filename,
            "total_chunks": len(chunks),
            "total_vectors": len(vectors_to_upsert),
            "namespace": namespace,
            "index": self.index_name,
            "region": self.region,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
        
        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Vectors: {stats['total_vectors']}")
        print(f"  Chunk Size: {CHUNK_SIZE} chars")
        print(f"  Chunk Overlap: {CHUNK_OVERLAP} chars")
        print("=" * 80)
        
        return stats
    
    def delete_contract(self, user_id: str, contract_id: str) -> Dict:
        """
        Delete all vectors for a specific contract using metadata filter
        
        The vector DB is the authoritative source - no pickle files or local dictionaries.
        
        Args:
            user_id: UUID of the user
            contract_id: UUID of the contract to delete
            
        Returns:
            Dict with deletion statistics
        """
        namespace = f"{user_id}-namespace"
        
        print("\n" + "=" * 80)
        print(f"DELETING CONTRACT: {contract_id}")
        print("=" * 80)
        print(f"Namespace: {namespace}")
        
        try:
            # Delete all vectors matching the contract_id filter
            self.index.delete(
                namespace=namespace,
                filter={"contract_id": contract_id}
            )
            
            print(f"✓ Successfully deleted all vectors for contract {contract_id}")
            print("=" * 80)
            
            return {
                "status": "success",
                "contract_id": contract_id,
                "namespace": namespace
            }
        except Exception as e:
            print(f"✗ Error deleting contract: {e}")
            print("=" * 80)
            return {
                "status": "error",
                "contract_id": contract_id,
                "error": str(e)
            }
    
    def replace_contract(self, 
                        pdf_path: str,
                        user_id: str,
                        project_id: str,
                        project_name: str,
                        contract_id: str,
                        contract_filename: str) -> Dict:
        """
        Replace an existing contract by deleting old chunks and uploading new ones
        
        Args:
            pdf_path: Path to the PDF file
            user_id: UUID of the user
            project_id: UUID of the project
            project_name: Name of the project
            contract_id: UUID of the contract (from project_files table)
            contract_filename: Original filename of the contract
            
        Returns:
            Dict with replacement statistics
        """
        print("\n" + "=" * 80)
        print(f"REPLACING CONTRACT: {contract_filename}")
        print("=" * 80)
        
        # Step 1: Delete existing contract chunks
        delete_result = self.delete_contract(user_id, contract_id)
        
        if delete_result["status"] != "success":
            return {
                "status": "error",
                "error": f"Failed to delete existing contract: {delete_result.get('error')}"
            }
        
        # Step 2: Ingest new contract
        ingest_result = self.ingest_contract(
            pdf_path=pdf_path,
            user_id=user_id,
            project_id=project_id,
            project_name=project_name,
            contract_id=contract_id,
            contract_filename=contract_filename
        )
        
        print("\n" + "=" * 80)
        print("CONTRACT REPLACEMENT COMPLETE")
        print("=" * 80)
        
        return {
            "status": "success",
            "operation": "replace",
            **ingest_result
        }


# Example usage
if __name__ == "__main__":
    # Example: Ingest a sample contract
    ingestion = ContractIngestion(region="US")
    
    # Sample data (replace with actual values)
    sample_pdf = Path(__file__).parent.parent / "sample_docs" / "Scenario 2 ' Home' - Romes_Lebron Contract.pdf"
    
    if sample_pdf.exists():
        stats = ingestion.ingest_contract(
            pdf_path=str(sample_pdf),
            user_id="test-user-123",
            project_id="test-project-456",
            project_name="Home - Romes",
            contract_id="test-contract-789",
            contract_filename="Romes_Lebron Contract.pdf"
        )
        
        print("\nIngestion Stats:")
        print(json.dumps(stats, indent=2))
    else:
        print(f"Sample PDF not found at {sample_pdf}")
