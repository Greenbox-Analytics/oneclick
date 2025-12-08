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
import tiktoken
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
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 600
BATCH_SIZE = 100

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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by text-embedding-3-small
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page information
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with page_number and text
        """
        print(f"Extracting text from: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # Only include pages with content
                pages.append({
                    "page_number": page_num,
                    "text": text
                })
        
        doc.close()
        print(f"Extracted {len(pages)} pages from PDF")
        return pages
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text_by_tokens(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk text into 300-600 token segments while preserving page information
        
        Args:
            pages: List of page dicts with page_number and text
            
        Returns:
            List of chunk dicts with text, page_number, and chunk_index
        """
        chunks = []
        chunk_index = 0
        
        for page_data in pages:
            page_num = page_data["page_number"]
            text = page_data["text"]
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_tokens = self.count_tokens(para)
                
                # If adding this paragraph exceeds max tokens, save current chunk
                if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current_chunk:
                    if current_tokens >= MIN_CHUNK_TOKENS:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page_number": page_num,
                            "chunk_index": chunk_index,
                            "token_count": current_tokens
                        })
                        chunk_index += 1
                        current_chunk = para + "\n\n"
                        current_tokens = para_tokens
                    else:
                        # Current chunk too small, add paragraph anyway
                        current_chunk += para + "\n\n"
                        current_tokens += para_tokens
                else:
                    current_chunk += para + "\n\n"
                    current_tokens += para_tokens
            
            # Save remaining chunk from this page if it meets minimum
            if current_chunk.strip() and current_tokens >= MIN_CHUNK_TOKENS:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "token_count": current_tokens
                })
                chunk_index += 1
            elif current_chunk.strip():
                # If last chunk is too small, append to previous chunk if exists
                if chunks:
                    chunks[-1]["text"] += "\n\n" + current_chunk.strip()
                    chunks[-1]["token_count"] += current_tokens
                else:
                    # First chunk, keep it even if small
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "token_count": current_tokens
                    })
                    chunk_index += 1
        
        print(f"Created {len(chunks)} chunks (token range: {MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS})")
        return chunks
    
    def generate_deterministic_id(self, chunk_text: str, contract_id: str, 
                                  page_number: int, chunk_index: int, 
                                  metadata: Dict) -> str:
        """
        Generate deterministic vector ID using SHA256
        
        Args:
            chunk_text: The text content of the chunk
            contract_id: UUID of the contract
            page_number: Page number in PDF
            chunk_index: Index of chunk
            metadata: Full metadata dict
            
        Returns:
            SHA256 hash as hex string
        """
        # Create a deterministic string from all inputs
        id_components = {
            "chunk_text": chunk_text,
            "contract_id": contract_id,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "metadata": json.dumps(metadata, sort_keys=True)
        }
        
        id_string = json.dumps(id_components, sort_keys=True)
        return hashlib.sha256(id_string.encode()).hexdigest()
    
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
        pages = self.extract_text_from_pdf(pdf_path)
        
        if not pages:
            raise ValueError("No text content found in PDF")
        
        # Step 2: Chunk text by tokens
        chunks = self.chunk_text_by_tokens(pages)
        
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
                "contract_file": contract_filename,
                "region": self.region,
                "uploaded_at": uploaded_at,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "chunk_text": chunk["text"]  # Store for retrieval
            }
            
            vector_id = self.generate_deterministic_id(
                chunk_text=chunk["text"],
                contract_id=contract_id,
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                metadata=metadata
            )
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": all_embeddings[i],
                "metadata": metadata
            })
        
        # Step 5: Upsert to Pinecone (using user_id as namespace for isolation)
        namespace = f"{user_id}_data"
        
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
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "total_vectors": len(vectors_to_upsert),
            "namespace": namespace,
            "index": self.index_name,
            "region": self.region
        }
        
        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print(f"  Pages: {stats['total_pages']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Vectors: {stats['total_vectors']}")
        print("=" * 80)
        
        return stats
    
    def delete_contract(self, user_id: str, contract_id: str) -> Dict:
        """
        Delete all vectors for a specific contract
        
        Args:
            user_id: UUID of the user
            contract_id: UUID of the contract to delete
            
        Returns:
            Dict with deletion statistics
        """
        namespace = f"{user_id}_data"
        
        print(f"Deleting contract {contract_id} from namespace {namespace}")
        
        # Query to get all vector IDs for this contract
        # Note: Pinecone delete by metadata filter is available in some plans
        # For now, we'll use the deterministic ID pattern
        
        # Alternative: Use delete with filter (if available in your Pinecone plan)
        try:
            self.index.delete(
                namespace=namespace,
                filter={"contract_id": contract_id}
            )
            print(f"Deleted all vectors for contract {contract_id}")
            return {"status": "success", "contract_id": contract_id}
        except Exception as e:
            print(f"Error deleting contract: {e}")
            return {"status": "error", "error": str(e)}


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
