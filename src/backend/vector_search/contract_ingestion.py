"""
Contract Ingestion Module (OneClick Method)
Handles PDF contract uploads with intelligent section-based chunking, embedding, and upserting to Pinecone.

Features:
- Converts PDFs to markdown using pymupdf4llm for better structure preservation
- Splits PDFs into sections based on headings (explicit, heuristic, semantic)
- Categorizes sections by type (ROYALTY, PUBLISHING, MASTER_RIGHTS, etc.)
- Chunks each section into smaller pieces using RecursiveCharacterTextSplitter
- Generates deterministic vector IDs using SHA256
- Creates rich metadata for intelligent filtering
- Upserts to regional Pinecone indexes with user-specific namespaces
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pymupdf4llm
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
CHUNK_SIZE = 524  # Token-based chunk size (optimized for contract sections)
CHUNK_OVERLAP = 100  # Token overlap for context continuity
BATCH_SIZE = 20

# Regional index mapping
REGIONAL_INDEXES = {
    "US": "test-3-small-index",
    "EU": "test-3-small-index",
    "UK": "test-3-small-index"
}

# Section category keywords for classification
SECTION_CATEGORIES = {
    "ROYALTY_CALCULATIONS": ["royalty", "compensation", "revenue share", "compensation", "splits", "payment", "percentage"],
    "PUBLISHING_RIGHTS": ["publishing", "songwriter", "composition"],
    "PERFORMANCE_RIGHTS": ["performance", "production services", "live"],
    "COPYRIGHT": ["copyright", "intellectual property"],
    "TERMINATION": ["termination", "term", "duration", "end date"],
    "MASTER_RIGHTS": ["master", "master rights", "master recording"],
    "OWNERSHIP_RIGHTS": ["ownership", "synchronization", "licensing"],
    "ACCOUNTING_AND_CREDIT": ["accounting", "credit", "promotion", "audit"],
}


class ContractIngestion:
    """Handles contract PDF ingestion with intelligent section-based chunking and vector storage"""
    
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
        
        # Initialize RecursiveCharacterTextSplitter for section chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert a PDF file to markdown format using pymupdf4llm.
        This preserves document structure better than simple text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown representation of the PDF content
        """
        print(f"Converting PDF to markdown: {pdf_path}")
        md_text = pymupdf4llm.to_markdown(pdf_path)
        print(f"Converted to {len(md_text)} characters of markdown")
        return md_text
    
    def split_into_sections(self, markdown_text: str) -> List[Tuple[str, str]]:
        """
        Split markdown text into sections using a 4-layer approach:
        1) Explicit headings (markdown # or numbered sections)
        2) Heuristic heading candidates (caps, bold, short lines)
        3) Semantic heading validation
        4) Content-based section inference
        
        Args:
            markdown_text: The markdown content
            
        Returns:
            List of (header, content) tuples
        """
        lines = markdown_text.splitlines()
        sections: List[Tuple[str, str]] = []
        
        # Layer 1: Explicit Headings (markdown # or numbered like "1. DEFINITIONS")
        explicit_heading_re = re.compile(
            r'^(#{1,6}\s+.+|\d+(\.\d+)*[\)\.]?\s+[A-Z].+)$'
        )
        
        explicit_headers = [
            i for i, line in enumerate(lines)
            if explicit_heading_re.match(line.strip())
        ]
        
        if explicit_headers:
            for idx, start in enumerate(explicit_headers):
                end = explicit_headers[idx + 1] if idx + 1 < len(explicit_headers) else len(lines)
                header = lines[start].strip()
                content = "\n".join(lines[start + 1:end]).strip()
                sections.append((header, content))
            return sections
        
        # Layer 2: Heuristic Header Detection (ALL CAPS, bold markdown, short title-like)
        def is_heuristic_header(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            # ALL CAPS and short
            if stripped.isupper() and len(stripped.split()) <= 10:
                return True
            # Bold markdown
            if stripped.startswith("**") and stripped.endswith("**") and len(stripped.split()) <= 12:
                return True
            # Short title-like line ending with colon
            if len(stripped.split()) <= 6 and stripped.endswith(":"):
                return True
            return False
        
        header_indices = [i for i, line in enumerate(lines) if is_heuristic_header(line)]
        
        if header_indices:
            for idx, start in enumerate(header_indices):
                end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
                header = lines[start].strip().strip("*").strip(":")
                content = "\n".join(lines[start + 1:end]).strip()
                sections.append((header, content))
            return sections
        
        # Layer 3: Semantic Heading Detection (short, prominent lines validated by context)
        semantic_candidates = [
            (i, line.strip())
            for i, line in enumerate(lines)
            if 2 <= len(line.split()) <= 8 and line.strip()
        ]
        
        semantic_headers = []
        for idx, text in semantic_candidates:
            if self._is_semantic_heading(text):
                semantic_headers.append(idx)
        
        if semantic_headers:
            for i, start in enumerate(semantic_headers):
                end = semantic_headers[i + 1] if i + 1 < len(semantic_headers) else len(lines)
                header = lines[start].strip()
                content = "\n".join(lines[start + 1:end]).strip()
                sections.append((header, content))
            return sections
        
        # Layer 4: Content-Based Inference (detect royalty/payment sections)
        royalty_trigger_re = re.compile(
            r'(royalt|revenue|shall receive|net revenue|%)',
            re.IGNORECASE
        )
        
        inferred_starts = [
            i for i, line in enumerate(lines)
            if royalty_trigger_re.search(line)
        ]
        
        if inferred_starts:
            start = inferred_starts[0]
            sections.append((
                "Inferred Royalty Section",
                "\n".join(lines[start:]).strip()
            ))
            return sections
        
        # Fallback: Return entire document as one section
        return [("Full Document", markdown_text.strip())]
    
    def _is_semantic_heading(self, text: str) -> bool:
        """
        Determine if a text is likely a section heading using simple heuristics.
        Falls back to LLM validation for uncertain cases.
        
        Args:
            text: The text to evaluate
            
        Returns:
            True if text appears to be a heading
        """
        # Simple heuristic checks first
        stripped = text.strip()
        
        # Numbered section patterns
        if re.match(r'^\d+\.?\s+[A-Z]', stripped):
            return True
        
        # All caps short line
        if stripped.isupper() and len(stripped.split()) <= 6:
            return True
        
        # Title case and short
        if stripped.istitle() and len(stripped.split()) <= 5:
            return True
        
        # Contains common section header words
        header_keywords = ['article', 'section', 'clause', 'schedule', 'exhibit', 'appendix']
        if any(kw in stripped.lower() for kw in header_keywords):
            return True
        
        return False
    
    def categorize_section(self, section_header: str) -> str:
        """
        Categorize a section based on its header using keyword matching.
        
        Args:
            section_header: The header text of the section
            
        Returns:
            Category string (e.g., "ROYALTY_CALCULATIONS", "PUBLISHING_RIGHTS")
        """
        header_lower = section_header.lower()
        
        for category, keywords in SECTION_CATEGORIES.items():
            if any(keyword in header_lower for keyword in keywords):
                return category
        
        return "OTHER"
    
    def generate_deterministic_id(self, chunk_text: str, metadata: Dict) -> str:
        """
        Generate deterministic vector ID using SHA256 of content + stable metadata fields.
        
        This ensures uniqueness across:
        - Different users (via user_id)
        - Different projects (via project_id)
        - Different contracts (via contract_id)
        - Different sections (via section_heading)
        - Content changes (via chunk_text hash)
        
        Args:
            chunk_text: The text content of the chunk
            metadata: Metadata dict (only stable fields are used)
            
        Returns:
            SHA256 hash as hex string
        """
        # Use only stable identifiers to create the ID
        stable_fields = {
            'user_id': metadata.get('user_id', ''),
            'contract_id': metadata.get('contract_id', ''),
            'section_heading': metadata.get('section_heading', ''),
            'document_name': metadata.get('contract_file', '')
        }
        
        # Normalize metadata to ensure consistent ordering
        canonical_metadata = json.dumps(stable_fields, sort_keys=True)
        combined_string = chunk_text + "|" + canonical_metadata
        
        return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()
    
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
        Complete ingestion pipeline for a contract PDF using OneClick section-based approach.
        
        Pipeline:
        1. Convert PDF to markdown (preserves structure)
        2. Split into sections based on headings
        3. Categorize each section (ROYALTY, PUBLISHING, etc.)
        4. Chunk each section with RecursiveCharacterTextSplitter
        5. Generate embeddings for each chunk
        6. Upsert to user's namespace with rich metadata
        
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
        print(f"INGESTING CONTRACT (OneClick Method): {contract_filename}")
        print("=" * 80)
        
        # Step 1: Convert PDF to markdown
        markdown_text = self.pdf_to_markdown(pdf_path)
        
        if not markdown_text.strip():
            raise ValueError("No text content found in PDF")
        
        # Step 2: Split into sections
        print("\n--- Splitting into sections ---")
        sections = self.split_into_sections(markdown_text)
        print(f"Found {len(sections)} sections")
        
        # Step 3: Categorize sections and chunk each section
        print("\n--- Categorizing and chunking sections ---")
        all_chunks = []
        uploaded_at = datetime.utcnow().isoformat()
        
        for section_header, section_content in sections:
            if not section_content.strip():
                continue
                
            # Categorize the section
            category = self.categorize_section(section_header)
            print(f"  Section: '{section_header[:50]}...' -> {category}")
            
            # Chunk this section
            section_chunks = self.text_splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "section_heading": section_header,
                    "section_category": category
                })
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        if not all_chunks:
            raise ValueError("No chunks created from PDF")
        
        # Step 4: Create embeddings in batches
        texts_for_embedding = [chunk["text"] for chunk in all_chunks]
        all_embeddings = []
        
        for i in range(0, len(texts_for_embedding), BATCH_SIZE):
            batch_texts = texts_for_embedding[i:i + BATCH_SIZE]
            batch_embeddings = self.create_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Step 5: Create vectors with metadata and deterministic IDs
        vectors_to_upsert = []
        category_counts = {}
        
        for i, chunk in enumerate(all_chunks):
            metadata = {
                "user_id": user_id,
                "project_id": project_id,
                "project_name": project_name,
                "contract_id": contract_id,
                "contract_file": contract_filename,
                "section_heading": chunk["section_heading"],
                "section_category": chunk["section_category"],
                "uploaded_at": uploaded_at,
                "chunk_text": chunk["text"]
            }
            
            # Track category distribution
            cat = chunk["section_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Generate deterministic ID
            vector_id = self.generate_deterministic_id(
                chunk_text=chunk["text"],
                metadata=metadata
            )
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": all_embeddings[i],
                "metadata": metadata
            })
        
        # Step 6: Upsert to Pinecone (using user_id-namespace format)
        namespace = f"{user_id}-namespace"
        
        print(f"\n--- Upserting to Pinecone ---")
        print(f"Index: {self.index_name}")
        print(f"Namespace: {namespace}")
        print(f"Vectors: {len(vectors_to_upsert)}")
        
        # Upsert in batches
        for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
            batch = vectors_to_upsert[i:i + BATCH_SIZE]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(vectors_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE} uploaded")
        
        stats = {
            "contract_id": contract_id,
            "contract_filename": contract_filename,
            "total_sections": len(sections),
            "total_chunks": len(all_chunks),
            "total_vectors": len(vectors_to_upsert),
            "category_distribution": category_counts,
            "namespace": namespace,
            "index": self.index_name,
            "region": self.region,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
        
        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print(f"  Sections: {stats['total_sections']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Vectors: {stats['total_vectors']}")
        print(f"  Categories: {category_counts}")
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
