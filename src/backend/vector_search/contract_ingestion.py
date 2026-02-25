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
import json
from typing import Dict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from vector_search.helpers import (
    pdf_to_markdown,
    split_into_sections,
    categorize_section,
    create_embeddings,
    generate_deterministic_id,
    detect_and_extract_tables,
    linearize_table,
    categorize_table_content,
    split_table_if_oversized
)

# Load environment variables
load_dotenv()

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Configuration
CHUNK_SIZE = 524  # Token-based chunk size for text sections
CHUNK_OVERLAP = 100  # Token overlap for text sections
TABLE_CHUNK_SIZE = 1048  # 2x chunk size for sections surrounding tables
TABLE_CHUNK_OVERLAP = 200  # 2x overlap for sections surrounding tables
BATCH_SIZE = 20


class ContractIngestion:
    """Handles contract PDF ingestion with intelligent section-based chunking and vector storage"""
    
    def __init__(self):
        """
        Initialize the contract ingestion handler
        
        Uses PINECONE_INDEX_NAME from environment variables
        """
        self.index_name = PINECONE_INDEX_NAME
        self.index = pc.Index(self.index_name)
        
        # Text splitter for contracts without tables
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # Larger text splitter for contracts with tables (2x size/overlap)
        self.table_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TABLE_CHUNK_SIZE,
            chunk_overlap=TABLE_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def ingest_contract(self, 
                       pdf_path: str,
                       user_id: str,
                       project_id: str,
                       project_name: str,
                       contract_id: str,
                       contract_filename: str) -> Dict:
        """
        Complete ingestion pipeline for a contract PDF.
        
        Pipeline:
        1. Convert PDF to markdown (preserves structure)
        2. Detect and extract tables as atomic blocks
        3. Split remaining text into sections based on headings
        4. Categorize each section (ROYALTY, PUBLISHING, etc.)
        5. Chunk text sections with RecursiveCharacterTextSplitter
        6. Process table blocks: linearize for embedding, keep raw for LLM
        7. Generate embeddings for all chunks
        8. Upsert to user's namespace with rich metadata
        """
        print("\n" + "=" * 80)
        print(f"INGESTING CONTRACT (OneClick Method): {contract_filename}")
        print("=" * 80)
        
        # Step 1: Convert PDF to markdown
        markdown_text = pdf_to_markdown(pdf_path)
        
        if not markdown_text.strip():
            raise ValueError("No text content found in PDF")
        
        # Step 2: Detect and extract tables
        print("\n--- Table Detection ---")
        has_table, table_blocks, text_without_tables = detect_and_extract_tables(markdown_text)
        print(f"  Table detected: {has_table}")
        if has_table:
            print(f"  Found {len(table_blocks)} table block(s)")
            for idx, tb in enumerate(table_blocks):
                print(f"    Table {idx + 1}: {len(tb.raw_text)} chars, lines {tb.start_line}-{tb.end_line}")
                if tb.preceding_context:
                    print(f"      Context: '{tb.preceding_context[:80]}...'")
        
        # Step 3: Split remaining text into sections
        print("\n--- Splitting text into sections ---")
        sections = split_into_sections(text_without_tables)
        print(f"Found {len(sections)} sections")
        
        # Step 4: Categorize and chunk text sections
        # Use larger chunks when the contract contains tables for better context preservation
        active_splitter = self.table_text_splitter if has_table else self.text_splitter
        active_chunk_size = TABLE_CHUNK_SIZE if has_table else CHUNK_SIZE
        active_chunk_overlap = TABLE_CHUNK_OVERLAP if has_table else CHUNK_OVERLAP
        print(f"\n--- Categorizing and chunking text sections (chunk_size={active_chunk_size}, overlap={active_chunk_overlap}) ---")
        all_chunks = []
        uploaded_at = datetime.utcnow().isoformat()
        
        for section_header, section_content in sections:
            if not section_content.strip():
                continue
                
            category = categorize_section(section_header)
            print(f"  Section: '{section_header[:50]}...' -> {category}")
            
            section_chunks = active_splitter.split_text(section_content)
            
            for chunk_text in section_chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "embedding_text": chunk_text,
                    "section_heading": section_header,
                    "section_category": category,
                    "content_type": "text"
                })
        
        text_chunk_count = len(all_chunks)
        print(f"  Text chunks: {text_chunk_count}")
        
        # Step 5: Process table blocks as atomic chunks
        if has_table:
            print("\n--- Processing table blocks ---")
            table_chunk_count = 0
            for idx, tb in enumerate(table_blocks):
                sub_tables = split_table_if_oversized(tb)
                for sub_idx, sub_tb in enumerate(sub_tables):
                    linearized = linearize_table(sub_tb.raw_text, sub_tb.preceding_context)
                    category = categorize_table_content(sub_tb.raw_text, sub_tb.preceding_context)
                    
                    # The chunk_text seen by the LLM includes context + raw table
                    chunk_text_for_llm = sub_tb.raw_text
                    if sub_tb.preceding_context:
                        chunk_text_for_llm = sub_tb.preceding_context + "\n\n" + sub_tb.raw_text
                    
                    section_heading = sub_tb.preceding_context.splitlines()[0] if sub_tb.preceding_context else "Table"
                    
                    all_chunks.append({
                        "text": chunk_text_for_llm,
                        "embedding_text": linearized,
                        "section_heading": section_heading,
                        "section_category": category,
                        "content_type": "table"
                    })
                    table_chunk_count += 1
                    
                    label = f"Table {idx + 1}" + (f".{sub_idx + 1}" if len(sub_tables) > 1 else "")
                    print(f"  {label}: category={category}, linearized={len(linearized)} chars")
            
            print(f"  Table chunks: {table_chunk_count}")
        
        total_chunks = len(all_chunks)
        print(f"\nTotal chunks created: {total_chunks} ({text_chunk_count} text + {total_chunks - text_chunk_count} table)")
        
        if not all_chunks:
            raise ValueError("No chunks created from PDF")
        
        # Step 6: Create embeddings in batches (using embedding_text, not chunk_text)
        texts_for_embedding = [chunk["embedding_text"] for chunk in all_chunks]
        all_embeddings = []
        
        for i in range(0, len(texts_for_embedding), BATCH_SIZE):
            batch_texts = texts_for_embedding[i:i + BATCH_SIZE]
            batch_embeddings = create_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Step 7: Create vectors with metadata and deterministic IDs
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
                "chunk_text": chunk["text"],
                "content_type": chunk["content_type"],
                "has_table": has_table
            }
            
            cat = chunk["section_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            vector_id = generate_deterministic_id(
                chunk_text=chunk["text"],
                metadata=metadata
            )
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": all_embeddings[i],
                "metadata": metadata
            })
        
        # Step 8: Upsert to Pinecone
        namespace = f"{user_id}-namespace"
        
        print("\n--- Upserting to Pinecone ---")
        print(f"Index: {self.index_name}")
        print(f"Namespace: {namespace}")
        print(f"Vectors: {len(vectors_to_upsert)}")
        
        for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
            batch = vectors_to_upsert[i:i + BATCH_SIZE]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(vectors_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE} uploaded")
        
        stats = {
            "contract_id": contract_id,
            "contract_filename": contract_filename,
            "total_sections": len(sections),
            "total_chunks": total_chunks,
            "text_chunks": text_chunk_count,
            "table_chunks": total_chunks - text_chunk_count,
            "has_table": has_table,
            "total_vectors": len(vectors_to_upsert),
            "category_distribution": category_counts,
            "namespace": namespace,
            "index": self.index_name,
            "chunk_size": active_chunk_size,
            "chunk_overlap": active_chunk_overlap
        }
        
        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print(f"  Sections: {stats['total_sections']}")
        print(f"  Chunks: {stats['total_chunks']} ({stats['text_chunks']} text + {stats['table_chunks']} table)")
        print(f"  Table detected: {has_table}")
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
    ingestion = ContractIngestion()
    
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
