import os
import re
import hashlib
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import pymupdf4llm
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Section category keywords for classification
SECTION_CATEGORIES = {
    "ROYALTY_CALCULATIONS": ["royalty", "compensation", "revenue share", "splits", "payment", "percentage"],
    "PUBLISHING_RIGHTS": ["publishing", "songwriter", "composition"],
    "PERFORMANCE_RIGHTS": ["performance", "production services", "live"],
    "COPYRIGHT": ["copyright", "intellectual property"],
    "TERMINATION": ["termination", "term", "duration", "end date"],
    "MASTER_RIGHTS": ["master", "master rights", "master recording"],
    "OWNERSHIP_RIGHTS": ["ownership", "synchronization", "licensing"],
    "ACCOUNTING_AND_CREDIT": ["accounting", "credit", "promotion", "audit"],
}


# ============================================================================
# PDF PROCESSING
# ============================================================================

def pdf_to_markdown(pdf_path: str) -> str:
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


def split_into_sections(markdown_text: str) -> List[Tuple[str, str]]:
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
        if is_semantic_heading(text):
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


def is_semantic_heading(text: str) -> bool:
    """
    Determine if a text is likely a section heading using simple heuristics.
    
    Args:
        text: The text to evaluate
        
    Returns:
        True if text appears to be a heading
    """
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


# ============================================================================
# SECTION CATEGORIZATION
# ============================================================================

def categorize_section(section_header: str) -> str:
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


# ============================================================================
# EMBEDDINGS
# ============================================================================

def create_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Create embeddings using OpenAI embedding model.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model to use (default: text-embedding-3-small)
        
    Returns:
        List of embedding vectors
    """
    print(f"Creating embeddings for {len(texts)} chunks...")
    response = openai_client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


def create_query_embedding(query: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Create embedding for a single query string.
    
    Args:
        query: Query text to embed
        model: Embedding model to use (default: text-embedding-3-small)
        
    Returns:
        Embedding vector
    """
    response = openai_client.embeddings.create(
        model=model,
        input=[query]
    )
    return response.data[0].embedding


# ============================================================================
# VECTOR ID GENERATION
# ============================================================================

def generate_deterministic_id(chunk_text: str, metadata: Dict) -> str:
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
