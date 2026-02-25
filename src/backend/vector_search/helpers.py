import os
import re
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import pymupdf4llm
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Global OpenAI client (lazy initialization)
openai_client = None

def get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance (lazy initialization)"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise RuntimeError(
                "Missing required environment variable: OPENAI_API_KEY"
            )
        
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
    return openai_client

# Section category keywords for classification
SECTION_CATEGORIES = {
    "ROYALTY_CALCULATIONS": ["royalty", "compensation", "revenue share", "splits", "payment", "percentage", "master"],
    "PUBLISHING_RIGHTS": ["publishing", "songwriter", "composition"],
    "PERFORMANCE_RIGHTS": ["performance", "production services", "live"],
    "COPYRIGHT": ["copyright", "intellectual property"],
    "TERMINATION": ["termination", "term", "duration", "end date"],
    "MASTER_RIGHTS": ["master rights", "master recording"],
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
# TABLE DETECTION, EXTRACTION, AND LINEARIZATION
# ============================================================================

@dataclass
class TableBlock:
    raw_text: str
    preceding_context: str
    start_line: int
    end_line: int


def detect_and_extract_tables(markdown_text: str) -> Tuple[bool, List[TableBlock], str]:
    """
    Detect and extract all markdown tables from text.
    
    Returns tables as atomic blocks and the remaining text with tables replaced
    by [TABLE_REMOVED] placeholders so section splitting still works.
    
    Returns:
        (has_table, table_blocks, text_without_tables)
    """
    lines = markdown_text.splitlines()
    table_blocks: List[TableBlock] = []

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2

    def _is_separator_line(line: str) -> bool:
        return bool(re.search(r'\|[\s:]*-{2,}[\s:]*\|', line))

    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]) or _is_separator_line(lines[i]):
            table_start = i
            # Walk backward to include the header row if we landed on a separator
            while table_start > 0 and _is_table_line(lines[table_start - 1]):
                table_start -= 1

            table_end = i + 1
            while table_end < len(lines) and (_is_table_line(lines[table_end]) or _is_separator_line(lines[table_end])):
                table_end += 1

            table_lines = lines[table_start:table_end]

            # Validate: a real table needs at least a header + separator + one data row
            has_separator = any(_is_separator_line(l) for l in table_lines)
            has_data_rows = sum(1 for l in table_lines if _is_table_line(l) and not _is_separator_line(l)) >= 2
            if has_separator and has_data_rows:
                # Gather preceding context (up to 3 non-empty lines before the table)
                context_lines = []
                j = table_start - 1
                while j >= 0 and len(context_lines) < 3:
                    stripped = lines[j].strip()
                    if stripped and not _is_table_line(lines[j]):
                        context_lines.insert(0, stripped)
                    elif not stripped and context_lines:
                        break
                    j -= 1

                table_blocks.append(TableBlock(
                    raw_text="\n".join(table_lines),
                    preceding_context="\n".join(context_lines),
                    start_line=table_start,
                    end_line=table_end
                ))

            i = table_end
        else:
            i += 1

    # Build text_without_tables by replacing table regions with placeholder
    if not table_blocks:
        return False, [], markdown_text

    result_lines = list(lines)
    # Process in reverse so indices stay valid
    for tb in reversed(table_blocks):
        result_lines[tb.start_line:tb.end_line] = ["[TABLE_REMOVED]"]

    text_without_tables = "\n".join(result_lines)
    return True, table_blocks, text_without_tables


def linearize_table(table_text: str, preceding_context: str = "") -> str:
    """
    Convert a markdown table into natural-language sentences for embedding.
    
    The embedding model sees this linearized form (good semantic similarity),
    while the raw table is stored separately for LLM extraction (good accuracy).
    
    Example:
        | Party  | Share | Type        |
        |--------|-------|-------------|
        | Artist | 15%   | Net Revenue |
        
        Becomes: "Party: Artist, Share: 15%, Type: Net Revenue."
    """
    lines = table_text.strip().splitlines()
    
    # Find header row (first non-separator table line)
    header_row = None
    separator_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.search(r'\|[\s:]*-{2,}[\s:]*\|', stripped):
            separator_idx = idx
            break
    
    if separator_idx is None or separator_idx == 0:
        # No clear header/separator structure — return cleaned text
        cleaned = re.sub(r'[|\-]+', ' ', table_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return f"{preceding_context}\n{cleaned}" if preceding_context else cleaned
    
    header_row = lines[separator_idx - 1]
    headers = [h.strip() for h in header_row.split('|') if h.strip()]
    
    data_lines = lines[separator_idx + 1:]
    
    sentences = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith('|'):
            continue
        cells = [c.strip() for c in stripped.split('|') if c.strip()]
        if not cells:
            continue
        
        pairs = []
        for col_idx, cell in enumerate(cells):
            col_name = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
            pairs.append(f"{col_name}: {cell}")
        sentences.append(", ".join(pairs) + ".")
    
    linearized = " ".join(sentences)
    
    if preceding_context:
        return f"{preceding_context}\n{linearized}"
    return linearized


def categorize_table_content(table_text: str, preceding_context: str = "") -> str:
    """
    Categorize a table by running keyword matching against its cell values
    and preceding context. This ensures tables get the right category even
    when the section header is vague.
    """
    combined = (preceding_context + " " + table_text).lower()
    
    for category, keywords in SECTION_CATEGORIES.items():
        if any(keyword in combined for keyword in keywords):
            return category
    
    return "OTHER"


def split_table_if_oversized(table_block: TableBlock, max_chars: int = 4000) -> List[TableBlock]:
    """
    If a table exceeds max_chars, split it into sub-tables by row groups,
    preserving the header row in each sub-chunk.
    """
    if len(table_block.raw_text) <= max_chars:
        return [table_block]
    
    lines = table_block.raw_text.splitlines()
    
    # Find header and separator
    header_line = None
    separator_line = None
    data_start = 0
    for idx, line in enumerate(lines):
        if re.search(r'\|[\s:]*-{2,}[\s:]*\|', line):
            separator_line = line
            if idx > 0:
                header_line = lines[idx - 1]
                data_start = idx + 1
            else:
                data_start = idx + 1
            break
    
    if header_line is None or separator_line is None:
        return [table_block]
    
    preamble = header_line + "\n" + separator_line
    preamble_len = len(preamble) + 1  # +1 for joining newline
    
    data_rows = lines[data_start:]
    sub_tables: List[TableBlock] = []
    current_rows: List[str] = []
    current_len = preamble_len
    
    for row in data_rows:
        row_len = len(row) + 1
        if current_rows and (current_len + row_len) > max_chars:
            sub_text = preamble + "\n" + "\n".join(current_rows)
            sub_tables.append(TableBlock(
                raw_text=sub_text,
                preceding_context=table_block.preceding_context,
                start_line=table_block.start_line,
                end_line=table_block.end_line
            ))
            current_rows = []
            current_len = preamble_len
        
        current_rows.append(row)
        current_len += row_len
    
    if current_rows:
        sub_text = preamble + "\n" + "\n".join(current_rows)
        sub_tables.append(TableBlock(
            raw_text=sub_text,
            preceding_context=table_block.preceding_context,
            start_line=table_block.start_line,
            end_line=table_block.end_line
        ))
    
    return sub_tables if sub_tables else [table_block]


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
    client = get_openai_client()
    response = client.embeddings.create(
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
    client = get_openai_client()
    response = client.embeddings.create(
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


# ============================================================================
# ONECLICK ROYALTY CALCULATION HELPERS
# ============================================================================

def calculate_royalty_payments(
    contract_path: str,
    statement_path: str,
    user_id: str,
    contract_id: str,
    api_key: str = None,
    contract_ids: List[str] = None
) -> List[Dict]:
    """
    Calculate royalty payments from a contract and royalty statement.
    
    This is a helper function that wraps the RoyaltyCalculator to provide
    a simpler interface for the OneClick feature.
    
    Args:
        contract_path: Path to the contract PDF file (not used, kept for compatibility)
        statement_path: Path to the royalty statement Excel file
        user_id: User ID for querying Pinecone
        contract_id: Contract ID for querying Pinecone (single)
        api_key: Optional OpenAI API key (uses env var if not provided)
        contract_ids: Optional list of contract IDs for multi-contract calculation
        
    Returns:
        List of payment dictionaries with keys:
            - song_title: str
            - party_name: str
            - role: str
            - royalty_type: str
            - percentage: float
            - total_royalty: float
            - amount_to_pay: float
            - terms: Optional[str]
    """
    from oneclick.royalty_calculator import RoyaltyCalculator
    
    # Initialize calculator
    calculator = RoyaltyCalculator(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    # Calculate payments
    if contract_ids and len(contract_ids) > 0:
        # Multi-contract mode
        payments = calculator.calculate_payments_from_contract_ids(
            contract_ids=contract_ids,
            user_id=user_id,
            statement_path=statement_path
        )
    else:
        # Single contract mode
        payments = calculator.calculate_payments(
            contract_path=contract_path,
            statement_path=statement_path,
            user_id=user_id,
            contract_id=contract_id
        )
    
    # Convert to dictionaries for easier JSON serialization
    payment_dicts = []
    for payment in payments:
        payment_dicts.append({
            'song_title': payment.song_title,
            'party_name': payment.party_name,
            'role': payment.role,
            'royalty_type': payment.royalty_type,
            'percentage': payment.percentage,
            'total_royalty': payment.total_royalty,
            'amount_to_pay': payment.amount_to_pay,
            'terms': payment.terms
        })
    
    return payment_dicts


def save_royalty_payments_to_excel(
    payments: List[Dict],
    output_path: str,
    api_key: str = None
) -> None:
    """
    Save royalty payments to an Excel file with formatting.
    
    Args:
        payments: List of payment dictionaries (from calculate_royalty_payments)
        output_path: Path where Excel file should be saved
        api_key: Optional OpenAI API key (uses env var if not provided)
    """
    from oneclick.royalty_calculator import RoyaltyCalculator, RoyaltyPayment
    
    # Initialize calculator
    calculator = RoyaltyCalculator(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    # Convert dictionaries back to RoyaltyPayment objects
    payment_objects = []
    for p in payments:
        payment_objects.append(RoyaltyPayment(
            song_title=p['song_title'],
            party_name=p['party_name'],
            role=p['role'],
            royalty_type=p['royalty_type'],
            percentage=p['percentage'],
            total_royalty=p['total_royalty'],
            amount_to_pay=p['amount_to_pay'],
            terms=p.get('terms')
        ))
    
    # Save to Excel
    calculator.save_payments_to_excel(payment_objects, output_path)
