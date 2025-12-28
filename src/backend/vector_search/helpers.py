# Helpers file qith functions that support the vector search and upsert operations 

import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Tuple
import hashlib
import json

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Convert pdf to markdown
def pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown format using pymupdf4llm.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        str: The markdown representation of the PDF content.
    """
    return pymupdf4llm.to_markdown(file_path)


def split_into_sections(markdown_text: str) -> List[Tuple[str, str]]:
    """
    Split markdown text into sections using a 4-layer approach:
    1) Explicit headings (markdown / numbered)
    2) Heuristic heading candidates (caps, bold, short lines)
    3) Semantic heading validation (LLM hook)
    4) Content-based section inference

    Returns list of (header, content).
    """

    lines = markdown_text.splitlines()
    sections: List[Tuple[str, str]] = []

    # -----------------------------
    # Layer 1: Explicit Headings
    # -----------------------------
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

    # --------------------------------
    # Layer 2: Heuristic Header Detection
    # --------------------------------
    def is_heuristic_header(line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        # ALL CAPS, short
        if stripped.isupper() and len(stripped.split()) <= 10:
            return True

        # Bold markdown
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped.split()) <= 12:
            return True

        # Short title-like line
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

    # --------------------------------
    # Layer 3: Semantic Heading Detection (LLM hook)
    # --------------------------------
    # Idea:
    # - Identify candidate lines (short, prominent)
    # - Ask LLM: "Is this a legal contract section heading?"
    # - Keep only those marked as YES

    semantic_candidates = [
        (i, line.strip())
        for i, line in enumerate(lines)
        if 2 <= len(line.split()) <= 8
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

    # --------------------------------
    # Layer 4: Content-Based Inference
    # --------------------------------
    # Detect implicit royalty / payment section starts

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

    # --------------------------------
    # Fallback
    # --------------------------------
    return [("Full Document", markdown_text.strip())]


def categorize_sections(sections: list) -> list:
    """
    Categorize each section based on its header.
    
    Args:
        sections (list): List of tuples containing section headers and content.
        
    Returns:
        list: List of tuples containing section category, header, and content.
    """
    categorized_sections = []
    for header, content in sections:
        category = section_categorizer(header)
        categorized_sections.append((category, header, content))
    return categorized_sections


def is_bold(text: str) -> bool:
    """
    Check if the given text is bold using pymupdf4llm formatting features.
    
    Args:
        text (str): The text to check.
    Returns:
        bool: True if the text is bold, False otherwise.
    """


def section_categorizer(section_header: str) -> str:
    """
    Categorize a section based on its header using simple keyword matching.
    
    Args:
        section_header (str): The header of the section.
        
    Returns:
        str: The category of the section.
    """
    header_lower = section_header.lower()
    if any(keyword in header_lower for keyword in ["royalty", "compensation", "revenue share", "compensation", "splits"]):
        return "ROYALTY_CALCULATIONS"
    elif any(keyword in header_lower for keyword in ["publishing"]):
        return "PUBLISHING_RIGHTS"
    elif any(keyword in header_lower for keyword in ["performance", "production services"]):
        return "PERFORMANCE_RIGHTS"
    elif any(keyword in header_lower for keyword in ["copyright"]):
        return "COPYRIGHT"
    elif any(keyword in header_lower for keyword in ["termination", "term"]):
        return "TERMINATION"
    elif any(keyword in header_lower for keyword in ["master", "master rights"]):
        return "MASTER_RIGHTS"
    elif any(keyword in header_lower for keyword in ["ownership"]):
        return "OWNERSHIP_RIGHTS"
    elif any(keyword in header_lower for keyword in ["copyright"]):
        return "COPYRIGHT"
    elif any(keyword in header_lower for keyword in ["accounting", "credit", "promotion"]):
        return "ACCOUNTING_AND_CREDIT"
    else:
        return "Other"
    

def is_semantic_heading(text: str) -> bool:
    """
    Stub function to determine if a text is a semantic heading.
    In a real implementation, this would call an LLM to validate.
    
    Args:
        text (str): The text to evaluate.
    """
    
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    prompt = f"Is the following text a section heading in a legal contract? Answer 'yes' or 'no'.\n\nText: '''{text}'''"

    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies section headings in legal contracts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000
    )   

    if 'yes' in response.choices[0].message.content.lower():
        return True
    return False

    
def create_vector_id(content: str, metadata: dict) -> str:
    """
    Create a deterministic vector ID based on chunk text and stable metadata fields.
    
    Args:
        content (str): The text content of the chunk.
        metadata (dict): Metadata dictionary.
    Returns:
        str: Deterministic vector ID.
    """
    # Use only stable identifiers to create the ID
    # Exclude: upload_time (changes each run), chunk_text (redundant with content)
    stable_fields = {
        'user_id': metadata.get('user_id', ''),
        'contract_id': metadata.get('contract_id', ''),
        'section_heading': metadata.get('section_heading', ''),
        'document_name': metadata.get('document_name', '')
    }
    
    # Normalize metadata to ensure consistent ordering
    normalized_metadata = json.dumps(stable_fields, sort_keys=True)
    normalized = f"{content}|{normalized_metadata}"

    return str(hashlib.sha256(normalized.encode('utf-8')).hexdigest())

def create_embeddings(openai_client, texts):
    """Create embeddings using OpenAI text-embedding-3-small model"""
    print(f"Creating embeddings for {len(texts)} chunks...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    print(f"Created {len(embeddings)} embeddings")
    return embeddings
