"""
Configuration for Contract RAG System

This file contains all configuration settings for:
- Regional Pinecone indexes
- Embedding models
- Chunking parameters
- Search parameters
- LLM settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================

# Regional index mapping
# Use ONE shared index per region, NOT per-user or per-project
REGIONAL_INDEXES = {
    "US": "test-3-small-index",
    "EU": "test-3-small-index",
    "UK": "test-3-small-index"
}

# Default region
DEFAULT_REGION = "US"

# Pinecone API configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============================================================================
# OPENAI CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # Optional: for custom endpoints

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Dimensions for text-embedding-3-small

# LLM model for RAG
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # Can be changed to gpt-5-nano when available
LLM_TEMPERATURE = 0.3  # Low temperature for factual accuracy
LLM_MAX_TOKENS = 1000

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Token-based chunking (300-600 tokens per chunk)
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 600

# Tokenizer encoding (used by text-embedding-3-small)
TOKENIZER_ENCODING = "cl100k_base"

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Default number of results to retrieve
DEFAULT_TOP_K = 8

# Minimum similarity threshold for chatbot responses
# Only answer questions if highest similarity >= 0.5
MIN_SIMILARITY_THRESHOLD = 0.5

# Maximum context length to send to LLM (in characters)
MAX_CONTEXT_LENGTH = 8000

# ============================================================================
# BATCH PROCESSING
# ============================================================================

# Batch size for embedding creation and vector upserts
BATCH_SIZE = 20

# ============================================================================
# METADATA SCHEMA
# ============================================================================

# Required metadata fields for each vector
REQUIRED_METADATA_FIELDS = [
    "user_id",          # UUID of the user (for isolation)
    "project_id",       # UUID of the project
    "project_name",     # Name of the project
    "contract_id",      # UUID of the contract (from project_files table)
    "contract_file",    # Original filename
    "region",           # Region code (US, EU, UK)
    "uploaded_at",      # ISO timestamp
    "page_number",      # Page number in PDF
    "chunk_index",      # Index of chunk within document
    "chunk_text"        # The actual text content
]

# Optional metadata fields
OPTIONAL_METADATA_FIELDS = [
    "token_count",      # Number of tokens in chunk
    "artist_id",        # UUID of the artist (if applicable)
    "artist_name"       # Name of the artist (if applicable)
]

# ============================================================================
# NAMESPACE CONFIGURATION
# ============================================================================

# Namespace pattern: {user_id}_data
# This provides user-level isolation within shared regional indexes
def get_namespace(user_id: str) -> str:
    """
    Get namespace for a user
    
    Args:
        user_id: UUID of the user
        
    Returns:
        Namespace string
    """
    return f"{user_id}_data"

# ============================================================================
# VECTOR ID GENERATION
# ============================================================================

# Use SHA256 for deterministic vector IDs
# This ensures re-uploading the same contract creates the same IDs
VECTOR_ID_HASH_ALGORITHM = "sha256"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that required environment variables are set"""
    errors = []
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not found in environment variables")
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not found in environment variables")
    
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return True

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

CHATBOT_SYSTEM_PROMPT = """You are a specialized contract analysis assistant for the music industry. Your role is to answer questions about music contracts accurately and precisely.

CRITICAL RULES:
1. ONLY answer based on the provided contract contracts - do not use external knowledge
2. If the answer is not explicitly stated in the contracts, respond with: "I don't know based on the available documents."
3. Always cite the source (contract file and page number) when providing information
4. Be precise with numbers, percentages, dates, and legal terms
5. If multiple contracts contain relevant information, clearly distinguish between them
6. Do not make assumptions or inferences beyond what is explicitly stated
7. If asked about something not in the contracts, acknowledge the limitation

Your answers should be:
- Accurate and grounded in the provided text
- Clear and concise
- Properly cited with sources
- Professional and helpful"""

# ============================================================================
# LOGGING
# ============================================================================

# Enable verbose logging
VERBOSE_LOGGING = True

# Log file path (optional)
LOG_FILE_PATH = None  # Set to a path to enable file logging

# ============================================================================
# EXPORT ALL SETTINGS
# ============================================================================

__all__ = [
    # Pinecone
    'REGIONAL_INDEXES',
    'DEFAULT_REGION',
    'PINECONE_API_KEY',
    
    # OpenAI
    'OPENAI_API_KEY',
    'OPENAI_BASE_URL',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMENSIONS',
    'DEFAULT_LLM_MODEL',
    'LLM_TEMPERATURE',
    'LLM_MAX_TOKENS',
    
    # Chunking
    'MIN_CHUNK_TOKENS',
    'MAX_CHUNK_TOKENS',
    'TOKENIZER_ENCODING',
    
    # Search
    'DEFAULT_TOP_K',
    'MIN_SIMILARITY_THRESHOLD',
    'MAX_CONTEXT_LENGTH',
    
    # Batch processing
    'BATCH_SIZE',
    
    # Metadata
    'REQUIRED_METADATA_FIELDS',
    'OPTIONAL_METADATA_FIELDS',
    
    # Namespace
    'get_namespace',
    
    # Vector ID
    'VECTOR_ID_HASH_ALGORITHM',
    
    # System prompts
    'CHATBOT_SYSTEM_PROMPT',
    
    # Logging
    'VERBOSE_LOGGING',
    'LOG_FILE_PATH',
    
    # Validation
    'validate_config'
]
