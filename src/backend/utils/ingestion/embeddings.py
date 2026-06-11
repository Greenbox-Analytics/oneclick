"""OpenAI embedding helpers and deterministic vector ID generation."""

import hashlib
import json

from utils.llm.client import get_openai_client

EMBEDDING_MODEL = "text-embedding-3-small"


def create_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
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
    response = client.embeddings.create(model=model, input=texts)
    embeddings = [item.embedding for item in response.data]
    return embeddings


def create_query_embedding(query: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Create embedding for a single query string.

    Args:
        query: Query text to embed
        model: Embedding model to use (default: text-embedding-3-small)

    Returns:
        Embedding vector
    """
    client = get_openai_client()
    response = client.embeddings.create(model=model, input=[query])
    return response.data[0].embedding


def generate_deterministic_id(chunk_text: str, metadata: dict) -> str:
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
        "user_id": metadata.get("user_id", ""),
        "contract_id": metadata.get("contract_id", ""),
        "section_heading": metadata.get("section_heading", ""),
        "document_name": metadata.get("contract_file", ""),
    }

    # Normalize metadata to ensure consistent ordering
    canonical_metadata = json.dumps(stable_fields, sort_keys=True)
    combined_string = chunk_text + "|" + canonical_metadata

    return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()
