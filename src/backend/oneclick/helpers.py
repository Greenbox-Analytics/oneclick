"""
Helper functions for OneClick royalty calculations
Contains utility methods for song matching and normalization
"""

import re
import os
from typing import Dict, Optional, Tuple, List
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def normalize_title(title: str) -> str:
    """
    Normalize work title for matching against royalty statements.
    
    Args:
        title: Song/work title to normalize
        
    Returns:
        Normalized title string (lowercase, no punctuation, etc.)
    """
    if not title:
        return ""
    
    clean = title.lower().strip()
    
    # Remove label prefixes like "title:", "song:", etc.
    clean = re.sub(r'^(title|song|work|track)\s*:\s*', '', clean)
    
    # Remove parentheses or brackets like "(remix)" or "[live]"
    clean = re.sub(r'\(.*?\)|\[.*?\]', '', clean)
    
    # Remove punctuation and extra spaces
    clean = re.sub(r'[^a-z0-9\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean)
    
    return clean


def find_matching_song(
    song_title: str, 
    song_totals: Dict[str, float]
) -> Tuple[Optional[str], float]:
    """
    Find matching song in royalty statement with fuzzy matching.
    
    Aggregates amounts from all entries in song_totals that match the song_title
    based on the matching strategies. This ensures that if a song appears multiple
    times with slight variations (e.g. "Song A" and "Song A (Remix)"), all revenue
    is captured.
    
    Args:
        song_title: Title from contract
        song_totals: Dictionary of song titles to amounts from statement
        
    Returns:
        Tuple of (best_match_title, total_aggregated_amount) or (None, 0.0) if not found
    """
    if not song_title or not song_totals:
        return (None, 0.0)
    
    song_title_norm = normalize_title(song_title)
    total_amount = 0.0
    matched_titles = []
    
    # Iterate through all statement entries and sum up matches
    for title, amount in song_totals.items():
        title_norm = normalize_title(title)
        is_match = False
        
        # Strategy 1: Exact match (case-insensitive normalized)
        if title_norm == song_title_norm:
            is_match = True
            
        # Strategy 2: Partial match (contains or is contained)
        elif song_title_norm in title_norm or title_norm in song_title_norm:
            # Must be at least 70% of the length to avoid false positives
            min_len = min(len(song_title_norm), len(title_norm))
            max_len = max(len(song_title_norm), len(title_norm))
            
            if max_len > 0 and min_len / max_len >= 0.7:
                is_match = True
        
        # Strategy 3: Very fuzzy match (first 3 words) - Fallback
        if not is_match:
            song_words = song_title_norm.split()[:3]
            if len(song_words) >= 2:
                title_words = title_norm.split()[:3]
                matches = sum(1 for w in song_words if w in title_words)
                if matches >= 2:
                    # Safety check on length ratio
                    min_len = min(len(song_title_norm), len(title_norm))
                    max_len = max(len(song_title_norm), len(title_norm))
                    if max_len > 0 and min_len / max_len >= 0.6:
                         is_match = True

        if is_match:
            total_amount += amount
            matched_titles.append(title)
    
    if matched_titles:
        # Return the first matched title as representative, and the SUM of all amounts
        return (matched_titles[0], total_amount)
    
    return (None, 0.0)


ROLE_SIMPLIFICATIONS = {
    "lyrical writer": "writer",
    "lyrical writer (songwriter)": "writer",
    "lyrical writer (credited as a sole lyrical writer)": "writer",
    "songwriter": "writer",
    "writer": "writer",
    "producer": "producer",
    "artist": "artist",
    "label": "label",
    "distributor": "distributor",
    "manager": "manager",
    "mixer": "mixer",
    "remixer": "remixer",
    "publisher": "publisher",
    "company": "label",
    "licensor": "licensor",
    "licensee": "licensee",
}


def simplify_role(role: str) -> str:
    """
    Simplify a potentially verbose role string into concise, standardized terms.
    Handles combined roles separated by semicolons.
    
    Examples:
        "lyrical writer (credited as a sole lyrical writer)" -> "Writer"
        "producer; lyrical writer (songwriter)" -> "Producer; Writer"
    """
    parts = [r.strip() for r in role.split(";")]
    simplified = set()
    for part in parts:
        simplified.add(ROLE_SIMPLIFICATIONS.get(part.lower(), part))
    return "; ".join(sorted(simplified))


def normalize_name(name: str) -> str:
    """
    Normalize party/artist name for comparison.
    
    Args:
        name: Party or artist name to normalize
        
    Returns:
        Normalized name string
    """
    if not name:
        return ""
    # Remove role annotations, lowercase, strip whitespace
    clean = re.sub(r'\(.*?\)', '', name).strip().lower()
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean)
    return clean

def find_chunks_with_text(
    search_term: str,
    contract_id: str,
    user_id: str,
    index_name: str = "test-3-small-index",
    top_k: int = 2000
) -> List[Dict]:
    """
    Find all chunks in Pinecone namespace that contain a specific text string for a given contract.
    
    Args:
        search_term: The text string to search for (case-insensitive)
        contract_id: The contract ID to filter by
        user_id: The user ID for the namespace
        index_name: The Pinecone index name (default: "test-3-small-index")
        top_k: Max number of chunks to retrieve for filtering (default: 2000)
        
    Returns:
        List of dictionaries containing match details (id, score, text, section)
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        namespace = f"{user_id}-namespace"

        # Create a dummy vector of the correct dimension (assuming 1536 for text-embedding-3-small)
        dummy_vector = [0.0] * 1536 

        # Query Pinecone
        results = index.query(
            namespace=namespace,
            vector=dummy_vector,
            filter={
                "contract_id": {"$eq": contract_id}
            },
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for match in results.matches:
            chunk_text = match.metadata.get("chunk_text", "")
            # Case-insensitive check
            if search_term.lower() in chunk_text.lower():
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "text": chunk_text,
                    "section": match.metadata.get("section_heading", "N/A")
                })
        
        return matches

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []
