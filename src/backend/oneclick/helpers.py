"""
Helper functions for OneClick royalty calculations
Contains utility methods for song matching and normalization
"""

import re
from typing import Dict, Optional, Tuple


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
    
    Tries multiple matching strategies:
    1. Exact match (case-insensitive)
    2. Normalized match (without extra whitespace/punctuation)
    3. Partial match (contains or is contained)
    
    Args:
        song_title: Title from contract
        song_totals: Dictionary of song titles to amounts from statement
        
    Returns:
        Tuple of (matched_title, total_amount) or (None, 0.0) if not found
    """
    if not song_title or not song_totals:
        return (None, 0.0)
    
    song_title_norm = normalize_title(song_title)
    
    # Strategy 1: Exact match (case-insensitive)
    for title, amount in song_totals.items():
        if normalize_title(title) == song_title_norm:
            return (title, amount)
    
    # Strategy 2: Partial match (contains or is contained)
    for title, amount in song_totals.items():
        title_norm = normalize_title(title)
        if song_title_norm in title_norm or title_norm in song_title_norm:
            # Must be at least 70% of the length to avoid false positives
            min_len = min(len(song_title_norm), len(title_norm))
            max_len = max(len(song_title_norm), len(title_norm))
            if min_len / max_len >= 0.7:
                return (title, amount)
    
    # Strategy 3: Very fuzzy match (first 3 words)
    song_words = song_title_norm.split()[:3]
    if len(song_words) >= 2:
        for title, amount in song_totals.items():
            title_words = normalize_title(title).split()[:3]
            matches = sum(1 for w in song_words if w in title_words)
            if matches >= 2:
                return (title, amount)
    
    return (None, 0.0)


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
