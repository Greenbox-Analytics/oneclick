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
