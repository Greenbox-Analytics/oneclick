"""
Shared text-normalization helpers used by OneClick, Zoe, and any future tool that
needs to compare song titles, party names, or role strings.

Functions:
- normalize_title: prepare song/work titles for fuzzy matching against statements.
- find_matching_song: fuzzy-match a contract song title against a statement dict and
  aggregate the matched amounts.
- simplify_role: collapse verbose role strings ("lyrical writer (songwriter)") into a
  small canonical taxonomy (writer / producer / label / etc.).
- normalize_name: prepare party/artist names for case-insensitive comparison.
"""

import re

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
    clean = re.sub(r"^(title|song|work|track)\s*:\s*", "", clean)

    # Remove parentheses or brackets like "(remix)" or "[live]"
    clean = re.sub(r"\(.*?\)|\[.*?\]", "", clean)

    # Remove punctuation and extra spaces
    clean = re.sub(r"[^a-z0-9\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean)

    return clean.strip()


def titles_match(title_a: str, title_b: str) -> bool:
    """
    Decide whether two song/work titles refer to the same track, using the same
    fuzzy strategies the royalty calculator relies on. Order-independent.

    Strategies (any one is sufficient):
      1. Exact match on normalized titles.
      2. One normalized title contains the other, and the shorter is >= 70% of
         the longer (guards against short-word false positives).
      3. First-3-words overlap (>= 2 shared words) with a >= 60% length ratio.

    Args:
        title_a: A song/work title (e.g. from a contract).
        title_b: Another song/work title (e.g. from a statement line).

    Returns:
        True if the titles are considered the same track.
    """
    a_norm = normalize_title(title_a)
    b_norm = normalize_title(title_b)
    if not a_norm or not b_norm:
        return False

    # Strategy 1: Exact match (case-insensitive normalized)
    if a_norm == b_norm:
        return True

    min_len = min(len(a_norm), len(b_norm))
    max_len = max(len(a_norm), len(b_norm))

    # Strategy 2: Partial match (contains or is contained)
    if a_norm in b_norm or b_norm in a_norm:
        if max_len > 0 and min_len / max_len >= 0.7:
            return True

    # Strategy 3: Very fuzzy match (first 3 words) - Fallback
    a_words = a_norm.split()[:3]
    if len(a_words) >= 2:
        b_words = b_norm.split()[:3]
        matches = sum(1 for w in a_words if w in b_words)
        if matches >= 2 and max_len > 0 and min_len / max_len >= 0.6:
            return True

    return False


def find_matching_song(song_title: str, song_totals: dict[str, float]) -> tuple[str | None, float]:
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

    total_amount = 0.0
    matched_titles = []

    # Iterate through all statement entries and sum up matches
    for title, amount in song_totals.items():
        if titles_match(song_title, title):
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
    "lyrical writer (credited as sole lyrical writer/songwriter)": "writer",
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

# Ordered keyword fallback for when exact dict lookup misses a variant
ROLE_KEYWORDS = [
    ("songwriter", "writer"),
    ("writer", "writer"),
    ("producer", "producer"),
    ("artist", "artist"),
    ("label", "label"),
    ("company", "label"),
    ("distributor", "distributor"),
    ("manager", "manager"),
    ("mixer", "mixer"),
    ("remixer", "remixer"),
    ("publisher", "publisher"),
    ("licensor", "licensor"),
    ("licensee", "licensee"),
]


def simplify_role(role: str) -> str:
    """
    Simplify a potentially verbose role string into concise, standardized terms.
    Handles combined roles separated by semicolons.
    Uses exact dict lookup first, then keyword substring fallback.

    Examples:
        "lyrical writer (credited as a sole lyrical writer)" -> "writer"
        "producer; lyrical writer (songwriter)" -> "producer; writer"
    """
    parts = [r.strip() for r in role.split(";")]
    simplified = set()
    for part in parts:
        lower = part.lower()
        # Fast path: exact match
        match = ROLE_SIMPLIFICATIONS.get(lower)
        if match:
            simplified.add(match)
            continue
        # Fallback: keyword substring match
        found = False
        for keyword, simple in ROLE_KEYWORDS:
            if keyword in lower:
                simplified.add(simple)
                found = True
                break
        if not found:
            simplified.add(part)
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
    clean = re.sub(r"\(.*?\)", "", name).strip().lower()
    # Remove extra whitespace
    clean = re.sub(r"\s+", " ", clean)
    return clean
