"""FastAPI router for Spotify track lookup (Add Work wizard)."""

import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from integrations.spotify.service import get_track, search_tracks

router = APIRouter()


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Track name to search for"),
    limit: int = Query(10, ge=1, le=50),
    market: str = Query("US", min_length=2, max_length=2),
    user_id: str = Depends(get_current_user_id),
):
    """Search Spotify for tracks. Returns the formatted, slimmed-down match list."""
    try:
        tracks = await search_tracks(q, limit=limit, market=market)
        return {"tracks": tracks}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Spotify API error: {e.response.status_code}")


@router.get("/tracks/{track_id}")
async def track_detail(
    track_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Fetch full metadata for a single track id (used after the user picks a match)."""
    try:
        track = await get_track(track_id)
        return track
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Track not found")
        raise HTTPException(status_code=502, detail=f"Spotify API error: {e.response.status_code}")
