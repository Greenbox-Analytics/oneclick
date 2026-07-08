"""Spotify Web API client using Client Credentials flow (app-only, no user OAuth).

Used by the Metadata Registry Add Work wizard to look up released-track metadata
(ISRC, label, release date, duration, popularity, cover art).
"""

import asyncio
import base64
import os
import time

import httpx

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"

# Cached access token + expiry timestamp shared across requests
_token_cache: dict = {"access_token": None, "expires_at": 0.0}
_token_lock = asyncio.Lock()


async def _get_access_token() -> str:
    """Return a cached access token, refreshing if expired or unset."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set")

    now = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] - 30 > now:
        return _token_cache["access_token"]

    async with _token_lock:
        # Re-check after acquiring the lock — another coroutine may have refreshed.
        if _token_cache["access_token"] and _token_cache["expires_at"] - 30 > time.time():
            return _token_cache["access_token"]

        creds = f"{client_id}:{client_secret}".encode()
        auth_header = base64.b64encode(creds).decode()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                SPOTIFY_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
            )
            resp.raise_for_status()
            data = resp.json()

        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + int(data.get("expires_in", 3600))
        return _token_cache["access_token"]


def _format_track(track: dict) -> dict:
    """Reduce a Spotify track payload to the fields the Add Work wizard needs."""
    album = track.get("album") or {}
    images = album.get("images") or []
    cover = images[0]["url"] if images else None
    # _resolved_genre is set by get_track after a follow-up call to /artists/{id};
    # search responses don't have it (Spotify's track endpoint doesn't carry genre).
    genre = track.get("_resolved_genre")
    artists = track.get("artists") or []
    return {
        "id": track.get("id"),
        "title": track.get("name"),
        "artist": ", ".join(a["name"] for a in artists),
        # Credited artists with the only role distinction Spotify provides:
        # the first artist is the main artist, the rest are featured.
        "artists": [
            {
                "name": a.get("name"),
                "role": "Main artist" if i == 0 else "Featured artist",
                "spotify_url": (a.get("external_urls") or {}).get("spotify"),
            }
            for i, a in enumerate(artists)
        ],
        "album": album.get("name"),
        "release_date": album.get("release_date"),
        "year": int(album.get("release_date", "0000")[:4]) if album.get("release_date") else None,
        "duration_ms": track.get("duration_ms"),
        "explicit": track.get("explicit", False),
        "popularity": track.get("popularity", 0),
        "isrc": (track.get("external_ids") or {}).get("isrc"),
        "upc": (album.get("external_ids") or {}).get("upc"),
        "label": album.get("label"),
        "genre": genre.title() if isinstance(genre, str) and genre else None,
        "cover_url": cover,
        "spotify_url": (track.get("external_urls") or {}).get("spotify"),
    }


async def search_tracks(query: str, limit: int = 10, market: str = "US") -> list[dict]:
    """Search Spotify for tracks matching the query string."""
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{SPOTIFY_API}/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": min(max(limit, 1), 50), "market": market},
        )
        resp.raise_for_status()
        items = (resp.json().get("tracks") or {}).get("items") or []
        return [_format_track(t) for t in items]


async def get_track(track_id: str) -> dict:
    """Fetch a single track by Spotify id. Returns full metadata including ISRC/UPC/label/genre."""
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{SPOTIFY_API}/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        track = resp.json()

        # /tracks/{id} returns album without external_ids.upc; fetch the album to get it
        album_id = (track.get("album") or {}).get("id")
        if album_id:
            album_resp = await client.get(
                f"{SPOTIFY_API}/albums/{album_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if album_resp.status_code == 200:
                track["album"] = album_resp.json()

        # Genre lives on the artist endpoint, not the track endpoint. Take the
        # primary artist's first genre — Spotify often returns several; the
        # first is the most strongly associated.
        artists = track.get("artists") or []
        primary_artist_id = artists[0]["id"] if artists and artists[0].get("id") else None
        if primary_artist_id:
            artist_resp = await client.get(
                f"{SPOTIFY_API}/artists/{primary_artist_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if artist_resp.status_code == 200:
                genres = artist_resp.json().get("genres") or []
                if genres:
                    track["_resolved_genre"] = genres[0]

        return _format_track(track)
