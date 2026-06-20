"""Unit tests for the Spotify track formatter.

_format_track is a pure function — no network — so we can assert its output
shape directly. The key behavior under test: the credited-artist list exposes
each artist with the only role distinction Spotify provides (main vs featured),
alongside genre and label.
"""

from integrations.spotify.service import _format_track


def _track(**overrides):
    base = {
        "id": "track123",
        "name": "Test Song",
        "artists": [
            {"id": "a1", "name": "Main Artist", "external_urls": {"spotify": "https://open.spotify.com/artist/a1"}},
            {"id": "a2", "name": "Feature One", "external_urls": {"spotify": "https://open.spotify.com/artist/a2"}},
            {"id": "a3", "name": "Feature Two"},
        ],
        "duration_ms": 200000,
        "explicit": False,
        "popularity": 50,
        "external_ids": {"isrc": "USRC17607831"},
        "external_urls": {"spotify": "https://open.spotify.com/track/track123"},
        "album": {
            "name": "Test Album",
            "release_date": "2024-05-01",
            "external_ids": {"upc": "0123456789012"},
            "label": "Test Label",
            "images": [{"url": "https://img/cover.jpg"}],
        },
        "_resolved_genre": "afrobeats",
    }
    base.update(overrides)
    return base


def test_format_track_exposes_credited_artists_with_roles():
    out = _format_track(_track())
    assert out["artists"] == [
        {"name": "Main Artist", "role": "Main artist", "spotify_url": "https://open.spotify.com/artist/a1"},
        {"name": "Feature One", "role": "Featured artist", "spotify_url": "https://open.spotify.com/artist/a2"},
        {"name": "Feature Two", "role": "Featured artist", "spotify_url": None},
    ]
    # Existing joined string is preserved for callers that rely on it.
    assert out["artist"] == "Main Artist, Feature One, Feature Two"


def test_format_track_carries_genre_and_label():
    out = _format_track(_track())
    assert out["genre"] == "Afrobeats"  # title-cased
    assert out["label"] == "Test Label"


def test_format_track_handles_missing_artists():
    out = _format_track(_track(artists=[]))
    assert out["artists"] == []
    assert out["artist"] == ""
