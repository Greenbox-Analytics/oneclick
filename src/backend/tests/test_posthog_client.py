"""Tests for the PostHog HogQL client — primarily the ingest-vs-API host derivation.

The bug this guards against: POSTHOG_HOST is the ingest endpoint
(us.i.posthog.com) but HogQL queries must hit the app host (us.posthog.com).
Calling /api/projects/.../query against the ingest host returns a 4xx and
manifests in the admin UI as 'Analytics unavailable. posthog_query_failed:
HTTPStatusError'.
"""

from admin.posthog_client import PostHogClient


def test_api_host_strips_ingest_segment_us(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://us.i.posthog.com")
    monkeypatch.delenv("POSTHOG_API_HOST", raising=False)
    client = PostHogClient()
    assert client.api_host == "https://us.posthog.com"


def test_api_host_strips_ingest_segment_eu(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.i.posthog.com")
    monkeypatch.delenv("POSTHOG_API_HOST", raising=False)
    client = PostHogClient()
    assert client.api_host == "https://eu.posthog.com"


def test_api_host_override_wins(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://us.i.posthog.com")
    monkeypatch.setenv("POSTHOG_API_HOST", "https://custom.posthog.example")
    client = PostHogClient()
    assert client.api_host == "https://custom.posthog.example"


def test_api_host_passes_through_when_no_ingest_segment(monkeypatch):
    """If POSTHOG_HOST is already the app host, leave it alone."""
    monkeypatch.setenv("POSTHOG_HOST", "https://us.posthog.com")
    monkeypatch.delenv("POSTHOG_API_HOST", raising=False)
    client = PostHogClient()
    assert client.api_host == "https://us.posthog.com"


def test_ingest_host_unchanged(monkeypatch):
    """The ingest host stays as set — analytics.py needs the original value."""
    monkeypatch.setenv("POSTHOG_HOST", "https://us.i.posthog.com")
    client = PostHogClient()
    assert client.ingest_host == "https://us.i.posthog.com"
