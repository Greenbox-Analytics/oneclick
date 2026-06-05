"""Tests for stale-cache re-conversion decision and OneClick marker-stripping."""

from zoe_chatbot.helpers import markdown_has_page_markers, strip_page_markers


def _needs_reconvert(cached: str | None) -> bool:
    """Mirrors the endpoint guard: re-convert when empty OR missing page markers."""
    return not cached or not markdown_has_page_markers(cached)


def test_reconvert_decision():
    assert _needs_reconvert(None) is True
    assert _needs_reconvert("") is True
    assert _needs_reconvert("legacy markdown, no markers") is True
    assert _needs_reconvert("intro\n\n[[PAGE 1]]\n\nbody") is False


def test_oneclick_consumes_marker_free_text():
    marked = "intro\n\n[[PAGE 1]]\n\n50% to A\n\n[[PAGE 2]]\n\n50% to B"
    cleaned = strip_page_markers(marked)
    assert "[[PAGE" not in cleaned
    assert "50% to A" in cleaned and "50% to B" in cleaned
