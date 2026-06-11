"""Tests for the shared, lazily-initialized OpenAI client at utils/llm/client.py.

The client is a module-level singleton. Each test must reset that cache so the
lazy-init code path actually runs.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_client_singleton():
    """Reset the lazy singleton between tests so each test exercises lazy init."""
    import utils.llm.client as c

    c.openai_client = None
    yield
    c.openai_client = None


def test_returns_singleton(monkeypatch):
    """Two calls return the SAME object — only one OpenAI(...) is constructed."""
    import utils.llm.client as c

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_openai_class = MagicMock(return_value=MagicMock(name="openai_instance"))
    monkeypatch.setattr(c, "OpenAI", fake_openai_class)

    a = c.get_openai_client()
    b = c.get_openai_client()

    assert a is b
    fake_openai_class.assert_called_once()


def test_raises_when_api_key_missing(monkeypatch):
    import utils.llm.client as c

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        c.get_openai_client()


def test_uses_openai_base_url_when_set(monkeypatch):
    import utils.llm.client as c

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example")
    fake_openai_class = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(c, "OpenAI", fake_openai_class)

    c.get_openai_client()

    kwargs = fake_openai_class.call_args.kwargs
    assert kwargs.get("api_key") == "sk-test"
    assert kwargs.get("base_url") == "https://proxy.example"


def test_omits_base_url_when_unset(monkeypatch):
    import utils.llm.client as c

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    fake_openai_class = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(c, "OpenAI", fake_openai_class)

    c.get_openai_client()

    assert fake_openai_class.call_args.kwargs.get("base_url") is None
