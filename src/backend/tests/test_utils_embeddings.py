"""Tests for the OpenAI embedding helpers and deterministic vector IDs
in utils/ingestion/embeddings.py.

These helpers had zero direct coverage before the refactor.
"""

from unittest.mock import MagicMock

from utils.ingestion.embeddings import (
    EMBEDDING_MODEL,
    create_embeddings,
    create_query_embedding,
    generate_deterministic_id,
)

# ─── generate_deterministic_id — pure function, no mocks needed ──────────────


def test_generate_deterministic_id_is_stable():
    md = {"user_id": "u1", "contract_id": "c1", "section_heading": "Royalty", "contract_file": "deal.pdf"}
    a = generate_deterministic_id("hello world", md)
    b = generate_deterministic_id("hello world", md)
    assert a == b
    # Sanity: SHA256 produces 64 hex chars.
    assert len(a) == 64
    assert all(ch in "0123456789abcdef" for ch in a)


def test_generate_deterministic_id_differs_on_content_change():
    md = {"user_id": "u1", "contract_id": "c1"}
    a = generate_deterministic_id("hello", md)
    b = generate_deterministic_id("world", md)
    assert a != b


def test_generate_deterministic_id_differs_on_contract_id():
    a = generate_deterministic_id("hello", {"user_id": "u1", "contract_id": "c1"})
    b = generate_deterministic_id("hello", {"user_id": "u1", "contract_id": "c2"})
    assert a != b


def test_generate_deterministic_id_differs_on_user_id():
    a = generate_deterministic_id("hello", {"user_id": "u1", "contract_id": "c1"})
    b = generate_deterministic_id("hello", {"user_id": "u2", "contract_id": "c1"})
    assert a != b


def test_generate_deterministic_id_ignores_unstable_metadata():
    """Unknown metadata keys are intentionally not hashed. Adding a page_number
    or score must not change the ID — that's what makes vector upserts idempotent."""
    base = {"user_id": "u1", "contract_id": "c1"}
    a = generate_deterministic_id("hello", base)
    b = generate_deterministic_id("hello", {**base, "page_number": 5, "score": 0.87, "extra": "x"})
    assert a == b


def test_generate_deterministic_id_uses_contract_file_for_document_name():
    """The function reads metadata['contract_file'] but exposes it internally as
    'document_name'. Same file path → same ID; different file → different ID."""
    a = generate_deterministic_id("hello", {"contract_file": "deal_v1.pdf"})
    b = generate_deterministic_id("hello", {"contract_file": "deal_v2.pdf"})
    assert a != b


# ─── create_embeddings / create_query_embedding — mock the shared client ─────


def _mock_client_returning(vectors: list[list[float]]):
    """Build a mock OpenAI client whose embeddings.create returns the given vectors."""
    client = MagicMock()
    fake_data = [MagicMock(embedding=v) for v in vectors]
    client.embeddings.create.return_value = MagicMock(data=fake_data)
    return client


def test_create_embeddings_calls_openai_once_with_full_batch(monkeypatch):
    client = _mock_client_returning([[0.1, 0.2], [0.3, 0.4]])
    import utils.ingestion.embeddings as e

    monkeypatch.setattr(e, "get_openai_client", lambda: client)

    create_embeddings(["a", "b"])

    client.embeddings.create.assert_called_once()
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs.get("input") == ["a", "b"]
    assert kwargs.get("model") == EMBEDDING_MODEL


def test_create_embeddings_returns_vectors_in_order(monkeypatch):
    client = _mock_client_returning([[0.1, 0.2], [0.3, 0.4]])
    import utils.ingestion.embeddings as e

    monkeypatch.setattr(e, "get_openai_client", lambda: client)

    result = create_embeddings(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_create_query_embedding_wraps_single_string_in_list(monkeypatch):
    client = _mock_client_returning([[0.9, 0.8]])
    import utils.ingestion.embeddings as e

    monkeypatch.setattr(e, "get_openai_client", lambda: client)

    vec = create_query_embedding("hello")

    assert vec == [0.9, 0.8]
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs.get("input") == ["hello"]
    assert kwargs.get("model") == EMBEDDING_MODEL


def test_embedding_model_constant_is_text_embedding_3_small():
    """Pin the default model so a silent dependency change is loud."""
    assert EMBEDDING_MODEL == "text-embedding-3-small"
