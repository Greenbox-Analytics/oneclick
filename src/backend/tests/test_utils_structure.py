"""Structural invariants of the utils/ refactor.

These tests lock the *shape* of the move — what was deleted is still deleted,
what was relocated is at the new path, and patch-where-used re-bindings still
point at the same callable as the utils source. A future refactor that breaks
any of these will fail loudly here rather than silently in production.
"""

import importlib
import os
from unittest.mock import MagicMock

import pytest

# ─── new symbols importable from utils/ ──────────────────────────────────────


def test_all_utils_modules_import_cleanly():
    """Every symbol the refactor created is reachable at its new path."""
    from utils.contract_parsing.basis_detection import (  # noqa: F401
        BasisFinding,
        audit_contract_basis,
        log_basis_finding,
    )
    from utils.contract_parsing.models import (  # noqa: F401
        ContractData,
        Party,
        RoyaltyShare,
        Work,
    )
    from utils.contract_parsing.parser import (  # noqa: F401
        STREAMING_EQUIVALENT_TERMS,
        MusicContractParser,
    )
    from utils.ingestion.embeddings import (  # noqa: F401
        EMBEDDING_MODEL,
        create_embeddings,
        create_query_embedding,
        generate_deterministic_id,
    )
    from utils.ingestion.pdf_markdown import (  # noqa: F401
        markdown_has_page_markers,
        pdf_to_markdown,
        strip_page_markers,
    )
    from utils.ingestion.sections import (  # noqa: F401
        SECTION_CATEGORIES,
        categorize_section,
        is_semantic_heading,
        split_into_sections,
    )
    from utils.ingestion.tables import (  # noqa: F401
        TableBlock,
        categorize_table_content,
        detect_and_extract_tables,
        linearize_table,
        split_table_if_oversized,
    )
    from utils.llm.client import get_openai_client  # noqa: F401
    from utils.text.normalize import (  # noqa: F401
        find_matching_song,
        normalize_name,
        normalize_title,
        simplify_role,
    )


# ─── deleted modules stay deleted ────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_name",
    ["oneclick.helpers", "oneclick.contract_parser", "oneclick.nuance_adjuster"],
)
def test_deleted_oneclick_modules_are_actually_gone(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


# ─── the markdown context file traveled with the parser ──────────────────────


def test_oneclick_context_md_lives_next_to_parser():
    """`MusicContractParser` reads oneclick_context.md via os.path.dirname(__file__).
    If the markdown isn't beside parser.py, parser import would FileNotFoundError."""
    from utils.contract_parsing import parser as parser_module

    expected = os.path.join(os.path.dirname(parser_module.__file__), "oneclick_context.md")
    assert os.path.isfile(expected), f"missing: {expected}"


def test_oneclick_context_md_does_not_remain_in_oneclick():
    """The original location should be gone — the move was a move, not a copy."""
    from oneclick import royalty_calculator as rc_module

    stale = os.path.join(os.path.dirname(rc_module.__file__), "oneclick_context.md")
    assert not os.path.isfile(stale), f"stale copy left behind: {stale}"


# ─── patch-where-used re-bindings still resolve to the utils source ──────────


def test_patch_where_used_rebinding_audit_contract_basis():
    """Existing tests do @patch("oneclick.royalty_calculator.audit_contract_basis", ...).
    That only works if the name re-bound at module-top in royalty_calculator.py IS the
    same object as the source in utils.contract_parsing.basis_detection."""
    from oneclick import royalty_calculator
    from utils.contract_parsing import basis_detection

    assert royalty_calculator.audit_contract_basis is basis_detection.audit_contract_basis
    assert royalty_calculator.log_basis_finding is basis_detection.log_basis_finding


def test_patch_where_used_rebinding_reference_search_embedding():
    """test_reference_search.py patches knowledge.reference_search.create_query_embedding;
    keep that binding equal to the utils source."""
    from knowledge import reference_search
    from utils.ingestion import embeddings

    assert reference_search.create_query_embedding is embeddings.create_query_embedding


def test_shared_openai_client_used_by_royalty_calculator():
    """OneClick's payable-column LLM fallback now uses the shared lazy client."""
    from oneclick import royalty_calculator
    from utils.llm import client as llm_client

    assert royalty_calculator.get_openai_client is llm_client.get_openai_client


def test_shared_openai_client_used_by_embeddings():
    from utils.ingestion import embeddings
    from utils.llm import client as llm_client

    assert embeddings.get_openai_client is llm_client.get_openai_client


def test_shared_openai_client_used_by_parser(monkeypatch):
    """`MusicContractParser.__init__` must call the shared `get_openai_client`,
    not construct its own OpenAI instance. Patch the parser's binding and assert
    the constructed parser's client is the sentinel."""
    from utils.contract_parsing import parser as parser_module

    sentinel = MagicMock(name="sentinel_openai_client")
    monkeypatch.setattr(parser_module, "get_openai_client", lambda: sentinel)

    p = parser_module.MusicContractParser(api_key="sk-test")
    assert p.openai_client is sentinel


# ─── zoe_chatbot/helpers.py shrunk to just the OneClick wrappers ─────────────


def test_zoe_helpers_exposes_oneclick_wrappers():
    import zoe_chatbot.helpers as zh

    assert callable(getattr(zh, "calculate_royalty_payments", None))
    assert callable(getattr(zh, "save_royalty_payments_to_excel", None))


@pytest.mark.parametrize(
    "moved_name",
    [
        # Moved to utils/llm/client.py
        "get_openai_client",
        # Moved to utils/ingestion/pdf_markdown.py
        "pdf_to_markdown",
        "markdown_has_page_markers",
        "strip_page_markers",
        # Moved to utils/ingestion/sections.py
        "split_into_sections",
        "is_semantic_heading",
        "categorize_section",
        "SECTION_CATEGORIES",
        # Moved to utils/ingestion/tables.py
        "TableBlock",
        "detect_and_extract_tables",
        "linearize_table",
        "categorize_table_content",
        "split_table_if_oversized",
        # Moved to utils/ingestion/embeddings.py
        "EMBEDDING_MODEL",
        "create_embeddings",
        "create_query_embedding",
        "generate_deterministic_id",
    ],
)
def test_zoe_helpers_does_not_re_export_moved_names(moved_name):
    """Catch accidental re-bloat. If someone re-adds one of these to
    zoe_chatbot.helpers, they're undoing the refactor."""
    import zoe_chatbot.helpers as zh

    assert not hasattr(zh, moved_name), f"{moved_name!r} should live in utils/, not zoe_chatbot/helpers.py"


# ─── load-bearing logger name preserved across the move ──────────────────────


def test_audit_logger_name_preserved():
    """Several tests use caplog.at_level(logger="oneclick.audit"). Renaming the
    logger string would silently make those tests stop capturing."""
    from utils.contract_parsing.basis_detection import audit_logger

    assert audit_logger.name == "oneclick.audit"
