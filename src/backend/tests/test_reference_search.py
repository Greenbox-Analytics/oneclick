"""Tests for the shared reference retriever gating policy."""

from unittest.mock import MagicMock, patch

from knowledge.reference_search import ReferencePassage, search_reference


def _match(score, page=10):
    return {
        "score": score,
        "metadata": {
            "text": f"passage at {score}",
            "section_path": "Part V ▸ Record Deals",
            "page_start": page,
            "page_end": page,
            "book_title": "The Music Business (Passman)",
        },
    }


def _index_returning(scores):
    index = MagicMock()
    index.query.return_value = {"matches": [_match(s) for s in scores]}
    return index


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_strict_mode_filters_below_threshold(_embed):
    index = _index_returning([0.72, 0.61, 0.45, 0.30])
    out = search_reference("q", min_score=0.6, floor_count=0, index=index)
    assert [round(p.score, 2) for p in out] == [0.72, 0.61]
    assert all(isinstance(p, ReferencePassage) for p in out)


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_floor_adds_top_ranked_below_threshold(_embed):
    index = _index_returning([0.55, 0.50, 0.40, 0.25, 0.10])
    out = search_reference("q", min_score=0.6, floor_count=5, floor_min=0.2, index=index)
    assert [round(p.score, 2) for p in out] == [0.55, 0.50, 0.40, 0.25]  # 0.10 dropped


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_results_sorted_desc_and_carry_metadata(_embed):
    index = _index_returning([0.61, 0.90])
    out = search_reference("q", min_score=0.6, floor_count=0, index=index)
    assert [round(p.score, 2) for p in out] == [0.90, 0.61]
    assert out[0].section_path == "Part V ▸ Record Deals"
    assert out[0].book_title == "The Music Business (Passman)"
    assert out[0].page_start == 10


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_empty_when_nothing_qualifies(_embed):
    index = _index_returning([0.15, 0.10])
    out = search_reference("q", min_score=0.6, floor_count=0, index=index)
    assert out == []


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_handles_sdk_style_attribute_objects(_embed):
    # Production Pinecone returns objects with .score/.metadata attributes, not dicts.
    md = {
        "text": "attr passage",
        "section_path": "Part V",
        "page_start": 5,
        "page_end": 6,
        "book_title": "Passman",
    }
    match = MagicMock(score=0.8, metadata=md)
    index = MagicMock()
    index.query.return_value = MagicMock(matches=[match])
    out = search_reference("q", min_score=0.6, floor_count=0, index=index)
    assert len(out) == 1
    assert out[0].score == 0.8
    assert out[0].section_path == "Part V"
    assert out[0].page_start == 5


@patch("knowledge.reference_search.create_query_embedding", return_value=[0.0] * 1536)
def test_floor_count_larger_than_candidates_is_safe(_embed):
    index = MagicMock()
    index.query.return_value = {"matches": [_match(0.40), _match(0.30)]}
    out = search_reference("q", min_score=0.6, floor_count=10, floor_min=0.2, index=index)
    assert [round(p.score, 2) for p in out] == [0.40, 0.30]  # both kept, no error
