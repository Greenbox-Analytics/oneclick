"""The working set's contract-name lookup covers selection + carried markdown."""

from main import _name_lookup_ids


def test_union_of_selected_and_markdown_keys():
    assert sorted(_name_lookup_ids(["c2"], {"c1": "x", "c2": "y"})) == ["c1", "c2"]


def test_selection_only_when_no_carried():
    assert _name_lookup_ids(["c2"], {"c2": "y"}) == ["c2"]


def test_empty_inputs():
    assert _name_lookup_ids(None, None) == []
    assert _name_lookup_ids([], {}) == []
