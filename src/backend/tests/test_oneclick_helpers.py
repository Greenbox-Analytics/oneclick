from oneclick.helpers import normalize_title


def test_parenthetical_variant_matches_base():
    assert normalize_title("Rude Gyal") == normalize_title("Rude Gyal (Master Recording)")


def test_bracket_variant_matches_base():
    assert normalize_title("Rude Gyal") == normalize_title("Rude Gyal [Live]")


def test_output_has_no_trailing_or_leading_whitespace():
    result = normalize_title("  Rude Gyal (Master Recording)  ")
    assert result == result.strip()
    assert result == "rude gyal"


def test_empty_input_returns_empty_string():
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_label_prefix_is_stripped():
    assert normalize_title("Title: Rude Gyal") == "rude gyal"


def test_punctuation_is_stripped():
    assert normalize_title("Rude Gyal!") == "rude gyal"
