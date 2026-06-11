"""Unit tests for OneClick helper utilities.

Covers:
- normalize_title           (oneclick.helpers)
- find_matching_song        (oneclick.helpers)
- simplify_role             (oneclick.helpers)
- normalize_name            (oneclick.helpers)
- is_streaming_equivalent_royalty_type + STREAMING_EQUIVALENT_TERMS
                            (oneclick.royalty_calculator / contract_parser)
- RoyaltyCalculator._calculate_payments_from_data streaming-share filter
"""

import pytest

from oneclick.royalty_calculator import (
    RoyaltyCalculator,
    is_streaming_equivalent_royalty_type,
)
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work
from utils.contract_parsing.parser import STREAMING_EQUIVALENT_TERMS
from utils.text.normalize import (
    find_matching_song,
    normalize_name,
    normalize_title,
    simplify_role,
)

# ---------------------------------------------------------------------------
# normalize_title
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# is_streaming_equivalent_royalty_type — expanded STREAMING_EQUIVALENT_TERMS
# ---------------------------------------------------------------------------


# Pre-existing terms — guard against accidental removals during the expansion.
def test_streaming_term_streaming_matches():
    assert is_streaming_equivalent_royalty_type("streaming") is True


def test_streaming_term_master_matches():
    assert is_streaming_equivalent_royalty_type("master") is True


def test_streaming_term_producer_matches():
    assert is_streaming_equivalent_royalty_type("producer") is True


def test_streaming_term_master_royalties_matches():
    assert is_streaming_equivalent_royalty_type("master royalties") is True


def test_streaming_term_sound_recording_royalty_splits_matches():
    assert is_streaming_equivalent_royalty_type("sound recording royalty splits") is True


# Newly added terms — one per category from the expansion.
def test_streaming_term_dpd_matches_case_insensitively():
    # "DPD" stored in title-case in the list, but the function lowercases
    # the input, so callers can pass any casing.
    assert is_streaming_equivalent_royalty_type("DPD") is True
    assert is_streaming_equivalent_royalty_type("dpd") is True


def test_streaming_term_neighbouring_rights_matches():
    assert is_streaming_equivalent_royalty_type("neighbouring rights") is True


def test_streaming_term_back_end_royalties_matches():
    assert is_streaming_equivalent_royalty_type("back-end royalties") is True


def test_streaming_term_distribution_revenue_matches():
    assert is_streaming_equivalent_royalty_type("distribution revenue") is True


def test_streaming_term_exploitation_income_matches():
    assert is_streaming_equivalent_royalty_type("exploitation income") is True


def test_streaming_term_per_stream_royalty_matches():
    assert is_streaming_equivalent_royalty_type("per-stream royalty") is True


def test_streaming_term_sound_engineering_matches():
    assert is_streaming_equivalent_royalty_type("sound engineering") is True


def test_streaming_term_mixer_royalties_matches():
    assert is_streaming_equivalent_royalty_type("mixer royalties") is True


def test_streaming_term_net_receipts_matches():
    assert is_streaming_equivalent_royalty_type("net receipts") is True


def test_streaming_term_soundexchange_royalties_matches():
    assert is_streaming_equivalent_royalty_type("soundexchange royalties") is True


# Substring + case behavior.
def test_streaming_term_substring_in_longer_label_matches():
    assert is_streaming_equivalent_royalty_type("Master Recording Royalty (Featured Artist)") is True


def test_streaming_term_all_caps_streaming_revenue_matches():
    assert is_streaming_equivalent_royalty_type("STREAMING REVENUE") is True


def test_streaming_term_mixed_case_producer_points_matches():
    assert is_streaming_equivalent_royalty_type("Producer Points") is True


# Negative cases — publishing-side and unrelated labels must NOT match.
def test_streaming_term_publishing_does_not_match():
    assert is_streaming_equivalent_royalty_type("publishing") is False


def test_streaming_term_writer_share_does_not_match():
    assert is_streaming_equivalent_royalty_type("writer share") is False


def test_streaming_term_mechanical_royalties_does_not_match():
    assert is_streaming_equivalent_royalty_type("mechanical royalties") is False


def test_streaming_term_sync_license_fee_does_not_match():
    assert is_streaming_equivalent_royalty_type("sync license fee") is False


def test_streaming_term_empty_string_does_not_match():
    assert is_streaming_equivalent_royalty_type("") is False


def test_streaming_term_none_does_not_match():
    assert is_streaming_equivalent_royalty_type(None) is False


def test_streaming_terms_list_includes_each_new_category():
    """Regression-lock: at least one representative term per newly-added category
    must remain in STREAMING_EQUIVALENT_TERMS."""
    for term in [
        "DPD",
        "neighbouring rights",
        "back-end royalties",
        "distribution revenue",
        "exploitation income",
        "per-stream royalty",
        "sound engineering",
        "mixer royalties",
        "net receipts",
        "SoundExchange royalties",
    ]:
        assert term in STREAMING_EQUIVALENT_TERMS, f"Missing expected term: {term!r}"


# ---------------------------------------------------------------------------
# find_matching_song — fuzzy match + aggregation
# ---------------------------------------------------------------------------


def test_find_matching_song_exact_match_returns_amount():
    matched, total = find_matching_song("Blue Sky", {"Blue Sky": 100.0})
    assert matched == "Blue Sky"
    assert total == 100.0


def test_find_matching_song_partial_match_within_70_percent_length():
    # "sunshine" (8) vs "sunshines" (9) → ratio 0.889, passes 70% guard.
    matched, total = find_matching_song("Sunshine", {"Sunshines": 42.0})
    assert matched == "Sunshines"
    assert total == 42.0


def test_find_matching_song_three_word_fallback_match():
    # Neither exact nor 70% partial, but first-3-words overlap ≥ 2 and ratio ≥ 0.6.
    # "blue sky in summer" (18) vs "blue sky on sunday" (18) → 2 word matches, ratio 1.0.
    matched, total = find_matching_song(
        "Blue Sky in Summer",
        {"Blue Sky on Sunday": 75.0},
    )
    assert matched == "Blue Sky on Sunday"
    assert total == 75.0


def test_find_matching_song_aggregates_across_parenthetical_variants():
    # All three statement entries normalize to "blue sky" → exact-match aggregation.
    matched, total = find_matching_song(
        "Blue Sky",
        {
            "Blue Sky": 100.0,
            "Blue Sky (Remix)": 50.0,
            "Blue Sky (Live)": 25.0,
        },
    )
    assert matched in {"Blue Sky", "Blue Sky (Remix)", "Blue Sky (Live)"}
    assert total == 175.0


def test_find_matching_song_no_match_returns_none_and_zero():
    matched, total = find_matching_song("Mountain Echo", {"Blue Sky": 100.0, "Crimson Tide": 50.0})
    assert matched is None
    assert total == 0.0


def test_find_matching_song_empty_title_returns_none_and_zero():
    matched, total = find_matching_song("", {"Blue Sky": 100.0})
    assert matched is None
    assert total == 0.0


def test_find_matching_song_empty_song_totals_returns_none_and_zero():
    matched, total = find_matching_song("Blue Sky", {})
    assert matched is None
    assert total == 0.0


def test_find_matching_song_length_ratio_guard_blocks_short_against_long():
    # "sky" (3) is contained in "blue sky on a sunday afternoon" (30), but
    # the 70% length guard (Strategy 2) and the 2-word minimum (Strategy 3)
    # both reject this — preventing false positives.
    matched, total = find_matching_song("Sky", {"Blue Sky on a Sunday Afternoon": 1000.0})
    assert matched is None
    assert total == 0.0


# ---------------------------------------------------------------------------
# simplify_role
# ---------------------------------------------------------------------------


def test_simplify_role_exact_dict_hit_songwriter_to_writer():
    assert simplify_role("songwriter") == "writer"


def test_simplify_role_exact_dict_hit_company_to_label():
    assert simplify_role("company") == "label"


def test_simplify_role_keyword_fallback_verbose_writer_variant():
    assert simplify_role("lyrical writer (credited as a sole lyrical writer)") == "writer"


def test_simplify_role_combined_roles_sorted_and_semicolon_joined():
    # "producer; lyrical writer (songwriter)" → {"producer", "writer"} → "producer; writer".
    assert simplify_role("producer; lyrical writer (songwriter)") == "producer; writer"


def test_simplify_role_unknown_role_passes_through():
    # No exact dict hit and no keyword match → original part is preserved.
    assert simplify_role("caterer") == "caterer"


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


def test_normalize_name_lowercases_and_trims():
    assert normalize_name("  Jane Doe  ") == "jane doe"


def test_normalize_name_strips_parenthetical_annotation():
    assert normalize_name("Jane Doe (Producer)") == "jane doe"


def test_normalize_name_collapses_internal_whitespace():
    assert normalize_name("Jane    Doe") == "jane doe"


def test_normalize_name_empty_or_none_returns_empty_string():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


# ---------------------------------------------------------------------------
# RoyaltyCalculator._calculate_payments_from_data — streaming-share filter
# ---------------------------------------------------------------------------


def _make_calculator() -> RoyaltyCalculator:
    """Build a RoyaltyCalculator without invoking __init__ (skips OpenAI key check)."""
    return RoyaltyCalculator.__new__(RoyaltyCalculator)


def _contract(parties, works, shares) -> ContractData:
    return ContractData(parties=parties, works=works, royalty_shares=shares)


class TestStreamingShareFilter:
    """Verify _calculate_payments_from_data filters royalty_shares via
    is_streaming_equivalent_royalty_type before producing payments."""

    def test_master_share_produces_payment(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert len(payments) == 1
        p = payments[0]
        assert p.song_title == "Blue Sky"
        assert p.party_name == "Jane Doe"
        assert p.role == "Producer"
        assert p.percentage == 50.0
        assert p.total_royalty == 1000.0
        assert p.amount_to_pay == 500.0

    def test_publishing_share_is_filtered_out(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="John Smith", role="Songwriter")],
            works=[Work(title="Blue Sky")],
            shares=[RoyaltyShare(party_name="John Smith", royalty_type="publishing", percentage=25.0)],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert payments == []

    def test_mixed_shares_keep_only_streaming_equivalent(self):
        calc = _make_calculator()
        data = _contract(
            parties=[
                Party(name="Jane Doe", role="Producer"),
                Party(name="John Smith", role="Songwriter"),
            ],
            works=[Work(title="Blue Sky")],
            shares=[
                RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0),
                RoyaltyShare(party_name="John Smith", royalty_type="publishing", percentage=25.0),
            ],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert len(payments) == 1
        assert payments[0].party_name == "Jane Doe"
        assert payments[0].royalty_type == "master"

    def test_newly_supported_term_flows_through_filter(self):
        # Regression-lock: a term added in the STREAMING_EQUIVALENT_TERMS expansion
        # must produce a payment end-to-end through _calculate_payments_from_data.
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[
                RoyaltyShare(party_name="Jane Doe", royalty_type="back-end royalties", percentage=10.0),
            ],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert len(payments) == 1
        assert payments[0].amount_to_pay == 100.0

    def test_unmatched_work_yields_no_payment_and_no_error(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Mountain Echo")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert payments == []

    def test_empty_works_raises_value_error(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        with pytest.raises(ValueError, match="No works found"):
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

    def test_empty_royalty_shares_raises_value_error(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[],
        )

        with pytest.raises(ValueError, match="No royalty shares found"):
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

    def test_empty_song_totals_raises_value_error(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        with pytest.raises(ValueError, match="No songs found"):
            calc._calculate_payments_from_data(data, {})
