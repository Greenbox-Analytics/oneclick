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
    CalculationError,
    RoyaltyCalculator,
    is_streaming_earnable_share,
    is_streaming_equivalent_royalty_type,
)
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work
from utils.contract_parsing.parser import (
    NON_STREAMING_PAYOR_CONTEXT_PHRASES,
    NON_STREAMING_PAYOR_TERMS,
    STREAMING_EQUIVALENT_TERMS,
)
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


def test_streaming_term_bare_mixer_matches():
    # Bare "mixer" parallels bare "producer" / "master" in the allowlist —
    # LLMs sometimes simplify royalty_type to just "mixer" (per the extraction
    # prompt's instruction to pick from {Streaming, Master, Publishing,
    # Producer, Mixer, Remixer}).
    assert is_streaming_equivalent_royalty_type("mixer") is True


def test_streaming_term_net_receipts_matches():
    assert is_streaming_equivalent_royalty_type("net receipts") is True


def test_streaming_term_soundexchange_royalties_does_not_match():
    # SoundExchange royalties are intentionally excluded from streaming-equivalent
    # terms — they're a separate US neighbouring-rights collection stream and
    # shouldn't be lumped into streaming for splits calculations.
    assert is_streaming_equivalent_royalty_type("soundexchange royalties") is False


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
    ]:
        assert term in STREAMING_EQUIVALENT_TERMS, f"Missing expected term: {term!r}"


# ---------------------------------------------------------------------------
# is_streaming_earnable_share — allowlist + payor denylist
# ---------------------------------------------------------------------------


def test_streaming_earnable_master_share_with_clean_terms_is_true():
    share = RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0, terms="paid quarterly")
    assert is_streaming_earnable_share(share) is True


def test_streaming_earnable_publishing_share_fails_allowlist():
    share = RoyaltyShare(party_name="John Smith", royalty_type="publishing", percentage=25.0)
    assert is_streaming_earnable_share(share) is False


def test_streaming_earnable_soundexchange_in_terms_is_excluded():
    # Mirrors the live bug: LLM tagged royalty_type='streaming' but the terms
    # describe a SoundExchange Letter of Direction.
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="streaming",
        percentage=8.33,
        terms='SoundExchange Letter of Direction: 8.33% of featured performer royalties for "No Assumptions".',
    )
    assert is_streaming_earnable_share(share) is False


def test_streaming_earnable_sound_exchange_two_words_in_terms_is_excluded():
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="streaming",
        percentage=8.33,
        terms="payable via Sound Exchange for non-interactive digital performance.",
    )
    assert is_streaming_earnable_share(share) is False


def test_streaming_earnable_ascap_in_terms_is_excluded():
    share = RoyaltyShare(
        party_name="John Smith",
        royalty_type="streaming",
        percentage=50.0,
        terms="performance royalties collected via ASCAP and remitted quarterly.",
    )
    assert is_streaming_earnable_share(share) is False


def test_streaming_earnable_bmi_word_boundary_avoids_bmg_false_positive():
    # "BMI" must match as a whole word, not inside "BMG" (a label, not a PRO).
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="master",
        percentage=50.0,
        terms="all royalties submitted to BMG for distribution.",
    )
    assert is_streaming_earnable_share(share) is True


def test_streaming_earnable_bmi_as_whole_word_is_excluded():
    share = RoyaltyShare(
        party_name="John Smith",
        royalty_type="streaming",
        percentage=50.0,
        terms="public performance royalties payable by BMI.",
    )
    assert is_streaming_earnable_share(share) is False


def test_streaming_earnable_bare_payor_mention_in_eg_clause_is_not_excluded():
    # Regression for the live false-positive: a 10% Master Royalty share that
    # mentions SoundExchange only as one example of "Direct Monies" must NOT
    # be excluded by the denylist. The payor of this share is the label/company,
    # not SoundExchange — there is no payment-direction context.
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="mixer royalties",
        percentage=10.0,
        terms=(
            'Schedule 1: 10% "Master Royalty". Royalty is calculated on Gross '
            "Revenues from exploitation of the Master(s). Pro-rata if bundled "
            "with other masters; applies to Direct Monies (e.g., SoundExchange) "
            "at same percentage."
        ),
    )
    assert is_streaming_earnable_share(share) is True


def test_streaming_earnable_bare_income_from_payor_is_not_excluded():
    # "income from SoundExchange" is a contextual mention (one of several
    # income sources) — there's no payment-direction verb tying SoundExchange
    # to THIS share, so it must NOT be excluded.
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="master",
        percentage=15.0,
        terms="Royalty calculated on gross revenues, including income from SoundExchange.",
    )
    assert is_streaming_earnable_share(share) is True


def test_streaming_earnable_lod_to_payor_phrasing_is_excluded():
    # Regression-lock for the inverse LOD phrasing: "Letter of Direction to X"
    # must trip the denylist just like "X Letter of Direction" does.
    share = RoyaltyShare(
        party_name="Jane Doe",
        royalty_type="streaming",
        percentage=8.33,
        terms="Producer has issued a Letter of Direction to SoundExchange for the Master.",
    )
    assert is_streaming_earnable_share(share) is False


def test_non_streaming_payor_terms_list_includes_each_category():
    """Regression-lock: at least one representative term per category must
    remain in NON_STREAMING_PAYOR_TERMS."""
    for term in [
        "soundexchange",
        "the mlc",
        "ascap",
        "bmi",
        "sesac",
        "gema",
        "socan",
    ]:
        assert term in NON_STREAMING_PAYOR_TERMS, f"Missing expected term: {term!r}"


def test_non_streaming_payor_context_phrases_cover_each_payment_direction():
    """Regression-lock: at least one representative phrase per payment-direction
    category must remain in NON_STREAMING_PAYOR_CONTEXT_PHRASES so the
    denylist keeps catching real direct-payor relationships. Phrases use
    \\s+ as a word separator, so we check for substrings split on whitespace."""
    joined = " | ".join(NON_STREAMING_PAYOR_CONTEXT_PHRASES)
    # Each tuple is (required_keyword, human-readable description for failure message)
    required_keywords = [
        ("payable", "payable-by phrasing"),
        ("via", "via phrasing"),
        ("letter", "Letter of Direction phrasing"),
        ("shall", "shall pay phrasing"),
    ]
    for kw, desc in required_keywords:
        assert kw in joined, f"Missing payment-direction phrase ({desc}): {kw!r}"


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

    def test_publishing_only_share_raises_no_streaming_earnable_shares(self):
        # When all shares fail the streaming-earnable filter, the calculator
        # raises a structured CalculationError instead of silently returning [].
        # This lets the API surface a per-reason explanation to the user.
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="John Smith", role="Songwriter")],
            works=[Work(title="Blue Sky")],
            shares=[RoyaltyShare(party_name="John Smith", royalty_type="publishing", percentage=25.0)],
        )

        with pytest.raises(CalculationError) as exc_info:
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert exc_info.value.code == "NO_STREAMING_EARNABLE_SHARES"
        assert "streaming" in exc_info.value.user_message.lower()
        assert exc_info.value.suggestion

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

    def test_soundexchange_share_with_streaming_type_is_excluded_from_payouts(self):
        # Regression-lock for the live bug: a share extracted with
        # royalty_type='streaming' but terms describing a SoundExchange LOD
        # must NOT produce a payout from a DSP streaming statement.
        calc = _make_calculator()
        data = _contract(
            parties=[
                Party(name="Jane Doe", role="Producer"),
                Party(name="John Smith", role="Songwriter"),
            ],
            works=[Work(title="Blue Sky")],
            shares=[
                RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0),
                RoyaltyShare(
                    party_name="Jane Doe",
                    royalty_type="streaming",
                    percentage=8.33,
                    terms="SoundExchange Letter of Direction: 8.33% of featured performer royalties.",
                ),
                RoyaltyShare(party_name="John Smith", royalty_type="publishing", percentage=25.0),
            ],
        )

        payments = calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        assert len(payments) == 1
        assert payments[0].party_name == "Jane Doe"
        assert payments[0].royalty_type == "master"
        assert payments[0].amount_to_pay == 500.0

    def test_unmatched_work_raises_no_song_matches_with_song_lists(self):
        # Regression-lock for the live UX case: contract covers "Mountain Echo"
        # but the statement only has "Blue Sky" — calculator must raise
        # NO_SONG_MATCHES carrying both song lists for the frontend's
        # side-by-side comparison UI.
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Mountain Echo")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        with pytest.raises(CalculationError) as exc_info:
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})

        err = exc_info.value
        assert err.code == "NO_SONG_MATCHES"
        assert err.details["contract_works"] == ["Mountain Echo"]
        assert err.details["statement_songs"] == ["Blue Sky"]
        assert err.details["statement_song_total_count"] == 1

    def test_no_song_matches_truncates_statement_song_list_to_cap(self):
        # The details payload caps statement_songs at _STATEMENT_SONGS_PREVIEW_CAP
        # (20) so SSE payload size stays bounded. statement_song_total_count
        # preserves the true total for the "+N more" UI affordance.
        from oneclick.royalty_calculator import _STATEMENT_SONGS_PREVIEW_CAP

        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Mountain Echo")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )
        song_totals = {f"Song {i}": 100.0 for i in range(50)}

        with pytest.raises(CalculationError) as exc_info:
            calc._calculate_payments_from_data(data, song_totals)

        err = exc_info.value
        assert err.code == "NO_SONG_MATCHES"
        assert len(err.details["statement_songs"]) == _STATEMENT_SONGS_PREVIEW_CAP
        assert err.details["statement_song_total_count"] == 50

    def test_empty_works_raises_no_works_in_contract(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        # CalculationError subclasses ValueError so existing broad catchers
        # keep working; the .code is the new structured contract.
        with pytest.raises(ValueError) as exc_info:
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})
        assert isinstance(exc_info.value, CalculationError)
        assert exc_info.value.code == "NO_WORKS_IN_CONTRACT"
        assert exc_info.value.suggestion

    def test_empty_royalty_shares_raises_no_royalty_shares_in_contract(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[],
        )

        with pytest.raises(ValueError) as exc_info:
            calc._calculate_payments_from_data(data, {"Blue Sky": 1000.0})
        assert isinstance(exc_info.value, CalculationError)
        assert exc_info.value.code == "NO_ROYALTY_SHARES_IN_CONTRACT"
        assert exc_info.value.suggestion

    def test_empty_song_totals_raises_statement_empty(self):
        calc = _make_calculator()
        data = _contract(
            parties=[Party(name="Jane Doe", role="Producer")],
            works=[Work(title="Blue Sky")],
            shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="master", percentage=50.0)],
        )

        with pytest.raises(ValueError) as exc_info:
            calc._calculate_payments_from_data(data, {})
        assert isinstance(exc_info.value, CalculationError)
        assert exc_info.value.code == "STATEMENT_EMPTY"
        assert exc_info.value.suggestion
