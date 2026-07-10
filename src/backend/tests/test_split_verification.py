"""Unit tests for the advisory split-verification pass (blind report -> Python verdict)."""

import json
from unittest.mock import MagicMock

from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work
from utils.contract_parsing.split_verification import verify_royalty_shares

CONTRACT = (
    "ROYALTIES. Label shall pay Artist fifty percent (50%) of Net Receipts from streaming. "
    "Producer shall receive a royalty of twenty percent (20%) of gross receipts."
)


def _client_returning(payload: dict):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return client


def _contract_data(shares, default_basis=None):
    return ContractData(
        parties=[Party(name="Artist", role="Artist"), Party(name="Producer", role="Producer")],
        works=[Work(title="Song A")],
        royalty_shares=shares,
        default_basis=default_basis,
    )


def _finding(
    party="Artist",
    rtype="Streaming",
    pct=50.0,
    basis="net",
    quote="pay Artist fifty percent (50%) of Net Receipts from streaming",
):
    return {
        "party_name": party,
        "royalty_type": rtype,
        "contract_percentage": pct,
        "contract_basis": basis,
        "contract_quote": quote,
        "note": "Artist gets 50% of net.",
    }


def _verify(shares, payload, contract=CONTRACT, default_basis=None):
    return verify_royalty_shares(
        contract,
        _contract_data(shares, default_basis=default_basis),
        openai_client=_client_returning(payload),
    )


# ── Verdicts ─────────────────────────────────────────────────────────────────────


def test_correct_extraction_is_verified():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    review = _verify(shares, {"findings": [_finding()]})
    assert review.overall == "verified"
    assert review.checked == 1 and review.flagged == 0
    assert review.findings[0].verdict == "verified"


def test_wrong_percentage_is_mismatch():
    # Extraction said 40%, contract (and blind report) say 50%.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=40.0, basis="net")]
    review = _verify(shares, {"findings": [_finding(pct=50.0)]})
    assert review.overall == "needs_review"
    assert review.findings[0].verdict == "mismatch"
    assert review.findings[0].contract_percentage == 50.0


def test_pct_match_but_basis_mismatch_is_mismatch():
    # Contract says net; extraction (effective basis) is gross -> money-relevant mismatch.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="gross")]
    review = _verify(shares, {"findings": [_finding(basis="net")]})
    assert review.findings[0].verdict == "mismatch"


def test_contract_silent_on_basis_is_not_flagged():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    review = _verify(shares, {"findings": [_finding(basis=None)]})
    assert review.findings[0].verdict == "verified"


def test_basis_compared_against_effective_basis():
    # Share is silent; contract default_basis="net"; report says net -> consistent -> verified.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis=None)]
    review = _verify(shares, {"findings": [_finding(basis="net")]}, default_basis="net")
    assert review.findings[0].verdict == "verified"


# ── Guardrails ───────────────────────────────────────────────────────────────────


def test_hallucinated_quote_is_unverified():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = _verify(shares, {"findings": [_finding(quote="Artist shall receive 50% of everything forever")]})
    assert review.findings[0].verdict == "unverified"
    assert review.overall == "needs_review"


def test_real_quote_but_number_absent_is_unverified():
    # Quote is verbatim-present but contains no "50" -> number-in-quote guardrail fires.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = _verify(shares, {"findings": [_finding(quote="Producer shall receive a royalty of twenty percent")]})
    assert review.findings[0].verdict == "unverified"


def test_non_finite_percentage_is_unverified():
    # 1e999 is valid JSON (parses to inf); NaN parses too. Neither may reach the
    # serialized payload — json.dumps would emit Infinity/NaN and break JSON.parse.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    for bad in (float("nan"), float("inf")):
        review = _verify(shares, {"findings": [_finding(pct=bad)]})
        assert review.findings[0].verdict == "unverified"
        assert review.findings[0].contract_percentage is None


def test_float_percentage_matches_integer_digits_in_quote():
    # 50.0 must match "50%" in the quote (formatting normalization).
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    review = _verify(shares, {"findings": [_finding(pct=50.0)]})
    assert review.findings[0].verdict == "verified"


def test_word_form_only_percentage_is_unverified():
    # Contract states the number only in words -> digit check can't confirm -> safe direction.
    contract = "ROYALTIES. Label shall pay Artist fifty percent of Net Receipts from streaming revenue."
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    payload = {"findings": [_finding(quote="pay Artist fifty percent of Net Receipts from streaming revenue")]}
    review = _verify(shares, payload, contract=contract)
    assert review.findings[0].verdict == "unverified"


def test_prompt_never_contains_extracted_values():
    # Anchoring guard: the outgoing prompt must not leak the values under test.
    # The PAIRS section is the only per-share content — extracted percentage/basis
    # must never appear there (the static rules text legitimately mentions "net"/"gross").
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=41.7, basis="net")]
    client = _client_returning({"findings": []})
    verify_royalty_shares(CONTRACT, _contract_data(shares), openai_client=client)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    pairs_section = prompt.split("PAIRS TO LOOK UP:")[1].split("Rules:")[0]
    assert "Artist" in pairs_section and "Streaming" in pairs_section
    assert "41.7" not in prompt  # extracted percentage appears nowhere at all
    assert "net" not in pairs_section.lower()  # extracted basis withheld from the lookup list


def test_no_report_for_share_is_unverified():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = _verify(shares, {"findings": []})
    assert review.findings[0].verdict == "unverified"
    assert review.overall == "needs_review"


def test_model_rephrased_party_name_still_matches():
    # The model may expand/shorten a name ("Artist" -> "The Artist"); the unambiguous
    # containment fallback must still match the report to the share.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    review = _verify(shares, {"findings": [_finding(party="The Artist")]})
    assert review.findings[0].verdict == "verified"


def test_ambiguous_rephrased_name_stays_unverified():
    # Two reports whose names both contain the share's name -> ambiguous -> no match.
    shares = [RoyaltyShare(party_name="Ali", royalty_type="Streaming", percentage=50.0, basis="net")]
    payload = {"findings": [_finding(party="Ali Prime"), _finding(party="Ali Beats")]}
    review = _verify(shares, payload)
    assert review.findings[0].verdict == "unverified"


# ── Degradation ──────────────────────────────────────────────────────────────────


def test_llm_error_returns_unavailable():
    client = MagicMock()
    client.chat.completions.create.side_effect = TimeoutError("deadline")
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = verify_royalty_shares(CONTRACT, _contract_data(shares), openai_client=client)
    assert review.overall == "unavailable"
    assert review.findings == []


def test_empty_markdown_returns_unavailable():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = verify_royalty_shares("", _contract_data(shares), openai_client=MagicMock())
    assert review.overall == "unavailable"


def test_no_shares_returns_unavailable():
    review = verify_royalty_shares(CONTRACT, _contract_data([]), openai_client=MagicMock())
    assert review.overall == "unavailable"


def test_timeout_is_passed_to_openai_call():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    client = _client_returning({"findings": []})
    verify_royalty_shares(CONTRACT, _contract_data(shares), openai_client=client, timeout=10.0)
    assert client.chat.completions.create.call_args.kwargs["timeout"] == 10.0


def test_fallback_never_steals_another_shares_report():
    # C1 pin: extraction wrongly gave Ali the 20% that belongs to Ali Beats; the model
    # reported only the Ali Beats clause. Ali must NOT verify against that report.
    contract = (
        "Ali shall receive fifty percent (50%) of streaming income. "
        "Ali Beats shall receive twenty percent (20%) of streaming income."
    )
    shares = [
        RoyaltyShare(party_name="Ali", royalty_type="Streaming", percentage=20.0),
        RoyaltyShare(party_name="Ali Beats", royalty_type="Streaming", percentage=20.0),
    ]
    payload = {
        "findings": [
            _finding(
                party="Ali Beats",
                rtype="Streaming",
                pct=20.0,
                basis=None,
                quote="Ali Beats shall receive twenty percent (20%) of streaming income",
            )
        ]
    }
    review = _verify(shares, payload, contract=contract)
    by_party = {f.party_name: f for f in review.findings}
    assert by_party["Ali Beats"].verdict == "verified"
    assert by_party["Ali"].verdict == "unverified"
    assert review.overall == "needs_review"


def test_capitalized_basis_still_flags_mismatch():
    # C2 pin: model says "Net" (capitalized); extracted effective basis is gross -> mismatch.
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="gross")]
    review = _verify(shares, {"findings": [_finding(basis="Net")]})
    assert review.findings[0].verdict == "mismatch"


def test_capitalized_basis_match_verifies():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    review = _verify(shares, {"findings": [_finding(basis="NET")]})
    assert review.findings[0].verdict == "verified"


def test_top_level_list_output_is_unavailable():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="[]"))])
    review = verify_royalty_shares(CONTRACT, _contract_data(shares), openai_client=client)
    assert review.overall == "unavailable"


def test_non_list_findings_is_unavailable():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = _verify(shares, {"findings": 5})
    assert review.overall == "unavailable"


def test_numeric_quote_degrades_not_raises():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0)]
    review = _verify(
        shares,
        {
            "findings": [
                {
                    "party_name": "Artist",
                    "royalty_type": "Streaming",
                    "contract_percentage": 50.0,
                    "contract_basis": None,
                    "contract_quote": 50,
                    "note": None,
                }
            ]
        },
    )
    assert review.findings[0].verdict == "unverified"


def test_boolean_percentage_is_not_a_number():
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=1.0)]
    review = _verify(shares, {"findings": [_finding(pct=True)]})
    assert review.findings[0].verdict == "unverified"


def test_pct_in_quote_boundaries():
    from utils.contract_parsing.split_verification import _pct_in_quote

    assert _pct_in_quote(50.0, "a rate of fifty percent (50%) of receipts")
    assert _pct_in_quote(50.0, "equal to 50 percent of net receipts")
    assert _pct_in_quote(12.5, "twelve and a half percent (12.5%)")
    assert not _pct_in_quote(50.0, "an advance recoupable at 150% of cost")
    assert not _pct_in_quote(50.0, "a rate of 50.5% of receipts")
    assert not _pct_in_quote(50.0, "within 50 days of receipt")
    assert not _pct_in_quote(50.0, "as described in Section 50.")


def test_duplicate_party_type_shares_cannot_both_verify():
    # Two shares for the same (party, type) with different percentages judged against one
    # report: at most one can verify; the overall must flag.
    shares = [
        RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net"),
        RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=30.0, basis="net"),
    ]
    review = _verify(shares, {"findings": [_finding(pct=50.0)]})
    verdicts = sorted(f.verdict for f in review.findings)
    assert verdicts == ["mismatch", "verified"]
    assert review.overall == "needs_review"


def test_quote_normalization_tolerates_pdf_artifacts():
    # Curly quotes / em-dash in the model's quote vs straight/hyphen in the contract.
    contract = 'ROYALTIES. The "Artist" shall receive fifty percent (50%) - of Net Receipts.'
    shares = [RoyaltyShare(party_name="Artist", royalty_type="Streaming", percentage=50.0, basis="net")]
    payload = {"findings": [_finding(quote="The “Artist” shall receive fifty percent (50%) — of Net Receipts")]}
    review = _verify(shares, payload, contract=contract)
    assert review.findings[0].verdict == "verified"


def test_fuzzy_report_claimable_by_two_shares_matches_neither():
    # A truncated report name ("Ali B") containment-matches BOTH shares -> ambiguous
    # attribution -> neither share may verify from it.
    contract = (
        "Ali shall receive fifty percent (50%) of streaming income. "
        "Ali Beats shall receive twenty percent (20%) of streaming income."
    )
    shares = [
        RoyaltyShare(party_name="Ali", royalty_type="Streaming", percentage=20.0),
        RoyaltyShare(party_name="Ali Beats", royalty_type="Streaming", percentage=20.0),
    ]
    payload = {
        "findings": [
            _finding(
                party="Ali B",
                rtype="Streaming",
                pct=20.0,
                basis=None,
                quote="Ali Beats shall receive twenty percent (20%) of streaming income",
            )
        ]
    }
    review = _verify(shares, payload, contract=contract)
    assert all(f.verdict == "unverified" for f in review.findings)
    assert review.overall == "needs_review"
