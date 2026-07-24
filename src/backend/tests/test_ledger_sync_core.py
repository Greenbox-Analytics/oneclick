"""Pure-core tests: aggregation, corroboration, conflict, resolution."""

from oneclick.royalties import ledger_sync as ls


def _p(party, song, amount, pct, sources):
    return {
        "party_name": party,
        "song_title": song,
        "royalty_type": "streaming",
        "percentage": pct,
        "amount_to_pay": amount,
        "role": "artist",
        "source_contract_ids": sources,
    }


def test_keys():
    assert ls.song_key("  Home  ") == "home"
    assert ls.type_key(None) == "streaming"
    assert ls.type_key("Digital Streaming") == "streaming"


def test_same_contract_shares_sum():
    agg, conflicts = ls.aggregate_payments(
        [_p("Kenji", "Home", 2.0, 20.0, ["K"]), _p("Kenji", "Home", 1.0, 10.0, ["K"])], ["K"]
    )
    assert not conflicts
    (entry,) = agg.values()
    assert entry["amount"] == 3.0 and entry["percentage"] == 30.0


def test_corroboration_not_additive():
    agg, conflicts = ls.aggregate_payments(
        [_p("Romes", "Home", 3.0, 30.0, ["K"]), _p("Romes", "Home", 3.0, 30.0, ["L"])], ["K", "L"]
    )
    assert not conflicts
    (entry,) = agg.values()
    assert entry["amount"] == 3.0
    assert entry["sources"] == {"K", "L"}


def test_conflict_detected_and_resolved():
    payments = [_p("Romes", "Home", 3.0, 30.0, ["K"]), _p("Romes", "Home", 3.5, 35.0, ["L"])]
    agg, conflicts = ls.aggregate_payments(payments, ["K", "L"])
    assert len(conflicts) == 1
    assert conflicts[0]["song_key"] == "home"

    resolutions = [
        {"party_key": "romes", "song_key": "home", "royalty_type_key": "streaming", "governing_contract_id": "L"}
    ]
    agg, conflicts = ls.aggregate_payments(payments, ["K", "L"], resolutions=resolutions)
    assert not conflicts
    (entry,) = agg.values()
    assert entry["amount"] == 3.5 and entry["sources"] == {"L"}


def test_resolution_governing_absent_from_all_groups_drops_identity():
    """Three-way case: two in-run contracts conflict AND the user resolves in
    favor of a stored ABSENT contract. The identity must drop out entirely
    (gate 4 preserves the stored line) — falling back to all groups would
    re-raise the in-run conflict forever."""
    payments = [_p("Romes", "Home", 3.0, 30.0, ["K"]), _p("Romes", "Home", 3.5, 35.0, ["M"])]
    resolutions = [
        {"party_key": "romes", "song_key": "home", "royalty_type_key": "streaming", "governing_contract_id": "L"}
    ]
    agg, conflicts = ls.aggregate_payments(payments, ["K", "M"], resolutions=resolutions)
    assert not conflicts and not agg


def test_legacy_payments_attribute_to_run_set():
    p = _p("Kenji", "Home", 2.0, 20.0, None)
    del p["source_contract_ids"]
    agg, conflicts = ls.aggregate_payments([p], ["K", "L"])
    (entry,) = agg.values()
    assert entry["sources"] == {"K", "L"}
