"""Tests for net-vs-gross royalty calculation: expense allocation + payout math.

Covers:
- allocate_expenses: tagged, untagged-proportional, zero-gross equal split, mixed,
  and tagged-to-absent-track (dropped).
- _calculate_payments_from_data: gross vs net payout, mixed-basis on one song,
  net base never goes negative.
"""

import pytest

from oneclick.royalty_calculator import RoyaltyCalculator, allocate_expenses
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work

# ─── allocate_expenses ──────────────────────────────────────────────────────


class TestAllocateExpenses:
    def test_tagged_expense_assigned_to_its_song(self):
        song_totals = {"Song A": 1000.0, "Song B": 1000.0}
        expenses = [{"amount": 200.0, "work_titles": ["Song A"]}]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == 200.0
        assert result["Song B"] == 0.0

    def test_tagged_to_multiple_songs_applies_full_amount_to_each(self):
        song_totals = {"Song A": 1000.0, "Song B": 1000.0}
        expenses = [{"amount": 200.0, "work_titles": ["Song A", "Song B"]}]

        result = allocate_expenses(song_totals, expenses)

        # Full amount counts against each linked track.
        assert result["Song A"] == 200.0
        assert result["Song B"] == 200.0

    def test_untagged_allocated_proportionally_to_gross(self):
        song_totals = {"Song A": 750.0, "Song B": 250.0}
        expenses = [{"amount": 100.0, "work_titles": []}]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == pytest.approx(75.0)
        assert result["Song B"] == pytest.approx(25.0)

    def test_untagged_zero_gross_splits_equally(self):
        song_totals = {"Song A": 0.0, "Song B": 0.0}
        expenses = [{"amount": 100.0, "work_titles": []}]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == pytest.approx(50.0)
        assert result["Song B"] == pytest.approx(50.0)

    def test_tagged_to_absent_track_is_dropped(self):
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 500.0, "work_titles": ["Some Other Track"]}]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == 0.0

    def test_mixed_tagged_and_untagged(self):
        song_totals = {"Song A": 500.0, "Song B": 500.0}
        expenses = [
            {"amount": 100.0, "work_titles": ["Song A"]},  # tagged → A
            {"amount": 200.0, "work_titles": []},  # untagged → split 50/50
        ]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == pytest.approx(200.0)  # 100 + 100
        assert result["Song B"] == pytest.approx(100.0)  # 0 + 100

    def test_no_expenses_returns_zeros(self):
        song_totals = {"Song A": 100.0}
        assert allocate_expenses(song_totals, None) == {"Song A": 0.0}
        assert allocate_expenses(song_totals, []) == {"Song A": 0.0}

    def test_negative_and_invalid_amounts_ignored(self):
        song_totals = {"Song A": 100.0}
        expenses = [
            {"amount": -50.0, "work_titles": []},
            {"amount": "oops", "work_titles": []},
        ]

        result = allocate_expenses(song_totals, expenses)

        assert result["Song A"] == 0.0


# ─── payout math ────────────────────────────────────────────────────────────


def _calc():
    """A RoyaltyCalculator without invoking heavy __init__ (we only test the
    pure payout method, which uses no instance state)."""
    return RoyaltyCalculator.__new__(RoyaltyCalculator)


def _contract(shares, default_basis=None):
    return ContractData(
        parties=[Party("Alice", "producer")],
        works=[Work("Song A", "song")],
        royalty_shares=shares,
        default_basis=default_basis,
    )


class TestPayoutBasis:
    def test_gross_basis_ignores_expenses(self):
        calc = _calc()
        contract = _contract([RoyaltyShare("Alice", "streaming", 50.0, basis="gross")])
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 400.0, "work_titles": ["Song A"]}]

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        assert len(payments) == 1
        p = payments[0]
        assert p.basis == "gross"
        assert p.amount_to_pay == pytest.approx(500.0)  # 1000 * 50%
        assert p.expenses_applied == 0.0
        assert p.net_amount == pytest.approx(1000.0)

    def test_net_basis_subtracts_expenses(self):
        calc = _calc()
        contract = _contract([RoyaltyShare("Alice", "streaming", 50.0, basis="net")])
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 400.0, "work_titles": ["Song A"]}]

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        p = payments[0]
        assert p.basis == "net"
        assert p.gross_amount == pytest.approx(1000.0)
        assert p.expenses_applied == pytest.approx(400.0)
        assert p.net_amount == pytest.approx(600.0)
        assert p.amount_to_pay == pytest.approx(300.0)  # (1000-400) * 50%

    def test_default_basis_applies_when_share_silent(self):
        calc = _calc()
        contract = _contract([RoyaltyShare("Alice", "streaming", 50.0)], default_basis="net")
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 200.0, "work_titles": []}]

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        assert payments[0].basis == "net"
        assert payments[0].amount_to_pay == pytest.approx(400.0)  # (1000-200) * 50%

    def test_silent_contract_defaults_to_gross(self):
        calc = _calc()
        contract = _contract([RoyaltyShare("Alice", "streaming", 50.0)])
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 200.0, "work_titles": []}]

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        assert payments[0].basis == "gross"
        assert payments[0].amount_to_pay == pytest.approx(500.0)

    def test_mixed_basis_on_same_song(self):
        calc = _calc()
        contract = ContractData(
            parties=[Party("Alice", "producer"), Party("Bob", "artist")],
            works=[Work("Song A", "song")],
            royalty_shares=[
                RoyaltyShare("Alice", "streaming", 20.0, basis="net"),
                RoyaltyShare("Bob", "streaming", 50.0, basis="gross"),
            ],
        )
        song_totals = {"Song A": 1000.0}
        expenses = [{"amount": 200.0, "work_titles": ["Song A"]}]

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        by_party = {p.party_name: p for p in payments}
        assert by_party["Alice"].amount_to_pay == pytest.approx(160.0)  # (1000-200)*20%
        assert by_party["Bob"].amount_to_pay == pytest.approx(500.0)  # 1000*50%

    def test_net_base_never_negative(self):
        calc = _calc()
        contract = _contract([RoyaltyShare("Alice", "streaming", 50.0, basis="net")])
        song_totals = {"Song A": 100.0}
        expenses = [{"amount": 500.0, "work_titles": ["Song A"]}]  # exceeds gross

        payments = calc._calculate_payments_from_data(contract, song_totals, expenses)

        assert payments[0].net_amount == 0.0
        assert payments[0].amount_to_pay == 0.0


# ─── /oneclick/recalculate-net endpoint ─────────────────────────────────────


class TestRecalculateNetEndpoint:
    def test_recompute_net_after_expense_edit(self, client):
        """Editing the expense set re-derives net-row payouts; gross rows unchanged."""
        body = {
            "payments": [
                {
                    "song_title": "Song A",
                    "party_name": "Alice",
                    "percentage": 20.0,
                    "basis": "net",
                    "gross_amount": 1000.0,
                    "total_royalty": 1000.0,
                },
                {
                    "song_title": "Song A",
                    "party_name": "Bob",
                    "percentage": 50.0,
                    "basis": "gross",
                    "gross_amount": 1000.0,
                    "total_royalty": 1000.0,
                },
            ],
            "expenses": [{"amount": 300.0, "work_titles": ["Song A"]}],
        }

        resp = client.post("/oneclick/recalculate-net", json=body)

        assert resp.status_code == 200
        payments = {p["party_name"]: p for p in resp.json()["payments"]}
        # Net: (1000 - 300) * 20% = 140
        assert payments["Alice"]["amount_to_pay"] == pytest.approx(140.0)
        assert payments["Alice"]["expenses_applied"] == pytest.approx(300.0)
        assert payments["Alice"]["net_amount"] == pytest.approx(700.0)
        # Gross row untouched by expenses: 1000 * 50% = 500
        assert payments["Bob"]["amount_to_pay"] == pytest.approx(500.0)
        assert payments["Bob"]["expenses_applied"] == 0.0

    def test_recompute_with_no_expenses_yields_gross_for_net_rows(self, client):
        body = {
            "payments": [
                {
                    "song_title": "Song A",
                    "party_name": "Alice",
                    "percentage": 20.0,
                    "basis": "net",
                    "gross_amount": 1000.0,
                    "total_royalty": 1000.0,
                }
            ],
            "expenses": [],
        }

        resp = client.post("/oneclick/recalculate-net", json=body)

        assert resp.status_code == 200
        p = resp.json()["payments"][0]
        assert p["amount_to_pay"] == pytest.approx(200.0)  # (1000 - 0) * 20%
        assert p["net_amount"] == pytest.approx(1000.0)
