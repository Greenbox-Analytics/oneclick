"""Engine tests against the in-memory Supabase fake."""

# Spec §6 test-matrix coverage (docs/superpowers/specs/2026-07-23-oneclick-ledger-reconciliation-design.md), row → covering test:
#  1 PARTIAL test_ledger_sync_core.py::test_corroboration_not_additive — single-run corroboration; no {K}→{K,L} sequential engine run
#  2 test_ledger_sync_engine.py::test_run_scoped_authority_wipe_bug_dead
#  3 PARTIAL test_ledger_sync_engine.py::test_identical_rerun_writes_nothing — settled-stays-settled (coverage) not asserted
#  4 PARTIAL test_ledger_sync_engine.py::test_execute_replace_supersedes_and_repoints — net delta both directions not asserted
#  5 test_ledger_sync_core.py::test_conflict_detected_and_resolved
#  6 test_ledger_sync_engine.py::test_cross_run_conflict_names_absent_contract
#  7 test_ledger_sync_engine.py::test_remove_contract_sole_vs_shared
#  8 test_royalties_aggregation.py::TestOverpaymentCredit::test_overpaid_bucket_yields_credit_not_negative_owed
#  9 test_calculator_provenance.py::test_corroborating_shares_union_sources
# 10 test_calculator_provenance.py::test_share_does_not_apply_to_other_contracts_work
# 11 test_royalties_credit_netting.py::test_credit_nets_payout_total
# 12 test_ledger_sync_engine.py::test_statement_hash_adoption_reuses_bucket
# 13 PARTIAL test_oneclick_stream_gates.py::TestCacheHitSyncGates::test_cache_hit_gate_emits_needs_confirmation — one gate exercised, not every gate
# 14 test_ledger_sync_engine.py::test_tombstone_refuses_with_no_writes
# 15 test_ledger_sync_engine.py::test_empty_sources_protected_then_adopted
# 16 manual (migration SQL; run-once)
# 17 test_ledger_sync_engine.py::test_payee_locked_survives_rerun
# 18 test_royalties_credit_netting.py::test_no_cross_currency_netting
# 19 test_royalties_credit_netting.py::test_revert_blocked_after_move
# 20 test_ledger_sync_engine.py::test_contract_adoption_deferred_until_gates_pass_then_applied
# 21 test_ledger_sync_engine.py::test_null_statement_hashes_never_adopt
# 22 test_oneclick_confirm_gates.py::TestConfirmGateSequencing::test_gate_raise_returns_409_and_writes_no_cache
# 23 test_royalties_delete.py::TestDeleteProjectRoyaltyEntriesPurgeAudit::test_records_one_manual_purge_row_per_deleted_row_with_original_data
# 24 test_ledger_sync_engine.py::test_cross_run_resolution_favoring_absent_contract_preserves_stored
# 25 PARTIAL test_ledger_sync_engine.py::test_revision_none_persists_dismissal_no_reprompt + ::test_revision_dismissal_holds_reverse_direction — single-candidate pairs only
# 26 test_ledger_sync_engine.py::test_locked_line_multi_party_song_routes_only_own_party
# 27 test_ledger_sync_engine.py::test_replace_merges_coverage_on_collision
# 28 test_ledger_sync_engine.py::test_expense_change_same_percentages_no_false_conflict

import pytest

from oneclick.royalties import ledger_sync as ls
from tests.fake_supabase import USER, FakeDB


def _line(
    db,
    payee="p-kenji",
    song="home",
    amount=2.0,
    stmt="A",
    sources=(("K", "Kenji.pdf", "hk"),),
    locked=False,
    locked_party=None,
    pct=20.0,
):
    row = {
        "id": f"line-{len(db.tables['royalty_lines'].rows) + 1}",
        "user_id": USER,
        "payee_id": payee,
        "royalty_statement_id": stmt,
        "project_id": "P",
        "song_title": song.title(),
        "song_key": song,
        "royalty_type_key": "streaming",
        "royalty_type": "streaming",
        "percentage": pct,
        "amount_owed": amount,
        "statement_currency": "USD",
        "period_start": "2024-01-01",
        "period_end": "2024-03-31",
        "payee_locked": locked,
        "locked_party_key": locked_party,
        "calculation_id": None,
        # Hash/name DERIVED from the statement id: distinct statements must be
        # distinct by default, or gate 2 silently adopts them into one bucket
        # and revision/adoption tests assert against the wrong gate.
        "statement_content_hash": f"h{stmt.lower()}",
        "statement_file_name": f"stmt-{stmt.lower()}.xlsx",
        "source_contracts": [{"id": i, "name": n, "hash": h} for i, n, h in sources],
    }
    db.tables["royalty_lines"].rows.append(row)
    return row


def _payee(db, pid, name):
    db.tables["royalty_payees"].rows.append(
        {"id": pid, "user_id": USER, "normalized_name": name, "display_name": name.title()}
    )


def _sync(db, payments, cids, stmt="A", statement_file=None, **kw):
    return ls.gated_sync(
        db,
        USER,
        calculation_id="calc-9",
        royalty_statement_id=stmt,
        project_id="P",
        results={"payments": payments},
        statement_currency="USD",
        period_start="2024-01-01",
        period_end="2024-03-31",
        contract_ids=cids,
        statement_file=statement_file or {"name": "stmt-a.xlsx", "content_hash": "ha"},
        contract_files={c: {"name": f"{c}.pdf", "content_hash": f"h{c.lower()}"} for c in cids},
        **kw,
    )


def _pay(party, song, amount, sources):
    return {
        "party_name": party,
        "song_title": song,
        "royalty_type": "streaming",
        "percentage": 20.0,
        "amount_to_pay": amount,
        "role": "artist",
        "source_contract_ids": sources,
    }


def test_tombstone_refuses_with_no_writes():
    db = FakeDB()
    db.tables["royalty_statement_supersessions"].rows.append(
        {"user_id": USER, "old_statement_id": "A", "new_statement_id": "B", "kind": "superseded"}
    )
    with pytest.raises(ls.SyncGateError) as e:
        _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"])
    assert e.value.gate == "superseded"
    assert not db.tables["royalty_lines"].rows


def test_run_scoped_authority_wipe_bug_dead():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-lebron", "lebron")
    _line(db, payee="p-kenji", amount=2.0, sources=(("K", "Kenji.pdf", "hk"),))
    _line(db, payee="p-lebron", amount=2.5, sources=(("L", "Lebron.pdf", "hl"),))
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"])
    lebron = [r for r in db.tables["royalty_lines"].rows if r["payee_id"] == "p-lebron"]
    assert lebron and lebron[0]["amount_owed"] == 2.5


def test_cross_run_conflict_names_absent_contract():
    db = FakeDB()
    _payee(db, "p-romes", "romes")
    _line(db, payee="p-romes", amount=3.0, pct=30.0, sources=(("L", "Lebron Contract.pdf", "hl"),))
    with pytest.raises(ls.SyncGateError) as e:
        _sync(db, [{**_pay("Romes", "Home", 3.5, ["K"]), "percentage": 35.0}], ["K"])
    assert e.value.gate == "conflict"
    assert "Lebron Contract.pdf" in str(e.value.payload)


def test_empty_sources_protected_then_adopted():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-ghost", "ghost")
    _line(db, payee="p-ghost", song="away", amount=9.0, sources=())  # legacy, unknown owner
    _line(db, payee="p-kenji", song="home", amount=1.0, sources=())  # legacy, will be adopted
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"])
    rows = {(r["payee_id"], r["song_key"]): r for r in db.tables["royalty_lines"].rows}
    assert rows[("p-ghost", "away")]["amount_owed"] == 9.0  # never stale-deleted
    kenji = rows[("p-kenji", "home")]
    assert kenji["amount_owed"] == 2.0 and kenji["source_contracts"][0]["id"] == "K"


def test_payee_locked_survives_rerun():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-split", "kenji jr")
    _line(db, payee="p-split", amount=2.0, locked=True, locked_party="kenji", sources=(("K", "Kenji.pdf", "hk"),))
    _sync(db, [_pay("Kenji", "Home", 2.4, ["K"])], ["K"])
    rows = db.tables["royalty_lines"].rows
    assert len(rows) == 1 and rows[0]["payee_id"] == "p-split" and rows[0]["amount_owed"] == 2.4


def test_locked_line_multi_party_song_routes_only_own_party():
    """The locked match is per ORIGINAL party: Lebron's payment on the same song
    must create its own line, never land on Kenji's split line."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-lebron", "lebron")
    _payee(db, "p-split", "kenji jr")
    _line(db, payee="p-split", amount=2.0, locked=True, locked_party="kenji", sources=(("K", "Kenji.pdf", "hk"),))
    _sync(
        db,
        [
            {**_pay("Kenji", "Home", 2.4, ["K"])},
            {**_pay("Lebron", "Home", 2.5, ["K"]), "percentage": 25.0},
        ],
        ["K"],
    )
    rows = {r["payee_id"]: r for r in db.tables["royalty_lines"].rows}
    assert rows["p-split"]["amount_owed"] == 2.4  # Kenji's split line updated
    assert rows["p-lebron"]["amount_owed"] == 2.5  # Lebron got his OWN line
    assert len(db.tables["royalty_lines"].rows) == 2


def test_cross_run_resolution_favoring_absent_contract_preserves_stored():
    """Picking the absent contract's number must leave the stored line's amount
    untouched — not overwrite it with the run's number."""
    db = FakeDB()
    _payee(db, "p-romes", "romes")
    _line(db, payee="p-romes", amount=3.0, pct=30.0, sources=(("L", "Lebron Contract.pdf", "hl"),))
    _sync(
        db,
        [{**_pay("Romes", "Home", 3.5, ["K"]), "percentage": 35.0}],
        ["K"],
        conflict_resolutions=[
            {"party_key": "romes", "song_key": "home", "royalty_type_key": "streaming", "governing_contract_id": "L"}
        ],
    )
    (row,) = db.tables["royalty_lines"].rows
    assert row["amount_owed"] == 3.0 and row["percentage"] == 30.0


def test_revision_none_persists_dismissal_no_reprompt():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, stmt="B")  # other DSP, same quarter
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"], revision_decision={"none": True})
    assert any(r["kind"] == "not_related" for r in db.tables["royalty_statement_supersessions"].rows)
    # Re-run WITHOUT a decision: dismissed pair must not re-prompt.
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"])  # no SyncGateError raised


def test_expense_change_same_percentages_no_false_conflict():
    """Amounts differ (expense recalc) but percentages match → update, no gate."""
    db = FakeDB()
    _payee(db, "p-romes", "romes")
    _line(db, payee="p-romes", amount=3.0, pct=30.0, sources=(("L", "Lebron.pdf", "hl"),))
    _sync(db, [{**_pay("Romes", "Home", 2.7, ["K"]), "percentage": 30.0}], ["K"])
    (row,) = db.tables["royalty_lines"].rows
    assert row["amount_owed"] == 2.7


def test_contract_adoption_deferred_until_gates_pass_then_applied():
    """Same content-hash, new file id: a raising gate leaves sources untouched;
    a clean run re-points them with ONE 'adopted' history row."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-romes", "romes")
    # Source hash must equal what _sync derives for K_new: f"h{'K_new'.lower()}"
    _line(db, payee="p-kenji", amount=2.0, sources=(("K_old", "Kenji.pdf", "hk_new"),))
    # Absent-contract disagreement on Romes forces a cross-run conflict raise:
    _line(db, payee="p-romes", amount=3.0, pct=30.0, sources=(("L", "Lebron.pdf", "hl"),))
    payments = [
        _pay("Kenji", "Home", 2.0, ["K_new"]),
        {**_pay("Romes", "Home", 3.5, ["K_new"]), "percentage": 35.0},
    ]
    with pytest.raises(ls.SyncGateError):
        _sync(db, payments, ["K_new"])
    kenji_line = next(r for r in db.tables["royalty_lines"].rows if r["payee_id"] == "p-kenji")
    assert kenji_line["source_contracts"][0]["id"] == "K_old"  # deferred: raise = zero writes
    assert not db.tables["royalty_ledger_history"].rows

    _sync(
        db,
        payments,
        ["K_new"],
        conflict_resolutions=[
            {"party_key": "romes", "song_key": "home", "royalty_type_key": "streaming", "governing_contract_id": "L"}
        ],
    )
    kenji_line = next(r for r in db.tables["royalty_lines"].rows if r["payee_id"] == "p-kenji")
    assert kenji_line["source_contracts"][0]["id"] == "K_new"  # adopted
    assert [h for h in db.tables["royalty_ledger_history"].rows if h["action"] == "adopted"]


def test_identical_rerun_writes_nothing():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, sources=(("K", "Kenji.pdf", "hk"),))
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"])
    assert not db.tables["royalty_ledger_history"].rows


def test_execute_replace_supersedes_and_repoints():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, stmt="A")
    db.tables["royalty_payouts"].rows.append({"id": "po-1", "user_id": USER, "status": "paid"})
    db.tables["royalty_payout_coverage"].rows.append(
        {
            "id": "cov-1",
            "payout_id": "po-1",
            "payee_id": "p-kenji",
            "project_id": "P",
            "royalty_statement_id": "A",
            "covered_amount": 2.0,
            "moved_from": None,
        }
    )
    db.tables["royalty_calculations"].rows.append({"id": "calc-old", "royalty_statement_id": "A", "user_id": USER})
    ls.execute_replace(db, USER, old_statement_id="A", new_statement_id="B", project_id="P")
    assert not [r for r in db.tables["royalty_lines"].rows if r["royalty_statement_id"] == "A"]
    cov = db.tables["royalty_payout_coverage"].rows[0]
    assert cov["royalty_statement_id"] == "B" and cov["moved_from"]["statement_id"] == "A"
    assert db.tables["royalty_statement_supersessions"].rows[0]["old_statement_id"] == "A"
    assert not db.tables["royalty_calculations"].rows
    actions = {h["action"] for h in db.tables["royalty_ledger_history"].rows}
    assert {"superseded", "coverage_moved"} <= actions


def test_statement_hash_adoption_reuses_bucket():
    """Row 12: reuploaded statement with a known content hash adopts the OLD
    bucket silently — no new statement bucket, the OLD line is updated."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    row = _line(db, payee="p-kenji", amount=2.0, stmt="OLD")
    row["statement_content_hash"] = "ha"  # same bytes as the run's statement A
    effective = _sync(db, [_pay("Kenji", "Home", 2.4, ["K"])], ["K"], stmt="A")
    assert effective == "OLD"
    (line,) = db.tables["royalty_lines"].rows
    assert line["royalty_statement_id"] == "OLD" and line["amount_owed"] == 2.4


def test_null_statement_hashes_never_adopt():
    """Row 21: null content hashes on both sides never match — no silent merge;
    the overlap falls through to the revision gate instead."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    row = _line(db, payee="p-kenji", amount=2.0, stmt="B")
    row["statement_content_hash"] = None
    with pytest.raises(ls.SyncGateError) as e:
        _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"], statement_file={"name": "x", "content_hash": None})
    assert e.value.gate == "revision"


def test_replace_merges_coverage_on_collision():
    """Row 27: the payout already covers the NEW bucket → amounts merge into one
    row (summed, moved_from SET), source row deleted, move history recorded."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, stmt="A")
    db.tables["royalty_payouts"].rows.append({"id": "po-1", "user_id": USER, "status": "paid"})
    for cov_id, stmt, amount in (("cov-a", "A", 2.0), ("cov-b", "B", 3.0)):
        db.tables["royalty_payout_coverage"].rows.append(
            {
                "id": cov_id,
                "payout_id": "po-1",
                "payee_id": "p-kenji",
                "project_id": "P",
                "royalty_statement_id": stmt,
                "covered_amount": amount,
                "moved_from": None,
            }
        )
    ls.execute_replace(db, USER, old_statement_id="A", new_statement_id="B", project_id="P")
    (cov,) = db.tables["royalty_payout_coverage"].rows
    assert cov["id"] == "cov-b" and cov["royalty_statement_id"] == "B"
    assert cov["covered_amount"] == 5.0
    assert cov["moved_from"] == {"statement_id": "A", "project_id": "P", "action": "revision_replace"}
    assert any(h["action"] == "coverage_moved" for h in db.tables["royalty_ledger_history"].rows)


def test_revision_dismissal_holds_reverse_direction():
    """Row 25 (reverse): after dismissing A↔B as unrelated during A's sync, a
    later sync FOR B overlapping A's lines must not re-prompt."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, stmt="B")
    _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"], revision_decision={"none": True})
    assert any(r["kind"] == "not_related" for r in db.tables["royalty_statement_supersessions"].rows)
    # Statement A now holds lines; sync FOR B overlaps them — no gate raised.
    _sync(
        db,
        [_pay("Kenji", "Home", 2.1, ["K"])],
        ["K"],
        stmt="B",
        statement_file={"name": "stmt-b.xlsx", "content_hash": "hb"},
    )


def test_replace_decision_not_among_candidates_fails_closed():
    """C1: a client-supplied replace id the gate never offered must raise, not
    execute — no supersession, no line deletion."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _line(db, payee="p-kenji", amount=2.0, stmt="B")  # real candidate is B
    with pytest.raises(ls.SyncGateError) as e:
        _sync(db, [_pay("Kenji", "Home", 2.0, ["K"])], ["K"], revision_decision={"replace": "unrelated-stmt-id"})
    assert e.value.gate == "revision"
    assert not db.tables["royalty_statement_supersessions"].rows
    (line,) = db.tables["royalty_lines"].rows
    assert line["royalty_statement_id"] == "B" and line["amount_owed"] == 2.0


def test_execute_replace_scoped_to_callers_payouts():
    """C1: coverage rows hanging off ANOTHER user's payout are never re-pointed."""
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    db.tables["royalty_payouts"].rows.append({"id": "po-mine", "user_id": USER, "status": "paid"})
    db.tables["royalty_payouts"].rows.append({"id": "po-other", "user_id": "other", "status": "paid"})
    for cov_id, payout in (("cov-mine", "po-mine"), ("cov-other", "po-other")):
        db.tables["royalty_payout_coverage"].rows.append(
            {
                "id": cov_id,
                "payout_id": payout,
                "payee_id": "p-kenji",
                "project_id": "P",
                "royalty_statement_id": "A",
                "covered_amount": 2.0,
                "moved_from": None,
            }
        )
    ls.execute_replace(db, USER, old_statement_id="A", new_statement_id="B", project_id="P")
    by_id = {r["id"]: r for r in db.tables["royalty_payout_coverage"].rows}
    assert by_id["cov-mine"]["royalty_statement_id"] == "B"
    assert by_id["cov-other"]["royalty_statement_id"] == "A" and by_id["cov-other"]["moved_from"] is None


def test_remove_contract_sole_vs_shared():
    db = FakeDB()
    _payee(db, "p-kenji", "kenji")
    _payee(db, "p-romes", "romes")
    _line(db, payee="p-kenji", amount=2.0, sources=(("K", "Kenji.pdf", "hk"),))
    _line(db, payee="p-romes", amount=3.0, sources=(("K", "Kenji.pdf", "hk"), ("L", "Lebron.pdf", "hl")))
    ls.remove_contract_from_ledger(db, USER, "K")
    rows = db.tables["royalty_lines"].rows
    assert len(rows) == 1 and rows[0]["payee_id"] == "p-romes"
    assert [e["id"] for e in rows[0]["source_contracts"]] == ["L"]
    actions = [h["action"] for h in db.tables["royalty_ledger_history"].rows]
    assert "deleted" in actions and "source_removed" in actions
