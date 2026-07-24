"""Unit tests for oneclick.royalties.ingest — compute_statement_meta + statement_meta +
normalize_name + upsert_payee + persist_statement_rows + sync_calc_royalties (delegation
to the gated ledger_sync engine)."""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties.ingest import (
    compute_statement_meta,
    normalize_name,
    persist_statement_rows,
    statement_meta,
    upsert_payee,
)
from oneclick.royalties.ledger_sync import SyncGateError
from oneclick.royalty_calculator import StatementRow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALC_ID = "calc-0000-0000-0000-0000-000000000001"
USER_ID = "user-0000-0000-0000-0000-000000000001"
STMT_ID = "stmt-0000-0000-0000-0000-000000000001"
PROJECT_ID = "proj-0000-0000-0000-0000-000000000001"
PAYEE_ID = "pyee-0000-0000-0000-0000-000000000001"
WORK_ID = "work-0000-0000-0000-0000-000000000001"


class _TableBuilder:
    """Minimal chainable builder.  execute() is a MagicMock for per-test control.
    delete() returns a fresh _TableBuilder so delete().eq().execute() works.
    insert() records the payload and returns self.
    """

    def __init__(self, execute_data=None):
        self.execute = MagicMock(return_value=MagicMock(data=execute_data if execute_data is not None else []))
        self._insert_calls = []
        self._delete_eq_args = []

    # --- chainable helpers ---------------------------------------------------

    def select(self, *a, **kw):
        return self

    def eq(self, *a, **kw):
        self._delete_eq_args.append((a, kw))
        return self

    def insert(self, payload, **kw):
        self._insert_calls.append(payload)
        return self

    def delete(self):
        # Return self so that .delete().eq().execute() chains through self.
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self


def _make_multi_table_db(table_map: dict) -> MagicMock:
    """Return a mock db whose .table(name) returns a configured _TableBuilder.

    table_map: {table_name: _TableBuilder}
    Missing tables get an empty-result builder.
    All .table() calls are tracked via db.table.call_args_list.
    """
    db = MagicMock()

    def _side(name):
        return table_map.get(name, _TableBuilder())

    db.table.side_effect = _side
    return db


def _make_db_mock(rows: list[dict]) -> MagicMock:
    """Return a minimal Supabase-client mock whose chained call
    .table(...).select(...).eq(...).execute() returns MagicMock(data=rows).
    Mirrors the MockQueryBuilder pattern used across the test suite but kept
    inline to keep this test file self-contained and avoid conftest coupling.
    """
    builder = MagicMock()
    # Each chainable method returns the same builder so the chain resolves.
    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.execute.return_value = MagicMock(data=rows)

    db = MagicMock()
    db.table.return_value = builder
    return db


# ---------------------------------------------------------------------------
# compute_statement_meta — pure in-memory helper
# ---------------------------------------------------------------------------


class TestComputeStatementMeta:
    def test_returns_min_max_and_sum_for_dated_rows(self):
        rows = [
            StatementRow(song_title="Song A", net_payable=100.0, sale_date="2024-01-15"),
            StatementRow(song_title="Song B", net_payable=50.0, sale_date="2024-03-20"),
            StatementRow(song_title="Song C", net_payable=25.0, sale_date=None),
        ]
        result = compute_statement_meta(rows)

        assert result["period_start"] == "2024-01-15"
        assert result["period_end"] == "2024-03-20"
        assert result["statement_total"] == 175.0

    def test_empty_list_returns_none_none_zero(self):
        result = compute_statement_meta([])

        assert result == {"period_start": None, "period_end": None, "statement_total": 0}

    def test_all_null_dates_returns_none_bounds(self):
        rows = [
            StatementRow(song_title="Song A", net_payable=10.0, sale_date=None),
            StatementRow(song_title="Song B", net_payable=20.0, sale_date=None),
        ]
        result = compute_statement_meta(rows)

        assert result["period_start"] is None
        assert result["period_end"] is None
        assert result["statement_total"] == 30.0

    def test_single_row_period_start_equals_period_end(self):
        rows = [StatementRow(song_title="Song A", net_payable=42.5, sale_date="2024-06-01")]
        result = compute_statement_meta(rows)

        assert result["period_start"] == "2024-06-01"
        assert result["period_end"] == "2024-06-01"
        assert result["statement_total"] == 42.5


# ---------------------------------------------------------------------------
# statement_meta — DB-aggregation helper
# ---------------------------------------------------------------------------


class TestStatementMeta:
    def test_returns_min_max_and_sum_from_db_rows(self):
        rows = [
            {"sale_date": "2024-02-01", "net_payable": 200.0},
            {"sale_date": "2024-04-30", "net_payable": 80.0},
            {"sale_date": None, "net_payable": 15.0},
        ]
        db = _make_db_mock(rows)

        result = statement_meta(db, CALC_ID)

        assert result["period_start"] == "2024-02-01"
        assert result["period_end"] == "2024-04-30"
        assert result["statement_total"] == 295.0

        # Verify the right table and calculation_id were used.
        db.table.assert_called_once_with("royalty_statement_rows")
        db.table.return_value.select.assert_called_once_with("sale_date, net_payable")
        db.table.return_value.eq.assert_called_once_with("calculation_id", CALC_ID)

    def test_empty_data_returns_none_none_zero(self):
        db = _make_db_mock([])

        result = statement_meta(db, CALC_ID)

        assert result == {"period_start": None, "period_end": None, "statement_total": 0}

    def test_none_data_attribute_treated_as_empty(self):
        builder = MagicMock()
        builder.select.return_value = builder
        builder.eq.return_value = builder
        builder.execute.return_value = MagicMock(data=None)
        db = MagicMock()
        db.table.return_value = builder

        result = statement_meta(db, CALC_ID)

        assert result == {"period_start": None, "period_end": None, "statement_total": 0}


# ---------------------------------------------------------------------------
# normalize_name — pure helper
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_collapses_internal_whitespace_and_lowercases(self):
        assert normalize_name("  John   SMITH ") == "john smith"

    def test_empty_string_returns_empty(self):
        assert normalize_name("") == ""

    def test_none_returns_empty(self):
        assert normalize_name(None) == ""

    def test_already_normalized_is_idempotent(self):
        assert normalize_name("jane doe") == "jane doe"


# ---------------------------------------------------------------------------
# upsert_payee — DB upsert helper
# ---------------------------------------------------------------------------


class TestUpsertPayee:
    def test_existing_row_returns_id_without_insert(self):
        """When the select finds a matching row, return its id and never insert."""
        payees_builder = _TableBuilder(execute_data=[{"id": PAYEE_ID}])
        db = _make_multi_table_db({"royalty_payees": payees_builder})

        result = upsert_payee(db, USER_ID, "John Smith")

        assert result == PAYEE_ID
        # No insert payload captured.
        assert payees_builder._insert_calls == []

    def test_missing_row_inserts_and_returns_new_id(self):
        """When the select returns no rows, insert and return the new row's id."""
        new_id = "pyee-new-0000-0000-0000-0000-000000000002"
        payees_builder = _TableBuilder(execute_data=[])
        # After insert, execute() should return the new row.
        payees_builder.execute = MagicMock(
            side_effect=[
                MagicMock(data=[]),  # first call: select → empty
                MagicMock(data=[{"id": new_id}]),  # second call: insert → new row
            ]
        )
        db = _make_multi_table_db({"royalty_payees": payees_builder})

        result = upsert_payee(db, USER_ID, "Jane Doe")

        assert result == new_id
        # Insert was called once with the correct payload.
        assert len(payees_builder._insert_calls) == 1
        inserted = payees_builder._insert_calls[0]
        assert inserted["user_id"] == USER_ID
        assert inserted["normalized_name"] == "jane doe"
        assert inserted["display_name"] == "Jane Doe"

    def test_normalized_name_used_in_select(self):
        """The select filter uses the normalized (lowercased, collapsed) name."""
        payees_builder = _TableBuilder(execute_data=[{"id": PAYEE_ID}])
        # Spy on eq calls to confirm normalized_name arg.
        eq_args = []
        orig_eq = payees_builder.eq

        def _spy_eq(*a, **kw):
            eq_args.append(a)
            return orig_eq(*a, **kw)

        payees_builder.eq = _spy_eq
        db = _make_multi_table_db({"royalty_payees": payees_builder})

        upsert_payee(db, USER_ID, "  ALICE   WONDER  ")

        # One of the eq calls should filter on "alice wonder".
        assert any(a == ("normalized_name", "alice wonder") for a in eq_args)


# ---------------------------------------------------------------------------
# persist_statement_rows — delete-then-insert
# ---------------------------------------------------------------------------


class TestPersistStatementRows:
    def test_deletes_by_calculation_id_then_inserts_rows(self):
        """Existing rows deleted; new rows inserted with the passed currency."""
        stmt_builder = _TableBuilder()
        # Track what was passed to execute() at different stages.
        execute_calls = []

        def _execute():
            # Alternate: first call is delete, second is insert.
            if not execute_calls:
                execute_calls.append("delete")
                return MagicMock(data=[])
            execute_calls.append("insert")
            return MagicMock(data=[])

        stmt_builder.execute = MagicMock(side_effect=_execute)
        db = _make_multi_table_db({"royalty_statement_rows": stmt_builder})

        rows = [
            StatementRow(song_title="Track A", net_payable=50.0, sale_date="2024-01-01"),
            StatementRow(song_title="Track B", net_payable=30.0, sale_date="2024-02-01"),
        ]
        persist_statement_rows(db, USER_ID, CALC_ID, rows, "USD")

        # delete called first, then insert.
        assert execute_calls == ["delete", "insert"]
        # Insert captured 2 rows.
        assert len(stmt_builder._insert_calls) == 1
        inserted_rows = stmt_builder._insert_calls[0]
        assert len(inserted_rows) == 2
        # Currency is the passed value, not a per-row value.
        assert all(r["currency"] == "USD" for r in inserted_rows)
        assert inserted_rows[0]["song_title"] == "Track A"
        assert inserted_rows[1]["song_title"] == "Track B"

    def test_empty_rows_skips_insert(self):
        """If rows is empty, delete still runs but insert is never called."""
        stmt_builder = _TableBuilder()
        execute_calls = []

        def _execute():
            execute_calls.append("called")
            return MagicMock(data=[])

        stmt_builder.execute = MagicMock(side_effect=_execute)
        db = _make_multi_table_db({"royalty_statement_rows": stmt_builder})

        persist_statement_rows(db, USER_ID, CALC_ID, [], "GBP")

        # Only the delete execute was called, no insert.
        assert execute_calls == ["called"]
        assert stmt_builder._insert_calls == []

    def test_currency_from_arg_not_per_row(self):
        """Verify currency comes from the function arg even if StatementRow has its own field."""
        stmt_builder = _TableBuilder()
        execute_calls = []

        def _execute():
            execute_calls.append("called")
            return MagicMock(data=[])

        stmt_builder.execute = MagicMock(side_effect=_execute)
        db = _make_multi_table_db({"royalty_statement_rows": stmt_builder})

        rows = [StatementRow(song_title="X", net_payable=1.0)]
        persist_statement_rows(db, USER_ID, CALC_ID, rows, "EUR")

        inserted = stmt_builder._insert_calls[0]
        assert inserted[0]["currency"] == "EUR"


# ---------------------------------------------------------------------------
# sync_calc_royalties — loads file metadata, delegates to ledger_sync.gated_sync
# ---------------------------------------------------------------------------


class TestSyncCalcRoyalties:
    """sync_calc_royalties must ALWAYS reach the gated sync engine, even when the
    best-effort statement-rows step fails, and it must propagate SyncGateError
    (never swallow it)."""

    def test_calls_gated_sync_even_when_persist_raises(self):
        from oneclick.royalties import ingest

        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        rows = [StatementRow(song_title="A", net_payable=10.0, sale_date="2026-01-05")]
        with (
            patch.object(ingest, "persist_statement_rows", side_effect=RuntimeError("table missing")) as p_persist,
            patch("oneclick.royalties.ledger_sync.gated_sync", return_value="stmt1") as p_gated,
        ):
            ingest.sync_calc_royalties(
                db,
                "u1",
                "calc1",
                "stmt1",
                "proj1",
                {"payments": [{"party_name": "X", "song_title": "A", "amount_to_pay": 5}]},
                "USD",
                contract_ids=["c1"],
                statement_rows=rows,
            )
        p_persist.assert_called_once()
        p_gated.assert_called_once()
        # positional args: (db, user_id, calc_id, stmt_id, project_id, results, currency,
        #                   period_start, period_end, contract_ids)
        args = p_gated.call_args.args
        assert args[7] == args[8]  # fallback period: start == end
        assert isinstance(args[7], str) and len(args[7]) == 10

    def test_cache_hit_path_calls_gated_sync_when_statement_meta_raises(self):
        from oneclick.royalties import ingest

        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        with (
            patch.object(ingest, "statement_meta", side_effect=RuntimeError("table missing")),
            patch("oneclick.royalties.ledger_sync.gated_sync", return_value="stmt1") as p_gated,
        ):
            ingest.sync_calc_royalties(
                db,
                "u1",
                "calc1",
                "stmt1",
                "proj1",
                {"payments": []},
                "USD",
                contract_ids=["c1"],
                statement_rows=None,
            )
        p_gated.assert_called_once()

    def test_uses_derived_period_on_success(self):
        from oneclick.royalties import ingest

        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        rows = [
            StatementRow(song_title="A", net_payable=10.0, sale_date="2026-01-05"),
            StatementRow(song_title="B", net_payable=5.0, sale_date="2026-03-31"),
        ]
        with (
            patch.object(ingest, "persist_statement_rows"),
            patch("oneclick.royalties.ledger_sync.gated_sync", return_value="stmt1") as p_gated,
        ):
            ingest.sync_calc_royalties(
                db,
                "u1",
                "calc1",
                "stmt1",
                "proj1",
                {"payments": []},
                "USD",
                contract_ids=["c1"],
                statement_rows=rows,
            )
        args = p_gated.call_args.args
        assert args[7] == "2026-01-05"
        assert args[8] == "2026-03-31"

    def test_sync_calc_royalties_delegates_to_gated_sync(self):
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"file_name": "stmt.xlsx", "content_hash": "h1"}
        ]
        with patch("oneclick.royalties.ledger_sync.gated_sync", return_value="stmt-1") as g:
            from oneclick.royalties.ingest import sync_calc_royalties

            out = sync_calc_royalties(
                db, "u1", "calc-1", "stmt-1", "proj-1", {"payments": []}, "USD", contract_ids=["K"]
            )
        assert out == "stmt-1"
        assert g.call_args.kwargs["contract_files"]["K"]["content_hash"] == "h1"

    def test_persist_rows_false_skips_persist_but_derives_real_period(self):
        """persist_rows=False (confirm endpoint's gate-check call): rows are only
        used to derive the period, never written to royalty_statement_rows."""
        from oneclick.royalties import ingest

        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        rows = [
            StatementRow(song_title="A", net_payable=10.0, sale_date="2026-02-01"),
            StatementRow(song_title="B", net_payable=5.0, sale_date="2026-02-20"),
        ]
        with (
            patch.object(ingest, "persist_statement_rows") as p_persist,
            patch("oneclick.royalties.ledger_sync.gated_sync", return_value="stmt-1") as p_gated,
        ):
            ingest.sync_calc_royalties(
                db,
                "u1",
                "calc-1",
                "stmt-1",
                "proj-1",
                {"payments": []},
                "USD",
                contract_ids=["K"],
                statement_rows=rows,
                persist_rows=False,
            )
        p_persist.assert_not_called()
        args = p_gated.call_args.args
        assert args[7] == "2026-02-01"  # derived from the rows, not today
        assert args[8] == "2026-02-20"

    def test_sync_gate_error_propagates(self):
        """SyncGateError from gated_sync is NOT swallowed — it must reach the caller."""
        from oneclick.royalties import ingest

        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        with (
            patch("oneclick.royalties.ledger_sync.gated_sync", side_effect=SyncGateError("revision", {})),
            pytest.raises(SyncGateError) as exc_info,
        ):
            ingest.sync_calc_royalties(db, "u1", None, "stmt-1", "proj-1", {"payments": []}, "USD", contract_ids=["K"])
        assert exc_info.value.gate == "revision"
