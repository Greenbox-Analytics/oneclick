"""Tests for delete_project_royalty_entries in oneclick/royalties/service.py,
the DELETE /projects/{project_id}/entries router endpoint, the
GET /contracts/{contract_id}/impact router endpoint, and the ledger-guardrail
wiring in main.py's DELETE /contracts/{contract_id} endpoint.

All tests mock the Supabase client (or use the in-memory FakeDB). No real DB
or network calls.

Key invariants verified:
  1. Coverage rows are deleted by project_id.
  2. Calculations are deleted by their ids (not by project_id directly).
  3. royalty_lines ARE deleted explicitly by project_id (they outlive their calc
     under ON DELETE SET NULL); royalty_statement_rows is left as-is.
  4. When there are no calcs, the calc-delete call is skipped.
  5. Returns {"deleted_calculations": N, "project_id": ...}.
  6. Router returns 403 when caller does not own the project.
  7. Router returns 200 and calls the service when the caller owns the project.
  8. GET /contracts/{contract_id}/impact reports lines/backed-amounts/paid-buckets.
  9. delete_project_royalty_entries history-records every purged coverage/line row.
  10. DELETE /contracts/{contract_id} calls remove_contract_from_ledger before the
      project_files delete, and aborts with 500 (no project_files delete) if it raises.
"""

from unittest.mock import MagicMock, patch

from oneclick.royalties import service
from tests.conftest import MockQueryBuilder
from tests.fake_supabase import FakeDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-aaa"
PROJECT_ID = "proj-111"
CALC_ID_1 = "calc-aaa"
CALC_ID_2 = "calc-bbb"
CONTRACT_ID = "contract-999"
ARTIST_ID = "artist-001"

# ---------------------------------------------------------------------------
# MockDB
# ---------------------------------------------------------------------------


class MockDB:
    """Minimal mock of the Supabase client for delete tests.

    Tracks:
      deleted_tables: list of (table_name, filter_type, filter_value) tuples
                      representing every .delete().execute() call.
    """

    def __init__(self, calc_rows=None, payout_rows=None):
        # calc_rows is what royalty_calculations.select().eq().eq().execute() returns
        self.calc_rows = calc_rows if calc_rows is not None else []
        # payout_rows is what royalty_payouts.select().eq().execute() returns.
        # Default to one payout so coverage delete is not skipped in baseline tests.
        self.payout_rows = payout_rows if payout_rows is not None else [{"id": "payout-default", "user_id": USER_ID}]

        # Side-effect capture
        self.deleted_tables: list[tuple] = []  # (table, filter_col, filter_val)
        self._select_table = None

    def table(self, name):
        return _TableProxy(self, name)


class _TableProxy:
    def __init__(self, db: MockDB, name: str):
        self._db = db
        self._name = name
        self._filters: dict = {}
        self._in_filter: tuple | None = None  # (col, vals)
        self._pending_delete = False

    def select(self, *a, **kw):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def in_(self, col, vals):
        self._in_filter = (col, list(vals))
        return self

    def delete(self):
        self._pending_delete = True
        return self

    def execute(self):
        db = self._db
        name = self._name

        if self._pending_delete:
            # Record eq filters
            for col, val in self._filters.items():
                db.deleted_tables.append((name, "eq", col, val))
            # Record in_ filter if present
            if self._in_filter:
                col, vals = self._in_filter
                db.deleted_tables.append((name, "in_", col, vals))
            return MagicMock(data=[])

        # SELECT — royalty_calculations and royalty_payouts are queried in the service function
        if name == "royalty_calculations":
            rows = db.calc_rows
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        if name == "royalty_payouts":
            rows = db.payout_rows
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        return MagicMock(data=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calc(calc_id, project_id=PROJECT_ID, user_id=USER_ID):
    return {"id": calc_id, "project_id": project_id, "user_id": user_id}


# ---------------------------------------------------------------------------
# Tests: service.delete_project_royalty_entries
# ---------------------------------------------------------------------------


class TestDeleteProjectRoyaltyEntries:
    def test_returns_correct_count_and_project_id(self):
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1), _make_calc(CALC_ID_2)])
        result = service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        assert result == {"deleted_calculations": 2, "project_id": PROJECT_ID}

    def test_deletes_coverage_by_project_id(self):
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1)])
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        cov_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_payout_coverage"]
        # There are now two delete filter records: one eq(project_id) and one in_(payout_id)
        assert len(cov_deletes) >= 1
        # Must be filtered by project_id (eq filter present)
        eq_deletes = [e for e in cov_deletes if e[1] == "eq" and e[2] == "project_id"]
        assert len(eq_deletes) == 1
        assert eq_deletes[0][3] == PROJECT_ID
        # Must also be scoped to user's payout ids (in_ filter present)
        in_deletes = [e for e in cov_deletes if e[1] == "in_" and e[2] == "payout_id"]
        assert len(in_deletes) == 1

    def test_deletes_calcs_by_id_list(self):
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1), _make_calc(CALC_ID_2)])
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        calc_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_calculations"]
        assert len(calc_deletes) == 1
        _, filter_type, col, vals = calc_deletes[0]
        assert filter_type == "in_"
        assert col == "id"
        assert set(vals) == {CALC_ID_1, CALC_ID_2}

    def test_no_calcs_skips_calc_delete_call(self):
        """When there are no calculations, we must NOT call .delete() on royalty_calculations."""
        db = MockDB(calc_rows=[])
        result = service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        calc_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_calculations"]
        assert calc_deletes == [], "Must not call delete on royalty_calculations when calc_ids is empty"
        assert result == {"deleted_calculations": 0, "project_id": PROJECT_ID}

    def test_no_calcs_still_deletes_coverage(self):
        """Even with zero calcs, coverage for the project should be cleared (when payouts exist)."""
        db = MockDB(calc_rows=[])  # uses default payout so coverage delete is not skipped
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        cov_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_payout_coverage"]
        # At least the eq(project_id) filter entry must be present
        assert any(e[1] == "eq" and e[2] == "project_id" for e in cov_deletes)

    def test_deletes_royalty_lines_by_project(self):
        """royalty_lines ARE deleted explicitly by project_id — under ON DELETE SET NULL
        they outlive their calc, so the calc deletion would not remove them."""
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1)])
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        line_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_lines"]
        assert line_deletes, "royalty_lines must be deleted explicitly by project_id"

    def test_does_not_delete_royalty_statement_rows_explicitly(self):
        """royalty_statement_rows is not deleted here (it carries no project_id; any
        orphaned breakdown rows are harmless and don't affect earned/owed)."""
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1)])
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        stmt_row_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_statement_rows"]
        assert stmt_row_deletes == [], "royalty_statement_rows is not deleted by this function"

    def test_does_not_delete_royalty_payouts(self):
        """royalty_payouts must NOT be deleted — orphan state is derived at read time."""
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1)])
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        payout_deletes = [entry for entry in db.deleted_tables if entry[0] == "royalty_payouts"]
        assert payout_deletes == [], "royalty_payouts must never be deleted by this function"

    def test_single_calc_returns_count_one(self):
        db = MockDB(calc_rows=[_make_calc(CALC_ID_1)])
        result = service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)
        assert result["deleted_calculations"] == 1

    def test_project_id_in_return_matches_input(self):
        db = MockDB(calc_rows=[])
        result = service.delete_project_royalty_entries(db, USER_ID, "some-other-project")
        assert result["project_id"] == "some-other-project"


# ---------------------------------------------------------------------------
# Tests: router HTTP layer (ownership + smoke)
# ---------------------------------------------------------------------------


def _make_router_client(
    owns_project: bool = True,
    service_return=None,
    service_side_effect=None,
):
    """Build a minimal FastAPI app with the royalties router. Auth/gating bypassed.

    Patches:
      - auth dependency → returns USER_ID
      - gated_feature → no-op
      - _get_supabase → trivial MagicMock
      - oneclick.royalties.router._verify_owns_project → controlled return
      - service.delete_project_royalty_entries → controlled return/side_effect
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth import get_current_user_id
    from oneclick.royalties.router import router

    app = FastAPI()
    app.include_router(router)

    async def _mock_user_id():
        return USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    gating_patcher = patch("oneclick.royalties.router.gated_feature", return_value=None)
    supabase_patcher = patch("oneclick.royalties.router._get_supabase", return_value=MagicMock())
    ownership_patcher = patch(
        "oneclick.royalties.router._verify_owns_project",
        return_value=owns_project,
    )

    if service_side_effect is not None:
        svc_patcher = patch(
            "oneclick.royalties.service.delete_project_royalty_entries",
            side_effect=service_side_effect,
        )
    else:
        svc_patcher = patch(
            "oneclick.royalties.service.delete_project_royalty_entries",
            return_value=service_return
            if service_return is not None
            else {"deleted_calculations": 0, "project_id": PROJECT_ID},
        )

    gating_patcher.start()
    supabase_patcher.start()
    ownership_patcher.start()
    svc_patcher.start()

    client = TestClient(app, raise_server_exceptions=False)

    yield client

    gating_patcher.stop()
    supabase_patcher.stop()
    ownership_patcher.stop()
    svc_patcher.stop()


class TestDeleteProjectEntriesRouter:
    def test_non_owner_returns_403(self):
        """When _verify_owns_project returns False, endpoint must return 403."""
        for client in _make_router_client(owns_project=False):
            resp = client.delete(f"/projects/{PROJECT_ID}/entries")
            assert resp.status_code == 403

    def test_owner_returns_200(self):
        """When _verify_owns_project returns True, endpoint must return 200."""
        for client in _make_router_client(owns_project=True):
            resp = client.delete(f"/projects/{PROJECT_ID}/entries")
            assert resp.status_code == 200

    def test_owner_response_contains_deleted_calculations(self):
        """200 response body must include deleted_calculations and project_id."""
        return_val = {"deleted_calculations": 3, "project_id": PROJECT_ID}
        for client in _make_router_client(owns_project=True, service_return=return_val):
            resp = client.delete(f"/projects/{PROJECT_ID}/entries")
            assert resp.status_code == 200
            body = resp.json()
            assert body["deleted_calculations"] == 3
            assert body["project_id"] == PROJECT_ID

    def test_service_is_called_for_owner(self):
        """Service function must be invoked when the caller owns the project."""
        with (
            patch(
                "oneclick.royalties.service.delete_project_royalty_entries",
                return_value={"deleted_calculations": 0, "project_id": PROJECT_ID},
            ) as mock_svc,
            patch("oneclick.royalties.router.gated_feature", return_value=None),
            patch("oneclick.royalties.router._get_supabase", return_value=MagicMock()),
            patch("oneclick.royalties.router._verify_owns_project", return_value=True),
        ):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from auth import get_current_user_id
            from oneclick.royalties.router import router

            app = FastAPI()
            app.include_router(router)

            async def _mock_user_id():
                return USER_ID

            app.dependency_overrides[get_current_user_id] = _mock_user_id
            client = TestClient(app, raise_server_exceptions=False)
            client.delete(f"/projects/{PROJECT_ID}/entries")
            mock_svc.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: GET /contracts/{contract_id}/impact (router, FakeDB)
# ---------------------------------------------------------------------------


def _make_impact_router_client(db):
    """Build a minimal FastAPI app with the royalties router wired to `db` (e.g. FakeDB).

    No ownership patch needed — the endpoint scopes every query to user_id itself
    (see router docstring), so only auth/gating/db need to be stubbed.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth import get_current_user_id
    from oneclick.royalties.router import router

    app = FastAPI()
    app.include_router(router)

    async def _mock_user_id():
        return USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    gating_patcher = patch("oneclick.royalties.router.gated_feature", return_value=None)
    supabase_patcher = patch("oneclick.royalties.router._get_supabase", return_value=db)

    gating_patcher.start()
    supabase_patcher.start()

    client = TestClient(app, raise_server_exceptions=False)

    yield client

    gating_patcher.stop()
    supabase_patcher.stop()


class TestContractLedgerImpactRouter:
    def test_reports_lines_backed_currencies_and_paid_bucket_coverage(self):
        db = FakeDB()
        db.tables["royalty_lines"].rows.extend(
            [
                {
                    "id": "line-1",
                    "user_id": USER_ID,
                    "source_contracts": [{"id": CONTRACT_ID}],
                    "statement_currency": "USD",
                    "amount_owed": 100,
                    "payee_id": "payee-1",
                    "royalty_statement_id": "stmt-1",
                    "project_id": PROJECT_ID,
                },
                {
                    "id": "line-2",
                    "user_id": USER_ID,
                    "source_contracts": [{"id": CONTRACT_ID}],
                    "statement_currency": "EUR",
                    "amount_owed": 50,
                    "payee_id": "payee-2",
                    "royalty_statement_id": "stmt-2",
                    "project_id": PROJECT_ID,
                },
            ]
        )
        db.tables["royalty_payouts"].rows.append({"id": "payout-1", "user_id": USER_ID, "status": "paid"})
        # Only payee-1's bucket has paid coverage — payee-2's does not.
        db.tables["royalty_payout_coverage"].rows.append(
            {
                "payout_id": "payout-1",
                "payee_id": "payee-1",
                "royalty_statement_id": "stmt-1",
                "project_id": PROJECT_ID,
            }
        )

        for client in _make_impact_router_client(db):
            resp = client.get(f"/contracts/{CONTRACT_ID}/impact")
            assert resp.status_code == 200
            assert resp.json() == {
                "lines": 2,
                "backed": {"USD": 100.0, "EUR": 50.0},
                "buckets_with_paid_coverage": 1,
            }

    def test_no_matching_lines_returns_all_zero(self):
        for client in _make_impact_router_client(FakeDB()):
            resp = client.get(f"/contracts/{CONTRACT_ID}/impact")
            assert resp.status_code == 200
            assert resp.json() == {"lines": 0, "backed": {}, "buckets_with_paid_coverage": 0}


# ---------------------------------------------------------------------------
# Tests: delete_project_royalty_entries purge audit trail (FakeDB)
# ---------------------------------------------------------------------------


def _seed_purge_fixture():
    """FakeDB with one paid payout, two covered buckets, and two royalty_lines —
    all scoped to PROJECT_ID/USER_ID — for delete_project_royalty_entries to purge."""
    db = FakeDB()
    db.tables["royalty_payouts"].rows.append({"id": "payout-1", "user_id": USER_ID, "status": "paid"})
    db.tables["royalty_payout_coverage"].rows.extend(
        [
            {"id": "cov-1", "project_id": PROJECT_ID, "payout_id": "payout-1", "payee_id": "payee-1"},
            {"id": "cov-2", "project_id": PROJECT_ID, "payout_id": "payout-1", "payee_id": "payee-2"},
        ]
    )
    db.tables["royalty_lines"].rows.extend(
        [
            {"id": "line-1", "project_id": PROJECT_ID, "user_id": USER_ID, "amount_owed": 10},
            {"id": "line-2", "project_id": PROJECT_ID, "user_id": USER_ID, "amount_owed": 20},
        ]
    )
    return db


class TestDeleteProjectRoyaltyEntriesPurgeAudit:
    """A manual purge is destructive and outside the ledger's normal sync gates,
    so it must leave the same audit trail as the automated sync paths."""

    def test_records_one_manual_purge_row_per_deleted_row_with_original_data(self):
        db = _seed_purge_fixture()
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)

        history_rows = db.tables["royalty_ledger_history"].rows
        assert len(history_rows) == 4
        assert all(r["action"] == "manual_purge" for r in history_rows)
        assert all(r["cause"] == "manual_purge" for r in history_rows)
        assert all(r["user_id"] == USER_ID for r in history_rows)

        old_rows = [r["old_row"] for r in history_rows]
        assert {"id": "cov-1", "project_id": PROJECT_ID, "payout_id": "payout-1", "payee_id": "payee-1"} in old_rows
        assert {"id": "line-1", "project_id": PROJECT_ID, "user_id": USER_ID, "amount_owed": 10} in old_rows

    def test_deletes_coverage_and_lines_but_keeps_payouts(self):
        db = _seed_purge_fixture()
        service.delete_project_royalty_entries(db, USER_ID, PROJECT_ID)

        assert db.tables["royalty_payout_coverage"].rows == []
        assert db.tables["royalty_lines"].rows == []
        assert len(db.tables["royalty_payouts"].rows) == 1


# ---------------------------------------------------------------------------
# Tests: main.py DELETE /contracts/{contract_id} ledger-guardrail wiring
# ---------------------------------------------------------------------------

CONTRACT_FILE_PATH = f"{ARTIST_ID}/{PROJECT_ID}/contract/deal.pdf"


def _make_contract_delete_router(call_order):
    """Table router for DELETE /contracts/{id}, mirroring verify_user_owns_contract
    + delete_contract's call sequence in main.py:

      1. project_files  — ownership check (select project_id)
      2. artists         — get_user_artist_ids
      3. projects        — verify project belongs to artist (+ later, event-emit lookup)
      4. project_files   — fetch full contract record
      5. project_files   — delete (records "project_files_delete" into call_order)
    """
    state = {"project_files_call": 0}

    def _record_delete():
        call_order.append("project_files_delete")
        return MagicMock(data=[])

    def _router(name):
        builder = MockQueryBuilder()
        if name == "artists":
            builder.execute.return_value = MagicMock(data=[{"id": ARTIST_ID}], count=1)
        elif name == "projects":
            builder.execute.return_value = MagicMock(data=[{"id": PROJECT_ID}], count=1)
        elif name == "project_files":
            call_num = state["project_files_call"]
            state["project_files_call"] += 1
            if call_num == 0:
                builder.execute.return_value = MagicMock(data=[{"project_id": PROJECT_ID}], count=1)
            elif call_num == 1:
                builder.execute.return_value = MagicMock(
                    data=[
                        {
                            "id": CONTRACT_ID,
                            "project_id": PROJECT_ID,
                            "file_name": "deal.pdf",
                            "file_path": CONTRACT_FILE_PATH,
                        }
                    ],
                    count=1,
                )
            else:
                builder.delete.return_value.eq.return_value.execute.side_effect = _record_delete
        return builder

    return _router


class TestContractDeleteLedgerWiring:
    """DELETE /contracts/{contract_id} must clear the contract from the ledger
    before touching storage/project_files, and abort (no delete) if that fails."""

    def test_ledger_cleanup_runs_before_project_files_delete(self, client, mock_supabase):
        call_order = []
        mock_supabase.table.side_effect = _make_contract_delete_router(call_order)

        with patch(
            "oneclick.royalties.ledger_sync.remove_contract_from_ledger",
            side_effect=lambda *a, **k: call_order.append("ledger_cleanup"),
        ) as mock_ledger:
            response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 200
        mock_ledger.assert_called_once()
        assert call_order == ["ledger_cleanup", "project_files_delete"]

    def test_ledger_cleanup_failure_returns_500_and_skips_project_files_delete(self, client, mock_supabase):
        call_order = []
        mock_supabase.table.side_effect = _make_contract_delete_router(call_order)

        with patch(
            "oneclick.royalties.ledger_sync.remove_contract_from_ledger",
            side_effect=RuntimeError("boom"),
        ):
            response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 500
        assert "Ledger cleanup failed" in response.json()["detail"]
        assert "project_files_delete" not in call_order
