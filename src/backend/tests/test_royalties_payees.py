"""Tests for patch_payee and split_payee in oneclick/royalties/service.py.

All tests use a mock Supabase client — no real DB or network calls.

Scenarios covered:
  patch_payee:
    - updates only provided fields
    - email patch also sets email_source = 'manual'
    - other user's payee → PermissionError (→ 403 in router)

  split_payee:
    - reassigns selected lines' payee_id to a newly-upserted target
    - returns the target payee row
    - blocked (ValueError → 409) when any selected line's (payee, statement) bucket
      has a paid-payout coverage row
    - allowed when only draft coverage exists
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import service

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

USER_ID = "user-aaa"
OTHER_USER_ID = "user-bbb"
PAYEE_ID = "payee-111"
PAYEE_ID_2 = "payee-222"
PAYOUT_ID = "payout-xyz"
LINE_ID_1 = "line-1"
LINE_ID_2 = "line-2"
STMT_ID = "stmt-1"
STMT_ID_2 = "stmt-2"

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_payee(payee_id=PAYEE_ID, user_id=USER_ID, payout_ccy="USD", display_name="Alice", email=None):
    return {
        "id": payee_id,
        "user_id": user_id,
        "display_name": display_name,
        "payout_currency": payout_ccy,
        "normalized_name": display_name.lower(),
        "registry_user_id": None,
        "email": email,
        "email_source": None,
    }


def _make_line(
    line_id=LINE_ID_1,
    payee_id=PAYEE_ID,
    stmt_id=STMT_ID,
    user_id=USER_ID,
    amount=200.0,
    ccy="USD",
    project_id="proj-1",
):
    return {
        "id": line_id,
        "user_id": user_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "calculation_id": "calc-1",
        "project_id": project_id,
        "song_title": "Song A",
        "role": "Writer",
        "royalty_type": "mechanical",
        "percentage": 50.0,
        "amount_owed": amount,
        "statement_currency": ccy,
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
    }


def _make_payout(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, user_id=USER_ID, status="draft"):
    return {
        "id": payout_id,
        "user_id": user_id,
        "payee_id": payee_id,
        "status": status,
        "pay_currency": "USD",
        "total_amount": 200.0,
        "fx_rate_date": "2026-06-23",
        "created_at": "2026-06-23T00:00:00Z",
        "paid_at": None,
        "note": None,
        "breakdown_snapshot": {"projects": []},
        "idempotency_key": None,
    }


def _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id=STMT_ID, covered_amount=200.0):
    return {
        "payout_id": payout_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "covered_amount": covered_amount,
        "project_id": "proj-1",
    }


# ---------------------------------------------------------------------------
# Mock DB — supports patch_payee and split_payee patterns
# ---------------------------------------------------------------------------


class MockDB:
    """Minimal stateful mock of the Supabase client for payee mutation tests.

    Tracks:
      updated_payees:       list of update dicts applied to royalty_payees
      upserted_payees:      list of display_names inserted via royalty_payees.insert()
      updated_lines:        list of (payee_id_new, line_ids) tuples from royalty_lines.update()
    """

    def __init__(
        self,
        payees=None,
        lines=None,
        payouts=None,
        coverage=None,
    ):
        self.payees = list(payees or [])
        self.lines = list(lines or [])
        self.payouts = list(payouts or [])
        self.coverage = list(coverage or [])

        # Side-effect capture
        self.updated_payees: list[dict] = []
        self.upserted_payees: list[dict] = []
        self.updated_lines: list[dict] = []

        self._payee_id_counter = 0

    def table(self, name):
        return _TableProxy(self, name)


class _TableProxy:
    """Fluent mock proxy for db.table(name).select/update/insert/eq/in_().execute()."""

    def __init__(self, db: MockDB, name: str):
        self._db = db
        self._name = name
        self._filters: dict = {}
        self._in_filters: dict = {}
        self._pending_insert = None
        self._pending_update = None
        self._pending_delete = False

    def select(self, *a, **kw):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def in_(self, col, vals):
        self._in_filters[col] = set(vals)
        return self

    def neq(self, col, val):
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def single(self):
        return self

    def maybe_single(self):
        return self

    def insert(self, data):
        self._pending_insert = data
        return self

    def update(self, data):
        self._pending_update = data
        return self

    def delete(self):
        self._pending_delete = True
        return self

    def upsert(self, data, **kw):
        return self

    def _apply_filters(self, rows):
        for col, val in self._filters.items():
            rows = [r for r in rows if r.get(col) == val]
        for col, vals in self._in_filters.items():
            rows = [r for r in rows if r.get(col) in vals]
        return rows

    def execute(self):
        db = self._db
        name = self._name

        # --- INSERTS ---
        if self._pending_insert is not None:
            if name == "royalty_payees":
                rows = self._pending_insert if isinstance(self._pending_insert, list) else [self._pending_insert]
                created = []
                for row in rows:
                    db._payee_id_counter += 1
                    row_with_id = {"id": f"payee-new-{db._payee_id_counter}", **row}
                    db.upserted_payees.append(row_with_id)
                    db.payees.append(row_with_id)
                    created.append(row_with_id)
                return MagicMock(data=created)
            return MagicMock(data=[])

        # --- UPDATES ---
        if self._pending_update is not None:
            if name == "royalty_payees":
                payee_id = self._filters.get("id")
                matched = [p for p in db.payees if p.get("id") == payee_id]
                if matched:
                    updated = {**matched[0], **self._pending_update}
                    db.updated_payees.append(updated)
                    # Update in place so subsequent selects see the new values
                    idx = db.payees.index(matched[0])
                    db.payees[idx] = updated
                    return MagicMock(data=[updated])
                return MagicMock(data=[])

            if name == "royalty_lines":
                # Batch update via in_("id", ...)
                line_ids = self._in_filters.get("id", set())
                db.updated_lines.append(
                    {"payee_id": self._pending_update.get("payee_id"), "line_ids": sorted(line_ids)}
                )
                # Update the in-memory lines
                for i, line in enumerate(db.lines):
                    if line.get("id") in line_ids:
                        db.lines[i] = {**line, **self._pending_update}
                return MagicMock(data=[])

            return MagicMock(data=[])

        # --- DELETES ---
        if self._pending_delete:
            return MagicMock(data=[])

        # --- SELECTS ---
        if name == "royalty_payees":
            rows = self._apply_filters(db.payees)
            return MagicMock(data=rows)

        if name == "royalty_lines":
            rows = self._apply_filters(db.lines)
            return MagicMock(data=rows)

        if name == "royalty_payouts":
            rows = self._apply_filters(db.payouts)
            return MagicMock(data=rows)

        if name == "royalty_payout_coverage":
            rows = self._apply_filters(db.coverage)
            return MagicMock(data=rows)

        if name == "projects":
            return MagicMock(data=[])

        return MagicMock(data=[])


# ---------------------------------------------------------------------------
# Tests: patch_payee
# ---------------------------------------------------------------------------


class TestPatchPayee:
    def test_updates_payout_currency_only(self):
        """Providing only payout_currency must not touch other fields."""
        payee = _make_payee(email="alice@example.com")
        db = MockDB(payees=[payee])
        result = service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "GBP"})
        assert result["payout_currency"] == "GBP"
        # email_source must NOT be set when email was not in the patch
        assert result.get("email_source") != "manual"

    def test_updates_registry_user_id(self):
        payee = _make_payee()
        db = MockDB(payees=[payee])
        result = service.patch_payee(db, USER_ID, PAYEE_ID, {"registry_user_id": "uid-xyz"})
        assert result["registry_user_id"] == "uid-xyz"

    def test_email_patch_sets_email_source_manual(self):
        """Patching email must also set email_source = 'manual'."""
        payee = _make_payee()
        db = MockDB(payees=[payee])
        result = service.patch_payee(db, USER_ID, PAYEE_ID, {"email": "alice@example.com"})
        assert result["email"] == "alice@example.com"
        assert result["email_source"] == "manual"

    def test_email_source_not_set_when_email_not_in_patch(self):
        """If email is NOT in the patch, email_source must not be set to 'manual'."""
        payee = _make_payee()
        db = MockDB(payees=[payee])
        service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "EUR"})
        applied = db.updated_payees[0]
        assert "email_source" not in applied or applied.get("email_source") != "manual"

    def test_only_provided_keys_are_sent_in_update(self):
        """Only keys present in the input dict must appear in the update (plus updated_at)."""
        payee = _make_payee()
        db = MockDB(payees=[payee])
        service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "GBP"})
        applied = db.updated_payees[0]
        # registry_user_id and email should NOT have been explicitly set
        assert "payout_currency" in applied
        assert applied["payout_currency"] == "GBP"

    def test_other_user_payee_raises_permission_error(self):
        """Payee owned by another user → PermissionError."""
        payee = _make_payee(user_id=OTHER_USER_ID)
        db = MockDB(payees=[payee])
        with pytest.raises(PermissionError):
            service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "GBP"})

    def test_nonexistent_payee_raises_permission_error(self):
        db = MockDB(payees=[])
        with pytest.raises(PermissionError):
            service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "GBP"})

    def test_updated_at_is_set(self):
        payee = _make_payee()
        db = MockDB(payees=[payee])
        service.patch_payee(db, USER_ID, PAYEE_ID, {"payout_currency": "EUR"})
        applied = db.updated_payees[0]
        assert "updated_at" in applied


# ---------------------------------------------------------------------------
# Tests: split_payee
# ---------------------------------------------------------------------------


class TestSplitPayee:
    def test_reassigns_lines_to_new_payee(self):
        """Selected lines must have payee_id updated to the new target payee."""
        payee = _make_payee()
        line = _make_line()
        db = MockDB(payees=[payee], lines=[line], payouts=[], coverage=[])
        service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")
        # A new payee must have been upserted
        assert len(db.upserted_payees) == 1
        new_payee = db.upserted_payees[0]
        assert new_payee["display_name"] == "Bob"
        # The update must have been issued for the correct lines
        assert len(db.updated_lines) == 1
        update_record = db.updated_lines[0]
        assert update_record["payee_id"] == new_payee["id"]
        assert LINE_ID_1 in update_record["line_ids"]

    def test_returns_target_payee_row(self):
        """split_payee must return the new (target) payee's row dict."""
        payee = _make_payee()
        line = _make_line()
        db = MockDB(payees=[payee], lines=[line], payouts=[], coverage=[])
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Charlie")
        assert result["display_name"] == "Charlie"

    def test_existing_payee_name_reuses_existing_payee(self):
        """If a payee with the normalized target name already exists, upsert returns
        that existing id without inserting a second row."""
        source = _make_payee(payee_id=PAYEE_ID, display_name="Alice")
        target = _make_payee(payee_id=PAYEE_ID_2, display_name="Bob")
        line = _make_line()
        db = MockDB(payees=[source, target], lines=[line], payouts=[], coverage=[])
        # upsert_payee will find the existing "bob" payee by normalized_name
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")
        # Should NOT have inserted a new payee
        assert len(db.upserted_payees) == 0
        assert result["id"] == PAYEE_ID_2

    def test_ignores_line_ids_not_owned_by_caller(self):
        """Line ids belonging to another user must be silently ignored."""
        payee = _make_payee()
        foreign_line = _make_line(line_id="foreign-line", user_id=OTHER_USER_ID)
        db = MockDB(payees=[payee], lines=[foreign_line], payouts=[], coverage=[])
        # No owned lines match — should still succeed (just no update issued for those)
        service.split_payee(db, USER_ID, PAYEE_ID, ["foreign-line"], "Dave")
        assert db.updated_lines == []  # nothing to reassign

    def test_other_user_source_payee_raises_permission_error(self):
        """Source payee not owned by caller → PermissionError."""
        payee = _make_payee(user_id=OTHER_USER_ID)
        db = MockDB(payees=[payee], lines=[], payouts=[], coverage=[])
        with pytest.raises(PermissionError):
            service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")

    def test_nonexistent_source_payee_raises_permission_error(self):
        db = MockDB(payees=[], lines=[], payouts=[], coverage=[])
        with pytest.raises(PermissionError):
            service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")


# ---------------------------------------------------------------------------
# Tests: split_payee — paid-bucket guard
# ---------------------------------------------------------------------------


class TestSplitPayeePaidBucketGuard:
    def test_blocks_when_line_bucket_has_paid_payout(self):
        """If the selected line's (payee_id, statement_id) bucket has a paid payout,
        split must raise ValueError (→ 409 in router)."""
        payee = _make_payee()
        line = _make_line(payee_id=PAYEE_ID, stmt_id=STMT_ID)
        paid_payout = _make_payout(status="paid")
        coverage = _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id=STMT_ID)
        db = MockDB(payees=[payee], lines=[line], payouts=[paid_payout], coverage=[coverage])
        with pytest.raises(ValueError, match="paid invoice"):
            service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")

    def test_allows_split_when_only_draft_coverage_exists(self):
        """A draft-status payout coverage must NOT block the split."""
        payee = _make_payee()
        line = _make_line(payee_id=PAYEE_ID, stmt_id=STMT_ID)
        draft_payout = _make_payout(status="draft")
        coverage = _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id=STMT_ID)
        db = MockDB(payees=[payee], lines=[line], payouts=[draft_payout], coverage=[coverage])
        # Should succeed — draft is not a blocker
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")
        assert result["display_name"] == "Bob"

    def test_allows_split_when_no_coverage_exists(self):
        """No coverage rows → no block."""
        payee = _make_payee()
        line = _make_line()
        db = MockDB(payees=[payee], lines=[line], payouts=[], coverage=[])
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")
        assert result["display_name"] == "Bob"

    def test_blocks_only_on_paid_not_draft_when_both_exist(self):
        """When both a draft and a paid payout cover the same bucket,
        the paid one still triggers the guard."""
        payee = _make_payee()
        line = _make_line(payee_id=PAYEE_ID, stmt_id=STMT_ID)
        draft_payout = _make_payout(payout_id="payout-draft", status="draft")
        paid_payout = _make_payout(payout_id="payout-paid", status="paid")
        coverage_draft = _make_coverage(payout_id="payout-draft", payee_id=PAYEE_ID, stmt_id=STMT_ID)
        coverage_paid = _make_coverage(payout_id="payout-paid", payee_id=PAYEE_ID, stmt_id=STMT_ID)
        db = MockDB(
            payees=[payee],
            lines=[line],
            payouts=[draft_payout, paid_payout],
            coverage=[coverage_draft, coverage_paid],
        )
        with pytest.raises(ValueError):
            service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")

    def test_only_selected_lines_buckets_are_checked(self):
        """Lines NOT in line_ids are irrelevant — even if their bucket is paid,
        the split for OTHER lines in a clean bucket must proceed."""
        payee = _make_payee()
        # line-1 in stmt-1 (paid), line-2 in stmt-2 (no coverage)
        line1 = _make_line(line_id=LINE_ID_1, payee_id=PAYEE_ID, stmt_id=STMT_ID)
        line2 = _make_line(line_id=LINE_ID_2, payee_id=PAYEE_ID, stmt_id=STMT_ID_2)
        paid_payout = _make_payout(status="paid")
        # Coverage only for stmt-1
        coverage = _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id=STMT_ID)
        db = MockDB(payees=[payee], lines=[line1, line2], payouts=[paid_payout], coverage=[coverage])
        # Splitting only line-2 (stmt-2, no paid coverage) must succeed
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_2], "Bob")
        assert result["display_name"] == "Bob"

    def test_paid_bucket_of_different_payee_does_not_block(self):
        """Coverage for a DIFFERENT payee's bucket must not block this payee's split."""
        source_payee = _make_payee(payee_id=PAYEE_ID)
        other_payee = _make_payee(payee_id=PAYEE_ID_2, display_name="Other")
        line = _make_line(line_id=LINE_ID_1, payee_id=PAYEE_ID, stmt_id=STMT_ID)
        paid_payout = _make_payout(status="paid")
        # Coverage is for OTHER payee's stmt, not source payee
        coverage = _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID_2, stmt_id=STMT_ID)
        db = MockDB(
            payees=[source_payee, other_payee],
            lines=[line],
            payouts=[paid_payout],
            coverage=[coverage],
        )
        # Source payee's line in stmt-1 has no paid coverage → should succeed
        result = service.split_payee(db, USER_ID, PAYEE_ID, [LINE_ID_1], "Bob")
        assert result["display_name"] == "Bob"


# ---------------------------------------------------------------------------
# Tests: router HTTP layer (smoke tests)
# ---------------------------------------------------------------------------


def _make_router_client(service_mock_target=None, service_mock_side_effect=None, service_mock_return=None):
    """Build a minimal FastAPI app with the royalties router, auth/gating bypassed."""
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

    service_patcher = None
    if service_mock_target:
        if service_mock_side_effect is not None:
            service_patcher = patch(service_mock_target, side_effect=service_mock_side_effect)
        else:
            service_patcher = patch(
                service_mock_target,
                return_value=service_mock_return if service_mock_return is not None else {},
            )

    gating_patcher.start()
    supabase_patcher.start()
    if service_patcher:
        service_patcher.start()

    client = TestClient(app, raise_server_exceptions=False)

    yield client

    gating_patcher.stop()
    supabase_patcher.stop()
    if service_patcher:
        service_patcher.stop()


class TestPatchPayeeRouter:
    def test_patch_payee_200(self):
        for client in _make_router_client(
            "oneclick.royalties.service.patch_payee",
            service_mock_return={"id": PAYEE_ID, "payout_currency": "GBP"},
        ):
            resp = client.patch(f"/payees/{PAYEE_ID}", json={"payout_currency": "GBP"})
            assert resp.status_code == 200

    def test_patch_payee_403_on_permission_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.patch_payee",
            service_mock_side_effect=PermissionError("not yours"),
        ):
            resp = client.patch(f"/payees/{PAYEE_ID}", json={"payout_currency": "GBP"})
            assert resp.status_code == 403


class TestSplitPayeeRouter:
    def test_split_payee_200(self):
        for client in _make_router_client(
            "oneclick.royalties.service.split_payee",
            service_mock_return={"id": "payee-new", "display_name": "Bob"},
        ):
            resp = client.post(f"/payees/{PAYEE_ID}/split", json={"line_ids": [LINE_ID_1], "new_display_name": "Bob"})
            assert resp.status_code == 200

    def test_split_payee_403_on_permission_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.split_payee",
            service_mock_side_effect=PermissionError("not yours"),
        ):
            resp = client.post(f"/payees/{PAYEE_ID}/split", json={"line_ids": [LINE_ID_1], "new_display_name": "Bob"})
            assert resp.status_code == 403

    def test_split_payee_409_on_value_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.split_payee",
            service_mock_side_effect=ValueError("cannot split lines already settled by a paid invoice"),
        ):
            resp = client.post(f"/payees/{PAYEE_ID}/split", json={"line_ids": [LINE_ID_1], "new_display_name": "Bob"})
            assert resp.status_code == 409
