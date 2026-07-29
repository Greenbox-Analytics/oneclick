"""Cap-raise requests: a member asks for a higher monthly limit, an admin
approves it by writing org_members.monthly_cap.

Nothing moves, which is what makes this simple: approving is idempotent by
nature, there is no pool-insufficient failure to map, and no duplicate-transfer
replay to read back from the ledger.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import service
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
ORG_ID = "20000000-0000-0000-0000-000000000001"
MEMBER_ID = "40000000-0000-0000-0000-000000000001"
REQUEST_ID = "70000000-0000-0000-0000-000000000001"
POOL_WALLET = "50000000-0000-0000-0000-000000000002"


def _db_seq(seqs):
    """Same helper as tests/test_orgs_service.py's `_db_seq`: table_name ->
    list of execute() return values, consumed in call order. rpc() is a
    fresh MagicMock configurable per test."""
    counters = {k: 0 for k in seqs}

    def _side(name):
        b = MockQueryBuilder()
        if name in seqs:
            i = min(counters[name], len(seqs[name]) - 1)
            counters[name] += 1
            b.execute.return_value = seqs[name][i]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


def _pending_request(**overrides):
    base = {
        "id": REQUEST_ID,
        "org_id": ORG_ID,
        "org_member_id": MEMBER_ID,
        "requested_cap": 100,
        "note": None,
        "status": "pending",
        "resolved_by": None,
        "resolved_cap": None,
        "resolved_at": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# orgs.service.submit_credit_request
# ---------------------------------------------------------------------------


async def test_submit_credit_request_requires_active_member(monkeypatch):
    """Non-member (or a suspended/removed seat — is_org_member only counts
    status='active' rows) is gated by authz.require_member -> 404."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.submit_credit_request(MagicMock(), U2, ORG_ID, 100, "please")
    assert exc_info.value.status_code == 404


async def test_submit_credit_request_inserts_with_caller_member_id(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data={"id": MEMBER_ID}, count=1)
        elif name == "credit_requests":

            def _insert(payload):
                captured["payload"] = payload
                return b

            b.insert = _insert
            b.execute.return_value = MagicMock(data=[_pending_request()], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.submit_credit_request(db, U1, ORG_ID, 100, "please")

    assert captured["payload"] == {
        "org_id": ORG_ID,
        "org_member_id": MEMBER_ID,
        "requested_cap": 100,
        "note": "please",
    }
    assert result["request"]["id"] == REQUEST_ID
    assert result["org_member_id"] == MEMBER_ID


async def test_submit_credit_request_omits_optional_fields_when_not_provided(monkeypatch):
    """requested_cap=None / note=None must NOT be written into the
    payload at all (the column default / NULL applies)."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data={"id": MEMBER_ID}, count=1)
        elif name == "credit_requests":

            def _insert(payload):
                captured["payload"] = payload
                return b

            b.insert = _insert
            b.execute.return_value = MagicMock(data=[_pending_request(requested_cap=None, note=None)], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.submit_credit_request(db, U1, ORG_ID, None, None)

    assert captured["payload"] == {"org_id": ORG_ID, "org_member_id": MEMBER_ID}


async def test_submit_credit_request_duplicate_pending_maps_to_error(monkeypatch):
    """KEY TEST: the DB partial unique index (org_member_id, WHERE
    status='pending') violation surfaces as a 23505 on the INSERT — caught
    and mapped to DuplicatePendingRequestError (409 at the router)."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data={"id": MEMBER_ID}, count=1)
        elif name == "credit_requests":
            b.insert.side_effect = Exception("duplicate key value violates unique constraint")
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with pytest.raises(service.DuplicatePendingRequestError):
        await service.submit_credit_request(db, U1, ORG_ID, 100, None)


async def test_submit_credit_request_other_insert_errors_propagate_unmapped(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data={"id": MEMBER_ID}, count=1)
        elif name == "credit_requests":
            b.insert.side_effect = RuntimeError("boom")
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with pytest.raises(RuntimeError):
        await service.submit_credit_request(db, U1, ORG_ID, 100, None)


# ---------------------------------------------------------------------------
# orgs.service.list_credit_requests
# ---------------------------------------------------------------------------


async def test_list_credit_requests_requires_membership(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.list_credit_requests(MagicMock(), U2, ORG_ID)
    assert exc_info.value.status_code == 404


async def test_list_credit_requests_admin_sees_all(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    rows = [_pending_request(id="r1"), _pending_request(id="r2", org_member_id="other-member")]
    db = _db_seq({"credit_requests": [MagicMock(data=rows, count=2)]})

    result = await service.list_credit_requests(db, U1, ORG_ID)

    assert [r["id"] for r in result] == ["r1", "r2"]
    # Non-admin path queries org_members for the caller's own id — admin
    # path must NOT do that extra lookup.
    org_members_calls = [c for c in db.table.call_args_list if c.args[0] == "org_members"]
    assert org_members_calls == []


async def test_list_credit_requests_member_sees_only_own(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data={"id": MEMBER_ID}, count=1)
        elif name == "credit_requests":
            original_eq = b.eq

            def _eq(*a, **kw):
                if a and a[0] == "org_member_id":
                    captured["org_member_id_filter"] = a[1]
                return original_eq(*a, **kw)

            b.eq = _eq
            b.execute.return_value = MagicMock(data=[_pending_request()], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.list_credit_requests(db, U2, ORG_ID)

    assert captured["org_member_id_filter"] == MEMBER_ID
    assert len(result) == 1


# ---------------------------------------------------------------------------
# orgs.service.approve_credit_request
# ---------------------------------------------------------------------------


async def test_approve_credit_request_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.approve_credit_request(MagicMock(), U2, ORG_ID, REQUEST_ID, 100)
    assert exc_info.value.status_code == 403


async def test_approve_credit_request_unknown_request_404(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"credit_requests": [MagicMock(data=None, count=0)]})
    with pytest.raises(service.CreditRequestNotFoundError):
        await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 100)


async def test_approve_credit_request_already_resolved_409(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"credit_requests": [MagicMock(data=_pending_request(status="approved"), count=1)]})
    with pytest.raises(service.CreditRequestAlreadyResolvedError):
        await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 100)


async def test_approve_credit_request_writes_the_member_cap(monkeypatch):
    """Approving a request raises the member's ceiling. No money moves, so there
    is no transfer to order against the status write and no pool-insufficient
    failure mode to map."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    updates = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_requests":
            # One builder serves the maybe_single read (a dict) and the UPDATE
            # echo (a list) — switch shape once the update lands.
            b.execute.return_value = MagicMock(data=_pending_request(), count=1)
            original_update = b.update

            def _update(payload, *a, **kw):
                updates.setdefault("credit_requests", []).append(payload)
                b.execute.return_value = MagicMock(data=[{**_pending_request(), **payload}], count=1)
                return original_update(payload, *a, **kw)

            b.update = _update
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": MEMBER_ID}], count=1)
            original_update = b.update

            def _update(payload, *a, **kw):
                updates.setdefault("org_members", []).append(payload)
                return original_update(payload, *a, **kw)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 2500)

    assert updates["org_members"] == [{"monthly_cap": 2500}]
    assert updates["credit_requests"][0]["status"] == "approved"
    assert updates["credit_requests"][0]["resolved_cap"] == 2500
    db.rpc.assert_not_called()


async def test_approve_credit_request_records_admin_chosen_cap(monkeypatch):
    """The admin's chosen cap is what's recorded — not what the member asked
    for."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq(
        {
            "credit_requests": [
                MagicMock(data=_pending_request(requested_cap=100), count=1),
                MagicMock(
                    data=[{**_pending_request(), "status": "approved", "resolved_cap": 2500, "resolved_by": U1}],
                    count=1,
                ),
            ]
        }
    )

    # Admin chooses 2500, even though the member only asked for 100.
    result = await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 2500)

    assert result["status"] == "approved"
    assert result["resolved_cap"] == 2500


async def test_approve_credit_request_is_idempotent_on_retry(monkeypatch):
    """Setting a ceiling twice lands the same ceiling — which is why this path
    needs none of the duplicate-replay read-back the allocation version had."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    caps = []

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_requests":
            # maybe_single read returns the dict; the UPDATE echo needs a list.
            b.execute.return_value = MagicMock(data=_pending_request(), count=1)
            original_update = b.update

            def _update_req(payload, *a, **kw):
                b.execute.return_value = MagicMock(data=[{**_pending_request(), **payload}], count=1)
                return original_update(payload, *a, **kw)

            b.update = _update_req
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": MEMBER_ID}], count=1)
            original_update = b.update

            def _update(payload, *a, **kw):
                caps.append(payload["monthly_cap"])
                return original_update(payload, *a, **kw)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 2500)
    await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, 2500)

    assert caps == [2500, 2500]


async def test_approve_credit_request_rejects_negative_cap(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"credit_requests": [MagicMock(data=_pending_request(), count=1)]})
    with pytest.raises(ValueError, match="cap must be >= 0"):
        await service.approve_credit_request(db, U1, ORG_ID, REQUEST_ID, -1)


async def test_deny_credit_request_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.deny_credit_request(MagicMock(), U2, ORG_ID, REQUEST_ID, None)
    assert exc_info.value.status_code == 403


async def test_deny_credit_request_unknown_request_404(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"credit_requests": [MagicMock(data=None, count=0)]})
    with pytest.raises(service.CreditRequestNotFoundError):
        await service.deny_credit_request(db, U1, ORG_ID, REQUEST_ID, None)


async def test_deny_credit_request_already_resolved_409(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"credit_requests": [MagicMock(data=_pending_request(status="denied"), count=1)]})
    with pytest.raises(service.CreditRequestAlreadyResolvedError):
        await service.deny_credit_request(db, U1, ORG_ID, REQUEST_ID, None)


async def test_deny_credit_request_status_only_no_rpc(monkeypatch):
    """KEY TEST: deny is a status-only transition — no money moves. Two
    separate db.table("credit_requests") calls happen here (the initial
    SELECT, then the UPDATE) — each needs its own execute() return shape
    (a bare dict for maybe_single vs a one-row list for update), so this
    uses a call counter rather than a single shared execute() config."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}
    calls = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_requests":
            calls["n"] += 1
            if calls["n"] == 1:
                b.execute.return_value = MagicMock(data=_pending_request(), count=1)
            else:
                original_update = b.update

                def _update(payload, *a, **kw):
                    captured["payload"] = payload
                    return original_update(payload, *a, **kw)

                b.update = _update
                b.execute.return_value = MagicMock(
                    data=[{**_pending_request(), "status": "denied", "resolved_by": U1}], count=1
                )
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.deny_credit_request(db, U1, ORG_ID, REQUEST_ID, "not this month")

    assert captured["payload"]["status"] == "denied"
    assert captured["payload"]["resolved_by"] == U1
    assert captured["payload"]["note"] == "not this month"
    assert "resolved_at" in captured["payload"]
    db.rpc.assert_not_called()
    assert result["status"] == "denied"


async def test_deny_credit_request_without_note_leaves_note_field_untouched(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}
    calls = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_requests":
            calls["n"] += 1
            if calls["n"] == 1:
                b.execute.return_value = MagicMock(data=_pending_request(), count=1)
            else:
                original_update = b.update

                def _update(payload, *a, **kw):
                    captured["payload"] = payload
                    return original_update(payload, *a, **kw)

                b.update = _update
                b.execute.return_value = MagicMock(
                    data=[{**_pending_request(), "status": "denied", "resolved_by": U1}], count=1
                )
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.deny_credit_request(db, U1, ORG_ID, REQUEST_ID, None)

    assert "note" not in captured["payload"]


# ---------------------------------------------------------------------------
# Router: POST/GET /orgs/{org_id}/credit-requests, approve, deny
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _licensing_on_by_default(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


class TestFlagGate:
    """Every /orgs/*/credit-requests* route is defined on the SAME router as
    every other /orgs/* route, which gates the whole surface on
    LICENSING_ENABLED via a router-level dependency (orgs.router.
    require_licensing) — these routes get that 404-when-off behavior for
    free. Verified directly here (not just inherited from
    test_orgs_router.py's TestFlagGate) per the plan's instruction to add
    these route names to the flag-gate pattern."""

    ROUTES = [
        ("POST", f"/orgs/{ORG_ID}/credit-requests", {"requested_cap": 100}),
        ("GET", f"/orgs/{ORG_ID}/credit-requests", None),
        ("POST", f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", {"credits": 100}),
        ("POST", f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/deny", {"note": "hi"}),
    ]

    @pytest.mark.parametrize("method,path,body", ROUTES)
    def test_route_404_when_flag_unset(self, method, path, body, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        kwargs = {"json": body} if body is not None else {}
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 404

    @pytest.mark.parametrize("method,path,body", ROUTES)
    def test_route_404_when_flag_explicitly_false(self, method, path, body, client, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "false")
        kwargs = {"json": body} if body is not None else {}
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 404


def test_submit_router_ok_fires_analytics_and_schedules_email(client):
    with (
        patch(
            "orgs.router.service.submit_credit_request",
            new=AsyncMock(
                return_value={"request": {"id": REQUEST_ID, "status": "pending"}, "org_member_id": MEMBER_ID}
            ),
        ),
        patch("orgs.router._send_credit_request_email_background") as mock_bg,
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests", json={"requested_cap": 100, "note": "please"})

    assert resp.status_code == 200
    assert resp.json()["id"] == REQUEST_ID
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "credit_request_submitted"
    mock_bg.assert_called_once()
    assert mock_bg.call_args.kwargs["org_id"] == ORG_ID
    assert mock_bg.call_args.kwargs["request_id"] == REQUEST_ID
    assert mock_bg.call_args.kwargs["requested_cap"] == 100


def test_submit_router_duplicate_pending_409(client):
    from orgs.service import DuplicatePendingRequestError

    with patch(
        "orgs.router.service.submit_credit_request",
        new=AsyncMock(side_effect=DuplicatePendingRequestError("You already have a pending request.")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests", json={"requested_cap": 100})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "You already have a pending request."


def test_submit_router_rejects_zero_requested_cap_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/credit-requests", json={"requested_cap": 0})
    assert resp.status_code == 422


def test_submit_router_rejects_over_limit_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/credit-requests", json={"requested_cap": 10_000_001})
    assert resp.status_code == 422


def test_submit_router_allows_omitted_requested_cap(client):
    with (
        patch(
            "orgs.router.service.submit_credit_request",
            new=AsyncMock(return_value={"request": {"id": REQUEST_ID}, "org_member_id": MEMBER_ID}),
        ),
        patch("orgs.router._send_credit_request_email_background"),
        patch("orgs.router.analytics_capture"),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests", json={})
    assert resp.status_code == 200


def test_list_router_ok(client):
    with patch("orgs.router.service.list_credit_requests", new=AsyncMock(return_value=[{"id": REQUEST_ID}])):
        resp = client.get(f"/orgs/{ORG_ID}/credit-requests")
    assert resp.status_code == 200
    assert resp.json() == {"requests": [{"id": REQUEST_ID}]}


def test_approve_router_ok_fires_analytics_with_resolved_cap(client):
    with (
        patch(
            "orgs.router.service.approve_credit_request",
            new=AsyncMock(return_value={"id": REQUEST_ID, "status": "approved", "resolved_cap": 250}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={"cap": 2500})

    assert resp.status_code == 200
    assert resp.json()["resolved_cap"] == 250
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "credit_request_resolved"
    props = mock_capture.call_args.args[2]
    assert props["status"] == "approved"
    assert props["cap"] == 250


def test_approve_router_not_found_404(client):
    from orgs.service import CreditRequestNotFoundError

    with patch(
        "orgs.router.service.approve_credit_request",
        new=AsyncMock(side_effect=CreditRequestNotFoundError("Credit request not found")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={"cap": 2500})
    assert resp.status_code == 404


def test_approve_router_already_resolved_409(client):
    from orgs.service import CreditRequestAlreadyResolvedError

    with patch(
        "orgs.router.service.approve_credit_request",
        new=AsyncMock(side_effect=CreditRequestAlreadyResolvedError("This request has already been resolved")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={"cap": 2500})
    assert resp.status_code == 409


def test_approve_router_rejects_negative_cap_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={"cap": -1})
    assert resp.status_code == 422


def test_approve_router_rejects_missing_credits_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={})
    assert resp.status_code == 422


def test_approve_router_rejects_over_limit_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/approve", json={"credits": 1_000_001})
    assert resp.status_code == 422


def test_deny_router_ok_fires_analytics_with_null_credits(client):
    with (
        patch(
            "orgs.router.service.deny_credit_request",
            new=AsyncMock(return_value={"id": REQUEST_ID, "status": "denied"}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/deny", json={"note": "not now"})

    assert resp.status_code == 200
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "credit_request_resolved"
    props = mock_capture.call_args.args[2]
    assert props["status"] == "denied"
    assert props["credits"] is None


def test_deny_router_not_found_404(client):
    from orgs.service import CreditRequestNotFoundError

    with patch(
        "orgs.router.service.deny_credit_request",
        new=AsyncMock(side_effect=CreditRequestNotFoundError("Credit request not found")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/deny", json={})
    assert resp.status_code == 404


def test_deny_router_already_resolved_409(client):
    from orgs.service import CreditRequestAlreadyResolvedError

    with patch(
        "orgs.router.service.deny_credit_request",
        new=AsyncMock(side_effect=CreditRequestAlreadyResolvedError("This request has already been resolved")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/deny", json={})
    assert resp.status_code == 409


def test_deny_router_allows_missing_note(client):
    with (
        patch(
            "orgs.router.service.deny_credit_request",
            new=AsyncMock(return_value={"id": REQUEST_ID, "status": "denied"}),
        ),
        patch("orgs.router.analytics_capture"),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/credit-requests/{REQUEST_ID}/deny", json={})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Background email task: admins-only recipients
# ---------------------------------------------------------------------------


def test_credit_request_email_background_sends_to_active_admins_only():
    """KEY TEST: recipients are ACTIVE admin members' emails only — resolved
    via the auth admin API (org_members carries user_id, not email)."""
    from orgs import router as orgs_router

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"name": "Acme"}, count=1)
        elif name == "profiles":
            b.execute.return_value = MagicMock(data={"full_name": "Requester Name"}, count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"user_id": "admin-1"}, {"user_id": "admin-2"}], count=2)
        return b

    fake_db = MagicMock()
    fake_db.table.side_effect = _side

    def _get_user_by_id(user_id):
        emails = {"admin-1": "admin1@example.com", "admin-2": "admin2@example.com"}
        return MagicMock(user=MagicMock(email=emails.get(user_id)))

    fake_db.auth.admin.get_user_by_id.side_effect = _get_user_by_id

    with (
        patch("supabase.create_client", return_value=fake_db),
        patch("orgs.emails.send_credit_request_email") as mock_send,
    ):
        orgs_router._send_credit_request_email_background(
            db_url="http://fake",
            db_key="fake-key",
            org_id=ORG_ID,
            request_id=REQUEST_ID,
            requester_user_id=U1,
            requested_cap=100,
            note="please",
        )

    mock_send.assert_called_once()
    kwargs = mock_send.call_args.kwargs
    assert sorted(kwargs["recipient_emails"]) == ["admin1@example.com", "admin2@example.com"]
    assert kwargs["org_name"] == "Acme"
    assert kwargs["requester_name"] == "Requester Name"
    assert kwargs["requested_cap"] == 100
    assert kwargs["note"] == "please"


def test_credit_request_email_background_no_admins_skips_send():
    from orgs import router as orgs_router

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"name": "Acme"}, count=1)
        elif name == "profiles":
            b.execute.return_value = MagicMock(data={"full_name": "Requester"}, count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    fake_db = MagicMock()
    fake_db.table.side_effect = _side

    with (
        patch("supabase.create_client", return_value=fake_db),
        patch("orgs.emails.send_credit_request_email") as mock_send,
    ):
        orgs_router._send_credit_request_email_background(
            db_url="http://fake",
            db_key="fake-key",
            org_id=ORG_ID,
            request_id=REQUEST_ID,
            requester_user_id=U1,
            requested_cap=100,
            note=None,
        )

    mock_send.assert_not_called()
