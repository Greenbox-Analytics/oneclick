"""Mock-based tests for orgs.projects (Licensing Phase C, Task 2 — owner
link/unlink + admin project list) and the orgs router endpoints that wrap
them. Mirrors tests/test_orgs_service.py + tests/test_orgs_router.py's
idioms (the `_db_seq` per-table call-order helper is copied here rather than
imported, same self-contained-fixture posture test_orgs_router.py's
TestGetOrgUsageService takes for the same reason: this module shouldn't
depend on another test module's private helpers).

LICENSING_ENABLED is OFF by default in tests; the autouse fixture below turns
it on so the router-level tests exercise the real routes, matching
test_orgs_router.py's convention."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from postgrest.exceptions import APIError

from orgs import projects as org_projects
from tests.conftest import MockQueryBuilder

ORG_ID = "20000000-0000-0000-0000-000000000001"
OTHER_ORG_ID = "20000000-0000-0000-0000-000000000002"
PROJECT_ID = "10000000-0000-0000-0000-000000000001"
USER_ID = "00000000-0000-0000-0000-000000000001"
OTHER_USER_ID = "00000000-0000-0000-0000-000000000002"
LINK_ID = "70000000-0000-0000-0000-000000000001"
SEAT_MEMBER_ID = "30000000-0000-0000-0000-000000000001"
TARGET_USER_ID = "00000000-0000-0000-0000-000000000003"
PM_ROW_ID = "40000000-0000-0000-0000-000000000001"


def _db_seq(seqs):
    """seqs: dict table_name -> list of execute() return values, consumed in
    call order. rpc() always returns a fresh MagicMock."""
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


def _org_row(**overrides):
    base = {"id": ORG_ID, "status": "active", "archived_at": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# link_project — no-leak 404 pairs (Task 2 AC 1)
# ---------------------------------------------------------------------------


class TestLinkProjectNoLeak404s:
    """Two independent no-existence-oracle pairs:
    (unknown project | not owner) -> "Project not found"
    (unknown org | no active seat | org not ACTIVE) -> "Organization not found"
    Every case in a pair must produce the IDENTICAL body as its sibling.
    """

    async def test_unknown_project_404(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value=None))
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Project not found"

    async def test_non_owner_role_same_body_as_unknown_project(self, monkeypatch):
        """seat-holder-non-owner: caller has SOME role but not 'owner' — must
        produce the identical body as an unknown project."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="editor"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: True)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Project not found"

    async def test_unknown_org_404(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: False)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Organization not found"

    async def test_owner_without_active_seat_same_body_as_unknown_org(self, monkeypatch):
        """owner-without-seat: caller owns the project but holds no active
        seat in org_id — must produce the identical body as an unknown org."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: False)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Organization not found"

    async def test_pending_org_404_even_for_active_seat_holder(self, monkeypatch):
        """A PENDING org confers nothing — 404s just like an unknown org,
        even though the caller genuinely holds an active seat there."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: True)

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=_org_row(status="pending"), count=1)
            return b

        db = MagicMock()
        db.table.side_effect = _side
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Organization not found"

    async def test_suspended_org_404(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: True)

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=_org_row(status="suspended"), count=1)
            return b

        db = MagicMock()
        db.table.side_effect = _side
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Organization not found"

    async def test_archived_org_404_even_if_status_active(self, monkeypatch):
        """archived_at set but status still 'active' (archive_org doesn't
        touch status) must still 404 — archived_at is the load-bearing check."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: True)

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(
                    data=_org_row(status="active", archived_at="2026-07-01T00:00:00+00:00"), count=1
                )
            return b

        db = MagicMock()
        db.table.side_effect = _side
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Organization not found"


# ---------------------------------------------------------------------------
# link_project — duplicate/other-org 409 (rule 8) + happy path
# ---------------------------------------------------------------------------


class TestLinkProjectDuplicateAndSuccess:
    def _authorized(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        monkeypatch.setattr(org_projects.authz, "is_org_member", lambda *a: True)

    async def test_duplicate_link_to_same_org_409(self, monkeypatch):
        self._authorized(monkeypatch)
        db = _db_seq(
            {
                "organizations": [MagicMock(data=_org_row(), count=1)],
                "org_project_links": [MagicMock(data={"id": LINK_ID, "org_id": ORG_ID}, count=1)],
            }
        )
        with pytest.raises(org_projects.ProjectAlreadyLinkedError) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert str(exc_info.value) == "This project is already linked to an organization — unlink it first."

    async def test_duplicate_link_to_different_org_409_same_copy(self, monkeypatch):
        """rule 8: ANY existing link (even to a DIFFERENT org than the one
        being requested) 409s with the exact same copy."""
        self._authorized(monkeypatch)
        db = _db_seq(
            {
                "organizations": [MagicMock(data=_org_row(), count=1)],
                "org_project_links": [MagicMock(data={"id": LINK_ID, "org_id": OTHER_ORG_ID}, count=1)],
            }
        )
        with pytest.raises(org_projects.ProjectAlreadyLinkedError) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert str(exc_info.value) == "This project is already linked to an organization — unlink it first."

    async def test_success_inserts_link_row(self, monkeypatch):
        self._authorized(monkeypatch)
        captured = {}
        link_calls = {"n": 0}

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=_org_row(), count=1)
            elif name == "org_project_links":
                link_calls["n"] += 1
                if link_calls["n"] == 1:
                    b.execute.return_value = MagicMock(data=None, count=0)  # existing-link check: none
                else:
                    original_insert = b.insert

                    def _capture_insert(payload, *a, **kw):
                        captured["insert_payload"] = payload
                        return original_insert(payload, *a, **kw)

                    b.insert = _capture_insert
                    b.execute.return_value = MagicMock(
                        data=[{"id": LINK_ID, "org_id": ORG_ID, "project_id": PROJECT_ID, "linked_by": USER_ID}],
                        count=1,
                    )
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert result["org_id"] == ORG_ID
        assert result["project_id"] == PROJECT_ID
        assert captured["insert_payload"] == {"org_id": ORG_ID, "project_id": PROJECT_ID, "linked_by": USER_ID}

    def _db_probe_clear_then_insert_raises(self, error):
        """Probe finds no link; the INSERT then raises `error` (the race where
        a concurrent duplicate slipped between probe and INSERT)."""
        link_calls = {"n": 0}

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=_org_row(), count=1)
            elif name == "org_project_links":
                link_calls["n"] += 1
                if link_calls["n"] == 1:
                    b.execute.return_value = MagicMock(data=None, count=0)  # probe: no link yet
                else:
                    b.execute.side_effect = error
            return b

        db = MagicMock()
        db.table.side_effect = _side
        return db

    async def test_concurrent_duplicate_unique_violation_maps_to_409(self, monkeypatch):
        """The probe+INSERT pair is not atomic: a concurrent double-submit can
        pass both probes and hit UNIQUE(project_id) at the DB. The unique
        violation (23505) must surface as the SAME rule-8 409 copy, not a 500."""
        self._authorized(monkeypatch)
        db = self._db_probe_clear_then_insert_raises(
            APIError({"message": "duplicate key value violates unique constraint", "code": "23505"})
        )
        with pytest.raises(org_projects.ProjectAlreadyLinkedError) as exc_info:
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert str(exc_info.value) == "This project is already linked to an organization — unlink it first."

    async def test_non_unique_apierror_on_insert_reraises(self, monkeypatch):
        """Only 23505 maps to the 409 — any other DB error propagates untouched
        rather than masquerading as 'already linked'."""
        self._authorized(monkeypatch)
        db = self._db_probe_clear_then_insert_raises(
            APIError({"message": "permission denied for table org_project_links", "code": "42501"})
        )
        with pytest.raises(APIError):
            await org_projects.link_project(db, USER_ID, ORG_ID, PROJECT_ID)


# ---------------------------------------------------------------------------
# unlink_project
# ---------------------------------------------------------------------------


class TestUnlinkProject:
    async def test_requires_owner_404(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="editor"))
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.unlink_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Project not found"

    async def test_no_link_at_all_404(self, monkeypatch):
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        db = _db_seq({"org_project_links": [MagicMock(data=None, count=0)]})
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.unlink_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "This project is not linked to this organization."

    async def test_link_belongs_to_different_org_404(self, monkeypatch):
        """The link exists, but for a DIFFERENT org than the one in the path
        — must 404, not silently unlink the wrong org's grant."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        db = _db_seq({"org_project_links": [MagicMock(data={"id": LINK_ID, "org_id": OTHER_ORG_ID}, count=1)]})
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.unlink_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "This project is not linked to this organization."

    async def test_deletes_only_provenance_matched_rows_and_returns_count(self, monkeypatch):
        """KEY TEST (rule 3): unlink must delete ONLY project_members rows
        where org_id = THIS org AND project_id = THIS project — organic rows
        (org_id NULL) and other-org rows must survive. The mock can't enforce
        a real filter, so this asserts BOTH halves of the guarantee: (1) the
        exact filter args passed to .eq() are org_id/THIS-org and
        project_id/THIS-project (the entire safety mechanism), and (2) the
        reported count matches only the rows that filter would match — an
        organic row and an other-org row are included in the fixture but
        deliberately excluded from the configured matched set."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))

        matched_rows = [
            {"id": "m1", "project_id": PROJECT_ID, "org_id": ORG_ID, "user_id": "u-a"},
            {"id": "m2", "project_id": PROJECT_ID, "org_id": ORG_ID, "user_id": "u-b"},
        ]
        builders = {}

        def _side(name):
            b = MockQueryBuilder()
            builders[name] = b
            if name == "org_project_links":
                b.execute.return_value = MagicMock(data={"id": LINK_ID, "org_id": ORG_ID}, count=1)
                b.delete.return_value = b  # link-row delete chains through self
            elif name == "project_members":
                b.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
                    data=matched_rows, count=len(matched_rows)
                )
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.unlink_project(db, USER_ID, ORG_ID, PROJECT_ID)

        assert result == {"revoked": 2}
        pm_builder = builders["project_members"]
        pm_builder.delete.return_value.eq.assert_called_once_with("org_id", ORG_ID)
        pm_builder.delete.return_value.eq.return_value.eq.assert_called_once_with("project_id", PROJECT_ID)

    async def test_revocation_happens_before_link_row_delete(self, monkeypatch):
        """The link row delete happens on org_project_links AFTER the
        project_members revocation completes (rule 3's ordering: if a crash
        lands between the two writes, the WORSE outcome — link row survives
        a completed revocation — is the one that can happen, never access
        surviving a deleted link)."""
        monkeypatch.setattr(org_projects, "get_user_role", AsyncMock(return_value="owner"))
        order = []

        def _fake_revoke(sb, org_id, *, user_id=None, project_id=None):
            order.append("revoke")
            return 5

        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", _fake_revoke)

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_project_links":
                b.execute.return_value = MagicMock(data={"id": LINK_ID, "org_id": ORG_ID}, count=1)

                def _tracked_delete(*a, **kw):
                    order.append("link_delete")
                    return b  # self-chain: .eq("id", ...).execute() resolves harmlessly

                b.delete = MagicMock(side_effect=_tracked_delete)
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.unlink_project(db, USER_ID, ORG_ID, PROJECT_ID)
        assert result == {"revoked": 5}
        assert order == ["revoke", "link_delete"]


# ---------------------------------------------------------------------------
# revoke_org_granted_memberships — standalone helper coverage (Task 4 will
# extend call sites, not the signature; this asserts the current filters).
# ---------------------------------------------------------------------------


class TestRevokeOrgGrantedMemberships:
    def _builder_with_chain(self, n_eq, data):
        b = MockQueryBuilder()
        node = b.delete.return_value
        for _ in range(n_eq):
            node = node.eq.return_value
        node.execute.return_value = MagicMock(data=data, count=len(data))
        return b

    def test_org_only_filter_when_no_narrowing(self):
        b = self._builder_with_chain(1, [{"id": "m1"}])
        db = MagicMock()
        db.table.return_value = b
        count = org_projects.revoke_org_granted_memberships(db, ORG_ID)
        assert count == 1
        b.delete.return_value.eq.assert_called_once_with("org_id", ORG_ID)

    def test_narrowed_by_user_id(self):
        b = self._builder_with_chain(2, [{"id": "m1"}])
        db = MagicMock()
        db.table.return_value = b
        count = org_projects.revoke_org_granted_memberships(db, ORG_ID, user_id=USER_ID)
        assert count == 1
        b.delete.return_value.eq.assert_called_once_with("org_id", ORG_ID)
        b.delete.return_value.eq.return_value.eq.assert_called_once_with("user_id", USER_ID)

    def test_narrowed_by_project_id(self):
        b = self._builder_with_chain(2, [{"id": "m1"}, {"id": "m2"}])
        db = MagicMock()
        db.table.return_value = b
        count = org_projects.revoke_org_granted_memberships(db, ORG_ID, project_id=PROJECT_ID)
        assert count == 2
        b.delete.return_value.eq.assert_called_once_with("org_id", ORG_ID)
        b.delete.return_value.eq.return_value.eq.assert_called_once_with("project_id", PROJECT_ID)

    def test_returns_zero_when_nothing_matches(self):
        b = self._builder_with_chain(1, [])
        db = MagicMock()
        db.table.return_value = b
        assert org_projects.revoke_org_granted_memberships(db, ORG_ID) == 0


# ---------------------------------------------------------------------------
# list_org_projects (admin console view)
# ---------------------------------------------------------------------------


class TestListOrgProjects:
    async def test_requires_admin_403(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.list_org_projects(db, OTHER_USER_ID, ORG_ID)
        assert exc_info.value.status_code == 403

    async def test_empty_when_no_links(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: True)
        db = _db_seq({"org_project_links": [MagicMock(data=[], count=0)]})
        result = await org_projects.list_org_projects(db, USER_ID, ORG_ID)
        assert result == []

    async def test_shape_with_owner_email_from_org_members_seat(self, monkeypatch):
        """Owner holds an active seat in this org — ownerEmail must come off
        org_members.email with ZERO fallback auth lookups."""
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: True)
        fallback_calls = []
        monkeypatch.setattr(
            org_projects, "_resolve_user_email", lambda *a: fallback_calls.append(a) or "should-not-be-used"
        )

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_project_links":
                b.execute.return_value = MagicMock(
                    data=[{"id": LINK_ID, "project_id": PROJECT_ID, "created_at": "2026-07-20T00:00:00+00:00"}],
                    count=1,
                )
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": PROJECT_ID, "name": "Album One"}], count=1)
            elif name == "project_members":
                b.execute.return_value = MagicMock(
                    data=[
                        {"project_id": PROJECT_ID, "user_id": OTHER_USER_ID, "role": "owner", "org_id": None},
                        {"project_id": PROJECT_ID, "user_id": "u-granted", "role": "editor", "org_id": ORG_ID},
                    ],
                    count=2,
                )
            elif name == "org_members":
                b.execute.return_value = MagicMock(
                    data=[{"user_id": OTHER_USER_ID, "email": "owner@example.com", "status": "active"}], count=1
                )
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.list_org_projects(db, USER_ID, ORG_ID)
        assert len(result) == 1
        row = result[0]
        assert row["projectId"] == PROJECT_ID
        assert row["name"] == "Album One"
        assert row["ownerEmail"] == "owner@example.com"
        assert row["linkedAt"] == "2026-07-20T00:00:00+00:00"
        assert row["orgGrantedMemberCount"] == 1
        assert fallback_calls == []

    async def test_shape_owner_email_falls_back_when_owner_has_no_seat(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(org_projects, "_resolve_user_email", lambda db, uid: f"{uid}@fallback.example.com")

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_project_links":
                b.execute.return_value = MagicMock(
                    data=[{"id": LINK_ID, "project_id": PROJECT_ID, "created_at": "2026-07-20T00:00:00+00:00"}],
                    count=1,
                )
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": PROJECT_ID, "name": "Album One"}], count=1)
            elif name == "project_members":
                b.execute.return_value = MagicMock(
                    data=[{"project_id": PROJECT_ID, "user_id": OTHER_USER_ID, "role": "owner", "org_id": None}],
                    count=1,
                )
            elif name == "org_members":
                b.execute.return_value = MagicMock(data=[], count=0)  # owner holds no seat here
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.list_org_projects(db, USER_ID, ORG_ID)
        assert result[0]["ownerEmail"] == f"{OTHER_USER_ID}@fallback.example.com"

    async def test_org_granted_member_count_excludes_owner_and_other_org_rows(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(org_projects, "_resolve_user_email", lambda *a: None)

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_project_links":
                b.execute.return_value = MagicMock(
                    data=[{"id": LINK_ID, "project_id": PROJECT_ID, "created_at": "2026-07-20T00:00:00+00:00"}],
                    count=1,
                )
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": PROJECT_ID, "name": "Album One"}], count=1)
            elif name == "project_members":
                b.execute.return_value = MagicMock(
                    data=[
                        {"project_id": PROJECT_ID, "user_id": OTHER_USER_ID, "role": "owner", "org_id": None},
                        {"project_id": PROJECT_ID, "user_id": "u-organic", "role": "viewer", "org_id": None},
                        {"project_id": PROJECT_ID, "user_id": "u-this-org-1", "role": "editor", "org_id": ORG_ID},
                        {"project_id": PROJECT_ID, "user_id": "u-this-org-2", "role": "viewer", "org_id": ORG_ID},
                        {
                            "project_id": PROJECT_ID,
                            "user_id": "u-other-org",
                            "role": "editor",
                            "org_id": OTHER_ORG_ID,
                        },
                    ],
                    count=5,
                )
            elif name == "org_members":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await org_projects.list_org_projects(db, USER_ID, ORG_ID)
        assert result[0]["orgGrantedMemberCount"] == 2


# ---------------------------------------------------------------------------
# set_org_project_member_role (Task 3 AC 1) — PUT .../members/{member_id}
# ---------------------------------------------------------------------------


def _authorize_admin(monkeypatch):
    monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: True)


class TestSetOrgProjectMemberRoleNoLeak404s:
    async def test_unknown_or_inactive_seat_member_404(self, monkeypatch):
        """Covers unknown member_id, wrong org, AND suspended/removed status
        identically — the mocked query simply returns no row, which is what
        the real `.eq("status", "active")` filter would produce for any of
        those three cases."""
        _authorize_admin(monkeypatch)
        db = _db_seq({"org_members": [MagicMock(data=None, count=0)]})
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.set_org_project_member_role(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Member not found"

    async def test_unlinked_project_404(self, monkeypatch):
        """Project doesn't exist, or isn't linked to THIS org (unlinked or
        linked to a DIFFERENT org) — all collapse to the same 404 body."""
        _authorize_admin(monkeypatch)
        db = _db_seq(
            {
                "org_members": [MagicMock(data={"user_id": TARGET_USER_ID}, count=1)],
                "org_project_links": [MagicMock(data=None, count=0)],
            }
        )
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.set_org_project_member_role(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Project not found"

    async def test_non_admin_caller_403(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.set_org_project_member_role(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor")
        assert exc_info.value.status_code == 403


class TestSetOrgProjectMemberRoleDecisionTree:
    def _side_for(self, pm_rows_by_call, captured):
        """pm_rows_by_call: list of MagicMock execute() return values consumed
        in order for the "project_members" table. The 2nd+ call (INSERT or
        UPDATE) has its payload captured into `captured`."""
        pm_calls = {"n": 0}

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_members":
                b.execute.return_value = MagicMock(data={"user_id": TARGET_USER_ID}, count=1)
            elif name == "org_project_links":
                b.execute.return_value = MagicMock(data={"id": LINK_ID}, count=1)
            elif name == "project_members":
                pm_calls["n"] += 1
                idx = min(pm_calls["n"] - 1, len(pm_rows_by_call) - 1)
                b.execute.return_value = pm_rows_by_call[idx]
                if pm_calls["n"] == 2:
                    original_insert = b.insert
                    original_update = b.update

                    def _capture_insert(payload, *a, **kw):
                        captured["insert_payload"] = payload
                        return original_insert(payload, *a, **kw)

                    def _capture_update(payload, *a, **kw):
                        captured["update_payload"] = payload
                        return original_update(payload, *a, **kw)

                    b.insert = _capture_insert
                    b.update = _capture_update
            return b

        db = MagicMock()
        db.table.side_effect = _side
        return db, pm_calls

    async def test_fresh_insert_stamps_org_id(self, monkeypatch):
        """rule 2: no existing row -> INSERT is the ONLY place org_id is
        stamped. Assert the exact insert payload."""
        _authorize_admin(monkeypatch)
        captured = {}
        db, pm_calls = self._side_for(
            [
                MagicMock(data=None, count=0),  # no existing row
                MagicMock(
                    data=[
                        {
                            "id": PM_ROW_ID,
                            "project_id": PROJECT_ID,
                            "user_id": TARGET_USER_ID,
                            "role": "editor",
                            "org_id": ORG_ID,
                        }
                    ],
                    count=1,
                ),
            ],
            captured,
        )

        result = await org_projects.set_org_project_member_role(
            db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor"
        )

        assert result["status"] == "granted"
        assert result["member"]["org_id"] == ORG_ID
        assert captured["insert_payload"] == {
            "project_id": PROJECT_ID,
            "user_id": TARGET_USER_ID,
            "role": "editor",
            "org_id": ORG_ID,
        }
        assert pm_calls["n"] == 2

    async def test_org_granted_role_update_keeps_org_id(self, monkeypatch):
        """rule 2: existing row already org_id = this org -> role UPDATE
        only, provenance untouched (org_id isn't in the update payload)."""
        _authorize_admin(monkeypatch)
        captured = {}
        db, pm_calls = self._side_for(
            [
                MagicMock(data={"id": PM_ROW_ID, "role": "editor", "org_id": ORG_ID}, count=1),
                MagicMock(
                    data=[
                        {
                            "id": PM_ROW_ID,
                            "project_id": PROJECT_ID,
                            "user_id": TARGET_USER_ID,
                            "role": "admin",
                            "org_id": ORG_ID,
                        }
                    ],
                    count=1,
                ),
            ],
            captured,
        )

        result = await org_projects.set_org_project_member_role(
            db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "admin"
        )

        assert result["status"] == "granted"
        assert result["member"]["org_id"] == ORG_ID
        assert captured["update_payload"] == {"role": "admin"}
        assert pm_calls["n"] == 2

    async def test_organic_row_is_byte_identical_noop(self, monkeypatch):
        """rule 2: an existing ORGANIC row (org_id IS NULL) must never be
        touched — no insert, no update, row survives byte-identical."""
        _authorize_admin(monkeypatch)
        captured = {}
        db, pm_calls = self._side_for(
            [MagicMock(data={"id": PM_ROW_ID, "role": "viewer", "org_id": None}, count=1)], captured
        )

        result = await org_projects.set_org_project_member_role(
            db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor"
        )

        assert result == {"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}
        assert pm_calls["n"] == 1  # only the read — no second (write) call
        assert "insert_payload" not in captured
        assert "update_payload" not in captured

    async def test_other_org_granted_row_treated_as_organic(self, monkeypatch):
        """A row granted by a DIFFERENT org is, from THIS org's perspective,
        exactly as untouchable as an organic row — same no-op shape, same
        byte-identical guarantee."""
        _authorize_admin(monkeypatch)
        captured = {}
        db, pm_calls = self._side_for(
            [MagicMock(data={"id": PM_ROW_ID, "role": "editor", "org_id": OTHER_ORG_ID}, count=1)], captured
        )

        result = await org_projects.set_org_project_member_role(
            db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor"
        )

        assert result == {"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}
        assert pm_calls["n"] == 1
        assert "insert_payload" not in captured
        assert "update_payload" not in captured

    async def test_owner_target_409(self, monkeypatch):
        """The owner row always has org_id IS NULL — this must resolve to
        the SPECIFIC owner 409, not the generic organic no-op, and the row
        must never be touched (checked ahead of the organic branch)."""
        _authorize_admin(monkeypatch)
        captured = {}
        db, pm_calls = self._side_for(
            [MagicMock(data={"id": PM_ROW_ID, "role": "owner", "org_id": None}, count=1)], captured
        )

        with pytest.raises(org_projects.ProjectMemberIsOwnerError) as exc_info:
            await org_projects.set_org_project_member_role(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID, "editor")
        assert str(exc_info.value) == "The project owner's access can't be managed by the organization."
        assert pm_calls["n"] == 1


# ---------------------------------------------------------------------------
# remove_org_project_member (Task 3 AC 2) — DELETE .../members/{member_id}
# ---------------------------------------------------------------------------


class TestRemoveOrgProjectMemberNoLeak404s:
    async def test_unknown_or_inactive_seat_member_404(self, monkeypatch):
        _authorize_admin(monkeypatch)
        db = _db_seq({"org_members": [MagicMock(data=None, count=0)]})
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Member not found"

    async def test_unlinked_project_404(self, monkeypatch):
        _authorize_admin(monkeypatch)
        db = _db_seq(
            {
                "org_members": [MagicMock(data={"user_id": TARGET_USER_ID}, count=1)],
                "org_project_links": [MagicMock(data=None, count=0)],
            }
        )
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Project not found"

    async def test_non_admin_caller_403(self, monkeypatch):
        monkeypatch.setattr(org_projects.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)
        assert exc_info.value.status_code == 403


class TestRemoveOrgProjectMemberDecisionTree:
    def _base_side(self, pm_row):
        def _side(name):
            b = MockQueryBuilder()
            if name == "org_members":
                b.execute.return_value = MagicMock(data={"user_id": TARGET_USER_ID}, count=1)
            elif name == "org_project_links":
                b.execute.return_value = MagicMock(data={"id": LINK_ID}, count=1)
            elif name == "project_members":
                b.execute.return_value = MagicMock(data=pm_row, count=1 if pm_row else 0)
            return b

        db = MagicMock()
        db.table.side_effect = _side
        return db

    async def test_org_granted_row_deleted_via_revoke_helper(self, monkeypatch):
        """AC 2: reuse `revoke_org_granted_memberships` with BOTH narrowing
        filters (project_id AND user_id) on top of the org_id filter."""
        _authorize_admin(monkeypatch)
        db = self._base_side({"id": PM_ROW_ID, "role": "editor", "org_id": ORG_ID})
        fake_revoke = MagicMock(return_value=1)
        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

        result = await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)

        assert result == {"status": "revoked", "revoked": 1}
        fake_revoke.assert_called_once_with(db, ORG_ID, user_id=TARGET_USER_ID, project_id=PROJECT_ID)

    async def test_organic_row_survives_no_revoke_call(self, monkeypatch):
        _authorize_admin(monkeypatch)
        db = self._base_side({"id": PM_ROW_ID, "role": "viewer", "org_id": None})
        fake_revoke = MagicMock(return_value=0)
        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

        result = await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)

        assert result == {"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}
        fake_revoke.assert_not_called()

    async def test_other_org_granted_row_treated_as_organic_no_revoke_call(self, monkeypatch):
        _authorize_admin(monkeypatch)
        db = self._base_side({"id": PM_ROW_ID, "role": "editor", "org_id": OTHER_ORG_ID})
        fake_revoke = MagicMock(return_value=0)
        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

        result = await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)

        assert result == {"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}
        fake_revoke.assert_not_called()

    async def test_owner_target_409_no_revoke_call(self, monkeypatch):
        _authorize_admin(monkeypatch)
        db = self._base_side({"id": PM_ROW_ID, "role": "owner", "org_id": None})
        fake_revoke = MagicMock(return_value=0)
        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

        with pytest.raises(org_projects.ProjectMemberIsOwnerError) as exc_info:
            await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)
        assert str(exc_info.value) == "The project owner's access can't be managed by the organization."
        fake_revoke.assert_not_called()

    async def test_no_existing_row_still_calls_revoke_and_reports_zero(self, monkeypatch):
        """No project_members row at all for this target on this project —
        there is nothing org-granted to revoke, so this reports revoked: 0
        rather than the "organic" shape (which would incorrectly claim the
        member has independent access)."""
        _authorize_admin(monkeypatch)
        db = self._base_side(None)
        fake_revoke = MagicMock(return_value=0)
        monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

        result = await org_projects.remove_org_project_member(db, USER_ID, ORG_ID, PROJECT_ID, SEAT_MEMBER_ID)

        assert result == {"status": "revoked", "revoked": 0}
        fake_revoke.assert_called_once_with(db, ORG_ID, user_id=TARGET_USER_ID, project_id=PROJECT_ID)


# ---------------------------------------------------------------------------
# Router wiring — success/analytics/error-mapping/flag-gate for the 3 new
# routes. Business-logic edge cases are covered above at the service level.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _licensing_on_by_default(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


class TestLinkRouteWiring:
    def test_link_ok_fires_analytics(self, client):
        with (
            patch(
                "orgs.router.org_projects.link_project",
                new=AsyncMock(return_value={"org_id": ORG_ID, "project_id": PROJECT_ID}),
            ),
            patch("orgs.router.analytics_capture") as mock_capture,
        ):
            resp = client.post(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 200
        assert resp.json()["org_id"] == ORG_ID
        mock_capture.assert_called_once()
        assert mock_capture.call_args.args[1] == "org_project_linked"

    def test_link_duplicate_maps_to_409(self, client):
        with patch(
            "orgs.router.org_projects.link_project",
            new=AsyncMock(
                side_effect=org_projects.ProjectAlreadyLinkedError(
                    "This project is already linked to an organization — unlink it first."
                )
            ),
        ):
            resp = client.post(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 409

    def test_link_404_propagates_from_service(self, client):
        with patch(
            "orgs.router.org_projects.link_project",
            new=AsyncMock(side_effect=HTTPException(status_code=404, detail="Project not found")),
        ):
            resp = client.post(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 404

    def test_link_route_404_when_flag_off(self, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = client.post(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 404


class TestUnlinkRouteWiring:
    def test_unlink_ok_fires_analytics_with_revoked_count(self, client):
        with (
            patch("orgs.router.org_projects.unlink_project", new=AsyncMock(return_value={"revoked": 3})),
            patch("orgs.router.analytics_capture") as mock_capture,
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 200
        assert resp.json() == {"revoked": 3}
        mock_capture.assert_called_once()
        assert mock_capture.call_args.args[1] == "org_project_unlinked"
        assert mock_capture.call_args.args[2]["revoked"] == 3

    def test_unlink_404_propagates_from_service(self, client):
        with patch(
            "orgs.router.org_projects.unlink_project",
            new=AsyncMock(side_effect=HTTPException(status_code=404, detail="Project not found")),
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 404

    def test_unlink_route_404_when_flag_off(self, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/link")
        assert resp.status_code == 404


class TestListOrgProjectsRouteWiring:
    def test_list_ok(self, client):
        payload = [
            {
                "projectId": PROJECT_ID,
                "name": "Album",
                "ownerEmail": "a@b.com",
                "linkedAt": "t",
                "orgGrantedMemberCount": 1,
            }
        ]
        with patch("orgs.router.org_projects.list_org_projects", new=AsyncMock(return_value=payload)):
            resp = client.get(f"/orgs/{ORG_ID}/projects")
        assert resp.status_code == 200
        assert resp.json() == {"projects": payload}

    def test_list_denied_for_non_admin_403(self, client):
        with patch(
            "orgs.router.org_projects.list_org_projects",
            new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
        ):
            resp = client.get(f"/orgs/{ORG_ID}/projects")
        assert resp.status_code == 403

    def test_list_route_404_when_flag_off(self, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = client.get(f"/orgs/{ORG_ID}/projects")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Router wiring — Task 3's two admin-membership routes. No analytics_capture
# patch/assert here: projects/router.py fires no event for its own
# add_member/update_member_role/remove_member, so Task 3 deliberately emits
# none either (see the router's inline comment).
# ---------------------------------------------------------------------------


class TestSetOrgProjectMemberRoleRouteWiring:
    def test_ok_granted(self, client):
        with patch(
            "orgs.router.org_projects.set_org_project_member_role",
            new=AsyncMock(return_value={"status": "granted", "member": {"role": "editor"}}),
        ):
            resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "editor"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "granted"

    def test_ok_organic_noop(self, client):
        with patch(
            "orgs.router.org_projects.set_org_project_member_role",
            new=AsyncMock(return_value={"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}),
        ):
            resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "editor"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "organic"

    def test_owner_target_maps_to_409(self, client):
        with patch(
            "orgs.router.org_projects.set_org_project_member_role",
            new=AsyncMock(
                side_effect=org_projects.ProjectMemberIsOwnerError(
                    "The project owner's access can't be managed by the organization."
                )
            ),
        ):
            resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "editor"})
        assert resp.status_code == 409

    def test_404_propagates_from_service(self, client):
        with patch(
            "orgs.router.org_projects.set_org_project_member_role",
            new=AsyncMock(side_effect=HTTPException(status_code=404, detail="Member not found")),
        ):
            resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "editor"})
        assert resp.status_code == 404

    def test_invalid_role_422(self, client):
        """The Literal-typed body model rejects 'owner' (and any other
        non-viewer/editor/admin value) before the request ever reaches the
        service layer."""
        resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "owner"})
        assert resp.status_code == 422

    def test_route_404_when_flag_off(self, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = client.put(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}", json={"role": "editor"})
        assert resp.status_code == 404


class TestRemoveOrgProjectMemberRouteWiring:
    def test_ok_revoked(self, client):
        with patch(
            "orgs.router.org_projects.remove_org_project_member",
            new=AsyncMock(return_value={"status": "revoked", "revoked": 1}),
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "revoked", "revoked": 1}

    def test_ok_organic_noop(self, client):
        with patch(
            "orgs.router.org_projects.remove_org_project_member",
            new=AsyncMock(return_value={"status": "organic", "detail": org_projects._ORGANIC_NOOP_DETAIL}),
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "organic"

    def test_owner_target_maps_to_409(self, client):
        with patch(
            "orgs.router.org_projects.remove_org_project_member",
            new=AsyncMock(
                side_effect=org_projects.ProjectMemberIsOwnerError(
                    "The project owner's access can't be managed by the organization."
                )
            ),
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}")
        assert resp.status_code == 409

    def test_404_propagates_from_service(self, client):
        with patch(
            "orgs.router.org_projects.remove_org_project_member",
            new=AsyncMock(side_effect=HTTPException(status_code=404, detail="Project not found")),
        ):
            resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}")
        assert resp.status_code == 404

    def test_route_404_when_flag_off(self, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = client.delete(f"/orgs/{ORG_ID}/projects/{PROJECT_ID}/members/{SEAT_MEMBER_ID}")
        assert resp.status_code == 404
