"""Licensing Phase C — Task 5: resource -> billing-org resolution.

Covers `EntitlementsService.resolve_billing_org_for_project` and
`resolve_billing_org_for_resource` — pure additions, no existing method is
touched (check_credits/can/get_for_user are Task 6/7 territory).

Reuses `test_billing_context`'s filter-aware `_ctx_store`/`_ctx_supabase` mock:
the shared no-op `MockQueryBuilder` ignores `.eq(...)`, so it could never
distinguish an active seat from a suspended one, or an active org from a
pending one. `_ctx_supabase` actually applies eq/in_ predicates and logs every
select's predicates, which is what lets the "licensing off -> zero queries"
test assert an empty log rather than just an empty result.
"""

from subscriptions.service import EntitlementsService
from tests.conftest import TEST_USER_ID
from tests.test_billing_context import MEMBER, ORG, _ctx_supabase, _member, _org

USER = TEST_USER_ID
PROJECT = "9b1d0000-0000-0000-0000-000000000001"
PROJECT_2 = "9b1d0000-0000-0000-0000-000000000002"
CONTRACT_1 = "f11e0000-0000-0000-0000-000000000001"
CONTRACT_2 = "f11e0000-0000-0000-0000-000000000002"
UNRESOLVABLE = "f11e0000-0000-0000-0000-00000000dead"

EXPECTED_CTX = {
    "org_id": ORG,
    "org_name": "Acme Records",
    "org_member_id": MEMBER,
    "role": "member",
    "project_id": PROJECT,
}


def _link(project_id=PROJECT, org_id=ORG):
    return {"project_id": project_id, "org_id": org_id}


def _base_data(*, org_status="active", member_status="active", link_project_id=PROJECT, project_files=None):
    return {
        "org_project_links": [_link(project_id=link_project_id)],
        "organizations": [_org(status=org_status)],
        "org_members": [_member(status=member_status)],
        "project_files": project_files or [],
    }


# ===========================================================================
# resolve_billing_org_for_project
# ===========================================================================


class TestResolveBillingOrgForProject:
    def test_linked_seat_active_returns_ctx_with_project_id(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data())
        ctx = EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT)
        assert ctx == EXPECTED_CTX

    def test_linked_no_seat_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data()
        data["org_members"] = []
        sb = _ctx_supabase(data)
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None

    def test_linked_pending_org_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data(org_status="pending"))
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None

    def test_linked_archived_org_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data()
        data["organizations"] = [_org(status="active", archived_at="2026-01-01T00:00:00+00:00")]
        sb = _ctx_supabase(data)
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None

    def test_unlinked_project_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data()
        data["org_project_links"] = []
        sb = _ctx_supabase(data)
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None

    def test_suspended_seat_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data(member_status="suspended"))
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None

    def test_licensing_off_returns_none_and_makes_zero_queries(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        sb = _ctx_supabase(_base_data())
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None
        assert not sb._log  # no table reads at all — the short-circuit is before any query

    def test_resolver_exception_returns_none_not_raise(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data())
        sb.table.side_effect = RuntimeError("boom")
        assert EntitlementsService(sb).resolve_billing_org_for_project(USER, PROJECT) is None


# ===========================================================================
# resolve_billing_org_for_resource
# ===========================================================================


class TestResolveBillingOrgForResource:
    def test_project_id_direct_delegates(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data())
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, project_id=PROJECT)
        assert ctx == EXPECTED_CTX

    def test_no_resource_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data())
        assert EntitlementsService(sb).resolve_billing_org_for_resource(USER) is None

    def test_single_contract_chain_resolves(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": PROJECT}])
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1])
        assert ctx == EXPECTED_CTX

    def test_multi_contract_same_project_resolves(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(
            project_files=[
                {"id": CONTRACT_1, "project_id": PROJECT},
                {"id": CONTRACT_2, "project_id": PROJECT},
            ]
        )
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1, CONTRACT_2])
        assert ctx == EXPECTED_CTX

    def test_multi_contract_different_projects_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(
            project_files=[
                {"id": CONTRACT_1, "project_id": PROJECT},
                {"id": CONTRACT_2, "project_id": PROJECT_2},
            ]
        )
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1, CONTRACT_2])
        assert ctx is None

    def test_one_unresolvable_contract_id_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": PROJECT}])
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(
            USER, contract_file_ids=[CONTRACT_1, UNRESOLVABLE]
        )
        assert ctx is None

    def test_contract_with_null_project_id_returns_none(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": None}])
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1])
        assert ctx is None

    def test_multi_contract_unanimous_but_no_seat_returns_none(self, monkeypatch):
        """Unanimity resolves to ONE project, but the caller has no seat there —
        falls through resolve_billing_org_for_project's own no-seat branch."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": PROJECT}])
        data["org_members"] = []
        sb = _ctx_supabase(data)
        ctx = EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1])
        assert ctx is None

    def test_licensing_off_returns_none_and_makes_zero_queries(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": PROJECT}])
        sb = _ctx_supabase(data)
        assert EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1]) is None
        assert not sb._log

    def test_resolver_exception_returns_none_not_raise(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _base_data(project_files=[{"id": CONTRACT_1, "project_id": PROJECT}])
        sb = _ctx_supabase(data)
        sb.table.side_effect = RuntimeError("boom")
        assert EntitlementsService(sb).resolve_billing_org_for_resource(USER, contract_file_ids=[CONTRACT_1]) is None

    def test_resolver_exception_on_project_id_path_returns_none_not_raise(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_base_data())
        sb.table.side_effect = RuntimeError("boom")
        assert EntitlementsService(sb).resolve_billing_org_for_resource(USER, project_id=PROJECT) is None
