from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder, _default_table_side_effect

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _not_owned_router(name):
    # Subscription tables must still return valid Pro data so gated_feature passes;
    # otherwise the EntitlementsService crashes before the ownership guard runs.
    if name in _SUBSCRIPTION_TABLES:
        return _default_table_side_effect(name)
    b = MockQueryBuilder()
    if name == "project_members":
        b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
    elif name == "artists":
        b.execute.return_value = MagicMock(data=[], count=0)  # owns nothing (fallback)
    elif name == "project_files":
        b.execute.return_value = MagicMock(data=[{"project_id": "victim-proj"}], count=1)
    return b


def test_calculate_rejects_unowned_inputs(client, mock_supabase):
    mock_supabase.table.side_effect = _not_owned_router
    resp = client.post(
        "/oneclick/calculate-royalties",
        json={"project_id": "victim-proj", "royalty_statement_file_id": "stmt-1", "contract_ids": ["c1"]},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Access denied"


def test_stream_rejects_unowned_inputs(client, mock_supabase):
    mock_supabase.table.side_effect = _not_owned_router
    resp = client.get(
        "/oneclick/calculate-royalties-stream",
        params={"project_id": "victim-proj", "royalty_statement_file_id": "stmt-1", "contract_ids": "c1"},
    )
    assert resp.status_code == 403


def test_confirm_rejects_unowned_inputs(client, mock_supabase):
    mock_supabase.table.side_effect = _not_owned_router
    # ConfirmCalculationRequest required fields: contract_ids, royalty_statement_id, project_id, results
    resp = client.post(
        "/oneclick/confirm",
        json={
            "royalty_statement_id": "stmt-1",
            "project_id": "victim-proj",
            "contract_ids": ["c1"],
            "results": {},
        },
    )
    assert resp.status_code == 403
