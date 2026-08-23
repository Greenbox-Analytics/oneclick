"""The `workspaceScope` half of the entitlements payload.

This field is the frontend's ONLY signal that workspace scoping is live:
useWorkspaceScope goes inert (no `?scope=` sent, `UNSCOPED_KEY` cache key)
whenever the key is absent, so the omission IS the rollback path and must be
byte-exact — not null, not {}.

Resolution deliberately rides the ACCESS predicate (live_org_ids), not the
billing one (_resolve_context): a pending/suspended org bills personal but is
still the caller's workspace. test_billing_context.py owns the billing side;
these tests pin the divergence.

Reuses test_billing_context's filter-aware store so resolve_scope's queries
(profiles → org_members → organizations) actually filter.
"""

import pytest

from subscriptions.service import EntitlementsService
from tests.test_billing_context import (
    ORG,
    USER,
    _ctx_supabase,
    _member,
    _org,
    _org_context_data,
    _profile,
)


@pytest.fixture
def scoping_on(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SCOPING_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def _payload(data):
    return EntitlementsService(_ctx_supabase(data)).get_for_user(USER).to_dict()


def test_flag_off_omits_the_key_entirely(monkeypatch):
    """Rollback: LICENSING on alone must not grow the payload — the frontend
    reads absence, so `None` serialized in would break the inert path."""
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.delenv("WORKSPACE_SCOPING_ENABLED", raising=False)

    assert "workspaceScope" not in _payload(_org_context_data())


def test_licensing_off_omits_the_key_even_with_scoping_on(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SCOPING_ENABLED", "true")
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)

    assert "workspaceScope" not in _payload(_org_context_data())


def test_stored_org_context_emits_the_org_scope(scoping_on):
    payload = _payload(_org_context_data())

    assert payload["workspaceScope"] == {"type": "org", "orgId": ORG, "orgName": "Acme Records"}


def test_no_stored_context_emits_personal(scoping_on):
    data = _org_context_data()
    data["profiles"] = [_profile(context_org=None)]

    assert _payload(data)["workspaceScope"] == {"type": "personal"}


def test_stale_stored_context_degrades_to_personal(scoping_on):
    """Offboarding: the stored org id points at a seat the caller no longer
    holds — the workspace quietly falls back to personal, mirroring
    resolve_scope, rather than erroring the whole entitlements read."""
    data = _org_context_data()
    data["org_members"] = []

    assert _payload(data)["workspaceScope"] == {"type": "personal"}


def test_pending_org_is_still_the_workspace(scoping_on):
    """THE divergence from billing: a pending org resolves to a PERSONAL
    billing context (_resolve_context parks it) but remains the caller's
    workspace — live_org_ids only excludes archived and lapsed. Keying the
    view off the billing answer would blank the org's whole roster."""
    data = _org_context_data(org_status="pending")
    payload = _payload(data)

    assert payload["billingContext"] == {"type": "personal"}
    assert payload["workspaceScope"]["type"] == "org"
    assert payload["workspaceScope"]["orgId"] == ORG


def test_lapsed_org_is_not_the_workspace(scoping_on):
    """A lapsed org is inert for EVERYONE (same posture as can_access_artist),
    so the stored context degrades to personal."""
    data = _org_context_data(org_status="lapsed")

    assert _payload(data)["workspaceScope"] == {"type": "personal"}


def test_suspended_seat_is_not_the_workspace(scoping_on):
    data = _org_context_data()
    data["org_members"] = [_member(status="suspended")]

    assert _payload(data)["workspaceScope"] == {"type": "personal"}


def test_org_name_rides_along_for_the_switcher_label(scoping_on):
    data = _org_context_data()
    data["organizations"] = [_org(name="Moonlight Collective")]

    assert _payload(data)["workspaceScope"]["orgName"] == "Moonlight Collective"
