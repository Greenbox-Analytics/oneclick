"""Licensing Phase B — Task 5: billing context + entitlements resolution.

Covers `EntitlementsService._resolve_context`, the org-context branch of
`get_for_user` (ENTERPRISE_SHAPE + seat wallet), the load-bearing RULE 11
(personal wallet never read/rolled in org context), pending-org persistence
(rule 7), the org-context storage-wall copy (rule 13), `availableContexts`,
personal-context byte-identical regression, and `PUT /me/billing-context`
(404-no-oracle parity).

Uses a purpose-built FILTER-AWARE mock (`_ctx_store`): the shared no-op
MockQueryBuilder ignores `.eq(...)`, so it could never distinguish an active
seat from a suspended one, a user wallet from a seat wallet, or prove the
personal wallet was never selected. `_CtxBuilder` actually applies eq/in_/
is_null predicates and LOGS every select's predicates so rule 11 can be pinned
precisely.
"""

from unittest.mock import MagicMock

import pytest

from subscriptions.models import Action
from subscriptions.service import EntitlementsService
from tests.conftest import TEST_USER_ID

USER = TEST_USER_ID
ORG = "0rg00000-0000-0000-0000-000000000001"
MEMBER = "mem00000-0000-0000-0000-000000000001"

FREE_TIER_ROW = {
    "tier": "free",
    "max_artists": 3,
    "max_projects": 3,
    "max_boards": -1,
    "max_tasks": 50,
    "max_storage_bytes": 1073741824,
    "max_split_sheets_per_month": 5,
    "max_oneclick_runs_per_month": 1,
    "zoe_enabled": False,
    "oneclick_enabled": True,
    "registry_enabled": False,
    "integrations_allowed": ["google_drive"],
    "monthly_credits": 50,
    "max_works": 10,
    "included_storage_bytes": 1073741824,
}
PRO_TIER_ROW = dict(FREE_TIER_ROW, tier="pro", monthly_credits=3000)

FAR_FUTURE = "2099-05-09T00:00:00+00:00"
STALE = "2020-01-01T00:00:00+00:00"

PRICES = [
    {"action": "zoe_message", "credits": 3},
    {"action": "oneclick_run", "credits": 21},
    {"action": "registry_parse", "credits": 12},
]


def _usage_row(storage=0):
    return {
        "user_id": USER,
        "total_storage_bytes": storage,
        "split_sheets_this_period": 0,
        "zoe_queries_this_period": 0,
        "oneclick_runs_this_period": 0,
        "period_start": "2026-05-09T00:00:00+00:00",
        "period_end": FAR_FUTURE,
    }


def _sub_row(tier="pro"):
    return {
        "user_id": USER,
        "tier": tier,
        "status": "active",
        "stripe_subscription_id": None,
        "stripe_price_id": None,
        "current_period_end": None,
        "cancel_at_period_end": False,
        "overage_enabled": False,
        "overage_cap_credits": None,
        "storage_overage_enabled": False,
    }


def _user_wallet(bundle=3000, reserve=0, period_end=STALE):
    return {
        "id": "wallet-personal",
        "owner_type": "user",
        "owner_id": USER,
        "bundle_balance": bundle,
        "reserve_balance": reserve,
        "overage_this_period": 0,
        "period_start": "2019-12-01T00:00:00+00:00",
        "period_end": period_end,
    }


def _seat_wallet(reserve=500, bundle=0):
    return {
        "id": "wallet-seat",
        "owner_type": "seat",
        "owner_id": MEMBER,
        "bundle_balance": bundle,
        "reserve_balance": reserve,
        "overage_this_period": 0,
        "period_start": None,
        "period_end": None,
    }


def _profile(context_org=None, is_admin=False):
    return {"id": USER, "billing_context_org_id": context_org, "is_admin": is_admin}


def _member(status="active", role="member"):
    return {"id": MEMBER, "org_id": ORG, "user_id": USER, "role": role, "status": status}


def _org(status="active", archived_at=None, name="Acme Records"):
    return {"id": ORG, "name": name, "status": status, "archived_at": archived_at}


# ---------------------------------------------------------------------------
# Filter-aware mock — applies eq/in_/is_null; logs select predicates.
# ---------------------------------------------------------------------------


class _CtxBuilder:
    def __init__(self, table, rows, log, updates):
        self._table = table
        self._rows = rows  # shared list reference — insert/upsert mutate it
        self._log = log
        self._updates = updates
        self._preds = []
        self._op = "select"

    # chainable no-ops
    def select(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def maybe_single(self, *a, **k):
        return self

    def single(self, *a, **k):
        return self

    def eq(self, col, val):
        self._preds.append(("eq", col, val))
        return self

    def neq(self, col, val):
        self._preds.append(("neq", col, val))
        return self

    def in_(self, col, vals):
        self._preds.append(("in", col, list(vals)))
        return self

    def is_(self, col, val):
        if val == "null":
            self._preds.append(("isnull", col, None))
        return self

    def update(self, payload):
        self._op = "update"
        self._updates.setdefault(self._table, []).append(payload)
        return self

    def insert(self, payload):
        self._op = "insert"
        if isinstance(payload, dict):
            self._rows.append(dict(payload))
        return self

    def upsert(self, payload, **k):
        self._op = "upsert"
        if isinstance(payload, dict):
            self._rows.append(dict(payload))
        return self

    def _match(self, row):
        for op, col, val in self._preds:
            rv = row.get(col)
            if op == "eq" and rv != val:
                return False
            if op == "neq" and rv == val:
                return False
            if op == "in" and rv not in val:
                return False
            if op == "isnull" and rv is not None:
                return False
        return True

    def execute(self):
        matched = [r for r in self._rows if self._match(r)]
        if self._op == "select":
            self._log.setdefault(self._table, []).append(list(self._preds))
            return MagicMock(data=matched, count=len(matched))
        # update/insert/upsert — return the (post-mutation) matched rows
        return MagicMock(data=matched or self._rows[-1:], count=len(matched) or 1)


def _ctx_store(data):
    """Return (table_side_effect_fn, log, updates, store)."""
    store = {k: [dict(r) for r in v] for k, v in data.items()}
    log: dict = {}
    updates: dict = {}

    def table(name):
        return _CtxBuilder(name, store.setdefault(name, []), log, updates)

    return table, log, updates, store


def _ctx_supabase(data):
    table_fn, log, updates, store = _ctx_store(data)
    sb = MagicMock()
    sb.table.side_effect = table_fn
    sb.rpc.return_value.execute.return_value = MagicMock(data=True)
    sb._log = log
    sb._updates = updates
    sb._store = store
    return sb


def _org_context_data(*, tier="pro", org_status="active", seat=_seat_wallet, user_wallet=_user_wallet, usage=0):
    return {
        "profiles": [_profile(context_org=ORG)],
        "subscriptions": [_sub_row(tier=tier)],
        "tier_entitlements": [PRO_TIER_ROW if tier == "pro" else FREE_TIER_ROW],
        "tier_overrides": [],
        "usage_counters": [_usage_row(usage)],
        "credit_prices": list(PRICES),
        "credit_ledger": [],
        "credit_wallets": [user_wallet(), seat()],
        "org_members": [_member()],
        "organizations": [_org(status=org_status)],
    }


# ===========================================================================
# RULE 11 — the personal wallet is UNTOUCHED in org context
# ===========================================================================


class TestRule11PersonalWalletUntouched:
    def test_org_context_never_reads_or_rolls_personal_wallet(self, monkeypatch):
        """THE rule-11 test: a Basic (pro) subscriber with a STALE personal period +
        3000 bundle reads entitlements in ACTIVE org context. The personal wallet
        must never be SELECTed (every credit_wallets query filters owner_type='seat',
        none 'user'), rollover_wallet must NEVER fire, the personal bundle stays 3000,
        and the credits block reflects the SEAT wallet (500)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())

        ent = EntitlementsService(sb).get_for_user(USER)

        # No rollover_wallet RPC at all.
        rpc_names = [c.args[0] for c in sb.rpc.call_args_list if c.args]
        assert "rollover_wallet" not in rpc_names

        # Every credit_wallets SELECT hit the SEAT wallet, never the personal one.
        wallet_queries = sb._log.get("credit_wallets", [])
        assert wallet_queries, "expected the seat wallet to be read"
        for preds in wallet_queries:
            assert ("eq", "owner_type", "seat") in preds
            assert ("eq", "owner_type", "user") not in preds

        # Personal wallet row is byte-for-byte unchanged (bundle still 3000).
        personal = next(w for w in sb._store["credit_wallets"] if w["owner_type"] == "user")
        assert personal["bundle_balance"] == 3000

        # Credits block is the seat wallet (reserve 500), not the personal 3000.
        assert ent.credits is not None
        assert ent.credits.balance == 500
        assert ent.credits.monthly_grant == 0
        payload = ent.to_dict()
        assert payload["credits"]["managedByOrg"] == {"orgId": ORG, "orgName": "Acme Records", "role": "member"}

    def test_org_context_does_not_mutate_subscriptions_tier(self, monkeypatch):
        """Enterprise shape is synthesized — subscriptions.tier is never written."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())
        EntitlementsService(sb).get_for_user(USER)
        assert not sb._updates.get("subscriptions")


# ===========================================================================
# ENTERPRISE_SHAPE in org context
# ===========================================================================


class TestEnterpriseShape:
    def test_caps_unlimited_counts_finite_storage(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "12345")
        sb = _ctx_supabase(_org_context_data(tier="free"))  # personal tier free — must NOT leak
        ent = EntitlementsService(sb).get_for_user(USER)

        assert ent.caps.max_artists == -1
        assert ent.caps.max_projects == -1
        assert ent.caps.max_works == -1
        assert ent.caps.max_oneclick_runs_per_month == -1
        assert ent.caps.max_split_sheets_per_month == -1
        # FINITE storage (rule 12) — max == included == the env value, NOT -1.
        assert ent.caps.max_storage_bytes == 12345
        assert ent.caps.included_storage_bytes == 12345
        # All features on regardless of the (free) personal tier.
        assert ent.features.zoe_enabled and ent.features.oneclick_enabled and ent.features.registry_enabled
        assert ent.features.integrations_allowed == ["google_drive", "slack"]

    def test_available_contexts_lists_personal_plus_active_seat(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())
        payload = EntitlementsService(sb).get_for_user(USER).to_dict()
        assert payload["availableContexts"] == [
            {"type": "personal"},
            {"type": "org", "orgId": ORG, "orgName": "Acme Records", "role": "member", "pending": False},
        ]


# ===========================================================================
# Storage-wall copy (rule 13) — org context points at SUPPORT
# ===========================================================================


class TestStorageWallCopy:
    def test_org_context_storage_wall_points_at_support(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "1000")
        sb = _ctx_supabase(_org_context_data(usage=1000))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1)
        assert not r.allowed
        assert r.reason == "Your organization seat's storage is full. Contact support to discuss options."

    def test_personal_context_storage_wall_keeps_legacy_copy(self, monkeypatch):
        """Regression: a personal-context wall must NOT get the support copy."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=None)],
            "subscriptions": [_sub_row(tier="free")],
            "tier_entitlements": [FREE_TIER_ROW],
            "tier_overrides": [],
            "usage_counters": [_usage_row(FREE_TIER_ROW["max_storage_bytes"])],
            "credit_prices": list(PRICES),
            "credit_ledger": [],
            "credit_wallets": [_user_wallet(period_end=FAR_FUTURE)],
            "org_members": [],
            "organizations": [],
        }
        sb = _ctx_supabase(data)
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1)
        assert not r.allowed
        assert "Contact support" not in (r.reason or "")


# ===========================================================================
# Forged / dead references — personal shape, NO org data, preference cleared
# ===========================================================================


class TestForgedContextFallsClosed:
    @pytest.mark.parametrize(
        ("members", "orgs"),
        [
            pytest.param([], [_org(status="active")], id="no_seat"),
            pytest.param([_member(status="active")], [], id="nonexistent_org"),
            pytest.param([_member(status="suspended")], [_org(status="active")], id="suspended_seat"),
            pytest.param([_member(status="removed")], [_org(status="active")], id="removed_seat"),
            pytest.param(
                [_member(status="active")], [_org(status="active", archived_at=FAR_FUTURE)], id="archived_org"
            ),
            pytest.param([_member(status="active")], [_org(status="suspended")], id="suspended_org"),
        ],
    )
    def test_dead_reference_falls_to_personal_and_clears(self, monkeypatch, members, orgs):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = _org_context_data()
        data["org_members"] = members
        data["organizations"] = orgs
        sb = _ctx_supabase(data)

        ent = EntitlementsService(sb).get_for_user(USER)
        payload = ent.to_dict()

        # Personal shape: pro tier caps (finite, from PRO_TIER_ROW), NOT enterprise.
        assert ent.managed_by_org is None
        assert "managedByOrg" not in (payload.get("credits") or {})
        # No org name/role anywhere in the credits block.
        assert payload["credits"]["balance"] == 3000  # personal wallet, not seat 500
        # Stale preference lazily cleared.
        prof_updates = sb._updates.get("profiles", [])
        assert {"billing_context_org_id": None} in prof_updates

    def test_resolve_context_returns_none_for_dead_ref(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        data = _org_context_data()
        data["org_members"] = []  # no seat
        sb = _ctx_supabase(data)
        assert EntitlementsService(sb)._resolve_context(USER) is None


# ===========================================================================
# Pending org — persists (rule 7)
# ===========================================================================


class TestPendingOrgPersists:
    def test_pending_context_personal_shape_but_not_cleared(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data(org_status="pending"))

        ent = EntitlementsService(sb).get_for_user(USER)
        payload = ent.to_dict()

        # Confers NOTHING yet — personal shape, no managedByOrg.
        assert ent.managed_by_org is None
        assert "managedByOrg" not in (payload.get("credits") or {})
        assert payload["credits"]["balance"] == 3000  # personal wallet

        # Preference NOT cleared (rule 7).
        assert not sb._updates.get("profiles")

        # availableContexts still surfaces the seat, marked pending.
        assert {
            "type": "org",
            "orgId": ORG,
            "orgName": "Acme Records",
            "role": "member",
            "pending": True,
        } in payload["availableContexts"]

    def test_resolve_context_pending_marker_not_active(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data(org_status="pending"))
        ctx = EntitlementsService(sb)._resolve_context(USER)
        assert ctx is not None and ctx["pending"] is True
        # And it did NOT clear the preference.
        assert not sb._updates.get("profiles")


# ===========================================================================
# Licensing OFF + personal byte-identical
# ===========================================================================


class TestPersonalAndLicensingOff:
    def test_licensing_off_short_circuits_resolution(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        sb = _ctx_supabase(_org_context_data())
        assert EntitlementsService(sb)._resolve_context(USER) is None
        # Short-circuit: no table reads at all in resolution.
        assert not sb._log

    def test_licensing_off_payload_has_no_available_contexts(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())  # even with a stored org preference
        payload = EntitlementsService(sb).get_for_user(USER).to_dict()
        assert "availableContexts" not in payload
        assert "billingContext" not in payload
        assert "managedByOrg" not in (payload.get("credits") or {})

    def test_licensing_on_personal_only_adds_available_contexts_key(self, monkeypatch):
        """The ONLY allowed delta when licensing is ON (personal context) is the
        availableContexts + billingContext keys — everything else is byte-identical."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = _org_context_data()
        data["profiles"] = [_profile(context_org=None)]  # personal
        data["org_members"] = []  # no seats

        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        off = EntitlementsService(_ctx_supabase(data)).get_for_user(USER).to_dict()

        monkeypatch.setenv("LICENSING_ENABLED", "true")
        on = EntitlementsService(_ctx_supabase(data)).get_for_user(USER).to_dict()

        assert "availableContexts" not in off
        assert "billingContext" not in off
        assert on.pop("availableContexts") == [{"type": "personal"}]
        assert on.pop("billingContext") == {"type": "personal"}
        assert on == off  # identical once the two new keys are removed


# ===========================================================================
# Task 3 (licensing follow-ups plan) — billingContext field + credits-off
# org-context gating. This is the four-cell (licensing, credits) x (off, on)
# flag matrix; it IS the acceptance criterion, not prose claims of "identical".
# ===========================================================================


class TestBillingContextFieldMatrix:
    def test_licensing_off_credits_off_cell(self, monkeypatch):
        """Cell (licensing off, credits off): no billingContext, no
        availableContexts, no credits block — a stored org preference is
        simply irrelevant when licensing is off."""
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        sb = _ctx_supabase(_org_context_data())
        payload = EntitlementsService(sb).get_for_user(USER).to_dict()
        assert "billingContext" not in payload
        assert "availableContexts" not in payload
        assert payload["credits"] is None

    def test_licensing_off_credits_on_cell(self, monkeypatch):
        """Cell (licensing off, credits on): still no billingContext /
        availableContexts — licensing gates both, independent of credits."""
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())
        payload = EntitlementsService(sb).get_for_user(USER).to_dict()
        assert "billingContext" not in payload
        assert "availableContexts" not in payload

    def test_licensing_on_credits_on_org_context_cell(self, monkeypatch):
        """Cell (licensing on, credits on), org context: TODAY's payload
        (seat-wallet credits block + credits.managedByOrg) PLUS the new
        top-level billingContext key — a pure addition, no-op on the gate."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_context_data())
        payload = EntitlementsService(sb).get_for_user(USER).to_dict()

        assert payload["billingContext"] == {
            "type": "org",
            "orgId": ORG,
            "orgName": "Acme Records",
            "role": "member",
        }
        # Today's payload is unchanged: seat-wallet credits block, managedByOrg.
        assert payload["credits"]["balance"] == 500
        assert payload["credits"]["managedByOrg"] == {"orgId": ORG, "orgName": "Acme Records", "role": "member"}

    def test_licensing_on_credits_off_org_context_cell(self, monkeypatch):
        """Cell (licensing on, credits off), org context — THE behavior change.
        credits is None (no seat-wallet block, mirroring the personal branch);
        billingContext still identifies the org (managed_by_org is set
        unconditionally); and — the key regression guard for round-3 directive
        2 — the seat wallet is NEVER read or created: no credit_wallets query
        fires at all (read_or_create_seat_wallet's only DB access is
        `.table("credit_wallets")`, so its absence from the query log proves
        the read+create never happened)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        sb = _ctx_supabase(_org_context_data())

        ent = EntitlementsService(sb).get_for_user(USER)
        payload = ent.to_dict()

        assert ent.credits is None
        assert payload["credits"] is None
        assert payload["billingContext"] == {
            "type": "org",
            "orgId": ORG,
            "orgName": "Acme Records",
            "role": "member",
        }

        # NO seat-wallet read+create call at all.
        assert "credit_wallets" not in sb._log
        assert not sb._updates.get("credit_wallets")

    def test_licensing_on_credits_off_personal_context_cell(self, monkeypatch):
        """Personal-context sibling of the cell above: billingContext is still
        emitted (licensing on) but type=='personal'; credits stays None
        (pre-existing behavior, unaffected by this task)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        data = _org_context_data()
        data["profiles"] = [_profile(context_org=None)]
        data["org_members"] = []
        sb = _ctx_supabase(data)

        ent = EntitlementsService(sb).get_for_user(USER)
        payload = ent.to_dict()

        assert ent.credits is None
        assert payload["credits"] is None
        assert payload["billingContext"] == {"type": "personal"}

    def test_licensing_on_credits_off_org_storage_cap_deny_still_support_copy(self, monkeypatch):
        """Directive 1 (verify-not-move): the storage-wall support copy (rule
        13, service.py's UPLOAD_BYTES branch) keys off `managed_by_org` alone,
        independent of credits_enabled() — so gating the credits block does
        NOT regress it, even with credits fully off."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "1000")
        sb = _ctx_supabase(_org_context_data(usage=1000))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1)
        assert not r.allowed
        assert r.reason == "Your organization seat's storage is full. Contact support to discuss options."


class TestToDictRegressionSnapshot:
    def test_to_dict_personal_shape_regression_snapshot(self):
        """Deep-equal against a captured pre-licensing shape: to_dict for a personal
        Entitlements (no managed_by_org, available_contexts=None) is byte-identical."""
        from datetime import datetime

        from subscriptions.models import Caps, CreditsInfo, Entitlements, Features, Usage

        ent = Entitlements(
            user_id=USER,
            tier="pro",
            status="active",
            caps=Caps(
                max_artists=-1,
                max_projects=-1,
                max_tasks=-1,
                max_storage_bytes=-1,
                max_split_sheets_per_month=-1,
                max_oneclick_runs_per_month=-1,
                monthly_credits=3000,
                max_works=-1,
                included_storage_bytes=107374182400,
            ),
            features=Features(
                zoe_enabled=True,
                oneclick_enabled=True,
                registry_enabled=True,
                integrations_allowed=["google_drive", "slack"],
            ),
            usage=Usage(
                total_storage_bytes=0,
                split_sheets_this_period=0,
                zoe_queries_this_period=0,
                oneclick_runs_this_period=0,
                period_end=datetime.fromisoformat("2099-05-09T00:00:00+00:00"),
            ),
            has_overrides=False,
            credits=CreditsInfo(
                bundle_balance=3000,
                reserve_balance=0,
                monthly_grant=3000,
                overage_this_period=0,
                overage_enabled=False,
                overage_cap_credits=None,
                storage_overage_enabled=False,
                period_end=None,
                prices={"zoe_message": 3, "oneclick_run": 21, "registry_parse": 12},
            ),
        )
        assert ent.to_dict() == {
            "tier": "pro",
            "status": "active",
            "caps": {
                "maxArtists": -1,
                "maxProjects": -1,
                "maxTasks": -1,
                "maxStorageBytes": -1,
                "maxSplitSheetsPerMonth": -1,
                "maxOneclickRunsPerMonth": -1,
                "maxWorks": -1,
                "includedStorageBytes": 107374182400,
                "monthlyCredits": 3000,
            },
            "features": {
                "zoeEnabled": True,
                "oneclickEnabled": True,
                "registryEnabled": True,
                "integrationsAllowed": ["google_drive", "slack"],
            },
            "usage": {
                "totalStorageBytes": 0,
                "splitSheetsThisPeriod": 0,
                "zoeQueriesThisPeriod": 0,
                "oneclickRunsThisPeriod": 0,
                "periodEnd": "2099-05-09T00:00:00+00:00",
            },
            "credits": {
                "balance": 3000,
                "bundleBalance": 3000,
                "reserveBalance": 0,
                "monthlyGrant": 3000,
                "overageThisPeriod": 0,
                "overageEnabled": False,
                "overageCapCredits": None,
                "storageOverageEnabled": False,
                "periodEnd": None,
                "prices": {"zoeMessage": 3, "oneclickRun": 21, "registryParse": 12},
            },
            "hasOverrides": False,
            "degraded": False,
            "subscription": {
                "stripeSubscriptionId": None,
                "stripePriceId": None,
                "currentPeriodEnd": None,
                "cancelAtPeriodEnd": False,
                "planPeriod": None,
            },
        }


# ===========================================================================
# PUT /me/billing-context — 404-no-oracle parity + accept/null
# ===========================================================================


def _wire_client(mock_supabase, data):
    table_fn, log, updates, store = _ctx_store(data)
    mock_supabase.table.side_effect = table_fn
    return updates, store


class TestPutBillingContext:
    def test_null_switches_to_personal_always(self, client, mock_supabase):
        updates, _ = _wire_client(mock_supabase, {"profiles": [_profile(context_org=ORG)]})
        resp = client.put("/me/billing-context", json={"orgId": None})
        assert resp.status_code == 200
        assert resp.json() == {"context": "personal"}
        assert {"billing_context_org_id": None} in updates.get("profiles", [])

    def test_active_seat_active_org_accepted(self, client, mock_supabase):
        updates, _ = _wire_client(
            mock_supabase,
            {"profiles": [_profile()], "org_members": [_member()], "organizations": [_org(status="active")]},
        )
        resp = client.put("/me/billing-context", json={"orgId": ORG})
        assert resp.status_code == 200
        assert resp.json() == {"context": "org", "orgId": ORG}
        assert {"billing_context_org_id": ORG} in updates.get("profiles", [])

    def test_pending_org_accepted(self, client, mock_supabase):
        """Rule 7: PUT accepts a pending org the caller holds an active seat in."""
        _wire_client(
            mock_supabase,
            {"profiles": [_profile()], "org_members": [_member()], "organizations": [_org(status="pending")]},
        )
        resp = client.put("/me/billing-context", json={"orgId": ORG})
        assert resp.status_code == 200
        assert resp.json() == {"context": "org", "orgId": ORG}

    @pytest.mark.parametrize(
        ("members", "orgs"),
        [
            pytest.param([], [_org(status="active")], id="no_seat"),
            pytest.param([], [], id="nonexistent_org"),
            pytest.param([_member(status="suspended")], [_org(status="active")], id="suspended_seat"),
            pytest.param(
                [_member(status="active")], [_org(status="active", archived_at=FAR_FUTURE)], id="archived_org"
            ),
            pytest.param([_member(status="active")], [_org(status="suspended")], id="suspended_org"),
        ],
    )
    def test_404_parity_no_oracle(self, client, mock_supabase, members, orgs):
        """Identical 404 status + detail for nonexistent org, no seat, suspended
        seat, archived org, suspended org — no existence oracle (rule 7)."""
        _wire_client(mock_supabase, {"profiles": [_profile()], "org_members": members, "organizations": orgs})
        resp = client.put("/me/billing-context", json={"orgId": ORG})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Organization not found"


# ===========================================================================
# Licensing Phase C — Task 7: caps derivation in can() (CREATE_WORK counts +
# UPLOAD_BYTES storage). Ambient stays PERSONAL (billing_context_org_id=None) —
# the derivation keys on the RESOURCE's project link + the caller's seat, NOT on
# ambient context. Rule 9 collision fix: storage derivation fires ONLY when the
# caller IS the storage-counter owner. Rule 4: upgrade-only, never restricts.
# ===========================================================================

GB = 1024 * 1024 * 1024
PROJECT = "proj0000-0000-0000-0000-000000000001"
HOST = "h0st0000-0000-0000-0000-000000000009"


def _link(project_id=PROJECT, org_id=ORG):
    return {"id": "link-1", "project_id": project_id, "org_id": org_id}


def _phase_c_data(*, tier="free", storage=0, linked=True, seat=True, org_status="active"):
    """Single-user, ambient-PERSONAL scenario: the project MAY be linked to an
    org where the caller MAY hold an active seat. Credits deliberately off — caps
    derivation is credits-independent, and leaving them off keeps the mock free of
    wallet reads."""
    return {
        "profiles": [_profile(context_org=None)],  # ambient PERSONAL — the Task 7 point
        "subscriptions": [_sub_row(tier=tier)],
        "tier_entitlements": [FREE_TIER_ROW if tier == "free" else PRO_TIER_ROW],
        "tier_overrides": [],
        "usage_counters": [_usage_row(storage)],
        "credit_prices": list(PRICES),
        "credit_ledger": [],
        "credit_wallets": [],
        "org_members": [_member()] if seat else [],
        "organizations": [_org(status=org_status)],
        "org_project_links": [_link()] if linked else [],
    }


class TestCreateWorkCapsDerivation:
    def test_work_11_in_linked_project_allowed(self, monkeypatch):
        """THE §5 mismatch: free cap 10, ambient personal, creating work #11 in a
        project linked to an org where the caller holds a seat → enterprise -1
        count cap applies → ALLOWED."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_phase_c_data())
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, resource_project_id=PROJECT, current_count=10)
        assert r.allowed is True
        assert "org_project_links" in sb._log  # derivation actually consulted the link

    def test_unlinked_project_uses_personal_cap(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_phase_c_data(linked=False))
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, resource_project_id=PROJECT, current_count=10)
        assert r.allowed is False
        assert r.reason == "You've reached your limit of 10 registered works."

    def test_linked_but_no_seat_uses_personal_cap(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_phase_c_data(seat=False))
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, resource_project_id=PROJECT, current_count=10)
        assert r.allowed is False

    def test_linked_pending_org_uses_personal_cap(self, monkeypatch):
        """Derivation requires an ACTIVE org (rule 4) — a pending org confers nothing."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_phase_c_data(org_status="pending"))
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, resource_project_id=PROJECT, current_count=10)
        assert r.allowed is False

    def test_no_resource_project_id_byte_identical(self, monkeypatch):
        """No resource id → resolver is never called, personal cap enforced."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = _ctx_supabase(_phase_c_data())
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, current_count=10)
        assert r.allowed is False
        assert "org_project_links" not in sb._log

    def test_licensing_off_no_derivation(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        sb = _ctx_supabase(_phase_c_data())
        r = EntitlementsService(sb).can(USER, Action.CREATE_WORK, resource_project_id=PROJECT, current_count=10)
        assert r.allowed is False
        assert "org_project_links" not in sb._log  # resolver short-circuits before any query


class TestUploadBytesCapsDerivation:
    def test_free_user_beyond_personal_cap_allowed_in_linked_project(self, monkeypatch):
        """Free personal cap 1GB, already storing 2GB (over personal), uploading
        into a linked project where the caller holds a seat → org's 10GB seat
        storage applies → ALLOWED. Caller IS the owner (host defaults to None)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(10 * GB))
        sb = _ctx_supabase(_phase_c_data(storage=2 * GB))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1, resource_project_id=PROJECT)
        assert r.allowed is True
        assert "org_project_links" in sb._log

    def test_over_seat_storage_denies_with_support_copy(self, monkeypatch):
        """Seat storage 2GB (> free 1GB → the binding cap); already at 2GB → deny
        with the round-5 support copy, not the legacy owner-limit copy."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(2 * GB))
        sb = _ctx_supabase(_phase_c_data(storage=2 * GB))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1, resource_project_id=PROJECT)
        assert r.allowed is False
        assert r.reason == "Your organization seat's storage is full. Contact support to discuss options."

    def test_unlinked_project_uses_personal_cap_legacy_copy(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(10 * GB))
        sb = _ctx_supabase(_phase_c_data(storage=1 * GB, linked=False))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1, resource_project_id=PROJECT)
        assert r.allowed is False
        assert "project owner's storage limit" in r.reason
        assert "Contact support" not in r.reason

    def test_upgrade_only_never_shrinks_larger_personal_headroom(self, monkeypatch):
        """Rule 4 upgrade-only: a personal cap LARGER than the seat storage is not
        shrunk by the link. Free 1GB personal, seat 512MB, already storing 800MB,
        upload 1 byte → still allowed (personal 1GB is more permissive and kept)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(512 * 1024 * 1024))
        sb = _ctx_supabase(_phase_c_data(storage=800 * 1024 * 1024))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1, resource_project_id=PROJECT)
        assert r.allowed is True

    def test_no_resource_project_id_byte_identical(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(10 * GB))
        sb = _ctx_supabase(_phase_c_data(storage=1 * GB))
        r = EntitlementsService(sb).can(USER, Action.UPLOAD_BYTES, size=1)
        assert r.allowed is False  # personal 1GB cap, no derivation
        assert "org_project_links" not in sb._log


# ---------------------------------------------------------------------------
# THE collision fix (rule 9): a seat-holding COLLABORATOR uploading to SOMEONE
# ELSE's linked project keeps the host-scoped check UNTOUCHED — no enterprise
# substitution, byte-identical deny copy, and NO org_project_links query.
# ---------------------------------------------------------------------------


def _collision_data():
    """USER holds a seat in ORG and PROJECT is linked to ORG — BUT the upload
    targets HOST's project (host_user_id=HOST). Storage is owner-scoped, so the
    host's counter/cap is authoritative and derivation MUST NOT fire."""
    return {
        "profiles": [
            _profile(context_org=None),
            {"id": HOST, "billing_context_org_id": None, "is_admin": False},
        ],
        "subscriptions": [_sub_row(tier="free"), {**_sub_row(tier="free"), "user_id": HOST}],
        "tier_entitlements": [FREE_TIER_ROW],
        "tier_overrides": [],
        "usage_counters": [
            {**_usage_row(0), "user_id": USER},
            {**_usage_row(1 * GB), "user_id": HOST},  # host storage at the free 1GB cap
        ],
        "credit_prices": list(PRICES),
        "credit_ledger": [],
        "credit_wallets": [],
        "org_members": [_member()],  # USER's active seat
        "organizations": [_org(status="active")],
        "org_project_links": [_link()],  # project IS linked...
    }


class TestUploadCollaboratorCollision:
    def test_collaborator_upload_to_host_linked_project_keeps_host_scope(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(500 * GB))
        sb = _ctx_supabase(_collision_data())
        # host_user_id=HOST != USER → owner-scoped host path; derivation is NOT
        # attempted even though PROJECT is linked and USER holds a seat.
        r = EntitlementsService(sb).can(
            USER, Action.UPLOAD_BYTES, size=1, host_user_id=HOST, resource_project_id=PROJECT
        )
        assert r.allowed is False
        # Legacy host-scoped copy — NO enterprise substitution.
        assert "project owner's storage limit" in r.reason
        assert "Contact support" not in r.reason
        # The load-bearing assertion: derivation never touched the link table.
        assert "org_project_links" not in sb._log
