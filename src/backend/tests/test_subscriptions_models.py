"""Tests for the subscriptions models module — type shape and enum coverage."""

from datetime import UTC, datetime


class TestCaps:
    def test_unlimited_sentinel(self):
        from subscriptions.models import Caps

        c = Caps(
            max_artists=-1,
            max_projects=-1,
            max_tasks=-1,
            max_storage_bytes=-1,
            max_split_sheets_per_month=-1,
            max_oneclick_runs_per_month=-1,
        )
        assert c.max_artists == -1


class TestFeatures:
    def test_default_integrations_list(self):
        from subscriptions.models import Features

        f = Features(
            zoe_enabled=False, oneclick_enabled=False, registry_enabled=False, integrations_allowed=["google_drive"]
        )
        assert "google_drive" in f.integrations_allowed
        assert "slack" not in f.integrations_allowed


class TestEntitlements:
    def test_serializable_to_dict(self):
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="00000000-0000-0000-0000-000000000001",
            tier="free",
            status="active",
            caps=Caps(3, 3, 50, 1073741824, 5, 1),
            features=Features(False, False, False, ["google_drive"]),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
            degraded=False,
        )
        d = ent.to_dict()
        assert d["tier"] == "free"
        assert d["caps"]["maxArtists"] == 3
        assert "maxOneclickRunsPerMonth" in d["caps"]
        assert d["caps"]["maxOneclickRunsPerMonth"] == 1
        assert d["features"]["integrationsAllowed"] == ["google_drive"]
        assert d["usage"]["splitSheetsThisPeriod"] == 0
        assert d["hasOverrides"] is False
        assert d["degraded"] is False
        # Stripe billing sub-object always present; None for free users
        assert "subscription" in d
        assert d["subscription"]["stripeSubscriptionId"] is None
        assert d["subscription"]["cancelAtPeriodEnd"] is False
        assert d["subscription"]["planPeriod"] is None

    def test_to_dict_subscription_stripe_fields(self):
        """to_dict() exposes stripe billing fields when set."""
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="u1",
            tier="pro",
            status="active",
            caps=Caps(-1, -1, -1, -1, -1, -1),
            features=Features(True, True, True, ["google_drive", "slack"]),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
            stripe_subscription_id="sub_abc123",
            stripe_price_id="price_monthly_pro",
            current_period_end=datetime(2026, 6, 9, tzinfo=UTC),
            cancel_at_period_end=True,
        )
        d = ent.to_dict()
        sub = d["subscription"]
        assert sub["stripeSubscriptionId"] == "sub_abc123"
        assert sub["stripePriceId"] == "price_monthly_pro"
        assert sub["currentPeriodEnd"] == "2026-06-09T00:00:00+00:00"
        assert sub["cancelAtPeriodEnd"] is True
        assert sub["planPeriod"] == "monthly"

    def test_to_dict_annual_plan_period(self):
        """plan_period is 'annual' when price_id contains 'annual'."""
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="u2",
            tier="pro",
            status="active",
            caps=Caps(-1, -1, -1, -1, -1, -1),
            features=Features(True, True, True, []),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
            stripe_price_id="price_annual_pro",
        )
        d = ent.to_dict()
        assert d["subscription"]["planPeriod"] == "annual"

    def test_degraded_flag_default_false(self):
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="x",
            tier="free",
            status="active",
            caps=Caps(0, 0, 0, 0, 0, 0),
            features=Features(False, False, False, []),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
        )
        assert ent.degraded is False


class TestAction:
    def test_all_actions_present(self):
        from subscriptions.models import Action

        expected = {
            "create_artist",
            "create_project",
            "create_task",
            "create_work",
            "upload_bytes",
            "generate_split_sheet",
            "use_zoe",
            "use_oneclick",
            "use_registry",
            "use_integration",
        }
        assert {a.value for a in Action} == expected

    def test_docstring_mentions_host_user_id_and_current_count(self):
        """Doc must communicate the host-wins ctx and the count semantics."""
        from subscriptions.models import Action

        doc = Action.__doc__ or ""
        assert "host_user_id" in doc
        assert "current_count" in doc


class TestCheckResult:
    def test_allowed_shape(self):
        from subscriptions.models import CheckResult

        r = CheckResult(allowed=True, reason=None, upgrade_required=False)
        assert r.allowed is True

    def test_denied_shape(self):
        from subscriptions.models import CheckResult

        r = CheckResult(allowed=False, reason="At cap", upgrade_required=True)
        assert r.reason == "At cap"
        assert r.upgrade_required is True


class TestUsageNewFields:
    def test_to_dict_includes_zoe_and_oneclick_counters(self):
        from datetime import UTC, datetime

        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="x",
            tier="free",
            status="active",
            caps=Caps(0, 0, 0, 0, 0, 0),
            features=Features(False, False, False, []),
            usage=Usage(0, 1, 7, 3, datetime.now(UTC)),
            has_overrides=False,
        )
        d = ent.to_dict()
        assert d["usage"]["zoeQueriesThisPeriod"] == 7
        assert d["usage"]["oneclickRunsThisPeriod"] == 3


class TestOverridePayload:
    def test_all_fields_optional(self):
        from subscriptions.models import OverridePayload

        p = OverridePayload()
        assert p.model_dump(exclude_none=True) == {}

    def test_negative_expires_days_rejected(self):
        from pydantic import ValidationError

        from subscriptions.models import OverridePayload

        try:
            OverridePayload(expires_days=-5)
            raise AssertionError("expected ValidationError")
        except ValidationError:
            pass

    def test_unlimited_sentinel_accepted(self):
        from subscriptions.models import OverridePayload

        p = OverridePayload(max_artists=-1)
        assert p.max_artists == -1


class TestCreditModels:
    def test_credit_action_keys_match_seeded_prices(self):
        from subscriptions.models import CreditAction

        assert {a.value for a in CreditAction} == {"zoe_message", "oneclick_run", "registry_parse"}

    def test_create_work_action_exists(self):
        from subscriptions.models import Action

        assert Action.CREATE_WORK == "create_work"

    def test_to_dict_includes_credits_block(self):
        from subscriptions.models import Caps, CreditsInfo, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="u1",
            tier="pro_max",
            status="active",
            caps=Caps(
                max_artists=-1,
                max_projects=-1,
                max_tasks=-1,
                max_storage_bytes=-1,
                max_split_sheets_per_month=-1,
                max_oneclick_runs_per_month=-1,
                monthly_credits=8000,
                max_works=-1,
                included_storage_bytes=268435456000,
            ),
            features=Features(True, True, True, ["google_drive", "slack"]),
            usage=Usage(0, 0, 0, 0, datetime(2026, 8, 1, tzinfo=UTC)),
            has_overrides=False,
            credits=CreditsInfo(
                bundle_balance=7000,
                reserve_balance=100,
                monthly_grant=8000,
                overage_this_period=0,
                overage_enabled=False,
                overage_cap_credits=None,
                storage_overage_enabled=False,
                period_end=datetime(2026, 8, 1, tzinfo=UTC),
                prices={"zoe_message": 3, "oneclick_run": 21, "registry_parse": 12},
            ),
        )
        d = ent.to_dict()
        c = d["credits"]
        assert c["balance"] == 7100  # bundle + reserve
        assert c["bundleBalance"] == 7000
        assert c["monthlyGrant"] == 8000
        assert c["prices"]["oneclickRun"] == 21
        assert d["caps"]["maxWorks"] == -1

    def test_to_dict_credits_none_when_absent(self):
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="u1",
            tier="pro",
            status="active",
            caps=Caps(-1, -1, -1, -1, -1, -1),
            features=Features(True, True, True, []),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
        )
        assert ent.to_dict()["credits"] is None

    def test_credit_check_result_defaults(self):
        from subscriptions.models import CreditCheckResult

        r = CreditCheckResult(allowed=True, price=3)
        assert r.use_overage is False and r.reason is None

    def test_credit_grant_shape(self):
        from subscriptions.models import CreditGrant

        g = CreditGrant(request_id="abc", action="zoe_message", price=3, kind="debit", enabled=True)
        assert g.kind in ("debit", "overage_debit")
