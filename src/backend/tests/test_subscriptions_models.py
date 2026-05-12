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
            caps=Caps(3, 3, 50, 1073741824, 5),
            features=Features(False, False, False, ["google_drive"]),
            usage=Usage(0, 0, 0, 0, datetime.now(UTC)),
            has_overrides=False,
            degraded=False,
        )
        d = ent.to_dict()
        assert d["tier"] == "free"
        assert d["caps"]["maxArtists"] == 3
        assert d["features"]["integrationsAllowed"] == ["google_drive"]
        assert d["usage"]["splitSheetsThisPeriod"] == 0
        assert d["hasOverrides"] is False
        assert d["degraded"] is False

    def test_degraded_flag_default_false(self):
        from subscriptions.models import Caps, Entitlements, Features, Usage

        ent = Entitlements(
            user_id="x",
            tier="free",
            status="active",
            caps=Caps(0, 0, 0, 0, 0),
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
            caps=Caps(0, 0, 0, 0, 0),
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
