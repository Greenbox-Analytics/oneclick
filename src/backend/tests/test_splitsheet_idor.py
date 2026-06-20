"""IDOR tests for splitsheet: save_to_artist ownership check."""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# Route: POST /splitsheet/generate
# Model: SplitSheetRequest
#   save_to_artist: bool
#   artist_id: str | None
#   project_id: str | None

OWN_ARTIST_ID = "own-artist-00-0000-0000-0000-000000000001"
VICTIM_ARTIST_ID = "victim-artist-0000-0000-0000-000000000001"
OWN_PROJECT_ID = "own-proj-0000-0000-0000-0000-000000000001"
VICTIM_PROJECT_ID = "victim-proj-0000-0000-0000-000000000001"

_PRO_TIER_ROW = {
    "tier": "pro",
    "max_artists": -1,
    "max_projects": -1,
    "max_boards": -1,
    "max_tasks": -1,
    "max_storage_bytes": -1,
    "max_split_sheets_per_month": -1,
    "max_oneclick_runs_per_month": -1,
    "zoe_enabled": True,
    "oneclick_enabled": True,
    "registry_enabled": True,
    "integrations_allowed": ["google_drive", "slack", "notion"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
_PRO_SUB_ROW = {
    "id": "s-default",
    "user_id": TEST_USER_ID,
    "tier": "pro",
    "status": "active",
    "stripe_customer_id": None,
    "stripe_subscription_id": None,
    "stripe_price_id": None,
    "current_period_start": None,
    "current_period_end": None,
    "cancel_at_period_end": False,
    "canceled_at": None,
    "created_at": "2026-05-01T00:00:00+00:00",
    "updated_at": "2026-05-01T00:00:00+00:00",
}
_DEFAULT_USAGE_ROW = {
    "user_id": TEST_USER_ID,
    "total_storage_bytes": 0,
    "split_sheets_this_period": 0,
    "zoe_queries_this_period": 0,
    "oneclick_runs_this_period": 0,
    "period_start": "2026-05-09T00:00:00+00:00",
    "period_end": "2099-05-09T00:00:00+00:00",
    "updated_at": "2026-05-09T00:00:00+00:00",
}
_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_builder(name):
    b = MockQueryBuilder()
    if name == "subscriptions":
        b.execute.return_value = MagicMock(data=[_PRO_SUB_ROW], count=1)
    elif name == "tier_entitlements":
        b.execute.return_value = MagicMock(data=[_PRO_TIER_ROW], count=1)
    elif name == "tier_overrides":
        b.execute.return_value = MagicMock(data=[], count=0)
    elif name == "usage_counters":
        b.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
    elif name == "profiles":
        b.execute.return_value = MagicMock(data=[], count=0)
    return b


MINIMAL_CONTRIBUTOR = {
    "name": "Alice",
    "role": "Composer",
    "publishing_percentage": 100.0,
}

VALID_PAYLOAD = {
    "work_title": "Test Song",
    "work_type": "single",
    "split_type": "both",
    "date": "2026-01-01",
    "format": "pdf",
    "contributors": [MINIMAL_CONTRIBUTOR],
}


class TestSplitSheetSaveToArtistIDOR:
    def test_save_to_foreign_artist_returns_403(self, client, mock_supabase):
        """POST /splitsheet/generate with save_to_artist=true using a victim's artist_id → 403."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            # verify_user_owns_artist → checks artists table
            if name == "artists":
                # caller does NOT own VICTIM_ARTIST_ID
                b.execute.return_value = MagicMock(data=[])
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        payload = {
            **VALID_PAYLOAD,
            "save_to_artist": True,
            "artist_id": VICTIM_ARTIST_ID,
            "project_id": VICTIM_PROJECT_ID,
        }

        # PDF generation itself still runs; we need to mock the generators to avoid
        # importing heavy PDF deps and to isolate the IDOR check.
        import io

        with patch("splitsheet.router.generate_split_sheet_pdf", return_value=io.BytesIO(b"%PDF-stub")):
            resp = client.post("/splitsheet/generate", json=payload)

        assert resp.status_code == 403

    def test_save_to_foreign_project_returns_403(self, client, mock_supabase):
        """POST /splitsheet/generate with save_to_artist=true using a victim's project_id → 403."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "artists":
                # artist check passes (caller owns the artist)
                b.execute.return_value = MagicMock(data=[{"id": OWN_ARTIST_ID}])
            elif name == "projects":
                # project does NOT belong to caller's artists
                b.execute.return_value = MagicMock(data=[])
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        payload = {
            **VALID_PAYLOAD,
            "save_to_artist": True,
            "artist_id": OWN_ARTIST_ID,
            "project_id": VICTIM_PROJECT_ID,
        }

        import io

        with patch("splitsheet.router.generate_split_sheet_pdf", return_value=io.BytesIO(b"%PDF-stub")):
            resp = client.post("/splitsheet/generate", json=payload)

        assert resp.status_code == 403

    def test_save_to_own_artist_and_project_succeeds(self, client, mock_supabase):
        """POST /splitsheet/generate with save_to_artist=true using caller's own ids → 200."""

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "artists":
                b.execute.return_value = MagicMock(data=[{"id": OWN_ARTIST_ID}])
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": OWN_PROJECT_ID}])
            elif name == "project_files":
                b.execute.return_value = MagicMock(data=[{"id": "pf-001"}])
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        payload = {
            **VALID_PAYLOAD,
            "save_to_artist": True,
            "artist_id": OWN_ARTIST_ID,
            "project_id": OWN_PROJECT_ID,
        }

        import io

        with patch("splitsheet.router.generate_split_sheet_pdf", return_value=io.BytesIO(b"%PDF-stub")):
            resp = client.post("/splitsheet/generate", json=payload)

        # Should succeed (200 with PDF stream) - storage calls are mocked by mock_supabase
        assert resp.status_code == 200

    def test_no_save_to_artist_skips_ownership_check(self, client, mock_supabase):
        """POST /splitsheet/generate with save_to_artist=false should not check ownership."""
        artist_check_called = [False]

        def _router(name):
            if name in _SUBSCRIPTION_TABLES:
                return _sub_builder(name)
            b = MockQueryBuilder()
            if name == "artists":
                artist_check_called[0] = True
            b.execute.return_value = MagicMock(data=[])
            return b

        mock_supabase.table.side_effect = _router

        payload = {
            **VALID_PAYLOAD,
            "save_to_artist": False,
        }

        import io

        with patch("splitsheet.router.generate_split_sheet_pdf", return_value=io.BytesIO(b"%PDF-stub")):
            resp = client.post("/splitsheet/generate", json=payload)

        assert resp.status_code == 200
        # No ownership check should have been made (artist_id is None, no save_to_artist)
        assert not artist_check_called[0]
