"""orgs/storage_guard.py — Task 12: upload_allowed + artist_subtree_bytes on
top of Task 5's pool_state/reactivation_allowed. Task 13 adds
bill_team_storage_overage (the sweep step).

pool_state and reactivation_allowed themselves (the archived/lapsed-included,
dissolved-excluded sum; the >10 GiB pro-like threshold) are covered by
tests/test_org_standing.py::TestPoolState / TestReactivationAllowed and are
NOT duplicated here — this file only adds what Task 12/13 are new for.
"""

import logging
from unittest.mock import MagicMock, patch

from orgs import storage_guard
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
ARTIST_ID = "30000000-0000-0000-0000-000000000001"
GB = 2**30


def _pool_db(*, org_rows, tier="basic", tier_row=None, is_admin=False):
    """Mirrors test_org_standing.py's `_storage_db` (Task 5) — same four
    tables pool_state reads, kept local so this file doesn't depend on a
    module owned by a parallel task."""
    db = MagicMock()

    def side_effect(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=org_rows, count=len(org_rows))
        elif name == "profiles":
            b.execute.return_value = MagicMock(data=[{"is_admin": is_admin}], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(data=[{"tier": tier}], count=1)
        elif name == "tier_entitlements":
            row = tier_row or {
                "tier": tier,
                "max_teams": 1,
                "max_team_members": 3,
                "team_storage_bytes": 10 * GB,
            }
            b.execute.return_value = MagicMock(data=[row], count=1)
        return b

    db.table.side_effect = side_effect
    return db


class TestUploadAllowed:
    def test_basic_owner_upload_within_pool_passes(self):
        db = _pool_db(org_rows=[{"storage_bytes": int(9.5 * GB)}], tier="basic")

        allowed, reason = storage_guard.upload_allowed(db, U1, int(0.4 * GB))

        assert allowed is True
        assert reason is None

    def test_basic_owner_upload_over_pool_fails_with_member_copy(self):
        db = _pool_db(org_rows=[{"storage_bytes": int(9.5 * GB)}], tier="basic")

        allowed, reason = storage_guard.upload_allowed(db, U1, int(0.6 * GB))

        assert allowed is False
        assert reason == storage_guard.TEAM_STORAGE_FULL_MSG

    def test_pro_like_owner_upload_passes_even_far_over_pool(self):
        db = _pool_db(
            org_rows=[{"storage_bytes": 150 * GB}],
            tier="pro",
            tier_row={"tier": "pro", "max_teams": 3, "max_team_members": 10, "team_storage_bytes": 100 * GB},
        )

        allowed, reason = storage_guard.upload_allowed(db, U1, GB)

        assert allowed is True
        assert reason is None

    def test_archived_and_lapsed_bytes_block_the_upload_wall(self):
        """pool_state sums archived+lapsed self_serve orgs (Task 5 anti-
        cycling — pinned directly in test_org_standing.py::TestPoolState).
        What's missing there and added here: that the sum actually reaches
        upload_allowed and blocks a real upload, not just that pool_state
        adds correctly in isolation."""
        db = _pool_db(
            org_rows=[{"storage_bytes": 6 * GB}, {"storage_bytes": 5 * GB}],  # e.g. one archived + one lapsed
            tier="basic",
        )

        allowed, reason = storage_guard.upload_allowed(db, U1, 1024)

        assert allowed is False
        assert reason == storage_guard.TEAM_STORAGE_FULL_MSG

    def test_landing_exactly_on_the_pool_boundary_is_allowed(self):
        """used + size == pool is a fit, not an overage — pins the inclusive
        `<=` in upload_allowed's own check."""
        db = _pool_db(org_rows=[{"storage_bytes": 9 * GB}], tier="basic")

        allowed, reason = storage_guard.upload_allowed(db, U1, GB)  # 9 + 1 == 10 (the pool)

        assert allowed is True
        assert reason is None


class TestArtistSubtreeBytes:
    """Mirrors recalc_team_storage's two joins (20260803000003_team_storage.sql)
    scoped to one artist instead of a whole team."""

    def test_sums_project_files_and_audio_files_across_the_artist(self):
        def side_effect(name):
            b = MockQueryBuilder()
            if name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": "p1"}, {"id": "p2"}], count=2)
            elif name == "project_files":
                b.execute.return_value = MagicMock(data=[{"file_size": 3 * GB}, {"file_size": None}], count=2)
            elif name == "audio_folders":
                b.execute.return_value = MagicMock(data=[{"id": "f1"}], count=1)
            elif name == "audio_files":
                b.execute.return_value = MagicMock(data=[{"file_size": 2 * GB}], count=1)
            return b

        db = MagicMock()
        db.table.side_effect = side_effect

        assert storage_guard.artist_subtree_bytes(db, ARTIST_ID) == 5 * GB

    def test_no_projects_or_folders_skips_the_file_queries(self):
        """Empty project/folder id lists must not fire an IN (...) query with
        no ids — the join has nothing to filter on."""

        def side_effect(name):
            b = MockQueryBuilder()
            if name in ("projects", "audio_folders"):
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name in ("project_files", "audio_files"):
                raise AssertionError(f"{name} queried with no ids to filter on")
            return b

        db = MagicMock()
        db.table.side_effect = side_effect

        assert storage_guard.artist_subtree_bytes(db, ARTIST_ID) == 0


def _billing_db(
    *,
    storage_bytes,
    owner_id=U1,
    tier="pro",
    tier_row=None,
    is_admin=False,
    stripe_customer_id="cus_123",
    last_invoiced_period=None,
    wallet_period_start="2026-08-01T00:00:00+00:00",
):
    """Single-owner mock for bill_team_storage_overage / _bill_one_owner.

    The `organizations` row carries both `storage_bytes` (pool_state) and
    `covered_by` (the outer owner scan) so one row serves both call sites.

    Unlike `_pool_db`/`_storage_db` above (conftest's MockQueryBuilder, whose
    chain methods are plain Python methods, not spies), this mirrors
    `_sweep_mock_supabase`'s cached-MagicMock-per-table shape: Task 13
    reads AND updates `subscriptions`, and callers need `.update(...)` to be
    an assertable mock, which MockQueryBuilder's `update` is not.
    """
    builders: dict = {}

    def get_builder(name):
        if name in builders:
            return builders[name]
        b = MagicMock()
        for chain_method in ("select", "eq", "update", "in_", "is_", "limit", "order"):
            getattr(b, chain_method).return_value = b
        if name == "organizations":
            rows = [{"storage_bytes": storage_bytes, "covered_by": owner_id}]
            b.execute.return_value = MagicMock(data=rows, count=len(rows))
        elif name == "profiles":
            b.execute.return_value = MagicMock(data=[{"is_admin": is_admin}], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(
                data=[
                    {
                        "tier": tier,
                        "stripe_customer_id": stripe_customer_id,
                        "last_team_storage_invoiced_period": last_invoiced_period,
                    }
                ],
                count=1,
            )
        elif name == "tier_entitlements":
            row = tier_row or {
                "tier": tier,
                "max_teams": 3,
                "max_team_members": 10,
                "team_storage_bytes": 100 * GB,
            }
            b.execute.return_value = MagicMock(data=[row], count=1)
        elif name == "credit_wallets":
            data = [{"period_start": wallet_period_start}] if wallet_period_start else []
            b.execute.return_value = MagicMock(data=data, count=len(data))
        else:
            b.execute.return_value = MagicMock(data=[], count=0)
        builders[name] = b
        return b

    db = MagicMock()
    db.table.side_effect = get_builder
    return db, builders


class TestBillTeamStorageOverageScan:
    """The owner scan itself — one query, deduped, None-covered_by dropped —
    kept separate from per-owner billing logic (below), which is exercised
    through a single-owner db so pool_state's own filtered read stays
    coherent with the outer scan's unfiltered mock rows."""

    def test_scans_distinct_non_null_owners_once_each(self):
        db = MagicMock()

        def side_effect(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(
                    data=[
                        {"covered_by": U1},
                        {"covered_by": U1},  # second org, same owner — must not double count
                        {"covered_by": U2},
                        {"covered_by": None},  # no covering owner — must be skipped, not crash
                    ],
                    count=4,
                )
            return b

        db.table.side_effect = side_effect

        seen = []

        def fake_bill_one(sb, owner_id):
            seen.append(owner_id)
            return False

        with patch.object(storage_guard, "_bill_one_owner", side_effect=fake_bill_one):
            count = storage_guard.bill_team_storage_overage(db)

        assert sorted(seen) == [U1, U2]
        assert count == 0

    def test_one_owner_failure_does_not_abort_the_scan(self, caplog):
        db = MagicMock()

        def side_effect(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=[{"covered_by": U1}, {"covered_by": U2}], count=2)
            return b

        db.table.side_effect = side_effect

        def fake_bill_one(sb, owner_id):
            if owner_id == U1:
                raise RuntimeError("boom")
            return True

        with (
            patch.object(storage_guard, "_bill_one_owner", side_effect=fake_bill_one),
            caplog.at_level(logging.ERROR),
        ):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 1  # U2 still billed despite U1 blowing up
        assert any("bill_team_storage_overage failed" in r.getMessage() for r in caplog.records)


class TestBillTeamStorageOverageBilling:
    """Per-owner billing/stamping logic, exercised through the public
    bill_team_storage_overage(sb) entrypoint against a single-owner db."""

    def test_over_pool_pro_owner_bills_exact_cents_and_stamps(self):
        # pool = 100 GB (tier_row default), used = 110 GB → 10 GB overage.
        # 10 * $0.025 * 100 = 25.0 cents — a clean value, no rounding ambiguity.
        db, builders = _billing_db(storage_bytes=110 * GB)
        fake_item = MagicMock(id="ii_1")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item

        with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        kwargs = fake_stripe.InvoiceItem.create.call_args.kwargs
        assert kwargs["customer"] == "cus_123"
        assert kwargs["amount"] == 25
        assert kwargs["currency"] == "usd"
        assert kwargs["description"] == "Team storage: 10 GB over your included 100 GB"
        assert kwargs["idempotency_key"] == "teamstorage:00000000-0000-0000-0000-000000000001:2026-08-01T00:00:00+00:00"
        assert "invoice" not in kwargs  # pending item — folds into the next real invoice
        builders["subscriptions"].update.assert_called_once_with(
            {"last_team_storage_invoiced_period": "2026-08-01T00:00:00+00:00"}
        )

    def test_same_period_rerun_does_not_double_invoice(self):
        # Stamp already equals the wallet's current period_start → skip entirely.
        db, builders = _billing_db(
            storage_bytes=110 * GB,
            last_invoiced_period="2026-08-01T00:00:00+00:00",
        )
        fake_stripe = MagicMock()

        with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 0
        fake_stripe.InvoiceItem.create.assert_not_called()
        builders["subscriptions"].update.assert_not_called()

    def test_under_pool_stamps_without_invoicing(self):
        db, builders = _billing_db(storage_bytes=50 * GB)  # well under the 100 GB pool
        fake_stripe = MagicMock()

        with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 0
        fake_stripe.InvoiceItem.create.assert_not_called()
        builders["subscriptions"].update.assert_called_once_with(
            {"last_team_storage_invoiced_period": "2026-08-01T00:00:00+00:00"}
        )

    def test_stripe_failure_does_not_stamp(self):
        db, builders = _billing_db(storage_bytes=110 * GB)
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.side_effect = RuntimeError("stripe down")

        with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 0
        builders["subscriptions"].update.assert_not_called()  # retried next sweep

    def test_admin_without_stripe_customer_stamps_and_warns_never_invoices(self, caplog):
        db, builders = _billing_db(
            storage_bytes=110 * GB,
            is_admin=True,
            stripe_customer_id=None,
        )
        fake_stripe = MagicMock()

        with (
            patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe),
            caplog.at_level(logging.WARNING),
        ):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 0
        fake_stripe.InvoiceItem.create.assert_not_called()
        builders["subscriptions"].update.assert_called_once_with(
            {"last_team_storage_invoiced_period": "2026-08-01T00:00:00+00:00"}
        )
        assert any("no stripe customer" in r.getMessage() for r in caplog.records)

    def test_basic_non_pro_like_owner_is_never_billed(self):
        db, builders = _billing_db(
            storage_bytes=110 * GB,
            tier="basic",
            tier_row={"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * GB},
        )
        fake_stripe = MagicMock()

        with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
            count = storage_guard.bill_team_storage_overage(db)

        assert count == 0
        fake_stripe.InvoiceItem.create.assert_not_called()
        # pool_state itself reads `subscriptions` for the tier lookup, but the
        # billing path (customer/stamp read, stamp write) never fires once
        # is_pro_like is False — no stamp is ever written.
        builders["subscriptions"].update.assert_not_called()
