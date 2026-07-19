"""Task 12: Daily billing sweep — auth gate + rollover/storage/overage/annual behavior.

Covers:
  - POST /internal/billing-sweep auth gate (403/503/200/disabled) via the shared `client` fixture
  - billing_sweep() business logic, called directly against purpose-built mock supabases.

Two mock shapes are used deliberately:
  - `_sweep_mock_supabase` — a cached-builder-per-table mock whose filter methods are
    no-ops (mirrors tests/test_credits_stripe.py). Fine for single-user scenarios where we
    assert on the Stripe/RPC call args rather than on which rows a filter admitted.
  - `_filter_aware_supabase` / `_FilterBuilder` — a builder that ACTUALLY applies
    eq/neq/lt/gt/gte/in_ predicates on execute. Load-bearing for the stale-filter and
    paid-only tests: the no-op mock would let a broken filter pass silently, so those
    filters are pinned with a fake that would fail if the filter were dropped.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


def _sweep_mock_supabase(table_data: dict):
    """Cached-builder-per-table mock; filter methods are no-ops (return the same rows).

    `.rpc(...).execute()` defaults to `MagicMock(data=True)` (rollover succeeded); override
    via the returned sb when a test needs a different rollover outcome. table_data maps
    table name -> rows; tables not present default to an empty list.
    """
    builders: dict = {}

    def get_builder(name):
        if name not in builders:
            b = MagicMock()
            for chain_method in (
                "select",
                "eq",
                "neq",
                "update",
                "upsert",
                "insert",
                "in_",
                "order",
                "limit",
                "is_",
                "gte",
                "gt",
                "lt",
            ):
                getattr(b, chain_method).return_value = b
            b.execute.return_value = MagicMock(data=list(table_data.get(name, [])))
            builders[name] = b
        return builders[name]

    sb = MagicMock()
    sb.table.side_effect = get_builder
    sb.rpc.return_value.execute.return_value = MagicMock(data=True)
    return sb, builders


class _FilterBuilder:
    """Query builder that applies eq/neq/lt/gt/gte/in_ predicates on execute().

    Used only for the load-bearing-filter tests. `is_` is treated as a pass-through
    (JSON-path null filtering isn't what those tests probe). insert/update return self
    so chained `.eq(...).execute()` works; their results are unused by the sweep here.
    A FRESH builder is returned per `sb.table()` call so predicates never leak between
    queries.
    """

    def __init__(self, rows):
        self._rows = rows
        self._preds = []
        self.insert = MagicMock(return_value=self)
        self.update = MagicMock(return_value=self)
        self.upsert = MagicMock(return_value=self)

    def select(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def is_(self, *a, **k):
        return self

    def eq(self, col, val):
        self._preds.append(("eq", col, val))
        return self

    def neq(self, col, val):
        self._preds.append(("neq", col, val))
        return self

    def lt(self, col, val):
        self._preds.append(("lt", col, val))
        return self

    def gt(self, col, val):
        self._preds.append(("gt", col, val))
        return self

    def gte(self, col, val):
        self._preds.append(("gte", col, val))
        return self

    def in_(self, col, vals):
        self._preds.append(("in", col, list(vals)))
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
            if op == "lt" and not (rv is not None and rv < val):
                return False
            if op == "gt" and not (rv is not None and rv > val):
                return False
            if op == "gte" and not (rv is not None and rv >= val):
                return False
        return True

    def execute(self):
        return MagicMock(data=[r for r in self._rows if self._match(r)])


def _filter_aware_supabase(table_data: dict):
    sb = MagicMock()
    sb.table.side_effect = lambda name: _FilterBuilder(list(table_data.get(name, [])))
    sb.rpc.return_value.execute.return_value = MagicMock(data=True)
    return sb


def _fake_stripe(item_id="ii_x", invoice_id="in_x"):
    stripe = MagicMock()
    stripe.InvoiceItem.create.return_value = MagicMock(id=item_id)
    stripe.Invoice.create.return_value = MagicMock(id=invoice_id)
    return stripe


def _iso_days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Auth gate — driven through the shared `client` fixture + default mocks
# ---------------------------------------------------------------------------


class TestSweepAuth:
    def test_403_without_token(self, client, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        resp = client.post("/internal/billing-sweep")
        assert resp.status_code == 403

    def test_503_when_unconfigured(self, client, monkeypatch):
        monkeypatch.delenv("SWEEP_TOKEN", raising=False)
        resp = client.post("/internal/billing-sweep", headers={"X-Sweep-Token": "x"})
        assert resp.status_code == 503

    def test_200_with_token(self, client, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = client.post("/internal/billing-sweep", headers={"X-Sweep-Token": "s3cret"})
        assert resp.status_code == 200
        assert set(resp.json()) >= {"walletsRolled", "storageBilled", "annualInvoiced"}

    def test_disabled_flag_short_circuits(self, client, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        resp = client.post("/internal/billing-sweep", headers={"X-Sweep-Token": "s3cret"})
        assert resp.json().get("disabled") is True


# ---------------------------------------------------------------------------
# Rollover — stale filter is load-bearing (filter-aware mock)
# ---------------------------------------------------------------------------


class TestSweepRollover:
    async def test_stale_filter_honored_only_stale_wallet_rolls(self, monkeypatch):
        """A stale wallet + a fresh wallet -> rollover_wallet fires ONLY for the stale one.

        Uses the filter-aware mock so `.lt("period_end", now)` actually scopes the scan;
        the no-op mock would return both wallets and hide a dropped filter.
        """
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = _filter_aware_supabase(
            {
                "tier_entitlements": [{"tier": "pro", "monthly_credits": 3000, "included_storage_bytes": -1}],
                "subscriptions": [
                    # customer None -> steps 1/3 skip; monthly -> step 4 skips. Only step 2 runs.
                    {
                        "user_id": "u_stale",
                        "tier": "pro",
                        "stripe_customer_id": None,
                        "stripe_price_id": None,
                        "storage_overage_enabled": False,
                    },
                    {
                        "user_id": "u_fresh",
                        "tier": "pro",
                        "stripe_customer_id": None,
                        "stripe_price_id": None,
                        "storage_overage_enabled": False,
                    },
                ],
                "credit_wallets": [
                    {
                        "id": "wallet-stale",
                        "owner_type": "user",
                        "owner_id": "u_stale",
                        "period_end": "2020-01-01T00:00:00+00:00",
                    },
                    {
                        "id": "wallet-fresh",
                        "owner_type": "user",
                        "owner_id": "u_fresh",
                        "period_end": "2099-01-01T00:00:00+00:00",
                    },
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["walletsRolled"] == 1
        rollover_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"]
        assert len(rollover_calls) == 1
        payload = rollover_calls[0].args[1]
        assert payload["p_wallet_id"] == "wallet-stale"
        assert payload["p_monthly_grant"] == 3000  # tier resolved from subs_by_uid, not a re-query


# ---------------------------------------------------------------------------
# Paid-only — the .in_("tier", PAID_TIERS) filter scopes billing (filter-aware)
# ---------------------------------------------------------------------------


class TestSweepPaidOnly:
    async def test_free_tier_sub_never_billed(self, monkeypatch):
        """Over-limit paid user IS storage-billed; over-limit free user is NOT.

        With the filter-aware mock, the free sub is excluded by `.in_("tier", PAID_TIERS)`
        before any billing loop, so only the paid customer's InvoiceItem is created. If the
        filter were dropped, the free customer would be billed too and this test would fail.
        """
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        one_gb = 1_073_741_824
        sb = _filter_aware_supabase(
            {
                "tier_entitlements": [
                    {"tier": "pro", "monthly_credits": 3000, "included_storage_bytes": one_gb},
                    {"tier": "free", "monthly_credits": 0, "included_storage_bytes": one_gb},
                ],
                "subscriptions": [
                    {
                        "user_id": "u_paid",
                        "tier": "pro",
                        "stripe_customer_id": "cus_paid",
                        "stripe_price_id": "price_monthly",
                        "storage_overage_enabled": True,
                    },
                    {
                        "user_id": "u_free",
                        "tier": "free",
                        "stripe_customer_id": "cus_free",
                        "stripe_price_id": None,
                        "storage_overage_enabled": True,
                    },
                ],
                "usage_counters": [
                    {"user_id": "u_paid", "total_storage_bytes": 2 * one_gb},
                    {"user_id": "u_free", "total_storage_bytes": 50 * one_gb},
                ],
                "credit_wallets": [
                    {
                        "id": "wallet-paid",
                        "owner_type": "user",
                        "owner_id": "u_paid",
                        "period_end": "2099-01-01T00:00:00+00:00",
                    }
                ],
                "credit_ledger": [],
            }
        )
        fake_stripe = _fake_stripe(item_id="ii_paid_storage")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["storageBilled"] == 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        assert fake_stripe.InvoiceItem.create.call_args.kwargs["customer"] == "cus_paid"
        billed_customers = [c.kwargs["customer"] for c in fake_stripe.InvoiceItem.create.call_args_list]
        assert "cus_free" not in billed_customers


# ---------------------------------------------------------------------------
# Storage overage — insert-first ordering, once-per-period idempotency
# ---------------------------------------------------------------------------


def _storage_setup(user_id, storage_overage_enabled, total_storage_bytes, included_storage_bytes, ledger_rows=None):
    sub_row = {
        "user_id": user_id,
        "tier": "pro",
        "stripe_customer_id": f"cus_{user_id}",
        "stripe_price_id": "price_monthly",
        "storage_overage_enabled": storage_overage_enabled,
    }
    return _sweep_mock_supabase(
        {
            "tier_entitlements": [
                {"tier": "pro", "monthly_credits": 3000, "included_storage_bytes": included_storage_bytes}
            ],
            "subscriptions": [sub_row],
            "usage_counters": [{"user_id": user_id, "total_storage_bytes": total_storage_bytes}],
            "credit_wallets": [
                {
                    "id": f"wallet-{user_id}",
                    "owner_type": "user",
                    "owner_id": user_id,
                    "period_end": "2099-01-01T00:00:00+00:00",
                }
            ],
            "credit_ledger": ledger_rows or [],
        }
    )


class TestSweepStorage:
    async def test_storage_billed_insert_first_then_backfills_item_id(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _storage_setup(
            "u2",
            storage_overage_enabled=True,
            total_storage_bytes=2_147_483_648,  # 2 GB
            included_storage_bytes=1_073_741_824,  # 1 GB included -> 1 GB over
        )
        fake_stripe = _fake_stripe(item_id="ii_storage_1")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["storageBilled"] == 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        create_kwargs = fake_stripe.InvoiceItem.create.call_args.kwargs
        assert create_kwargs["amount"] == 5  # 1 GB over * $0.05/GB * 100 cents
        assert create_kwargs["customer"] == "cus_u2"

        # Insert-first: the ledger row is written WITHOUT invoice_item_id...
        insert_payload = builders["credit_ledger"].insert.call_args[0][0]
        assert insert_payload["kind"] == "storage_bill"
        assert "invoice_item_id" not in insert_payload["metadata"]
        assert "id" in insert_payload  # client-generated so the backfill can target it
        ledger_id = insert_payload["id"]

        # ...then backfilled with the item id, keyed on the same client-generated id.
        # (update().eq() runs on the shared credit_ledger builder whose eq accumulates
        # every call, so assert the id target was among them rather than the last one.)
        update_payload = builders["credit_ledger"].update.call_args[0][0]
        assert update_payload["metadata"]["invoice_item_id"] == "ii_storage_1"
        builders["credit_ledger"].update.return_value.eq.assert_any_call("id", ledger_id)

    async def test_storage_not_rebilled_when_period_already_billed(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        # Existing storage_bill row for the SAME period_end as the wallet -> re-run no-ops.
        sb, builders = _storage_setup(
            "u2",
            storage_overage_enabled=True,
            total_storage_bytes=2_147_483_648,
            included_storage_bytes=1_073_741_824,
            ledger_rows=[
                {
                    "id": "l-existing",
                    "kind": "storage_bill",
                    "metadata": {"period_end": "2099-01-01T00:00:00+00:00", "invoice_item_id": "ii_prev_storage"},
                }
            ],
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["storageBilled"] == 0
        fake_stripe.InvoiceItem.create.assert_not_called()
        builders["credit_ledger"].insert.assert_not_called()

    async def test_storage_skipped_when_not_opted_in(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _storage_setup(
            "u3",
            storage_overage_enabled=False,
            total_storage_bytes=5_000_000_000,
            included_storage_bytes=1_073_741_824,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["storageBilled"] == 0
        fake_stripe.InvoiceItem.create.assert_not_called()

    async def test_storage_skipped_when_usage_under_included(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _storage_setup(
            "u4",
            storage_overage_enabled=True,
            total_storage_bytes=1_000_000,
            included_storage_bytes=107_374_182_400,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["storageBilled"] == 0
        fake_stripe.InvoiceItem.create.assert_not_called()


# ---------------------------------------------------------------------------
# Overage (step 3) positive path — bill_pending_overage actually bills
# ---------------------------------------------------------------------------


class TestSweepOverage:
    async def test_overage_billed_for_paid_user(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.02")
        from subscriptions.sweep import billing_sweep

        # Monthly plan (step 4 skipped), storage disabled (step 1 skipped): only step 3 runs.
        # Unbilled overage_debit row (no invoice_item_id) -> bill_pending_overage creates one.
        sb, builders = _sweep_mock_supabase(
            {
                "tier_entitlements": [{"tier": "pro", "monthly_credits": 3000, "included_storage_bytes": -1}],
                "subscriptions": [
                    {
                        "user_id": "u10",
                        "tier": "pro",
                        "stripe_customer_id": "cus_u10",
                        "stripe_price_id": "price_monthly",
                        "storage_overage_enabled": False,
                    }
                ],
                "credit_wallets": [
                    {
                        "id": "wallet-u10",
                        "owner_type": "user",
                        "owner_id": "u10",
                        "period_end": "2099-01-01T00:00:00+00:00",
                    }
                ],
                "credit_ledger": [
                    {"id": "l-ov", "metadata": {"credits_billed": 21}, "delta": 0, "action": "oneclick_run"}
                ],
            }
        )
        fake_stripe = _fake_stripe(item_id="ii_overage_1")

        # bill_pending_overage -> bill_overage_row uses overage_billing's OWN stripe ref.
        with (
            patch("main.get_supabase_client", return_value=sb),
            patch(
                "subscriptions.overage_billing.stripe_client_module.get_stripe",
                return_value=fake_stripe,
            ),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["overageBilled"] >= 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        assert fake_stripe.InvoiceItem.create.call_args.kwargs["amount"] == 42  # 21 * $0.02 * 100


# ---------------------------------------------------------------------------
# Annual overage — MONTHLY cadence, decoupled from who rolled the wallet
# ---------------------------------------------------------------------------


def _annual_setup(user_id, price_id, ledger_rows, last_standalone_invoice_at=None, rpc_data=True):
    sub_row = {
        "user_id": user_id,
        "tier": "pro_max",
        "stripe_customer_id": f"cus_{user_id}",
        "stripe_price_id": price_id,
        "storage_overage_enabled": False,
    }
    sb, builders = _sweep_mock_supabase(
        {
            "tier_entitlements": [{"tier": "pro_max", "monthly_credits": 8000, "included_storage_bytes": -1}],
            "subscriptions": [sub_row],
            "credit_wallets": [
                {
                    "id": f"wallet-{user_id}",
                    "owner_type": "user",
                    "owner_id": user_id,
                    "period_end": "2099-01-01T00:00:00+00:00",
                    "last_standalone_invoice_at": last_standalone_invoice_at,
                }
            ],
            "credit_ledger": ledger_rows,
        }
    )
    sb.rpc.return_value.execute.return_value = MagicMock(data=rpc_data)
    return sb, builders


class TestSweepAnnual:
    async def test_a_null_timestamp_fires_stamps_and_records_cadence(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u5",
            "price_annual_xyz",
            [{"id": "ledger-annual-1", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe(invoice_id="in_annual_1")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 1
        fake_stripe.Invoice.create.assert_called_once_with(customer="cus_u5", auto_advance=True)
        assert builders["credit_ledger"].update.call_args[0][0]["metadata"]["swept"] is True
        # cadence timestamp recorded so next-day re-run no-ops
        assert "last_standalone_invoice_at" in builders["credit_wallets"].update.call_args[0][0]

    async def test_b_recent_invoice_within_27d_does_not_fire(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u6",
            "price_annual_xyz",
            [{"id": "ledger-annual-2", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
            last_standalone_invoice_at=_iso_days_ago(5),
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 0
        fake_stripe.Invoice.create.assert_not_called()

    async def test_c_stale_invoice_beyond_27d_fires(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u7",
            "price_annual_xyz",
            [{"id": "ledger-annual-3", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
            last_standalone_invoice_at=_iso_days_ago(30),
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 1
        fake_stripe.Invoice.create.assert_called_once()

    async def test_d_storage_bill_only_still_fires(self, monkeypatch):
        """Critical 2 regression guard: an annual user with ONLY a storage_bill overage
        (no credit overage) must still get a standalone invoice."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u8",
            "price_annual_xyz",
            [{"id": "ledger-storage-1", "kind": "storage_bill", "metadata": {"invoice_item_id": "ii_storage_prev"}}],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 1
        fake_stripe.Invoice.create.assert_called_once()

    async def test_e_lazy_rolled_active_user_still_fires(self, monkeypatch):
        """Critical 1 regression guard (load-bearing): the lazy get_for_user path already
        rolled this active annual user's wallet, so the sweep's rollover_wallet RPC returns
        FALSE (user is NOT in any 'rolled this sweep' set). The standalone invoice must STILL
        fire, gated purely on the cadence timestamp + unswept items."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u9",
            "price_annual_xyz",
            [{"id": "ledger-annual-5", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
            last_standalone_invoice_at=None,
            rpc_data=False,  # rollover_wallet reports already-rolled / period-not-ended
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["walletsRolled"] == 0  # sweep did not roll it — lazy path did
        assert result["annualInvoiced"] == 1  # ...but the invoice still fires
        fake_stripe.Invoice.create.assert_called_once()

    async def test_monthly_plan_no_annual_invoice(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u11",
            "price_monthly_xyz",
            [{"id": "ledger-annual-6", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 0
        fake_stripe.Invoice.create.assert_not_called()

    async def test_no_unswept_items_no_invoice(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u12",
            "price_annual_xyz",
            [
                {
                    "id": "ledger-annual-7",
                    "kind": "overage_debit",
                    "metadata": {"invoice_item_id": "ii_prev", "swept": True},
                }
            ],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.sweep.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 0
        fake_stripe.Invoice.create.assert_not_called()
