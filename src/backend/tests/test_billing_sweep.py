"""Task 12: Daily billing sweep — auth gate + rollover/overage/annual behavior.

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
    """Query builder that applies eq/neq/lt/gt/gte/in_/is_ predicates on execute().

    Used only for the load-bearing-filter tests. `is_(col, "null")` applies a real
    IS NULL filter (needed by the licensing allowance-sweep org scan); any other
    value passed to `is_` is a pass-through no-op. insert/update return self so
    chained `.eq(...).execute()` works; their results are unused by the sweep here.
    A FRESH builder is returned per `sb.table()` call so predicates never leak
    between queries.
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

    def is_(self, col, val):
        if val == "null":
            self._preds.append(("isnull", col, None))
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
            if op == "isnull" and rv is not None:
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
        assert set(resp.json()) >= {"walletsRolled", "overageBilled", "annualInvoiced"}

    def test_disabled_flag_short_circuits(self, client, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        resp = client.post("/internal/billing-sweep", headers={"X-Sweep-Token": "s3cret"})
        body = resp.json()
        assert body.get("disabled") is True
        # Regression (Task 10): the credits-disabled early-return is BYTE-IDENTICAL —
        # the licensing allowance/grandfather keys must never appear here, whether or
        # not LICENSING_ENABLED is set.
        assert set(body.keys()) == {"walletsRolled", "overageBilled", "annualInvoiced", "disabled"}


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
                    },
                    {
                        "user_id": "u_fresh",
                        "tier": "pro",
                        "stripe_customer_id": None,
                        "stripe_price_id": None,
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
# Rollover — per-user tier_overrides.monthly_credits wins over tier default
# ---------------------------------------------------------------------------


class TestSweepOverrideGrants:
    def _setup(self, override_rows):
        return _sweep_mock_supabase(
            {
                "tier_entitlements": [{"tier": "free", "monthly_credits": 50, "included_storage_bytes": -1}],
                "subscriptions": [],  # free user — not in the paid set
                "tier_overrides": override_rows,
                "credit_wallets": [
                    {
                        "id": "w-ovr",
                        "owner_type": "user",
                        "owner_id": "u-tester",
                        "period_end": _iso_days_ago(3),
                    }
                ],
            }
        )

    async def test_override_monthly_credits_wins_over_tier_default(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, _ = self._setup([{"user_id": "u-tester", "monthly_credits": 5000, "reason": "tester", "expires_at": None}])
        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")
        assert result["walletsRolled"] == 1
        assert sb.rpc.call_args[0][1]["p_monthly_grant"] == 5000

    async def test_expired_override_falls_back_to_tier_default(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, _ = self._setup(
            [{"user_id": "u-tester", "monthly_credits": 5000, "reason": "tester", "expires_at": _iso_days_ago(1)}]
        )
        with patch("main.get_supabase_client", return_value=sb):
            await billing_sweep(x_sweep_token="s3cret")
        assert sb.rpc.call_args[0][1]["p_monthly_grant"] == 50

    async def test_tester_revoked_marker_ignored(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, _ = self._setup(
            [{"user_id": "u-tester", "monthly_credits": 5000, "reason": "tester_revoked", "expires_at": None}]
        )
        with patch("main.get_supabase_client", return_value=sb):
            await billing_sweep(x_sweep_token="s3cret")
        assert sb.rpc.call_args[0][1]["p_monthly_grant"] == 50


# ---------------------------------------------------------------------------
# Paid-only — the .in_("tier", PAID_TIERS) filter scopes billing (filter-aware)
# ---------------------------------------------------------------------------


class TestSweepPaidOnly:
    async def test_free_tier_sub_never_billed(self, monkeypatch):
        """A paid user with unbilled overage IS billed; a free user with the same
        ledger row is NOT.

        With the filter-aware mock, the free sub is excluded by `.in_("tier", PAID_TIERS)`
        before any billing loop, so only the paid customer's InvoiceItem is created. If the
        filter were dropped, the free customer would be billed too and this test would fail.
        """
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.02")
        from subscriptions.sweep import billing_sweep

        sb = _filter_aware_supabase(
            {
                "tier_entitlements": [
                    {"tier": "pro", "monthly_credits": 3000},
                    {"tier": "free", "monthly_credits": 0},
                ],
                "subscriptions": [
                    {
                        "user_id": "u_paid",
                        "tier": "pro",
                        "stripe_customer_id": "cus_paid",
                        "stripe_price_id": "price_monthly",
                    },
                    {
                        "user_id": "u_free",
                        "tier": "free",
                        "stripe_customer_id": "cus_free",
                        "stripe_price_id": None,
                    },
                ],
                "credit_wallets": [
                    {
                        "id": "wallet-paid",
                        "owner_type": "user",
                        "owner_id": "u_paid",
                        "period_end": "2099-01-01T00:00:00+00:00",
                    },
                    {
                        "id": "wallet-free",
                        "owner_type": "user",
                        "owner_id": "u_free",
                        "period_end": "2099-01-01T00:00:00+00:00",
                    },
                ],
                "credit_ledger": [
                    {
                        "id": "l-paid",
                        "wallet_id": "wallet-paid",
                        "kind": "overage_debit",
                        "delta": 0,
                        "action": "oneclick_run",
                        "metadata": {"credits_billed": 21},
                    },
                    {
                        "id": "l-free",
                        "wallet_id": "wallet-free",
                        "kind": "overage_debit",
                        "delta": 0,
                        "action": "oneclick_run",
                        "metadata": {"credits_billed": 21},
                    },
                ],
            }
        )
        fake_stripe = _fake_stripe(item_id="ii_paid_overage")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["overageBilled"] == 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        assert fake_stripe.InvoiceItem.create.call_args.kwargs["customer"] == "cus_paid"
        billed_customers = [c.kwargs["customer"] for c in fake_stripe.InvoiceItem.create.call_args_list]
        assert "cus_free" not in billed_customers


# ---------------------------------------------------------------------------
# Annual overage — MONTHLY cadence, decoupled from who rolled the wallet
# ---------------------------------------------------------------------------


def _annual_setup(user_id, price_id, ledger_rows, last_standalone_invoice_at=None, rpc_data=True):
    sub_row = {
        "user_id": user_id,
        "tier": "pro_max",
        "stripe_customer_id": f"cus_{user_id}",
        "stripe_price_id": price_id,
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 1
        fake_stripe.Invoice.create.assert_called_once()
        kwargs = fake_stripe.Invoice.create.call_args.kwargs
        assert kwargs["customer"] == "cus_u5"
        assert kwargs["auto_advance"] is True
        # Prefix-only check: re-deriving today's date here would flake if the
        # test straddles UTC midnight between the sweep call and the assert.
        assert kwargs["idempotency_key"].startswith("annual:wallet-u5:")
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 1
        fake_stripe.Invoice.create.assert_called_once()

    async def test_d_single_floating_item_still_fires(self, monkeypatch):
        """Critical 2 regression guard: an annual user with a single already-priced
        floating item must still get a standalone invoice."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u8",
            "price_annual_xyz",
            [{"id": "ledger-ov-1", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_overage_prev"}}],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe()

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
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
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 0
        fake_stripe.Invoice.create.assert_not_called()

    async def test_consumed_items_stamp_swept_without_invoice(self, monkeypatch):
        """Items already attached to a renewal invoice (via invoice.created):
        Stripe rejects the empty standalone invoice — rows must be stamped
        swept and the cadence recorded so the sweep doesn't retry daily."""
        import stripe as stripe_lib

        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u9",
            "price_annual_xyz",
            [{"id": "ledger-consumed-1", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_attached"}}],
            last_standalone_invoice_at=None,
        )
        fake_stripe = _fake_stripe()
        fake_stripe.Invoice.create.side_effect = stripe_lib.InvalidRequestError(
            "Nothing to invoice for customer", None, code="invoice_no_customer_line_items"
        )

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe),
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["annualInvoiced"] == 0  # nothing actually invoiced
        assert builders["credit_ledger"].update.call_args[0][0]["metadata"]["swept"] is True
        assert "last_standalone_invoice_at" in builders["credit_wallets"].update.call_args[0][0]


# ---------------------------------------------------------------------------
# Licensing Phase B — the monthly credit dispersal. Filter-aware mock: the
# organizations predicates (status, archived_at IS NULL,
# monthly_dispersal_credits > 0) are load-bearing, so a no-op filter mock would
# hide a broken query.
# ---------------------------------------------------------------------------


class TestSweepDispersal:
    def _sb(self, *, dispersal=10000, period_end=None, status="active", archived_at=None):
        return _filter_aware_supabase(
            {
                "organizations": [
                    {
                        "id": "org1",
                        "status": status,
                        "archived_at": archived_at,
                        "monthly_dispersal_credits": dispersal,
                    }
                ],
                "credit_wallets": [
                    {"id": "pool-org1", "owner_type": "org", "owner_id": "org1", "period_end": period_end}
                ],
            }
        )

    async def test_first_dispersal_anchors_period_to_the_month(self, monkeypatch):
        """A pool that has never been on a cycle (period_end NULL) gets its
        period anchored to the start of NEXT month, not now()+1mo — member cap
        counters reset against this boundary, so every org has to share the same
        calendar-month alignment or "this month's spend" means nothing."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = self._sb(period_end=None)

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 1
        calls = [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"]
        assert len(calls) == 1
        payload = calls[0].args[1]
        assert payload["p_wallet_id"] == "pool-org1"
        assert payload["p_monthly_grant"] == 10000
        end = datetime.fromisoformat(payload["p_new_period_end"])
        assert (end.day, end.hour, end.minute, end.second) == (1, 0, 0, 0)

    async def test_open_period_is_not_dispersed_again(self, monkeypatch):
        """Idempotency: the pool's period is still open, so nothing is due. This
        is why the sweep can run daily without topping an org up 30 times."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        future = (datetime.now(UTC) + timedelta(days=20)).isoformat()
        sb = self._sb(period_end=future)

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 0
        assert [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"] == []

    async def test_lapsed_period_steps_forward_from_the_stored_end(self, monkeypatch):
        """The new period steps from the STORED period_end, not from now(), so a
        sweep that runs late doesn't silently shorten or shift the cycle."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        stored_end = datetime(2020, 3, 1, tzinfo=UTC)
        sb = self._sb(period_end=stored_end.isoformat())

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 1
        payload = [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"][0].args[1]
        end = datetime.fromisoformat(payload["p_new_period_end"])
        assert end > datetime.now(UTC)
        assert end.day == stored_end.day  # cycle day preserved

    async def test_no_contract_org_is_skipped(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = self._sb(dispersal=0)

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 0
        assert [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"] == []

    async def test_pending_and_archived_orgs_are_skipped(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        for kwargs in ({"status": "pending"}, {"archived_at": "2026-01-01T00:00:00+00:00"}):
            sb = self._sb(**kwargs)
            with patch("main.get_supabase_client", return_value=sb):
                result = await billing_sweep(x_sweep_token="s3cret")
            assert result["orgsDispersed"] == 0, kwargs
            assert [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"] == [], kwargs

    async def test_missing_pool_wallet_skips_the_org(self, monkeypatch):
        """Wallet creation is the app layer's job (lazy on the first org-context
        read or purchase), not the sweep's."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = _filter_aware_supabase(
            {
                "organizations": [
                    {"id": "org1", "status": "active", "archived_at": None, "monthly_dispersal_credits": 10000}
                ],
                "credit_wallets": [],
            }
        )

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 0
        assert [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"] == []

    async def test_licensing_off_never_queries_organizations(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        from subscriptions.sweep import billing_sweep

        sb, builders = _sweep_mock_supabase({})

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 0
        assert [c for c in sb.table.call_args_list if c.args[0] == "organizations"] == []
