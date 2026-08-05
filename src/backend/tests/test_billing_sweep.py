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
        self._negate_next = False
        self.insert = MagicMock(return_value=self)
        self.update = MagicMock(return_value=self)
        self.upsert = MagicMock(return_value=self)
        self.delete = MagicMock(return_value=self)

    def select(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    @property
    def not_(self):
        # Supports `.not_.is_(col, "null")` — the org-grant reconciliation's
        # "org_id IS NOT NULL" provenance filter.
        self._negate_next = True
        return self

    def is_(self, col, val):
        if val == "null":
            self._preds.append(("notnull" if self._negate_next else "isnull", col, None))
        self._negate_next = False
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
            if op == "notnull" and rv is None:
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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
        # FIX 3: annual is now matched against the STRIPE_PRICE_*_ANNUAL env vars.
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_annual_xyz")
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

    async def test_real_stripe_price_id_matching_env_var_is_annual(self, monkeypatch):
        """FIX 3 (the actual bug): real Stripe price ids are opaque
        (price_1AbC...) and never contain "annual" — the old substring check
        made is_annual always False, so annual overage floated unbilled.
        Matching against the configured env var must catch it."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_1AbCdEfGh0RealId")
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u13",
            "price_1AbCdEfGh0RealId",
            [{"id": "ledger-real-1", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
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

    async def test_price_id_containing_annual_but_not_configured_is_not_annual(self, monkeypatch):
        """Pins the fix direction: the substring heuristic is GONE — a price id
        that happens to contain "annual" but matches no STRIPE_PRICE_*_ANNUAL
        env var is not treated as an annual plan."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("STRIPE_PRICE_ANNUAL", raising=False)
        monkeypatch.delenv("STRIPE_PRICE_PRO_MAX_ANNUAL", raising=False)
        from subscriptions.sweep import billing_sweep

        sb, builders = _annual_setup(
            "u14",
            "price_annual_xyz",
            [{"id": "ledger-sub-1", "kind": "overage_debit", "metadata": {"invoice_item_id": "ii_prev"}}],
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

    async def test_archived_and_suspended_orgs_are_skipped(self, monkeypatch):
        """Archived orgs never receive dispersal; nor do suspended ones (the
        status IN-list is ('active','pending') only)."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        for kwargs in ({"status": "suspended"}, {"archived_at": "2026-01-01T00:00:00+00:00"}):
            sb = self._sb(**kwargs)
            with patch("main.get_supabase_client", return_value=sb):
                result = await billing_sweep(x_sweep_token="s3cret")
            assert result["orgsDispersed"] == 0, kwargs
            assert [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"] == [], kwargs

    async def test_pending_org_is_dispersed_and_activation_checked(self, monkeypatch):
        """FIX 2: the sweep disperses to PENDING orgs too, and — because the
        dispersal counts toward the activation floor — runs the SAME shared
        activation check the pack-purchase webhook uses after a successful
        rollover."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = self._sb(status="pending")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("orgs.wallets.maybe_activate_org") as mock_activate,
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 1
        calls = [c for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"]
        assert len(calls) == 1 and calls[0].args[1]["p_wallet_id"] == "pool-org1"
        mock_activate.assert_called_once_with(sb, "org1", "pool-org1")

    async def test_active_org_dispersal_skips_activation_check(self, monkeypatch):
        """Activation only ever moves pending -> active; an already-active org's
        dispersal must not even attempt the check."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = self._sb(status="active")

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("orgs.wallets.maybe_activate_org") as mock_activate,
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 1
        mock_activate.assert_not_called()

    async def test_pending_org_open_period_no_dispersal_no_activation(self, monkeypatch):
        """A pending org whose pool period is still open gets neither a second
        dispersal nor an activation check this run."""
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        future = (datetime.now(UTC) + timedelta(days=20)).isoformat()
        sb = self._sb(status="pending", period_end=future)

        with (
            patch("main.get_supabase_client", return_value=sb),
            patch("orgs.wallets.maybe_activate_org") as mock_activate,
        ):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orgsDispersed"] == 0
        mock_activate.assert_not_called()

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


# ---------------------------------------------------------------------------
# FIX 5b — org-grant reconciliation: the offboard/archive revocation paths are
# best-effort, so the sweep deletes any org-granted project_members row whose
# backing seat is no longer active (or whose org is archived/gone).
# ---------------------------------------------------------------------------


def _filter_aware_supabase_with_builders(table_data: dict):
    """Like _filter_aware_supabase, but records every builder per table so a
    test can reach the builder a delete ran on (fresh builder per table()
    call, so the plain helper can't expose it)."""
    sb = MagicMock()
    created: dict = {}

    def _mk(name):
        b = _FilterBuilder(list(table_data.get(name, [])))
        created.setdefault(name, []).append(b)
        return b

    sb.table.side_effect = _mk
    sb.rpc.return_value.execute.return_value = MagicMock(data=True)
    return sb, created


class TestSweepGrantReconciliation:
    def _data(self):
        return {
            "project_members": [
                # Backing seat still ACTIVE -> kept.
                {"id": "pm-live", "org_id": "org1", "user_id": "u-active", "project_id": "p1"},
                # Backing seat suspended -> stale.
                {"id": "pm-suspended-seat", "org_id": "org1", "user_id": "u-suspended", "project_id": "p1"},
                # Backing seat removed -> stale.
                {"id": "pm-removed-seat", "org_id": "org1", "user_id": "u-removed", "project_id": "p2"},
                # Backing org archived -> stale even though its seat is active.
                {"id": "pm-archived-org", "org_id": "org2", "user_id": "u-arch", "project_id": "p3"},
                # Organic row (org_id NULL) -> excluded by the provenance filter.
                {"id": "pm-organic", "org_id": None, "user_id": "u-organic", "project_id": "p1"},
            ],
            "organizations": [
                {"id": "org1", "archived_at": None},
                {"id": "org2", "archived_at": "2026-01-01T00:00:00+00:00"},
            ],
            "org_members": [
                {"org_id": "org1", "user_id": "u-active", "status": "active"},
                {"org_id": "org1", "user_id": "u-suspended", "status": "suspended"},
                {"org_id": "org1", "user_id": "u-removed", "status": "removed"},
                {"org_id": "org2", "user_id": "u-arch", "status": "active"},
            ],
        }

    async def test_stale_grants_deleted_live_and_organic_kept(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb, builders = _filter_aware_supabase_with_builders(self._data())

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orphanGrantsRevoked"] == 3
        delete_builders = [b for b in builders["project_members"] if b.delete.called]
        assert len(delete_builders) == 1
        deleted_ids = next(v for op, col, v in delete_builders[0]._preds if op == "in" and col == "id")
        assert sorted(deleted_ids) == ["pm-archived-org", "pm-removed-seat", "pm-suspended-seat"]

    async def test_no_stale_grants_no_delete(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        data = self._data()
        data["project_members"] = [
            {"id": "pm-live", "org_id": "org1", "user_id": "u-active", "project_id": "p1"},
            {"id": "pm-organic", "org_id": None, "user_id": "u-organic", "project_id": "p1"},
        ]
        sb, builders = _filter_aware_supabase_with_builders(data)

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orphanGrantsRevoked"] == 0
        assert not any(b.delete.called for b in builders["project_members"])

    async def test_licensing_off_skips_reconciliation(self, monkeypatch):
        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        from subscriptions.sweep import billing_sweep

        sb, builders = _filter_aware_supabase_with_builders(self._data())

        with patch("main.get_supabase_client", return_value=sb):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["orphanGrantsRevoked"] == 0
        assert "project_members" not in builders


# ---------------------------------------------------------------------------
# FIX 12 — charge-leak DETECTION: ai_usage_log success rows from users with no
# credit debit at all in the window are counted + warned about. Detection only.
# ---------------------------------------------------------------------------


class TestSweepChargeLeakDetection:
    def _data(self):
        recent = _iso_days_ago(0)  # now-ish, inside the 1-day window
        return {
            "ai_usage_log": [
                # u-leak did successful LLM work but has NO debit -> flagged.
                {"id": "log-leak", "user_id": "u-leak", "success": True, "cache_hit": False, "created_at": recent},
                # u-paid has a personal-wallet debit -> not flagged.
                {"id": "log-paid", "user_id": "u-paid", "success": True, "cache_hit": False, "created_at": recent},
                # u-org spent from a pool (via org_member metadata) -> not flagged.
                {"id": "log-org", "user_id": "u-org", "success": True, "cache_hit": False, "created_at": recent},
                # Cache hits are free by design -> excluded from detection.
                {"id": "log-cache", "user_id": "u-cache", "success": True, "cache_hit": True, "created_at": recent},
                # Failures are uncharged by design -> excluded.
                {"id": "log-fail", "user_id": "u-fail", "success": False, "cache_hit": False, "created_at": recent},
                # Old rows are outside the 1-day window.
                {
                    "id": "log-old",
                    "user_id": "u-old",
                    "success": True,
                    "cache_hit": False,
                    "created_at": _iso_days_ago(5),
                },
            ],
            "credit_ledger": [
                {"wallet_id": "w-paid", "kind": "debit", "metadata": {}, "created_at": recent},
                {"wallet_id": "w-pool", "kind": "debit", "metadata": {"org_member_id": "m-org"}, "created_at": recent},
            ],
            "credit_wallets": [
                {"id": "w-paid", "owner_type": "user", "owner_id": "u-paid"},
                {"id": "w-pool", "owner_type": "org", "owner_id": "org1"},
            ],
            "org_members": [
                {"id": "m-org", "user_id": "u-org", "status": "active"},
            ],
        }

    async def test_counts_and_warns_only_undebited_users(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        from subscriptions.sweep import billing_sweep

        sb = _filter_aware_supabase(self._data())

        with patch("main.get_supabase_client", return_value=sb), caplog.at_level(logging.WARNING):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["chargeLeaks"] == 1
        leak_logs = [r for r in caplog.records if "charge-leak detection" in r.getMessage()]
        assert len(leak_logs) == 1
        assert "log-leak" in leak_logs[0].getMessage()

    async def test_no_usage_rows_skips_quietly(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("SWEEP_TOKEN", "s3cret")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.sweep import billing_sweep

        sb = _filter_aware_supabase({})

        with patch("main.get_supabase_client", return_value=sb), caplog.at_level(logging.WARNING):
            result = await billing_sweep(x_sweep_token="s3cret")

        assert result["chargeLeaks"] == 0
        assert not [r for r in caplog.records if "charge-leak detection" in r.getMessage()]
