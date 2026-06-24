"""Unit tests for oneclick.royalties.fx — get_rates + convert + _boc_cad_rates.

All external dependencies (Supabase client, httpx) are mocked. No real network
calls or database access are made.

Mock strategy:
  - Supabase: inline MockQueryBuilder pattern (same style as test_royalties_ingest.py)
  - httpx: unittest.mock.patch on httpx.Client used as a context manager
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties.fx import _boc_cad_rates, convert, get_rates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RATES_USD = {"gbp": 0.79, "eur": 0.92, "cad": 1.36}
_RESOLVED_DATE = "2024-06-15"
_API_PAYLOAD_USD = {"date": _RESOLVED_DATE, "usd": _RATES_USD}


class _ChainBuilder:
    """Minimal chainable supabase-py mock. execute() is a MagicMock so
    tests can set its return_value to control what the 'query' returns."""

    def __init__(self, execute_result=None):
        self.execute = MagicMock(return_value=MagicMock(data=execute_result if execute_result is not None else []))

    def select(self, *a, **kw):
        return self

    def eq(self, *a, **kw):
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def upsert(self, *a, **kw):
        return self


def _make_db(table_builder_map: dict | None = None):
    """Return a mock db whose .table(name) returns a configured _ChainBuilder.

    table_builder_map: {table_name: _ChainBuilder} — missing tables get an
    empty-result builder.
    """
    db = MagicMock()
    table_builder_map = table_builder_map or {}

    def _side_effect(name):
        return table_builder_map.get(name, _ChainBuilder())

    db.table.side_effect = _side_effect
    return db


def _mock_httpx_response(payload: dict, status_code: int = 200):
    """Return a mock that simulates an httpx Response with .json() and .raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_client_context(response):
    """Return a mock that acts as `httpx.Client(...)` used as a context manager,
    where `client.get(...)` returns *response*.
    """
    mock_client = MagicMock()
    mock_client.get.return_value = response
    # __enter__ returns the client itself; __exit__ is a no-op
    context_manager = MagicMock()
    context_manager.__enter__ = MagicMock(return_value=mock_client)
    context_manager.__exit__ = MagicMock(return_value=False)
    return context_manager, mock_client


# ---------------------------------------------------------------------------
# get_rates — cache hit (dated on)
# ---------------------------------------------------------------------------


class TestGetRatesCacheHit:
    def test_dated_on_cache_hit_returns_cached_rates_no_http(self):
        """When on != 'latest' and a cache row exists, return cached rates with
        no HTTP call made."""
        cached_rates = {"gbp": 0.80, "eur": 0.93}
        builder = _ChainBuilder(execute_result=[{"rate_date": "2024-01-01", "rates": cached_rates}])
        db = _make_db({"fx_rate_snapshots": builder})

        with patch("httpx.Client") as mock_httpx_client:
            result = get_rates(db, "USD", on="2024-01-01")

        assert result == cached_rates
        mock_httpx_client.assert_not_called()


# ---------------------------------------------------------------------------
# get_rates — cache miss → primary fetch
# ---------------------------------------------------------------------------


class TestGetRatesPrimaryFetch:
    def test_cache_miss_fetches_primary_upserts_and_returns_rates(self):
        """On a cache miss (empty db result), the primary CDN is fetched,
        result is upserted with on_conflict='base,rate_date', and rates returned."""
        # Cache miss — empty result
        cache_builder = _ChainBuilder(execute_result=[])
        # Track upsert calls
        upsert_mock = MagicMock(return_value=cache_builder)
        cache_builder.upsert = upsert_mock

        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(_API_PAYLOAD_USD)
        ctx, mock_client = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = get_rates(db, "USD", on="2024-06-15")

        # Rates returned correctly
        assert result == _RATES_USD

        # Upsert called with correct payload and on_conflict
        upsert_mock.assert_called_once_with(
            {
                "base": "usd",
                "rate_date": _RESOLVED_DATE,
                "rates": _RATES_USD,
            },
            on_conflict="base,rate_date",
        )

    def test_resolved_date_comes_from_response_json_not_on_param(self):
        """The stored rate_date should be json['date'], not the 'on' parameter."""
        cache_builder = _ChainBuilder(execute_result=[])
        upsert_mock = MagicMock(return_value=cache_builder)
        cache_builder.upsert = upsert_mock

        db = _make_db({"fx_rate_snapshots": cache_builder})

        # The API resolves "latest" → a specific date in the response
        payload = {"date": "2024-06-20", "usd": {"gbp": 0.79}}
        resp = _mock_httpx_response(payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = get_rates(db, "usd", on="latest")

        assert result == {"gbp": 0.79}
        # resolved date from json, not "latest"
        call_kwargs = upsert_mock.call_args
        assert call_kwargs[0][0]["rate_date"] == "2024-06-20"

    def test_base_normalised_to_lowercase(self):
        """Uppercase base currency is lowercased before the API call and upsert."""
        cache_builder = _ChainBuilder(execute_result=[])
        upsert_mock = MagicMock(return_value=cache_builder)
        cache_builder.upsert = upsert_mock

        db = _make_db({"fx_rate_snapshots": cache_builder})

        payload = {"date": "2024-06-20", "usd": {"eur": 0.92}}
        resp = _mock_httpx_response(payload)
        ctx, mock_client = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = get_rates(db, "USD", on="latest")

        assert result == {"eur": 0.92}
        # URL should contain lowercase base
        url_called = mock_client.get.call_args[0][0]
        assert "usd" in url_called
        assert "USD" not in url_called


# ---------------------------------------------------------------------------
# get_rates — primary fails → fallback
# ---------------------------------------------------------------------------


class TestGetRatesFallback:
    def test_primary_raises_fallback_url_used(self):
        """When the primary CDN raises an exception, the fallback URL is tried."""
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        # Primary raises a connection error
        primary_ctx = MagicMock()
        primary_ctx.__enter__ = MagicMock(side_effect=Exception("connection refused"))
        primary_ctx.__exit__ = MagicMock(return_value=False)

        # Fallback succeeds
        fallback_payload = {"date": "2024-06-15", "usd": {"gbp": 0.79}}
        fallback_resp = _mock_httpx_response(fallback_payload)
        fallback_ctx, fallback_client = _mock_client_context(fallback_resp)

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return primary_ctx
            return fallback_ctx

        with patch("httpx.Client", side_effect=_side_effect):
            result = get_rates(db, "USD", on="latest")

        assert result == {"gbp": 0.79}
        # Fallback URL must contain "currency-api.pages.dev"
        fallback_url_called = fallback_client.get.call_args[0][0]
        assert "currency-api.pages.dev" in fallback_url_called


# ---------------------------------------------------------------------------
# get_rates — both raise → stale cache
# ---------------------------------------------------------------------------


class TestGetRatesBothFail:
    def test_both_raise_returns_newest_cached_snapshot(self):
        """When both CDN URLs fail, the newest cached snapshot is returned."""
        stale_rates = {"gbp": 0.78, "eur": 0.91}
        stale_builder = _ChainBuilder(execute_result=[{"rate_date": "2024-05-01", "rates": stale_rates}])
        db = _make_db({"fx_rate_snapshots": stale_builder})

        # Both HTTP calls raise
        def _raise_ctx(*args, **kwargs):
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(side_effect=Exception("timeout"))
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch("httpx.Client", side_effect=_raise_ctx):
            result = get_rates(db, "USD", on="latest")

        assert result == stale_rates

    def test_both_raise_no_cache_returns_empty_dict(self):
        """When both URLs fail and there is no cached snapshot, returns {}."""
        empty_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": empty_builder})

        def _raise_ctx(*args, **kwargs):
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(side_effect=Exception("timeout"))
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch("httpx.Client", side_effect=_raise_ctx):
            result = get_rates(db, "USD", on="latest")

        assert result == {}

    def test_both_raise_stale_query_uses_desc_order_limit_1(self):
        """The stale-fallback query orders by rate_date desc, limit 1."""
        stale_rates = {"gbp": 0.77}
        stale_builder = _ChainBuilder(execute_result=[{"rate_date": "2024-04-01", "rates": stale_rates}])

        # Spy on order/limit calls
        order_calls = []
        limit_calls = []

        def _spy_order(*a, **kw):
            order_calls.append((a, kw))
            return stale_builder

        def _spy_limit(*a, **kw):
            limit_calls.append(a)
            return stale_builder

        stale_builder.order = _spy_order
        stale_builder.limit = _spy_limit

        db = _make_db({"fx_rate_snapshots": stale_builder})

        def _raise_ctx(*args, **kwargs):
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(side_effect=Exception("timeout"))
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch("httpx.Client", side_effect=_raise_ctx):
            result = get_rates(db, "USD", on="latest")

        assert result == stale_rates
        # Verify ordering
        assert any("rate_date" in str(c) for c in order_calls)
        assert any(kw.get("desc") is True for _, kw in order_calls)
        assert limit_calls and limit_calls[-1][0] == 1


# ---------------------------------------------------------------------------
# _boc_cad_rates — unit tests
# ---------------------------------------------------------------------------


class TestBocCadRates:
    def test_happy_path_returns_cad_per_unit(self):
        """Mock httpx to return a Valet-shaped payload; verify correct CAD-per-unit dict."""
        boc_payload = {
            "observations": [{"d": "2026-06-23", "FXUSDCAD": {"v": "1.37"}}],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = _boc_cad_rates(db, ["USD"])

        assert result["cad"] == 1.0
        assert result["usd"] == pytest.approx(1.37)

    def test_cad_always_1_0(self):
        """CAD is always returned as 1.0 regardless of fetched data."""
        boc_payload = {
            "observations": [{"d": "2026-06-23", "FXGBPCAD": {"v": "1.74"}}],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = _boc_cad_rates(db, ["GBP", "CAD"])

        assert result["cad"] == 1.0
        assert result["gbp"] == pytest.approx(1.74)

    def test_uncovered_code_omitted(self):
        """A currency code BoC doesn't return (no series entry) is omitted from result."""
        # BoC returns nothing for NGN (no series)
        boc_payload = {
            "observations": [{"d": "2026-06-23", "FXUSDCAD": {"v": "1.37"}}],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = _boc_cad_rates(db, ["USD", "NGN"])

        assert "usd" in result
        assert "ngn" not in result

    def test_network_failure_falls_back_to_stale_cache(self):
        """On network failure, returns stale cached values without raising."""
        stale = {"usd": 1.35, "gbp": 1.70}
        cache_builder = _ChainBuilder(execute_result=[{"rate_date": "2026-06-20", "rates": stale}])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        def _raise_ctx(*args, **kwargs):
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(side_effect=Exception("timeout"))
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch("httpx.Client", side_effect=_raise_ctx):
            result = _boc_cad_rates(db, ["USD", "GBP"])

        assert result["usd"] == pytest.approx(1.35)
        assert result["gbp"] == pytest.approx(1.70)
        assert result["cad"] == 1.0

    def test_multiple_observations_picks_latest_non_null(self):
        """When multiple observations exist, iterates in reverse to pick latest non-null v."""
        boc_payload = {
            "observations": [
                {"d": "2026-06-21", "FXUSDCAD": {"v": "1.35"}},
                {"d": "2026-06-22", "FXUSDCAD": {"v": "1.36"}},
                {"d": "2026-06-23", "FXUSDCAD": {"v": "1.37"}},
            ],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx):
            result = _boc_cad_rates(db, ["USD"])

        # Latest non-null is 1.37
        assert result["usd"] == pytest.approx(1.37)

    def test_cached_codes_not_refetched(self):
        """Codes already in cache are not refetched from BoC."""
        cached = {"usd": 1.37}
        cache_builder = _ChainBuilder(execute_result=[{"rate_date": "2026-06-23", "rates": cached}])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        with patch("httpx.Client") as mock_httpx_client:
            result = _boc_cad_rates(db, ["USD"])

        # No HTTP call because USD was already cached
        mock_httpx_client.assert_not_called()
        assert result["usd"] == pytest.approx(1.37)
        assert result["cad"] == 1.0


# ---------------------------------------------------------------------------
# convert — hybrid routing
# ---------------------------------------------------------------------------


class TestConvert:
    def test_same_currency_returns_amount_unchanged(self):
        """Same-currency conversion short-circuits, no HTTP call made."""
        db = _make_db()
        with patch("httpx.Client") as mock_httpx_client:
            result = convert(db, 100.0, "USD", "USD")

        assert result == 100.0
        mock_httpx_client.assert_not_called()

    def test_same_currency_case_insensitive(self):
        """Same-currency check is case-insensitive."""
        db = _make_db()
        with patch("httpx.Client") as mock_httpx_client:
            result = convert(db, 50.0, "usd", "USD")

        assert result == 50.0
        mock_httpx_client.assert_not_called()

    def test_boc_path_usd_to_gbp_no_fawazahmed0(self):
        """convert(db, 100, 'USD', 'GBP') uses BoC triangulation; get_rates NOT called."""
        db = _make_db()
        boc_result = {"usd": 1.37, "gbp": 1.74, "cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result) as mock_boc,
            patch("oneclick.royalties.fx.get_rates") as mock_fawaz,
        ):
            result = convert(db, 100.0, "USD", "GBP")

        expected = 100.0 * 1.37 / 1.74
        assert result == pytest.approx(expected)
        mock_boc.assert_called_once()
        mock_fawaz.assert_not_called()

    def test_fallback_for_uncovered_ngn(self):
        """convert(db, 100, 'USD', 'NGN'): BoC lacks NGN -> falls through to fawazahmed0."""
        db = _make_db()
        # BoC only returns USD (no NGN)
        boc_result = {"usd": 1.37, "cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result),
            patch("oneclick.royalties.fx.get_rates", return_value={"ngn": 1500.0}) as mock_fawaz,
        ):
            result = convert(db, 100.0, "USD", "NGN")

        assert result == pytest.approx(100.0 * 1500.0)
        mock_fawaz.assert_called_once()

    def test_all_unavailable_returns_unconverted(self):
        """When BoC lacks both AND fawazahmed0 returns empty, amount returned unconverted (no raise)."""
        db = _make_db()
        # BoC returns only cad (neither usd nor xyz)
        boc_result = {"cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result),
            patch("oneclick.royalties.fx.get_rates", return_value={}),
        ):
            result = convert(db, 42.0, "USD", "XYZ")

        assert result == pytest.approx(42.0)

    def test_convert_mid_market_no_spread_usd_to_cad_via_boc(self):
        """USD->CAD via BoC: cad[usd]/cad[cad] = rate/1.0 = rate."""
        db = _make_db()
        rate = 1.36
        boc_result = {"usd": rate, "cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result),
            patch("oneclick.royalties.fx.get_rates") as mock_fawaz,
        ):
            result = convert(db, 200.0, "USD", "CAD")

        assert result == pytest.approx(200.0 * rate)
        mock_fawaz.assert_not_called()

    def test_usd_to_gbp_via_boc_triangulation_math(self):
        """Triangulation math: amount * cad_per_usd / cad_per_gbp."""
        db = _make_db()
        boc_result = {"usd": 1.37, "gbp": 1.74, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "USD", "GBP")

        assert result == pytest.approx(100.0 * 1.37 / 1.74)

    def test_missing_target_currency_returns_amount_unconverted(self):
        """When neither BoC nor fawazahmed0 has JPY, convert returns unconverted amount."""
        db = _make_db()
        # BoC has USD but not JPY
        boc_result = {"usd": 1.37, "cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result),
            patch("oneclick.royalties.fx.get_rates", return_value={"eur": 0.92}),
        ):
            result = convert(db, 100.0, "USD", "JPY")

        assert result == pytest.approx(100.0)

    def test_boc_zero_rate_falls_through_to_fawazahmed0(self):
        """If BoC returns a zero rate for target (division guard), falls through to fawazahmed0."""
        db = _make_db()
        # cad[t] == 0 — the guard `cad[t]` is falsy, so we skip BoC path
        boc_result = {"usd": 1.37, "gbp": 0.0, "cad": 1.0}

        with (
            patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result),
            patch("oneclick.royalties.fx.get_rates", return_value={"gbp": 0.79}) as mock_fawaz,
        ):
            result = convert(db, 100.0, "USD", "GBP")

        # Falls back to fawazahmed0 rate
        assert result == pytest.approx(100.0 * 0.79)
        mock_fawaz.assert_called_once()
