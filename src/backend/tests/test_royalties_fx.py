"""Unit tests for oneclick.royalties.fx — convert + _boc_cad_rates.

All external dependencies (Supabase client, httpx) are mocked. No real network
calls or database access are made.

Mock strategy:
  - Supabase: inline MockQueryBuilder pattern (same style as test_royalties_ingest.py)
  - httpx: unittest.mock.patch on httpx.Client used as a context manager
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import fx as _fxmod
from oneclick.royalties.fx import _boc_cad_rates, convert

# ---------------------------------------------------------------------------
# Autouse fixture — clear in-process memo before + after every test so the
# cache never leaks state between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_fx_memo():
    _fxmod._clear_rate_memo()
    yield
    _fxmod._clear_rate_memo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# convert — BoC-only routing
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

    def test_usd_to_gbp_via_boc_triangulation(self):
        """convert(db, 100, 'USD', 'GBP') uses BoC triangulation."""
        db = _make_db()
        boc_result = {"usd": 1.37, "gbp": 1.74, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result) as mock_boc:
            result = convert(db, 100.0, "USD", "GBP")

        expected = 100.0 * 1.37 / 1.74
        assert result == pytest.approx(expected)
        mock_boc.assert_called_once()

    def test_eur_to_gbp_via_boc_triangulation(self):
        """convert(db, 100, 'EUR', 'GBP') triangulates via CAD."""
        db = _make_db()
        boc_result = {"eur": 1.45, "gbp": 1.74, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "EUR", "GBP")

        assert result == pytest.approx(100.0 * 1.45 / 1.74)

    def test_usd_to_cad_via_boc(self):
        """USD->CAD via BoC: cad[usd]/cad[cad] = rate/1.0 = rate."""
        db = _make_db()
        rate = 1.36
        boc_result = {"usd": rate, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 200.0, "USD", "CAD")

        assert result == pytest.approx(200.0 * rate)

    def test_cad_to_usd_via_boc(self):
        """CAD->USD via BoC: 1.0 / cad[usd]."""
        db = _make_db()
        boc_result = {"usd": 1.36, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 136.0, "CAD", "USD")

        assert result == pytest.approx(136.0 * 1.0 / 1.36)

    def test_usd_to_gbp_triangulation_math(self):
        """Triangulation math: amount * cad_per_usd / cad_per_gbp."""
        db = _make_db()
        boc_result = {"usd": 1.37, "gbp": 1.74, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "USD", "GBP")

        assert result == pytest.approx(100.0 * 1.37 / 1.74)

    def test_missing_rate_on_missing_none_returns_none(self):
        """convert(db, amount, 'NGN', 'USD', on_missing='none') returns None when BoC lacks NGN."""
        db = _make_db()
        # BoC only has USD, not NGN
        boc_result = {"usd": 1.37, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "NGN", "USD", on_missing="none")

        assert result is None

    def test_missing_rate_default_returns_amount_unconverted(self):
        """convert(db, amount, 'NGN', 'USD') with default on_missing returns amount unchanged."""
        db = _make_db()
        # BoC only has USD, not NGN
        boc_result = {"usd": 1.37, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 42.0, "NGN", "USD")

        assert result == pytest.approx(42.0)

    def test_boc_zero_rate_target_on_missing_none_returns_none(self):
        """If BoC returns a zero rate for target (division guard falsy), on_missing='none' → None."""
        db = _make_db()
        # cad[t] == 0 — the guard `cad[t]` is falsy
        boc_result = {"usd": 1.37, "gbp": 0.0, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "USD", "GBP", on_missing="none")

        assert result is None

    def test_boc_zero_rate_target_default_returns_amount(self):
        """If BoC returns a zero rate for target, default on_missing returns amount unconverted."""
        db = _make_db()
        boc_result = {"usd": 1.37, "gbp": 0.0, "cad": 1.0}

        with patch("oneclick.royalties.fx._boc_cad_rates", return_value=boc_result):
            result = convert(db, 100.0, "USD", "GBP")

        assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Memoisation tests — prove that repeated calls hit network only once
# ---------------------------------------------------------------------------


class TestBocCadRatesMemo:
    def test_second_boc_call_reuses_memo_network_called_once(self):
        """Two _boc_cad_rates calls for the same codes: httpx.Client invoked only once."""
        boc_payload = {
            "observations": [{"d": "2026-06-23", "FXUSDCAD": {"v": "1.37"}, "FXGBPCAD": {"v": "1.74"}}],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx) as mock_httpx_cls:
            r1 = _boc_cad_rates(db, ["USD"])
            r2 = _boc_cad_rates(db, ["USD"])

        assert r1["usd"] == pytest.approx(1.37)
        assert r2["usd"] == pytest.approx(1.37)
        # Network hit only once; second call served from in-process memo
        assert mock_httpx_cls.call_count == 1

    def test_memo_serves_superset_of_codes_without_refetch(self):
        """If memo already has USD + GBP, a second call for just USD reuses memo without fetching."""
        boc_payload = {
            "observations": [{"d": "2026-06-23", "FXUSDCAD": {"v": "1.37"}, "FXGBPCAD": {"v": "1.74"}}],
            "seriesDetail": {},
        }
        cache_builder = _ChainBuilder(execute_result=[])
        db = _make_db({"fx_rate_snapshots": cache_builder})

        resp = _mock_httpx_response(boc_payload)
        ctx, _ = _mock_client_context(resp)

        with patch("httpx.Client", return_value=ctx) as mock_httpx_cls:
            # First call: fetches USD + GBP; populates memo
            r1 = _boc_cad_rates(db, ["USD", "GBP"])
            # Second call: only asks for USD — should be satisfied from memo
            r2 = _boc_cad_rates(db, ["USD"])

        assert r1["usd"] == pytest.approx(1.37)
        assert r1["gbp"] == pytest.approx(1.74)
        assert r2["usd"] == pytest.approx(1.37)
        # Only one network call total
        assert mock_httpx_cls.call_count == 1

    def test_memo_cleared_between_tests_via_autouse(self):
        """Sanity-check: the autouse fixture clears the memo, so this test starts clean."""
        # The memo must be empty at the start of every test
        assert _fxmod._RATE_MEMO == {}
