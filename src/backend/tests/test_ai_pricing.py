"""Tests for subscriptions/ai_pricing.py — model rates + cost estimation."""

import pytest

from subscriptions.ai_pricing import MODEL_RATES, estimate_cost_usd


class TestEstimateCostUsd:
    def test_uncached_call(self):
        # gpt-5-mini: in $0.25/1M, out $2.00/1M
        cost = estimate_cost_usd("gpt-5-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.25 + 2.00)

    def test_cached_discount_applied(self):
        # gpt-5.2: in $1.75, cached $0.175, out $14.00 per 1M
        cost = estimate_cost_usd("gpt-5.2", input_tokens=10_000, output_tokens=2_000, cached_tokens=4_000)
        expected = (6_000 * 1.75 + 4_000 * 0.175 + 2_000 * 14.00) / 1e6
        assert cost == pytest.approx(expected)

    def test_cached_exceeding_input_is_clamped(self):
        # Defensive: cached_tokens can never exceed input_tokens in the math.
        cost = estimate_cost_usd("gpt-5-mini", input_tokens=100, output_tokens=0, cached_tokens=500)
        expected = (100 * 0.025) / 1e6  # all input billed at cached rate
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("gpt-99-turbo", 100, 100) is None

    def test_embedding_model_present(self):
        assert "text-embedding-3-small" in MODEL_RATES
        cost = estimate_cost_usd("text-embedding-3-small", 1_000_000, 0)
        assert cost == pytest.approx(0.02)
