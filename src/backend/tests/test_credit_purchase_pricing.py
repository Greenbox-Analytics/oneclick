"""Custom credit purchase: pricing math + Stripe line-item construction.

The client sends a credit COUNT and the server derives the amount, so these
functions are the only thing standing between a tampered request and a wrong
charge. Pure functions — no Stripe, no DB.
"""

import pytest

from subscriptions.credit_purchase import (
    MAX_CUSTOM_CREDITS,
    MIN_CUSTOM_CREDITS,
    credits_line_item,
    custom_config,
    per_credit_cents,
    price_cents_for_credits,
    validate_custom_credits,
)


class TestPriceCentsForCredits:
    def test_default_rate_is_two_cents_per_credit(self, monkeypatch):
        monkeypatch.delenv("CREDIT_OVERAGE_USD", raising=False)
        assert per_credit_cents() == 2
        assert price_cents_for_credits(MIN_CUSTOM_CREDITS) == 500  # $5.00
        assert price_cents_for_credits(1_000) == 2_000  # $20.00
        assert price_cents_for_credits(MAX_CUSTOM_CREDITS) == 200_000  # $2,000.00

    def test_matches_the_overage_rate_dial(self, monkeypatch):
        """Buying a credit and burning one on overage must cost the same, or
        the cheaper path becomes an arbitrage."""
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.05")
        assert price_cents_for_credits(1_000) == 5_000

    def test_fractional_rate_rounds_once_on_the_total(self, monkeypatch):
        """2.5c/credit must not round to 2c/credit and undercharge by 20%."""
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.025")
        assert per_credit_cents() == 2.5
        assert price_cents_for_credits(1_000) == 2_500

    def test_float_dust_does_not_shift_a_cent(self, monkeypatch):
        # 0.02 * 100 is 2.0000000000000004 in IEEE754; naive int() truncation
        # would price 1,000 credits at 1,999 cents.
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.02")
        assert price_cents_for_credits(1_000) == 2_000
        assert price_cents_for_credits(333) == 666

    @pytest.mark.parametrize("rate", ["0", "-0.02"])
    def test_non_positive_rate_raises_instead_of_charging_zero(self, monkeypatch, rate):
        monkeypatch.setenv("CREDIT_OVERAGE_USD", rate)
        with pytest.raises(ValueError):
            price_cents_for_credits(1_000)

    def test_non_positive_credits_raise(self):
        with pytest.raises(ValueError):
            price_cents_for_credits(0)


class TestValidateCustomCredits:
    def test_accepts_the_bounds(self):
        assert validate_custom_credits(MIN_CUSTOM_CREDITS) == MIN_CUSTOM_CREDITS
        assert validate_custom_credits(MAX_CUSTOM_CREDITS) == MAX_CUSTOM_CREDITS

    @pytest.mark.parametrize("credits", [0, -100, MIN_CUSTOM_CREDITS - 1])
    def test_rejects_below_minimum(self, credits):
        with pytest.raises(ValueError):
            validate_custom_credits(credits)

    def test_rejects_above_maximum(self):
        with pytest.raises(ValueError):
            validate_custom_credits(MAX_CUSTOM_CREDITS + 1)

    @pytest.mark.parametrize("credits", ["500", 500.5, None, True])
    def test_rejects_non_integers(self, credits):
        # `True` is an int subclass in Python and would otherwise buy 1 credit.
        with pytest.raises(ValueError):
            validate_custom_credits(credits)


class TestCreditsLineItem:
    def test_builds_ad_hoc_price_data(self, monkeypatch):
        monkeypatch.delenv("STRIPE_CREDITS_PRODUCT_ID", raising=False)
        item = credits_line_item(price_cents=1_000, name="500 Msanii credits")
        assert item == {
            "price_data": {
                "currency": "usd",
                "unit_amount": 1_000,
                "product_data": {"name": "500 Msanii credits"},
            },
            "quantity": 1,
        }

    def test_reuses_a_configured_product(self, monkeypatch):
        monkeypatch.setenv("STRIPE_CREDITS_PRODUCT_ID", "prod_credits")
        item = credits_line_item(price_cents=1_000, name="500 Msanii credits")
        assert item["price_data"]["product"] == "prod_credits"
        assert "product_data" not in item["price_data"]

    def test_blank_product_env_falls_back_to_product_data(self, monkeypatch):
        monkeypatch.setenv("STRIPE_CREDITS_PRODUCT_ID", "   ")
        item = credits_line_item(price_cents=1_000, name="500 Msanii credits")
        assert "product_data" in item["price_data"]

    def test_rejects_a_free_line_item(self):
        with pytest.raises(ValueError):
            credits_line_item(price_cents=0, name="free credits")


class TestCustomConfig:
    def test_shape_is_the_api_contract(self, monkeypatch):
        monkeypatch.delenv("CREDIT_OVERAGE_USD", raising=False)
        assert custom_config() == {
            "minCredits": MIN_CUSTOM_CREDITS,
            "maxCredits": MAX_CUSTOM_CREDITS,
            "perCreditCents": 2,
        }
