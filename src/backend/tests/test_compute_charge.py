"""compute_charge is the one charge formula shared by product and partner
paths. These pin its three terms in isolation; test_credits_service.py pins
them through debit_for_action, and test_partner_calculate.py through the
partner endpoint.

Amounts are LITERALS, not expressions over the metadata the same call
produced — asserting `amount == 30 + meta["tail_credits"]` moves both sides
together and passes with the tail term broken."""

from subscriptions.ai_pricing import TAIL_FREE_TOKENS, compute_charge

ACTION = "partner_oneclick_run"


def test_partner_action_has_its_own_pinned_allowance():
    # Explicit entry, not the dict fallback: a retune of DEFAULT_TAIL_FREE_TOKENS
    # must not silently reprice the partner action.
    assert TAIL_FREE_TOKENS[ACTION] == 6_500


def test_unmeasured_charges_base_and_says_so():
    amount, meta = compute_charge(ACTION, 30, None, None)
    assert amount == 30
    assert meta["measurable"] is False
    assert meta["metered"] is False
    assert "input_tokens" not in meta


def test_cache_hit_measures_zero_and_still_pays_base():
    amount, meta = compute_charge(ACTION, 30, 0, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0})
    assert amount == 30
    assert meta["measurable"] is True and meta["metered"] is False and meta["tail_credits"] == 0


def test_base_is_a_floor_a_cheap_short_run_never_charges_less():
    # 3,500 tokens is well inside the 6,500 allowance: no tail, and measured (5)
    # is below the base. The charge is the base, never the measured cost.
    usage = {"cost_usd": 0.01, "input_tokens": 3_000, "output_tokens": 500, "calls": 1}
    amount, meta = compute_charge(ACTION, 30, 5, usage)
    assert amount == 30
    assert meta["tail_credits"] == 0 and meta["metered"] is False


def test_metered_wins_when_above_base():
    # Inside the allowance, so the tail contributes nothing and `measured`
    # alone lifts the charge.
    usage = {"cost_usd": 0.30, "input_tokens": 5_000, "output_tokens": 1_000, "calls": 1}
    amount, meta = compute_charge(ACTION, 30, 45, usage)
    assert amount == 45
    assert meta["metered"] is True and meta["tail_credits"] == 0


def test_size_tail_wins_for_a_long_cheap_run():
    # $0.06 over 21,000 tokens with 6,500 free: the excess is pro-rated
    # (0.06 x 14,500/21,000 = $0.0414) -> 7 credits, so 30 + 7. Metered (3) is
    # far below the base; the tail is what lifts the charge.
    usage = {"cost_usd": 0.06, "input_tokens": 20_000, "output_tokens": 1_000, "calls": 2}
    amount, meta = compute_charge(ACTION, 30, 3, usage)
    assert amount == 37
    assert meta["tail_credits"] == 7
    assert meta["metered"] is True
    assert meta["input_tokens"] == 20_000 and meta["llm_calls"] == 2


def test_tail_wins_when_measured_is_also_above_base():
    # Both non-base terms clear the base; max() takes base + tail (54) over
    # measured (45).
    usage = {"cost_usd": 0.20, "input_tokens": 28_000, "output_tokens": 2_000, "calls": 4}
    amount, meta = compute_charge(ACTION, 30, 45, usage)
    assert amount == 54
    assert meta["tail_credits"] == 24 and meta["metered_credits"] == 45
