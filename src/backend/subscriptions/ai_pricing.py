"""Authoritative model-rate table + cost estimator for AI usage logging.

SOURCE OF TRUTH for real-cost computation (`ai_usage_log.cost_usd`).
The pricing dashboard (subscriptions/pricing_model/index.html, JS `MODELS` map)
carries a planning-tool copy of these rates — when OpenAI pricing changes,
update BOTH (the dashboard README says the same).

Rates are USD per 1M tokens, standard tier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRate:
    input_usd: float  # $/1M input tokens
    cached_input_usd: float  # $/1M cached input tokens
    output_usd: float  # $/1M output tokens


MODEL_RATES: dict[str, ModelRate] = {
    "gpt-5.2": ModelRate(1.75, 0.175, 14.00),
    "gpt-5": ModelRate(1.25, 0.125, 10.00),
    "gpt-5-mini": ModelRate(0.25, 0.025, 2.00),
    "gpt-5.4-mini": ModelRate(0.75, 0.075, 4.50),
    "text-embedding-3-small": ModelRate(0.02, 0.02, 0.0),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float | None:
    """Real USD cost of one API call, honoring the cached-input discount.

    Returns None for unknown models — callers log a warning and store NULL,
    never crash a user-facing request over a rate-table gap.
    """
    rate = MODEL_RATES.get(model)
    if rate is None:
        return None
    cached = min(max(cached_tokens or 0, 0), max(input_tokens, 0))
    uncached = max(input_tokens, 0) - cached
    return (uncached * rate.input_usd + cached * rate.cached_input_usd + max(output_tokens, 0) * rate.output_usd) / 1e6
