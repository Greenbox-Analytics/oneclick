"""Authoritative model-rate table + cost estimator for AI usage logging.

SOURCE OF TRUTH for real-cost computation (`ai_usage_log.cost_usd`) and for the
overage rate quoted to users. The pricing dashboard
(subscriptions/pricing_model/index.html) needs the rates in JS, so it keeps a
copy — but its entry point (pricing_model/open.py) diffs that copy against this
table and refuses to open when they disagree, so drift can't go unnoticed.

Rates are USD per 1M tokens, standard tier.
"""

import math
import os
from dataclasses import dataclass


def overage_usd_per_credit() -> float:
    """USD charged per credit of pay-per-use overage.

    SOURCE OF TRUTH for the rate: the Stripe biller (overage_billing.py) and the
    rate shown in the UI (Entitlements.to_dict → CreditsUsageCard) both read it
    here, so changing CREDIT_OVERAGE_USD can never leave the two disagreeing.
    """
    return float(os.getenv("CREDIT_OVERAGE_USD", "0.02"))


def credit_markup() -> float:
    """Multiplier applied to real LLM cost when converting it into credits.

    Same dial as the pricing dashboard's "target markup on COGS" (tab 1): a
    credit sells for `overage_usd_per_credit()`, so credits = COGS x markup /
    price-per-credit. 3.0 is the dashboard baseline and reproduces the flat
    prices this replaced (zoe 3 / registry 12 / oneclick 21) at their modeled
    token counts.
    """
    return float(os.getenv("CREDIT_MARKUP", "3.0"))


def credits_for_cost(cost_usd: float) -> int:
    """Credits owed for a MEASURED USD cost of LLM work.

    Rounded UP, so any real spend costs at least 1 credit; zero cost (a cache
    hit, or an action that made no LLM call at all) is genuinely free.

    The round() before ceil kills float dust: 0.10 * 3 / 0.02 evaluates to
    15.000000000000002 in IEEE754, which a naive ceil bills as 16.
    """
    if cost_usd <= 0:
        return 0
    per_credit = overage_usd_per_credit()
    if per_credit <= 0:
        return 0
    return max(1, math.ceil(round(cost_usd * credit_markup() / per_credit, 6)))


# ---------------------------------------------------------------------------
# Size tail: the free token allowance before a run starts costing extra
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Before this, the tail threshold was not a decision anyone
# made — it fell out of the base rate. max(base, metered) starts charging extra
# only once metered exceeds the base, i.e. at cost > base / markup x
# price-per-credit = $0.20 for a 30-credit action. Measured runs cost ~$0.015,
# so the implied threshold sat at ~13x the largest run anyone had ever done and
# a 60-page contract cost exactly what a 3-page one did. Setting the allowance
# explicitly decouples "when does size start to matter" from "what does this
# action cost", which were only ever welded together by arithmetic.
#
# CALIBRATED 2026-08-27 against ai_usage_log (39 calls) + credit_ledger token
# metadata, reconstructed into runs. Sample is SMALL (12 runs) — re-run
# scripts/ against a bigger window before trusting these to two significant
# figures. Observed, tokens per run:
#
#   oneclick + registry   median 4,824   p90  5,105   max  8,056   (~7.4 pages)
#   zoe                   median 27,164  p90 35,971   max 63,115
#
# Zoe burns ~6x a document run because retrieval reads across the corpus, and
# its base is 5 rather than 30 — so ONE global allowance would have quietly
# tripled the price of a median Zoe message. Hence per-action values.
#
# ponytail: a dict, not a credit_prices column. These want tuning as data
# accumulates, but n=12 means they want tuning from a dashboard by a human, not
# hot-swapping in prod. Move them into credit_prices (public read, next to the
# base) if a deploy-per-tune ever becomes the annoying part.
DEFAULT_TAIL_FREE_TOKENS = 6_500  # ~10 pages at ~650 tokens/page

TAIL_FREE_TOKENS: dict[str, int] = {
    # Document extraction: ~10 pages included, then ~0.3 credits/page.
    "oneclick_run": 6_500,
    # Same document-extraction shape as oneclick_run, same allowance. The dict
    # fallback would give this number anyway; the explicit entry is what a
    # future retune will look for.
    "partner_oneclick_run": 6_500,
    "registry_parse": 6_500,
    "partner_registry_parse": 6_500,
    # Retrieval-heavy. 30,000 sits just above the observed median so the
    # advertised 5 still holds for a normal question, and is deliberately kept
    # at or below what the base is worth in COGS (5 / 150 = $0.033) so the
    # allowance can never be more generous than the base already paid for.
    "zoe_message": 30_000,
    # No LLM call at all, so no tail can ever fire. Present for completeness.
    "split_sheet": 6_500,
    "partner_split_sheet": 6_500,
}


def tail_free_tokens(action: str) -> int:
    """Tokens a single run of `action` gets before the size tail engages.

    CREDIT_TAIL_FREE_TOKENS overrides every action at once — an ops dial for
    turning the tail off (set it very high) without a deploy, not a per-action
    knob.
    """
    override = os.getenv("CREDIT_TAIL_FREE_TOKENS")
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    return TAIL_FREE_TOKENS.get(action, DEFAULT_TAIL_FREE_TOKENS)


def credits_for_excess(cost_usd: float, total_tokens: int, free_tokens: int) -> int:
    """Credits owed for the portion of a run ABOVE its free token allowance.

    The excess cost is pro-rated by token share rather than recomputed from the
    per-call rows: the accumulator holds one total, and which specific tokens
    were "the free ones" is not a real distinction. Pro-rating keeps the
    expensive/cheap model mix of the actual run instead of assuming the excess
    was all cheap input.

    0 when the run fits in its allowance, when nothing was measured, or when the
    allowance is disabled — so a cache hit (0 tokens, 0 cost) never tails.
    """
    if total_tokens <= free_tokens or total_tokens <= 0 or cost_usd <= 0:
        return 0
    return credits_for_cost(cost_usd * (1.0 - free_tokens / total_tokens))


def compute_charge(action: str, base: int, measured: int | None, usage: dict | None) -> tuple[int, dict]:
    """THE charge for one metered action, and the ledger metadata that explains it.

    Three terms, and the max() of all three is the charge (2026-08-27):

      base              the published price of the deliverable, and a hard
                        floor. Never charge below it.
      measured          what the run actually cost, in credits (real OpenAI
                        cost x CREDIT_MARKUP / price-per-credit). Keeps the
                        guarantee that we never sell a run below COGS,
                        whatever the allowance is set to.
      base + size tail  a run gets tail_free_tokens(action) of tokens
                        included and pays for what it burns past that. This
                        is what makes a 60-page contract cost more than a
                        3-page one.

    `measured is None` = unmeasurable (no tracked scope, or a model missing
    from MODEL_RATES): charge the base, and say so in the metadata. A cache
    hit measures 0 and still pays the base — the user got the same
    deliverable; the dedupe request_id is what stops a second charge.

    Metadata: `metered` means "did a term ABOVE the base decide the amount",
    NOT "was cost readable" — under base rates `measurable` is true on nearly
    every LLM-touching row while the amount is still the base. `tail_credits`
    and `metered_credits` record what each term wanted, so a ledger row alone
    answers "charged for size, or for cost?" — the recalibration question.

    Callers: EntitlementsService.debit_for_action (product) and the partner
    calculate endpoint. There must never be a third max() anywhere.
    """
    usage = usage or {}
    tail = 0
    if measured is not None:
        tail = credits_for_excess(
            usage.get("cost_usd", 0.0),
            (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
            tail_free_tokens(action),
        )
    amount = base if measured is None else max(base, measured, base + tail)
    metadata = {
        "estimated": base,
        "base": base,
        "measurable": measured is not None,
        "metered_credits": measured,
        "tail_credits": tail,
        "free_tokens": tail_free_tokens(action),
        "metered": amount > base,
    }
    if usage:
        metadata.update(
            {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "llm_calls": usage.get("calls", 0),
                "cost_usd": round(usage.get("cost_usd", 0.0), 6),
            }
        )
    return amount, metadata


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
