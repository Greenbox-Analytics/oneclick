"""Read the model garden (config_panel/model_garden.yaml) and answer "which model?".

Call sites ask by SLOT — model_for("zoe_routing") — and never name a model id.
Precedence per slot: MODEL_<SLOT> env > the slot's legacy_env > the YAML value.

Loaded once per process (the YAML is baked into the image). A parse error
propagates: a broken garden is a config bug to fix, not something to paper over
with a stale fallback that hides which model is actually running.
"""

import logging
import os
from functools import cache
from pathlib import Path

import yaml

from subscriptions.ai_pricing import MODEL_RATES

logger = logging.getLogger(__name__)

GARDEN_PATH = Path(__file__).resolve().parents[2] / "config_panel" / "model_garden.yaml"


@cache
def _garden() -> dict:
    return yaml.safe_load(GARDEN_PATH.read_text()) or {}


@cache
def _warn_unpriced(model: str, slot: str) -> None:
    """Once per (model, slot) for the process — functools.cache as a log-spam latch."""
    logger.warning(
        "Model %r (slot %r) is not in MODEL_RATES — ai_usage_log.cost_usd will be NULL "
        "for these calls. Add a rate in subscriptions/ai_pricing.py.",
        model,
        slot,
    )


def model_for(slot: str) -> str:
    """Model id for a job slot. Raises KeyError on an unknown slot (typo = bug).

    An unpriced model is warned about, not rejected: cost tracking degrades but a
    user-facing request must never fail over a rate-table gap.
    """
    entry = _garden()[slot]
    legacy_env = entry.get("legacy_env")
    model = os.getenv(f"MODEL_{slot.upper()}") or (os.getenv(legacy_env) if legacy_env else None) or entry["model"]
    if model not in MODEL_RATES:
        _warn_unpriced(model, slot)
    return model
