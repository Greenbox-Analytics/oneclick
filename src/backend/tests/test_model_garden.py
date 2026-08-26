"""Model garden: YAML contents, env precedence, priced models."""

import pytest
import yaml

from subscriptions.ai_pricing import MODEL_RATES
from utils.llm.model_garden import GARDEN_PATH, model_for


@pytest.fixture
def no_model_env(monkeypatch):
    """Strip every override — the repo .env sets the legacy vars, so tests of the
    YAML values must not read whatever the developer happens to have configured."""
    for slot, entry in yaml.safe_load(GARDEN_PATH.read_text()).items():
        monkeypatch.delenv(f"MODEL_{slot.upper()}", raising=False)
        if entry.get("legacy_env"):
            monkeypatch.delenv(entry["legacy_env"], raising=False)


def test_yaml_values_preserve_todays_models(no_model_env):
    """Regression guard: the garden must not silently move a tool to a new model."""
    assert {slot: model_for(slot) for slot in yaml.safe_load(GARDEN_PATH.read_text())} == {
        "zoe": "gpt-5-mini",
        "zoe_routing": "gpt-5-mini",
        "zoe_citations": "gpt-5-mini",
        "zoe_large": "gpt-5",
        "contract_parser": "gpt-5.2",
        "oneclick_columns": "gpt-5.4-mini",
    }


def test_every_yaml_model_is_priced():
    """An unpriced model would null out ai_usage_log.cost_usd for that tool."""
    for slot, entry in yaml.safe_load(GARDEN_PATH.read_text()).items():
        assert entry["model"] in MODEL_RATES, f"{slot}: {entry['model']} missing from MODEL_RATES"


def test_slot_env_var_wins_over_legacy_and_yaml(no_model_env, monkeypatch):
    monkeypatch.setenv("OPENAI_LLM_MODEL", "gpt-5")
    monkeypatch.setenv("MODEL_ZOE", "gpt-5.4-mini")
    assert model_for("zoe") == "gpt-5.4-mini"


def test_legacy_env_still_honored(no_model_env, monkeypatch):
    """Deployed GSM secrets must keep working after the constants were removed."""
    monkeypatch.setenv("OPENAI_LLM_MODEL", "gpt-5.4-mini")
    assert model_for("zoe") == "gpt-5.4-mini"
    assert model_for("zoe_routing") == "gpt-5.4-mini"
    assert model_for("oneclick_columns") == "gpt-5.4-mini"  # no legacy_env — its own YAML value


def test_unpriced_model_is_used_not_rejected(no_model_env, monkeypatch):
    monkeypatch.setenv("MODEL_ZOE", "some-unpriced-model")
    assert model_for("zoe") == "some-unpriced-model"


def test_unknown_slot_raises():
    with pytest.raises(KeyError):
        model_for("nope")
