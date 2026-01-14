"""Tests for model registry and configuration."""

import pytest

from apex_harness.models import (
    MODEL_REGISTRY,
    ModelConfig,
    get_model_config,
    list_available_models,
)


def test_model_registry_exists():
    """Verify model registry is populated."""
    assert len(MODEL_REGISTRY) > 0
    assert "claude" in MODEL_REGISTRY
    assert "gemini" in MODEL_REGISTRY


def test_model_config_structure():
    """Verify ModelConfig has required fields."""
    config = MODEL_REGISTRY["claude"]
    assert isinstance(config, ModelConfig)
    assert config.name
    assert config.identifier
    assert config.report_prefix
    assert config.status_csv_prefix
    assert config.n_trials > 0
    assert config.max_workers > 0
    assert config.timeout > 0


def test_get_model_config_valid():
    """Test retrieving valid model configuration."""
    config = get_model_config("claude")
    assert config.name == "Claude Sonnet 4.5"
    assert config.identifier == "claude-sonnet-4-5-20250929"
    assert config.report_prefix == "claude"


def test_get_model_config_invalid():
    """Test error handling for invalid model."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_config("nonexistent-model")


def test_list_available_models():
    """Test listing all available models."""
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "claude" in models
    assert "gemini" in models
    # Verify sorted
    assert models == sorted(models)


def test_all_models_have_consistent_structure():
    """Ensure all models in registry have valid configurations."""
    for model_name, config in MODEL_REGISTRY.items():
        assert isinstance(config, ModelConfig)
        assert config.name, f"Model {model_name} missing name"
        assert config.identifier, f"Model {model_name} missing identifier"
        assert config.report_prefix, f"Model {model_name} missing report_prefix"
        assert (
            config.status_csv_prefix
        ), f"Model {model_name} missing status_csv_prefix"
        assert config.n_trials > 0, f"Model {model_name} invalid n_trials"
        assert config.max_workers > 0, f"Model {model_name} invalid max_workers"
        assert config.timeout > 0, f"Model {model_name} invalid timeout"


def test_model_configurations_match_legacy():
    """Verify configurations match original scripts."""
    # Claude
    claude = get_model_config("claude")
    assert claude.identifier == "claude-sonnet-4-5-20250929"
    assert claude.report_prefix == "claude"
    assert claude.n_trials == 3
    assert claude.max_workers == 3
    assert claude.timeout == 3600

    # XAI
    xai = get_model_config("xai")
    assert xai.identifier == "xai/grok-4"
    assert xai.report_prefix == "grok"
    assert xai.status_csv_prefix == "xai"

    # Gemini
    gemini = get_model_config("gemini")
    assert gemini.identifier == "gemini/gemini-3-pro-preview"
    assert gemini.report_prefix == "gemini"

    # DeepSeek
    deepseek = get_model_config("deepseek")
    assert (
        deepseek.identifier
        == "fireworks_ai/accounts/fireworks/models/deepseek-v3p2"
    )
    assert deepseek.report_prefix == "deepseek"
    assert deepseek.requires_env_var == "FIREWORKS_API_KEY"

    # Qwen
    qwen = get_model_config("qwen")
    assert (
        qwen.identifier
        == "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"
    )
    assert qwen.report_prefix == "fireworks"
    assert qwen.max_workers == 2  # Different from others
