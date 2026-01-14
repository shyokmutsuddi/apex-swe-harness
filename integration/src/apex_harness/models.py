"""
Model configuration registry for different AI models.

This module implements the Strategy pattern for model selection and configuration.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific AI model."""

    name: str
    """Display name of the model"""

    identifier: str
    """Model identifier used in API calls"""

    report_prefix: str
    """Prefix for generated report names"""

    status_csv_prefix: str
    """Prefix for status CSV files"""

    n_trials: int = 3
    """Number of trials to run per task"""

    max_workers: int = 3
    """Maximum number of parallel workers"""

    timeout: int = 3600
    """Timeout in seconds per task"""

    requires_env_var: Optional[str] = None
    """Required environment variable for API access"""


# Model Registry - Single source of truth for all model configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "claude": ModelConfig(
        name="Claude Sonnet 4.5",
        identifier="claude-sonnet-4-5-20250929",
        report_prefix="claude",
        status_csv_prefix="claude",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
    "opus": ModelConfig(
        name="Claude Opus 4",
        identifier="anthropic/claude-opus-4-20250514",
        report_prefix="opus",
        status_csv_prefix="opus",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
    "xai": ModelConfig(
        name="xAI Grok 4",
        identifier="xai/grok-4",
        report_prefix="grok",
        status_csv_prefix="xai",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
    "gemini": ModelConfig(
        name="Gemini 3 Pro Preview",
        identifier="gemini/gemini-3-pro-preview",
        report_prefix="gemini",
        status_csv_prefix="gemini",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
    "deepseek": ModelConfig(
        name="DeepSeek V3.2",
        identifier="fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        report_prefix="deepseek",
        status_csv_prefix="deepseek",
        n_trials=3,
        max_workers=3,
        timeout=3600,
        requires_env_var="FIREWORKS_API_KEY",
    ),
    "codex": ModelConfig(
        name="Codex",
        identifier="codex-model-identifier",  # Update with actual identifier
        report_prefix="codex",
        status_csv_prefix="codex",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
    "qwen": ModelConfig(
        name="Qwen 3 Coder 480B",
        identifier="fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
        report_prefix="fireworks",
        status_csv_prefix="fireworks",
        n_trials=3,
        max_workers=2,
        timeout=3600,
        requires_env_var="FIREWORKS_API_KEY",
    ),
    "kimi": ModelConfig(
        name="Kimi",
        identifier="kimi-model-identifier",  # Update with actual identifier
        report_prefix="kimi",
        status_csv_prefix="kimi",
        n_trials=3,
        max_workers=3,
        timeout=3600,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name.

    Args:
        model_name: Name of the model (e.g., 'claude', 'gemini')

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If model name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models() -> list[str]:
    """Get list of all available model names."""
    return sorted(MODEL_REGISTRY.keys())
