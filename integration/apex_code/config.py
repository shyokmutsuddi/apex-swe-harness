"""Global configuration constants for apex-code."""

# Model name constants
QWEN3_CODER_480B = (
    "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"
)
GEMINI_3_PRO_PREVIEW = "gemini/gemini-3-pro-preview"
GPT5_1_CODEX = "gpt-5.1-codex"

# Models that don't support temperature control (require temperature = 1.0)
# Note: Use string values to avoid circular imports with harness.models
MODELS_NOT_SUPPORTING_TEMP = [
    "gpt-5",  # GPT5 model
    "gpt-5-codex",  # GPT5 Codex model
    "gpt-5.1-codex",  # GPT5.1 Codex model
    # Add future models here that don't support temperature control
    # "future-model-name",
]

# Default temperature for models that support temperature control
DEFAULT_TEMPERATURE = 0.1

# Required temperature for models that don't support temperature control
REQUIRED_TEMPERATURE_1_0 = 1.0

# Services that have MCP servers
SERVICES_WITH_MCP: dict[str, str] = {
    "zammad": "zammad",
    "mattermost": "mattermost",
    # Plane service may appear as plane-api or plane; normalize to one token
    "plane-api": "plane",
    "plane": "plane",
    "grafana": "grafana",
    "prometheus": "prometheus",
    "espocrm": "espocrm",
    "medusa": "medusa",
}
