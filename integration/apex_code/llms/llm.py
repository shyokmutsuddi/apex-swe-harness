"""LiteLLM implementation for all providers."""

import os

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
)
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base_llm import BaseLLM, ContextLengthExceededError, OutputLengthExceededError
from .mock_llm import MockLLM
from .oracle_model import OracleModel
from .utils import get_model_temperature


class LiteLLM(BaseLLM):
    """LiteLLM implementation for all providers."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key

        # Set model-specific temperature (reasoning models don't support temperature)
        self.temperature = get_model_temperature(model_name, temperature)
        # limits is set by BaseLLM.__init__

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        retry=retry_if_not_exception_type(
            (
                ContextLengthExceededError,
                OutputLengthExceededError,
                LiteLLMAuthenticationError,
            )
        ),
    )
    def call(self, prompt: str, **kwargs) -> str:
        """Call the LLM with a prompt and return the response."""
        try:
            messages = self.conversation_history + [{"role": "user", "content": prompt}]

            # Prepare completion kwargs
            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "api_key": self.api_key,
                "drop_params": True,  # Drop unsupported params gracefully
            }

            completion_kwargs.update(kwargs)

            # Add beta headers for claude-sonnet-4-20250514 to enable 1M context
            if (
                self.model_name == "claude-sonnet-4-20250514"
                or self.model_name == "claude-sonnet-4-5-20250929"
            ):
                completion_kwargs["extra_headers"] = {
                    "anthropic-beta": "context-1m-2025-08-07"
                }

            # Enable thinking/reasoning content for all models
            completion_kwargs["reasoning_effort"] = "high"

            # Set temperature to 1.0 for reasoning models if needed (Anthropic reasoning models often require temp=1 or non-zero)
            # Actually, standard behavior for reasoning models is usually fixed temperature or temp=1
            if (
                self.model_name == "claude-sonnet-4-5-20250929"
                or self.model_name == "claude-opus-4-5-20251101"
            ):
                completion_kwargs["temperature"] = 1.0

            response = litellm.completion(**completion_kwargs)

            output = response.choices[0].message.content

            # Extract token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0

            return output

        except LiteLLMContextWindowExceededError:
            raise ContextLengthExceededError
        except LiteLLMAuthenticationError:
            raise  # Re-raise as-is
        except Exception as e:
            raise e


def get_provider_for_model(model: str) -> str:
    """Auto-detect provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gpt"):
        return "openai"
    elif model.startswith("gemini"):
        return "google"
    elif model.startswith("xai/"):
        return "xai"
    elif model.startswith("meta_llama/"):
        return "meta_llama"
    elif model.startswith("fireworks_ai/") or model.startswith("accounts/fireworks/") or model.startswith("fireworks/"):
        return "fireworks_ai"
    elif model == "oracle":
        return "oracle"
    elif model == "mock":
        return "mock"
    else:
        raise ValueError(f"Unknown model: {model}")


def get_api_key_env_for_model(model: str) -> str:
    """Get the API key environment variable for a model."""
    provider = get_provider_for_model(model)
    api_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "meta_llama": "LLAMA_API_KEY",
        "fireworks_ai": "FIREWORKS_API_KEY",
        "oracle": None,
        "mock": None,
    }
    return api_key_map[provider]


def create_llm(
    model_name: str,
    api_key: str | None = None,
    task_dir: str | None = None,
    container_name: str | None = None,
) -> BaseLLM:
    """Create an LLM from model name."""
    provider = get_provider_for_model(model_name)

    if provider == "oracle":
        if not task_dir:
            raise ValueError("Oracle model requires task_dir parameter")
        from pathlib import Path

        return OracleModel(
            task_dir=Path(task_dir),
            container_name=container_name,
        )
    elif provider == "mock":
        return MockLLM(model_name)
    else:
        if not api_key:
            api_key_env = get_api_key_env_for_model(model_name)
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found for {model_name}")

        return LiteLLM(model_name, api_key)
