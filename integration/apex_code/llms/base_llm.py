"""Base LLM interface."""

import bisect
import logging
from abc import ABC, abstractmethod
from typing import Any

import litellm

# Import LiteLLM utilities for model info
try:
    from litellm.utils import get_max_tokens, get_model_info
except ImportError:
    get_max_tokens = None
    get_model_info = None

logger = logging.getLogger(__name__)


class ContextLengthExceededError(Exception):
    """Raised when the LLM response indicates the context length was exceeded."""

    pass


class OutputLengthExceededError(Exception):
    """Raised when the LLM response was truncated due to max_tokens limit."""

    def __init__(self, message: str, truncated_response: str | None = None):
        super().__init__(message)
        self.truncated_response = truncated_response


class ParseError(Exception):
    """Raised when parsing LLM response fails."""

    pass


class BaseLLM(ABC):
    """Base LLM interface with integrated conversation management."""

    def __init__(self, model_name: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Skip token limit lookup for special models
        if model_name in ("oracle", "mock", "unknown"):
            self.max_tokens = 0 if model_name == "oracle" else 4096
        # Special handling for 1M context window models
        elif model_name in (
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-3-pro-preview",
        ):
            self.max_tokens = 1_000_000  # 1M context
        # Special handling for Claude Opus 4.5 (200k context)
        elif model_name == "claude-opus-4-5-20251101":
            self.max_tokens = 200_000  # 200k context
        # Special handling for GPT-5.1-codex with 272k context window
        elif model_name == "gpt-5.1-codex":
            self.max_tokens = 272_000  # 272k context
        # Special handling for Qwen3 Coder with 262k context window
        elif "qwen3-coder" in model_name.lower():
            self.max_tokens = 262_144  # 262k context
        # Special handling for DeepSeek V3.2 with 164k context window
        elif "deepseek" in model_name.lower():
            self.max_tokens = 163_840  # 164k context
        # Special handling for Kimi K2 models with 128k context window
        elif "kimi-k2" in model_name.lower():
            self.max_tokens = 128_000  # 128k context
        else:
            try:
                self.max_tokens = litellm.get_max_tokens(model_name)
            except Exception:
                logger.warning(
                    f"Failed to get max tokens for {model_name}, using default"
                )
                self.max_tokens = 4096  # Default reasonable limit

        self.conversation_history: list[dict[str, Any]] = []
        self.total_tokens_used = 0

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Call the LLM with a prompt and return the response."""
        pass

    def count_tokens(self, messages: list[dict]) -> int:
        """Count tokens in messages using litellm.utils.token_counter."""
        try:
            return litellm.utils.token_counter(self.model_name, messages)
        except Exception:
            # Fallback to simple estimation if token_counter fails
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return total_chars // 4  # Rough estimation: 4 chars per token

    # Conversation management methods
    def add_to_conversation(self, user_message: str, assistant_message: str) -> None:
        """Add a user-assistant exchange to conversation history."""
        self.conversation_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

    def get_conversation_status(self) -> dict[str, Any]:
        """Get conversation status."""
        token_count = self.count_tokens(self.conversation_history)
        return {
            "model": self.model_name,
            "current_tokens": token_count,
            "limit": self.max_tokens,
            "fits": token_count <= self.max_tokens,
            "total_tokens_used": self.total_tokens_used,
            "should_trim": self.should_trim_conversation(),
        }

    def should_trim_conversation(self) -> bool:
        """Check if conversation should be trimmed."""
        token_count = self.count_tokens(self.conversation_history)
        return token_count > self.max_tokens * 0.8

    def trim_conversation(self, target_tokens: int | None = None) -> int:
        """
        Trim conversation history to fit within limits.

        Args:
            target_tokens: Target token count (defaults to 60% of recommended limit)

        Returns:
            Number of tokens after trimming
        """
        if target_tokens is None:
            target_tokens = int(self.max_tokens * 0.6)

        # Simple strategy: keep the most recent messages that fit
        current_tokens = self.count_tokens(self.conversation_history)

        if current_tokens <= target_tokens:
            return current_tokens

        # Use bisect to find the right number of messages to keep
        # Create a list of token counts for different message counts
        token_counts = []
        for i in range(len(self.conversation_history) + 1):
            test_messages = self.conversation_history[-i:] if i > 0 else []
            token_counts.append(self.count_tokens(test_messages))

        # Find the largest number of messages that fits within target_tokens
        best_trim = bisect.bisect_right(token_counts, target_tokens) - 1
        best_trim = max(0, best_trim)  # Ensure we don't go negative

        # Apply the trim
        self.conversation_history[:] = self.conversation_history[-best_trim:]
        return self.count_tokens(self.conversation_history)

    def log_token_usage(self, episode_num: int, response_tokens: int) -> None:
        """Log token usage for monitoring."""
        self.total_tokens_used += response_tokens

        logger.info(
            f"Episode {episode_num}: +{response_tokens} tokens "
            f"(Total: {self.total_tokens_used})"
        )

    def manage_conversation_tokens(
        self, logger_instance, episode_num: int, elapsed_minutes: float
    ) -> None:
        """Manage conversation tokens and perform trimming if needed."""
        status = self.get_conversation_status()

        if logger_instance:
            logger_instance._log(
                f"Episode {episode_num} (elapsed: {elapsed_minutes:.1f}m): {status['current_tokens']}/{status['limit']} tokens"
            )

        # Progressive context trimming
        trim_threshold = int(self.max_tokens * 0.9)
        if status["current_tokens"] > trim_threshold:
            if logger_instance:
                logger_instance._log(
                    f"Trimming conversation: {status['current_tokens']} tokens"
                )

            # Trim to 60% of recommended limit
            trimmed_tokens = self.trim_conversation()
            if logger_instance:
                logger_instance._log(f"Trimmed to {trimmed_tokens} tokens")

        elif len(self.conversation_history) > 200:
            if logger_instance:
                logger_instance._log(
                    f"Episode-based trimming: {len(self.conversation_history)} messages"
                )

            # Keep last 150 messages (75 episodes)
            self.conversation_history[:] = self.conversation_history[-150:]
            trimmed_tokens = self.count_tokens(self.conversation_history)
            if logger_instance:
                logger_instance._log(
                    f"Kept last 150 messages ({trimmed_tokens} tokens)"
                )
