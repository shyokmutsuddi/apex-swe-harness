"""Mock LLM implementation for testing."""

import time

from .base_llm import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, model_name: str = "mock-llm"):
        super().__init__(model_name=model_name)

    def call(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        time.sleep(0.1)

        return """I'm a mock LLM working on this task. I'll make a minimal attempt but this will likely fail the real tests.

<tool_use>
{
  "tool": "terminal",
  "command": "echo 'Mock LLM - minimal attempt' && echo 'This is not a real solution' > mock_output.txt"
}
</tool_use>

This is a mock response that should fail evaluation."""
