"""Mock LLM Provider for testing."""

from typing import Any

from helixir.llm.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider that returns predefined responses.

    Used for E2E testing without actual LLM calls.
    """

    def __init__(self):
        self.responses = []
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate mock response.

        Returns:
            Predefined response or generic mock response
        """
        self.call_count += 1

        if self.responses:
            return self.responses.pop(0)

        if "extract" in prompt.lower():
            return '{"facts": [{"content": "Paris is the capital of France", "certainty": 100}]}'
        if "decision" in prompt.lower():
            return '{"operation": "ADD", "confidence": 90, "reason": "New information"}'
        if "entities" in prompt.lower():
            return '{"entities": [{"name": "Paris", "type": "Location"}, {"name": "France", "type": "Country"}]}'
        if "concepts" in prompt.lower():
            return '{"concepts": [{"name": "Geography", "type": "Domain"}]}'
        return '{"response": "mock response"}'

    def set_next_response(self, response: str):
        """Set next response to return."""
        self.responses.append(response)

    def reset(self):
        """Reset mock state."""
        self.responses.clear()
        self.call_count = 0
