"""OpenAI LLM provider."""

import logging
from typing import Any

from helixir.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (GPT-4, GPT-3.5, etc.)."""

    def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._client is not None:
            return

        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.info("OpenAI client initialized (model=%s)", self.model)
        except ImportError as e:
            msg = "OpenAI package not installed. Run: pip install openai"
            raise ImportError(msg) from e

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate completion using OpenAI API."""
        self.initialize()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            metadata = {
                "provider": "openai",
                "model": self.model,
                "tokens_prompt": response.usage.prompt_tokens,
                "tokens_completion": response.usage.completion_tokens,
                "tokens_total": response.usage.total_tokens,
            }

            return content, metadata

        except Exception as e:
            logger.exception("OpenAI generation failed: %s", e)
            raise

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"
