"""Cerebras Inference LLM provider - world's fastest AI inference.

Cerebras delivers 70x faster inference than GPUs:
- Llama 3.3-70B: ~2,100 tokens/sec
- Llama 3.1-8B: ~2,000 tokens/sec
- Llama 4 Maverick 400B: ~2,500 tokens/sec (world record!)

Free Tier: 1 million tokens/day
Pricing: Starting at $0.10 per million tokens

Perfect for:
- Real-time AI agents
- Multi-step reasoning
- High-volume extraction tasks
- Memory processing at scale
"""

import logging
from typing import Any

from helixir.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class CerebrasProvider(BaseLLMProvider):
    """
    Cerebras Inference provider - ultra-fast LLM inference.

    Supported models:
    - llama-3.3-70b (recommended for balanced speed/quality)
    - llama-3.1-8b (fastest, good for simple tasks)
    - llama-3.1-70b (high quality, very fast)
    - llama-4-scout-17b-16e-instruct (Llama 4 latest)
    - llama-4-maverick-17b-128e-instruct (preview)

    Usage:
        provider = CerebrasProvider(
            api_key="your-api-key",
            model="llama-3.3-70b",
            base_url="https://api.cerebras.ai/v1"
        )

    Free tier:
        - 1 million tokens per day
        - 30 requests per minute
        - Sign up at: https://cloud.cerebras.ai
    """

    def initialize(self) -> None:
        """Initialize Cerebras client."""
        if self._client is not None:
            return

        try:
            from cerebras.cloud.sdk import Cerebras

            self._client = Cerebras(
                api_key=self.api_key,
            )
            logger.info("Cerebras client initialized (model=%s, speed=⚡ultra-fast)", self.model)
            logger.debug("Cerebras base URL: https://api.cerebras.ai/v1")
        except ImportError as e:
            msg = (
                "Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk\n"
                "Or: uv add cerebras-cloud-sdk"
            )
            raise ImportError(msg) from e

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate completion using Cerebras Inference API.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            response_format: Optional "json_object" for structured output

        Returns:
            Tuple of (content, metadata)

        Note:
            Cerebras supports structured outputs with JSON schema enforcement.
            For complex extraction, consider using instructor library.
        """
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
                "provider": "cerebras",
                "model": self.model,
                "base_url": "https://api.cerebras.ai/v1",
            }

            if hasattr(response, "usage") and response.usage:
                metadata.update(
                    {
                        "tokens_prompt": response.usage.prompt_tokens,
                        "tokens_completion": response.usage.completion_tokens,
                        "tokens_total": response.usage.total_tokens,
                    }
                )

            logger.debug(
                "Cerebras generation completed: %d prompt + %d completion tokens",
                metadata.get("tokens_prompt", 0),
                metadata.get("tokens_completion", 0),
            )

            return content, metadata

        except Exception as e:
            logger.exception("Cerebras generation failed: %s", e)
            raise

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "cerebras"
