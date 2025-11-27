"""Ollama LLM provider for local models."""

import logging
from typing import Any

import httpx

from helixir.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider for local models.

    Supports models like gemma, llama2, mistral, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "gemma2", "llama2", "mistral")
            api_key: Not used for Ollama (kept for interface compatibility)
            base_url: Ollama server URL (default: http://localhost:11434)
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional arguments
        """
        super().__init__(model, api_key, base_url, temperature, **kwargs)
        if not self.base_url:
            self.base_url = "http://localhost:11434"

    def initialize(self) -> None:
        """Initialize Ollama client (httpx)."""

        if self._client is not None:
            return

        self._client = httpx.Client(timeout=600.0)
        logger.info("Ollama client initialized (model=%s, url=%s)", self.model, self.base_url)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate completion using Ollama API."""
        self.initialize()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        if response_format == "json_object":
            payload["format"] = "json"

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            content = result["message"]["content"]

            metadata = {
                "provider": "ollama",
                "model": self.model,
                "base_url": self.base_url,
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
                "total_duration_ms": result.get("total_duration", 0) / 1_000_000,
            }

            logger.debug(
                "Ollama generation complete: %d tokens, %.2fms",
                result.get("eval_count", 0),
                metadata["total_duration_ms"],
            )

            return content, metadata

        except httpx.HTTPStatusError as e:
            logger.exception("Ollama HTTP error: %s", e)
            raise
        except Exception as e:
            logger.exception("Ollama generation failed: %s", e)
            raise

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "ollama"

    def __del__(self) -> None:
        """Clean up httpx client."""
        if self._client:
            self._client.close()
