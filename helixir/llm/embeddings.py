"""Embedding generation via Ollama and other providers."""

import logging

import httpx

from helixir.core.cache import EmbeddingCache
from helixir.toolkit.misc_toolbox import float_event

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings via multiple providers (Ollama, OpenAI, HuggingFace).

    Architecture:
    - Supports multiple embedding providers
    - LRU cache for performance
    - Automatic provider detection

    Supported Providers:
    - ollama: Local models (nomic-embed-text, mxbai-embed-large, etc.)
    - openai: OpenAI Embeddings API (text-embedding-3-small/large)
    - huggingface: HuggingFace models (future)

    Usage:
        >>>
        >>> generator = EmbeddingGenerator(
        ...     provider="ollama", ollama_url="http://localhost:11434", model="nomic-embed-text"
        ... )
        >>>
        >>>
        >>> generator = EmbeddingGenerator(
        ...     provider="openai", api_key="sk-...", model="text-embedding-3-small"
        ... )
        >>>
        >>> vector = await generator.generate("Paris is the capital of France")
    """

    def __init__(
        self,
        provider: str = "ollama",
        ollama_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        api_key: str | None = None,
        timeout: int = 30,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
    ):
        """
        Initialize EmbeddingGenerator.

        Args:
            provider: Embedding provider ("ollama", "openai", "huggingface")
            ollama_url: Ollama API URL (for provider="ollama")
            model: Embedding model name
            api_key: API key (for OpenAI/HuggingFace)
            timeout: Request timeout in seconds
            cache_size: Maximum number of embeddings to cache (default 10k)
            cache_ttl: Cache TTL in seconds (default 1 hour)
        """
        self.provider = provider.lower()
        self.ollama_url = ollama_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

        self._cache = EmbeddingCache(maxsize=cache_size, ttl=cache_ttl)

        if self.provider not in {"ollama", "openai", "huggingface"}:
            logger.warning(f"Unknown provider '{self.provider}', falling back to ollama")
            self.provider = "ollama"

        if self.provider in {"openai", "huggingface"} and not self.api_key:
            raise ValueError(f"Provider '{self.provider}' requires api_key")

        logger.info(
            f"EmbeddingGenerator initialized: provider={self.provider}, model={model}, "
            f"cache={cache_size} items, ttl={cache_ttl}s"
        )

    async def generate(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text to embed
            use_cache: Whether to use cache (default True)

        Returns:
            List of floats (dimensions depend on model)

        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If response format is invalid
        """
        float_event("llm.embed.start", text_len=len(text), provider=self.provider)

        if not text.strip():
            float_event("llm.embed.error", error="Empty text")
            raise ValueError("Cannot generate embedding for empty text")

        if use_cache:
            cached = self._cache.get(text)
            if cached is not None:
                logger.debug(f"Cache HIT for: {text[:50]}...")
                float_event("llm.embed.cache_hit", text_len=len(text))
                return cached

        try:
            logger.debug(
                f"Cache MISS - generating embedding (provider={self.provider}): {text[:50]}..."
            )

            if self.provider == "ollama":
                embedding = await self._generate_ollama(text)
            elif self.provider == "openai":
                embedding = await self._generate_openai(text)
            else:
                raise ValueError(f"Provider '{self.provider}' not implemented yet")

            if use_cache:
                self._cache.set(text, embedding)

            float_event("llm.embed.success", dims=len(embedding))
            return embedding

        except httpx.HTTPError as e:
            logger.exception(f"Embedding generation failed: {e}")
            float_event("llm.embed.error", error=str(e))
            raise

    async def _generate_ollama(self, text: str) -> list[float]:
        """Generate embedding via Ollama."""
        float_event("llm.embed.ollama_call", model=self.model)

        response = await self.client.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()

        data = response.json()
        if "embedding" not in data:
            raise ValueError(f"Invalid Ollama response: {data}")

        return data["embedding"]

    async def _generate_openai(self, text: str) -> list[float]:
        """Generate embedding via OpenAI."""
        float_event("llm.embed.openai_call", model=self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={"model": self.model, "input": text},
        )
        response.raise_for_status()

        data = response.json()
        if "data" not in data or not data["data"]:
            raise ValueError(f"Invalid OpenAI response: {data}")

        return data["data"][0]["embedding"]

    def get_cache_stats(self) -> dict:
        """
        Get embedding cache statistics.

        Returns:
            Dict with cache stats (hits, misses, size, hit_rate)
        """
        return self._cache.stats

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
