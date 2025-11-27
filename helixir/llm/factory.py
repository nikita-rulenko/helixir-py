"""
LLM Provider Factory - creation and management of LLM providers.

Solves problems:
1. Hardcoded providers in extractor.py, decision_engine.py, helixir_client.py
2. SOLID/DRY principle violations
3. Duplicated provider creation logic

Architecture:
- Provider Registry: easy to add new providers
- Factory Pattern: create providers by name
- Singleton: one LLM/Embedding provider instance per client
"""

import logging
from typing import TYPE_CHECKING, Any

from helixir.core.exceptions import ConfigurationError
from helixir.llm.embeddings import EmbeddingGenerator

if TYPE_CHECKING:
    from helixir.core.config import HelixMemoryConfig
    from helixir.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


LLM_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}


def register_llm_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """
    Register a new LLM provider.

    Args:
        name: Provider name (cerebras, openai, ollama)
        provider_class: Provider class
    """
    LLM_PROVIDER_REGISTRY[name.lower()] = provider_class
    logger.debug(f"Registered LLM provider: {name}")


def _lazy_load_providers() -> None:
    """
    Lazy load providers.

    Import only when needed to avoid circular imports.
    """
    if LLM_PROVIDER_REGISTRY:
        return

    from helixir.llm.providers.cerebras import CerebrasProvider
    from helixir.llm.providers.ollama import OllamaProvider
    from helixir.llm.providers.openai import OpenAIProvider

    register_llm_provider("cerebras", CerebrasProvider)
    register_llm_provider("ollama", OllamaProvider)
    register_llm_provider("openai", OpenAIProvider)


class LLMProviderFactory:
    """
    Factory for creating LLM providers.

    Uses Provider Registry for extensibility.
    """

    @staticmethod
    def create_from_config(config: HelixMemoryConfig) -> BaseLLMProvider:
        """
        Create LLM provider from configuration.

        Args:
            config: HelixMemoryConfig with LLM settings

        Returns:
            BaseLLMProvider instance

        Raises:
            ConfigurationError: If provider is unknown or settings are missing
        """
        _lazy_load_providers()

        provider_name = config.llm_provider.lower()

        if provider_name not in LLM_PROVIDER_REGISTRY:
            available = ", ".join(LLM_PROVIDER_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown LLM provider: {provider_name}. Available providers: {available}"
            )

        provider_class = LLM_PROVIDER_REGISTRY[provider_name]

        if provider_name in {"cerebras", "openai"} and not config.llm_api_key:
            raise ConfigurationError(
                f"Provider '{provider_name}' requires llm_api_key. "
                f"Set it in config.yaml or via HELIX_LLM_API_KEY env variable"
            )

        if provider_name == "ollama" and not config.llm_base_url:
            logger.warning(
                "Ollama provider without llm_base_url, using default: http://localhost:11434"
            )

        provider = provider_class(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
        )

        logger.info(
            f"Created LLM provider: {provider_name} (model={config.llm_model}, "
            f"temperature={config.llm_temperature})"
        )

        return provider

    @staticmethod
    def create(
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Create LLM provider directly.

        Args:
            provider: Provider name (cerebras, openai, ollama)
            model: Model name
            api_key: API key (for cerebras/openai)
            base_url: Base URL (for ollama)
            temperature: Generation temperature
            **kwargs: Additional parameters

        Returns:
            BaseLLMProvider instance
        """
        _lazy_load_providers()

        provider_name = provider.lower()

        if provider_name not in LLM_PROVIDER_REGISTRY:
            available = ", ".join(LLM_PROVIDER_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown LLM provider: {provider_name}. Available providers: {available}"
            )

        provider_class = LLM_PROVIDER_REGISTRY[provider_name]

        return provider_class(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )


class EmbeddingProviderFactory:
    """
    Factory for creating Embedding providers.

    Singleton pattern - one instance per client.
    """

    @staticmethod
    def create_from_config(config: HelixMemoryConfig) -> EmbeddingGenerator:
        """
        Create Embedding provider from configuration.

        Args:
            config: HelixMemoryConfig with Embedding settings

        Returns:
            EmbeddingGenerator instance

        Raises:
            ConfigurationError: If settings are missing
        """
        provider_name = config.embedding_provider.lower()

        if provider_name == "openai" and not config.embedding_api_key:
            raise ConfigurationError(
                "Provider 'openai' for embeddings requires embedding_api_key. "
                "Set it in config.yaml or via HELIX_EMBEDDING_API_KEY env variable"
            )

        embedder = EmbeddingGenerator(
            provider=config.embedding_provider,
            ollama_url=config.embedding_url,
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            timeout=config.timeout,
        )

        logger.info(
            f"Created Embedding provider: {provider_name} "
            f"(model={config.embedding_model}, url={config.embedding_url})"
        )

        return embedder

    @staticmethod
    def create(
        provider: str,
        model: str,
        ollama_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: int = 30,
    ) -> EmbeddingGenerator:
        """
        Create Embedding provider directly.

        Args:
            provider: Provider name (ollama, openai, huggingface)
            model: Model name
            ollama_url: Ollama server URL
            api_key: API key (for openai)
            timeout: Request timeout

        Returns:
            EmbeddingGenerator instance
        """
        return EmbeddingGenerator(
            provider=provider,
            ollama_url=ollama_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )


class ProviderSingleton:
    """
    Singleton for LLM and Embedding providers.

    Guarantees that each HelixirClient has only one instance of each provider.
    """

    def __init__(self, config: HelixMemoryConfig):
        """
        Initialize Singleton.

        Args:
            config: HelixMemoryConfig
        """
        self.config = config
        self._llm_provider: BaseLLMProvider | None = None
        self._embedding_provider: EmbeddingGenerator | None = None

    @property
    def llm(self) -> BaseLLMProvider:
        """
        Get LLM provider (singleton).

        Returns:
            BaseLLMProvider instance
        """
        if self._llm_provider is None:
            self._llm_provider = LLMProviderFactory.create_from_config(self.config)
            logger.debug("LLM provider singleton created")
        return self._llm_provider

    @property
    def embedding(self) -> EmbeddingGenerator:
        """
        Get Embedding provider (singleton).

        Returns:
            EmbeddingGenerator instance
        """
        if self._embedding_provider is None:
            self._embedding_provider = EmbeddingProviderFactory.create_from_config(self.config)
            logger.debug("Embedding provider singleton created")
        return self._embedding_provider
