"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LLM provider.

        Args:
            model: Model name
            api_key: Optional API key
            base_url: Optional custom base URL
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.kwargs = kwargs
        self._client: Any = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate completion from the LLM.

        Args:
            system_prompt: System message
            user_prompt: User message
            response_format: Optional response format ("json_object" or None)

        Returns:
            Tuple of (response_text, metadata_dict)
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
