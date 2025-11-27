"""Core module for Helixir - main entry point for the framework."""

from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.core.exceptions import (
    HelixConnectionError,
    HelixMemoryError,
    HelixMemoryOperationError,
    OntologyError,
    QueryError,
    ReasoningError,
    SchemaError,
    ValidationError,
)
from helixir.core.helixir_client import HelixirClient

__all__ = [
    "HelixConnectionError",
    "HelixDBClient",
    "HelixMemoryConfig",
    "HelixMemoryError",
    "HelixMemoryOperationError",
    "HelixirClient",
    "OntologyError",
    "QueryError",
    "ReasoningError",
    "SchemaError",
    "ValidationError",
]
