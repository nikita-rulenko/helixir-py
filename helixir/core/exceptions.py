"""Custom exceptions for Helix Memory SDK."""


class HelixMemoryError(Exception):
    """Base exception for all Helix Memory errors."""


class ConfigurationError(HelixMemoryError):
    """Raised when configuration is invalid."""


class HelixConnectionError(HelixMemoryError):
    """Raised when connection to HelixDB fails."""


class QueryError(HelixMemoryError):
    """Raised when a HelixQL query fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        super().__init__(message)
        self.query = query


class ValidationError(HelixMemoryError):
    """Raised when input validation fails."""


class SchemaError(HelixMemoryError):
    """Raised when schema operations fail."""


class OntologyError(HelixMemoryError):
    """Raised when ontology operations fail."""


class HelixMemoryOperationError(HelixMemoryError):
    """Raised when memory operations fail."""


class ReasoningError(HelixMemoryError):
    """Raised when reasoning operations fail."""
