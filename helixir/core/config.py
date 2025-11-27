"""Configuration management for Helix Memory SDK."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class FloatConfig(BaseSettings):
    """
    Float Controller Configuration.

    Controls event-driven execution tracking system.
    """

    enabled: bool = Field(
        default=False, description="Enable float collection (True=tests/debug, False=production)"
    )
    auto_enable_in_tests: bool = Field(
        default=True, description="Auto-enable when running in test environment"
    )
    max_events: int = Field(default=10000, description="Max floats to keep in memory (0=unlimited)")
    auto_print_report: bool = Field(default=False, description="Print float report on context exit")

    model_config = SettingsConfigDict(
        env_prefix="HELIX_FLOAT_",
        extra="ignore",
    )


class HelixMemoryConfig(BaseSettings):
    """
    Configuration for Helix Memory SDK.

    Can be loaded from:
    - Environment variables (prefix: HELIX_)
    - YAML file
    - Direct initialization

    Example:
        >>> config = HelixMemoryConfig(host="localhost", port=6969)
        >>> config = HelixMemoryConfig.from_yaml("config.yaml")
        >>> config = HelixMemoryConfig()
    """

    model_config = SettingsConfigDict(
        env_prefix="HELIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    host: str = Field(
        default="localhost",
        description="HelixDB host address",
    )
    port: int = Field(
        default=6969,
        ge=1,
        le=65535,
        description="HelixDB port",
    )
    instance: str = Field(
        default="dev",
        description="HelixDB instance name",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for HelixDB (if auth enabled)",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed requests",
    )

    schema_dir: Path = Field(
        default=Path(__file__).parent.parent / "schema",
        description="Directory containing HelixQL schema files",
    )
    auto_load_schema: bool = Field(
        default=True,
        description="Automatically load schema on initialization",
    )

    llm_provider: str = Field(
        default="cerebras",
        description="LLM provider for text generation (cerebras, openai, ollama)",
    )
    llm_model: str = Field(
        default="llama3.3-70b",
        description="LLM model name for text generation",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM provider (required for Cerebras/OpenAI)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="LLM provider base URL (for Ollama: http://localhost:11434)",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM temperature for text generation",
    )

    embedding_provider: str = Field(
        default="ollama",
        description="Embedding provider (ollama, openai, huggingface)",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name (nomic-embed-text recommended: 768d, 8K context)",
    )
    embedding_url: str = Field(
        default="http://localhost:11434",
        description="Embedding service URL (for Ollama)",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for embedding provider (for OpenAI)",
    )

    default_certainty: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Default certainty score for memories",
    )
    default_importance: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Default importance score for memories",
    )
    conflict_resolution: str = Field(
        default="prefer_newer",
        description="Conflict resolution strategy (prefer_newer, prefer_higher_certainty, manual)",
    )

    default_search_limit: int = Field(
        default=10,
        ge=1,
        description="Default number of search results",
    )
    default_search_mode: str = Field(
        default="recent",
        description="Default search mode (recent/contextual/deep/full)",
    )
    vector_search_enabled: bool = Field(
        default=True,
        description="Enable vector search",
    )
    graph_search_enabled: bool = Field(
        default=True,
        description="Enable graph traversal search",
    )
    bm25_search_enabled: bool = Field(
        default=True,
        description="Enable BM25 full-text search",
    )

    search_mode_recent_limit: int = Field(
        default=10,
        ge=1,
        description="Max results for RECENT mode",
    )
    search_mode_contextual_limit: int = Field(
        default=20,
        ge=1,
        description="Max results for CONTEXTUAL mode",
    )
    search_mode_deep_limit: int = Field(
        default=50,
        ge=1,
        description="Max results for DEEP mode",
    )
    search_mode_full_limit: int = Field(
        default=100,
        ge=1,
        description="Max results for FULL mode",
    )

    @field_validator("schema_dir")
    @classmethod
    def validate_schema_dir(cls, v: Path) -> Path:
        """Ensure schema directory is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    @classmethod
    def find_config_yaml(cls) -> Path | None:
        """
        Search for config.yaml in standard locations.

        Search order:
        1. Current working directory
        2. Project root (parent of helixir package)
        3. User home directory

        Returns:
            Path to config.yaml if found, None otherwise
        """
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
            Path.home() / ".helixdb" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> HelixMemoryConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file. If None, searches standard locations.

        Returns:
            HelixMemoryConfig instance

        Raises:
            FileNotFoundError: If path is specified but file not found

        Example:
            >>>
            >>> config = HelixMemoryConfig.from_yaml("config.yaml")
            >>>
            >>>
            >>> config = HelixMemoryConfig.from_yaml()
        """
        if path is None:
            path = cls.find_config_yaml()
            if path is None:
                raise FileNotFoundError(
                    "Config file not found. Searched:\n"
                    "  1. ./config.yaml\n"
                    "  2. <project_root>/config.yaml\n"
                    "  3. ~/.helixdb/config.yaml"
                )
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        import os

        result_data = {}

        for key, value in yaml_data.items():
            env_key = f"HELIX_{key.upper()}"
            if env_key in os.environ:
                continue
            result_data[key] = value

        return cls(**result_data)

    def to_yaml(self, path: Path | str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration

        Example:
            >>> config.to_yaml("config.yaml")
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @property
    def base_url(self) -> str:
        """Get base URL for HelixDB API."""
        return f"http://{self.host}:{self.port}"

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"HelixMemoryConfig(host={self.host!r}, port={self.port}, instance={self.instance!r})"
        )
