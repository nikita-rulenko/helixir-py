"""
Helixir - Multi-Agent Memory Framework powered by HelixDB.

Main Features:
- LLM-powered memory extraction (atomic facts)
- Multi-agent support (separate memory contexts)
- Graph-based memory relations
- Vector search + BM25 + graph traversal
- HelixDB backend (fast graph + vectors)

Quick Start:
    >>> from helixir import HelixirClient
    >>> client = HelixirClient()
    >>> await client.add("I love Python", user_id="user123")
    >>> results = await client.search("What do I like?", user_id="user123")

Architecture:
    User → HelixirClient → ToolingManager (LLM-agent) → Toolboxes → HelixDB
"""

__version__ = "0.1.0"

from pathlib import Path as _Path
import sys as _sys

_helixir_root = _Path(__file__).parent
for _submodule in ["core", "llm", "toolkit", "cli", "mcp"]:
    _new_name = f"helixir.{_submodule}"
    _old_name = f"helix_memory.{_submodule}"
    if _new_name in _sys.modules:
        _sys.modules[_old_name] = _sys.modules[_new_name]

if "helixir" in _sys.modules:
    _sys.modules["helix_memory"] = _sys.modules["helixir"]

del _sys, _Path, _helixir_root, _submodule, _new_name, _old_name


from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.core.exceptions import (
    HelixConnectionError,
    HelixMemoryError,
    OntologyError,
    QueryError,
    ReasoningError,
    ValidationError,
)
from helixir.core.helixir_client import HelixirClient
from helixir.llm import EmbeddingGenerator, LLMExtractor, OllamaProvider, OpenAIProvider

__all__ = [
    "EmbeddingGenerator",
    "HelixConnectionError",
    "HelixDBClient",
    "HelixMemoryConfig",
    "HelixMemoryError",
    "HelixirClient",
    "LLMExtractor",
    "OllamaProvider",
    "OntologyError",
    "OpenAIProvider",
    "QueryError",
    "ReasoningError",
    "ValidationError",
    "__version__",
]
