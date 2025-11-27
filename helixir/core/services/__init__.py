"""
Core Services - Shared services for Helixir framework.

Services:
- IDResolutionService: External→Internal ID mapping with cache
- BatchIDResolver: Bulk ID resolution for performance
- ChunkingService: Event-driven semantic chunking pipeline
- LinkBuilder: Edge creation for chunk graphs

Design Principles:
- Weak coupling (separate modules)
- Strong cohesion (single responsibility)
- Observable (Float integration)
- Testable (dependency injection)
"""

from .batch_resolver import BatchIDResolver, BatchResolutionError
from .chunking_service import ChunkingService
from .id_resolution import IDResolutionService
from .link_builder import LinkBuilder

__all__ = [
    "BatchIDResolver",
    "BatchResolutionError",
    "ChunkingService",
    "IDResolutionService",
    "LinkBuilder",
]
