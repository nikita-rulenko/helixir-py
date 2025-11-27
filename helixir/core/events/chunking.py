"""
Chunking Events - Event definitions for chunking pipeline.

Event Flow:
    MemoryCreatedEvent
        ↓
    ChunkingStartedEvent
        ↓
    ChunkCreatedEvent (x N chunks)
        ↓
    ChunkLinkedEvent (x N links)
        ↓
    ChunkChainedEvent (x N-1 chains)
        ↓
    ChunkingCompleteEvent

Each event is immutable and carries all context needed for handlers.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import BaseEvent

if TYPE_CHECKING:
    from uuid import UUID


@dataclass(frozen=True)
class MemoryCreatedEvent(BaseEvent):
    """
    Emitted when a new Memory node is created.

    Triggers the chunking pipeline if content exceeds threshold.

    Attributes:
        memory_id: External memory identifier (String)
        content: Full text content to chunk
        user_id: User who created the memory
        internal_id: Internal HelixDB UUID (if already resolved)
        extra_metadata: Additional memory metadata
    """

    memory_id: str
    content: str
    user_id: str
    internal_id: UUID | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        return "memory.created"

    @property
    def needs_chunking(self) -> bool:
        """Check if content is large enough to require chunking."""

        return len(self.content) > 1000


@dataclass(frozen=True)
class ChunkingStartedEvent(BaseEvent):
    """
    Emitted when chunking pipeline begins.

    Attributes:
        memory_id: External memory identifier
        internal_id: Resolved internal UUID
        content_length: Total content size
        estimated_chunks: Estimated number of chunks
        chunking_strategy: Strategy used (e.g., "semantic", "fixed")
    """

    memory_id: str
    internal_id: UUID
    content_length: int
    estimated_chunks: int
    chunking_strategy: str = "semantic"

    @property
    def event_type(self) -> str:
        return "chunking.started"


@dataclass(frozen=True)
class ChunkCreatedEvent(BaseEvent):
    """
    Emitted when a single chunk is created in DB.

    Attributes:
        chunk_id: External chunk identifier (String)
        chunk_internal_id: Internal HelixDB UUID
        parent_memory_id: External parent memory ID
        parent_internal_id: Internal parent UUID
        position: Chunk position in sequence (0-indexed)
        content: Chunk text content
        token_count: Number of tokens in chunk
        total_chunks: Total number of chunks in sequence
    """

    chunk_id: str
    chunk_internal_id: UUID
    parent_memory_id: str
    parent_internal_id: UUID
    position: int
    content: str
    token_count: int
    total_chunks: int

    @property
    def event_type(self) -> str:
        return "chunk.created"

    @property
    def is_first(self) -> bool:
        """Check if this is the first chunk."""
        return self.position == 0

    @property
    def is_last(self) -> bool:
        """Check if this is the last chunk."""
        return self.position == self.total_chunks - 1


@dataclass(frozen=True)
class ChunkLinkedEvent(BaseEvent):
    """
    Emitted when HAS_CHUNK edge is created (Memory → Chunk).

    Attributes:
        chunk_id: External chunk identifier
        parent_memory_id: External parent memory ID
        edge_id: Internal edge UUID (if available)
        position: Chunk position
    """

    chunk_id: str
    parent_memory_id: str
    edge_id: UUID | None = None
    position: int = 0

    @property
    def event_type(self) -> str:
        return "chunk.linked"


@dataclass(frozen=True)
class ChunkChainedEvent(BaseEvent):
    """
    Emitted when NEXT_CHUNK edge is created (Chunk → Chunk).

    Attributes:
        from_chunk_id: Source chunk (position N)
        to_chunk_id: Target chunk (position N+1)
        edge_id: Internal edge UUID (if available)
        position: Position of source chunk
    """

    from_chunk_id: str
    to_chunk_id: str
    edge_id: UUID | None = None
    position: int = 0

    @property
    def event_type(self) -> str:
        return "chunk.chained"


@dataclass(frozen=True)
class ChunkingCompleteEvent(BaseEvent):
    """
    Emitted when entire chunking pipeline completes.

    Attributes:
        memory_id: External memory identifier
        chunks_created: Number of chunks created
        links_created: Number of HAS_CHUNK edges
        chains_created: Number of NEXT_CHUNK edges
        duration_ms: Pipeline duration in milliseconds
        success: Whether pipeline completed successfully
        error: Error message if failed
    """

    memory_id: str
    chunks_created: int
    links_created: int
    chains_created: int
    duration_ms: float
    success: bool = True
    error: str | None = None

    @property
    def event_type(self) -> str:
        return "chunking.complete"

    @property
    def all_edges_created(self) -> bool:
        """Verify all expected edges were created."""
        expected_links = self.chunks_created
        expected_chains = max(0, self.chunks_created - 1)
        return self.links_created == expected_links and self.chains_created == expected_chains


@dataclass(frozen=True)
class ChunkingFailedEvent(BaseEvent):
    """
    Emitted when chunking pipeline fails.

    Attributes:
        memory_id: External memory identifier
        stage: Pipeline stage where failure occurred
        error: Error message
        chunks_created: Number of chunks created before failure
        retry_count: Number of retry attempts
    """

    memory_id: str
    stage: str
    error: str
    chunks_created: int = 0
    retry_count: int = 0

    @property
    def event_type(self) -> str:
        return "chunking.failed"


__all__ = [
    "ChunkChainedEvent",
    "ChunkCreatedEvent",
    "ChunkLinkedEvent",
    "ChunkingCompleteEvent",
    "ChunkingFailedEvent",
    "ChunkingStartedEvent",
    "MemoryCreatedEvent",
]
