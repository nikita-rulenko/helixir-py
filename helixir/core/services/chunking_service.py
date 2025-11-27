"""
Chunking Service - Event-driven chunking pipeline orchestrator.

Architecture:
    MemoryCreatedEvent
        ↓
    ChunkingService.on_memory_created()
        ↓
    1. Check if needs chunking (content size)
    2. Resolve parent Memory ID (external → internal UUID)
    3. Split content using Chonkie SemanticChunker
    4. Emit ChunkingStartedEvent
        ↓
    5. Create MemoryChunk nodes in parallel
    6. Emit ChunkCreatedEvent for each
        ↓
    LinkBuilder listens and creates edges
        ↓
    ChunkingCompleteEvent

Responsibilities:
- Orchestrate chunking pipeline
- Use Chonkie for semantic chunking
- Emit events for observability
- Handle errors gracefully
- Track performance via Float
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from chonkie import SemanticChunker, SentenceChunker

from helixir.core.events import (
    BaseEvent,
    ChunkCreatedEvent,
    ChunkingCompleteEvent,
    ChunkingFailedEvent,
    ChunkingStartedEvent,
    EventHandler,
    MemoryCreatedEvent,
    get_event_bus,
)
from helixir.toolkit.misc_toolbox.float_controller import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.core.services.id_resolution import IDResolutionService


class ChunkingService(EventHandler):
    """
    Event-driven chunking service using Chonkie.

    Listens to MemoryCreatedEvent and orchestrates:
    1. ID resolution (external → internal UUID)
    2. Content splitting (Chonkie SemanticChunker)
    3. Chunk creation (parallel DB operations)
    4. Event emission (observability)

    Features:
    - Semantic chunking (meaning-aware)
    - Configurable thresholds
    - Parallel chunk creation
    - Full Float tracking
    - Error resilience

    Usage:
        service = ChunkingService(
            db_client=db,
            id_resolver=resolver
        )

        bus = get_event_bus()
        bus.register(service)

    """

    def __init__(
        self,
        db_client: HelixDBClient,
        id_resolver: IDResolutionService,
        chunk_size: int = 1024,
        similarity_threshold: float = 0.7,
        min_chunk_length: int = 1000,
        min_sentences_per_chunk: int = 2,
        use_semantic: bool = True,
    ):
        """
        Initialize Chunking Service.

        Args:
            db_client: HelixDB client for creating chunks
            id_resolver: ID resolution service for UUID lookup
            chunk_size: Maximum tokens per chunk
            similarity_threshold: Semantic similarity threshold (0-1) - only for SemanticChunker
            min_chunk_length: Minimum content length to trigger chunking
            min_sentences_per_chunk: Minimum sentences per chunk
            use_semantic: If True, use SemanticChunker (embeddings); if False, use SentenceChunker (fast)
        """
        self._db = db_client
        self._id_resolver = id_resolver
        self._min_chunk_length = min_chunk_length

        if use_semantic:
            float_event(
                "chunking_service.initializing",
                chunker="SemanticChunker",
                chunk_size=chunk_size,
                threshold=similarity_threshold,
            )
            self._chunker = SemanticChunker(
                embedding_model="minishlab/potion-base-32M",
                threshold=similarity_threshold,
                chunk_size=chunk_size,
                min_sentences_per_chunk=min_sentences_per_chunk,
            )
        else:
            float_event(
                "chunking_service.initializing",
                chunker="SentenceChunker",
                chunk_size=chunk_size,
            )
            self._chunker = SentenceChunker(
                chunk_size=chunk_size,
                chunk_overlap=128,
                min_sentences_per_chunk=min_sentences_per_chunk,
            )

        float_event(
            "chunking_service.initialized",
            chunker=self._chunker.__class__.__name__,
            chunk_size=chunk_size,
            min_chunk_length=min_chunk_length,
        )

    @property
    def event_types(self) -> list[str]:
        """Events this service handles."""
        return ["memory.created"]

    async def handle(self, event: BaseEvent) -> None:
        """
        Handle MemoryCreatedEvent - orchestrate chunking pipeline.

        Args:
            event: MemoryCreatedEvent instance
        """
        if not isinstance(event, MemoryCreatedEvent):
            return

        float_event(
            "chunking_service.event_received",
            memory_id=event.memory_id,
            content_length=len(event.content),
        )

        if not event.needs_chunking:
            float_event(
                "chunking_service.skipped",
                memory_id=event.memory_id,
                reason="content_too_short",
                length=len(event.content),
                threshold=self._min_chunk_length,
            )
            return

        start_time = time.time()

        try:
            await self._process_chunking(event)

            duration_ms = (time.time() - start_time) * 1000
            float_event(
                "chunking_service.pipeline_success",
                memory_id=event.memory_id,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            float_event(
                "chunking_service.pipeline_failed",
                memory_id=event.memory_id,
                error=str(e),
                duration_ms=duration_ms,
            )

            bus = get_event_bus()
            await bus.emit(
                ChunkingFailedEvent(
                    memory_id=event.memory_id,
                    stage="chunking_pipeline",
                    error=str(e),
                    correlation_id=event.correlation_id,
                )
            )

            raise

    async def _process_chunking(self, event: MemoryCreatedEvent) -> None:
        """
        Execute chunking pipeline.

        Steps:
        1. Resolve parent Memory internal UUID
        2. Split content using Chonkie
        3. Emit ChunkingStartedEvent
        4. Create MemoryChunk nodes in parallel
        5. Emit ChunkCreatedEvent for each
        6. Emit ChunkingCompleteEvent

        Args:
            event: MemoryCreatedEvent to process
        """
        memory_id = event.memory_id
        content = event.content

        float_event("chunking_service.resolving_id", memory_id=memory_id)

        internal_id = event.internal_id or await self._id_resolver.resolve(memory_id)

        float_event(
            "chunking_service.id_resolved",
            memory_id=memory_id,
            internal_id=str(internal_id),
        )

        float_event(
            "chunking_service.splitting_content",
            memory_id=memory_id,
            content_length=len(content),
        )

        chonkie_chunks = self._chunker.chunk(content)

        float_event(
            "chunking_service.content_split",
            memory_id=memory_id,
            chunks_count=len(chonkie_chunks),
            avg_tokens=sum(c.token_count for c in chonkie_chunks) // len(chonkie_chunks),
        )

        bus = get_event_bus()
        await bus.emit(
            ChunkingStartedEvent(
                memory_id=memory_id,
                internal_id=internal_id,
                content_length=len(content),
                estimated_chunks=len(chonkie_chunks),
                chunking_strategy="semantic",
                correlation_id=event.correlation_id,
            )
        )

        float_event(
            "chunking_service.creating_chunks",
            memory_id=memory_id,
            count=len(chonkie_chunks),
        )

        tasks = [
            self._create_chunk(
                parent_memory_id=memory_id,
                parent_internal_id=internal_id,
                chonkie_chunk=chonkie_chunk,
                position=idx,
                total_chunks=len(chonkie_chunks),
                correlation_id=event.correlation_id,
            )
            for idx, chonkie_chunk in enumerate(chonkie_chunks)
        ]

        chunk_events = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [e for e in chunk_events if isinstance(e, ChunkCreatedEvent)]
        errors = [e for e in chunk_events if isinstance(e, Exception)]

        if errors:
            float_event(
                "chunking_service.chunk_creation_errors",
                memory_id=memory_id,
                errors_count=len(errors),
                successful_count=len(successful),
            )
            raise errors[0]

        await bus.emit(
            ChunkingCompleteEvent(
                memory_id=memory_id,
                chunks_created=len(successful),
                links_created=0,
                chains_created=0,
                duration_ms=0,
                success=True,
                correlation_id=event.correlation_id,
            )
        )

    async def _create_chunk(
        self,
        parent_memory_id: str,
        parent_internal_id: Any,
        chonkie_chunk: Any,
        position: int,
        total_chunks: int,
        correlation_id: Any,
    ) -> ChunkCreatedEvent:
        """
        Create a single MemoryChunk node in DB.

        Args:
            parent_memory_id: External parent memory ID
            parent_internal_id: Internal parent UUID
            chonkie_chunk: Chonkie Chunk object
            position: Chunk position in sequence (0-indexed)
            total_chunks: Total number of chunks
            correlation_id: Event correlation ID

        Returns:
            ChunkCreatedEvent

        Raises:
            Exception: If chunk creation fails
        """
        chunk_id = f"{parent_memory_id}_chunk_{position}"

        float_event(
            "chunking_service.chunk_creation_started",
            chunk_id=chunk_id,
            position=position,
        )

        try:
            result = await self._db.execute_query(
                "addMemoryChunk",
                {
                    "chunk_id": chunk_id,
                    "parent_id": parent_internal_id,
                    "position": position,
                    "content": chonkie_chunk.text,
                    "token_count": chonkie_chunk.token_count,
                    "created_at": "{{timestamp}}",
                },
            )

            chunk_internal_id = result.get("id") if result else None

            float_event(
                "chunking_service.chunk_created",
                chunk_id=chunk_id,
                chunk_internal_id=str(chunk_internal_id),
                position=position,
                token_count=chonkie_chunk.token_count,
            )

            event = ChunkCreatedEvent(
                chunk_id=chunk_id,
                chunk_internal_id=chunk_internal_id,
                parent_memory_id=parent_memory_id,
                parent_internal_id=parent_internal_id,
                position=position,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                total_chunks=total_chunks,
                correlation_id=correlation_id,
            )

            bus = get_event_bus()
            await bus.emit(event)

            return event

        except Exception as e:
            float_event(
                "chunking_service.chunk_creation_failed",
                chunk_id=chunk_id,
                position=position,
                error=str(e),
            )
            raise


__all__ = ["ChunkingService"]
