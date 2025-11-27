"""
Link Builder - Event-driven edge creation for chunking pipeline.

Architecture:
    ChunkCreatedEvent
        ↓
    LinkBuilder.on_chunk_created()
        ↓
    1. HAS_CHUNK edge (Memory → Chunk) - already created by addMemoryChunk!
    2. Emit ChunkLinkedEvent
        ↓
    (Collect all chunks)
        ↓
    3. Create NEXT_CHUNK edges (Chunk → Chunk chain)
    4. Emit ChunkChainedEvent for each

Responsibilities:
- Listen to ChunkCreatedEvent
- Track chunking progress per memory
- Create NEXT_CHUNK edges when chunks complete
- Emit events for observability
- Handle edge cases (single chunk, errors)

Design:
- Stateful (tracks chunks per memory_id)
- One edge at a time (reliable > fast)
- Full Float tracking
- Error resilient
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from helixir.core.events import (
    BaseEvent,
    ChunkChainedEvent,
    ChunkCreatedEvent,
    ChunkLinkedEvent,
    EventHandler,
    get_event_bus,
)
from helixir.toolkit.misc_toolbox.float_controller import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient


class LinkBuilder(EventHandler):
    """
    Event-driven edge builder for chunking pipeline.

    Listens to ChunkCreatedEvent and:
    1. Emits ChunkLinkedEvent (HAS_CHUNK already created)
    2. Collects all chunks for a memory
    3. Creates NEXT_CHUNK edges (Chunk → Chunk chain)
    4. Emits ChunkChainedEvent for each edge

    Features:
    - Stateful chunk tracking (per memory_id)
    - Sequential edge creation (reliable)
    - Full Float observability
    - Error handling per edge

    Usage:
        builder = LinkBuilder(db_client)

        bus = get_event_bus()
        bus.register(builder)

    """

    def __init__(self, db_client: HelixDBClient):
        """
        Initialize Link Builder.

        Args:
            db_client: HelixDB client for creating edges
        """
        self._db = db_client

        self._chunks_by_memory: dict[str, list[ChunkCreatedEvent]] = defaultdict(list)

        self._expected_chunks: dict[str, int] = {}

        float_event("link_builder.initialized")

    @property
    def event_types(self) -> list[str]:
        """Events this builder handles."""
        return ["chunk.created"]

    async def handle(self, event: BaseEvent) -> None:
        """
        Handle ChunkCreatedEvent - build edges.

        Args:
            event: ChunkCreatedEvent instance
        """
        if not isinstance(event, ChunkCreatedEvent):
            return

        memory_id = event.parent_memory_id

        float_event(
            "link_builder.event_received",
            memory_id=memory_id,
            chunk_id=event.chunk_id,
            position=event.position,
        )

        bus = get_event_bus()
        await bus.emit(
            ChunkLinkedEvent(
                chunk_id=event.chunk_id,
                parent_memory_id=memory_id,
                position=event.position,
                correlation_id=event.correlation_id,
            )
        )

        self._chunks_by_memory[memory_id].append(event)
        self._expected_chunks[memory_id] = event.total_chunks

        float_event(
            "link_builder.chunk_tracked",
            memory_id=memory_id,
            collected=len(self._chunks_by_memory[memory_id]),
            expected=event.total_chunks,
        )

        if len(self._chunks_by_memory[memory_id]) == event.total_chunks:
            float_event(
                "link_builder.all_chunks_collected",
                memory_id=memory_id,
                total=event.total_chunks,
            )

            await self._create_chunk_chain(memory_id, event.correlation_id)

            del self._chunks_by_memory[memory_id]
            del self._expected_chunks[memory_id]

    async def _create_chunk_chain(
        self,
        memory_id: str,
        correlation_id: Any,
    ) -> None:
        """
        Create NEXT_CHUNK edges (Chunk → Chunk chain).

        Args:
            memory_id: Parent memory ID
            correlation_id: Event correlation ID
        """
        chunks = self._chunks_by_memory[memory_id]

        chunks.sort(key=lambda c: c.position)

        float_event(
            "link_builder.chain_creation_started",
            memory_id=memory_id,
            chunks=len(chunks),
        )

        if len(chunks) == 1:
            float_event(
                "link_builder.single_chunk_no_chain",
                memory_id=memory_id,
            )
            return

        edges_created = 0
        errors = []

        for i in range(len(chunks) - 1):
            from_chunk = chunks[i]
            to_chunk = chunks[i + 1]

            try:
                await self._create_next_chunk_edge(
                    from_chunk=from_chunk,
                    to_chunk=to_chunk,
                    correlation_id=correlation_id,
                )
                edges_created += 1

            except Exception as e:
                float_event(
                    "link_builder.edge_creation_failed",
                    from_chunk=from_chunk.chunk_id,
                    to_chunk=to_chunk.chunk_id,
                    error=str(e),
                )
                errors.append((from_chunk.chunk_id, to_chunk.chunk_id, str(e)))

        float_event(
            "link_builder.chain_creation_complete",
            memory_id=memory_id,
            edges_created=edges_created,
            errors=len(errors),
        )

        if errors:
            for from_id, to_id, error in errors:
                float_event(
                    "link_builder.chain_partial_failure",
                    from_chunk=from_id,
                    to_chunk=to_id,
                    error=error,
                )

    async def _create_next_chunk_edge(
        self,
        from_chunk: ChunkCreatedEvent,
        to_chunk: ChunkCreatedEvent,
        correlation_id: Any,
    ) -> None:
        """
        Create a single NEXT_CHUNK edge.

        Args:
            from_chunk: Source chunk event
            to_chunk: Target chunk event
            correlation_id: Event correlation ID

        Raises:
            Exception: If edge creation fails
        """
        float_event(
            "link_builder.edge_creation_started",
            from_chunk=from_chunk.chunk_id,
            to_chunk=to_chunk.chunk_id,
        )

        try:
            result = await self._db.execute_query(
                "linkChunks",
                {
                    "from_chunk_id": from_chunk.chunk_internal_id,
                    "to_chunk_id": to_chunk.chunk_internal_id,
                },
            )

            edge_id = result.get("id") if result else None

            float_event(
                "link_builder.edge_created",
                from_chunk=from_chunk.chunk_id,
                to_chunk=to_chunk.chunk_id,
                edge_id=str(edge_id),
            )

            bus = get_event_bus()
            await bus.emit(
                ChunkChainedEvent(
                    from_chunk_id=from_chunk.chunk_id,
                    to_chunk_id=to_chunk.chunk_id,
                    edge_id=edge_id,
                    position=from_chunk.position,
                    correlation_id=correlation_id,
                )
            )

        except Exception as e:
            float_event(
                "link_builder.edge_creation_error",
                from_chunk=from_chunk.chunk_id,
                to_chunk=to_chunk.chunk_id,
                error=str(e),
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get LinkBuilder statistics for monitoring."""
        return {
            "pending_memories": len(self._chunks_by_memory),
            "total_chunks_tracked": sum(len(chunks) for chunks in self._chunks_by_memory.values()),
        }


__all__ = ["LinkBuilder"]
