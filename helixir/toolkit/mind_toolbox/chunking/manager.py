"""
ChunkingManager: Fast semantic chunking using semchunk.

Features:
- Uses semchunk for fast splitting
- Simple, predictable API
- No tokenizer loading issues
- Character-based token counting by default
"""

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any

import semchunk

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.llm.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ChunkingManager:
    """
    V2 ChunkingManager using semchunk for fast, semantic text splitting.

    Features:
    - Automatic chunking when text exceeds threshold
    - Semantic boundaries (sentences, paragraphs, etc.)
    - Optional embeddings for each chunk
    - Simple character-based token counting (no model downloads!)

    Example:
        >>> manager = ChunkingManagerV2(
        ...     client=db_client,
        ...     embedder=embedder,
        ...     threshold=500,
        ...     chunk_size=512,
        ... )
        >>> result = await manager.add_memory(
        ...     content="Long text...",
        ...     memory_id="mem_123",
        ... )
    """

    def __init__(
        self,
        client: HelixDBClient,
        embedder: EmbeddingGenerator | None = None,
        threshold: int = 500,
        chunk_size: int = 512,
        overlap: float = 0.1,
        enable_embeddings: bool = True,
    ):
        """
        Initialize ChunkingManagerV2.

        Args:
            client: HelixDB client
            embedder: Embedding generator (optional)
            threshold: Chunk if text > threshold chars
            chunk_size: Target chunk size in tokens (characters for simple counting)
            overlap: Overlap ratio (0.1 = 10% overlap between chunks)
            enable_embeddings: Whether to generate embeddings for chunks
        """
        self.client = client
        self.embedder = embedder
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_embeddings = enable_embeddings

        self.chunker = semchunk.chunkerify(
            lambda text: len(text),
            chunk_size=chunk_size,
        )

        logger.info(
            "ChunkingManager initialized (threshold=%d, chunk_size=%d, overlap=%.1f)",
            threshold,
            chunk_size,
            overlap,
        )

    def should_chunk(self, text: str) -> bool:
        """
        Determine if text should be chunked.

        Args:
            text: Input text

        Returns:
            True if text exceeds threshold
        """
        return len(text) > self.threshold

    async def add_memory(
        self,
        content: str,
        memory_id: str,
        memory_type: str = "fact",
        importance: int = 50,
        certainty: int = 50,
        source: str = "user",
        context_tags: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Add memory with automatic chunking if needed.

        Args:
            content: Memory content
            memory_id: User-facing memory ID
            memory_type: Type of memory
            importance: Importance score (0-100)
            certainty: Certainty score (0-100)
            source: Memory source
            context_tags: Comma-separated tags
            metadata: Additional metadata
            **kwargs: Additional fields

        Returns:
            Memory node data
        """
        if not self.should_chunk(content):
            logger.debug(f"Content below threshold ({len(content)} chars), storing directly")
            return await self._add_simple_memory(
                content=content,
                memory_id=memory_id,
                memory_type=memory_type,
                importance=importance,
                certainty=certainty,
                source=source,
                context_tags=context_tags,
                metadata=metadata or {},
                **kwargs,
            )

        logger.info(f"Chunking content ({len(content)} chars) into ~{self.chunk_size}-char chunks")

        chunks_text = self.chunker(content, overlap=self.overlap)

        logger.info(f"Created {len(chunks_text)} chunks")

        created_at = datetime.now(UTC)
        memory_result = await self.client.execute_query(
            "addMemory",
            {
                "memory_id": memory_id,
                "user_id": kwargs.get("user_id", "unknown"),
                "content": content,
                "memory_type": memory_type,
                "created_at": created_at.isoformat(),
                "updated_at": created_at.isoformat(),
                "certainty": certainty,
                "importance": importance,
                "context_tags": context_tags,
                "source": source,
                "metadata": str(metadata or {}),
            },
        )

        memory_internal_id = memory_result.get("memory", {}).get("id")
        if not memory_internal_id:
            raise ValueError(f"Failed to create Memory node: {memory_result}")

        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{memory_id}_chunk_{i}"

            chunk_result = await self.client.execute_query(
                "addChunk",
                {
                    "chunk_id": chunk_id,
                    "memory_id": memory_internal_id,
                    "content": chunk_text,
                    "position": i,
                    "token_count": len(chunk_text),
                    "created_at": created_at.isoformat(),
                },
            )

            chunk_internal_id = chunk_result.get("chunk", {}).get("id")

            if not chunk_internal_id:
                logger.warning(f"Failed to create chunk {i}: {chunk_result}")
                continue

            if self.enable_embeddings and self.embedder:
                try:
                    chunk_vector = await self.embedder.generate(chunk_text)

                    chunk_vector_rounded = [round(v, 2) for v in chunk_vector]

                    await self.client.execute_query(
                        "addChunkEmbedding",
                        {
                            "chunk_id": chunk_internal_id,
                            "vector_data": chunk_vector_rounded,
                        },
                    )
                    logger.debug(f"✅ Chunk {i} embedding created")
                except Exception as e:
                    logger.warning(f"Failed to create embedding for chunk {i}: {e}")

        logger.info(f"✅ Memory chunked: {len(chunks_text)} chunks created for {memory_id}")

        return memory_result

    async def _add_simple_memory(
        self,
        content: str,
        memory_id: str,
        memory_type: str,
        importance: int,
        certainty: int,
        source: str,
        context_tags: str,
        metadata: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Add memory without chunking (direct storage).

        Args:
            content: Memory content
            memory_id: Memory ID
            memory_type: Type of memory
            importance: Importance score
            certainty: Certainty score
            source: Source
            context_tags: Tags
            metadata: Metadata
            **kwargs: Additional fields

        Returns:
            Memory node data
        """
        created_at = datetime.now(UTC)

        result = await self.client.execute_query(
            "addMemory",
            {
                "memory_id": memory_id,
                "user_id": kwargs.get("user_id", "unknown"),
                "content": content,
                "memory_type": memory_type,
                "created_at": created_at.isoformat(),
                "updated_at": created_at.isoformat(),
                "certainty": certainty,
                "importance": importance,
                "context_tags": context_tags,
                "source": source,
                "metadata": str(metadata),
            },
        )

        logger.debug(f"Added simple memory: {memory_id}")
        return result
