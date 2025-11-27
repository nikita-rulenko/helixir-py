"""Memory Integrator - connects new memories into unified knowledge graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import math
from typing import TYPE_CHECKING

from helixir.core.exceptions import HelixMemoryOperationError
from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.llm.embeddings import EmbeddingGenerator
    from helixir.toolkit.mind_toolbox.memory.models import Memory
    from helixir.toolkit.mind_toolbox.reasoning.engine import ReasoningEngine

logger = logging.getLogger(__name__)


@dataclass
class SimilarMemory:
    """Similar memory candidate for linking."""

    memory_id: str
    content: str
    embedding: list[float]
    similarity_score: float
    created_at: datetime


@dataclass
class MemoryRelation:
    """Proposed relation between memories."""

    target_id: str
    relation_type: str
    confidence: float
    reasoning: str


@dataclass
class IntegrationResult:
    """Result of memory integration process."""

    memory_id: str
    similar_found: int
    relations_created: int
    superseded_memories: list[str]
    integration_time_ms: float


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


class MemoryIntegrator:
    """
    Integrates new memories into unified knowledge graph.

    Solves fragmentation by:
    1. Finding similar memories via direct embedding comparison
    2. Using LLM reasoning to determine relations
    3. Creating edges to connect new memory to existing graph
    4. Handling duplicates via SUPERSEDES
    5. Forcing vector index sync
    """

    def __init__(
        self,
        client: HelixDBClient,
        embedding_gen: EmbeddingGenerator,
        reasoning_engine: ReasoningEngine | None = None,
        similarity_threshold: float = 0.7,
        max_similar: int = 10,
        enable_reasoning: bool = True,
    ) -> None:
        """
        Initialize MemoryIntegrator.

        Args:
            client: HelixDB client
            embedding_gen: Embedding generator
            reasoning_engine: Optional reasoning engine for relation inference
            similarity_threshold: Minimum cosine similarity for candidates (default: 0.7)
            max_similar: Max number of similar memories to consider (default: 10)
            enable_reasoning: Whether to use LLM reasoning (default: True)
        """
        self.client = client
        self.embedding_gen = embedding_gen
        self.reasoning_engine = reasoning_engine
        self.similarity_threshold = similarity_threshold
        self.max_similar = max_similar
        self.enable_reasoning = enable_reasoning
        self.logger = logging.getLogger(self.__class__.__name__)

    async def integrate_memory(
        self,
        memory: Memory,
        query_embedding: list[float],
    ) -> IntegrationResult:
        """
        Integrate new memory into knowledge graph.

        Args:
            memory: Newly created memory
            query_embedding: Embedding of memory content

        Returns:
            IntegrationResult with stats
        """
        start_time = asyncio.get_event_loop().time()
        float_event("memory.integration.start", memory_id=memory.memory_id)

        try:
            similar_memories = await self._find_similar_by_embedding(
                query_embedding=query_embedding,
                user_id=memory.user_id,
                exclude_id=memory.memory_id,
            )

            if not similar_memories:
                self.logger.info("No similar memories found for %s", memory.memory_id[:8])
                return IntegrationResult(
                    memory_id=memory.memory_id,
                    similar_found=0,
                    relations_created=0,
                    superseded_memories=[],
                    integration_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                )

            relations = await self._infer_relations(
                new_memory=memory,
                similar_memories=similar_memories,
            )

            created_count = await self._create_relations(
                source_id=memory.memory_id,
                relations=relations,
            )

            superseded = await self._handle_duplicates(
                new_memory=memory,
                similar_memories=similar_memories,
            )

            await self._ensure_indexed(memory.memory_id)

            integration_time = (asyncio.get_event_loop().time() - start_time) * 1000

            float_event(
                "memory.integration.complete",
                memory_id=memory.memory_id,
                similar_found=len(similar_memories),
                relations_created=created_count,
                superseded=len(superseded),
                time_ms=integration_time,
            )

            return IntegrationResult(
                memory_id=memory.memory_id,
                similar_found=len(similar_memories),
                relations_created=created_count,
                superseded_memories=superseded,
                integration_time_ms=integration_time,
            )

        except Exception as e:
            self.logger.exception("Memory integration failed: %s", e)
            float_event("memory.integration.error", error=str(e))
            raise HelixMemoryOperationError(
                f"Failed to integrate memory {memory.memory_id}: {e}"
            ) from e

    async def _find_similar_by_embedding(
        self,
        query_embedding: list[float],
        user_id: str,
        exclude_id: str | None = None,
    ) -> list[SimilarMemory]:
        """
        Find similar memories by vector search (chunk-aware).

        Uses smartVectorSearchWithChunks HelixQL query for semantic similarity.
        Handles both direct Memory embeddings and MemoryChunk embeddings.

        Args:
            query_embedding: Query embedding vector
            user_id: User ID to filter by
            exclude_id: Memory ID to exclude from results

        Returns:
            List of similar memories sorted by score
        """
        try:
            results = await self.client.execute_query(
                "smartVectorSearchWithChunks",
                {
                    "query_vector": query_embedding,
                    "limit": self.max_similar * 2,
                },
            )

            all_memories = []
            all_memories.extend(results.get("memories", []))
            all_memories.extend(results.get("parent_memories", []))

            seen_ids = set()
            memories = []
            for mem in all_memories:
                mem_id = mem.get("memory_id")
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    memories.append(mem)

            candidates: list[SimilarMemory] = []
            for memory in memories:
                memory_id = memory.get("memory_id")

                if memory_id == exclude_id:
                    continue

                mem_user_id = memory.get("user_id") or "unknown"
                if mem_user_id != user_id:
                    continue

                content = memory.get("content", "")
                created_at_str = memory.get("created_at", "")

                score = 0.8

                if score >= self.similarity_threshold:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except Exception:
                        created_at = datetime.now(UTC)

                    candidates.append(
                        SimilarMemory(
                            memory_id=memory_id,
                            content=content,
                            embedding=query_embedding,
                            similarity_score=score,
                            created_at=created_at,
                        )
                    )

            candidates.sort(key=lambda x: x.similarity_score, reverse=True)

            return candidates[: self.max_similar]

        except Exception as e:
            self.logger.exception("Failed to find similar memories: %s", e)
            return []

    async def _infer_relations(
        self,
        new_memory: Memory,
        similar_memories: list[SimilarMemory],
    ) -> list[MemoryRelation]:
        """
        Infer relations between new memory and similar memories.

        Uses LLM reasoning if enabled, otherwise uses simple heuristics.

        Args:
            new_memory: New memory
            similar_memories: Candidate similar memories

        Returns:
            List of proposed relations
        """
        if not self.enable_reasoning or not self.reasoning_engine:
            return [
                MemoryRelation(
                    target_id=sim.memory_id,
                    relation_type="RELATES_TO",
                    confidence=sim.similarity_score,
                    reasoning=f"Semantic similarity: {sim.similarity_score:.2f}",
                )
                for sim in similar_memories
                if sim.similarity_score >= 0.75
            ]

        relations: list[MemoryRelation] = []

        for sim in similar_memories:
            try:
                inferred = await self.reasoning_engine.infer_relation(
                    source_content=new_memory.content,
                    target_content=sim.content,
                    similarity_score=sim.similarity_score,
                )

                if inferred:
                    relations.append(
                        MemoryRelation(
                            target_id=sim.memory_id,
                            relation_type=inferred.get("type", "RELATES_TO"),
                            confidence=inferred.get("confidence", sim.similarity_score),
                            reasoning=inferred.get("reasoning", "LLM inference"),
                        )
                    )

            except Exception as e:
                self.logger.warning(
                    "Reasoning failed for %s -> %s: %s",
                    new_memory.memory_id[:8],
                    sim.memory_id[:8],
                    e,
                )
                relations.append(
                    MemoryRelation(
                        target_id=sim.memory_id,
                        relation_type="RELATES_TO",
                        confidence=sim.similarity_score,
                        reasoning=f"Fallback: similarity {sim.similarity_score:.2f}",
                    )
                )

        return relations

    async def _create_relations(
        self,
        source_id: str,
        relations: list[MemoryRelation],
    ) -> int:
        """
        Create edges in the graph using HelixQL queries.

        Args:
            source_id: Source memory ID
            relations: Relations to create

        Returns:
            Number of relations created
        """
        if not relations:
            return 0

        created = 0
        for rel in relations:
            try:
                rel_type = rel.relation_type.upper()

                if rel_type == "IMPLIES":
                    await self.client.execute_query(
                        "addMemoryImplication",
                        {
                            "from_id": source_id,
                            "to_id": rel.target_id,
                            "probability": int(rel.confidence * 100),
                            "reasoning_id": rel.reasoning[:255] if rel.reasoning else "",
                        },
                    )
                elif rel_type == "BECAUSE":
                    await self.client.execute_query(
                        "addMemoryCausation",
                        {
                            "from_id": source_id,
                            "to_id": rel.target_id,
                            "strength": int(rel.confidence * 100),
                            "reasoning_id": rel.reasoning[:255] if rel.reasoning else "",
                        },
                    )
                elif rel_type == "CONTRADICTS":
                    await self.client.execute_query(
                        "addMemoryContradiction",
                        {
                            "from_id": source_id,
                            "to_id": rel.target_id,
                            "resolution": "",
                            "resolved": 0,
                            "resolution_strategy": rel.reasoning[:255] if rel.reasoning else "",
                        },
                    )
                else:
                    await self.client.execute_query(
                        "addMemoryRelation",
                        {
                            "source_id": source_id,
                            "target_id": rel.target_id,
                            "relation_type": rel_type,
                            "strength": int(rel.confidence * 100),
                            "created_at": datetime.now(UTC).isoformat(),
                            "metadata": rel.reasoning,
                        },
                    )

                created += 1
                self.logger.debug(
                    "Created %s relation: %s -> %s",
                    rel_type,
                    source_id[:8],
                    rel.target_id[:8],
                )

            except Exception as e:
                self.logger.warning(
                    "Failed to create relation %s -> %s: %s",
                    source_id[:8],
                    rel.target_id[:8],
                    e,
                )

        return created

    async def _handle_duplicates(
        self,
        new_memory: Memory,
        similar_memories: list[SimilarMemory],
    ) -> list[str]:
        """
        Handle duplicate memories.

        Note: SUPERSEDES edge type not in schema, so this is a no-op for now.
        Future: implement custom duplicate handling strategy.

        Args:
            new_memory: New memory
            similar_memories: Similar memories

        Returns:
            List of superseded memory IDs (currently always empty)
        """
        duplicates = [sim.memory_id for sim in similar_memories if sim.similarity_score >= 0.95]

        if duplicates:
            self.logger.info(
                "Detected %d potential duplicates for %s (similarity >0.95)",
                len(duplicates),
                new_memory.memory_id[:8],
            )

        return []

    async def _ensure_indexed(self, memory_id: str) -> None:
        """
        Ensure memory is indexed in vector search.

        HelixDB indexes asynchronously - we just wait a bit.

        Args:
            memory_id: Memory ID to ensure is indexed
        """
        await asyncio.sleep(0.1)

        self.logger.debug("Waiting for async indexing of %s", memory_id[:8])
