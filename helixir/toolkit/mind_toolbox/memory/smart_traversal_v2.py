"""
Smart Graph Traversal V2 - Vector-First Approach.

Key Innovation: Two-Phase Search
    Phase 1: Vector Search → Find semantically similar nodes (Top-K)
    Phase 2: Graph Expansion → Load logical connections from hits
    Phase 3: Combined Ranking → Semantic + Logical + Temporal scoring

Architecture:
    User Query → Embedding → Vector Search → Graph Expansion → Ranked Results

Advantages over V1:
- ✅ Precision: Vector finds semantically similar
- ✅ Context: Graph adds logical connections
- ✅ Performance: Only traverse neighborhoods of relevant nodes
- ✅ Relevance: Combined scoring (vector + edge + temporal)

Performance targets:
- Phase 1 (Vector): < 100ms
- Phase 2 (Graph): < 200ms per hit
- Phase 3 (Ranking): < 50ms
- Total: < 500ms for depth=2, top_k=10
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from helixir.core.cache import LRUCache
from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


EDGE_WEIGHTS = {
    "BECAUSE": 1.0,
    "IMPLIES": 0.9,
    "SIMILAR_TO": 0.75,
    "MEMORY_RELATION": 0.7,
    "EXTRACTED_ENTITY": 0.6,
    "CONTRADICTS": 0.4,
}
DEFAULT_EDGE_WEIGHT = 0.5


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity (0-1)."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    similarity = dot_product / (mag1 * mag2)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def calculate_temporal_freshness(created_at: str, decay_days: float = 30.0) -> float:
    """Calculate temporal freshness (0-1) with exponential decay."""
    try:
        created = datetime.fromisoformat(created_at)
        now = datetime.now(UTC)
        days_old = (now - created).total_seconds() / 86400
        freshness = math.exp(-days_old / decay_days)
        return max(0.0, min(1.0, freshness))
    except (ValueError, AttributeError):
        return 0.5


@dataclass
class SearchResult:
    """Single search result with combined scoring."""

    memory_id: str
    content: str
    vector_score: float
    graph_score: float
    temporal_score: float
    combined_score: float
    depth: int
    source: str
    edge_path: list[str] | None = None
    metadata: dict[str, Any] | None = None


class SmartTraversalV2:
    """
    Vector-First Graph Traversal.

    Two-phase search:
    1. Vector Search: Find semantically similar nodes (Top-K)
    2. Graph Expansion: Expand from hits via logical edges
    3. Combined Ranking: Merge and rank by combined score
    """

    def __init__(
        self,
        client: HelixDBClient,
        cache_size: int = 500,
        cache_ttl: int = 300,
    ):
        """
        Initialize SmartTraversalV2.

        Args:
            client: HelixDB client
            cache_size: LRU cache size
            cache_ttl: Cache TTL (seconds)
        """
        self.client = client
        self._cache: LRUCache[str, list[SearchResult]] = LRUCache(maxsize=cache_size, ttl=cache_ttl)
        logger.info("SmartTraversalV2 initialized")

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        vector_top_k: int = 10,
        graph_depth: int = 2,
        min_vector_score: float = 0.5,
        min_combined_score: float = 0.3,
        edge_types: list[str] | None = None,
        temporal_cutoff: datetime | None = None,
    ) -> list[SearchResult]:
        """
        Two-phase vector-first search.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            user_id: Optional user filter
            vector_top_k: Number of vector search results (Phase 1)
            graph_depth: Max graph expansion depth (Phase 2)
            min_vector_score: Min score for vector hits
            min_combined_score: Min score for final results
            edge_types: Whitelist edge types (None = all logical types)
            temporal_cutoff: Only return memories created after this datetime (None = no filter)

        Returns:
            Ranked search results with combined scoring
        """
        cache_key = f"{hash(tuple(query_embedding))}-{user_id}-{vector_top_k}-{graph_depth}"
        if cached := self._cache.get(cache_key):
            float_event("smart_traversal_v2.cache_hit")
            logger.debug("SmartTraversalV2 cache HIT")
            return cached

        float_event("smart_traversal_v2.cache_miss")
        logger.info(
            "SmartTraversalV2 search: query=%s, vector_top_k=%d, graph_depth=%d",
            query[:50],
            vector_top_k,
            graph_depth,
        )
        start_time = time.perf_counter()

        phase1_start = time.perf_counter()
        vector_hits = await self._vector_search_phase(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=vector_top_k,
            min_score=min_vector_score,
            temporal_cutoff=temporal_cutoff,
        )
        phase1_duration = time.perf_counter() - phase1_start
        float_event("smart_traversal_v2.phase1_duration", duration=phase1_duration)
        logger.info(
            "Phase 1 (Vector Search): %d hits in %.2fms",
            len(vector_hits),
            phase1_duration * 1000,
        )

        if not vector_hits:
            logger.warning("No vector hits found, returning empty results")
            return []

        phase2_start = time.perf_counter()
        graph_results = await self._graph_expansion_phase(
            vector_hits=vector_hits,
            query_embedding=query_embedding,
            max_depth=graph_depth,
            edge_types=edge_types or ["BECAUSE", "IMPLIES", "MEMORY_RELATION"],
        )
        phase2_duration = time.perf_counter() - phase2_start
        float_event("smart_traversal_v2.phase2_duration", duration=phase2_duration)
        logger.info(
            "Phase 2 (Graph Expansion): %d nodes in %.2fms",
            len(graph_results),
            phase2_duration * 1000,
        )

        phase3_start = time.perf_counter()
        all_results = vector_hits + graph_results
        ranked_results = self._rank_and_filter(
            results=all_results,
            min_combined_score=min_combined_score,
        )
        phase3_duration = time.perf_counter() - phase3_start
        float_event("smart_traversal_v2.phase3_duration", duration=phase3_duration)

        total_duration = time.perf_counter() - start_time
        float_event("smart_traversal_v2.total_duration", duration=total_duration)
        logger.info(
            "SmartTraversalV2 completed: %d results in %.2fms (P1: %.0fms, P2: %.0fms, P3: %.0fms)",
            len(ranked_results),
            total_duration * 1000,
            phase1_duration * 1000,
            phase2_duration * 1000,
            phase3_duration * 1000,
        )

        self._cache.set(cache_key, ranked_results)

        return ranked_results

    async def _vector_search_phase(
        self,
        query_embedding: list[float],
        user_id: str | None,
        top_k: int,
        min_score: float,
        temporal_cutoff: datetime | None = None,
    ) -> list[SearchResult]:
        """
        Phase 1: Vector search for semantically similar nodes (chunk-aware).

        Args:
            query_embedding: Query embedding
            user_id: Optional user filter
            top_k: Number of results
            min_score: Minimum similarity score
            temporal_cutoff: Only return memories created after this datetime (None = no filter)

        Returns:
            List of vector search results
        """
        try:
            results = await self.client.execute_query(
                "smartVectorSearchWithChunks",
                {
                    "query_vector": query_embedding,
                    "limit": top_k,
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

            logger.debug(f"smartVectorSearchWithChunks returned {len(memories)} unique memories")

            if temporal_cutoff:
                filtered_memories = []
                for memory in memories:
                    created_at_str = memory.get("created_at", "")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str
                            )
                            if created_at >= temporal_cutoff:
                                filtered_memories.append(memory)
                        except Exception:
                            logger.warning(f"Failed to parse created_at: {created_at_str}")
                            continue
                memories = filtered_memories
                logger.debug(f"After temporal filter: {len(memories)} memories")

            search_results = []
            for memory in memories:
                memory_id = memory.get("memory_id")
                if not memory_id:
                    continue

                vector_score = 0.8

                temporal_score = calculate_temporal_freshness(memory.get("created_at", ""))
                combined_score = vector_score * 0.7 + temporal_score * 0.3

                search_results.append(
                    SearchResult(
                        memory_id=memory_id,
                        content=memory.get("content", ""),
                        vector_score=vector_score,
                        graph_score=0.0,
                        temporal_score=temporal_score,
                        combined_score=combined_score,
                        depth=0,
                        source="vector",
                    )
                )

            logger.info(f"Vector search phase complete: {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.exception("Vector search phase failed: %s", e)
            return []

    async def _graph_expansion_phase(
        self,
        vector_hits: list[SearchResult],
        query_embedding: list[float],
        max_depth: int,
        edge_types: list[str],
    ) -> list[SearchResult]:
        """
        Phase 2: Expand from vector hits via graph edges.

        Args:
            vector_hits: Results from Phase 1
            query_embedding: Query embedding for semantic scoring
            max_depth: Maximum traversal depth
            edge_types: Allowed edge types

        Returns:
            Graph expansion results
        """
        graph_results = []
        visited = {hit.memory_id for hit in vector_hits}

        expansion_tasks = [
            self._expand_from_node(
                node_id=hit.memory_id,
                query_embedding=query_embedding,
                current_depth=0,
                max_depth=max_depth,
                edge_types=edge_types,
                visited=visited,
                parent_score=hit.vector_score,
            )
            for hit in vector_hits
        ]

        expansion_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)

        for result in expansion_results:
            if isinstance(result, list):
                graph_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Graph expansion task failed: %s", result)

        return graph_results

    async def _expand_from_node(
        self,
        node_id: str,
        query_embedding: list[float],
        current_depth: int,
        max_depth: int,
        edge_types: list[str],
        visited: set[str],
        parent_score: float,
    ) -> list[SearchResult]:
        """
        Recursively expand from single node via reasoning edges.

        Uses HelixQL query getMemoryLogicalConnections to fetch all
        logical connections (IMPLIES, BECAUSE, CONTRADICTS, MEMORY_RELATION).

        Args:
            node_id: Starting node
            query_embedding: Query embedding
            current_depth: Current depth
            max_depth: Max depth
            edge_types: Allowed edge types (unused - gets all reasoning edges)
            visited: Set of visited nodes
            parent_score: Parent node's vector score

        Returns:
            Expansion results
        """
        logger.debug(
            f"🔍 _expand_from_node called: node={node_id}, depth={current_depth}/{max_depth}"
        )

        if current_depth >= max_depth:
            return []

        try:
            results = await self.client.execute_query(
                "getMemoryLogicalConnections", {"memory_id": node_id}
            )

            expansion_results = []

            edge_collections = [
                ("IMPLIES", results.get("implies_out", []), 1.0),
                ("IMPLIES", results.get("implies_in", []), 0.9),
                ("BECAUSE", results.get("because_out", []), 0.95),
                ("BECAUSE", results.get("because_in", []), 0.85),
                ("CONTRADICTS", results.get("contradicts_out", []), 0.8),
                ("CONTRADICTS", results.get("contradicts_in", []), 0.8),
                ("MEMORY_RELATION", results.get("relation_out", []), 0.7),
                ("MEMORY_RELATION", results.get("relation_in", []), 0.6),
            ]

            for _edge_type, memories, edge_weight in edge_collections:
                if not memories:
                    continue

                for memory in memories:
                    target_id = memory.get("memory_id")
                    if not target_id or target_id in visited:
                        continue

                    visited.add(target_id)

                    content = memory.get("content", "")
                    created_at = memory.get("created_at", "")

                    semantic_sim = 0.5

                    try:
                        temporal = calculate_temporal_freshness(created_at)
                    except Exception:
                        temporal = 0.5

                    graph_score = edge_weight * parent_score

                    combined_score = semantic_sim * 0.3 + graph_score * 0.5 + temporal * 0.2

                    expansion_results.append(
                        SearchResult(
                            memory_id=target_id,
                            content=content,
                            vector_score=semantic_sim,
                            graph_score=graph_score,
                            temporal_score=temporal,
                            combined_score=combined_score,
                            depth=current_depth + 1,
                            source="graph",
                            edge_path=[node_id, target_id],
                        )
                    )

            logger.debug(
                f"Expanded from {node_id}: found {len(expansion_results)} connected memories"
            )

            if current_depth + 1 < max_depth and expansion_results:
                top_neighbors = sorted(
                    expansion_results, key=lambda r: r.combined_score, reverse=True
                )[:3]

                child_tasks = [
                    self._expand_from_node(
                        node_id=neighbor.memory_id,
                        query_embedding=query_embedding,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                        edge_types=edge_types,
                        visited=visited,
                        parent_score=neighbor.combined_score,
                    )
                    for neighbor in top_neighbors
                ]

                child_results = await asyncio.gather(*child_tasks, return_exceptions=True)

                for child_result in child_results:
                    if isinstance(child_result, list):
                        expansion_results.extend(child_result)
                    elif isinstance(child_result, Exception):
                        logger.warning(f"Child expansion failed: {child_result}")

            return expansion_results

        except Exception as e:
            logger.exception("Failed to expand from node %s: %s", node_id, e)
            return []

    def _rank_and_filter(
        self,
        results: list[SearchResult],
        min_combined_score: float,
    ) -> list[SearchResult]:
        """
        Phase 3: Rank and filter results by combined score.

        Args:
            results: All results (vector + graph)
            min_combined_score: Minimum score threshold

        Returns:
            Filtered and ranked results
        """
        unique_results: dict[str, SearchResult] = {}
        for result in results:
            if (
                result.memory_id not in unique_results
                or result.combined_score > unique_results[result.memory_id].combined_score
            ):
                unique_results[result.memory_id] = result

        filtered = [r for r in unique_results.values() if r.combined_score >= min_combined_score]
        ranked = sorted(filtered, key=lambda r: r.combined_score, reverse=True)

        logger.debug(
            "Ranking: %d unique results, %d after filtering (min_score=%.2f)",
            len(unique_results),
            len(ranked),
            min_combined_score,
        )

        return ranked

    def get_stats(self) -> dict[str, Any]:
        """Get traversal statistics."""
        cache_stats = self._cache.stats()
        return {
            "cache_size": cache_stats["size"],
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": (
                cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])
                if (cache_stats["hits"] + cache_stats["misses"]) > 0
                else 0.0
            ),
        }
