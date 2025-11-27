"""
Smart Graph Traversal - intelligent memory graph traversal.

Key differences from simple BFS:
1. Priority Queue instead of FIFO queue (A* approach)
2. Scoring based on edge strength + semantic similarity + temporal freshness
3. Query-aware pruning - skip irrelevant branches
4. Early stopping - stop when found enough quality results
5. Parallel batch fetching - load nodes in batches
6. Intelligent caching - cache frequent subgraph traversals

Architecture:
    User Query -> SmartGraphTraverser -> PriorityQueue -> Batch Fetcher -> Scored Results

Performance targets:
- < 500ms for depth=2
- < 1000ms for depth=3
- Cache hit rate > 60%
- Relevance precision > 0.8
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import heapq
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
    "DERIVED_FROM": 0.85,
    "SUPPORTS": 0.8,
    "SIMILAR_TO": 0.75,
    "MEMORY_RELATION": 0.7,
    "EVOLVES_INTO": 0.7,
    "SUPERSEDES": 0.65,
    "EXTRACTED_ENTITY": 0.6,
    "MENTIONS": 0.55,
    "INSTANCE_OF": 0.5,
    "BELONGS_TO_CATEGORY": 0.45,
    "CONTRADICTS": 0.4,
    "REFUTES": 0.35,
    "OCCURRED_IN": 0.2,
    "CREATED_IN": 0.1,
}

DEFAULT_EDGE_WEIGHT = 0.5


def calculate_edge_score(
    edge_type: str,
    edge_strength: int,
    semantic_similarity: float,
    temporal_freshness: float,
    current_depth: int,
    max_depth: int,
) -> float:
    """
    Calculate edge priority for traversal.

    Formula: score = type_weight * (strength/100) * semantic_sim * temporal * depth_penalty

    Args:
        edge_type: Edge type (IMPLIES, BECAUSE, etc)
        edge_strength: Edge strength (0-100)
        semantic_similarity: Cosine similarity with query (0-1)
        temporal_freshness: Freshness (0-1, exp decay)
        current_depth: Current traversal depth
        max_depth: Maximum depth

    Returns:
        Score (0-1), higher = better
    """
    type_weight = EDGE_WEIGHTS.get(edge_type, DEFAULT_EDGE_WEIGHT)

    normalized_strength = edge_strength / 100.0 if edge_strength else 0.5

    depth_penalty = 1.0 - (current_depth / (max_depth + 1))

    score = (
        type_weight * normalized_strength * semantic_similarity * temporal_freshness * depth_penalty
    )

    return max(0.0, min(1.0, score))


def calculate_temporal_freshness(created_at: str, decay_days: float = 30.0) -> float:
    """
    Calculate temporal freshness (0-1).

    Uses exponential decay: freshness = exp(-days_old / decay_constant)

    Args:
        created_at: ISO timestamp
        decay_days: Decay constant (default 30 days = e^-1)

    Returns:
        Freshness score (0-1), 1 = very fresh
    """
    try:
        created = datetime.fromisoformat(created_at)
        now = datetime.now(UTC)
        days_old = (now - created).total_seconds() / 86400
        freshness = math.exp(-days_old / decay_days)
        return max(0.0, min(1.0, freshness))
    except (ValueError, AttributeError):
        return 0.5


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Formula: cos(theta) = (A·B) / (||A|| * ||B||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity (0-1), 1 = identical
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))

    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(b * b for b in vec2)

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    similarity = dot_product / (mag1 * mag2)

    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


@dataclass
class TraversalNode:
    """
    Node for graph traversal with priority.

    Used in heapq (min-heap), so needs __lt__ for comparison.
    """

    memory_id: str
    depth: int
    score: float
    parent_id: str | None = None
    edge_type: str | None = None

    def __lt__(self, other: TraversalNode) -> bool:
        """For heapq: lower score = higher priority (use negative score)."""
        return self.score > other.score


class SmartGraphTraverser:
    """
    Smart graph traversal with prioritization and query-aware pruning.

    Features:
    - Priority-based traversal (not all edges are equal)
    - Semantic similarity filtering
    - Temporal freshness weighting
    - Early stopping
    - Batch node fetching
    - Result caching
    """

    def __init__(
        self,
        client: HelixDBClient,
        cache_size: int = 500,
        cache_ttl: int = 300,
    ):
        """
        Initialize SmartGraphTraverser.

        Args:
            client: HelixDB client for graph queries
            cache_size: LRU cache size for subgraphs
            cache_ttl: TTL for cache entries (seconds)
        """
        self.client = client

        self._traversal_cache: LRUCache[str, dict[str, Any]] = LRUCache(
            maxsize=cache_size,
            ttl=cache_ttl,
        )

        logger.info(
            "SmartGraphTraverser initialized (cache_size=%d, ttl=%ds)",
            cache_size,
            cache_ttl,
        )

    async def traverse(
        self,
        start_memory_id: str,
        query_embedding: list[float],
        max_depth: int = 2,
        max_results: int = 10,
        min_relevance: float = 0.3,
        edge_types: list[str] | None = None,
        *,
        include_start: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Smart graph traversal with prioritization.

        Args:
            start_memory_id: Starting node
            query_embedding: Query embedding for relevance scoring
            max_depth: Maximum traversal depth
            max_results: Maximum results (early stopping)
            min_relevance: Minimum relevance score (pruning threshold)
            edge_types: Whitelist edge types (None = all)
            include_start: Include start node in results

        Returns:
            List of scored memories [{memory_id, score, depth, path}, ...]
            Sorted by score descending
        """
        start_time = time.time()
        float_event(
            "smart_traversal.start",
            start_id=start_memory_id[:8],
            max_depth=max_depth,
            max_results=max_results,
        )

        cache_key = self._make_cache_key(
            start_memory_id,
            query_embedding,
            max_depth,
            max_results,
            edge_types,
        )

        if cached := self._traversal_cache.get(cache_key):
            logger.debug("Cache HIT for traversal %s", start_memory_id[:8])
            float_event("smart_traversal.cache_hit")
            return cached["results"]

        priority_queue: list[TraversalNode] = []
        visited: set[str] = set()
        results: list[dict[str, Any]] = []

        if include_start:
            start_node = TraversalNode(
                memory_id=start_memory_id,
                depth=0,
                score=1.0,
            )
            heapq.heappush(priority_queue, start_node)
        else:
            await self._expand_node(
                memory_id=start_memory_id,
                current_depth=0,
                max_depth=max_depth,
                query_embedding=query_embedding,
                min_relevance=min_relevance,
                edge_types=edge_types,
                visited=visited,
                priority_queue=priority_queue,
            )
            visited.add(start_memory_id)

        nodes_expanded = 0

        while priority_queue and len(results) < max_results:
            current = heapq.heappop(priority_queue)

            if current.memory_id in visited:
                continue

            visited.add(current.memory_id)
            nodes_expanded += 1

            results.append(
                {
                    "memory_id": current.memory_id,
                    "score": current.score,
                    "depth": current.depth,
                    "parent_id": current.parent_id,
                    "edge_type": current.edge_type,
                }
            )

            if len(results) >= max_results:
                logger.debug("Early stopping: found %d results", len(results))
                break

            if current.depth < max_depth:
                await self._expand_node(
                    memory_id=current.memory_id,
                    current_depth=current.depth,
                    max_depth=max_depth,
                    query_embedding=query_embedding,
                    min_relevance=min_relevance,
                    edge_types=edge_types,
                    visited=visited,
                    priority_queue=priority_queue,
                )

        results.sort(key=lambda x: x["score"], reverse=True)

        self._traversal_cache.set(cache_key, {"results": results})

        elapsed_ms = (time.time() - start_time) * 1000
        float_event(
            "smart_traversal.complete",
            elapsed_ms=elapsed_ms,
            results=len(results),
            nodes_expanded=nodes_expanded,
            cache_miss=True,
        )

        logger.info(
            "✅ Smart traversal complete: %d results, %d nodes expanded, %.1fms",
            len(results),
            nodes_expanded,
            elapsed_ms,
        )

        return results

    async def _expand_node(
        self,
        memory_id: str,
        current_depth: int,
        max_depth: int,
        query_embedding: list[float],
        min_relevance: float,
        edge_types: list[str] | None,
        visited: set[str],
        priority_queue: list[TraversalNode],
    ) -> int:
        """
        Expand node - add its neighbors to priority queue.

        Loads edges, calculates scores, filters by min_relevance.

        Returns:
            Number of edges expanded
        """
        try:
            edges_query = """
            MATCH (start:Memory {memory_id: $memory_id})-[edge]->(target:Memory)
            WHERE edge.valid_until IS NULL OR edge.valid_until > NOW()
            RETURN edge, target.memory_id as target_id,
                   target.created_at as created_at,
                   target.data as embedding,
                   type(edge) as edge_type,
                   COALESCE(edge.strength, edge.probability, edge.confidence, 50) as strength
            """

            result = await self.client.execute_query(
                "custom_query",
                {
                    "query": edges_query,
                    "params": {"memory_id": memory_id},
                },
            )

            if not result or "data" not in result:
                return 0

            edges = result["data"]
            if not edges:
                return 0

            if edge_types:
                edges = [e for e in edges if e.get("edge_type") in edge_types]

            scoring_tasks = [
                self._score_edge(
                    edge=edge,
                    query_embedding=query_embedding,
                    current_depth=current_depth,
                    max_depth=max_depth,
                )
                for edge in edges
            ]

            scored_edges = await asyncio.gather(*scoring_tasks, return_exceptions=True)

            added = 0
            for edge, score_result in zip(edges, scored_edges, strict=False):
                if isinstance(score_result, Exception):
                    logger.debug("Scoring error for edge: %s", score_result)
                    continue

                target_id, score = score_result

                if target_id in visited:
                    continue

                if score < min_relevance:
                    continue

                node = TraversalNode(
                    memory_id=target_id,
                    depth=current_depth + 1,
                    score=score,
                    parent_id=memory_id,
                    edge_type=edge.get("edge_type"),
                )

                heapq.heappush(priority_queue, node)
                added += 1

        except (KeyError, ValueError) as e:
            logger.warning("Failed to expand node %s: %s", memory_id[:8], e)

        return added

    async def _score_edge(
        self,
        edge: dict[str, Any],
        query_embedding: list[float],
        current_depth: int,
        max_depth: int,
    ) -> tuple[str, float]:
        """
        Score single edge for priority.

        Returns:
            (target_id, score)
        """
        target_id = edge["target_id"]
        edge_type = edge["edge_type"]
        strength = edge.get("strength", 50)
        created_at = edge.get("created_at", "")
        embedding = edge.get("embedding", [])

        semantic_sim = (
            cosine_similarity(query_embedding, embedding) if embedding else DEFAULT_EDGE_WEIGHT
        )

        temporal = calculate_temporal_freshness(created_at)

        score = calculate_edge_score(
            edge_type=edge_type,
            edge_strength=strength,
            semantic_similarity=semantic_sim,
            temporal_freshness=temporal,
            current_depth=current_depth,
            max_depth=max_depth,
        )

        return (target_id, score)

    def _make_cache_key(
        self,
        start_id: str,
        query_embedding: list[float],
        max_depth: int,
        max_results: int,
        edge_types: list[str] | None,
    ) -> str:
        """Create cache key for traversal parameters."""
        embedding_hash = hash(tuple(query_embedding[:10]))

        edge_types_str = ",".join(sorted(edge_types)) if edge_types else "all"

        return f"{start_id}:{embedding_hash}:d{max_depth}:r{max_results}:{edge_types_str}"

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._traversal_cache.stats()

    def clear_cache(self) -> None:
        """Clear traversal cache."""
        self._traversal_cache.clear()
        logger.info("Traversal cache cleared")
