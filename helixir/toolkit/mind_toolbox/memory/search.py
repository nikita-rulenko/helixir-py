"""Search engine for memory retrieval using vector, BM25, hybrid, and smart graph search."""

import asyncio
from datetime import UTC, datetime
import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

from helixir.core.cache import LRUCache
from helixir.toolkit.mind_toolbox.memory.models import Memory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helixir.core.client import HelixDBClient
    from helixir.toolkit.mind_toolbox.memory.smart_traversal_v2 import SmartTraversalV2

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result with score and metadata."""

    def __init__(
        self,
        memory: Memory,
        score: float,
        method: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize SearchResult.

        Args:
            memory: Memory object
            score: Relevance score (0.0-1.0)
            method: Search method used (vector, bm25, hybrid)
            metadata: Additional metadata
        """
        self.memory = memory
        self.score = score
        self.method = method
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        """String representation."""
        return f"SearchResult(memory_id={self.memory.memory_id[:8]}..., score={self.score:.3f}, method={self.method})"


class SearchEngine:
    """
    Search engine for memory retrieval.

    Supports:
    - Vector search (semantic similarity)
    - BM25 search (keyword-based)
    - Hybrid search (combination)
    - Smart graph search V2 (vector-first + graph expansion) 🚀
    """

    def __init__(
        self,
        client: HelixDBClient | None = None,
        cache_size: int = 500,
        cache_ttl: int = 300,
        *,
        enable_smart_traversal_v2: bool = True,
    ) -> None:
        """
        Initialize SearchEngine.

        Args:
            client: HelixDBClient instance (optional for E2E tests)
            cache_size: Vector search cache size (default 500)
            cache_ttl: Cache TTL in seconds (default 300 = 5 min)
            enable_smart_traversal_v2: Enable SmartTraversalV2 (vector-first)
        """
        self.client = client

        self._vector_cache = LRUCache[str, list[SearchResult]](
            maxsize=cache_size, ttl=float(cache_ttl)
        )

        self._smart_traversal_v2: SmartTraversalV2 | None = None
        if enable_smart_traversal_v2 and client:
            from helixir.toolkit.mind_toolbox.memory.smart_traversal_v2 import SmartTraversalV2

            self._smart_traversal_v2 = SmartTraversalV2(
                client=client,
                cache_size=cache_size,
                cache_ttl=cache_ttl,
            )
            logger.info("SmartTraversalV2 enabled")

        self._stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
        }

    def _make_cache_key(
        self,
        query: str,
        user_id: str | None,
        limit: int,
        min_score: float,
    ) -> str:
        """
        Create cache key for vector search.

        Args:
            query: Search query
            user_id: Optional user ID filter
            limit: Maximum results
            min_score: Minimum score

        Returns:
            Cache key (hash)
        """
        key_data = f"{query}|{user_id}|{limit}|{min_score}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase, no stopwords)
        """
        tokens = re.findall(r"\b\w+\b", text.lower())

        return [t for t in tokens if t not in self._stopwords and len(t) > 2]

    def _calculate_bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        avg_doc_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """
        Calculate BM25 score for a document.

        Args:
            query_tokens: Tokenized query
            doc_tokens: Tokenized document
            avg_doc_length: Average document length in corpus
            k1: BM25 parameter (term frequency saturation)
            b: BM25 parameter (length normalization)

        Returns:
            BM25 score
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_length = len(doc_tokens)
        score = 0.0

        doc_tf: dict[str, int] = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1

        for query_term in query_tokens:
            if query_term in doc_tf:
                tf = doc_tf[query_term]

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

                score += numerator / denominator

        if len(query_tokens) > 0:
            score /= len(query_tokens)

        return min(score, 1.0)

    async def vector_search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.5,
        *,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """
        Perform vector (semantic) search.

        Args:
            query: Search query
            user_id: Optional user ID filter
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0-1.0)
            use_cache: Whether to use cache (default True)

        Returns:
            List of SearchResult objects
        """
        if use_cache:
            cache_key = self._make_cache_key(query, user_id, limit, min_score)
            cached = self._vector_cache.get(cache_key)
            if cached is not None:
                logger.debug("Vector search cache HIT for: %s", query[:50])
                return cached

        try:
            result = await self.client.execute_query(
                "vectorSearch",
                {
                    "query": query,
                    "user_id": user_id or "",
                    "limit": limit,
                    "min_score": min_score,
                },
            )

            results = []
            if result and "memories" in result:
                for item in result["memories"]:
                    memory = Memory(
                        memory_id=item["memory_id"],
                        content=item["content"],
                        memory_type=item["memory_type"],
                        user_id=item["user_id"],
                        certainty=item.get("certainty", 100),
                        importance=item.get("importance", 50),
                        created_at=datetime.fromisoformat(item["created_at"]),
                        updated_at=datetime.fromisoformat(item["updated_at"]),
                        valid_from=datetime.fromisoformat(item["valid_from"]),
                        valid_until=(
                            datetime.fromisoformat(item["valid_until"])
                            if item.get("valid_until")
                            else None
                        ),
                    )

                    search_result = SearchResult(
                        memory=memory,
                        score=item.get("similarity_score", 0.0),
                        method="vector",
                        metadata={"embedding_distance": item.get("distance", 0.0)},
                    )
                    results.append(search_result)

            if use_cache:
                cache_key = self._make_cache_key(query, user_id, limit, min_score)
                self._vector_cache.set(cache_key, results)

            logger.info("Vector search returned %d results", len(results))
            return results

        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    async def bm25_search(
        self,
        query: str,
        memories: Sequence[Memory],
        limit: int = 10,
        min_score: float = 0.1,
    ) -> list[SearchResult]:
        """
        Perform BM25 (keyword) search.

        Args:
            query: Search query
            memories: List of memories to search
            limit: Maximum number of results
            min_score: Minimum BM25 score

        Returns:
            List of SearchResult objects
        """
        if not memories:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        total_length = sum(len(self._tokenize(m.content)) for m in memories)
        avg_doc_length = total_length / len(memories) if memories else 0

        scored_results = []
        for memory in memories:
            doc_tokens = self._tokenize(memory.content)
            score = self._calculate_bm25_score(query_tokens, doc_tokens, avg_doc_length)

            if score >= min_score:
                search_result = SearchResult(
                    memory=memory,
                    score=score,
                    method="bm25",
                    metadata={
                        "query_tokens": query_tokens,
                        "doc_length": len(doc_tokens),
                    },
                )
                scored_results.append(search_result)

        scored_results.sort(key=lambda x: x.score, reverse=True)

        logger.info("BM25 search returned %d results", len(scored_results[:limit]))
        return scored_results[:limit]

    async def hybrid_search(
        self,
        query: str,
        user_id: str | None = None,
        memories: Sequence[Memory] | None = None,
        limit: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[SearchResult]:
        """
        Perform hybrid search (vector + BM25) with PARALLEL execution.

        Args:
            query: Search query
            user_id: Optional user ID filter
            memories: Optional pre-fetched memories for BM25
            limit: Maximum number of results
            vector_weight: Weight for vector search (0.0-1.0)
            bm25_weight: Weight for BM25 search (0.0-1.0)

        Returns:
            List of SearchResult objects with combined scores
        """
        total_weight = vector_weight + bm25_weight
        vector_weight /= total_weight
        bm25_weight /= total_weight

        tasks = [self.vector_search(query, user_id, limit=limit * 2, min_score=0.3)]

        if memories:
            tasks.append(self.bm25_search(query, memories, limit=limit * 2, min_score=0.05))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_results = results[0] if not isinstance(results[0], Exception) else []
        bm25_results = (
            results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
        )

        combined_scores: dict[str, tuple[Memory, float, dict[str, Any]]] = {}

        for result in vector_results:
            memory_id = result.memory.memory_id
            score = result.score * vector_weight
            combined_scores[memory_id] = (result.memory, score, {"vector": result.score})

        for result in bm25_results:
            memory_id = result.memory.memory_id
            score = result.score * bm25_weight

            if memory_id in combined_scores:
                memory, existing_score, metadata = combined_scores[memory_id]
                metadata["bm25"] = result.score
                combined_scores[memory_id] = (memory, existing_score + score, metadata)
            else:
                combined_scores[memory_id] = (result.memory, score, {"bm25": result.score})

        final_results = [
            SearchResult(
                memory=memory,
                score=score,
                method="hybrid",
                metadata=metadata,
            )
            for memory, score, metadata in combined_scores.values()
        ]

        final_results.sort(key=lambda x: x.score, reverse=True)

        logger.info("Hybrid search returned %d results", len(final_results[:limit]))
        return final_results[:limit]

    async def smart_graph_search_v2(
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
        🚀 NEW: Smart graph search V2 (vector-first + graph expansion).

        Two-phase search:
        1. Vector search: Find semantically similar nodes (with temporal filtering)
        2. Graph expansion: Load logical connections from hits
        3. Combined ranking: Semantic + Graph + Temporal

        Args:
            query: User query text
            query_embedding: Query embedding vector
            user_id: Optional user filter
            vector_top_k: Number of vector search results (Phase 1)
            graph_depth: Max graph expansion depth (Phase 2)
            min_vector_score: Min score for vector hits
            min_combined_score: Min score for final results
            edge_types: Whitelist edge types (None = default logical types)
            temporal_cutoff: Only return memories created after this datetime (None = no filter)

        Returns:
            List of SearchResult objects with combined scoring
        """
        if not self._smart_traversal_v2:
            logger.warning("SmartTraversalV2 not enabled, falling back to vector search")
            return await self.vector_search(
                query=query,
                user_id=user_id,
                limit=vector_top_k,
                min_score=min_vector_score,
            )

        logger.info(
            "SmartGraphSearchV2: query=%s, vector_top_k=%d, graph_depth=%d",
            query[:50],
            vector_top_k,
            graph_depth,
        )

        traversal_results = await self._smart_traversal_v2.search(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            vector_top_k=vector_top_k,
            graph_depth=graph_depth,
            min_vector_score=min_vector_score,
            min_combined_score=min_combined_score,
            edge_types=edge_types,
            temporal_cutoff=temporal_cutoff,
        )

        search_results = []
        for result in traversal_results:
            memory = Memory(
                memory_id=result.memory_id,
                user_id=user_id or "unknown",
                content=result.content,
                memory_type="fact",
                created_at=datetime.now(UTC),
                agent_id=None,
            )

            search_result = SearchResult(
                memory=memory,
                score=result.combined_score,
                method="smart_graph_v2",
                metadata={
                    "vector_score": result.vector_score,
                    "graph_score": result.graph_score,
                    "temporal_score": result.temporal_score,
                    "depth": result.depth,
                    "source": result.source,
                },
            )
            search_results.append(search_result)

        logger.info("SmartGraphSearchV2 returned %d results", len(search_results))
        return search_results

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get vector search cache statistics.

        Returns:
            Cache stats dict
        """
        return self._vector_cache.stats

    def clear_cache(self) -> None:
        """Clear vector search cache."""
        self._vector_cache.clear()
        logger.info("Vector search cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        return "SearchEngine(methods=['vector', 'bm25', 'hybrid'])"
