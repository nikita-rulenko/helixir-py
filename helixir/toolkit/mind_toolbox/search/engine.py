"""
Unified Search Engine.

Facade for all search strategies with pluggable architecture.

Strategies:
- "smart" (default): SmartTraversalV2 (vector + graph)
- "hybrid": Vector + BM25
- "onto": OntoSearch (vector + concepts + graph)

Usage:
    >>> engine = UnifiedSearchEngine(client, ontology_manager)
    >>> results = await engine.search(
    ...     query="Python preferences",
    ...     query_embedding=embedding,
    ...     strategy="onto",
    ...     limit=10,
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from helixir.toolkit.mind_toolbox.search.models import OntoSearchConfig, SearchResult
from helixir.toolkit.mind_toolbox.search.onto_search import OntoSearchStrategy

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.toolkit.mind_toolbox.ontology import OntologyManager
    from helixir.toolkit.mind_toolbox.search.protocols import SearchStrategy

logger = logging.getLogger(__name__)


class SmartSearchAdapter:
    """
    Adapter for existing SmartTraversalV2.

    Wraps SmartTraversalV2 to implement SearchStrategy protocol.
    """

    def __init__(self, client: HelixDBClient) -> None:
        self.client = client
        self._smart_traversal = None

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Execute SmartTraversalV2 search."""
        if self._smart_traversal is None:
            from helixir.toolkit.mind_toolbox.memory.smart_traversal_v2 import SmartTraversalV2

            self._smart_traversal = SmartTraversalV2(client=self.client)

        graph_depth = kwargs.get("graph_depth", 2)
        min_vector_score = kwargs.get("min_vector_score", 0.5)
        min_combined_score = kwargs.get("min_combined_score", 0.3)
        temporal_cutoff = kwargs.get("temporal_cutoff")

        results = await self._smart_traversal.search(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            vector_top_k=limit * 2,
            graph_depth=graph_depth,
            min_vector_score=min_vector_score,
            min_combined_score=min_combined_score,
            temporal_cutoff=temporal_cutoff,
        )

        search_results: list[SearchResult] = []
        for r in results[:limit]:
            search_results.append(
                SearchResult(
                    memory_id=r.memory_id,
                    content=r.content,
                    score=r.combined_score,
                    method="smart",
                    vector_score=r.vector_score,
                    graph_score=r.graph_score,
                    temporal_score=r.temporal_score,
                    depth=r.depth,
                    source=r.source,
                    metadata=r.metadata or {},
                )
            )

        return search_results


class HybridSearchAdapter:
    """
    Adapter for existing hybrid search (vector + BM25).

    Wraps SearchEngine.hybrid_search to implement SearchStrategy protocol.
    """

    def __init__(self, client: HelixDBClient) -> None:
        self.client = client
        self._search_engine = None

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Execute hybrid search."""
        if self._search_engine is None:
            from helixir.toolkit.mind_toolbox.memory.search import SearchEngine

            self._search_engine = SearchEngine(client=self.client)

        vector_weight = kwargs.get("vector_weight", 0.6)
        bm25_weight = kwargs.get("bm25_weight", 0.4)

        results = await self._search_engine.hybrid_search(
            query=query,
            user_id=user_id,
            limit=limit,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        search_results: list[SearchResult] = []
        for r in results:
            search_results.append(
                SearchResult(
                    memory_id=r.memory.memory_id,
                    content=r.memory.content,
                    score=r.score,
                    method="hybrid",
                    vector_score=r.metadata.get("vector", 0.0),
                    depth=0,
                    source="hybrid",
                    metadata=r.metadata,
                )
            )

        return search_results


class UnifiedSearchEngine:
    """
    Unified search engine with pluggable strategies.

    Provides single interface for all search methods:
    - "smart": SmartTraversalV2 (vector-first + graph expansion)
    - "hybrid": Vector + BM25 hybrid search
    - "onto": Ontology-aware search (vector + concepts + graph)

    Attributes:
        client: HelixDBClient instance
        ontology: OntologyManager instance (optional, required for "onto")
        default_strategy: Default strategy name

    Example:
        >>> engine = UnifiedSearchEngine(client, ontology)
        >>>
        >>>
        >>> results = await engine.search(query, embedding)
        >>>
        >>>
        >>> results = await engine.search(query, embedding, strategy="onto")
        >>>
        >>>
        >>> engine.register_strategy("custom", MyCustomStrategy())
    """

    def __init__(
        self,
        client: HelixDBClient,
        ontology_manager: OntologyManager | None = None,
        default_strategy: str = "smart",
        onto_config: OntoSearchConfig | None = None,
    ) -> None:
        """
        Initialize UnifiedSearchEngine.

        Args:
            client: HelixDBClient instance
            ontology_manager: OntologyManager (required for "onto" strategy)
            default_strategy: Default strategy name
            onto_config: Configuration for OntoSearchStrategy
        """
        self.client = client
        self.ontology = ontology_manager
        self.default_strategy = default_strategy

        self._strategies: dict[str, SearchStrategy] = {}

        self._strategies["smart"] = SmartSearchAdapter(client)
        self._strategies["hybrid"] = HybridSearchAdapter(client)

        if ontology_manager:
            self._strategies["onto"] = OntoSearchStrategy(
                client=client,
                ontology_manager=ontology_manager,
                config=onto_config,
            )
            logger.info("OntoSearchStrategy registered")
        else:
            logger.info("OntoSearchStrategy not available (no ontology manager)")

        logger.info(
            "UnifiedSearchEngine initialized: strategies=%s, default=%s",
            list(self._strategies.keys()),
            default_strategy,
        )

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        strategy: str | None = None,
        mode: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Execute search with specified strategy.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            user_id: Optional user filter
            strategy: Strategy name (None = use default)
            mode: Search mode for OntoSearch ("recent", "contextual", "deep", "full")
            limit: Maximum number of results
            **kwargs: Strategy-specific parameters

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If strategy is not registered
        """
        strategy_name = strategy or self.default_strategy

        if strategy_name not in self._strategies:
            available = list(self._strategies.keys())
            msg = f"Unknown strategy: {strategy_name}. Available: {available}"
            raise ValueError(msg)

        logger.info(
            "Executing search: strategy=%s, mode=%s, query='%s'",
            strategy_name,
            mode or "default",
            query[:50],
        )

        if strategy_name == "onto" and mode is not None:
            return await self._strategies[strategy_name].search(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                mode=mode,
                **kwargs,
            )

        return await self._strategies[strategy_name].search(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            **kwargs,
        )

    def register_strategy(self, name: str, strategy: SearchStrategy) -> None:
        """
        Register custom search strategy.

        Args:
            name: Strategy name
            strategy: Strategy instance (must implement SearchStrategy)
        """
        self._strategies[name] = strategy
        logger.info("Registered custom strategy: %s", name)

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        return list(self._strategies.keys())

    def has_strategy(self, name: str) -> bool:
        """Check if strategy is available."""
        return name in self._strategies

    def __repr__(self) -> str:
        return (
            f"UnifiedSearchEngine(strategies={list(self._strategies.keys())}, "
            f"default={self.default_strategy})"
        )
