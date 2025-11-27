"""
Search Strategy Protocols.

Defines interfaces for pluggable search strategies.
All search strategies must implement SearchStrategy protocol.

Usage:
    class MyCustomSearch:
        async def search(self, query, query_embedding, ...) -> list[SearchResult]:
            ...

    strategy: SearchStrategy = MyCustomSearch()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from helixir.toolkit.mind_toolbox.search.models import ConceptMatch, SearchResult


@runtime_checkable
class SearchStrategy(Protocol):
    """
    Protocol for search strategies.

    All search strategies (smart, hybrid, onto) must implement this interface.
    This enables strategy pattern - swap strategies at runtime.

    Example:
        >>> class OntoSearchStrategy:
        ...     async def search(self, query, query_embedding, user_id, limit):
        ...         return results
        >>>
        >>> strategy: SearchStrategy = OntoSearchStrategy(...)
        >>> results = await strategy.search("Python", embedding, "user123", 10)
    """

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Execute search and return ranked results.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            user_id: Optional user filter
            limit: Maximum number of results
            **kwargs: Strategy-specific parameters

        Returns:
            List of SearchResult objects, ranked by relevance
        """
        ...


@runtime_checkable
class ConceptMatcher(Protocol):
    """
    Protocol for concept matching.

    Maps text to ontology concepts. Can be keyword-based or semantic.

    Example:
        >>> matcher: ConceptMatcher = KeywordConceptMatcher(ontology)
        >>> concepts = matcher.match("I love Python programming")
        >>>
    """

    def match(
        self,
        text: str,
        min_confidence: float = 0.3,
    ) -> list[ConceptMatch]:
        """
        Match text to ontology concepts.

        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of ConceptMatch objects with confidence scores
        """
        ...


@runtime_checkable
class GraphExpander(Protocol):
    """
    Protocol for graph expansion.

    Expands from seed nodes via graph edges (IMPLIES, BECAUSE, etc.).

    Example:
        >>> expander: GraphExpander = LogicalGraphExpander(client)
        >>> neighbors = await expander.expand(["mem_001"], depth=1)
    """

    async def expand(
        self,
        node_ids: list[str],
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Expand from seed nodes via graph edges.

        Args:
            node_ids: Starting node IDs
            depth: Maximum traversal depth
            edge_types: Allowed edge types (None = all logical types)

        Returns:
            List of neighbor nodes with edge metadata
        """
        ...


@runtime_checkable
class ResultRanker(Protocol):
    """
    Protocol for result ranking.

    Combines multiple scores (vector, graph, concept, temporal) into final ranking.

    Example:
        >>> ranker: ResultRanker = CombinedScoreRanker(weights)
        >>> ranked = ranker.rank(results)
    """

    def rank(
        self,
        results: list[SearchResult],
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        Rank results by combined score.

        Args:
            results: Unranked search results
            min_score: Minimum score threshold

        Returns:
            Ranked and filtered results
        """
        ...
