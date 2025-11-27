"""
Search Module - Unified search with pluggable strategies.

This module provides:
- UnifiedSearchEngine: Facade for all search strategies
- OntoSearchStrategy: Ontology-aware search (vector + concepts + graph)
- SearchStrategy protocol: Interface for custom strategies
- Data models: SearchResult, ConceptMatch, OntoSearchConfig

Strategies:
- "smart" (default): SmartTraversalV2 (vector-first + graph expansion)
- "hybrid": Vector + BM25 hybrid search
- "onto": Ontology-aware search with concept matching

Usage:
    >>> from helixir.toolkit.mind_toolbox.search import (
    ...     UnifiedSearchEngine,
    ...     OntoSearchStrategy,
    ...     OntoSearchConfig,
    ... )
    >>>
    >>>
    >>> engine = UnifiedSearchEngine(client, ontology_manager)
    >>>
    >>>
    >>> results = await engine.search(
    ...     query="Python preferences",
    ...     query_embedding=embedding,
    ...     strategy="onto",
    ...     limit=10,
    ... )
    >>>
    >>>
    >>> strategy = OntoSearchStrategy(client, ontology_manager)
    >>> results = await strategy.search(query, embedding)
"""

from helixir.toolkit.mind_toolbox.search.engine import (
    HybridSearchAdapter,
    SmartSearchAdapter,
    UnifiedSearchEngine,
)
from helixir.toolkit.mind_toolbox.search.models import ConceptMatch, OntoSearchConfig, SearchResult
from helixir.toolkit.mind_toolbox.search.onto_search import OntoSearchStrategy
from helixir.toolkit.mind_toolbox.search.protocols import (
    ConceptMatcher,
    GraphExpander,
    ResultRanker,
    SearchStrategy,
)
from helixir.toolkit.mind_toolbox.search.query_processor import ProcessedQuery, QueryProcessor

__all__ = [
    "ConceptMatch",
    "ConceptMatcher",
    "GraphExpander",
    "HybridSearchAdapter",
    "OntoSearchConfig",
    "OntoSearchStrategy",
    "ProcessedQuery",
    "QueryProcessor",
    "ResultRanker",
    "SearchResult",
    "SearchStrategy",
    "SmartSearchAdapter",
    "UnifiedSearchEngine",
]
