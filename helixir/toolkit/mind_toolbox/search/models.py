"""
Search Models.

Data classes for search operations:
- SearchResult: Single search result with scores
- ConceptMatch: Concept classification result
- OntoSearchConfig: Configuration for OntoSearch strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptMatch:
    """
    Result of concept matching.

    Represents a match between text and an ontology concept.

    Attributes:
        concept_id: Unique concept identifier (e.g., "Preference", "Skill")
        confidence: Match confidence (0.0-1.0)
        link_type: Type of link (INSTANCE_OF, BELONGS_TO_CATEGORY)
        match_type: How the match was found (mapper, classifier, processor_hint)

    Example:
        >>> match = ConceptMatch("Preference", 0.9, "INSTANCE_OF")
        >>> print(f"{match.concept_id}: {match.confidence:.0%}")
        Preference: 90%
    """

    concept_id: str
    confidence: float
    link_type: str = "INSTANCE_OF"
    match_type: str = "mapper"

    def __repr__(self) -> str:
        return f"ConceptMatch({self.concept_id}, {self.confidence:.2f}, {self.match_type})"


@dataclass
class SearchResult:
    """
    Single search result with multi-dimensional scoring.

    Combines scores from different sources:
    - vector_score: Semantic similarity from embedding search
    - graph_score: Relevance from graph connections
    - concept_score: Ontology concept overlap
    - temporal_score: Freshness/recency

    Attributes:
        memory_id: Memory identifier
        content: Memory content text
        score: Final combined score (0.0-1.0)
        method: Search method used (smart, hybrid, onto)
        vector_score: Semantic similarity score
        graph_score: Graph connection score
        concept_score: Concept overlap score
        temporal_score: Temporal freshness score
        depth: Distance from query (0 = direct hit)
        source: Result source (vector, graph, concept)
        metadata: Additional metadata

    Example:
        >>> result = SearchResult(
        ...     memory_id="mem_001",
        ...     content="I love Python",
        ...     score=0.85,
        ...     method="onto",
        ...     vector_score=0.9,
        ...     concept_score=0.8,
        ... )
    """

    memory_id: str
    content: str
    score: float
    method: str
    vector_score: float = 0.0
    graph_score: float = 0.0
    concept_score: float = 0.0
    tag_score: float = 0.0
    temporal_score: float = 0.0
    depth: int = 0
    source: str = "vector"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.memory_id[:8]}..., "
            f"score={self.score:.3f}, method={self.method})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
            "method": self.method,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "concept_score": self.concept_score,
            "tag_score": self.tag_score,
            "temporal_score": self.temporal_score,
            "depth": self.depth,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class OntoSearchConfig:
    """
    Configuration for OntoSearch strategy.

    Controls weights for combined scoring and search parameters.
    Can be auto-configured from SearchMode for seamless integration.

    Attributes:
        vector_weight: Weight for semantic similarity (default: 0.35)
        graph_weight: Weight for graph connections (default: 0.25)
        concept_weight: Weight for concept overlap (default: 0.25)
        tag_weight: Weight for tag matching (default: 0.05)
        temporal_weight: Weight for freshness (default: 0.1)
        min_score: Minimum combined score threshold (default: 0.3)
        graph_depth: Graph expansion depth (default: 1)
        vector_top_k: Number of vector search results (default: 20)
        concept_boost: Bonus for exact concept match (default: 0.15)
        temporal_decay_days: Half-life for temporal decay (default: 30)

    Example:
        >>>
        >>> config = OntoSearchConfig(vector_weight=0.5, concept_weight=0.3)
        >>>
        >>>
        >>> config = OntoSearchConfig.from_mode("contextual")

    Note:
        Weights should sum to 1.0 for normalized scoring.
        They will be auto-normalized if they don't.
    """

    vector_weight: float = 0.35
    graph_weight: float = 0.25
    concept_weight: float = 0.25
    tag_weight: float = 0.05
    temporal_weight: float = 0.1

    min_score: float = 0.3
    min_vector_score: float = 0.5
    min_concept_confidence: float = 0.3

    graph_depth: int = 1
    vector_top_k: int = 20
    max_concepts_per_query: int = 3

    temporal_filter_hours: float | None = None

    concept_boost: float = 0.15
    temporal_decay_days: float = 30.0

    def __post_init__(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (
            self.vector_weight
            + self.graph_weight
            + self.concept_weight
            + self.tag_weight
            + self.temporal_weight
        )
        if abs(total - 1.0) > 0.001:
            self.vector_weight /= total
            self.graph_weight /= total
            self.concept_weight /= total
            self.tag_weight /= total
            self.temporal_weight /= total

    @classmethod
    def from_mode(cls, mode: str) -> OntoSearchConfig:
        """
        Create config optimized for a SearchMode.

        Mode-specific tuning:
        - RECENT: Fast, vector-heavy, minimal concept overhead
        - CONTEXTUAL: Balanced, moderate concept boost
        - DEEP: Strong concept/tag boost, deep graph
        - FULL: Maximum concept/graph coverage

        Args:
            mode: Search mode string ("recent", "contextual", "deep", "full")

        Returns:
            OntoSearchConfig tuned for the mode
        """
        mode_configs = {
            "recent": cls(
                vector_weight=0.45,
                graph_weight=0.10,
                concept_weight=0.15,
                tag_weight=0.05,
                temporal_weight=0.25,
                min_score=0.3,
                min_vector_score=0.5,
                min_concept_confidence=0.4,
                graph_depth=1,
                vector_top_k=15,
                max_concepts_per_query=2,
                temporal_filter_hours=4.0,
                concept_boost=0.1,
                temporal_decay_days=1.0,
            ),
            "contextual": cls(
                vector_weight=0.35,
                graph_weight=0.20,
                concept_weight=0.25,
                tag_weight=0.10,
                temporal_weight=0.10,
                min_score=0.3,
                min_vector_score=0.5,
                min_concept_confidence=0.3,
                graph_depth=2,
                vector_top_k=20,
                max_concepts_per_query=3,
                temporal_filter_hours=30 * 24,
                concept_boost=0.15,
                temporal_decay_days=30.0,
            ),
            "deep": cls(
                vector_weight=0.25,
                graph_weight=0.25,
                concept_weight=0.30,
                tag_weight=0.10,
                temporal_weight=0.10,
                min_score=0.25,
                min_vector_score=0.4,
                min_concept_confidence=0.25,
                graph_depth=3,
                vector_top_k=30,
                max_concepts_per_query=5,
                temporal_filter_hours=90 * 24,
                concept_boost=0.20,
                temporal_decay_days=90.0,
            ),
            "full": cls(
                vector_weight=0.20,
                graph_weight=0.25,
                concept_weight=0.35,
                tag_weight=0.10,
                temporal_weight=0.10,
                min_score=0.2,
                min_vector_score=0.3,
                min_concept_confidence=0.2,
                graph_depth=4,
                vector_top_k=50,
                max_concepts_per_query=5,
                temporal_filter_hours=None,
                concept_boost=0.25,
                temporal_decay_days=365.0,
            ),
        }

        return mode_configs.get(mode.lower(), mode_configs["contextual"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "concept_weight": self.concept_weight,
            "tag_weight": self.tag_weight,
            "temporal_weight": self.temporal_weight,
            "min_score": self.min_score,
            "min_vector_score": self.min_vector_score,
            "graph_depth": self.graph_depth,
            "vector_top_k": self.vector_top_k,
            "temporal_filter_hours": self.temporal_filter_hours,
            "concept_boost": self.concept_boost,
            "temporal_decay_days": self.temporal_decay_days,
        }
