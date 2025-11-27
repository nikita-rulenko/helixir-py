"""
Search modes for memory retrieval with token cost control.

This module defines different search strategies optimized for different use cases
and token budgets. Each mode has different depth, breadth, and cost characteristics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class SearchMode(str, Enum):
    """
    Search modes controlling depth and cost of memory retrieval.

    Modes:
        RECENT: Only recent memories (last N hours), minimal graph traversal
            - Token cost: LOW (⚡)
            - Use case: Quick recent context, conversational queries
            - Default limits: 10 results, depth=1, last 4 hours

        CONTEXTUAL: Recent memories with moderate graph traversal
            - Token cost: MEDIUM (⚡⚡)
            - Use case: Balanced context with related memories
            - Default limits: 20 results, depth=2, last 30 days

        DEEP: Deep graph traversal with temporal expansion
            - Token cost: HIGH (⚡⚡⚡)
            - Use case: Complex queries requiring extensive context
            - Default limits: 50 results, depth=3, last 90 days

        FULL: Complete search across all memories and full graph
            - Token cost: VERY HIGH (⚡⚡⚡⚡)
            - Use case: Comprehensive analysis, data export
            - Default limits: 100 results, depth=4, no time limit
    """

    RECENT = "recent"
    CONTEXTUAL = "contextual"
    DEEP = "deep"
    FULL = "full"

    def get_defaults(self) -> dict[str, Any]:
        """
        Get default parameters for this search mode.

        Returns:
            Dictionary with:
                - max_results: Maximum number of results to return
                - graph_depth: Maximum graph traversal depth
                - temporal_days: Time window in days (None = no limit)
                - vector_weight: Weight for vector search (0.0-1.0)
                - bm25_weight: Weight for BM25 search (0.0-1.0)
                - include_relations: Whether to include graph relations
                - cost_estimate: Relative token cost multiplier
                - use_smart_traversal: Whether to use SmartTraversalV2
                - vector_top_k: Number of vector hits for smart traversal
                - min_vector_score: Minimum similarity for smart traversal
                - min_combined_score: Minimum combined score for smart traversal
        """
        defaults = {
            SearchMode.RECENT: {
                "max_results": 10,
                "graph_depth": 1,
                "temporal_days": 0.167,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "include_relations": True,
                "cost_estimate": 1.0,
                "description": "Fast search of recent memories (last 4 hours, low token cost)",
                "use_smart_traversal": True,
                "vector_top_k": 5,
                "min_vector_score": 0.6,
                "min_combined_score": 0.4,
            },
            SearchMode.CONTEXTUAL: {
                "max_results": 20,
                "graph_depth": 2,
                "temporal_days": 30,
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "include_relations": True,
                "cost_estimate": 2.5,
                "description": "Balanced search with context (medium token cost)",
                "use_smart_traversal": True,
                "vector_top_k": 10,
                "min_vector_score": 0.5,
                "min_combined_score": 0.3,
            },
            SearchMode.DEEP: {
                "max_results": 50,
                "graph_depth": 3,
                "temporal_days": 90,
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "include_relations": True,
                "cost_estimate": 5.0,
                "description": "Deep search with extended graph (high token cost)",
                "use_smart_traversal": True,
                "vector_top_k": 15,
                "min_vector_score": 0.4,
                "min_combined_score": 0.25,
            },
            SearchMode.FULL: {
                "max_results": 100,
                "graph_depth": 4,
                "temporal_days": None,
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "include_relations": True,
                "cost_estimate": 10.0,
                "description": "Full search across all history (very high token cost)",
                "use_smart_traversal": False,
                "vector_top_k": None,
                "min_vector_score": None,
                "min_combined_score": None,
            },
        }

        return defaults[self]

    def get_cost_warning(self) -> str | None:
        """
        Get warning message about token cost for this mode.

        Returns:
            Warning message or None for RECENT mode
        """
        warnings = {
            SearchMode.RECENT: None,
            SearchMode.CONTEXTUAL: (
                "⚠️  CONTEXTUAL mode will increase token usage by ~2.5x. "
                "Related memories from the graph will be loaded."
            ),
            SearchMode.DEEP: (
                "⚠️  DEEP mode will increase token usage by ~5x! "
                "Deep graph traversal will be performed. "
                "Make sure you have enough tokens."
            ),
            SearchMode.FULL: (
                "🚨 FULL mode will increase token usage by ~10x! "
                "Complete memory history will be loaded. "
                "This mode is suitable for data export or comprehensive analysis. "
                "Use only when necessary."
            ),
        }

        return warnings[self]

    @classmethod
    def from_string(cls, value: str | None) -> SearchMode:
        """
        Convert string to SearchMode, with fallback to RECENT.

        Args:
            value: Mode string ("recent", "contextual", "deep", "full")

        Returns:
            SearchMode enum value
        """
        if value is None:
            return cls.RECENT

        try:
            return cls(value.lower())
        except ValueError:
            return cls.RECENT


def estimate_token_cost(
    mode: SearchMode,
    num_results: int | None = None,
    graph_depth: int | None = None,
) -> dict[str, Any]:
    """
    Estimate token cost for a search operation.

    Args:
        mode: Search mode
        num_results: Override number of results
        graph_depth: Override graph depth

    Returns:
        Dictionary with:
            - base_cost: Base token cost estimate
            - result_cost: Per-result token cost
            - total_cost: Total estimated cost
            - cost_tier: Cost tier (low/medium/high/very_high)
    """
    defaults = mode.get_defaults()

    results = num_results or defaults["max_results"]
    depth = graph_depth or defaults["graph_depth"]

    base_cost_per_memory = 200
    relation_cost = 50

    graph_multiplier = 1 + (depth * 2)

    result_cost = base_cost_per_memory + (relation_cost * depth * 2)
    total_cost = result_cost * results * graph_multiplier

    if total_cost < 5000:
        tier = "low"
    elif total_cost < 15000:
        tier = "medium"
    elif total_cost < 50000:
        tier = "high"
    else:
        tier = "very_high"

    return {
        "base_cost": result_cost,
        "result_cost": result_cost * graph_multiplier,
        "total_cost": int(total_cost),
        "cost_tier": tier,
        "num_results": results,
        "graph_depth": depth,
        "mode": mode.value,
    }
