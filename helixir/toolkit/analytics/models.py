"""
Analytics models - data structures for database analytics.

Contains Pydantic models for:
- Storage metrics (size, distribution)
- Graph metrics (nodes, edges, orphans)
- Performance metrics (cache, latency)
- Growth metrics (trends, rates)
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StorageStats(BaseModel):
    """Storage statistics for the database."""

    total_size_bytes: int = Field(0, description="Total database size in bytes")
    total_size_mb: float = Field(0.0, description="Total size in MB")
    total_size_gb: float = Field(0.0, description="Total size in GB")

    size_by_type: dict[str, int] = Field(
        default_factory=dict, description="Storage size by memory type (bytes)"
    )
    size_by_user: dict[str, int] = Field(
        default_factory=dict, description="Storage size by user (bytes)"
    )

    total_memories: int = Field(0, description="Total number of memories")
    avg_memory_size: float = Field(0.0, description="Average memory size in bytes")
    largest_memories: list[tuple[str, int]] = Field(
        default_factory=list, description="Top 10 largest memories (id, size)"
    )

    vector_count: int = Field(0, description="Number of vector embeddings")
    vector_dimensions: int = Field(768, description="Vector dimensions")
    vector_storage_mb: float = Field(0.0, description="Storage used by vectors (MB)")

    chunks_count: int = Field(0, description="Number of memory chunks")
    chunks_storage_mb: float = Field(0.0, description="Storage used by chunks (MB)")

    collected_at: datetime = Field(
        default_factory=datetime.now, description="When stats were collected"
    )


class GraphStats(BaseModel):
    """Graph structure statistics."""

    node_counts: dict[str, int] = Field(
        default_factory=dict, description="Node counts by type (Memory, Entity, Concept, etc.)"
    )
    total_nodes: int = Field(0, description="Total nodes in graph")

    edge_counts: dict[str, int] = Field(
        default_factory=dict, description="Edge counts by type (MENTIONS, IMPLIES, etc.)"
    )
    total_edges: int = Field(0, description="Total edges in graph")

    orphaned_entities: int = Field(0, description="Entities with no memory references")
    orphaned_edges: int = Field(0, description="Edges pointing to deleted nodes")
    deleted_memories: int = Field(0, description="Soft-deleted memories")

    graph_density: float = Field(0.0, description="Graph density (edges/possible)")
    avg_degree: float = Field(0.0, description="Average node degree")

    collected_at: datetime = Field(
        default_factory=datetime.now, description="When stats were collected"
    )


class PerformanceStats(BaseModel):
    """Performance and efficiency statistics."""

    cache_hit_rate: float = Field(0.0, description="Cache hit rate (%)")
    cache_size: int = Field(0, description="Current cache size")
    cache_evictions: int = Field(0, description="Cache evictions count")

    total_queries: int = Field(0, description="Total queries executed")
    avg_query_latency_ms: float = Field(0.0, description="Average query latency (ms)")
    median_query_latency_ms: float = Field(0.0, description="Median query latency (ms)")
    p95_query_latency_ms: float = Field(0.0, description="95th percentile latency (ms)")

    slowest_queries: list[tuple[str, float]] = Field(
        default_factory=list, description="Top 10 slowest queries (name, latency_ms)"
    )

    error_count: int = Field(0, description="Total errors")
    error_rate: float = Field(0.0, description="Error rate (%)")

    collected_at: datetime = Field(
        default_factory=datetime.now, description="When stats were collected"
    )


class GrowthStats(BaseModel):
    """Growth and trend statistics."""

    memories_per_day: float = Field(0.0, description="Avg memories added per day")
    memories_per_week: float = Field(0.0, description="Avg memories added per week")
    growth_rate_percent: float = Field(0.0, description="Growth rate (%)")

    size_growth_mb_per_day: float = Field(0.0, description="Storage growth (MB/day)")
    size_growth_gb_per_week: float = Field(0.0, description="Storage growth (GB/week)")

    trend: str = Field("stable", description="Trend: growing, stable, declining")
    projection_7d: dict[str, Any] = Field(default_factory=dict, description="7-day projection")
    projection_30d: dict[str, Any] = Field(default_factory=dict, description="30-day projection")

    analysis_period_days: int = Field(7, description="Analysis period (days)")
    collected_at: datetime = Field(
        default_factory=datetime.now, description="When stats were collected"
    )


class AnalyticsSummary(BaseModel):
    """Combined analytics summary."""

    storage: StorageStats
    graph: GraphStats
    performance: PerformanceStats
    growth: GrowthStats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "storage": self.storage.model_dump(),
            "graph": self.graph.model_dump(),
            "performance": self.performance.model_dump(),
            "growth": self.growth.model_dump(),
        }

    def __repr__(self) -> str:
        return (
            f"AnalyticsSummary("
            f"storage={self.storage.total_size_mb:.1f}MB, "
            f"memories={self.storage.total_memories}, "
            f"nodes={self.graph.total_nodes})"
        )
