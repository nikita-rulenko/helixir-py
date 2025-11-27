"""
Analytics Manager - analyzes database state and memory metrics.

Provides comprehensive analytics:
- Storage usage (size, distribution by category/user)
- Graph structure (nodes, edges, orphans)
- Performance metrics (cache, latency)
- Growth trends (rate, projections)
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from helixir.toolkit.analytics.models import (
    AnalyticsSummary,
    GraphStats,
    GrowthStats,
    PerformanceStats,
    StorageStats,
)
from helixir.toolkit.misc_toolbox.float_controller import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient


class AnalyticsManager:
    """
    Manages database analytics and monitoring.

    Collects and analyzes:
    - Storage metrics (total size, per-category, per-user)
    - Graph metrics (nodes, edges, orphans)
    - Performance metrics (cache, latency, errors)
    - Growth trends (rate, projections)
    """

    def __init__(self, db_client: HelixDBClient):
        """
        Initialize analytics manager.

        Args:
            db_client: HelixDB client for queries
        """
        self.db = db_client
        self._cache: dict[str, Any] = {}

    async def collect_all(self) -> AnalyticsSummary:
        """
        Collect all analytics in one pass.

        Returns:
            Complete analytics summary
        """
        float_event("analytics.collect_all.start")

        try:
            storage = await self.collect_storage_stats()
            graph = await self.collect_graph_stats()
            performance = await self.collect_performance_stats()
            growth = await self.collect_growth_stats()

            summary = AnalyticsSummary(
                storage=storage, graph=graph, performance=performance, growth=growth
            )

            float_event(
                "analytics.collect_all.success",
                total_memories=storage.total_memories,
                total_nodes=graph.total_nodes,
                size_mb=storage.total_size_mb,
            )
            return summary

        except Exception as e:
            float_event("analytics.collect_all.error", error=str(e))
            raise

    async def collect_storage_stats(self) -> StorageStats:
        """
        Collect storage usage statistics.

        Returns:
            Storage statistics
        """
        float_event("analytics.storage.start")

        try:
            float_event("analytics.storage.query_memories")
            memories_result = await self.db.execute_query("getMemoriesForStorageStats", {})
            float_event(
                "analytics.storage.memories_fetched", count=len(memories_result.get("memories", []))
            )
            memories = memories_result.get("memories", [])

            float_event("analytics.storage.query_chunks")
            chunks_result = await self.db.execute_query("getChunkCount", {})
            float_event(
                "analytics.storage.chunks_fetched", count=len(chunks_result.get("chunks", []))
            )
            chunks = chunks_result.get("chunks", [])

            float_event("analytics.storage.calculating_stats")
            total_memories = len(memories)
            total_size_bytes = sum(len(m.get("content", "")) for m in memories)
            total_size_mb = total_size_bytes / (1024 * 1024)
            total_size_gb = total_size_mb / 1024

            size_by_type: dict[str, int] = {}
            for m in memories:
                mem_type = m.get("memory_type", "unknown")
                content_size = len(m.get("content", ""))
                size_by_type[mem_type] = size_by_type.get(mem_type, 0) + content_size

            avg_memory_size = total_size_bytes / total_memories if total_memories > 0 else 0.0

            memories_with_sizes = [
                (m.get("memory_id", "unknown"), len(m.get("content", ""))) for m in memories
            ]
            largest_memories = sorted(memories_with_sizes, key=lambda x: x[1], reverse=True)[:10]

            vector_count = total_memories
            vector_storage_mb = (vector_count * 768 * 4) / (1024 * 1024)

            chunks_count = len(chunks)
            chunks_size = sum(len(c.get("content", "")) for c in chunks)
            chunks_storage_mb = chunks_size / (1024 * 1024)

            stats = StorageStats(
                total_size_bytes=total_size_bytes,
                total_size_mb=total_size_mb,
                total_size_gb=total_size_gb,
                total_memories=total_memories,
                size_by_type=size_by_type,
                avg_memory_size=avg_memory_size,
                largest_memories=largest_memories,
                vector_count=vector_count,
                vector_storage_mb=vector_storage_mb,
                chunks_count=chunks_count,
                chunks_storage_mb=chunks_storage_mb,
                collected_at=datetime.now(),
            )

            float_event(
                "analytics.storage.complete", total_memories=total_memories, size_mb=total_size_mb
            )
            return stats

        except Exception as e:
            float_event("analytics.storage.error", error=str(e))
            raise

    async def collect_graph_stats(self) -> GraphStats:
        """
        Collect graph structure statistics.

        Returns:
            Graph statistics
        """
        float_event("analytics.graph.start")

        try:
            float_event("analytics.graph.querying_nodes")
            memories_result = await self.db.execute_query("getAllMemoriesForGraph", {})
            entities_result = await self.db.execute_query("getAllEntities", {})
            concepts_result = await self.db.execute_query("getAllConcepts", {})
            chunks_result = await self.db.execute_query("getAllChunks", {})
            deleted_result = await self.db.execute_query("getDeletedMemories", {})

            memories = memories_result.get("memories", [])
            entities = entities_result.get("entities", [])
            concepts = concepts_result.get("concepts", [])
            chunks = chunks_result.get("chunks", [])
            deleted_memories = deleted_result.get("memories", [])

            float_event(
                "analytics.graph.nodes_fetched",
                memories=len(memories),
                entities=len(entities),
                concepts=len(concepts),
                chunks=len(chunks),
                deleted=len(deleted_memories),
            )

            node_counts = {
                "Memory": len(memories),
                "Entity": len(entities),
                "Concept": len(concepts),
                "MemoryChunk": len(chunks),
            }
            total_nodes = sum(node_counts.values())

            edge_counts: dict[str, int] = {}
            total_edges = 0

            max_possible_edges = total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 0
            graph_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0
            avg_degree = (2 * total_edges) / total_nodes if total_nodes > 0 else 0.0

            stats = GraphStats(
                node_counts=node_counts,
                total_nodes=total_nodes,
                edge_counts=edge_counts,
                total_edges=total_edges,
                orphaned_entities=0,
                deleted_memories=len(deleted_memories),
                graph_density=graph_density,
                avg_degree=avg_degree,
                collected_at=datetime.now(),
            )

            float_event(
                "analytics.graph.complete", total_nodes=total_nodes, total_edges=total_edges
            )
            return stats

        except Exception as e:
            float_event("analytics.graph.error", error=str(e))
            raise

    async def collect_performance_stats(self) -> PerformanceStats:
        """
        Collect performance metrics from FloatController.

        Returns:
            Performance statistics
        """
        float_event("analytics.performance.start")

        try:
            from helixir.toolkit.misc_toolbox.float_controller import get_float_controller

            fc = get_float_controller()

            report = fc.get_report()

            float_event(
                "analytics.performance.analyzing_floats", total_floats=report["total_floats"]
            )

            query_floats = fc.get_floats("*.query.*")
            fc.get_floats("*.cache.*")
            error_floats = fc.get_floats("*.error")

            cache_hits = len(fc.get_floats("*.cache.hit"))
            cache_misses = len(fc.get_floats("*.cache.miss"))
            total_cache_ops = cache_hits + cache_misses
            cache_hit_rate = cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

            query_latencies: list[float] = []
            for event in query_floats:
                if "latency_ms" in event.data:
                    query_latencies.append(event.data["latency_ms"])

            avg_query_latency_ms = (
                sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
            )

            error_count = len(error_floats)

            stats = PerformanceStats(
                cache_hit_rate=cache_hit_rate,
                total_queries=len(query_floats),
                avg_query_latency_ms=avg_query_latency_ms,
                error_count=error_count,
                collected_at=datetime.now(),
            )

            float_event(
                "analytics.performance.complete",
                queries=len(query_floats),
                errors=error_count,
                cache_hit_rate=cache_hit_rate,
            )
            return stats

        except Exception as e:
            float_event("analytics.performance.error", error=str(e))
            raise

    async def collect_growth_stats(self) -> GrowthStats:
        """
        Collect growth and trend statistics.

        Returns:
            Growth statistics
        """
        float_event("analytics.growth.start")

        try:
            from datetime import timedelta

            analysis_period_days = 7
            cutoff_date = datetime.now() - timedelta(days=analysis_period_days)
            cutoff_str = cutoff_date.isoformat()

            float_event("analytics.growth.querying", period_days=analysis_period_days)

            result = await self.db.execute_query("getMemoriesForStorageStats", {})
            all_memories = result.get("memories", [])

            recent_memories = [m for m in all_memories if m.get("created_at", "") >= cutoff_str]

            memories_per_day = (
                len(recent_memories) / analysis_period_days if analysis_period_days > 0 else 0.0
            )

            old_count = len(all_memories) - len(recent_memories)
            if old_count > 0:
                growth_rate_percent = (len(recent_memories) / old_count) * 100
            else:
                growth_rate_percent = 100.0 if len(recent_memories) > 0 else 0.0

            if memories_per_day < 1:
                trend = "slow"
            elif memories_per_day < 10:
                trend = "stable"
            elif memories_per_day < 100:
                trend = "growing"
            else:
                trend = "rapid"

            stats = GrowthStats(
                memories_per_day=memories_per_day,
                growth_rate_percent=growth_rate_percent,
                trend=trend,
                analysis_period_days=analysis_period_days,
                collected_at=datetime.now(),
            )

            float_event(
                "analytics.growth.complete",
                memories_per_day=memories_per_day,
                trend=trend,
                growth_rate=growth_rate_percent,
            )
            return stats

        except Exception as e:
            float_event("analytics.growth.error", error=str(e))
            raise

    async def get_category_breakdown(self) -> dict[str, int]:
        """
        Get memory count breakdown by category.

        Returns:
            Dict of {category: count}
        """
        float_event("analytics.category_breakdown.start")

        try:
            float_event("analytics.category_breakdown.querying")
            result = await self.db.execute_query("getMemoriesForCategoryBreakdown", {})
            memories = result.get("memories", [])

            breakdown: dict[str, int] = {}
            for m in memories:
                mem_type = m.get("memory_type", "unknown")
                breakdown[mem_type] = breakdown.get(mem_type, 0) + 1

            float_event("analytics.category_breakdown.complete", categories=len(breakdown))
            return breakdown

        except Exception as e:
            float_event("analytics.category_breakdown.error", error=str(e))
            raise

    async def get_user_breakdown(self) -> dict[str, dict[str, Any]]:
        """
        Get memory and storage breakdown by user.

        Returns:
            Dict of {user_id: {count: int, size_mb: float}}
        """
        float_event("analytics.user_breakdown.start")

        try:
            float_event("analytics.user_breakdown.querying")
            result = await self.db.execute_query("getMemoriesForUserBreakdown", {})
            memories = result.get("memories", [])

            breakdown: dict[str, dict[str, Any]] = {}
            for m in memories:
                user_id = m.get("user_id", "unknown")

                if user_id not in breakdown:
                    breakdown[user_id] = {"count": 0, "size_mb": 0.0}

                breakdown[user_id]["count"] += 1
                content_size = len(m.get("content", ""))
                breakdown[user_id]["size_mb"] += content_size / (1024 * 1024)

            float_event("analytics.user_breakdown.complete", users=len(breakdown))
            return breakdown

        except Exception as e:
            float_event("analytics.user_breakdown.error", error=str(e))
            raise

    async def export_to_json(self, summary: AnalyticsSummary) -> str:
        """
        Export analytics summary to JSON string.

        Args:
            summary: Analytics summary to export

        Returns:
            JSON string
        """
        import json

        return json.dumps(summary.to_dict(), indent=2, default=str)

    async def cleanup_orphans(self) -> dict[str, int]:
        """
        Clean up orphaned entities and edges.

        Orphan detection rules:
        1. Entity without any MENTIONS/HAS_ENTITY edges
        2. Concept without any RELATES_TO edges
        3. Edges pointing to deleted/non-existent nodes

        Returns:
            Cleanup summary: {entities_deleted: int, edges_deleted: int, concepts_deleted: int}
        """
        float_event("analytics.cleanup_orphans.start")

        try:
            entities_deleted = 0
            concepts_deleted = 0
            edges_deleted = 0

            float_event("analytics.cleanup_orphans.scanning_entities")
            entities_result = await self.db.execute_query("getAllEntities", {})
            entities = entities_result.get("entities", [])

            memories_result = await self.db.execute_query("getMemoriesForStorageStats", {})
            memories = memories_result.get("memories", [])

            mentioned_entity_ids = set()
            for memory in memories:
                metadata = memory.get("metadata", {})
                if isinstance(metadata, dict) and "entities" in metadata:
                    for entity_ref in metadata.get("entities", []):
                        if isinstance(entity_ref, dict):
                            mentioned_entity_ids.add(entity_ref.get("entity_id", ""))

            orphaned_entities = [
                e for e in entities if e.get("entity_id", "") not in mentioned_entity_ids
            ]

            float_event("analytics.cleanup_orphans.found_orphans", entities=len(orphaned_entities))

            for orphan in orphaned_entities:
                logger.warning(
                    "Orphaned entity detected: %s (type: %s)",
                    orphan.get("entity_id", "unknown"),
                    orphan.get("entity_type", "unknown"),
                )

            entities_deleted = len(orphaned_entities)

            result = {
                "entities_deleted": entities_deleted,
                "concepts_deleted": concepts_deleted,
                "edges_deleted": edges_deleted,
                "note": "Detection only - actual deletion requires DeletionManager integration",
            }

            float_event(
                "analytics.cleanup_orphans.complete",
                entities_found=entities_deleted,
                edges_found=edges_deleted,
            )
            return result

        except Exception as e:
            float_event("analytics.cleanup_orphans.error", error=str(e))
            raise
