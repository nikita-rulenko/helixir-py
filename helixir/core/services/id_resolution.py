"""
ID Resolution Service - External→Internal ID mapping with smart caching.

Problem:
- SDK uses external IDs (memory_id: String) for API
- HelixDB uses internal UUIDs for edges (AddE::From(uuid))
- Need efficient resolution with cache invalidation

Solution:
- Hybrid cache: Event-driven + TTL + LRU
- Event invalidation on delete/update
- TTL as safety net (5 min)
- LRU eviction for memory control
- Full Float observability

Architecture:
    resolve(memory_id) → UUID
         ↓
    Cache hit? → return
         ↓
    Query HelixDB → cache → return
         ↓
    Event (delete/update) → invalidate
"""

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from helixir.core.cache import LRUCache
from helixir.core.events import BaseEvent, EventHandler
from helixir.toolkit.misc_toolbox.float_controller import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient


class IDResolutionService(EventHandler):
    """
    Resolve external IDs (String) to internal HelixDB UUIDs.

    Features:
    - Event-driven cache invalidation
    - TTL-based expiry (5 min default)
    - LRU eviction (10k entries default)
    - Float tracking for observability
    - Async batch resolution

    Cache Strategy:
    1. Event invalidation (immediate, precise)
    2. TTL expiry (safety net for missed events)
    3. LRU eviction (memory overflow protection)

    Usage:
        resolver = IDResolutionService(db_client)
        uuid = await resolver.resolve("mem_123")

        uuids = await resolver.resolve_many(["mem_1", "mem_2", "mem_3"])
    """

    def __init__(
        self,
        db_client: HelixDBClient,
        max_size: int = 10_000,
        ttl: int = 300,
    ):
        """
        Initialize ID Resolution Service.

        Args:
            db_client: HelixDB client for querying
            max_size: Maximum cache entries (LRU eviction)
            ttl: Time-to-live in seconds (300 = 5 min)
        """
        self._db = db_client
        self._max_size = max_size
        self._ttl = ttl

        self._cache = LRUCache[str, UUID](maxsize=max_size, ttl=float(ttl))

        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
        }

    async def resolve(self, memory_id: str) -> UUID:
        """
        Resolve external memory_id to internal UUID.

        Args:
            memory_id: External memory identifier (String)

        Returns:
            Internal HelixDB UUID

        Raises:
            ValueError: If memory_id not found in DB
        """
        float_event("id_resolver.resolve_requested", memory_id=memory_id)

        cached_uuid = self._cache.get(memory_id)
        if cached_uuid is not None:
            self._stats["hits"] += 1
            float_event(
                "id_resolver.cache_hit",
                memory_id=memory_id,
            )
            return cached_uuid

        self._stats["misses"] += 1
        float_event("id_resolver.cache_miss", memory_id=memory_id)

        uuid = await self._query_db(memory_id)

        self._cache.set(memory_id, uuid)
        float_event("id_resolver.cached", memory_id=memory_id, uuid=str(uuid))

        return uuid

    async def resolve_many(self, memory_ids: list[str]) -> dict[str, UUID]:
        """
        Batch resolve multiple memory IDs in parallel.

        More efficient than calling resolve() in a loop.

        Args:
            memory_ids: List of external memory identifiers

        Returns:
            Dict mapping memory_id → internal UUID
        """
        float_event(
            "id_resolver.batch_resolve_requested",
            count=len(memory_ids),
        )

        tasks = [self.resolve(mid) for mid in memory_ids]
        uuids = await asyncio.gather(*tasks, return_exceptions=True)

        result = {}
        errors = 0
        for memory_id, uuid in zip(memory_ids, uuids, strict=False):
            if isinstance(uuid, Exception):
                errors += 1
                float_event(
                    "id_resolver.batch_resolve_error",
                    memory_id=memory_id,
                    error=str(uuid),
                )
            else:
                result[memory_id] = uuid

        float_event(
            "id_resolver.batch_resolve_complete",
            requested=len(memory_ids),
            resolved=len(result),
            errors=errors,
        )

        return result

    async def _query_db(self, memory_id: str) -> UUID:
        """
        Query HelixDB for internal UUID by external memory_id.

        Args:
            memory_id: External memory identifier

        Returns:
            Internal HelixDB UUID

        Raises:
            ValueError: If memory not found
        """
        float_event("id_resolver.db_query_started", memory_id=memory_id)

        try:
            result = await self._db.execute_query("getMemory", {"memory_id": memory_id})

            if not result or "id" not in result:
                raise ValueError(f"Memory not found: {memory_id}")

            uuid = UUID(result["id"])

            float_event(
                "id_resolver.db_query_success",
                memory_id=memory_id,
                uuid=str(uuid),
            )

            return uuid

        except Exception as e:
            float_event(
                "id_resolver.db_query_failed",
                memory_id=memory_id,
                error=str(e),
            )
            raise

    @property
    def event_types(self) -> list[str]:
        """Events that trigger cache invalidation."""
        return ["memory.deleted", "memory.updated"]

    async def handle(self, event: BaseEvent) -> None:
        """
        Handle cache invalidation events.

        Called by EventBus when memory.deleted or memory.updated occurs.
        """
        memory_id = getattr(event, "memory_id", None)
        if not memory_id:
            return

        deleted = self._cache.delete(memory_id)
        if deleted:
            self._stats["invalidations"] += 1

            float_event(
                "id_resolver.cache_invalidated",
                memory_id=memory_id,
                reason=event.event_type,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        cache_stats = self._cache.stats

        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": cache_stats["size"],
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hit_rate": round(hit_rate, 3),
            "lru_stats": cache_stats,
            **self._stats,
        }

    def clear_cache(self) -> None:
        """Clear entire cache (for testing or manual reset)."""
        self._cache.clear()
        float_event("id_resolver.cache_cleared")


__all__ = ["IDResolutionService"]
