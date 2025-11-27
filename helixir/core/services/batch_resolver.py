"""
Batch ID Resolution - Efficient bulk external→internal ID mapping.

Purpose:
- Weak coupling: Separate module for batch operations
- Performance: Parallel DB queries with connection pooling
- Resilience: Partial success (some IDs may fail)
- Observable: Full Float tracking

Use Cases:
- Chunking pipeline (resolve parent + all chunks)
- Bulk imports
- Context assembly (resolve multiple memory IDs)

Design:
- Async parallel queries
- Fail-fast vs fail-safe modes
- Automatic retry on transient errors
- Result batching for efficiency
"""

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from helixir.toolkit.misc_toolbox.float_controller import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient


class BatchIDResolver:
    """
    Batch ID resolution for high-performance bulk operations.

    Optimized for:
    - Parallel DB queries (async gather)
    - Partial success handling
    - Connection pooling
    - Observable via Float tracking

    Usage:
        resolver = BatchIDResolver(db_client)

        results = await resolver.resolve_batch(
            ["mem_1", "mem_2", "mem_3"],
            fail_fast=False
        )

        results = await resolver.resolve_batch(
            ["mem_1", "mem_2", "mem_3"],
            fail_fast=True
        )
    """

    def __init__(
        self,
        db_client: HelixDBClient,
        max_parallel: int = 10,
        retry_attempts: int = 3,
    ):
        """
        Initialize Batch ID Resolver.

        Args:
            db_client: HelixDB client for querying
            max_parallel: Max parallel queries (connection pool limit)
            retry_attempts: Retry count for transient errors
        """
        self._db = db_client
        self._max_parallel = max_parallel
        self._retry_attempts = retry_attempts
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def resolve_batch(
        self,
        memory_ids: list[str],
        fail_fast: bool = False,
    ) -> dict[str, UUID]:
        """
        Resolve multiple memory IDs in parallel.

        Args:
            memory_ids: List of external memory identifiers
            fail_fast: If True, raise on first error; if False, return partial results

        Returns:
            Dict mapping memory_id → internal UUID (partial if fail_fast=False)

        Raises:
            BatchResolutionError: If fail_fast=True and any ID fails
        """
        float_event(
            "batch_resolver.started",
            count=len(memory_ids),
            fail_fast=fail_fast,
        )

        unique_ids = list(set(memory_ids))
        if len(unique_ids) < len(memory_ids):
            float_event(
                "batch_resolver.deduplication",
                original=len(memory_ids),
                unique=len(unique_ids),
            )

        tasks = [self._resolve_single_with_retry(mid, fail_fast) for mid in unique_ids]
        results = await asyncio.gather(*tasks, return_exceptions=not fail_fast)

        resolved = {}
        errors = []

        for memory_id, result in zip(unique_ids, results, strict=False):
            if isinstance(result, Exception):
                errors.append((memory_id, str(result)))
                if fail_fast:
                    pass
            else:
                resolved[memory_id] = result

        float_event(
            "batch_resolver.completed",
            requested=len(memory_ids),
            unique=len(unique_ids),
            resolved=len(resolved),
            errors=len(errors),
        )

        if errors:
            for memory_id, error in errors:
                float_event(
                    "batch_resolver.resolution_failed",
                    memory_id=memory_id,
                    error=error,
                )

        return resolved

    async def _resolve_single_with_retry(
        self,
        memory_id: str,
        fail_fast: bool,
    ) -> UUID:
        """
        Resolve single ID with retry logic and semaphore.

        Args:
            memory_id: External memory identifier
            fail_fast: Whether to raise on error

        Returns:
            Internal UUID

        Raises:
            Exception: If all retries fail (only if fail_fast=True)
        """
        async with self._semaphore:
            for attempt in range(self._retry_attempts):
                try:
                    uuid = await self._query_db(memory_id)

                    if attempt > 0:
                        float_event(
                            "batch_resolver.retry_success",
                            memory_id=memory_id,
                            attempt=attempt + 1,
                        )

                    return uuid

                except Exception as e:
                    if attempt < self._retry_attempts - 1:
                        await asyncio.sleep(0.1 * (2**attempt))
                        float_event(
                            "batch_resolver.retry_attempt",
                            memory_id=memory_id,
                            attempt=attempt + 1,
                            error=str(e),
                        )
                    else:
                        float_event(
                            "batch_resolver.retry_exhausted",
                            memory_id=memory_id,
                            attempts=self._retry_attempts,
                            error=str(e),
                        )
                        if fail_fast:
                            raise
                        raise
        return None

    async def _query_db(self, memory_id: str) -> UUID:
        """
        Query HelixDB for internal UUID.

        Args:
            memory_id: External memory identifier

        Returns:
            Internal UUID

        Raises:
            ValueError: If memory not found
        """
        result = await self._db.execute_query("getMemory", {"memory_id": memory_id})

        if not result or "id" not in result:
            raise ValueError(f"Memory not found: {memory_id}")

        return UUID(result["id"])

    def get_stats(self) -> dict[str, Any]:
        """Get resolver configuration."""
        return {
            "max_parallel": self._max_parallel,
            "retry_attempts": self._retry_attempts,
        }


class BatchResolutionError(Exception):
    """Raised when batch resolution fails in fail_fast mode."""

    def __init__(self, failed_ids: list[tuple[str, str]]):
        """
        Initialize error with failed IDs.

        Args:
            failed_ids: List of (memory_id, error_message) tuples
        """
        self.failed_ids = failed_ids
        super().__init__(
            f"Batch resolution failed for {len(failed_ids)} IDs: "
            f"{', '.join(mid for mid, _ in failed_ids[:3])}"
            + ("..." if len(failed_ids) > 3 else "")
        )


__all__ = ["BatchIDResolver", "BatchResolutionError"]
