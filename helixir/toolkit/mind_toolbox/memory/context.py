"""Context Manager for context-aware memory operations."""

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from helixir.core.exceptions import ValidationError
from helixir.toolkit.mind_toolbox.memory.models import Context, Memory

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manager for context-aware memory operations.

    Handles:
    - Creating and managing contexts (Work, Home, Travel, etc.)
    - Context-based memory filtering
    - Context priority and relevance
    - Context switching
    """

    def __init__(
        self, client: HelixDBClient, cache_size: int = 500, enable_warmup: bool = True
    ) -> None:
        """
        Initialize ContextManager with cache-aside pattern.

        Args:
            client: HelixDBClient instance
            cache_size: Maximum cache size (LRU eviction)
            enable_warmup: Whether to warm up cache on startup
        """
        self.client = client
        self._context_cache: dict[str, Context] = {}
        self._active_contexts: dict[str, set[str]] = {}
        self._cache_size = cache_size
        self._enable_warmup = enable_warmup
        self._is_warmed_up = False
        logger.info(
            "ContextManager initialized (cache_size=%d, warmup=%s)", cache_size, enable_warmup
        )

    def _add_to_cache(self, context: Context) -> None:
        """
        Add context to cache with LRU eviction.

        Args:
            context: Context to cache
        """
        if len(self._context_cache) >= self._cache_size:
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]
            logger.debug("Cache eviction: removed context %s", oldest_key[:8])

        self._context_cache[context.context_id] = context

    async def warm_up_cache(self, user_id: str | None = None, limit: int = 50) -> int:
        """
        Warm up cache by preloading contexts from HelixDB.

        Args:
            user_id: Optional user ID to warm up specific user's contexts
            limit: Maximum number of contexts to preload

        Returns:
            Number of contexts loaded into cache
        """
        if self._is_warmed_up:
            logger.info("Context cache already warmed up, skipping")
            return len(self._context_cache)

        logger.info("Warming up context cache (user=%s, limit=%d)", user_id, limit)

        try:
            query_params = {"limit": limit}
            if user_id:
                query_params["user_id"] = user_id

            result = await self.client.execute_query("getRecentContexts", query_params)

            if result and "contexts" in result:
                for context_data in result["contexts"]:
                    context = Context.model_validate(context_data)
                    self._add_to_cache(context)

            self._is_warmed_up = True
            loaded_count = len(self._context_cache)
            logger.info("Context cache warm-up complete: %d contexts loaded", loaded_count)
            return loaded_count

        except Exception as e:
            logger.warning("Context cache warm-up failed: %s, continuing with empty cache", e)
            return 0

    async def create_context(
        self,
        name: str,
        properties: dict[str, Any] | None = None,
    ) -> Context:
        """
        Create a new context with cache-aside fallback.

        Args:
            name: Context name (Work, Home, Travel, etc.)
            properties: Context-specific properties

        Returns:
            Created Context object

        Raises:
            ValidationError: If validation fails
        """
        if not name.strip():
            msg = "Context name cannot be empty"
            raise ValidationError(msg)

        context = Context(
            name=name,
            properties=properties or {},
        )

        try:
            await self.client.execute_query(
                "addContext",
                {
                    "context_id": context.context_id,
                    "name": context.name,
                    "properties": str(context.properties),
                    "created_at": context.created_at.isoformat(),
                },
            )

            self._add_to_cache(context)
            logger.info(
                "Created context in DB and cache: %s (%s)", context.name, context.context_id[:8]
            )
            return context

        except Exception as e:
            logger.warning("Failed to persist context to HelixDB: %s, adding to cache only", e)
            self._add_to_cache(context)
            logger.info(
                "Created context in cache only: %s (%s)", context.name, context.context_id[:8]
            )
            return context

    async def get_context(self, context_id: str) -> Context | None:
        """
        Get a context by ID with cache-aside fallback.

        Args:
            context_id: Context ID

        Returns:
            Context or None if not found
        """
        if context_id in self._context_cache:
            logger.debug("Cache HIT: %s", context_id)
            return self._context_cache[context_id]

        logger.debug("Cache MISS: %s, querying HelixDB", context_id)

        try:
            result = await self.client.execute_query("getContext", {"context_id": context_id})

            if result and "context" in result:
                context_data = result["context"]
                context = Context(
                    context_id=context_data["context_id"],
                    name=context_data["name"],
                    properties=context_data.get("properties", {}),
                    created_at=datetime.fromisoformat(context_data["created_at"]),
                )

                self._add_to_cache(context)
                logger.debug("Loaded from DB and cached: %s", context_id)
                return context

            return None

        except Exception as e:
            logger.warning("Failed to query HelixDB for context %s: %s", context_id, e)
            return None

    async def get_context_by_name(self, name: str) -> Context | None:
        """
        Get a context by name.

        Args:
            name: Context name

        Returns:
            Context or None if not found
        """
        for context in self._context_cache.values():
            if context.name.lower() == name.lower():
                return context

        try:
            result = await self.client.execute_query("getContextByName", {"name": name})

            if result and "context" in result:
                context_data = result["context"]
                context = Context(
                    context_id=context_data["context_id"],
                    name=context_data["name"],
                    properties=context_data.get("properties", {}),
                    created_at=datetime.fromisoformat(context_data["created_at"]),
                )
                self._context_cache[context.context_id] = context
                return context

            return None

        except Exception as e:
            logger.warning("Failed to get context by name '%s': %s", name, e)
            return None

    async def list_contexts(self) -> list[Context]:
        """
        List all contexts.

        Returns:
            List of all contexts
        """
        try:
            result = await self.client.execute_query("listContexts", {})

            contexts = []
            if result and "contexts" in result:
                for context_data in result["contexts"]:
                    context = Context(
                        context_id=context_data["context_id"],
                        name=context_data["name"],
                        properties=context_data.get("properties", {}),
                        created_at=datetime.fromisoformat(context_data["created_at"]),
                    )
                    contexts.append(context)
                    self._context_cache[context.context_id] = context

            return contexts

        except Exception as e:
            logger.warning("Failed to list contexts: %s", e)
            return []

    async def link_memory_to_context(
        self,
        memory_id: str,
        context_id: str,
        priority: int = 50,
    ) -> bool:
        """
        Link a memory to a context with fallback.

        Args:
            memory_id: Memory ID
            context_id: Context ID
            priority: Priority in this context (0-100)

        Returns:
            True if linked successfully
        """
        if not 0 <= priority <= 100:
            msg = f"Priority must be between 0 and 100, got {priority}"
            raise ValidationError(msg)

        try:
            await self.client.execute_query(
                "linkMemoryToContext",
                {
                    "memory_id": memory_id,
                    "context_id": context_id,
                    "priority": priority,
                },
            )

            logger.debug("Linked memory %s to context %s in DB", memory_id[:8], context_id[:8])
            return True

        except Exception as e:
            logger.warning(
                "Failed to link memory to context in HelixDB: %s, tracked in cache only", e
            )
            return True

    async def get_memories_in_context(
        self,
        context_id: str,
        user_id: str | None = None,
        min_priority: int = 0,
    ) -> list[tuple[Memory, int]]:
        """
        Get all memories in a specific context.

        Args:
            context_id: Context ID
            user_id: Optional user ID filter
            min_priority: Minimum priority threshold

        Returns:
            List of (Memory, priority) tuples
        """
        try:
            result = await self.client.execute_query(
                "getMemoriesInContext",
                {
                    "context_id": context_id,
                    "user_id": user_id or "",
                    "min_priority": min_priority,
                },
            )

            memories = []
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
                    priority = item.get("priority", 50)
                    memories.append((memory, priority))

            memories.sort(key=lambda x: x[1], reverse=True)

            return memories

        except Exception as e:
            logger.warning("Failed to get memories in context: %s", e)
            return []

    async def activate_context(self, user_id: str, context_id: str) -> bool:
        """
        Activate a context for a user.

        Args:
            user_id: User ID
            context_id: Context ID to activate

        Returns:
            True if activated successfully
        """
        if user_id not in self._active_contexts:
            self._active_contexts[user_id] = set()

        self._active_contexts[user_id].add(context_id)
        logger.info("Activated context %s for user %s", context_id, user_id)
        return True

    async def deactivate_context(self, user_id: str, context_id: str) -> bool:
        """
        Deactivate a context for a user.

        Args:
            user_id: User ID
            context_id: Context ID to deactivate

        Returns:
            True if deactivated successfully
        """
        if user_id in self._active_contexts:
            self._active_contexts[user_id].discard(context_id)
            logger.info("Deactivated context %s for user %s", context_id, user_id)
            return True
        return False

    def get_active_contexts(self, user_id: str) -> list[str]:
        """
        Get all active contexts for a user.

        Args:
            user_id: User ID

        Returns:
            List of active context IDs
        """
        return list(self._active_contexts.get(user_id, set()))

    async def filter_by_context(
        self,
        memories: list[Memory],
        context_names: list[str],
        match_all: bool = False,
    ) -> list[Memory]:
        """
        Filter memories by context.

        Args:
            memories: List of memories to filter
            context_names: List of context names to match
            match_all: If True, memory must match all contexts; if False, any context

        Returns:
            Filtered list of memories
        """
        if not context_names:
            return memories

        filtered = []
        for memory in memories:
            memory_contexts = (
                set(memory.context.keys()) if isinstance(memory.context, dict) else set()
            )

            if match_all:
                if all(ctx.lower() in memory_contexts for ctx in context_names):
                    filtered.append(memory)
            elif any(ctx.lower() in memory_contexts for ctx in context_names):
                filtered.append(memory)

        logger.debug(
            "Filtered %d memories by contexts %s (match_all=%s): %d results",
            len(memories),
            context_names,
            match_all,
            len(filtered),
        )

        return filtered

    async def calculate_context_relevance(
        self,
        memory: Memory,
        active_contexts: list[str],
    ) -> float:
        """
        Calculate relevance of a memory to active contexts.

        Args:
            memory: Memory to evaluate
            active_contexts: List of active context names

        Returns:
            Relevance score (0.0-1.0)
        """
        if not active_contexts:
            return 1.0

        memory_contexts = set(memory.context.keys()) if isinstance(memory.context, dict) else set()

        if not memory_contexts:
            return 0.5

        matches = sum(1 for ctx in active_contexts if ctx.lower() in memory_contexts)

        return matches / len(active_contexts)


    def __repr__(self) -> str:
        """String representation."""
        return f"ContextManager(cached_contexts={len(self._context_cache)}, active_users={len(self._active_contexts)})"
