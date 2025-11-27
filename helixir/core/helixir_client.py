"""
Helixir Client - main entry point for the framework.

Architecture:
- Automatic initialization of HelixDB + LLM + Embeddings
- High-level API (add, search, update, delete)
- Uses ToolingManager under the hood for all logic
- LLM agent manages toolboxes via function calling
"""

import logging
from typing import Any

from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.llm import LLMExtractor
from helixir.llm.factory import ProviderSingleton
from helixir.toolkit.analytics import AnalyticsManager, AnalyticsSummary
from helixir.toolkit.misc_toolbox import float_event
from helixir.toolkit.tooling_manager import ToolingManager

logger = logging.getLogger(__name__)


class HelixirClient:
    """
    Helixir Client - main facade of the framework.

    Automatically initializes:
    - HelixDB connection (database)
    - LLM provider (memory engine)
    - Embedding generator (vectors)
    - ToolingManager (LLM agent with function calling)

    Architecture:
        User Request
            ↓
        HelixirClient (high-level API)
            ↓
        ToolingManager (LLM agent)
            ↓
        Toolboxes (QueryBuilder, ChunkManager, SchemaManager)
            ↓
        HelixDB

    Usage:
        >>> config = HelixMemoryConfig(
        ...     host="localhost", port=6969, llm_provider="ollama", llm_model="gemma2"
        ... )
        >>> client = HelixirClient(config)
        >>> await client.add("I love Python", user_id="user123")
        >>> results = await client.search("Python", user_id="user123")
    """

    def __init__(self, config: HelixMemoryConfig | None = None):
        """
        Initialize Helixir Client.

        Args:
            config: Configuration object. If None, attempts to load from:
                    1. config.yaml (auto-search in standard locations)
                    2. Environment variables (HELIX_* prefix)
                    3. Default values

        Priority: config arg > config.yaml > env vars > defaults

        Example:
            >>>
            >>> client = HelixirClient()
            >>>
            >>>
            >>> config = HelixMemoryConfig.from_yaml("custom_config.yaml")
            >>> client = HelixirClient(config)
            >>>
            >>>
            >>> config = HelixMemoryConfig(llm_provider="cerebras", llm_model="llama3.3-70b")
            >>> client = HelixirClient(config)
        """
        if config is None:
            try:
                self.config = HelixMemoryConfig.from_yaml()
                logger.info("Loaded configuration from config.yaml")
            except FileNotFoundError:
                self.config = HelixMemoryConfig()
                logger.info("Using configuration from environment variables and defaults")
        else:
            self.config = config

        self._providers = ProviderSingleton(self.config)

        self.db = self._init_db()
        self.llm_provider = self._providers.llm
        self.extractor = self._init_extractor()
        self.embedder = self._providers.embedding

        self.tooling = ToolingManager(
            db_client=self.db,
            llm_extractor=self.extractor,
            embedder=self.embedder,
            config=self.config,
        )

        self.analytics = AnalyticsManager(db_client=self.db)

        logger.info(
            "HelixirClient initialized: db=%s:%d, llm=%s/%s, embedding=%s/%s",
            self.config.host,
            self.config.port,
            self.config.llm_provider,
            self.config.llm_model,
            self.config.embedding_provider,
            self.config.embedding_model,
        )

    def _init_db(self) -> HelixDBClient:
        """Initialize HelixDB client."""
        return HelixDBClient(self.config)

    def _init_extractor(self) -> LLMExtractor:
        """Initialize LLM extractor with configured provider."""
        return LLMExtractor(provider=self.llm_provider)

    async def add(
        self,
        message: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add memory with LLM-powered extraction.

        Pipeline:
        1. LLM extracts atomic facts (extractor)
        2. Generate embeddings (embedder)
        3. Vector search for duplicates (tooling)
        4. LLM decides: ADD/UPDATE/DELETE (tooling)
        5. Store in HelixDB (tooling)

        Args:
            message: Text to extract memories from
            user_id: User identifier
            agent_id: Optional agent identifier
            metadata: Optional additional metadata

        Returns:
            Dictionary with added memory IDs and stats

        Example:
            >>> result = await client.add("I love Python and FastAPI", user_id="user123")
            >>> print(result["memories_added"])
            2
        """
        float_event("client.add.start", user_id=user_id, message_len=len(message))

        try:
            result = await self.tooling.add_memory(
                message=message, user_id=user_id, agent_id=agent_id, metadata=metadata
            )

            float_event(
                "client.add.success",
                user_id=user_id,
                memories_added=result.get("memories_added", 0),
            )

            return result

        except Exception as e:
            float_event("client.add.error", user_id=user_id, error=str(e))
            raise

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int | None = None,
        search_mode: str = "recent",
        temporal_days: int | None = None,
        graph_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memories with LLM + vectors and flexible depth control.

        Args:
            query: Search query
            user_id: User identifier
            limit: Max number of results (default from config)
            search_mode: "recent", "contextual", "deep", or "full"
            temporal_days: Time window in days (None = use mode default)
            graph_depth: Graph traversal depth (None = use mode default)

        Returns:
            List of matching memories with scores

        Example:
            >>> results = await client.search(
            ...     "What programming languages do I like?", user_id="user123", search_mode="contextual"
            ... )
            >>> for mem in results:
            ...     print(mem["content"], mem["score"])
        """
        float_event("client.search.start", user_id=user_id, query_len=len(query))

        try:
            limit = limit or self.config.default_search_limit
            results = await self.tooling.search_memory(
                query=query,
                user_id=user_id,
                limit=limit,
                search_mode=search_mode,
                temporal_days=temporal_days,
                graph_depth=graph_depth,
            )

            float_event("client.search.success", user_id=user_id, results_count=len(results))

            return results

        except Exception as e:
            float_event("client.search.error", user_id=user_id, error=str(e))
            raise

    async def get(self, memory_id: str) -> dict[str, Any]:
        """
        Get memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory data with all properties

        Example:
            >>> memory = await client.get("mem_123")
            >>> print(memory["content"])
        """
        return await self.tooling.get_memory(memory_id)

    async def update(self, memory_id: str, new_content: str, user_id: str) -> dict[str, Any]:
        """
        Update memory content.

        Args:
            memory_id: Memory identifier
            new_content: New memory content
            user_id: User identifier

        Returns:
            Updated memory data

        Example:
            >>> result = await client.update("mem_123", "I love Python 3.14", user_id="user123")
        """
        return await self.tooling.update_memory(
            memory_id=memory_id, new_content=new_content, user_id=user_id
        )

    async def delete(self, memory_id: str) -> bool:
        """
        Delete memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False otherwise

        Example:
            >>> await client.delete("mem_123")
            True
        """
        return await self.tooling.delete_memory(memory_id)

    async def get_recent(
        self, user_id: str, limit: int = 10, hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Get recent memories for user.

        Args:
            user_id: User identifier
            limit: Max number of results
            hours: Time window in hours

        Returns:
            List of recent memories

        Example:
            >>> recent = await client.get_recent(user_id="user123", limit=5, hours=24)
        """
        return await self.tooling.get_recent_memories(user_id=user_id, limit=limit, hours=hours)

    async def get_graph(
        self, user_id: str, memory_id: str | None = None, depth: int = 2
    ) -> dict[str, Any]:
        """
        Get memory graph (nodes + edges).

        Args:
            user_id: User identifier
            memory_id: Optional starting memory (None = all user memories)
            depth: Graph traversal depth

        Returns:
            Graph structure with nodes and edges

        Example:
            >>> graph = await client.get_graph(user_id="user123", memory_id="mem_123", depth=2)
            >>> print(len(graph["nodes"]), len(graph["edges"]))
        """
        return await self.tooling.get_memory_graph(
            user_id=user_id, memory_id=memory_id, depth=depth
        )

    async def create_agent(
        self,
        agent_id: str,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create new agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            description: Optional description
            metadata: Optional metadata

        Returns:
            Created agent data

        Example:
            >>> agent = await client.create_agent(
            ...     agent_id="assistant_1",
            ...     name="Personal Assistant",
            ...     description="Helps with daily tasks",
            ... )
        """
        return await self.tooling.create_agent(
            agent_id=agent_id, name=name, description=description, metadata=metadata
        )

    async def get_agent_memories(self, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get all memories for specific agent.

        Args:
            agent_id: Agent identifier
            limit: Max number of results

        Returns:
            List of agent's memories
        """
        return await self.tooling.get_agent_memories(agent_id=agent_id, limit=limit)

    async def get_analytics(self) -> AnalyticsSummary:
        """
        Get comprehensive database analytics.

        Returns:
            Complete analytics summary with storage, graph, performance, growth

        Example:
            >>> analytics = await client.get_analytics()
            >>> print(f"Total size: {analytics.storage.total_size_mb:.2f} MB")
            >>> print(f"Total memories: {analytics.storage.total_memories}")
            >>> print(f"Nodes: {analytics.graph.total_nodes}")
        """
        return await self.analytics.collect_all()

    async def get_storage_stats(self):
        """Get storage usage statistics."""
        return await self.analytics.collect_storage_stats()

    async def get_graph_stats(self):
        """Get graph structure statistics."""
        return await self.analytics.collect_graph_stats()

    async def get_category_breakdown(self) -> dict[str, int]:
        """
        Get memory count by category.

        Returns:
            Dict of {category: count}

        Example:
            >>> breakdown = await client.get_category_breakdown()
            >>> print(breakdown)
            {'preference': 42, 'fact': 128, 'goal': 15}
        """
        return await self.analytics.get_category_breakdown()

    async def get_user_breakdown(self) -> dict[str, dict[str, int]]:
        """
        Get memory and storage by user.

        Returns:
            Dict of {user_id: {count: int, size_mb: float}}

        Example:
            >>> breakdown = await client.get_user_breakdown()
            >>> for user_id, stats in breakdown.items():
            ...     print(f"{user_id}: {stats['count']} memories, {stats['size_mb']:.2f} MB")
        """
        return await self.analytics.get_user_breakdown()

    async def close(self):
        """Close all connections and cleanup."""
        await self.db.close()
        await self.embedder.close()
        logger.info("HelixirClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HelixirClient(db={self.config.host}:{self.config.port}, "
            f"llm={self.config.llm_provider}/{self.config.llm_model})"
        )
