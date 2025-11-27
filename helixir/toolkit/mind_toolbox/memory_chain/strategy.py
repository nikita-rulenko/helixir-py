"""
MemoryChainStrategy - Search strategy that builds logical chains.
"""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

from helixir.toolkit.mind_toolbox.memory_chain.builder import ChainBuilder
from helixir.toolkit.mind_toolbox.memory_chain.models import MemoryChain, MemoryChainConfig

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.llm.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class ChainSearchResult:
    """Result of chain-based memory search."""

    query: str
    chains: list[MemoryChain] = field(default_factory=list)

    total_memories: int = 0
    total_chains: int = 0
    deepest_chain: int = 0

    memories: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._calculate_stats()

    def _calculate_stats(self) -> None:
        """Calculate aggregate statistics."""
        self.total_chains = len(self.chains)
        self.total_memories = sum(len(c.get_all_memories()) for c in self.chains)
        self.deepest_chain = max((c.total_depth for c in self.chains), default=0)

        seen = set()
        for chain in self.chains:
            for node in chain.get_all_memories():
                if node.memory_id not in seen:
                    seen.add(node.memory_id)
                    self.memories.append(
                        {
                            "memory_id": node.memory_id,
                            "content": node.content,
                            "memory_type": node.memory_type,
                            "chain_depth": node.depth,
                            "relation_type": node.relation_type.value
                            if node.relation_type
                            else None,
                        }
                    )

    def get_reasoning_trails(self) -> str:
        """Get all reasoning trails as formatted string."""
        trails = []
        for i, chain in enumerate(self.chains, 1):
            trails.append(f"=== Chain {i} ({chain.chain_type}) ===")
            trails.append(chain.get_reasoning_trail())
            trails.append("")
        return "\n".join(trails)


class MemoryChainStrategy:
    """
    Search strategy that builds logical chains through memory graph.

    Unlike flat search strategies, this follows logical relations
    (IMPLIES, BECAUSE, CONTRADICTS) to build reasoning trails.

    Usage:
        strategy = MemoryChainStrategy(client, embedder)
        result = await strategy.search("Why do I use Rust?", user_id="user1")

        for chain in result.chains:
            print(chain.get_reasoning_trail())
    """

    def __init__(
        self,
        client: HelixDBClient,
        embedder: EmbeddingGenerator,
        config: MemoryChainConfig | None = None,
    ) -> None:
        """
        Initialize MemoryChainStrategy.

        Args:
            client: HelixDB client
            embedder: Embedding generator for vector search
            config: Chain building configuration
        """
        self.client = client
        self.embedder = embedder
        self.config = config or MemoryChainConfig.default()

        self.chain_builder = ChainBuilder(client, self.config)

        logger.info("MemoryChainStrategy initialized")

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        config: MemoryChainConfig | None = None,
    ) -> ChainSearchResult:
        """
        Search for memories and build logical chains.

        Args:
            query: Search query
            user_id: Optional user ID filter
            limit: Max number of seed memories
            config: Optional override config

        Returns:
            ChainSearchResult with reasoning chains
        """
        logger.info("Chain search: '%s' (limit=%d)", query[:50], limit)

        if config:
            self.chain_builder.config = config

        seeds = await self._vector_search(query, limit)

        if not seeds:
            logger.warning("No seeds found for query: %s", query[:50])
            return ChainSearchResult(query=query)

        logger.info("Found %d seed memories", len(seeds))

        chains = await self.chain_builder.build_chains_from_seeds(
            seeds=seeds,
            query=query,
        )

        chains.sort(key=lambda c: (len(c.nodes), c.total_depth), reverse=True)

        result = ChainSearchResult(query=query, chains=chains)

        logger.info(
            "Chain search complete: %d chains, %d total memories, max depth %d",
            result.total_chains,
            result.total_memories,
            result.deepest_chain,
        )

        return result

    async def _vector_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Perform vector search to find seed memories.
        """
        try:
            embedding = await self.embedder.generate(query)

            result = await self.client.execute_query(
                "smartVectorSearchWithChunks",
                {
                    "query_vector": embedding,
                    "limit": limit,
                },
            )

            memories = []
            seen = set()

            for mem in result.get("memories", []):
                mem_id = mem.get("memory_id")
                if mem_id and mem_id not in seen:
                    seen.add(mem_id)
                    memories.append(mem)

            for mem in result.get("parent_memories", []):
                mem_id = mem.get("memory_id")
                if mem_id and mem_id not in seen:
                    seen.add(mem_id)
                    memories.append(mem)

            return memories

        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            return []

    async def search_causal(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> ChainSearchResult:
        """
        Search with causal chain focus (follow BECAUSE relations).

        Good for "Why?" questions.
        """
        return await self.search(
            query=query,
            user_id=user_id,
            limit=limit,
            config=MemoryChainConfig.causal_only(),
        )

    async def search_implications(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> ChainSearchResult:
        """
        Search with implication chain focus (follow IMPLIES relations).

        Good for "What follows?" questions.
        """
        return await self.search(
            query=query,
            user_id=user_id,
            limit=limit,
            config=MemoryChainConfig.implications_only(),
        )

    async def search_deep(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> ChainSearchResult:
        """
        Deep search following all relation types.

        Good for comprehensive context gathering.
        """
        return await self.search(
            query=query,
            user_id=user_id,
            limit=limit,
            config=MemoryChainConfig.deep_context(),
        )
