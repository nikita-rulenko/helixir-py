"""
ChainBuilder - Builds logical chains through memory graph.
"""

import logging
from typing import TYPE_CHECKING, Any

from helixir.toolkit.mind_toolbox.memory_chain.models import (
    ChainDirection,
    ChainNode,
    LogicalRelationType,
    MemoryChain,
    MemoryChainConfig,
)

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


class ChainBuilder:
    """
    Builds chains of memories connected by logical relations.

    Algorithm:
    1. Start from seed memory (from vector search)
    2. Query logical connections (IMPLIES, BECAUSE, CONTRADICTS)
    3. Select strongest/most relevant edge
    4. Move to connected memory
    5. Repeat until termination condition
    6. Return chain with reasoning trail
    """

    def __init__(
        self,
        client: HelixDBClient,
        config: MemoryChainConfig | None = None,
    ) -> None:
        """
        Initialize ChainBuilder.

        Args:
            client: HelixDB client
            config: Chain building configuration
        """
        self.client = client
        self.config = config or MemoryChainConfig.default()

        logger.info(
            "ChainBuilder initialized (max_depth=%d, direction=%s)",
            self.config.max_depth,
            self.config.direction.value,
        )

    async def build_chain(
        self,
        seed_memory_id: str,
        seed_content: str,
        seed_type: str = "fact",
        query: str = "",
    ) -> MemoryChain:
        """
        Build a logical chain starting from seed memory.

        Args:
            seed_memory_id: Starting memory ID
            seed_content: Content of seed memory
            seed_type: Type of seed memory
            query: Original search query (for context)

        Returns:
            MemoryChain with connected memories
        """
        logger.info("Building chain from seed: %s", seed_memory_id)

        seed = ChainNode(
            memory_id=seed_memory_id,
            content=seed_content,
            memory_type=seed_type,
            depth=0,
        )

        chain = MemoryChain(seed=seed, query=query)

        visited: set[str] = {seed_memory_id}

        await self._traverse_chain(
            chain=chain,
            current_id=seed_memory_id,
            current_depth=0,
            visited=visited,
        )

        logger.info(
            "Chain built: %d nodes, type=%s, depth=%d",
            len(chain.nodes),
            chain.chain_type,
            chain.total_depth,
        )

        return chain

    async def _traverse_chain(
        self,
        chain: MemoryChain,
        current_id: str,
        current_depth: int,
        visited: set[str],
    ) -> None:
        """
        Recursively traverse logical edges to build chain.
        """
        if current_depth >= self.config.max_depth:
            logger.debug("Max depth reached: %d", current_depth)
            return

        if len(chain.nodes) >= self.config.max_nodes:
            logger.debug("Max nodes reached: %d", len(chain.nodes))
            return

        connections = await self._get_logical_connections(current_id)

        edges_to_follow: list[dict[str, Any]] = []

        if self.config.direction in (ChainDirection.FORWARD, ChainDirection.BOTH):
            if self.config.follow_implies:
                for mem in connections.get("implies_out", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.IMPLIES,
                            "direction": "out",
                        }
                    )

            if self.config.follow_because:
                for mem in connections.get("because_out", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.BECAUSE,
                            "direction": "out",
                        }
                    )

            if self.config.follow_contradicts:
                for mem in connections.get("contradicts_out", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.CONTRADICTS,
                            "direction": "out",
                        }
                    )

            if self.config.follow_generic:
                for mem in connections.get("relation_out", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.RELATES_TO,
                            "direction": "out",
                        }
                    )

        if self.config.direction in (ChainDirection.BACKWARD, ChainDirection.BOTH):
            if self.config.follow_implies:
                for mem in connections.get("implies_in", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.IMPLIES,
                            "direction": "in",
                        }
                    )

            if self.config.follow_because:
                for mem in connections.get("because_in", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.BECAUSE,
                            "direction": "in",
                        }
                    )

            if self.config.follow_contradicts:
                for mem in connections.get("contradicts_in", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.CONTRADICTS,
                            "direction": "in",
                        }
                    )

            if self.config.follow_generic:
                for mem in connections.get("relation_in", []):
                    edges_to_follow.append(
                        {
                            "memory": mem,
                            "relation": LogicalRelationType.RELATES_TO,
                            "direction": "in",
                        }
                    )

        edges_to_follow = [
            e for e in edges_to_follow if e["memory"].get("memory_id") not in visited
        ]

        if not edges_to_follow:
            logger.debug("No more edges to follow from %s", current_id)
            return

        def edge_priority(e: dict) -> int:
            priorities = {
                LogicalRelationType.BECAUSE: 0,
                LogicalRelationType.IMPLIES: 1,
                LogicalRelationType.CONTRADICTS: 2,
                LogicalRelationType.RELATES_TO: 3,
            }
            return priorities.get(e["relation"], 99)

        edges_to_follow.sort(key=edge_priority)

        if not self.config.allow_branching:
            edges_to_follow = edges_to_follow[:1]
        else:
            edges_to_follow = edges_to_follow[: self.config.max_branches]

        for edge in edges_to_follow:
            mem = edge["memory"]
            mem_id = mem.get("memory_id", "")

            if not mem_id or mem_id in visited:
                continue

            visited.add(mem_id)

            node = ChainNode(
                memory_id=mem_id,
                content=mem.get("content", ""),
                memory_type=mem.get("memory_type", "fact"),
                relation_type=edge["relation"],
                relation_direction=edge["direction"],
                depth=current_depth + 1,
                raw_data=mem,
            )

            chain.add_node(node)

            logger.debug(
                "Added node: %s (%s %s) at depth %d",
                mem_id[:20],
                edge["direction"],
                edge["relation"].value,
                current_depth + 1,
            )

            await self._traverse_chain(
                chain=chain,
                current_id=mem_id,
                current_depth=current_depth + 1,
                visited=visited,
            )

    async def _get_logical_connections(self, memory_id: str) -> dict[str, list]:
        """
        Get all logical connections for a memory.

        Returns dict with keys:
        - implies_out, implies_in
        - because_out, because_in
        - contradicts_out, contradicts_in
        - relation_out, relation_in
        """
        try:
            return await self.client.execute_query(
                "getMemoryLogicalConnections",
                {"memory_id": memory_id},
            )
        except Exception as e:
            logger.warning("Failed to get connections for %s: %s", memory_id, e)
            return {}

    async def build_chains_from_seeds(
        self,
        seeds: list[dict[str, Any]],
        query: str = "",
    ) -> list[MemoryChain]:
        """
        Build chains from multiple seed memories.

        Args:
            seeds: List of seed memories from vector search
            query: Original search query

        Returns:
            List of MemoryChain objects
        """
        chains = []

        for seed in seeds:
            memory_id = seed.get("memory_id", "")
            if not memory_id:
                continue

            chain = await self.build_chain(
                seed_memory_id=memory_id,
                seed_content=seed.get("content", ""),
                seed_type=seed.get("memory_type", "fact"),
                query=query,
            )

            if chain.nodes:
                chains.append(chain)

        logger.info("Built %d chains from %d seeds", len(chains), len(seeds))
        return chains
