"""Reasoning Engine for managing logical relationships between memories."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from helixir.core.exceptions import ReasoningError, ValidationError
from helixir.toolkit.mind_toolbox.reasoning.models import (
    ReasoningChain,
    ReasoningRelation,
    ReasoningType,
)

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Manager for reasoning operations.

    Handles:
    - Creating reasoning relations (IMPLIES, BECAUSE, etc.)
    - Building reasoning chains
    - Graph traversal
    - Basic inference
    """

    def __init__(
        self,
        client: HelixDBClient,
        llm_provider: LLMProvider | None = None,
        cache_size: int = 1000,
        enable_warmup: bool = True,
    ) -> None:
        """
        Initialize ReasoningEngine with cache-aside pattern.

        Args:
            client: HelixDBClient instance
            llm_provider: Optional LLM provider for intelligent relation inference
            cache_size: Maximum cache size (LRU eviction)
            enable_warmup: Whether to warm up cache on startup
        """
        self.client = client
        self.llm_provider = llm_provider
        self._relation_cache: dict[str, ReasoningRelation] = {}
        self._cache_size = cache_size
        self._enable_warmup = enable_warmup
        self._is_warmed_up = False
        logger.info(
            "ReasoningEngine initialized (cache_size=%d, warmup=%s, llm=%s)",
            cache_size,
            enable_warmup,
            "enabled" if llm_provider else "disabled",
        )

    def _add_to_cache(self, relation: ReasoningRelation) -> None:
        """
        Add relation to cache with LRU eviction.

        Args:
            relation: ReasoningRelation to cache
        """
        if len(self._relation_cache) >= self._cache_size:
            oldest_key = next(iter(self._relation_cache))
            del self._relation_cache[oldest_key]
            logger.debug("Cache eviction: removed relation %s", oldest_key[:8])

        self._relation_cache[relation.relation_id] = relation

    async def warm_up_cache(self, memory_id: str | None = None, limit: int = 100) -> int:
        """
        Warm up cache by preloading reasoning relations from HelixDB.

        Args:
            memory_id: Optional memory ID to warm up specific memory's relations
            limit: Maximum number of relations to preload

        Returns:
            Number of relations loaded into cache
        """
        if self._is_warmed_up:
            logger.info("Reasoning cache already warmed up, skipping")
            return len(self._relation_cache)

        logger.info("Warming up reasoning cache (memory=%s, limit=%d)", memory_id, limit)

        try:
            query_params = {"limit": limit}
            if memory_id:
                query_params["memory_id"] = memory_id

            result = await self.client.execute_query("getRecentRelations", query_params)

            if result and "relations" in result:
                for relation_data in result["relations"]:
                    relation = ReasoningRelation.model_validate(relation_data)
                    self._add_to_cache(relation)

            self._is_warmed_up = True
            loaded_count = len(self._relation_cache)
            logger.info("Reasoning cache warm-up complete: %d relations loaded", loaded_count)
            return loaded_count

        except Exception as e:
            logger.warning("Reasoning cache warm-up failed: %s, continuing with empty cache", e)
            return 0

    async def add_relation(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relation_type: ReasoningType,
        strength: int = 50,
        confidence: int = 50,
        explanation: str | None = None,
        created_by: str = "system",
    ) -> ReasoningRelation:
        """
        Add a reasoning relation between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation_type: Type of relation (IMPLIES, BECAUSE, etc.)
            strength: Strength of relation (0-100)
            confidence: Confidence in relation (0-100)
            explanation: Optional explanation
            created_by: Creator identifier

        Returns:
            Created ReasoningRelation

        Raises:
            ValidationError: If validation fails
            ReasoningError: If creation fails
        """
        if from_memory_id == to_memory_id:
            msg = "Cannot create self-referential reasoning relation"
            raise ValidationError(msg)

        if not (0 <= strength <= 100) or not (0 <= confidence <= 100):
            msg = "Strength and confidence must be between 0 and 100"
            raise ValidationError(msg)

        try:
            relation = ReasoningRelation(
                from_memory_id=from_memory_id,
                to_memory_id=to_memory_id,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence,
                explanation=explanation,
                created_by=created_by,
            )

            await self.client.execute_query(
                "addReasoningRelation",
                {
                    "relation_id": relation.relation_id,
                    "from_memory_id": from_memory_id,
                    "to_memory_id": to_memory_id,
                    "relation_type": relation_type,
                    "strength": strength,
                    "confidence": confidence,
                    "explanation": explanation or "",
                    "created_by": created_by,
                    "created_at": relation.created_at.isoformat(),
                },
            )

            self._add_to_cache(relation)

            logger.info(
                "Added %s relation in DB and cache: %s → %s (strength=%d)",
                relation_type,
                from_memory_id[:8],
                to_memory_id[:8],
                strength,
            )

            return relation

        except ValidationError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to persist reasoning relation to HelixDB: %s, adding to cache only", e
            )
            self._add_to_cache(relation)
            logger.info(
                "Added %s relation in cache only: %s → %s (strength=%d)",
                relation_type,
                from_memory_id[:8],
                to_memory_id[:8],
                strength,
            )
            return relation

    async def get_relations(
        self,
        memory_id: str,
        relation_type: ReasoningType | None = None,
        direction: str = "outgoing",
    ) -> list[ReasoningRelation]:
        """
        Get all relations for a memory with cache-aside fallback.

        Args:
            memory_id: Memory ID
            relation_type: Optional filter by relation type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of ReasoningRelation objects
        """
        if direction not in {"outgoing", "incoming", "both"}:
            msg = f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'"
            raise ValidationError(msg)

        cached_relations = []
        for relation in self._relation_cache.values():
            if (direction == "outgoing" and relation.from_memory_id == memory_id) or (
                direction == "incoming" and relation.to_memory_id == memory_id
            ):
                if relation_type is None or relation.relation_type == relation_type:
                    cached_relations.append(relation)
            elif direction == "both":
                if memory_id in (relation.from_memory_id, relation.to_memory_id):
                    if relation_type is None or relation.relation_type == relation_type:
                        cached_relations.append(relation)

        if cached_relations:
            logger.debug(
                "Cache HIT: found %d relations for %s", len(cached_relations), memory_id[:8]
            )
            return cached_relations

        logger.debug("Cache MISS: querying HelixDB for relations of %s", memory_id[:8])

        try:
            result = await self.client.execute_query(
                "getReasoningRelations",
                {
                    "memory_id": memory_id,
                    "relation_type": relation_type or "",
                    "direction": direction,
                },
            )

            relations = []
            if result and "relations" in result:
                for rel_data in result["relations"]:
                    relation = ReasoningRelation(
                        relation_id=rel_data["relation_id"],
                        from_memory_id=rel_data["from_memory_id"],
                        to_memory_id=rel_data["to_memory_id"],
                        relation_type=rel_data["relation_type"],
                        strength=rel_data.get("strength", 50),
                        confidence=rel_data.get("confidence", 50),
                        explanation=rel_data.get("explanation"),
                        created_by=rel_data.get("created_by"),
                    )
                    relations.append(relation)
                    self._add_to_cache(relation)

            logger.debug("Loaded %d relations from DB for %s", len(relations), memory_id[:8])
            return relations

        except Exception as e:
            logger.warning(
                "Failed to get relations for memory %s: %s, returning empty list", memory_id, e
            )
            return []

    async def build_chain(
        self,
        start_memory_id: str,
        relation_type: ReasoningType | None = None,
        max_depth: int = 5,
    ) -> ReasoningChain:
        """
        Build a reasoning chain starting from a memory.

        Args:
            start_memory_id: Starting memory ID
            relation_type: Optional filter for relation types
            max_depth: Maximum chain depth

        Returns:
            ReasoningChain object
        """
        memory_ids = [start_memory_id]
        relation_types = []
        current_id = start_memory_id
        total_strength = 1.0

        try:
            for _ in range(max_depth):
                relations = await self.get_relations(
                    current_id, relation_type=relation_type, direction="outgoing"
                )

                if not relations:
                    break

                best_relation = max(relations, key=lambda r: r.strength)

                memory_ids.append(best_relation.to_memory_id)
                relation_types.append(best_relation.relation_type)
                total_strength *= best_relation.strength / 100.0

                current_id = best_relation.to_memory_id

            chain = ReasoningChain(
                memory_ids=memory_ids,
                relation_types=relation_types,
                total_strength=total_strength,
                length=len(memory_ids),
            )

            logger.debug(
                "Built reasoning chain: %d memories, strength=%.2f",
                chain.length,
                chain.total_strength,
            )

            return chain

        except Exception as e:
            msg = f"Failed to build reasoning chain: {e}"
            raise ReasoningError(msg) from e

    async def traverse_graph(
        self,
        start_memory_id: str,
        relation_types: list[ReasoningType] | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Traverse the reasoning graph starting from a memory.

        Args:
            start_memory_id: Starting memory ID
            relation_types: Optional list of relation types to follow
            max_depth: Maximum traversal depth

        Returns:
            Dict with graph structure: {memory_id: {relations: [...], children: [...]}}
        """
        graph: dict[str, Any] = {}
        visited = set()
        queue = [(start_memory_id, 0)]

        try:
            while queue:
                current_id, depth = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                all_relations = []
                if relation_types:
                    for rel_type in relation_types:
                        relations = await self.get_relations(
                            current_id, relation_type=rel_type, direction="outgoing"
                        )
                        all_relations.extend(relations)
                else:
                    all_relations = await self.get_relations(current_id, direction="outgoing")

                graph[current_id] = {
                    "relations": [
                        {
                            "to": rel.to_memory_id,
                            "type": rel.relation_type,
                            "strength": rel.strength,
                        }
                        for rel in all_relations
                    ],
                    "depth": depth,
                }

                for rel in all_relations:
                    if rel.to_memory_id not in visited:
                        queue.append((rel.to_memory_id, depth + 1))

            logger.debug(
                "Traversed graph from %s: %d nodes visited",
                start_memory_id[:8],
                len(graph),
            )

            return graph

        except Exception as e:
            msg = f"Failed to traverse graph: {e}"
            raise ReasoningError(msg) from e

    async def find_implications(
        self,
        memory_id: str,
        min_confidence: int = 50,
    ) -> list[tuple[str, int]]:
        """
        Find all memories implied by a given memory.

        Args:
            memory_id: Source memory ID
            min_confidence: Minimum confidence threshold

        Returns:
            List of (memory_id, confidence) tuples
        """
        implications = []

        try:
            relations = await self.get_relations(
                memory_id, relation_type="IMPLIES", direction="outgoing"
            )

            for rel in relations:
                if rel.confidence >= min_confidence:
                    implications.append((rel.to_memory_id, rel.confidence))

            implications.sort(key=lambda x: x[1], reverse=True)

            logger.debug("Found %d implications for memory %s", len(implications), memory_id[:8])

            return implications

        except Exception as e:
            logger.warning("Failed to find implications: %s", e)
            return []

    async def find_causes(
        self,
        memory_id: str,
        min_strength: int = 50,
    ) -> list[tuple[str, int]]:
        """
        Find all memories that cause a given memory (via BECAUSE relation).

        Args:
            memory_id: Target memory ID
            min_strength: Minimum strength threshold

        Returns:
            List of (memory_id, strength) tuples
        """
        causes = []

        try:
            relations = await self.get_relations(
                memory_id, relation_type="BECAUSE", direction="incoming"
            )

            for rel in relations:
                if rel.strength >= min_strength:
                    causes.append((rel.from_memory_id, rel.strength))

            causes.sort(key=lambda x: x[1], reverse=True)

            logger.debug("Found %d causes for memory %s", len(causes), memory_id[:8])

            return causes

        except Exception as e:
            logger.warning("Failed to find causes: %s", e)
            return []

    async def supersede_memory(
        self,
        old_memory_id: str,
        new_memory_id: str,
        reason: str | None = None,
    ) -> bool:
        """
        Mark one memory as superseding another.

        Args:
            old_memory_id: Memory being replaced
            new_memory_id: New memory
            reason: Optional reason for supersession

        Returns:
            True if successful
        """
        try:
            await self.add_relation(
                from_memory_id=new_memory_id,
                to_memory_id=old_memory_id,
                relation_type="SUPERSEDES",
                strength=100,
                confidence=100,
                explanation=reason,
            )

            logger.info("Memory %s supersedes %s", new_memory_id[:8], old_memory_id[:8])

            return True

        except Exception as e:
            logger.warning("Failed to supersede memory: %s", e)
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"ReasoningEngine(cached_relations={len(self._relation_cache)})"

    async def infer_relation(
        self,
        source_content: str,
        target_content: str,
        similarity_score: float,
    ) -> dict[str, Any] | None:
        """
        Infer the type of logical relation between two memories using LLM.

        Uses LLM to determine if memories have logical relationships:
        - IMPLIES: Source logically implies target
        - BECAUSE: Source is caused by target
        - SUPERSEDES: Source replaces/updates target
        - CONTRADICTS: Source contradicts target
        - RELATES_TO: General semantic relatedness (fallback)

        Args:
            source_content: Content of the new memory
            target_content: Content of the existing similar memory
            similarity_score: Semantic similarity score (0.0-1.0)

        Returns:
            Dict with:
                - type: Relation type (str)
                - confidence: Confidence score (0.0-1.0)
                - reasoning: Explanation (str)
            Or None if no relation should be created
        """
        if not self.llm_provider:
            if similarity_score >= 0.75:
                return {
                    "type": "RELATES_TO",
                    "confidence": similarity_score,
                    "reasoning": f"High semantic similarity: {similarity_score:.2f}",
                }
            return None

        try:
            user_prompt = f"""Analyze the logical relationship between these two memories:

NEW MEMORY: {source_content}

EXISTING MEMORY: {target_content}

Semantic similarity: {similarity_score:.2f}

Determine the STRONGEST logical relationship:
- IMPLIES: New memory logically implies the existing one (A→B)
- BECAUSE: New memory is caused by/justified by the existing one (A←B)
- SUPERSEDES: New memory replaces/updates the existing one (newer information)
- CONTRADICTS: Memories contradict each other (conflict)
- RELATES_TO: General semantic relatedness (use only if no stronger relation exists)
- NONE: No meaningful relation

Respond ONLY with valid JSON:
{{"type": "IMPLIES|BECAUSE|SUPERSEDES|CONTRADICTS|RELATES_TO|NONE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            system_prompt = "You are an expert at analyzing logical relationships between facts. Be precise and conservative. Only suggest strong relations when clearly justified."

            response, _metadata = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json_object",
            )

            import json

            try:
                result = json.loads(response.strip())

                relation_type = result.get("type", "").upper()
                if relation_type == "NONE":
                    return None

                if relation_type not in {
                    "IMPLIES",
                    "BECAUSE",
                    "SUPERSEDES",
                    "CONTRADICTS",
                    "RELATES_TO",
                }:
                    logger.warning(
                        "LLM returned invalid relation type: %s, falling back to RELATES_TO",
                        relation_type,
                    )
                    relation_type = "RELATES_TO"

                confidence = float(result.get("confidence", similarity_score))
                reasoning = result.get("reasoning", "LLM inference")

                logger.debug(
                    "Inferred %s relation (confidence=%.2f): %s",
                    relation_type,
                    confidence,
                    reasoning[:50],
                )

                return {
                    "type": relation_type,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to parse LLM response: %s, falling back to similarity-based",
                    e,
                )
                if similarity_score >= 0.75:
                    return {
                        "type": "RELATES_TO",
                        "confidence": similarity_score,
                        "reasoning": f"LLM parse failed, using similarity: {similarity_score:.2f}",
                    }
                return None

        except Exception as e:
            logger.warning("LLM relation inference failed: %s, using fallback", e)
            if similarity_score >= 0.75:
                return {
                    "type": "RELATES_TO",
                    "confidence": similarity_score,
                    "reasoning": f"LLM failed, using similarity: {similarity_score:.2f}",
                }
            return None
