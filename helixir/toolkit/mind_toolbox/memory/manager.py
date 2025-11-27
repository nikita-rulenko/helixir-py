"""
MemoryManager: Core memory operations.

Features:
- Uses internal_id from addMemory result for addMemoryEmbedding
- Simple, predictable flow
- Returns Memory object for compatibility with integrator
- Clear error handling
"""

from datetime import UTC, datetime
import logging
from typing import Any
from uuid import uuid4

from helixir.core.client import HelixDBClient
from helixir.llm.embeddings import EmbeddingGenerator
from helixir.toolkit.mind_toolbox.memory.models import Memory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    MemoryManager - core memory operations.

    Example:
        >>> manager = MemoryManager(client, embedder)
        >>> memory_id = await manager.add_memory(
        ...     content="Test",
        ...     user_id="user123",
        ...     memory_type="fact",
        ... )
    """

    def __init__(
        self,
        client: HelixDBClient,
        embedder: EmbeddingGenerator | None = None,
    ):
        """
        Initialize MemoryManager.

        Args:
            client: HelixDB client
            embedder: Embedding generator (optional)
        """
        self.client = client
        self.embedder = embedder

        logger.info(
            "MemoryManager initialized (embedder=%s)", "enabled" if embedder else "disabled"
        )

    async def add_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str = "fact",
        certainty: int = 80,
        importance: int = 50,
        source: str = "user",
        context_tags: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """
        Add memory with embedding.

        Args:
            content: Memory content
            user_id: User ID
            memory_type: Type of memory
            certainty: Certainty score (0-100)
            importance: Importance score (0-100)
            source: Source of memory
            context_tags: Comma-separated tags
            metadata: Additional metadata

        Returns:
            Memory object with internal_id populated

        Raises:
            Exception: If creation fails
        """
        logger.debug(f"Adding memory: {content[:50]}... (user={user_id})")

        memory_id = f"mem_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)

        try:
            mem_result = await self.client.execute_query(
                "addMemory",
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "content": content,
                    "memory_type": memory_type,
                    "created_at": created_at.isoformat(),
                    "updated_at": created_at.isoformat(),
                    "certainty": certainty,
                    "importance": importance,
                    "context_tags": context_tags,
                    "source": source,
                    "metadata": str(metadata or {}),
                },
            )

            internal_id = mem_result.get("memory", {}).get("id")

            if not internal_id:
                raise ValueError(f"addMemory did not return internal ID: {mem_result}")

            logger.debug(f"✅ Memory created: {memory_id} (internal: {internal_id})")

        except Exception as e:
            logger.error(f"Failed to create Memory node: {e}")
            raise

        if self.embedder:
            try:
                vector = await self.embedder.generate(content)

                await self.client.execute_query(
                    "addMemoryEmbedding",
                    {
                        "memory_id": internal_id,
                        "vector_data": vector,
                        "embedding_model": self.embedder.model,
                        "created_at": created_at.isoformat(),
                    },
                )

                logger.debug(f"✅ Embedding created for {memory_id}")

            except Exception as e:
                logger.warning(f"Failed to create embedding for {memory_id}: {e}")

        try:
            try:
                await self.client.execute_query("getUser", {"user_id": user_id})
            except Exception:
                await self.client.execute_query(
                    "addUser",
                    {"user_id": user_id, "name": user_id},
                )
                logger.debug(f"✅ Created user {user_id}")

            await self.client.execute_query(
                "linkUserToMemory",
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "context": "created",
                },
            )
            logger.debug(f"✅ Linked memory {memory_id} to user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to link memory to user: {e}")

        memory_data = mem_result.get("memory", {})

        try:
            db_created_at = datetime.fromisoformat(
                memory_data.get("created_at", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            db_created_at = created_at

        try:
            db_updated_at = datetime.fromisoformat(
                memory_data.get("updated_at", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            db_updated_at = created_at

        memory = Memory(
            memory_id=memory_data.get("memory_id", memory_id),
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            certainty=certainty,
            importance=importance,
            created_at=db_created_at,
            updated_at=db_updated_at,
            context_tags=context_tags,
            source=source,
            metadata=str(metadata or {}),
        )

        memory.__dict__["internal_id"] = internal_id

        logger.debug(f"✅ Memory object created with internal_id: {internal_id}")

        return memory

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """
        Get memory by ID.

        Args:
            memory_id: Memory ID (can be internal or user-facing)

        Returns:
            Memory data or None if not found
        """
        try:
            result = await self.client.execute_query(
                "getMemory",
                {"memory_id": memory_id},
            )
            return result.get("memory")
        except Exception as e:
            logger.warning(f"Failed to get memory {memory_id}: {e}")
            return None

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        vector: list[float] | None = None,
        certainty: int | None = None,
        importance: int | None = None,
        user_id: str | None = None,
        check_contradiction: bool = True,
    ) -> dict[str, Any] | None:
        """
        Update an existing memory using supersession pattern.

        Instead of modifying the old memory, creates a NEW memory with updated content
        and links it to the old one via SUPERSEDES edge. This preserves history and
        maintains graph integrity.

        Args:
            memory_id: Memory ID (user-facing) to update
            content: New content (required for supersession)
            vector: New embedding vector (required if content provided)
            certainty: New certainty (optional, inherits from old)
            importance: New importance (optional, inherits from old)
            user_id: User ID (optional, inherits from old)
            check_contradiction: Whether to check for contradiction with old content

        Returns:
            New memory data with supersession info, or None if failed
        """
        old_memory = await self.get_memory(memory_id)
        if not old_memory:
            logger.warning(f"Memory not found: {memory_id}")
            return None

        old_content = old_memory.get("content", "")
        old_user_id = old_memory.get("user_id", user_id or "unknown")

        if content is None or content == old_content:
            return await self._update_memory_metadata(memory_id, old_memory, certainty, importance)

        new_memory_id = f"mem_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)

        new_certainty = certainty if certainty is not None else old_memory.get("certainty", 80)
        new_importance = importance if importance is not None else old_memory.get("importance", 50)

        try:
            mem_result = await self.client.execute_query(
                "addMemory",
                {
                    "memory_id": new_memory_id,
                    "user_id": user_id or old_user_id,
                    "content": content,
                    "memory_type": old_memory.get("memory_type", "fact"),
                    "certainty": new_certainty,
                    "importance": new_importance,
                    "created_at": created_at.isoformat(),
                    "updated_at": created_at.isoformat(),
                    "context_tags": old_memory.get("context_tags", ""),
                    "source": "supersession",
                    "metadata": f'{{"supersedes": "{memory_id}"}}',
                },
            )

            new_internal_id = mem_result.get("memory", {}).get("id")
            if not new_internal_id:
                raise ValueError(f"Failed to create new memory: {mem_result}")

            logger.debug(f"✅ Created new memory: {new_memory_id} (supersedes {memory_id})")

            if vector and self.embedder:
                await self.client.execute_query(
                    "addMemoryEmbedding",
                    {
                        "memory_id": new_internal_id,
                        "vector_data": vector,
                        "embedding_model": self.embedder.model,
                        "created_at": created_at.isoformat(),
                    },
                )
                logger.debug(f"✅ Created embedding for new memory: {new_memory_id}")

            try:
                await self.client.execute_query(
                    "linkUserToMemory",
                    {
                        "user_id": user_id or old_user_id,
                        "memory_id": new_memory_id,
                        "context": "supersession",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to link user to new memory: {e}")

            is_contradiction = 0
            if check_contradiction:
                is_contradiction = (
                    1 if self._content_differs_significantly(old_content, content) else 0
                )

            await self.client.execute_query(
                "addMemorySupersession",
                {
                    "new_id": new_memory_id,
                    "old_id": memory_id,
                    "reason": "content_update",
                    "superseded_at": created_at.isoformat(),
                    "is_contradiction": is_contradiction,
                },
            )
            logger.debug(f"✅ Created SUPERSEDES edge: {new_memory_id} -> {memory_id}")

            if is_contradiction:
                try:
                    await self.client.execute_query(
                        "addMemoryContradiction",
                        {
                            "from_id": new_memory_id,
                            "to_id": memory_id,
                            "resolution": "superseded",
                            "resolved": 1,
                            "resolution_strategy": "newer_wins",
                        },
                    )
                    logger.debug(f"✅ Created CONTRADICTS edge: {new_memory_id} -> {memory_id}")
                except Exception as e:
                    logger.warning(f"Failed to create CONTRADICTS edge: {e}")

            await self._copy_reasoning_relations(memory_id, new_memory_id)

            return {
                "memory_id": new_memory_id,
                "content": content,
                "supersedes": memory_id,
                "is_contradiction": bool(is_contradiction),
                "created_at": created_at.isoformat(),
                **mem_result.get("memory", {}),
            }

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return None

    async def _update_memory_metadata(
        self,
        memory_id: str,
        memory_data: dict[str, Any],
        certainty: int | None,
        importance: int | None,
    ) -> dict[str, Any] | None:
        """Update only metadata (certainty, importance) without creating new memory."""
        internal_id = memory_data.get("id")
        if not internal_id:
            return None

        update_certainty = certainty if certainty is not None else memory_data.get("certainty", 80)
        update_importance = (
            importance if importance is not None else memory_data.get("importance", 50)
        )
        updated_at = datetime.now(UTC).isoformat()

        try:
            result = await self.client.execute_query(
                "updateMemoryById",
                {
                    "id": internal_id,
                    "content": memory_data.get("content", ""),
                    "certainty": update_certainty,
                    "importance": update_importance,
                    "updated_at": updated_at,
                },
            )
            logger.debug(f"✅ Updated memory metadata: {memory_id}")
            return result.get("updated")
        except Exception as e:
            logger.error(f"Failed to update memory metadata {memory_id}: {e}")
            return None

    def _content_differs_significantly(self, old: str, new: str) -> bool:
        """Simple heuristic to detect if content change is significant (potential contradiction)."""
        negation_words = [
            "not",
            "never",
            "don't",
            "doesn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "no longer",
            "actually",
            "but",
            "however",
            "instead",
        ]
        new_lower = new.lower()

        for word in negation_words:
            if word in new_lower and word not in old.lower():
                return True

        if ("love" in old.lower() and "hate" in new_lower) or (
            "hate" in old.lower() and "love" in new_lower
        ):
            return True
        if ("best" in old.lower() and "worst" in new_lower) or (
            "worst" in old.lower() and "best" in new_lower
        ):
            return True
        if ("prefer" in old.lower() and "avoid" in new_lower) or (
            "avoid" in old.lower() and "prefer" in new_lower
        ):
            return True

        return False

    async def _copy_reasoning_relations(self, old_memory_id: str, new_memory_id: str) -> None:
        """Copy reasoning relations from old memory to new memory."""
        try:
            outgoing = await self.client.execute_query(
                "getMemoryOutgoingRelations",
                {"memory_id": old_memory_id},
            )

            for edge in outgoing.get("implies_out", []):
                target_id = edge.get("to", {}).get("memory_id")
                if target_id:
                    try:
                        await self.client.execute_query(
                            "addMemoryImplication",
                            {
                                "from_id": new_memory_id,
                                "to_id": target_id,
                                "probability": edge.get("probability", 80),
                                "reasoning_id": f"copied_from_{old_memory_id}",
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to copy IMPLIES relation: {e}")

            for edge in outgoing.get("because_out", []):
                target_id = edge.get("to", {}).get("memory_id")
                if target_id:
                    try:
                        await self.client.execute_query(
                            "addMemoryCausation",
                            {
                                "from_id": new_memory_id,
                                "to_id": target_id,
                                "strength": edge.get("strength", 80),
                                "reasoning_id": f"copied_from_{old_memory_id}",
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to copy BECAUSE relation: {e}")

            for edge in outgoing.get("relations_out", []):
                target_id = edge.get("to", {}).get("memory_id")
                if target_id:
                    try:
                        await self.client.execute_query(
                            "addMemoryRelation",
                            {
                                "source_id": new_memory_id,
                                "target_id": target_id,
                                "relation_type": edge.get("relation_type", "related"),
                                "strength": edge.get("strength", 50),
                                "created_at": datetime.now(UTC).isoformat(),
                                "metadata": f'{{"copied_from": "{old_memory_id}"}}',
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to copy MEMORY_RELATION: {e}")

            logger.debug(f"✅ Copied reasoning relations from {old_memory_id} to {new_memory_id}")

        except Exception as e:
            logger.warning(f"Failed to copy reasoning relations: {e}")

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Note: Currently only soft-delete (marks as deleted).
        Full deletion with edge cleanup requires DeletionManager.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted successfully
        """
        logger.warning(f"delete_memory({memory_id}) - NOT IMPLEMENTED")
        return False

    async def link_memory_to_concept(
        self,
        memory_id: str,
        concept_id: str,
        confidence: int = 80,
        link_type: str = "INSTANCE_OF",
    ) -> bool:
        """
        Link a memory to a concept.

        Args:
            memory_id: Memory ID
            concept_id: Concept ID
            confidence: Confidence level (0-100)
            link_type: Type of link ("INSTANCE_OF" or "BELONGS_TO_CATEGORY")

        Returns:
            True if linked successfully
        """
        if link_type not in ("INSTANCE_OF", "BELONGS_TO_CATEGORY"):
            logger.error(f"Invalid link_type: {link_type}")
            return False

        try:
            if link_type == "INSTANCE_OF":
                await self.client.execute_query(
                    "linkMemoryToInstanceOf",
                    {
                        "memory_id": memory_id,
                        "concept_id": concept_id,
                        "confidence": confidence,
                    },
                )
            else:
                await self.client.execute_query(
                    "linkMemoryToCategory",
                    {
                        "memory_id": memory_id,
                        "concept_id": concept_id,
                        "relevance": confidence,
                    },
                )

            logger.debug(f"✅ Linked memory {memory_id[:8]} to concept {concept_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to link memory to concept: {e}")
            return False

    async def add_memory_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "RELATED_TO",
        strength: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Add a relation between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relation
            strength: Relation strength (0-100)
            metadata: Additional metadata

        Returns:
            Relation data or None if failed
        """
        try:
            result = await self.client.execute_query(
                "addMemoryRelation",
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "strength": strength,
                    "created_at": datetime.now(UTC).isoformat(),
                    "metadata": str(metadata or {}),
                },
            )
            logger.debug(f"✅ Added relation: {source_id[:8]} -> {target_id[:8]}")
            return result.get("relation")

        except Exception as e:
            logger.warning(f"Failed to add memory relation: {e}")
            return None
