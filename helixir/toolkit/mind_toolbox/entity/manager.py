"""Entity Manager for CRUD operations on entities."""

import json
import logging
from typing import TYPE_CHECKING, Any

from helixir.core.exceptions import HelixMemoryOperationError, ValidationError
from helixir.toolkit.mind_toolbox.memory.models import Entity, EntityType
from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


class EntityManager:
    """
    Manager for entity CRUD operations.

    Handles:
    - Creating, reading, updating entities
    - Entity deduplication
    - Linking entities to memories (EXTRACTED_ENTITY, MENTIONS edges)
    """

    def __init__(self, client: HelixDBClient, cache_size: int = 1000) -> None:
        """
        Initialize EntityManager with cache-aside pattern.

        Args:
            client: HelixDBClient instance
            cache_size: Maximum cache size (LRU eviction)
        """
        self.client = client
        self._entity_cache: dict[str, Entity] = {}
        self._name_to_id: dict[str, str] = {}
        self._cache_size = cache_size

        logger.info("EntityManager initialized (cache_size=%d)", cache_size)

    def _add_to_cache(self, entity: Entity) -> None:
        """
        Add entity to cache with LRU eviction.

        Args:
            entity: Entity object to cache
        """
        if len(self._entity_cache) >= self._cache_size:
            oldest_id = next(iter(self._entity_cache))
            evicted = self._entity_cache.pop(oldest_id)
            if evicted.name in self._name_to_id:
                del self._name_to_id[evicted.name]
            logger.debug("Cache eviction: %s (size: %d)", oldest_id, self._cache_size)

        self._entity_cache[entity.entity_id] = entity
        self._name_to_id[entity.name.lower()] = entity.entity_id

    async def create_entity(
        self, name: str, entity_type: str, properties: dict[str, Any] | None = None
    ) -> Entity:
        """
        Create a new entity.

        Args:
            name: Entity name
            entity_type: Type (Person, Organization, Location, System, Component, Resource, Concept, Process, Event)
            properties: Optional properties with subtype/technology/domain metadata

        Returns:
            Created Entity object

        Raises:
            ValidationError: If validation fails
            HelixMemoryOperationError: If creation fails
        """
        float_event("entity.create.start", name=name, entity_type=entity_type)

        if not name.strip():
            float_event("entity.create.error", error="Empty name")
            raise ValidationError("Entity name cannot be empty")

        valid_types = {t.value for t in EntityType}
        if entity_type not in valid_types:
            float_event("entity.create.error", error=f"Invalid type: {entity_type}")
            raise ValidationError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            )

        entity = Entity(name=name, entity_type=entity_type, properties=properties or {})

        try:
            await self.client.execute_query(
                "createEntity",
                {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "properties": json.dumps(entity.properties),
                    "aliases": "[]",
                },
            )

            self._add_to_cache(entity)

            logger.info("Created entity in DB and cache: %s (%s)", entity.name, entity.entity_type)

            float_event("entity.create.success", entity_id=entity.entity_id)

            return entity

        except Exception as e:
            logger.warning("Failed to persist entity to HelixDB: %s, adding to cache only", e)
            self._add_to_cache(entity)

            float_event("entity.create.fallback", error=str(e))

            return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """
        Get entity by ID with cache-aside pattern.

        Args:
            entity_id: Entity ID

        Returns:
            Entity or None if not found
        """
        float_event("entity.get.start", entity_id=entity_id)

        if entity_id in self._entity_cache:
            logger.debug("Cache HIT: %s", entity_id)
            float_event("entity.get.cache_hit", entity_id=entity_id)
            return self._entity_cache[entity_id]

        logger.debug("Cache MISS: %s, querying HelixDB", entity_id)
        float_event("entity.get.cache_miss", entity_id=entity_id)

        try:
            result = await self.client.execute_query("getEntity", {"entity_id": entity_id})

            if result and "entity" in result:
                entity_data = result["entity"]
                entity = Entity.model_validate(entity_data)

                self._add_to_cache(entity)
                logger.debug("Loaded from DB and cached: %s", entity_id)

                float_event("entity.get.success", entity_id=entity_id)
                return entity

            float_event("entity.get.not_found", entity_id=entity_id)
            return None

        except Exception as e:
            logger.warning("Failed to query HelixDB for entity %s: %s", entity_id, e)
            float_event("entity.get.error", entity_id=entity_id, error=str(e))
            return None

    async def get_or_create_entity(
        self, name: str, entity_type: str, properties: dict[str, Any] | None = None
    ) -> Entity:
        """
        Get entity by name or create if doesn't exist (deduplication).

        This is the key method for entity deduplication during extraction.

        Args:
            name: Entity name
            entity_type: Type (Person, Organization, Location, Object)
            properties: Optional properties

        Returns:
            Existing or newly created Entity
        """
        float_event("entity.get_or_create.start", name=name)

        normalized_name = name.lower()
        if normalized_name in self._name_to_id:
            entity_id = self._name_to_id[normalized_name]
            existing = self._entity_cache.get(entity_id)
            if existing:
                logger.debug("Entity found in cache: %s", name)
                float_event("entity.get_or_create.found", entity_id=entity_id)
                return existing

        try:
            result = await self.client.execute_query("getEntityByName", {"name": name})

            if result and "entity" in result:
                entity = Entity.model_validate(result["entity"])
                self._add_to_cache(entity)
                logger.debug("Entity found in DB: %s", name)
                float_event("entity.get_or_create.found_db", entity_id=entity.entity_id)
                return entity

        except Exception as e:
            logger.debug("Entity not found in DB: %s (%s)", name, e)

        logger.debug("Creating new entity: %s", name)
        float_event("entity.get_or_create.creating", name=name)
        return await self.create_entity(name, entity_type, properties)

    async def link_to_memory(
        self,
        entity_id: str,
        memory_id: str,
        edge_type: str = "EXTRACTED_ENTITY",
        confidence: int = 100,
        salience: int = 50,
        sentiment: str = "neutral",
    ) -> dict[str, Any]:
        """
        Create edge between Entity and Memory.

        Args:
            entity_id: Entity ID
            memory_id: Memory ID
            edge_type: "EXTRACTED_ENTITY" or "MENTIONS"
            confidence: Extraction confidence (0-100) for EXTRACTED_ENTITY
            salience: Entity importance in memory (0-100) for MENTIONS
            sentiment: Sentiment (positive/negative/neutral) for MENTIONS

        Returns:
            Created edge data

        Raises:
            ValidationError: If validation fails
            HelixMemoryOperationError: If creation fails
        """
        float_event(
            "entity.link.start", entity_id=entity_id, memory_id=memory_id, edge_type=edge_type
        )

        if edge_type not in {"EXTRACTED_ENTITY", "MENTIONS"}:
            raise ValidationError(
                f"Invalid edge_type: {edge_type}. Must be EXTRACTED_ENTITY or MENTIONS"
            )

        try:
            if edge_type == "EXTRACTED_ENTITY":
                result = await self.client.execute_query(
                    "linkExtractedEntity",
                    {
                        "memory_id": memory_id,
                        "entity_id": entity_id,
                        "confidence": confidence,
                        "method": "llm",
                    },
                )
            else:
                result = await self.client.execute_query(
                    "linkMentionsEntity",
                    {
                        "memory_id": memory_id,
                        "entity_id": entity_id,
                        "salience": salience,
                        "sentiment": sentiment,
                    },
                )

            logger.info(
                "Linked entity %s to memory %s (%s)", entity_id[:8], memory_id[:8], edge_type
            )

            float_event("entity.link.success", edge_type=edge_type)

            return result

        except Exception as e:
            logger.exception("Failed to link entity %s to memory %s: %s", entity_id, memory_id, e)

            float_event("entity.link.error", error=str(e))

            raise HelixMemoryOperationError(f"Failed to link entity to memory: {e}") from e

    async def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        """
        Get all entities linked to a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of Entity objects
        """
        float_event("entity.get_for_memory.start", memory_id=memory_id)

        try:
            result = await self.client.execute_query(
                "getEntitiesForMemory", {"memory_id": memory_id}
            )

            entities = []
            if result and "entities" in result:
                for entity_data in result["entities"]:
                    entity = Entity.model_validate(entity_data)
                    entities.append(entity)
                    self._add_to_cache(entity)

            logger.debug("Found %d entities for memory %s", len(entities), memory_id[:8])

            float_event("entity.get_for_memory.success", count=len(entities))

            return entities

        except Exception as e:
            logger.warning("Failed to get entities for memory %s: %s", memory_id, e)
            float_event("entity.get_for_memory.error", error=str(e))
            return []

    def __repr__(self) -> str:
        """String representation."""
        return f"EntityManager(cached_entities={len(self._entity_cache)})"
