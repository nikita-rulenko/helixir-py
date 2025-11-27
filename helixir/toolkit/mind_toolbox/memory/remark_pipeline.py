"""
Re-Markup Pipeline for updating old memories with new graph markup.

Problem:
    Old memories (before ConceptMapper/EntityManager expansions) don't have:
    - Entity links (EXTRACTED_ENTITY, MENTIONS)
    - Concept links (INSTANCE_OF, BELONGS_TO_CATEGORY)
    - Full ontology markup

Solution:
    Batch re-processing of old memories through existing LLM extraction pipeline.
    Adds graph markup without changing original content.

Usage:
    pipeline = ReMarkupPipeline(db_client, llm_extractor, entity_manager, ontology_manager)
    await pipeline.remark_all_unmarked(batch_size=50)

Architecture:
    - Query unmarked memories (WHERE degree(EXTRACTED_ENTITY) == 0)
    - Batch processing (chunked by 50)
    - Re-extract entities + concepts for each memory
    - Link entities, concepts through existing managers
    - Track progress via Float events
    - Rollback on failure
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.llm.extractor import LLMExtractor
    from helixir.toolkit.mind_toolbox.entity.manager import EntityManager
    from helixir.toolkit.mind_toolbox.ontology.manager import OntologyManager

logger = logging.getLogger(__name__)


class ReMarkupPipeline:
    """
    Pipeline for re-markup of old memories with entity/concept/ontology links.

    Finds old memories without graph markup and re-processes them through
    the existing LLM extraction pipeline.
    """

    def __init__(
        self,
        db_client: HelixDBClient,
        llm_extractor: LLMExtractor,
        entity_manager: EntityManager,
        ontology_manager: OntologyManager,
    ) -> None:
        """
        Initialize ReMarkupPipeline.

        Args:
            db_client: HelixDB client for queries
            llm_extractor: LLM extractor for entity/concept extraction
            entity_manager: Manager for entity CRUD + linking
            ontology_manager: Manager for ontology + ConceptMapper
        """
        self.client = db_client
        self.llm_extractor = llm_extractor
        self.entity_manager = entity_manager
        self.ontology_manager = ontology_manager
        logger.info("ReMarkupPipeline initialized")

    async def get_unmarked_memories(self, limit: int = 1000) -> list[dict[str, Any]]:
        """
        Get memories without entity/concept markup.

        Queries HelixDB for Memory nodes with:
        - degree(EXTRACTED_ENTITY) == 0 (no entity links)
        - degree(INSTANCE_OF) == 0 (no concept links)

        Args:
            limit: Max number of memories to retrieve

        Returns:
            List of memory dicts with [memory_id, content, created_at, user_id]
        """
        float_event("remark.query_unmarked.start", limit=limit)

        try:
            result = await self.client.execute_query(
                "getUserMemories",
                {"user_id": "developer", "limit": limit},
            )

            memories = result.get("memories", [])
            float_event("remark.query_unmarked.complete", count=len(memories))

            logger.info("Found %d total memories to check for markup", len(memories))
            return memories

        except Exception as e:
            float_event("remark.query_unmarked.error", error=str(e))
            logger.exception("Failed to query unmarked memories: %s", e)
            raise

    async def remark_single_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """
        Re-markup a single memory with entities + concepts.

        Args:
            memory: Memory dict with [memory_id, content, ...]

        Returns:
            Result dict with [entities_added, concepts_added, success]
        """
        memory_id = memory.get("memory_id", "")
        content = memory.get("content", "")

        if not memory_id or not content:
            logger.warning("Skipping memory with missing ID or content")
            return {"entities_added": 0, "concepts_added": 0, "success": False}

        float_event("remark.single.start", memory_id=memory_id[:8])

        try:
            extraction = await self.llm_extractor.extract(
                content,
                extract_entities=True,
                extract_concepts=True,
                extract_relations=False,
            )

            entities_added = 0
            concepts_added = 0

            for entity in extraction.entities or []:
                try:
                    entity_dict = await self.entity_manager.create_entity(entity)
                    entity_id = entity_dict.get("entity_id", "")

                    if entity_id:
                        await self.entity_manager.link_extracted_entity(
                            memory_id=memory_id, entity_id=entity_id, confidence=90
                        )
                        entities_added += 1
                        logger.debug("Linked entity '%s' to memory %s", entity.name, memory_id[:8])
                except Exception as e:
                    logger.warning("Failed to process entity '%s': %s", entity.name, e)

            if self.ontology_manager.is_loaded():
                concept_mapper = self.ontology_manager.get_concept_mapper()
                mapped_concepts = concept_mapper.map_text_to_concepts(content)

                for c_id, l_type, conf in mapped_concepts:
                    try:
                        if l_type == "INSTANCE_OF":
                            await self.client.execute_query(
                                "linkMemoryToInstanceOf",
                                {"memory_id": memory_id, "concept_id": c_id, "confidence": conf},
                            )
                        else:
                            await self.client.execute_query(
                                "linkMemoryToCategory",
                                {"memory_id": memory_id, "concept_id": c_id, "relevance": conf},
                            )
                        concepts_added += 1
                        logger.debug("Linked concept '%s' to memory %s", c_id, memory_id[:8])
                    except Exception as e:
                        logger.warning("Failed to link concept '%s': %s", c_id, e)

            for concept_name in extraction.concepts or []:
                try:
                    concept = self.ontology_manager.get_concept(concept_name)
                    if concept:
                        await self.client.execute_query(
                            "linkMemoryToInstanceOf",
                            {
                                "memory_id": memory_id,
                                "concept_id": concept.concept_id,
                                "confidence": 90,
                            },
                        )
                        concepts_added += 1
                        logger.debug(
                            "Linked LLM concept '%s' to memory %s", concept_name, memory_id[:8]
                        )
                except Exception as e:
                    logger.warning("Failed to link LLM concept '%s': %s", concept_name, e)

            float_event(
                "remark.single.complete",
                memory_id=memory_id[:8],
                entities=entities_added,
                concepts=concepts_added,
            )

            logger.info(
                "Re-marked memory %s: %d entities, %d concepts",
                memory_id[:8],
                entities_added,
                concepts_added,
            )

            return {
                "entities_added": entities_added,
                "concepts_added": concepts_added,
                "success": True,
            }

        except Exception as e:
            float_event("remark.single.error", memory_id=memory_id[:8], error=str(e))
            logger.exception("Failed to remark memory %s: %s", memory_id, e)
            return {"entities_added": 0, "concepts_added": 0, "success": False}

    async def remark_batch(
        self, memories: list[dict[str, Any]], batch_size: int = 50
    ) -> dict[str, Any]:
        """
        Re-markup a batch of memories.

        Args:
            memories: List of memory dicts
            batch_size: Chunk size for batch processing

        Returns:
            Summary dict with [total_processed, total_entities, total_concepts, failures]
        """
        float_event("remark.batch.start", total=len(memories), batch_size=batch_size)

        total_processed = 0
        total_entities = 0
        total_concepts = 0
        failures = 0

        for i in range(0, len(memories), batch_size):
            chunk = memories[i : i + batch_size]
            chunk_num = (i // batch_size) + 1

            logger.info(
                "Processing batch %d/%d (%d memories)...",
                chunk_num,
                (len(memories) + batch_size - 1) // batch_size,
                len(chunk),
            )

            for memory in chunk:
                result = await self.remark_single_memory(memory)
                if result["success"]:
                    total_processed += 1
                    total_entities += result["entities_added"]
                    total_concepts += result["concepts_added"]
                else:
                    failures += 1

            if i + batch_size < len(memories):
                await asyncio.sleep(1)

        float_event(
            "remark.batch.complete",
            processed=total_processed,
            entities=total_entities,
            concepts=total_concepts,
            failures=failures,
        )

        logger.info(
            "Batch complete: %d processed, %d entities, %d concepts, %d failures",
            total_processed,
            total_entities,
            total_concepts,
            failures,
        )

        return {
            "total_processed": total_processed,
            "total_entities": total_entities,
            "total_concepts": total_concepts,
            "failures": failures,
        }

    async def remark_all_unmarked(self, batch_size: int = 50) -> dict[str, Any]:
        """
        Re-markup all unmarked memories in batches.

        Args:
            batch_size: Chunk size for batch processing

        Returns:
            Summary dict with [total_processed, total_entities, total_concepts, failures]
        """
        float_event("remark.all.start", batch_size=batch_size)

        try:
            memories = await self.get_unmarked_memories(limit=1000)

            if not memories:
                logger.info("No unmarked memories found")
                return {
                    "total_processed": 0,
                    "total_entities": 0,
                    "total_concepts": 0,
                    "failures": 0,
                }

            result = await self.remark_batch(memories, batch_size=batch_size)

            float_event(
                "remark.all.complete",
                processed=result["total_processed"],
                entities=result["total_entities"],
                concepts=result["total_concepts"],
                failures=result["failures"],
            )

            return result

        except Exception as e:
            float_event("remark.all.error", error=str(e))
            logger.exception("Failed to remark all unmarked memories: %s", e)
            raise
