"""
ToolingManager - LLM-агент с function calling для управления памятью.

Архитектура:
- LLM-агент управляет toolboxes (QueryBuilder, ChunkManager, SchemaManager)
- Function calling для выбора правильных операций
- Координирует экстракцию, хранение, поиск памяти

Как работает:
    User: "Remember I love Python"
        ↓
    ToolingManager.add_memory()
        ↓
    1. LLMExtractor.extract() → atomic facts
    2. EmbeddingGenerator.generate() → vectors
    3. QueryBuilder.search_similar() → check duplicates
    4. LLM decides: ADD/UPDATE/NOOP
    5. QueryBuilder.create_memory() → store
        ↓
    HelixDB
"""


from datetime import datetime, timedelta
import logging
from typing import Any

from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.llm import EmbeddingGenerator, LLMExtractor
from helixir.llm.decision_engine import LLMDecisionEngine, MemoryOperation
from helixir.toolkit.mind_toolbox.entity import EntityManager
from helixir.toolkit.mind_toolbox.memory.integrator import MemoryIntegrator
from helixir.toolkit.mind_toolbox.ontology import OntologyManager
from helixir.toolkit.mind_toolbox.reasoning import ReasoningEngine
from helixir.toolkit.misc_toolbox import float_event

logger = logging.getLogger(__name__)


class ToolingManager:
    """
    LLM-агент для управления памятью с полной разметкой графа.

    Координирует:
    - Экстракцию (LLMExtractor)
    - Embeddings (EmbeddingGenerator)
    - Decision Engine (LLM function calling для ADD/UPDATE/DELETE/NOOP)
    - Разметку графа (Entities, Reasoning, Ontology)
    - Хранение в HelixDB

    Attributes:
        db_client: HelixDB connection
        llm_extractor: LLM для экстракции памяти
        embedder: Генератор векторов
        decision_engine: LLM для принятия решений
        memory_manager: Управление памятью
        entity_manager: Управление entity нодами
        ontology_manager: Управление онтологией
        reasoning_engine: Управление reasoning relations
        config: Конфигурация
    """

    def __init__(
        self,
        db_client: HelixDBClient,
        llm_extractor: LLMExtractor,
        embedder: EmbeddingGenerator,
        config: HelixMemoryConfig,
    ):
        """
        Initialize ToolingManager.

        Args:
            db_client: HelixDB client
            llm_extractor: LLM extractor for memory extraction
            embedder: Embedding generator
            config: Configuration
        """
        self.db = db_client
        self.llm_extractor = llm_extractor
        self.embedder = embedder
        self.config = config

        self.decision_engine = LLMDecisionEngine(llm_provider=llm_extractor.provider)

        from helixir.toolkit.mind_toolbox.memory.search import SearchEngine

        self.search_engine = SearchEngine(
            client=db_client,
            cache_size=1000,
            cache_ttl=300,
            enable_smart_traversal_v2=True,
        )

        from helixir.toolkit.mind_toolbox.memory.manager import MemoryManager

        self.memory_manager = MemoryManager(db_client, embedder=embedder)
        self.entity_manager = EntityManager(db_client)
        self.ontology_manager = OntologyManager(db_client)
        self.reasoning_engine = ReasoningEngine(
            db_client,
            llm_provider=llm_extractor.provider,
        )

        from helixir.toolkit.mind_toolbox.chunking import ChunkingManager

        self.chunking_manager = ChunkingManager(
            client=db_client,
            embedder=embedder,
            threshold=500,
            chunk_size=512,
            overlap=0.1,
            enable_embeddings=True,
        )

        self.memory_integrator = MemoryIntegrator(
            client=db_client,
            embedding_gen=embedder,
            reasoning_engine=self.reasoning_engine,
            similarity_threshold=0.7,
            max_similar=10,
            enable_reasoning=True,
        )

        logger.info(
            "ToolingManager initialized with LLM decision engine + graph markup tools"
        )


    async def add_memory(
        self,
        message: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add memory with LLM extraction and deduplication.

        Pipeline:
        1. Extract atomic facts (LLM)
        2. For each memory:
           a. Generate embedding
           b. Vector search for similar memories
           c. LLM decides: ADD/UPDATE/DELETE/NOOP
           d. Execute decision
        3. Return stats

        Args:
            message: Text to extract memories from
            user_id: User identifier
            agent_id: Optional agent identifier
            metadata: Optional metadata

        Returns:
            Dict with operation stats
        """
        float_event("tooling.add_memory.start", user_id=user_id, message_len=len(message))

        logger.info("Adding memory for user=%s: %s", user_id, message[:50])

        extraction = await self.llm_extractor.extract(
            text=message, user_id=user_id, extract_entities=True, extract_relations=True
        )

        float_event(
            "tooling.extraction_complete",
            memories=len(extraction.memories),
            entities=len(extraction.entities),
            relations=len(extraction.relations),
        )

        logger.info(
            "Extracted: %d memories, %d entities, %d relations",
            len(extraction.memories),
            len(extraction.entities),
            len(extraction.relations),
        )

        added_ids = []
        updated_ids = []
        deleted_ids = []
        skipped_count = 0

        memory_content_map: dict[str, Any] = {}
        for mem in extraction.memories:
            memory_content_map[mem.content] = mem

        for memory in extraction.memories:
            vector = await self.embedder.generate(memory.content)

            similar = []

            float_event("tooling.similar_found", count=len(similar))

            decision = await self.decision_engine.decide(
                new_memory=memory.content,
                similar_memories=similar,
                user_id=user_id,
                similarity_threshold=0.85,
            )

            logger.info(
                "Decision for '%s...': %s (confidence=%d)",
                memory.content[:30],
                decision.operation,
                decision.confidence,
            )

            memory_id = None

            if decision.operation == MemoryOperation.ADD:
                if self.chunking_manager.should_chunk(memory.content):
                    logger.info(
                        "📦 Content exceeds threshold (%d chars), using ChunkingManager",
                        len(memory.content),
                    )
                    created_memory_data = await self.chunking_manager.add_memory(
                        content=memory.content,
                        memory_id=None,
                        memory_type=memory.memory_type,
                        importance=memory.importance,
                        certainty=memory.certainty,
                        source="llm_extraction",
                        context_tags="",
                        metadata=memory.context or {},
                    )
                    memory_id = created_memory_data.get("memory_id")

                    from helixir.toolkit.mind_toolbox.memory.models import Memory

                    created_memory = Memory(
                        memory_id=memory_id,
                        content=memory.content[:200] + "..."
                        if len(memory.content) > 200
                        else memory.content,
                        memory_type=memory.memory_type,
                        user_id=user_id,
                        certainty=memory.certainty,
                        importance=memory.importance,
                        context=memory.context or {},
                        concepts=memory.concepts or [],
                    )
                else:
                    created_memory = await self.memory_manager.add_memory(
                        content=memory.content,
                        user_id=user_id,
                        memory_type=memory.memory_type,
                        certainty=memory.certainty,
                        importance=memory.importance,
                    )
                    memory_id = created_memory.memory_id

                added_ids.append(memory_id)

                float_event("tooling.memory.added", memory_id=memory_id)

                try:
                    integration_result = await self.memory_integrator.integrate_memory(
                        memory=created_memory,
                        query_embedding=vector,
                    )
                    logger.info(
                        "✅ Integrated %s: %d similar found, %d relations created, %d superseded",
                        memory_id[:8],
                        integration_result.similar_found,
                        integration_result.relations_created,
                        len(integration_result.superseded_memories),
                    )
                    float_event(
                        "tooling.memory.integrated",
                        memory_id=memory_id,
                        similar_found=integration_result.similar_found,
                        relations_created=integration_result.relations_created,
                        superseded=len(integration_result.superseded_memories),
                    )
                except Exception as e:
                    logger.warning("❌ Integration failed for %s: %s", memory_id[:8], e)


            elif decision.operation == MemoryOperation.UPDATE:
                if decision.target_memory_id and decision.merged_content:
                    merged_vector = await self.embedder.generate(decision.merged_content)

                    updated_memory = await self.memory_manager.update_memory(
                        memory_id=decision.target_memory_id,
                        content=decision.merged_content,
                        vector=merged_vector,
                        certainty=memory.certainty,
                        importance=memory.importance,
                        user_id=user_id,
                    )

                    if updated_memory:
                        memory_id = updated_memory.memory_id
                        updated_ids.append(memory_id)

                        float_event("tooling.memory.updated", memory_id=memory_id)
                else:
                    logger.warning("UPDATE decision without target_memory_id or merged_content")

            elif decision.operation == MemoryOperation.DELETE:
                if decision.target_memory_id:
                    await self.memory_manager.delete_memory(decision.target_memory_id)
                    deleted_ids.append(decision.target_memory_id)

                    float_event("tooling.memory.deleted", memory_id=decision.target_memory_id)

                    created_memory = await self.memory_manager.add_memory(
                        content=memory.content,
                        user_id=user_id,
                        memory_type=memory.memory_type,
                        certainty=memory.certainty,
                        importance=memory.importance,
                    )
                    memory_id = created_memory.memory_id
                    added_ids.append(memory_id)

                    float_event("tooling.memory.added", memory_id=memory_id)

                    try:
                        integration_result = await self.memory_integrator.integrate_memory(
                            memory=created_memory,
                            query_embedding=vector,
                        )
                        logger.info(
                            "✅ Integrated %s (after DELETE): %d similar, %d relations",
                            memory_id[:8],
                            integration_result.similar_found,
                            integration_result.relations_created,
                        )
                    except Exception as e:
                        logger.warning("❌ Integration failed for %s: %s", memory_id[:8], e)

            elif decision.operation == MemoryOperation.NOOP:
                skipped_count += 1
                logger.debug("Skipping duplicate memory: %s", memory.content[:50])
                float_event("tooling.memory.skipped", content=memory.content[:50])
                continue


            if memory_id:
                memory_content_map[memory.content] = (memory, memory_id)

                float_event("tooling.entity_markup.start", memory_id=memory_id)

                entity_count = 0
                for extracted_entity in extraction.entities:
                    try:
                        entity = await self.entity_manager.get_or_create_entity(
                            name=extracted_entity.name,
                            entity_type=extracted_entity.entity_type,
                            properties=extracted_entity.properties or {},
                        )

                        await self.entity_manager.link_to_memory(
                            entity_id=entity.entity_id,
                            memory_id=memory_id,
                            edge_type="EXTRACTED_ENTITY",
                            confidence=90,
                        )

                        entity_count += 1
                        logger.debug(
                            "Linked entity '%s' (%s) to memory %s",
                            entity.name,
                            entity.entity_type,
                            memory_id[:8],
                        )

                    except Exception as e:
                        logger.warning("Failed to create/link entity: %s", e)

                float_event("tooling.entity_markup.complete", count=entity_count)

                float_event("tooling.ontology_link.start", memory_id=memory_id)

                concept_count = 0

                if not self.ontology_manager.is_loaded():
                    await self.ontology_manager.load_base_ontology()

                try:
                    mapper = self.ontology_manager.get_mapper()
                    concept_mappings = mapper.map_to_concepts(
                        text=memory.content, memory_type=memory.memory_type, min_confidence=30
                    )

                    for concept_id, confidence, link_type in concept_mappings:
                        try:
                            await self.memory_manager.link_memory_to_concept(
                                memory_id=memory_id,
                                concept_id=concept_id,
                                confidence=confidence,
                                link_type=link_type,
                            )
                            concept_count += 1
                            logger.debug(
                                "Auto-linked memory %s to concept '%s' (%s, conf=%d)",
                                memory_id[:8],
                                concept_id,
                                link_type,
                                confidence,
                            )
                        except Exception as e:
                            logger.warning("Failed to auto-link concept %s: %s", concept_id, e)

                except Exception as e:
                    logger.warning("Failed to use ConceptMapper: %s", e)

                for concept_name in memory.concepts or []:
                    try:
                        concept = self.ontology_manager.get_concept(concept_name)
                        if concept:
                            await self.memory_manager.link_memory_to_concept(
                                memory_id=memory_id,
                                concept_id=concept.concept_id,
                                confidence=85,
                                link_type="INSTANCE_OF",
                            )
                            concept_count += 1
                            logger.debug(
                                "LLM-linked memory %s to concept '%s'", memory_id[:8], concept_name
                            )
                        else:
                            logger.warning("Concept not found in ontology: %s", concept_name)

                    except Exception as e:
                        logger.warning("Failed to link LLM concept %s: %s", concept_name, e)

                float_event("tooling.ontology_link.complete", count=concept_count)


        float_event("tooling.reasoning.start", relations=len(extraction.relations))

        relation_count = 0
        for relation in extraction.relations:
            try:
                from_data = memory_content_map.get(relation.from_memory_content)
                to_data = memory_content_map.get(relation.to_memory_content)

                if not from_data or not to_data:
                    logger.debug(
                        "Skipping relation (memories not found in current batch): %s → %s",
                        relation.from_memory_content[:40],
                        relation.to_memory_content[:40],
                    )
                    continue

                from_memory_id = from_data[1] if isinstance(from_data, tuple) else None
                to_memory_id = to_data[1] if isinstance(to_data, tuple) else None

                if from_memory_id and to_memory_id:
                    await self.reasoning_engine.add_relation(
                        from_memory_id=from_memory_id,
                        to_memory_id=to_memory_id,
                        relation_type=relation.relation_type,
                        strength=relation.strength,
                        confidence=relation.confidence,
                        explanation=relation.explanation,
                    )
                    relation_count += 1
                    logger.debug(
                        "Created %s relation: %s → %s",
                        relation.relation_type,
                        from_memory_id[:8],
                        to_memory_id[:8],
                    )

            except Exception as e:
                logger.warning("Failed to create reasoning relation: %s", e)

        float_event("tooling.reasoning.complete", count=relation_count)

        float_event(
            "tooling.add_memory.complete",
            added=len(added_ids),
            updated=len(updated_ids),
            deleted=len(deleted_ids),
            skipped=skipped_count,
            entities_created=len(extraction.entities),
            reasoning_relations=relation_count,
        )

        return {
            "added": added_ids,
            "updated": updated_ids,
            "deleted": deleted_ids,
            "skipped": skipped_count,
            "entities_extracted": len(extraction.entities),
            "reasoning_relations_created": relation_count,
            "metadata": extraction.metadata,
        }

    async def search_memory(
        self,
        query: str,
        user_id: str,
        limit: int | None = None,
        search_mode: str = "recent",
        temporal_days: float | None = None,
        graph_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Unified smart memory search with automatic mode selection.

        Uses SmartTraversalV2 (vector-first + graph expansion) for RECENT/CONTEXTUAL/DEEP modes,
        falls back to hybrid search for FULL mode.

        Args:
            query: Search query
            user_id: User identifier
            limit: Max results (None = use mode default)
            search_mode: "recent", "contextual", "deep", or "full"
            temporal_days: Time window in days (None = use mode default)
            graph_depth: Graph traversal depth (None = use mode default)

        Returns:
            List of matching memories with graph context
        """
        from datetime import UTC

        from helixir.core.search_modes import SearchMode

        float_event("tooling.search_memory.start", user_id=user_id, query_len=len(query))

        logger.info(
            "Searching memories: query=%s, user=%s, mode=%s",
            query[:50],
            user_id,
            search_mode,
        )

        mode = SearchMode.from_string(search_mode)
        mode_defaults = mode.get_defaults()

        if limit is None:
            limit = mode_defaults["max_results"]
        if temporal_days is None:
            temporal_days = mode_defaults["temporal_days"]
        if graph_depth is None:
            graph_depth = mode_defaults["graph_depth"]

        query_vector = await self.embedder.generate(query)

        temporal_cutoff = None
        if temporal_days is not None:
            temporal_cutoff = datetime.now(UTC) - timedelta(days=temporal_days)

        use_smart = mode_defaults.get("use_smart_traversal", False)

        try:
            if use_smart:
                vector_top_k = mode_defaults.get("vector_top_k", 10)
                min_vector_score = mode_defaults.get("min_vector_score", 0.5)
                min_combined_score = mode_defaults.get("min_combined_score", 0.3)

                logger.info(
                    "Using SmartTraversalV2: vector_top_k=%d, graph_depth=%d, temporal=%s",
                    vector_top_k,
                    graph_depth,
                    temporal_days or "all",
                )

                results = await self.search_engine.smart_graph_search_v2(
                    query=query,
                    query_embedding=query_vector,
                    user_id=user_id,
                    vector_top_k=vector_top_k,
                    graph_depth=graph_depth,
                    min_vector_score=min_vector_score,
                    min_combined_score=min_combined_score,
                    temporal_cutoff=temporal_cutoff,
                )

                results = results[:limit]

                results = [
                    {
                        "memory_id": r.memory.memory_id,
                        "content": r.memory.content,
                        "score": r.score,
                        "method": r.method,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]

            else:
                logger.info(
                    "Using hybrid search: limit=%d, temporal=%s",
                    limit,
                    temporal_days or "all",
                )

                search_results = await self.search_engine.hybrid_search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                )

                results = [
                    {
                        "memory_id": r.memory_id,
                        "content": r.content,
                        "score": r.score,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in search_results
                ]

        except Exception as e:
            logger.exception("Search failed: %s", e)
            results = []

        float_event(
            "tooling.search_memory.complete",
            results_count=len(results),
            search_mode=search_mode,
            temporal_days=temporal_days,
            use_smart=use_smart,
        )

        logger.info(
            "Search complete: found %d memories (mode=%s, strategy=%s)",
            len(results),
            search_mode,
            "smart_v2" if use_smart else "hybrid",
        )

        return results

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        """Get memory by ID."""
        return await self.memory_manager.get_memory(memory_id)

    async def update_memory(self, memory_id: str, new_content: str, user_id: str) -> dict[str, Any]:
        """
        Update memory content using supersession pattern.

        Creates a NEW memory with updated content and links it to the old one
        via SUPERSEDES edge. Preserves history and copies reasoning relations.
        """
        new_vector = await self.embedder.generate(new_content)

        return await self.memory_manager.update_memory(
            memory_id=memory_id,
            content=new_content,
            vector=new_vector,
            user_id=user_id,
        )

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID."""
        return await self.memory_manager.delete_memory(memory_id)

    async def get_recent_memories(
        self, user_id: str, limit: int = 10, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get recent memories."""
        results = await self.search_engine.search(
            query="",
            user_id=user_id,
            limit=limit,
            mode="recent",
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def get_memory_graph(
        self, user_id: str, memory_id: str | None = None, depth: int = 2
    ) -> dict[str, Any]:
        """Get memory graph."""
        if not memory_id:
            return {"nodes": [], "edges": [], "error": "memory_id required"}

        memory = await self.memory_manager.get_memory(memory_id)
        if not memory:
            return {"nodes": [], "edges": [], "error": "memory not found"}

        return {
            "nodes": [memory],
            "edges": [],
            "depth": depth,
        }


    async def create_agent(
        self,
        agent_id: str,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create new agent (stub - not implemented)."""
        logger.warning("create_agent not implemented")
        return {"agent_id": agent_id, "name": name, "status": "not_implemented"}

    async def get_agent_memories(self, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get agent memories (stub - not implemented)."""
        logger.warning("get_agent_memories not implemented")
        return []
