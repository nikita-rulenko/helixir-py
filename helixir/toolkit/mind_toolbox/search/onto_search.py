"""
Ontology-Aware Search Strategy.

OntoSearchStrategy combines:
- Vector search (semantic similarity)
- Concept matching (ontology overlap)
- Graph expansion (logical connections)
- Temporal scoring (freshness)

Pipeline:
    Query → Vector Search → Concept Classification → Concept Matching → Graph Expansion → Ranking

Key Innovation:
    Uses ontology concepts to boost relevance. If query is about "preferences"
    and memory is linked to "Preference" concept, it gets a concept_score boost.

Usage:
    >>> strategy = OntoSearchStrategy(client, ontology_manager)
    >>> results = await strategy.search(
    ...     query="What Python frameworks do I like?",
    ...     query_embedding=embedding,
    ...     user_id="developer",
    ...     limit=10,
    ... )
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from helixir.toolkit.mind_toolbox.search.models import ConceptMatch, OntoSearchConfig, SearchResult
from helixir.toolkit.mind_toolbox.search.query_processor import QueryProcessor

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.toolkit.mind_toolbox.ontology import OntologyManager

logger = logging.getLogger(__name__)


def parse_datetime_utc(dt_string: str) -> datetime | None:
    """
    Parse datetime string to UTC-aware datetime.

    Handles various formats:
    - ISO 8601 with Z suffix
    - ISO 8601 with +00:00 offset
    - ISO 8601 without timezone (assumes UTC)

    Args:
        dt_string: Datetime string to parse

    Returns:
        UTC-aware datetime or None if parsing fails
    """
    if not dt_string:
        return None

    try:
        dt_string = dt_string.replace("Z", "+00:00")

        try:
            dt = datetime.fromisoformat(dt_string)
        except ValueError:
            dt = datetime.fromisoformat(dt_string.split("+")[0])
            dt = dt.replace(tzinfo=UTC)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt
    except (ValueError, AttributeError):
        return None


def is_within_temporal_window(created_at: str, hours: float | None) -> bool:
    """
    Check if datetime is within temporal window.

    Args:
        created_at: ISO datetime string
        hours: Window in hours (None = no filter)

    Returns:
        True if within window or no filter, False otherwise
    """
    if hours is None:
        return True

    created = parse_datetime_utc(created_at)
    if created is None:
        return True

    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=hours)

    return created >= cutoff


def calculate_temporal_freshness(created_at: str, decay_days: float = 30.0) -> float:
    """
    Calculate temporal freshness score with exponential decay.

    Args:
        created_at: ISO datetime string
        decay_days: Half-life in days

    Returns:
        Freshness score (0.0-1.0), 1.0 = just created
    """
    created = parse_datetime_utc(created_at)
    if created is None:
        return 0.5

    now = datetime.now(UTC)
    days_old = (now - created).total_seconds() / 86400
    freshness = math.exp(-days_old / decay_days)
    return max(0.0, min(1.0, freshness))


class OntoSearchStrategy:
    """
    Ontology-aware search strategy.

    Implements SearchStrategy protocol with concept-based boosting.

    Pipeline:
        1. Vector Search → Top-K semantically similar memories
        2. Query Concept Classification → Extract concepts from query
        3. Memory Concept Loading → Get concepts for each memory
        4. Concept Scoring → Calculate overlap between query and memory concepts
        5. Graph Expansion → Load neighbors via logical edges (depth=1)
        6. Combined Ranking → vector + concept + graph + temporal

    Attributes:
        client: HelixDBClient for database queries
        ontology: OntologyManager for concept operations
        config: OntoSearchConfig with weights and parameters

    Example:
        >>> strategy = OntoSearchStrategy(client, ontology_manager)
        >>> results = await strategy.search(
        ...     query="Python preferences",
        ...     query_embedding=[0.1, 0.2, ...],
        ...     user_id="developer",
        ... )
    """

    def __init__(
        self,
        client: HelixDBClient,
        ontology_manager: OntologyManager,
        config: OntoSearchConfig | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Initialize OntoSearchStrategy.

        Args:
            client: HelixDBClient instance
            ontology_manager: OntologyManager instance (must be loaded)
            config: Optional configuration (uses defaults if None)
            mode: Optional search mode ("recent", "contextual", "deep", "full")
                  If provided, config is auto-generated from mode

        Priority: config > mode > defaults
        """
        self.client = client
        self.ontology = ontology_manager

        if config is not None:
            self.config = config
        elif mode is not None:
            self.config = OntoSearchConfig.from_mode(mode)
        else:
            self.config = OntoSearchConfig()

        self._current_mode = mode

        self.query_processor = QueryProcessor(enable_expansion=True, max_expansions=3)

        logger.info(
            "OntoSearchStrategy initialized: mode=%s, weights=[v=%.2f, g=%.2f, c=%.2f, tag=%.2f, t=%.2f]",
            mode or "default",
            self.config.vector_weight,
            self.config.graph_weight,
            self.config.concept_weight,
            self.config.tag_weight,
            self.config.temporal_weight,
        )

    def set_mode(self, mode: str) -> None:
        """
        Dynamically switch search mode.

        Updates config to match the new mode.

        Args:
            mode: Search mode ("recent", "contextual", "deep", "full")
        """
        self.config = OntoSearchConfig.from_mode(mode)
        self._current_mode = mode
        logger.info(
            "OntoSearch mode changed to '%s': weights=[v=%.2f, c=%.2f, tag=%.2f]",
            mode,
            self.config.vector_weight,
            self.config.concept_weight,
            self.config.tag_weight,
        )

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        limit: int = 10,
        mode: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Execute ontology-aware search.

        Args:
            query: User query text
            query_embedding: Query embedding vector
            user_id: Optional user filter
            limit: Maximum number of results
            mode: Optional search mode override ("recent", "contextual", "deep", "full")
            **kwargs: Additional parameters (graph_depth, concept_boost, etc.)

        Returns:
            List of SearchResult objects, ranked by combined score
        """
        time.perf_counter()

        original_config = self.config
        if mode is not None:
            self.config = OntoSearchConfig.from_mode(mode)

        try:
            return await self._execute_search(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                **kwargs,
            )
        finally:
            self.config = original_config

    async def _execute_search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None,
        limit: int,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Internal search execution."""
        start_time = time.perf_counter()

        graph_depth = kwargs.get("graph_depth", self.config.graph_depth)
        vector_top_k = kwargs.get("vector_top_k", self.config.vector_top_k)

        phase0_start = time.perf_counter()
        processed = self.query_processor.process(query)
        phase0_time = time.perf_counter() - phase0_start

        logger.info(
            "Phase 0 (QueryProc): intents=%s, concepts=%s, expanded=%s in %.0fms",
            processed.detected_intents,
            processed.concept_hints,
            processed.expanded_terms[:3],
            phase0_time * 1000,
        )

        logger.info(
            "OntoSearch: query='%s' → '%s', user=%s, limit=%d, depth=%d",
            query[:30],
            processed.enhanced_query[:50],
            user_id,
            limit,
            graph_depth,
        )

        phase1_start = time.perf_counter()
        vector_hits = await self._vector_search_phase(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=vector_top_k,
        )
        phase1_time = time.perf_counter() - phase1_start
        logger.info("Phase 1 (Vector): %d hits in %.0fms", len(vector_hits), phase1_time * 1000)

        if not vector_hits:
            logger.warning("No vector hits, returning empty results")
            return []

        phase2_start = time.perf_counter()
        query_concepts = self._classify_query_concepts(query)

        for hint in processed.concept_hints:
            if not any(c.concept_id == hint for c in query_concepts):
                query_concepts.append(
                    ConceptMatch(concept_id=hint, confidence=0.7, match_type="processor_hint")
                )

        query_tags = self._extract_query_tags(query)
        query_tags.extend([t for t in processed.expanded_terms if t not in query_tags])

        phase2_time = time.perf_counter() - phase2_start
        logger.info(
            "Phase 2 (Classify): %d concepts, %d tags in %.0fms: concepts=%s, tags=%s",
            len(query_concepts),
            len(query_tags),
            phase2_time * 1000,
            [c.concept_id for c in query_concepts],
            query_tags[:5],
        )

        phase3_start = time.perf_counter()
        scored_results = await self._score_by_concepts_and_tags_phase(
            vector_hits=vector_hits,
            query_concepts=query_concepts,
            query_tags=query_tags,
        )
        phase3_time = time.perf_counter() - phase3_start
        logger.info(
            "Phase 3 (Concepts+Tags): scored %d results in %.0fms",
            len(scored_results),
            phase3_time * 1000,
        )

        phase4_start = time.perf_counter()
        expanded_results = await self._graph_expansion_phase(
            results=scored_results,
            query_embedding=query_embedding,
            query_concepts=query_concepts,
            depth=graph_depth,
        )
        phase4_time = time.perf_counter() - phase4_start
        logger.info(
            "Phase 4 (Graph): expanded to %d results in %.0fms",
            len(expanded_results),
            phase4_time * 1000,
        )

        phase5_start = time.perf_counter()
        ranked_results = self._rank_results(expanded_results)
        phase5_time = time.perf_counter() - phase5_start

        total_time = time.perf_counter() - start_time
        logger.info(
            "OntoSearch complete: %d results in %.0fms (P1:%.0f P2:%.0f P3:%.0f P4:%.0f P5:%.0f)",
            len(ranked_results[:limit]),
            total_time * 1000,
            phase1_time * 1000,
            phase2_time * 1000,
            phase3_time * 1000,
            phase4_time * 1000,
            phase5_time * 1000,
        )

        return ranked_results[:limit]

    async def _vector_search_phase(
        self,
        query_embedding: list[float],
        user_id: str | None,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Phase 1: Vector search for semantically similar memories.

        Uses smartVectorSearchWithChunks to find Top-K similar memories.
        """
        try:
            results = await self.client.execute_query(
                "smartVectorSearchWithChunks",
                {"query_vector": query_embedding, "limit": top_k},
            )

            all_memories: list[dict[str, Any]] = []
            all_memories.extend(results.get("memories", []))
            all_memories.extend(results.get("parent_memories", []))

            seen_ids: set[str] = set()
            unique_memories: list[dict[str, Any]] = []
            for mem in all_memories:
                mem_id = mem.get("memory_id")
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    unique_memories.append(mem)

            temporal_filter = self.config.temporal_filter_hours
            if temporal_filter is not None:
                before_filter = len(unique_memories)
                unique_memories = [
                    mem
                    for mem in unique_memories
                    if is_within_temporal_window(mem.get("created_at", ""), temporal_filter)
                ]
                after_filter = len(unique_memories)
                if before_filter != after_filter:
                    logger.info(
                        "Temporal filter (%.1fh): %d → %d memories",
                        temporal_filter,
                        before_filter,
                        after_filter,
                    )

            search_results: list[SearchResult] = []
            for mem in unique_memories:
                memory_id = mem.get("memory_id", "")
                if not memory_id:
                    continue

                temporal_score = calculate_temporal_freshness(
                    mem.get("created_at", ""),
                    self.config.temporal_decay_days,
                )

                vector_score = 0.8

                search_results.append(
                    SearchResult(
                        memory_id=memory_id,
                        content=mem.get("content", ""),
                        score=0.0,
                        method="onto",
                        vector_score=vector_score,
                        temporal_score=temporal_score,
                        depth=0,
                        source="vector",
                        metadata={
                            "memory_type": mem.get("memory_type", ""),
                            "user_id": mem.get("user_id", ""),
                            "created_at": mem.get("created_at", ""),
                        },
                    )
                )

            return search_results

        except Exception as e:
            logger.exception("Vector search phase failed: %s", e)
            return []

    def _classify_query_concepts(self, query: str) -> list[ConceptMatch]:
        """
        Phase 2: Classify query into ontology concepts.

        Uses OntologyManager's mapper to find relevant concepts.
        """
        try:
            mapper = self.ontology.get_mapper()
            raw_matches = mapper.map_to_concepts(
                text=query,
                min_confidence=int(self.config.min_concept_confidence * 100),
            )

            concepts: list[ConceptMatch] = []
            for concept_id, confidence, link_type in raw_matches[
                : self.config.max_concepts_per_query
            ]:
                concepts.append(
                    ConceptMatch(
                        concept_id=concept_id,
                        confidence=confidence / 100.0,
                        link_type=link_type,
                    )
                )

            return concepts

        except Exception as e:
            logger.warning("Query concept classification failed: %s", e)
            return []

    def _extract_query_tags(self, query: str) -> list[str]:
        """
        Extract potential tags from query.

        Extracts:
        - Known technology terms (python, fastapi, rust, etc.)
        - Domain terms (work, personal, project, etc.)
        - Other significant words
        """
        import re

        known_tags = {
            "python",
            "fastapi",
            "rust",
            "javascript",
            "typescript",
            "react",
            "django",
            "flask",
            "nodejs",
            "docker",
            "kubernetes",
            "aws",
            "gcp",
            "postgresql",
            "mongodb",
            "redis",
            "helixdb",
            "ollama",
            "openai",
            "async",
            "api",
            "backend",
            "frontend",
            "database",
            "graph",
            "work",
            "personal",
            "project",
            "home",
            "travel",
            "health",
            "finance",
            "learning",
            "career",
            "family",
            "ai",
            "ml",
            "memory",
            "llm",
            "embedding",
            "vector",
            "search",
            "programming",
            "coding",
            "development",
            "architecture",
        }

        words = set(re.findall(r"\b\w+\b", query.lower()))

        return [word for word in words if word in known_tags]


    def _calculate_tag_overlap(
        self,
        query_tags: list[str],
        content: str,
    ) -> float:
        """
        Calculate tag overlap score.

        Checks if query tags appear in memory content.

        Args:
            query_tags: Tags extracted from query
            content: Memory content

        Returns:
            Overlap score (0.0-1.0)
        """
        if not query_tags:
            return 0.0

        content_lower = content.lower()
        matches = sum(1 for tag in query_tags if tag in content_lower)

        return matches / len(query_tags) if query_tags else 0.0

    async def _score_by_concepts_and_tags_phase(
        self,
        vector_hits: list[SearchResult],
        query_concepts: list[ConceptMatch],
        query_tags: list[str],
    ) -> list[SearchResult]:
        """
        Phase 3: Load memory concepts/tags and calculate scores.

        For each vector hit:
        1. Load its concepts via getMemoryConcepts
        2. Calculate concept overlap score
        3. Calculate tag overlap score (from content matching)
        4. Update concept_score and tag_score
        """
        tasks = [self._load_memory_concepts(result.memory_id) for result in vector_hits]
        memory_concepts_list = await asyncio.gather(*tasks, return_exceptions=True)

        for result, memory_concepts in zip(vector_hits, memory_concepts_list, strict=False):
            if isinstance(memory_concepts, Exception):
                logger.warning(
                    "Failed to load concepts for %s: %s", result.memory_id, memory_concepts
                )
                memory_concepts = []

            if query_concepts:
                concept_score = self._calculate_concept_overlap(
                    query_concepts=query_concepts,
                    memory_concepts=memory_concepts,
                )
                result.concept_score = concept_score

            if query_tags:
                tag_score = self._calculate_tag_overlap(
                    query_tags=query_tags,
                    content=result.content,
                )
                result.tag_score = tag_score

            result.metadata["concepts"] = memory_concepts

        return vector_hits

    async def _load_memory_concepts(self, memory_id: str) -> list[str]:
        """
        Load concepts for a single memory.

        Uses getMemoryConcepts query.
        """
        try:
            result = await self.client.execute_query(
                "getMemoryConcepts",
                {"memory_id": memory_id},
            )

            concepts: list[str] = []

            for concept in result.get("instance_of", []):
                concept_id = concept.get("concept_id")
                if concept_id:
                    concepts.append(concept_id)

            for concept in result.get("belongs_to", []):
                concept_id = concept.get("concept_id")
                if concept_id:
                    concepts.append(concept_id)

            return concepts

        except Exception as e:
            logger.debug("Failed to load concepts for %s: %s", memory_id, e)
            return []

    def _calculate_concept_overlap(
        self,
        query_concepts: list[ConceptMatch],
        memory_concepts: list[str],
    ) -> float:
        """
        Calculate concept overlap score.

        Score = sum(confidence) for matching concepts / max possible

        Args:
            query_concepts: Concepts from query classification
            memory_concepts: Concepts linked to memory

        Returns:
            Overlap score (0.0-1.0)
        """
        if not query_concepts or not memory_concepts:
            return 0.0

        total_score = 0.0
        max_score = sum(qc.confidence for qc in query_concepts)

        for qc in query_concepts:
            if qc.concept_id in memory_concepts:
                total_score += qc.confidence
                total_score += self.config.concept_boost

        return min(total_score / max_score, 1.0) if max_score > 0 else 0.0

    async def _graph_expansion_phase(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
        query_concepts: list[ConceptMatch],
        depth: int,
    ) -> list[SearchResult]:
        """
        Phase 4: Expand from vector hits via graph edges.

        For each result, load neighbors via getMemoryLogicalConnections.
        """
        if depth <= 0:
            return results

        seed_ids = {r.memory_id for r in results}
        all_results = list(results)

        expansion_tasks = [
            self._expand_from_memory(
                memory_id=result.memory_id,
                parent_score=result.vector_score,
                query_concepts=query_concepts,
                current_depth=0,
                max_depth=depth,
                visited=seed_ids.copy(),
            )
            for result in results
        ]

        expansion_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)

        for expansion in expansion_results:
            if isinstance(expansion, list):
                all_results.extend(expansion)
            elif isinstance(expansion, Exception):
                logger.warning("Graph expansion failed: %s", expansion)

        return all_results

    async def _expand_from_memory(
        self,
        memory_id: str,
        parent_score: float,
        query_concepts: list[ConceptMatch],
        current_depth: int,
        max_depth: int,
        visited: set[str],
    ) -> list[SearchResult]:
        """
        Expand from single memory via logical edges.
        """
        if current_depth >= max_depth:
            return []

        try:
            result = await self.client.execute_query(
                "getMemoryLogicalConnections",
                {"memory_id": memory_id},
            )

            expansion_results: list[SearchResult] = []

            edge_collections = [
                ("IMPLIES", result.get("implies_out", []), 0.9),
                ("IMPLIES", result.get("implies_in", []), 0.8),
                ("BECAUSE", result.get("because_out", []), 0.95),
                ("BECAUSE", result.get("because_in", []), 0.85),
                ("MEMORY_RELATION", result.get("relation_out", []), 0.7),
                ("MEMORY_RELATION", result.get("relation_in", []), 0.6),
            ]

            for edge_type, memories, edge_weight in edge_collections:
                if not memories:
                    continue

                for mem in memories:
                    target_id = mem.get("memory_id")
                    if not target_id or target_id in visited:
                        continue

                    visited.add(target_id)

                    graph_score = edge_weight * parent_score
                    temporal_score = calculate_temporal_freshness(
                        mem.get("created_at", ""),
                        self.config.temporal_decay_days,
                    )

                    memory_concepts = await self._load_memory_concepts(target_id)
                    concept_score = self._calculate_concept_overlap(query_concepts, memory_concepts)

                    expansion_results.append(
                        SearchResult(
                            memory_id=target_id,
                            content=mem.get("content", ""),
                            score=0.0,
                            method="onto",
                            vector_score=0.5,
                            graph_score=graph_score,
                            concept_score=concept_score,
                            temporal_score=temporal_score,
                            depth=current_depth + 1,
                            source="graph",
                            metadata={
                                "edge_type": edge_type,
                                "parent_id": memory_id,
                                "concepts": memory_concepts,
                            },
                        )
                    )

            return expansion_results

        except Exception as e:
            logger.debug("Failed to expand from %s: %s", memory_id, e)
            return []

    def _rank_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Phase 5: Calculate combined scores and rank results.

        Combined score = weighted sum of:
        - vector_score * vector_weight
        - graph_score * graph_weight
        - concept_score * concept_weight
        - temporal_score * temporal_weight
        """
        unique: dict[str, SearchResult] = {}
        for result in results:
            if result.memory_id not in unique:
                unique[result.memory_id] = result
            else:
                existing = unique[result.memory_id]
                if self._calculate_combined_score(result) > self._calculate_combined_score(
                    existing
                ):
                    unique[result.memory_id] = result

        for result in unique.values():
            result.score = self._calculate_combined_score(result)

        filtered = [r for r in unique.values() if r.score >= self.config.min_score]
        return sorted(filtered, key=lambda r: r.score, reverse=True)


    def _calculate_combined_score(self, result: SearchResult) -> float:
        """Calculate weighted combined score."""
        return (
            result.vector_score * self.config.vector_weight
            + result.graph_score * self.config.graph_weight
            + result.concept_score * self.config.concept_weight
            + result.tag_score * self.config.tag_weight
            + result.temporal_score * self.config.temporal_weight
        )

    def __repr__(self) -> str:
        return (
            f"OntoSearchStrategy(weights=[v={self.config.vector_weight:.2f}, "
            f"g={self.config.graph_weight:.2f}, c={self.config.concept_weight:.2f}, "
            f"t={self.config.temporal_weight:.2f}])"
        )
