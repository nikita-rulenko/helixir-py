"""Concept Mapper for linking memories to ontology concepts."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixir.toolkit.mind_toolbox.ontology.models import Concept

logger = logging.getLogger(__name__)


class ConceptMapper:
    """
    Maps text content to ontology concepts.

    Uses keyword matching and semantic analysis to determine which
    concepts from the base ontology are relevant to a given text.

    Handles two types of concept linking:
    - INSTANCE_OF: Memory is an instance of this concept (e.g., "I love Python" -> Preference)
    - BELONGS_TO_CATEGORY: Memory belongs to this category (e.g., work-related -> Work category)
    """

    def __init__(self, concepts_cache: dict[str, Concept]) -> None:
        """
        Initialize ConceptMapper.

        Args:
            concepts_cache: Dictionary of concept_id -> Concept
        """
        self.concepts_cache = concepts_cache
        self._build_keyword_index()

        logger.info("ConceptMapper initialized with %d concepts", len(concepts_cache))

    def _build_keyword_index(self) -> None:
        """Build keyword index for fast concept lookup."""
        self.keyword_to_concepts: dict[str, list[str]] = {}

        concept_keywords = {
            "Preference": ["like", "love", "prefer", "favorite", "enjoy", "hate", "dislike"],
            "Skill": [
                "can",
                "able to",
                "skilled at",
                "expert in",
                "know how",
                "proficient",
                "skill",
                "skills",
                "ability",
                "abilities",
                "capable",
                "competent",
                "do well",
                "good at",
                "experience in",
                "familiar with",
            ],
            "Trait": ["is", "am", "are", "personality", "character", "nature", "tends to"],
            "Goal": ["want", "goal", "aim", "plan", "wish", "hope", "intend", "going to"],
            "Opinion": ["think", "believe", "feel", "opinion", "view", "consider", "reckon"],
            "Fact": ["fact", "is", "has", "knows", "information", "data", "true"],
            "Action": ["did", "does", "doing", "performed", "executed", "ran", "action"],
            "Experience": ["experienced", "went through", "encounter", "witnessed", "lived"],
            "Achievement": [
                "completed",
                "finished",
                "achieved",
                "success",
                "accomplished",
                "milestone",
            ],
            "Person": ["person", "human", "user", "developer", "engineer", "colleague", "someone"],
            "Organization": ["company", "organization", "team", "group", "enterprise", "corp"],
            "Location": ["place", "location", "city", "country", "address", "site", "where"],
            "Object": ["object", "item", "thing", "device", "tool", "product"],
            "Technology": [
                "technology",
                "tech",
                "system",
                "software",
                "hardware",
                "platform",
                "framework",
                "library",
                "database",
                "api",
                "tool",
                "infrastructure",
                "service",
            ],
        }

        for concept_id, keywords in concept_keywords.items():
            if concept_id not in self.concepts_cache:
                logger.warning(
                    "Concept '%s' in keyword index but not in cache - skipping", concept_id
                )
                continue

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_to_concepts:
                    self.keyword_to_concepts[keyword_lower] = []
                self.keyword_to_concepts[keyword_lower].append(concept_id)

        logger.debug("Keyword index built with %d keywords", len(self.keyword_to_concepts))

    def map_to_concepts(
        self,
        text: str,
        memory_type: str | None = None,
        min_confidence: int = 30,
    ) -> list[tuple[str, int, str]]:
        """
        Map text to relevant ontology concepts.

        Args:
            text: Text to analyze
            memory_type: Optional memory type hint (fact, preference, event, etc.)
            min_confidence: Minimum confidence threshold (0-100)

        Returns:
            List of (concept_id, confidence, link_type) tuples
            link_type is either 'INSTANCE_OF' or 'BELONGS_TO_CATEGORY'

        Example:
            >>> mapper.map_to_concepts("I love Python programming")
            [('Preference', 85, 'INSTANCE_OF'), ('Skill', 45, 'BELONGS_TO_CATEGORY')]
        """
        ENTITY_TYPE_BLACKLIST = {"system", "component", "resource", "process", "event", "concept"}

        text_lower = text.lower()
        results: dict[str, tuple[int, str]] = {}

        if memory_type:
            type_to_concept = {
                "preference": ("Preference", 90, "INSTANCE_OF"),
                "skill": ("Skill", 90, "INSTANCE_OF"),
                "fact": ("Fact", 80, "INSTANCE_OF"),
                "goal": ("Goal", 90, "INSTANCE_OF"),
                "opinion": ("Opinion", 85, "INSTANCE_OF"),
                "trait": ("Trait", 85, "INSTANCE_OF"),
                "event": ("Event", 70, "INSTANCE_OF"),
            }

            if memory_type in type_to_concept:
                concept_id, confidence, link_type = type_to_concept[memory_type]
                if concept_id in self.concepts_cache:
                    results[concept_id] = (confidence, link_type)

        for keyword, concept_ids in self.keyword_to_concepts.items():
            if keyword in text_lower:
                for concept_id in concept_ids:
                    if concept_id.lower() in ENTITY_TYPE_BLACKLIST:
                        logger.debug(
                            "Skipping '%s' - it's an entity type, not ontology concept", concept_id
                        )
                        continue

                    if concept_id not in self.concepts_cache:
                        continue

                    base_confidence = 60 if concept_id not in results else 40

                    word_count = text_lower.count(keyword)
                    confidence = min(base_confidence + (word_count - 1) * 10, 95)

                    if concept_id in results:
                        existing_conf, existing_type = results[concept_id]
                        if confidence > existing_conf:
                            results[concept_id] = (confidence, existing_type)
                    else:
                        link_type = "BELONGS_TO_CATEGORY"
                        results[concept_id] = (confidence, link_type)

        filtered_results = [
            (concept_id, conf, link_type)
            for concept_id, (conf, link_type) in results.items()
            if conf >= min_confidence
        ]

        filtered_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Mapped text to %d concepts: %s", len(filtered_results), filtered_results)

        return filtered_results

    def suggest_concepts_for_memory(
        self,
        content: str,
        memory_type: str | None = None,
        top_n: int = 3,
    ) -> list[str]:
        """
        Suggest most relevant concept IDs for a memory.

        This is a convenience method that returns only concept IDs.

        Args:
            content: Memory content
            memory_type: Optional memory type
            top_n: Maximum number of concepts to return

        Returns:
            List of concept IDs
        """
        mappings = self.map_to_concepts(content, memory_type, min_confidence=30)
        return [concept_id for concept_id, _, _ in mappings[:top_n]]

    def get_concept_statistics(self) -> dict[str, Any]:
        """
        Get statistics about concept mapping capabilities.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_concepts": len(self.concepts_cache),
            "indexed_keywords": len(self.keyword_to_concepts),
            "mappable_concepts": len(
                {
                    concept_id
                    for concept_ids in self.keyword_to_concepts.values()
                    for concept_id in concept_ids
                }
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_concept_statistics()
        return (
            f"ConceptMapper(concepts={stats['total_concepts']}, "
            f"keywords={stats['indexed_keywords']}, "
            f"mappable={stats['mappable_concepts']})"
        )
