"""Ontology Manager for managing concept hierarchy."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from helixir.core.exceptions import OntologyError
from helixir.toolkit.mind_toolbox.ontology.classifier import ConceptClassifier
from helixir.toolkit.mind_toolbox.ontology.mapper import ConceptMapper
from helixir.toolkit.mind_toolbox.ontology.models import (
    Concept,
    ConceptRelation,
    ConceptType,
    OntologyStats,
)

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


class OntologyManager:
    """
    Manager for the ontology layer.

    Handles:
    - Loading base ontology from HelixQL files
    - CRUD operations for concepts
    - Hierarchy management (IS_A, HAS_SUBTYPE)
    - Concept classification
    """

    def __init__(self, client: HelixDBClient, schema_dir: Path | None = None) -> None:
        """
        Initialize OntologyManager.

        Args:
            client: HelixDBClient instance
            schema_dir: Directory containing HelixQL schema files
        """
        self.client = client
        self.schema_dir = schema_dir or Path(__file__).parent.parent / "schema"
        self._concepts_cache: dict[str, Concept] = {}
        self._relations_cache: list[ConceptRelation] = []
        self._is_loaded = False
        self._classifier: ConceptClassifier | None = None
        self._mapper: ConceptMapper | None = None

    async def load_base_ontology(self) -> None:
        """
        Load base ontology from HelixDB.

        This loads the fundamental 3-level concept hierarchy:
        - Level 1: Root (Thing)
        - Level 2: Major categories (Attribute, Event, Entity, Relation, State)
        - Level 3: Specific concepts (Preference, Skill, Fact, etc.)

        If ontology is not initialized in DB, it will be automatically created.

        Raises:
            OntologyError: If loading fails
        """
        logger.info("Loading base ontology from HelixDB")

        try:
            result = await self.client.execute_query("checkOntologyInitialized", {})

            if not result or not result.get("thing"):
                logger.info("Base ontology not found in DB - initializing...")
                await self.client.execute_query("initializeBaseOntology", {})
                logger.info("✅ Base ontology initialized in HelixDB")
            else:
                logger.info("Base ontology already exists in HelixDB")

            await self._load_concepts_from_db()

            self._is_loaded = True
            logger.info("Base ontology loaded successfully: %d concepts", len(self._concepts_cache))

        except Exception as e:
            msg = f"Failed to load base ontology: {e}"
            raise OntologyError(msg) from e

    async def load(self) -> None:
        """Alias for load_base_ontology() for convenience."""
        await self.load_base_ontology()

    async def _load_concepts_from_db(self) -> None:
        """Load all concepts from HelixDB into cache."""
        logger.debug("Loading concepts from HelixDB into cache")

        try:
            result = await self.client.execute_query("getAllConcepts", {})

            if not result or "concepts" not in result:
                logger.warning("No concepts returned from getAllConcepts query")
                return

            concepts_data = result["concepts"]

            for concept_node in concepts_data:
                concept = Concept(
                    concept_id=concept_node.get("concept_id", ""),
                    name=concept_node.get("name", ""),
                    type=ConceptType.ABSTRACT
                    if concept_node.get("level", 0) <= 2
                    else ConceptType.CONCRETE,
                    description=concept_node.get("description", ""),
                    parent_concept=concept_node.get("parent_id")
                    if concept_node.get("parent_id")
                    else None,
                )

                self._concepts_cache[concept.concept_id] = concept
                logger.debug("Loaded concept from DB: %s", concept.concept_id)

            for concept in self._concepts_cache.values():
                if concept.parent_concept:
                    self._relations_cache.append(
                        ConceptRelation(
                            from_concept=concept.parent_concept,
                            to_concept=concept.concept_id,
                            relation_type="HAS_SUBTYPE",
                        )
                    )

            logger.info(
                "Loaded %d concepts and %d relations from HelixDB",
                len(self._concepts_cache),
                len(self._relations_cache),
            )

        except Exception as e:
            logger.exception("Failed to load concepts from HelixDB: %s", e)
            raise

    async def add_concept(self, concept: Concept) -> Concept:
        """
        Add a new concept to the ontology.

        Args:
            concept: Concept to add

        Returns:
            Added concept

        Raises:
            OntologyError: If concept already exists or addition fails
        """
        if concept.concept_id in self._concepts_cache:
            msg = f"Concept already exists: {concept.concept_id}"
            raise OntologyError(msg)

        self._concepts_cache[concept.concept_id] = concept
        logger.debug("Added concept to cache: %s", concept.concept_id)
        return concept

    def get_concept(self, concept_id: str) -> Concept | None:
        """
        Get a concept by ID from cache.

        Args:
            concept_id: Concept ID

        Returns:
            Concept or None if not found
        """
        return self._concepts_cache.get(concept_id)

    async def add_relation(self, from_concept: str, to_concept: str, relation_type: str) -> None:
        """
        Add a relation between two concepts.

        Args:
            from_concept: Source concept ID
            to_concept: Target concept ID
            relation_type: Type of relation (IS_A, HAS_SUBTYPE)

        Raises:
            OntologyError: If relation addition fails
        """
        relation = ConceptRelation(
            from_concept=from_concept, to_concept=to_concept, relation_type=relation_type
        )

        try:
            await self.client.execute_query(
                "addConceptRelation",
                {
                    "from_concept": from_concept,
                    "to_concept": to_concept,
                    "relation_type": relation_type,
                },
            )

            self._relations_cache.append(relation)
            logger.debug("Added relation: %s", relation)

        except Exception as e:
            msg = f"Failed to add relation {relation}: {e}"
            raise OntologyError(msg) from e

    async def get_subtypes(self, concept_id: str) -> list[Concept]:
        """
        Get all direct subtypes of a concept.

        Args:
            concept_id: Concept ID

        Returns:
            List of subtype concepts
        """
        try:
            result = await self.client.execute_query("getSubtypes", {"concept_id": concept_id})

            subtypes = []
            if result and "concepts" in result:
                for concept_data in result["concepts"]:
                    concept = Concept(
                        concept_id=concept_data["concept_id"],
                        name=concept_data["name"],
                        type=ConceptType(concept_data["type"]),
                        description=concept_data.get("description", ""),
                        parent_concept=concept_data.get("parent_concept"),
                    )
                    subtypes.append(concept)

            return subtypes

        except Exception as e:
            logger.warning("Failed to get subtypes of %s: %s", concept_id, e)
            return []

    async def get_ancestors(self, concept_id: str) -> list[Concept]:
        """
        Get all ancestors of a concept (up to root).

        Args:
            concept_id: Concept ID

        Returns:
            List of ancestor concepts (from immediate parent to root)
        """
        ancestors = []
        current = await self.get_concept(concept_id)

        while current and current.parent_concept:
            parent = await self.get_concept(current.parent_concept)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_stats(self) -> OntologyStats:
        """
        Get statistics about the ontology from cache.

        Returns:
            OntologyStats with current statistics
        """
        abstract_count = sum(
            1 for c in self._concepts_cache.values() if c.type == ConceptType.ABSTRACT
        )
        concrete_count = len(self._concepts_cache) - abstract_count

        [
            c.concept_id for c in self._concepts_cache.values() if not c.parent_concept
        ]

        max_depth = 0
        for concept in self._concepts_cache.values():
            depth = 0
            current = concept
            while current and current.parent_concept:
                depth += 1
                current = self._concepts_cache.get(current.parent_concept)
                if depth > 10:
                    break
            max_depth = max(max_depth, depth)

        concepts_by_type = {
            "Abstract": abstract_count,
            "Concrete": concrete_count,
        }

        return OntologyStats(
            total_concepts=len(self._concepts_cache),
            total_relations=len(self._relations_cache),
            concepts_by_type=concepts_by_type,
            max_depth=max_depth,
        )

    def is_loaded(self) -> bool:
        """Check if ontology is loaded."""
        return self._is_loaded

    def get_classifier(self) -> ConceptClassifier:
        """
        Get concept classifier.

        Returns:
            ConceptClassifier instance

        Raises:
            OntologyError: If ontology is not loaded
        """
        if not self._is_loaded:
            msg = "Ontology must be loaded before using classifier"
            raise OntologyError(msg)

        if self._classifier is None:
            self._classifier = ConceptClassifier(self._concepts_cache)

        return self._classifier

    def get_mapper(self) -> ConceptMapper:
        """
        Get concept mapper.

        Returns:
            ConceptMapper instance

        Raises:
            OntologyError: If ontology is not loaded
        """
        if not self._is_loaded:
            msg = "Ontology must be loaded before using mapper"
            raise OntologyError(msg)

        if self._mapper is None:
            self._mapper = ConceptMapper(self._concepts_cache)

        return self._mapper

    def classify_text(self, text: str, min_confidence: float = 0.3) -> list[tuple[str, float]]:
        """
        Classify text into concepts.

        Args:
            text: Text to classify
            min_confidence: Minimum confidence threshold

        Returns:
            List of (concept_id, confidence) tuples

        Example:
            >>> manager.classify_text("I love Python")
            [('Preference', 0.8)]
        """
        classifier = self.get_classifier()
        return classifier.classify(text, min_confidence)

    def suggest_concepts_for_text(self, text: str, top_n: int = 3) -> list[str]:
        """
        Suggest most likely concepts for text.

        Args:
            text: Text to analyze
            top_n: Number of concepts to return

        Returns:
            List of concept IDs
        """
        classifier = self.get_classifier()
        return classifier.suggest_concepts(text, top_n)

    def map_memory_to_concepts(
        self,
        content: str,
        memory_type: str | None = None,
        min_confidence: int = 30,
    ) -> list[tuple[str, int, str]]:
        """
        Map memory content to relevant ontology concepts.

        Args:
            content: Memory content text
            memory_type: Optional memory type (fact, preference, event, etc.)
            min_confidence: Minimum confidence threshold (0-100)

        Returns:
            List of (concept_id, confidence, link_type) tuples

        Example:
            >>> manager.map_memory_to_concepts("I love Python", "preference")
            [('Preference', 90, 'INSTANCE_OF'), ('Skill', 45, 'BELONGS_TO_CATEGORY')]
        """
        mapper = self.get_mapper()
        return mapper.map_to_concepts(content, memory_type, min_confidence)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OntologyManager(concepts={len(self._concepts_cache)}, "
            f"relations={len(self._relations_cache)}, loaded={self._is_loaded})"
        )
