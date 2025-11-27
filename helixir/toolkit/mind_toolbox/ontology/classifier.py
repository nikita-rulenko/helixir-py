"""Concept classifier for automatic categorization."""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helixir.toolkit.mind_toolbox.ontology.models import Concept

logger = logging.getLogger(__name__)


class ConceptClassifier:
    """
    Classifier for automatically categorizing text into concepts.

    Uses keyword matching and pattern recognition to determine which
    concepts apply to a given piece of text.
    """

    def __init__(self, concepts: dict[str, Concept]) -> None:
        """
        Initialize ConceptClassifier.

        Args:
            concepts: Dictionary of concept_id -> Concept
        """
        self.concepts = concepts
        self._keyword_map: dict[str, list[str]] = {}
        self._build_keyword_map()

    def _build_keyword_map(self) -> None:
        """Build keyword mapping for each concept."""
        keyword_patterns = {
            "Preference": [
                r"\b(like|love|enjoy|prefer|favorite|hate|dislike)\b",
                r"\b(rather|better than)\b",
            ],
            "Skill": [
                r"\b(can|able to|know how to|expert|proficient|good at)\b",
                r"\b(skill|ability|capability|expertise)\b",
            ],
            "Fact": [
                r"\b(is|are|was|were|has|have|had)\b",
                r"\b(born|graduated|worked|studied|lives|lived)\b",
            ],
            "Opinion": [
                r"\b(think|believe|feel|consider|seems|appears)\b",
                r"\b(in my opinion|i believe|i think)\b",
            ],
            "Goal": [
                r"\b(want|plan|intend|aim|goal|objective|aspire)\b",
                r"\b(going to|will|hope to)\b",
            ],
            "Trait": [
                r"\b(is|are)\s+(very|quite|rather)?\s*(creative|patient|ambitious|careful)\b",
                r"\b(personality|character|nature|temperament)\b",
            ],
            "Action": [
                r"\b(do|did|done|doing|perform|execute)\b",
                r"\b(action|activity|task)\b",
            ],
            "Experience": [
                r"\b(experienced|went through|encountered|faced)\b",
                r"\b(experience|event|situation)\b",
            ],
            "Achievement": [
                r"\b(achieved|accomplished|succeeded|won|completed)\b",
                r"\b(achievement|success|accomplishment|milestone)\b",
            ],
            "Person": [
                r"\b(person|people|individual|colleague|friend|family)\b",
                r"\b(he|she|they|who)\b",
            ],
            "Organization": [
                r"\b(company|organization|corporation|institution|team)\b",
                r"\b(inc|ltd|corp|llc)\b",
            ],
            "Location": [
                r"\b(place|location|city|country|area|region)\b",
                r"\b(at|in|from)\s+[A-Z][a-z]+\b",
            ],
            "Object": [
                r"\b(object|item|thing|tool|device|product)\b",
                r"\b(car|phone|computer|book)\b",
            ],
        }

        for concept_id, patterns in keyword_patterns.items():
            if concept_id in self.concepts:
                self._keyword_map[concept_id] = patterns

        logger.debug("Built keyword map for %d concepts", len(self._keyword_map))

    def classify(self, text: str, min_confidence: float = 0.3) -> list[tuple[str, float]]:
        """
        Classify text into concepts.

        Args:
            text: Text to classify
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of (concept_id, confidence) tuples, sorted by confidence

        Example:
            >>> classifier.classify("I love Python programming")
            [('Preference', 0.8), ('Skill', 0.4)]
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for concept_id, patterns in self._keyword_map.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches += 1

            if matches > 0:
                score = min(matches / len(patterns), 1.0)
                scores[concept_id] = score

        results = [(cid, conf) for cid, conf in scores.items() if conf >= min_confidence]
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Classified text into %d concepts: %s", len(results), results)
        return results

    def classify_with_ancestors(
        self, text: str, concept_hierarchy: dict[str, str | None], min_confidence: float = 0.3
    ) -> list[tuple[str, float]]:
        """
        Classify text and include ancestor concepts.

        Args:
            text: Text to classify
            concept_hierarchy: Mapping of concept_id -> parent_concept_id
            min_confidence: Minimum confidence threshold

        Returns:
            List of (concept_id, confidence) tuples including ancestors
        """
        base_classifications = self.classify(text, min_confidence)
        result_map: dict[str, float] = {}

        for concept_id, confidence in base_classifications:
            result_map[concept_id] = confidence

            current_id = concept_id
            ancestor_confidence = confidence * 0.8

            while current_id in concept_hierarchy:
                parent_id = concept_hierarchy[current_id]
                if parent_id is None:
                    break

                if parent_id not in result_map or result_map[parent_id] < ancestor_confidence:
                    result_map[parent_id] = ancestor_confidence

                current_id = parent_id
                ancestor_confidence *= 0.8

        results = list(result_map.items())
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def suggest_concepts(self, text: str, top_n: int = 3) -> list[str]:
        """
        Suggest top N most likely concepts for text.

        Args:
            text: Text to analyze
            top_n: Number of concepts to return

        Returns:
            List of concept IDs (most likely first)
        """
        classifications = self.classify(text, min_confidence=0.1)
        return [cid for cid, _conf in classifications[:top_n]]

    def is_concept(self, text: str, concept_id: str, threshold: float = 0.5) -> bool:
        """
        Check if text strongly matches a specific concept.

        Args:
            text: Text to check
            concept_id: Concept to check against
            threshold: Confidence threshold

        Returns:
            True if confidence >= threshold
        """
        classifications = self.classify(text, min_confidence=threshold)
        return any(cid == concept_id for cid, _conf in classifications)

    def batch_classify(
        self, texts: Sequence[str], min_confidence: float = 0.3
    ) -> list[list[tuple[str, float]]]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify
            min_confidence: Minimum confidence threshold

        Returns:
            List of classification results (one per text)
        """
        return [self.classify(text, min_confidence) for text in texts]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConceptClassifier(concepts={len(self.concepts)}, keywords={len(self._keyword_map)})"
        )
