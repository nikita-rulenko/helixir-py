"""
Query Processor for intelligent search query enhancement.

Transforms user queries into enriched search queries with:
- Intent detection (what type of memory is being searched)
- Query expansion (synonyms, related terms)
- Concept hints (for OntoSearch integration)

This improves search quality by bridging the semantic gap between
how users ask questions and how memories are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixir.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


INTENT_PATTERNS: dict[str, list[str]] = {
    "preference": [
        r"\b(like|love|prefer|favorite|enjoy|fond of|into)\b",
        r"\b(what do i like|what are my favorites|my preferences)\b",
        r"\b(things i (like|love|enjoy|prefer))\b",
    ],
    "skill": [
        r"\b(can i|able to|know how|capable|proficient|skilled|expert)\b",
        r"\b(what (can|do) i (do|know)|my (skills|abilities|expertise))\b",
        r"\b(what am i good at|my strengths)\b",
    ],
    "goal": [
        r"\b(want to|goal|plan|aim|intend|aspire|wish)\b",
        r"\b(my (goals|plans|objectives|ambitions))\b",
        r"\b(what do i want|where am i going)\b",
    ],
    "fact": [
        r"\b(what is|tell me about|explain|describe|information about)\b",
        r"\b(how does|how do|what does)\b",
        r"\b(facts about|details about|specifics)\b",
    ],
    "opinion": [
        r"\b(think|believe|opinion|feel about|view on)\b",
        r"\b(what do i think|my (opinion|view|thoughts))\b",
        r"\b(how do i feel about)\b",
    ],
    "experience": [
        r"\b(did|have i|was i|when did|remember when)\b",
        r"\b(my (experience|history) with)\b",
        r"\b(what happened|past events)\b",
    ],
    "recent": [
        r"\b(today|yesterday|recently|lately|just now|this week)\b",
        r"\b(what (did|have) i (do|done)|current|latest)\b",
        r"\b(new|fresh|recent)\b",
    ],
}


EXPANSION_MAPPINGS: dict[str, list[str]] = {
    "like": ["love", "enjoy", "prefer", "fond of", "appreciate"],
    "love": ["like", "adore", "enjoy", "passionate about"],
    "prefer": ["like", "favor", "choose", "opt for"],
    "favorite": ["preferred", "beloved", "top", "best"],
    "can": ["able to", "capable of", "know how to", "proficient in"],
    "skill": ["ability", "expertise", "competence", "proficiency"],
    "good at": ["skilled in", "proficient at", "expert in", "talented at"],
    "know": ["understand", "familiar with", "experienced with"],
    "want": ["wish", "desire", "aim", "plan", "intend"],
    "goal": ["objective", "target", "aim", "ambition", "plan"],
    "plan": ["intend", "aim", "goal", "strategy"],
    "python": ["programming", "coding", "development", "backend"],
    "ai": ["artificial intelligence", "machine learning", "ml", "llm"],
    "database": ["db", "storage", "data", "persistence"],
    "api": ["endpoint", "rest", "graphql", "interface"],
    "today": ["now", "current", "recent", "latest"],
    "recently": ["lately", "just", "new", "fresh"],
}


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    original_query: str
    enhanced_query: str
    detected_intents: list[str] = field(default_factory=list)
    concept_hints: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    suggested_mode: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original_query,
            "enhanced": self.enhanced_query,
            "intents": self.detected_intents,
            "concepts": self.concept_hints,
            "expanded": self.expanded_terms,
            "mode": self.suggested_mode,
            "confidence": self.confidence,
        }


class QueryProcessor:
    """
    Processes user queries to improve search quality.

    Features:
    - Intent detection: Identifies what type of memory the user wants
    - Query expansion: Adds synonyms and related terms
    - Concept hints: Suggests ontology concepts for OntoSearch
    - Mode suggestion: Recommends search mode based on query

    Can work in two modes:
    - Rule-based (fast, no LLM required)
    - LLM-enhanced (better quality, requires LLM provider)

    Example:
        >>> processor = QueryProcessor()
        >>> result = processor.process("What do I like?")
        >>> print(result.detected_intents)
        >>> print(result.enhanced_query)
        >>> print(result.concept_hints)
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        enable_expansion: bool = True,
        max_expansions: int = 3,
    ) -> None:
        """
        Initialize QueryProcessor.

        Args:
            llm_provider: Optional LLM for enhanced processing
            enable_expansion: Whether to expand queries with synonyms
            max_expansions: Maximum expansion terms to add per word
        """
        self.llm_provider = llm_provider
        self.enable_expansion = enable_expansion
        self.max_expansions = max_expansions

        self._intent_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }

        logger.info(
            "QueryProcessor initialized: llm=%s, expansion=%s",
            "enabled" if llm_provider else "disabled",
            enable_expansion,
        )

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query to enhance search quality.

        Args:
            query: User's search query

        Returns:
            ProcessedQuery with enhanced query and metadata
        """
        query = query.strip()
        if not query:
            return ProcessedQuery(
                original_query=query,
                enhanced_query=query,
            )

        intents = self._detect_intents(query)

        concept_hints = self._intents_to_concepts(intents)

        expanded_terms = []
        if self.enable_expansion:
            expanded_terms = self._expand_query(query)

        enhanced_query = self._build_enhanced_query(query, expanded_terms)

        suggested_mode = self._suggest_mode(intents, query)

        confidence = self._calculate_confidence(intents, expanded_terms)

        result = ProcessedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            detected_intents=intents,
            concept_hints=concept_hints,
            expanded_terms=expanded_terms,
            suggested_mode=suggested_mode,
            confidence=confidence,
        )

        logger.debug(
            "Processed query: '%s' → intents=%s, concepts=%s, mode=%s",
            query[:30],
            intents,
            concept_hints,
            suggested_mode,
        )

        return result

    async def process_with_llm(self, query: str) -> ProcessedQuery:
        """
        Process query using LLM for better understanding.

        Falls back to rule-based if LLM unavailable.

        Args:
            query: User's search query

        Returns:
            ProcessedQuery with LLM-enhanced analysis
        """
        result = self.process(query)

        if not self.llm_provider:
            return result

        try:
            system_prompt = """You are a search query analyzer. Given a user query about their memories,
identify:
1. intent: What type of information they want (preference, skill, goal, fact, opinion, experience)
2. concepts: Related ontology concepts (Preference, Skill, Goal, Fact, Opinion, Technology, etc.)
3. expansion: 3-5 related terms/synonyms to improve search

Respond ONLY with valid JSON:
{"intent": "preference", "concepts": ["Preference"], "expansion": ["like", "love", "enjoy"]}"""

            user_prompt = f"Query: {query}"

            response, _ = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json_object",
            )

            import json

            llm_result = json.loads(response)

            if llm_result.get("intent"):
                if llm_result["intent"] not in result.detected_intents:
                    result.detected_intents.insert(0, llm_result["intent"])

            if llm_result.get("concepts"):
                for concept in llm_result["concepts"]:
                    if concept not in result.concept_hints:
                        result.concept_hints.append(concept)

            if llm_result.get("expansion"):
                for term in llm_result["expansion"]:
                    if term.lower() not in result.expanded_terms:
                        result.expanded_terms.append(term.lower())

            result.enhanced_query = self._build_enhanced_query(query, result.expanded_terms)
            result.confidence = min(result.confidence + 0.2, 1.0)

            logger.debug("LLM enhanced query: %s", result.to_dict())

        except Exception as e:
            logger.warning("LLM query processing failed: %s, using rule-based", e)

        return result

    def _detect_intents(self, query: str) -> list[str]:
        """Detect user intents from query."""
        intents = []
        query_lower = query.lower()

        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    if intent not in intents:
                        intents.append(intent)
                    break

        return intents

    def _intents_to_concepts(self, intents: list[str]) -> list[str]:
        """Map detected intents to ontology concepts."""
        intent_concept_map = {
            "preference": "Preference",
            "skill": "Skill",
            "goal": "Goal",
            "fact": "Fact",
            "opinion": "Opinion",
            "experience": "Experience",
            "recent": None,
        }

        concepts = []
        for intent in intents:
            concept = intent_concept_map.get(intent)
            if concept and concept not in concepts:
                concepts.append(concept)

        return concepts

    def _expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms and related terms."""
        expanded = []
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        for word in words:
            if word in EXPANSION_MAPPINGS:
                expansions = EXPANSION_MAPPINGS[word][: self.max_expansions]
                for exp in expansions:
                    if exp.lower() not in query_lower and exp.lower() not in expanded:
                        expanded.append(exp.lower())

        return expanded

    def _build_enhanced_query(self, query: str, expansions: list[str]) -> str:
        """Build enhanced query with expansions."""
        if not expansions:
            return query

        expansion_str = " ".join(expansions)
        return f"{query} {expansion_str}"

    def _suggest_mode(self, intents: list[str], query: str) -> str | None:
        """Suggest search mode based on intents and query."""
        query_lower = query.lower()

        if "recent" in intents:
            return "recent"

        if any(word in query_lower for word in ["today", "yesterday", "just", "now", "latest"]):
            return "recent"

        if any(
            word in query_lower for word in ["all", "everything", "complete", "full", "history"]
        ):
            return "deep"

        if intents:
            return "contextual"

        return None

    def _calculate_confidence(self, intents: list[str], expansions: list[str]) -> float:
        """Calculate confidence score for processing quality."""
        confidence = 0.3

        confidence += min(len(intents) * 0.15, 0.3)

        confidence += min(len(expansions) * 0.05, 0.2)

        return min(confidence, 1.0)

    def __repr__(self) -> str:
        return f"QueryProcessor(llm={'enabled' if self.llm_provider else 'disabled'})"
