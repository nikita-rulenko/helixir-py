"""LLM-based memory extraction from text."""

import json
import logging
from typing import Any

from helixir.core.exceptions import ValidationError
from helixir.llm.models import ExtractedEntity, ExtractedMemory, ExtractedRelation, ExtractionResult
from helixir.llm.providers import BaseLLMProvider, OllamaProvider, OpenAIProvider
from helixir.toolkit.misc_toolbox import float_event

logger = logging.getLogger(__name__)


class LLMExtractor:
    """
    Extracts structured memory information from text using LLM.

    Supports multiple LLM providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Ollama (gemma2, llama2, mistral, etc.)
    """

    def __init__(
        self,
        provider: str | BaseLLMProvider = "ollama",
        model: str = "gemma2",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        """
        Initialize LLMExtractor.

        Args:
            provider: LLM provider ("openai", "ollama") or BaseLLMProvider instance
            model: Model name
            api_key: API key for the provider (if needed)
            base_url: Optional custom API base URL
            temperature: Sampling temperature (0.0-1.0)
        """
        if isinstance(provider, BaseLLMProvider):
            self.provider = provider
        elif provider == "openai":
            self.provider = OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
            )
        elif provider == "cerebras":
            from helixir.llm.providers.cerebras import CerebrasProvider

            self.provider = CerebrasProvider(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        elif provider == "ollama":
            self.provider = OllamaProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
            )
        else:
            msg = f"Unsupported provider: {provider}. Use 'openai', 'cerebras', or 'ollama'"
            raise ValueError(msg)

        logger.info(
            "LLMExtractor initialized: provider=%s, model=%s",
            self.provider.get_provider_name(),
            self.provider.model,
        )

    async def extract(
        self,
        text: str,
        user_id: str,
        extract_entities: bool = True,
        extract_relations: bool = True,
        ontology_concepts: list[str] | None = None,
    ) -> ExtractionResult:
        """
        Extract memories, entities, and relations from text.

        Args:
            text: Input text to process
            user_id: User ID for context
            extract_entities: Whether to extract entities
            extract_relations: Whether to extract reasoning relations
            ontology_concepts: Optional list of known concepts for classification

        Returns:
            ExtractionResult with all extracted information
        """
        float_event("llm.extract.start", user_id=user_id, text_len=len(text))

        if not text.strip():
            float_event("llm.extract.error", error="Empty text")
            msg = "Input text cannot be empty"
            raise ValidationError(msg)

        prompt = self._build_extraction_prompt(
            text,
            extract_entities=extract_entities,
            extract_relations=extract_relations,
            ontology_concepts=ontology_concepts,
        )

        float_event("llm.extract.llm_call", provider=self.provider.get_provider_name())

        try:
            system_prompt = "You are an expert at extracting structured memory information from text. Always respond with valid JSON."
            content, metadata = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                response_format="json_object",
            )

            float_event("llm.extract.llm_response", tokens=metadata.get("eval_count", 0))

            result_dict = json.loads(content)

            result = self._parse_extraction_result(result_dict, user_id)

            result.metadata = {
                **metadata,
                "user_id": user_id,
            }

            logger.info(
                "Extracted %d memories, %d entities, %d relations",
                len(result.memories),
                len(result.entities),
                len(result.relations),
            )

            float_event(
                "llm.extract.success",
                memories=len(result.memories),
                entities=len(result.entities),
                relations=len(result.relations),
            )

            return result

        except json.JSONDecodeError as e:
            logger.exception("Failed to parse LLM response as JSON: %s", e)
            float_event("llm.extract.error", error=f"JSON parse: {e}")
            return ExtractionResult(
                metadata={
                    "provider": self.provider.get_provider_name(),
                    "model": self.provider.model,
                    "error": f"JSON parse error: {e}",
                }
            )
        except Exception as e:
            logger.exception("LLM extraction failed: %s", e)
            float_event("llm.extract.error", error=str(e))
            return ExtractionResult(
                metadata={
                    "provider": self.provider.get_provider_name(),
                    "model": self.provider.model,
                    "error": str(e),
                }
            )

    def _build_extraction_prompt(
        self,
        text: str,
        extract_entities: bool = True,
        extract_relations: bool = True,
        ontology_concepts: list[str] | None = None,
    ) -> str:
        """Build the extraction prompt for the LLM."""
        concepts_str = ""
        if ontology_concepts:
            concepts_str = f"\nAvailable concepts: {', '.join(ontology_concepts)}"

        prompt = f"""Extract structured memory information from the following text.

Text:
\"\"\"{text}\"\"\"

Extract the following:{concepts_str}

1. **Memories**: Individual facts, preferences, events, skills, goals, or opinions.
   - content: The actual memory
   - memory_type: fact, preference, event, skill, goal, opinion, trait
   - certainty: 0-100 (how confident you are in this information)
   - importance: 0-100 (how important this memory is)
   - concepts: List of ONLY base ontology concepts (see available concepts below)
   - context: Dict with boolean flags (work, personal, travel, etc.)

   **IMPORTANT for concepts field:**
   - Use ONLY concepts from base ontology: Preference, Skill, Fact, Opinion, Goal, Trait,
     Action, Experience, Achievement, Person, Organization, Location, Object, Technology
   - DO NOT add specific terms like "GPU", "Rust", "database", "AI" to concepts
   - Specific terms should go to entity properties
   - Example: "I love Python programming" →
     * concepts: ["Preference"]  (Python is Technology CONCEPT, not entity)
     * entities: [{{ "name": "Python", "type": "System" }}]  (System because it's a programming language/platform)

"""

        if extract_entities:
            prompt += """2. **Entities**: Extract entities. IMPORTANT: Use ONLY these exact entity types:

   **ALLOWED ENTITY TYPES (use exactly as written):**
   - Person (human individuals)
   - Organization (companies, teams, groups)
   - Location (places, addresses, regions)
   - System (complete systems: databases, frameworks, platforms)
   - Component (parts of systems: modules, classes, functions)
   - Resource (files, URLs, APIs, configs)
   - Concept (abstractions: patterns, principles, methodologies)
   - Process (activities: deployment, compilation, testing)
   - Event (occurrences: errors, milestones, releases)

   **CRITICAL: DO NOT confuse entity types with ontology concepts!**
   - ❌ NEVER use "Object" or "Technology" as entity_type (they are ontology concepts only)
   - ❌ NEVER use "System" in concepts field (it's an entity type only)
   - ✅ For physical objects → use entity_type: "Resource" or "Component"
   - ✅ For technologies → use entity_type: "System" (e.g. Python, HelixDB, Rust)

   For each entity:
   - name: Entity name (string)
   - entity_type: ONE of the 9 types listed above (EXACT spelling, case-sensitive)
   - properties: Metadata dict with ANY types (str, int, list, dict)

   DO NOT use other entity types like "Digital", "Physical", "Software", "Hardware", "Tool", "Database", "Object", "Technology".
   Map them to the allowed types:
   - Database/Framework/Platform/Programming Language → System
   - Module/Class/Function → Component
   - File/URL/API/Config → Resource
   - Pattern/Principle/Methodology → Concept
   - Physical objects/tools → Resource or Component

"""

        if extract_relations:
            prompt += """3. **Relations**: Logical relationships between memories.

   CRITICAL: Both from_memory_content and to_memory_content MUST be filled!
   - from_memory_content: FULL content of source memory (string, required)
   - to_memory_content: FULL content of target memory (string, required)
   - relation_type: IMPLIES, BECAUSE, CONTRADICTS, SUPERSEDES, DERIVED_FROM
   - strength: 0-100
   - confidence: 0-100
   - explanation: Why this relation exists

   Example:
   {{
     "from_memory_content": "Python is a programming language",
     "to_memory_content": "Python is used for AI development",
     "relation_type": "IMPLIES",
     "strength": 80,
     "confidence": 90,
     "explanation": "Programming languages enable development"
   }}

   If you cannot identify BOTH memories fully, skip the relation!

"""

        prompt += """Return a JSON object with this structure:
{
  "memories": [...],
  "entities": [...],
  "relations": [...],
  "summary": "Brief summary of what was extracted"
}

Be thorough but concise. Extract only meaningful information. Return ONLY the JSON, no additional text."""

        return prompt

    def _parse_extraction_result(
        self, result_dict: dict[str, Any], user_id: str
    ) -> ExtractionResult:
        """Parse LLM response into ExtractionResult."""
        memories = [ExtractedMemory(**mem_data) for mem_data in result_dict.get("memories", [])]

        entities = [ExtractedEntity(**ent_data) for ent_data in result_dict.get("entities", [])]

        relations = []
        for rel_data in result_dict.get("relations", []):
            try:
                if not rel_data.get("from_memory_content") or not rel_data.get("to_memory_content"):
                    logger.debug("Skipping relation with empty required fields: %s", rel_data)
                    continue
                relations.append(ExtractedRelation(**rel_data))
            except Exception as e:
                logger.debug("Failed to parse relation, skipping: %s - %s", rel_data, e)
                continue

        summary = result_dict.get("summary")

        return ExtractionResult(
            memories=memories,
            entities=entities,
            relations=relations,
            summary=summary,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMExtractor(provider={self.provider.get_provider_name()!r}, "
            f"model={self.provider.model!r})"
        )
