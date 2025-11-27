"""
LLM Decision Engine - determines what to do with new memory.

Workflow:
1. New memory arrives
2. Vector search finds similar memories
3. LLM analyzes and decides:
   - ADD: Add as new memory
   - UPDATE: Update existing memory
   - DELETE: Delete outdated memory (on conflict)
   - NOOP: Ignore (duplicate)

Architecture:
    New Memory
        ↓
    Vector Search (find similar)
        ↓
    LLM Decision (function calling)
        ↓
    Execute Decision
"""

from enum import Enum
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.llm.providers import BaseLLMProvider

logger = logging.getLogger(__name__)


class MemoryOperation(str, Enum):
    """Possible operations for memory management."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"
    SUPERSEDE = "SUPERSEDE"
    CONTRADICT = "CONTRADICT"


class MemoryDecision(BaseModel):
    """
    Decision made by LLM about memory operation.

    Attributes:
        operation: What to do (ADD/UPDATE/DELETE/NOOP/SUPERSEDE/CONTRADICT)
        target_memory_id: ID of memory to update/delete (if applicable)
        confidence: Confidence score (0-100)
        reasoning: Why this decision was made
        merged_content: New content if UPDATE (optional)
        supersedes_memory_id: Memory being superseded (temporal replacement)
        contradicts_memory_id: Memory that contradicts this one (logical conflict)
        relates_to: List of (memory_id, relation_type) for causal relations
    """

    operation: MemoryOperation = Field(..., description="Operation to perform")
    target_memory_id: str | None = Field(None, description="Target memory ID for UPDATE/DELETE")
    confidence: int = Field(..., ge=0, le=100, description="Confidence in decision")
    reasoning: str = Field(..., description="Explanation for decision")
    merged_content: str | None = Field(None, description="Merged content for UPDATE")

    supersedes_memory_id: str | None = Field(
        None, description="Memory ID being superseded (temporal replacement)"
    )
    contradicts_memory_id: str | None = Field(
        None, description="Memory ID that contradicts this one (logical conflict)"
    )
    relates_to: list[tuple[str, str]] | None = Field(
        None, description="List of (memory_id, relation_type) for IMPLIES/BECAUSE/etc"
    )


class LLMDecisionEngine:
    """
    LLM-powered decision engine for memory operations.

    Uses LLM with function calling to decide what to do with new memories
    based on existing similar memories.

    Approach:
    - Extracts new memory
    - Finds similar existing memories
    - LLM decides: ADD/UPDATE/DELETE/NOOP
    - Prevents duplicates and conflicting information

    Usage:
        >>> engine = LLMDecisionEngine(llm_provider)
        >>> decision = await engine.decide(
        ...     new_memory="I love Python", similar_memories=[...], user_id="user123"
        ... )
        >>> print(decision.operation)
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize decision engine.

        Args:
            llm_provider: LLM provider for decision making
        """
        self.llm = llm_provider
        logger.info(
            "LLMDecisionEngine initialized: provider=%s, model=%s",
            self.llm.get_provider_name(),
            self.llm.model,
        )

    async def decide(
        self,
        new_memory: str,
        similar_memories: list[dict[str, Any]],
        user_id: str,
        similarity_threshold: float = 0.92,
    ) -> MemoryDecision:
        """
        Decide what to do with new memory based on similar existing ones.

        Args:
            new_memory: New memory content to evaluate
            similar_memories: List of similar existing memories from vector search
            user_id: User identifier for context
            similarity_threshold: Threshold for considering memories similar (0-1)

        Returns:
            MemoryDecision with operation and reasoning

        Example:
            >>> decision = await engine.decide(
            ...     new_memory="I love Python programming",
            ...     similar_memories=[{"id": "mem_1", "content": "I like Python", "score": 0.92}],
            ...     user_id="user123",
            ... )
            >>> if decision.operation == MemoryOperation.UPDATE:
            ...     print(f"Update {decision.target_memory_id}")
        """
        logger.debug(
            "Making decision: new_memory='%s...', similar_count=%d",
            new_memory[:50],
            len(similar_memories),
        )

        float_event("llm.decide.start", user_id=user_id, similar_count=len(similar_memories))

        if not similar_memories:
            float_event("llm.decide.quick_add", reason="no_similar")
            return MemoryDecision(
                operation=MemoryOperation.ADD,
                confidence=100,
                reasoning="No similar memories found, adding as new.",
            )

        highly_similar = [m for m in similar_memories if m.get("score", 0) >= similarity_threshold]

        if not highly_similar:
            float_event("llm.decide.quick_add", reason="below_threshold")
            return MemoryDecision(
                operation=MemoryOperation.ADD,
                confidence=95,
                reasoning=f"No memories above {similarity_threshold} similarity threshold, adding as new.",
            )

        prompt = self._build_decision_prompt(
            new_memory=new_memory, similar_memories=highly_similar, user_id=user_id
        )

        float_event("llm.decide.llm_call", candidates=len(highly_similar))

        try:
            system_prompt = """You are a memory management expert. Analyze the new memory and similar existing memories to decide what operation to perform.

Your goal is to:
1. Prevent duplicate information
2. Keep memory coherent and up-to-date
3. Resolve conflicts (prefer newer information)
4. Maintain information quality

Always respond with valid JSON."""

            response, _metadata = self.llm.generate(
                system_prompt=system_prompt, user_prompt=prompt, response_format="json_object"
            )

            result = json.loads(response)
            decision = MemoryDecision(**result)

            logger.info(
                "Decision made: operation=%s, confidence=%d, target=%s",
                decision.operation,
                decision.confidence,
                decision.target_memory_id,
            )

            float_event(
                "llm.decide.success",
                operation=decision.operation.value,
                confidence=decision.confidence,
            )

            return decision

        except Exception as e:
            logger.exception("Failed to make decision: %s", e)
            float_event("llm.decide.error", error=str(e))
            return MemoryDecision(
                operation=MemoryOperation.ADD,
                confidence=50,
                reasoning=f"Decision failed ({e}), defaulting to ADD.",
            )

    def _build_decision_prompt(
        self, new_memory: str, similar_memories: list[dict[str, Any]], user_id: str
    ) -> str:
        """Build prompt for LLM decision."""
        similar_str = "\n".join(
            [
                f"  ID: {m['id']}\n"
                f"  Content: {m['content']}\n"
                f"  Similarity: {m.get('score', 0):.2f}\n"
                f"  Created: {m.get('created_at', 'unknown')}\n"
                for m in similar_memories
            ]
        )

        return f"""Analyze this new memory and decide what operation to perform.

**New Memory:**
"{new_memory}"

**Similar Existing Memories:**
{similar_str}

**User ID:** {user_id}

**Your Task:**
Decide what to do with the new memory. Choose ONE operation:

1. **ADD** - Add as completely new memory
   - Use when: Information is new and different
   - Example: "I love Python" vs "I love JavaScript" (different languages)

2. **UPDATE** - Update existing memory with new information
   - Use when: New memory enhances or extends existing one
   - Example: "I love Python" → "I love Python and use it daily"
   - Provide `merged_content` combining both memories

3. **DELETE** - Delete existing conflicting memory
   - Use when: New memory is correct and old one is wrong
   - Example: Outdated fact that needs removal
   - Specify which memory to delete via `target_memory_id`

4. **NOOP** - Ignore (duplicate or redundant)
   - Use when: Information already exists
   - Example: "I love Python" already stored as "I love Python"

5. **SUPERSEDE** - Replace old memory with evolved version
   - Use when: Preference/opinion changed over time
   - Example: "I love Python" → "I now prefer Rust over Python"
   - Set `supersedes_memory_id` to old memory ID
   - Creates SUPERSEDES edge: new_memory --SUPERSEDES--> old_memory

6. **CONTRADICT** - Mark logical conflict between memories
   - Use when: Two memories contradict but both might be valid (context-dependent)
   - Example: "I love Python" vs "I hate Python" (mood change, different context)
   - Set `contradicts_memory_id` to conflicting memory ID
   - Creates CONTRADICTS edge: new_memory --CONTRADICTS--> old_memory

**Response Format (JSON):**
{{
  "operation": "ADD|UPDATE|DELETE|NOOP|SUPERSEDE|CONTRADICT",
  "target_memory_id": "mem_xxx" or null,
  "confidence": 0-100,
  "reasoning": "Why you made this decision",
  "merged_content": "New combined content" or null (only for UPDATE),
  "supersedes_memory_id": "mem_xxx" or null (for SUPERSEDE),
  "contradicts_memory_id": "mem_xxx" or null (for CONTRADICT),
  "relates_to": [["mem_xxx", "IMPLIES"], ["mem_yyy", "BECAUSE"]] or null
}}

**Important:**
- **SUPERSEDE vs UPDATE**: SUPERSEDE for temporal evolution (opinion changed), UPDATE for adding details
- **CONTRADICT vs DELETE**: CONTRADICT keeps both memories for audit, DELETE removes one
- Be conservative with DELETE (only for clear errors)
- Use NOOP liberally to avoid duplicates
- Provide clear reasoning for your decision
"""

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMDecisionEngine(provider={self.llm.get_provider_name()!r}, "
            f"model={self.llm.model!r})"
        )
