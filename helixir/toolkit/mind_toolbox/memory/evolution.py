"""
Memory Evolution - manages memory changes over time.

Handles:
- SUPERSESSION: new information replaces old (preference changes, updates)
- CONTRADICTION: new information conflicts with old
- ENHANCEMENT: new information enriches old (adds details)

Architecture:
    ToolingManager → MemoryEvolution → MemoryManager + ReasoningEngine

Key Concepts:
- Temporal Validity: old memories get valid_until
- Reasoning Relations: SUPERSEDES, CONTRADICTS edges for audit trail
- Multi-Agent: each change has WHO, WHEN, WHY

Example:
    >>> evolution = MemoryEvolution(memory_manager, reasoning_engine)
    >>>
    >>> await evolution.handle_supersession(
    ...     old_memory_id="mem-001",
    ...     new_memory_id="mem-002",
    ...     reason="Preference changed from Python to Rust",
    ...     changed_by="agent_123",
    ... )
    >>>
    >>>
    >>>
    >>>
"""

from datetime import datetime
import logging
from typing import TYPE_CHECKING

from helixir.core.exceptions import HelixMemoryOperationError
from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.toolkit.mind_toolbox.memory.manager import MemoryManager
    from helixir.toolkit.mind_toolbox.reasoning.engine import ReasoningEngine

logger = logging.getLogger(__name__)


class MemoryEvolution:
    """
    Manager for memory evolution and change tracking.

    Coordinates between MemoryManager and ReasoningEngine to ensure
    proper graph markup when memories change over time.

    Key Operations:
    - handle_supersession: New memory replaces old (temporal change)
    - handle_contradiction: New memory conflicts with old (logical conflict)
    - handle_enhancement: New memory enriches old (adds details)
    """

    def __init__(self, memory_manager: MemoryManager, reasoning_engine: ReasoningEngine):
        """
        Initialize MemoryEvolution.

        Args:
            memory_manager: MemoryManager instance for CRUD operations
            reasoning_engine: ReasoningEngine for creating relations
        """
        self.memory_manager = memory_manager
        self.reasoning_engine = reasoning_engine
        logger.info("MemoryEvolution initialized")

    async def handle_supersession(
        self,
        old_memory_id: str,
        new_memory_id: str,
        reason: str | None = None,
        changed_by: str | None = None,
    ) -> bool:
        """
        Handle memory supersession (replacement).

        Process:
        1. Set valid_until=now on old memory (temporal boundary)
        2. Create SUPERSEDES edge (new → old) for audit trail
        3. Log evolution event

        Args:
            old_memory_id: Memory being replaced
            new_memory_id: New memory
            reason: Why this supersession happened
            changed_by: Agent/user who made the change

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If operation fails

        Example:
            >>>
            >>> await evolution.handle_supersession(
            ...     old_memory_id="mem-001",
            ...     new_memory_id="mem-002",
            ...     reason="Preference changed",
            ...     changed_by="agent_123",
            ... )
            >>>
        """
        float_event(
            "memory.evolution.supersession.start",
            old_id=old_memory_id,
            new_id=new_memory_id,
            changed_by=changed_by,
        )

        try:
            logger.info("Setting temporal boundary on old memory: %s", old_memory_id[:8])

            updated_memory = await self.memory_manager.update_memory(
                memory_id=old_memory_id, valid_until=datetime.now()
            )

            if not updated_memory:
                msg = f"Old memory not found: {old_memory_id}"
                raise HelixMemoryOperationError(msg)

            float_event(
                "memory.evolution.temporal_boundary_set",
                memory_id=old_memory_id,
                valid_until=updated_memory.valid_until.isoformat(),
            )

            logger.info("Creating SUPERSEDES edge: %s → %s", new_memory_id[:8], old_memory_id[:8])

            success = await self.reasoning_engine.supersede_memory(
                old_memory_id=old_memory_id,
                new_memory_id=new_memory_id,
                reason=reason or "Memory superseded",
            )

            if not success:
                logger.warning("Failed to create SUPERSEDES edge, but temporal boundary set")

            float_event(
                "memory.evolution.supersession.complete",
                old_id=old_memory_id,
                new_id=new_memory_id,
                edge_created=success,
            )

            logger.info(
                "✅ Memory supersession complete: %s supersedes %s",
                new_memory_id[:8],
                old_memory_id[:8],
            )

            return True

        except Exception as e:
            logger.exception("Failed to handle supersession: %s", e)
            float_event("memory.evolution.supersession.error", error=str(e))
            raise HelixMemoryOperationError(f"Supersession failed: {e}") from e

    async def handle_contradiction(
        self,
        existing_memory_id: str,
        new_memory_id: str,
        explanation: str | None = None,
        detected_by: str | None = None,
        confidence: int = 80,
    ) -> bool:
        """
        Handle memory contradiction (conflict).

        Process:
        1. Create CONTRADICTS edge (bidirectional: both conflict)
        2. Both memories stay active (for audit and resolution)
        3. Log conflict for later resolution

        Args:
            existing_memory_id: Existing memory
            new_memory_id: New conflicting memory
            explanation: Why they contradict
            detected_by: Agent/user who detected conflict
            confidence: Confidence in contradiction (0-100)

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If operation fails

        Example:
            >>>
            >>> await evolution.handle_contradiction(
            ...     existing_memory_id="mem-001",
            ...     new_memory_id="mem-002",
            ...     explanation="Performance benchmarks differ",
            ...     detected_by="agent_123",
            ... )
            >>>
        """
        float_event(
            "memory.evolution.contradiction.start",
            existing_id=existing_memory_id,
            new_id=new_memory_id,
            detected_by=detected_by,
        )

        try:
            logger.info(
                "Creating CONTRADICTS edge: %s ⇄ %s", new_memory_id[:8], existing_memory_id[:8]
            )

            await self.reasoning_engine.add_relation(
                from_memory_id=new_memory_id,
                to_memory_id=existing_memory_id,
                relation_type="CONTRADICTS",
                strength=confidence,
                confidence=confidence,
                explanation=explanation or "Memories contain conflicting information",
            )

            await self.reasoning_engine.add_relation(
                from_memory_id=existing_memory_id,
                to_memory_id=new_memory_id,
                relation_type="CONTRADICTS",
                strength=confidence,
                confidence=confidence,
                explanation=explanation or "Memories contain conflicting information",
            )

            float_event(
                "memory.evolution.contradiction.detected",
                existing_id=existing_memory_id,
                new_id=new_memory_id,
                confidence=confidence,
            )

            logger.warning(
                "⚠️  Memory contradiction detected and logged: %s ⇄ %s",
                new_memory_id[:8],
                existing_memory_id[:8],
            )

            return True

        except Exception as e:
            logger.exception("Failed to handle contradiction: %s", e)
            float_event("memory.evolution.contradiction.error", error=str(e))
            raise HelixMemoryOperationError(f"Contradiction handling failed: {e}") from e

    async def handle_enhancement(
        self,
        original_memory_id: str,
        enhanced_content: str,
        enhancement_reason: str | None = None,
        enhanced_by: str | None = None,
    ) -> bool:
        """
        Handle memory enhancement (enrichment with details).

        Process:
        1. Update original memory content in-place
        2. Update updated_at timestamp
        3. NO new memory node created (different from supersession!)
        4. Optionally log enhancement in metadata

        Args:
            original_memory_id: Memory to enhance
            enhanced_content: New enriched content
            enhancement_reason: Why it was enhanced
            enhanced_by: Agent/user who enhanced it

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If operation fails

        Example:
            >>>
            >>> await evolution.handle_enhancement(
            ...     original_memory_id="mem-001",
            ...     enhanced_content="User loves Python for data science and ML",
            ...     enhancement_reason="Added context about usage",
            ...     enhanced_by="agent_123",
            ... )
            >>>
        """
        float_event(
            "memory.evolution.enhancement.start",
            memory_id=original_memory_id,
            enhanced_by=enhanced_by,
        )

        try:
            logger.info("Enhancing memory: %s", original_memory_id[:8])

            updated_memory = await self.memory_manager.update_memory(
                memory_id=original_memory_id, content=enhanced_content
            )

            if not updated_memory:
                msg = f"Memory not found: {original_memory_id}"
                raise HelixMemoryOperationError(msg)

            float_event(
                "memory.evolution.enhancement.complete",
                memory_id=original_memory_id,
                content_length=len(enhanced_content),
            )

            logger.info("✅ Memory enhanced: %s", original_memory_id[:8])

            return True

        except Exception as e:
            logger.exception("Failed to handle enhancement: %s", e)
            float_event("memory.evolution.enhancement.error", error=str(e))
            raise HelixMemoryOperationError(f"Enhancement failed: {e}") from e

    def __repr__(self) -> str:
        """String representation."""
        return "MemoryEvolution(memory_manager, reasoning_engine)"
