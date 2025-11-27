"""Conflict detection and resolution for contradicting memories."""

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from helixir.core.exceptions import ReasoningError
from helixir.toolkit.mind_toolbox.reasoning.models import ConflictInfo

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient
    from helixir.toolkit.mind_toolbox.memory.models import Memory
    from helixir.toolkit.mind_toolbox.reasoning.engine import ReasoningEngine

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detector and resolver for conflicting memories.

    Handles:
    - Detecting contradictions between memories
    - Temporal conflicts (superseded memories)
    - Certainty-based conflict resolution
    - Multiple resolution strategies
    """

    def __init__(self, client: HelixDBClient, reasoning_engine: ReasoningEngine) -> None:
        """
        Initialize ConflictDetector.

        Args:
            client: HelixDBClient instance
            reasoning_engine: ReasoningEngine instance
        """
        self.client = client
        self.reasoning_engine = reasoning_engine
        self._conflict_cache: dict[str, ConflictInfo] = {}
        logger.info("ConflictDetector initialized")

    async def detect_contradictions(
        self,
        memory: Memory,
        candidate_memories: list[Memory],
    ) -> list[ConflictInfo]:
        """
        Detect contradictions between a memory and candidates.

        Args:
            memory: Memory to check
            candidate_memories: List of potentially conflicting memories

        Returns:
            List of detected conflicts
        """
        conflicts: list[ConflictInfo] = []

        try:
            for candidate in candidate_memories:
                if candidate.memory_id == memory.memory_id:
                    continue

                if await self._is_superseded(candidate.memory_id):
                    continue

                conflict = await self._check_contradiction(memory, candidate)
                if conflict:
                    conflicts.append(conflict)

            logger.debug(
                "Detected %d contradictions for memory %s",
                len(conflicts),
                memory.memory_id[:8],
            )

            return conflicts

        except Exception as e:
            logger.warning("Failed to detect contradictions: %s", e)
            return []

    async def _check_contradiction(
        self,
        memory_a: Memory,
        memory_b: Memory,
    ) -> ConflictInfo | None:
        """
        Check if two memories contradict each other.

        This is a simplified heuristic-based check. In a real system,
        this would use NLP/LLM to detect semantic contradictions.
        """
        contradicts_relations = await self.reasoning_engine.get_relations(
            memory_a.memory_id,
            relation_type="CONTRADICTS",
            direction="both",
        )

        for rel in contradicts_relations:
            if memory_b.memory_id in (rel.to_memory_id, rel.from_memory_id):
                return ConflictInfo(
                    memory_a_id=memory_a.memory_id,
                    memory_b_id=memory_b.memory_id,
                    conflict_type="explicit_contradiction",
                    severity=90,
                    resolution="manual_review",
                )

        if memory_a.user_id == memory_b.user_id and memory_a.memory_type == memory_b.memory_type:
            content_a_lower = memory_a.content.lower()
            content_b_lower = memory_b.content.lower()
            markers = ["not", "no longer", "never", "don't", "doesn't"]

            has_marker_a = any(marker in content_a_lower for marker in markers)
            has_marker_b = any(marker in content_b_lower for marker in markers)

            if (has_marker_a or has_marker_b) and self._has_overlapping_concepts(
                memory_a, memory_b
            ):
                return ConflictInfo(
                    memory_a_id=memory_a.memory_id,
                    memory_b_id=memory_b.memory_id,
                    conflict_type="semantic_contradiction",
                    severity=70,
                    resolution="prefer_higher_certainty",
                )

        if self._are_temporally_conflicting(memory_a, memory_b):
            return ConflictInfo(
                memory_a_id=memory_a.memory_id,
                memory_b_id=memory_b.memory_id,
                conflict_type="temporal_conflict",
                severity=60,
                resolution="prefer_newer",
            )

        return None

    def _has_overlapping_concepts(self, memory_a: Memory, memory_b: Memory) -> bool:
        """Check if two memories have overlapping concepts."""
        concepts_a = set(memory_a.concepts)
        concepts_b = set(memory_b.concepts)
        overlap = concepts_a & concepts_b
        return len(overlap) > 0

    def _are_temporally_conflicting(self, memory_a: Memory, memory_b: Memory) -> bool:
        """Check if memories are temporally conflicting."""
        now = datetime.now()

        a_is_expired = memory_a.valid_until and memory_a.valid_until < now
        b_is_expired = memory_b.valid_until and memory_b.valid_until < now

        return a_is_expired != b_is_expired

    async def _is_superseded(self, memory_id: str) -> bool:
        """Check if a memory has been superseded."""
        try:
            supersedes_relations = await self.reasoning_engine.get_relations(
                memory_id,
                relation_type="SUPERSEDES",
                direction="incoming",
            )
            return len(supersedes_relations) > 0
        except Exception:
            return False

    async def resolve_conflict(
        self,
        conflict: ConflictInfo,
        memories: dict[str, Memory],
    ) -> str | None:
        """
        Resolve a conflict using the suggested resolution strategy.

        Args:
            conflict: ConflictInfo with resolution strategy
            memories: Dict of memory_id -> Memory for lookup

        Returns:
            ID of the winning memory, or None if unresolved
        """
        memory_a = memories.get(conflict.memory_a_id)
        memory_b = memories.get(conflict.memory_b_id)

        if not memory_a or not memory_b:
            logger.warning("Cannot resolve conflict: memories not found")
            return None

        try:
            if conflict.resolution == "prefer_newer":
                winner = self._prefer_newer(memory_a, memory_b)
            elif conflict.resolution == "prefer_higher_certainty":
                winner = self._prefer_higher_certainty(memory_a, memory_b)
            elif conflict.resolution == "prefer_higher_importance":
                winner = self._prefer_higher_importance(memory_a, memory_b)
            elif conflict.resolution == "manual_review":
                logger.info(
                    "Conflict requires manual review: %s vs %s",
                    memory_a.memory_id[:8],
                    memory_b.memory_id[:8],
                )
                return None
            else:
                winner = self._default_resolution(memory_a, memory_b)

            loser = memory_a if winner == memory_b else memory_b
            await self.reasoning_engine.supersede_memory(
                old_memory_id=loser.memory_id,
                new_memory_id=winner.memory_id,
                reason=f"Conflict resolution: {conflict.resolution}",
            )

            logger.info(
                "Resolved conflict: %s wins over %s (strategy: %s)",
                winner.memory_id[:8],
                loser.memory_id[:8],
                conflict.resolution,
            )

            return winner.memory_id

        except Exception as e:
            msg = f"Failed to resolve conflict: {e}"
            raise ReasoningError(msg) from e

    def _prefer_newer(self, memory_a: Memory, memory_b: Memory) -> Memory:
        """Prefer the more recent memory."""
        return memory_a if memory_a.created_at > memory_b.created_at else memory_b

    def _prefer_higher_certainty(self, memory_a: Memory, memory_b: Memory) -> Memory:
        """Prefer the memory with higher certainty."""
        return memory_a if memory_a.certainty > memory_b.certainty else memory_b

    def _prefer_higher_importance(self, memory_a: Memory, memory_b: Memory) -> Memory:
        """Prefer the memory with higher importance."""
        return memory_a if memory_a.importance > memory_b.importance else memory_b

    def _default_resolution(self, memory_a: Memory, memory_b: Memory) -> Memory:
        """
        Default resolution: combine factors.

        Weighs: newness (40%), certainty (40%), importance (20%)
        """
        time_diff = abs((memory_a.created_at - memory_b.created_at).total_seconds())
        max(time_diff, 1)

        score_a = (
            (0.4 if memory_a.created_at > memory_b.created_at else 0.0)
            + (0.4 * memory_a.certainty / 100.0)
            + (0.2 * memory_a.importance / 100.0)
        )

        score_b = (
            (0.4 if memory_b.created_at > memory_a.created_at else 0.0)
            + (0.4 * memory_b.certainty / 100.0)
            + (0.2 * memory_b.importance / 100.0)
        )

        return memory_a if score_a > score_b else memory_b

    async def find_all_conflicts(
        self,
        memories: list[Memory],
    ) -> list[ConflictInfo]:
        """
        Find all conflicts within a list of memories.

        Args:
            memories: List of memories to check

        Returns:
            List of all detected conflicts
        """
        all_conflicts: list[ConflictInfo] = []

        for i, memory in enumerate(memories):
            candidates = memories[i + 1 :]
            conflicts = await self.detect_contradictions(memory, candidates)
            all_conflicts.extend(conflicts)

        logger.info("Found %d total conflicts in %d memories", len(all_conflicts), len(memories))

        return all_conflicts

    async def auto_resolve_conflicts(
        self,
        conflicts: list[ConflictInfo],
        memories: dict[str, Memory],
    ) -> dict[str, Any]:
        """
        Automatically resolve all conflicts that have resolution strategies.

        Args:
            conflicts: List of conflicts to resolve
            memories: Dict of memory_id -> Memory

        Returns:
            Dict with resolution stats
        """
        resolved = 0
        unresolved = 0
        winners: list[str] = []

        for conflict in conflicts:
            winner_id = await self.resolve_conflict(conflict, memories)
            if winner_id:
                resolved += 1
                winners.append(winner_id)
            else:
                unresolved += 1

        logger.info(
            "Auto-resolved %d/%d conflicts (%d require manual review)",
            resolved,
            len(conflicts),
            unresolved,
        )

        return {
            "total_conflicts": len(conflicts),
            "resolved": resolved,
            "unresolved": unresolved,
            "winners": winners,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ConflictDetector(cached_conflicts={len(self._conflict_cache)})"
