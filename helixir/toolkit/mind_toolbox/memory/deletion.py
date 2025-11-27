"""
Memory Deletion Manager - manages memory deletion with integrity preservation.

Handles:
- SOFT DELETE: mark as deleted, keep for audit trail
- HARD DELETE: permanently remove from database
- UNDELETE: restore soft-deleted memories
- ORPHAN CLEANUP: remove entities/edges without references

Architecture:
    ToolingManager → DeletionManager → MemoryManager + ReasoningEngine + EntityManager

Key Concepts:
- Soft Delete: is_deleted flag, all edges preserved
- Audit Trail: deleted_by, deleted_at for multi-agent tracking
- Referential Integrity: edges remain for reasoning chains
- Orphan Management: periodic cleanup of unused entities

Example:
    >>> deletion_mgr = DeletionManager(memory_manager, reasoning_engine, entity_manager)
    >>>
    >>> await deletion_mgr.soft_delete(
    ...     memory_id="mem-001", deleted_by="agent_123", reason="User requested forgetting"
    ... )
    >>>
    >>> await deletion_mgr.undelete(memory_id="mem-001", restored_by="admin")
    >>>
    >>> await deletion_mgr.hard_delete(
    ...     memory_id="mem-001",
    ...     cascade=False,
    ... )
"""

from enum import Enum
import logging
from typing import TYPE_CHECKING

from helixir.toolkit.misc_toolbox import float_event

if TYPE_CHECKING:
    from helixir.toolkit.mind_toolbox.entity import EntityManager
    from helixir.toolkit.mind_toolbox.memory.manager import MemoryManager
    from helixir.toolkit.mind_toolbox.reasoning.engine import ReasoningEngine

logger = logging.getLogger(__name__)


class DeletionStrategy(str, Enum):
    """Deletion strategies."""

    SOFT = "SOFT"
    HARD = "HARD"
    CASCADE = "CASCADE"


class DeletionManager:
    """
    Manager for memory deletion with integrity preservation.

    Ensures:
    - Audit trail for deletions (WHO, WHEN, WHY)
    - Referential integrity (edges handling)
    - Multi-agent safety (other agents see deletions)
    - Restore capability (soft delete)

    Default Strategy: SOFT DELETE
    - Reasoning chains remain intact
    - Other agents can see deletion history
    - Can be restored if needed
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        reasoning_engine: ReasoningEngine,
        entity_manager: EntityManager,
    ):
        """
        Initialize DeletionManager.

        Args:
            memory_manager: MemoryManager for CRUD operations
            reasoning_engine: ReasoningEngine for relations
            entity_manager: EntityManager for entity cleanup
        """
        self.memory_manager = memory_manager
        self.reasoning_engine = reasoning_engine
        self.entity_manager = entity_manager
        logger.info("DeletionManager initialized")

    async def soft_delete(self, memory_id: str, deleted_by: str, reason: str | None = None) -> bool:
        """
        Soft delete memory (mark as deleted, keep for audit).

        Process:
        1. Set is_deleted=True on Memory node
        2. Set deleted_at=now, deleted_by=agent_id
        3. Keep ALL edges intact (for audit trail)
        4. Reasoning chains remain valid
        5. Queries will filter out deleted memories by default

        Args:
            memory_id: Memory to delete
            deleted_by: Agent/user ID who deleted it
            reason: Optional reason for deletion

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If operation fails

        Example:
            >>>
            >>> await deletion_mgr.soft_delete(
            ...     memory_id="mem-001",
            ...     deleted_by="agent_123",
            ...     reason="User explicitly requested forgetting",
            ... )
            >>>
        """
        float_event("memory.deletion.soft.start", memory_id=memory_id, deleted_by=deleted_by)

        logger.warning(
            "⚠️  Soft delete not yet implemented! "
            "Need to add is_deleted field to Memory model first."
        )
        return False

    async def hard_delete(self, memory_id: str, deleted_by: str, cascade: bool = False) -> bool:
        """
        Hard delete memory (permanently remove from database).

        ⚠️  WARNING: This is DESTRUCTIVE and IRREVERSIBLE!
        Use only when absolutely necessary (e.g., GDPR compliance).

        Process:
        1. Remove Memory node from HelixDB
        2. Optionally cascade delete edges (if cascade=True)
        3. Log deletion event
        4. NO RESTORE POSSIBLE!

        Args:
            memory_id: Memory to delete
            deleted_by: Agent/user ID who deleted it
            cascade: If True, also delete all edges (dangerous!)

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If operation fails

        Example:
            >>>
            >>> await deletion_mgr.hard_delete(
            ...     memory_id="mem-001",
            ...     deleted_by="admin",
            ...     cascade=False,
            ... )
            >>>
        """
        float_event(
            "memory.deletion.hard.start",
            memory_id=memory_id,
            deleted_by=deleted_by,
            cascade=cascade,
        )

        logger.warning(
            "⚠️  Hard delete not yet implemented! Use with EXTREME caution - this is IRREVERSIBLE!"
        )
        return False

    async def undelete(self, memory_id: str, restored_by: str) -> bool:
        """
        Restore soft-deleted memory.

        Process:
        1. Set is_deleted=False
        2. Clear deleted_at, deleted_by
        3. Add restore metadata (restored_by, restored_at)
        4. Memory becomes active again

        Args:
            memory_id: Memory to restore
            restored_by: Agent/user ID who restored it

        Returns:
            True if successful

        Raises:
            HelixMemoryOperationError: If memory was hard-deleted

        Example:
            >>>
            >>> await deletion_mgr.undelete(memory_id="mem-001", restored_by="agent_123")
            >>>
        """
        float_event("memory.deletion.undelete.start", memory_id=memory_id, restored_by=restored_by)

        logger.warning("⚠️  Undelete not yet implemented! Need soft delete implementation first.")
        return False

    async def cleanup_orphans(self, dry_run: bool = True) -> dict[str, int]:
        """
        Cleanup orphaned entities and edges.

        Orphans:
        - Entities with no Memory references
        - Edges pointing to deleted memories
        - Unused Concept nodes

        Process:
        1. Find entities with ref_count = 0
        2. Find edges where source/target is deleted
        3. Optionally remove them (dry_run=False)
        4. Return statistics

        Args:
            dry_run: If True, only count orphans without deleting

        Returns:
            Dict with cleanup statistics

        Example:
            >>>
            >>> stats = await deletion_mgr.cleanup_orphans(dry_run=True)
            >>> print(f"Would delete {stats['orphaned_entities']} entities")
            >>>
            >>> stats = await deletion_mgr.cleanup_orphans(dry_run=False)
        """
        float_event("memory.deletion.cleanup.start", dry_run=dry_run)

        logger.warning(
            "⚠️  Orphan cleanup not yet implemented! This will be a periodic maintenance job."
        )

        return {
            "orphaned_entities": 0,
            "orphaned_edges": 0,
            "deleted_entities": 0,
            "deleted_edges": 0,
            "dry_run": dry_run,
        }

    def __repr__(self) -> str:
        """String representation."""
        return "DeletionManager(memory_manager, reasoning_engine, entity_manager)"
