"""
Data models for Memory Chain.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LogicalRelationType(str, Enum):
    """Types of logical relations between memories."""

    IMPLIES = "IMPLIES"
    BECAUSE = "BECAUSE"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "MEMORY_RELATION"

    @classmethod
    def from_string(cls, s: str) -> LogicalRelationType:
        """Convert string to enum, with fallback."""
        mapping = {
            "IMPLIES": cls.IMPLIES,
            "BECAUSE": cls.BECAUSE,
            "CONTRADICTS": cls.CONTRADICTS,
            "MEMORY_RELATION": cls.RELATES_TO,
        }
        return mapping.get(s.upper(), cls.RELATES_TO)


class ChainDirection(str, Enum):
    """Direction of chain traversal."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


@dataclass
class ChainNode:
    """A single node in a memory chain."""

    memory_id: str
    content: str
    memory_type: str

    relation_type: LogicalRelationType | None = None
    relation_direction: str = "out"

    depth: int = 0
    confidence: float = 1.0

    raw_data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        rel = f" ←{self.relation_type.value}← " if self.relation_type else ""
        return f"ChainNode(depth={self.depth}{rel}{self.content[:40]}...)"


@dataclass
class MemoryChain:
    """A chain of logically connected memories."""

    seed: ChainNode

    nodes: list[ChainNode] = field(default_factory=list)

    chain_type: str = "mixed"
    total_depth: int = 0
    query: str = ""

    implies_count: int = 0
    because_count: int = 0
    contradicts_count: int = 0

    def __post_init__(self) -> None:
        """Calculate statistics."""
        self._update_stats()

    def _update_stats(self) -> None:
        """Update chain statistics."""
        self.implies_count = sum(
            1 for n in self.nodes if n.relation_type == LogicalRelationType.IMPLIES
        )
        self.because_count = sum(
            1 for n in self.nodes if n.relation_type == LogicalRelationType.BECAUSE
        )
        self.contradicts_count = sum(
            1 for n in self.nodes if n.relation_type == LogicalRelationType.CONTRADICTS
        )

        if self.nodes:
            self.total_depth = max(n.depth for n in self.nodes)

        if self.contradicts_count > 0:
            self.chain_type = "contradiction"
        elif self.because_count > self.implies_count:
            self.chain_type = "causal"
        elif self.implies_count > self.because_count:
            self.chain_type = "implication"
        else:
            self.chain_type = "mixed"

    def add_node(self, node: ChainNode) -> None:
        """Add a node to the chain."""
        self.nodes.append(node)
        self._update_stats()

    def get_all_memories(self) -> list[ChainNode]:
        """Get all memories in chain including seed."""
        return [self.seed, *self.nodes]

    def get_reasoning_trail(self) -> str:
        """Get human-readable reasoning trail."""
        lines = [f"[SEED] {self.seed.content[:60]}..."]

        for node in self.nodes:
            arrow = "→" if node.relation_direction == "out" else "←"
            rel = node.relation_type.value if node.relation_type else "RELATED"
            lines.append(f"  {arrow} {rel} [{node.depth}] {node.content[:60]}...")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MemoryChain(seed={self.seed.memory_id}, nodes={len(self.nodes)}, type={self.chain_type})"


@dataclass
class MemoryChainConfig:
    """Configuration for memory chain building."""

    max_depth: int = 5
    direction: ChainDirection = ChainDirection.BOTH

    follow_implies: bool = True
    follow_because: bool = True
    follow_contradicts: bool = True
    follow_generic: bool = False

    min_confidence: float = 0.3
    max_nodes: int = 20

    allow_branching: bool = False
    max_branches: int = 3

    @classmethod
    def default(cls) -> MemoryChainConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def causal_only(cls) -> MemoryChainConfig:
        """Only follow BECAUSE relations (find causes)."""
        return cls(
            direction=ChainDirection.BACKWARD,
            follow_implies=False,
            follow_because=True,
            follow_contradicts=False,
        )

    @classmethod
    def implications_only(cls) -> MemoryChainConfig:
        """Only follow IMPLIES relations (find effects)."""
        return cls(
            direction=ChainDirection.FORWARD,
            follow_implies=True,
            follow_because=False,
            follow_contradicts=False,
        )

    @classmethod
    def deep_context(cls) -> MemoryChainConfig:
        """Deep traversal for maximum context."""
        return cls(
            max_depth=10,
            direction=ChainDirection.BOTH,
            follow_implies=True,
            follow_because=True,
            follow_contradicts=True,
            follow_generic=True,
            max_nodes=50,
        )
