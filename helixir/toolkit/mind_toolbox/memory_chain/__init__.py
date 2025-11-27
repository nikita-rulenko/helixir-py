"""
Memory Chain - Logical reasoning chains through memory graph.

This module builds chains of memories connected by logical relations
(IMPLIES, BECAUSE, CONTRADICTS) to provide deep context and reasoning trails.

Example:
    Query: "Why do I use Rust?"

    Chain:
    [1] "Rust is great for systems programming" (SEED - vector match)
        ↓ BECAUSE
    [2] "Rust has zero-cost abstractions" (cause)
        ↓ IMPLIES
    [3] "I write high-performance code in Rust" (effect)
        ↓ IMPLIES
    [4] "HelixDB is built in Rust" (application)
"""

from helixir.toolkit.mind_toolbox.memory_chain.builder import ChainBuilder
from helixir.toolkit.mind_toolbox.memory_chain.models import (
    ChainDirection,
    ChainNode,
    LogicalRelationType,
    MemoryChain,
    MemoryChainConfig,
)
from helixir.toolkit.mind_toolbox.memory_chain.strategy import MemoryChainStrategy

__all__ = [
    "ChainBuilder",
    "ChainDirection",
    "ChainNode",
    "LogicalRelationType",
    "MemoryChain",
    "MemoryChainConfig",
    "MemoryChainStrategy",
]
