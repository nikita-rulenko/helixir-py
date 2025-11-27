"""
Seed Base Memories for Helixir.

Loads documentation and usage instructions into memory.
"""

import asyncio
from pathlib import Path
import sys

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


SEED_MEMORIES = [
    {
        "content": "Helixir is a persistent memory framework for AI assistants, built on HelixDB graph-vector database.",
        "memory_type": "fact",
        "context_tags": "helixir,documentation,overview",
    },
    {
        "content": "Helixir stores memories as nodes in a graph with vector embeddings for semantic search.",
        "memory_type": "fact",
        "context_tags": "helixir,architecture,storage",
    },
    {
        "content": "Helixir supports reasoning relations: IMPLIES, BECAUSE, and CONTRADICTS to build logical chains.",
        "memory_type": "fact",
        "context_tags": "helixir,reasoning,relations",
    },
    {
        "content": "Memory types in Helixir: fact (objective info), opinion (subjective view), preference (likes/dislikes), skill (abilities), goal (objectives), event (occurrences).",
        "memory_type": "fact",
        "context_tags": "helixir,memory_types,documentation",
    },
    {
        "content": "Use search_memory for general queries. Modes: recent (last 4h), contextual (30 days), deep (90 days), full (all time).",
        "memory_type": "instruction",
        "context_tags": "helixir,search,usage",
    },
    {
        "content": "Use search_by_concept when looking for specific types like skills, preferences, or goals. Filter by concept_type parameter.",
        "memory_type": "instruction",
        "context_tags": "helixir,search,concepts",
    },
    {
        "content": "Use search_reasoning_chain to find logical connections. Use chain_mode='causal' for 'why' questions, 'forward' for implications.",
        "memory_type": "instruction",
        "context_tags": "helixir,search,reasoning",
    },
    {
        "content": "When adding memories, include relevant context_tags for better retrieval. Tags are comma-separated.",
        "memory_type": "instruction",
        "context_tags": "helixir,best_practices,tagging",
    },
    {
        "content": "For important facts, set higher importance score (0-100). For uncertain information, lower certainty score.",
        "memory_type": "instruction",
        "context_tags": "helixir,best_practices,scoring",
    },
    {
        "content": "I am an AI assistant with persistent memory powered by Helixir. I can remember our conversations and build logical connections between facts.",
        "memory_type": "fact",
        "context_tags": "self,identity,capabilities",
    },
]


async def seed_memories() -> bool:
    """Seed base memories into Helixir."""

    try:
        from helixir import HelixirClient
        from helixir.core.config import HelixMemoryConfig
    except ImportError:
        return False

    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        return False

    try:
        config = HelixMemoryConfig.from_yaml(str(config_path))
        client = HelixirClient(config)
    except Exception:
        return False


    success_count = 0
    for _i, memory in enumerate(SEED_MEMORIES, 1):
        try:
            result = await client.add(
                content=memory["content"],
                user_id="system",
                memory_type=memory.get("memory_type", "fact"),
                context_tags=memory.get("context_tags", ""),
            )

            if result.get("added"):
                success_count += 1
            else:
                pass

        except Exception:
            pass

    return success_count > 0


def main() -> None:
    """Main entry point."""
    success = asyncio.run(seed_memories())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
