"""
Leveling Structure - pyramid of Helixir framework levels.

Helixir is built on incremental levels (Level 0-5), where each level
adds new functionality on top of previous ones. This is called "Float System" -
where each level "floats" on the previous one.

Principles:
----------
1. Each level DEPENDS on previous ones
2. Schema ACCUMULATES (Level N includes Level 0..N-1)
3. Queries ACCUMULATE (can use entities from all previous levels)
4. Deployment is INCREMENTAL (cannot deploy Level 3 without Level 0-2)
5. Validation is BY LEVEL (Float validation before deployment)

Architecture:
-----------
    Level 0 (User)                     ← Base graph
        ↓
    Level 1 (Memory + Entity)          ← Memory CRUD
        ↓
    Level 2 (Context)                  ← Contexts and search
        ↓
    Level 3 (Temporal + UPDATE)        ← Temporal queries
        ↓
    Level 4 (Relations + Reasoning)    ← Causality and conflicts
        ↓
    Level 5 (Vectors + Embeddings)     ← Vector search
"""

from dataclasses import dataclass, field
from enum import IntEnum


class HelixirLevel(IntEnum):
    """Helixir framework levels."""

    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5


@dataclass
class LevelDefinition:
    """
    Framework level definition.

    Attributes:
        level: Level number
        name: Level name
        description: Functionality description
        schema_nodes: List of nodes this level adds
        schema_edges: List of edges this level adds
        schema_extends: List of nodes it extends (adds fields)
        queries: List of queries this level adds
        dependencies: Which levels it depends on
        notes: Additional notes/issues
    """

    level: HelixirLevel
    name: str
    description: str
    schema_nodes: list[str] = field(default_factory=list)
    schema_edges: list[str] = field(default_factory=list)
    schema_extends: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    dependencies: list[HelixirLevel] = field(default_factory=list)
    notes: str = ""


LEVEL_0 = LevelDefinition(
    level=HelixirLevel.LEVEL_0,
    name="User Management",
    description="Base level: user management",
    schema_nodes=["User"],
    schema_edges=[],
    queries=["addUser", "getUser"],
    dependencies=[],
    notes="Foundation. No User = no memory.",
)

LEVEL_1 = LevelDefinition(
    level=HelixirLevel.LEVEL_1,
    name="Memory CRUD",
    description="CRUD operations for memory and entities",
    schema_nodes=["Memory", "Entity"],
    schema_edges=["OWNS", "MENTIONS"],
    queries=[
        "addMemory",
        "getMemory",
        "addEntity",
        "getEntity",
        "getMemoriesByUser",
        "getEntitiesByMemory",
    ],
    dependencies=[HelixirLevel.LEVEL_0],
    notes="Framework core. Memory linked to User via OWNS.",
)

LEVEL_2 = LevelDefinition(
    level=HelixirLevel.LEVEL_2,
    name="Context & Search",
    description="Contexts and basic memory search",
    schema_nodes=["Context"],
    schema_edges=["IN_CONTEXT"],
    queries=[
        "addContext",
        "getContext",
        "getMemoriesByContext",
        "searchMemories",
        "searchMemoriesByKeyword",
    ],
    dependencies=[HelixirLevel.LEVEL_0, HelixirLevel.LEVEL_1],
    notes="Contexts for memory grouping. Search without vectors.",
)

LEVEL_3 = LevelDefinition(
    level=HelixirLevel.LEVEL_3,
    name="Temporal & Update",
    description="Temporal queries and UPDATE operations",
    schema_nodes=[],
    schema_edges=[],
    queries=["updateMemory", "getRecentMemories", "searchRecentMemories", "getMemoriesByDateRange"],
    dependencies=[HelixirLevel.LEVEL_0, HelixirLevel.LEVEL_1, HelixirLevel.LEVEL_2],
    notes=(
        "ISSUE: UPDATE in HelixQL requires two-query pattern:\n"
        "1) WHERE to get internal ID\n"
        "2) UPDATE with internal ID\n"
        "Parameters in WHERE must be constants!"
    ),
)

LEVEL_4 = LevelDefinition(
    level=HelixirLevel.LEVEL_4,
    name="Relations & Reasoning",
    description="Reasoning relations (causality and conflicts)",
    schema_nodes=["ReasoningRelation"],
    schema_edges=[
        "IMPLIES",
        "BECAUSE",
        "CONTRADICTS",
        "SUPERSEDES",
        "DERIVED_FROM",
        "SUPPORTS",
        "REFUTES",
    ],
    queries=[
        "addMemoryRelation",
        "getMemoryRelations",
        "getReasoningChain",
        "detectConflicts",
        "getRelatedMemories",
    ],
    dependencies=[HelixirLevel.LEVEL_1],
    notes=(
        "FRAMEWORK CORE! This is where reasoning happens - understanding WHY.\n"
        "Relations are built between Memory nodes."
    ),
)

LEVEL_5 = LevelDefinition(
    level=HelixirLevel.LEVEL_5,
    name="Vectors & Embeddings",
    description="Vector search and embeddings",
    schema_nodes=[],
    schema_edges=[],
    schema_extends=["Memory"],
    queries=[
        "addMemoryWithVector",
        "searchVectorMemories",
        "searchMemoriesByText",
        "searchHybrid",
    ],
    dependencies=[HelixirLevel.LEVEL_1],
    notes=(
        "ISSUE: Embed() in queries was unstable.\n"
        "SOLUTION: Client-side embeddings via LLM client.\n"
        "Schema extends Memory, adding vector and embedding_model fields."
    ),
)

LEVELS: dict[HelixirLevel, LevelDefinition] = {
    HelixirLevel.LEVEL_0: LEVEL_0,
    HelixirLevel.LEVEL_1: LEVEL_1,
    HelixirLevel.LEVEL_2: LEVEL_2,
    HelixirLevel.LEVEL_3: LEVEL_3,
    HelixirLevel.LEVEL_4: LEVEL_4,
    HelixirLevel.LEVEL_5: LEVEL_5,
}


def get_level_definition(level: HelixirLevel) -> LevelDefinition:
    """Get level definition."""
    return LEVELS[level]


def get_all_levels() -> list[LevelDefinition]:
    """Get all levels in ascending order."""
    return [LEVELS[HelixirLevel(i)] for i in range(6)]


def validate_level_dependencies(target_level: HelixirLevel) -> list[HelixirLevel]:
    """
    Validate dependencies: which levels need to be deployed before target_level.

    Args:
        target_level: Target level for deployment

    Returns:
        List of levels in correct deployment order (without target_level)

    Example:
        >>> validate_level_dependencies(HelixirLevel.LEVEL_3)
        [HelixirLevel.LEVEL_0, HelixirLevel.LEVEL_1, HelixirLevel.LEVEL_2]
    """
    definition = LEVELS[target_level]
    required = set(definition.dependencies)

    for dep in definition.dependencies:
        required.update(LEVELS[dep].dependencies)

    return sorted(required)


def get_deployment_order(max_level: HelixirLevel) -> list[HelixirLevel]:
    """
    Get deployment order of levels from 0 to max_level inclusive.

    Args:
        max_level: Maximum level for deployment

    Returns:
        List of levels in correct deployment order

    Example:
        >>> get_deployment_order(HelixirLevel.LEVEL_4)
        [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4]
    """
    return [HelixirLevel(i) for i in range(max_level + 1)]


def get_accumulated_schema(max_level: HelixirLevel) -> dict[str, list[str]]:
    """
    Get accumulated schema up to max_level inclusive.

    Args:
        max_level: Maximum level

    Returns:
        Dict with keys 'nodes', 'edges', 'extends'

    Example:
        >>> get_accumulated_schema(HelixirLevel.LEVEL_2)
        {
            'nodes': ['User', 'Memory', 'Entity', 'Context'],
            'edges': ['OWNS', 'MENTIONS', 'IN_CONTEXT'],
            'extends': []
        }
    """
    nodes = []
    edges = []
    extends = []

    for level in get_deployment_order(max_level):
        definition = LEVELS[level]
        nodes.extend(definition.schema_nodes)
        edges.extend(definition.schema_edges)
        extends.extend(definition.schema_extends)

    return {"nodes": nodes, "edges": edges, "extends": extends}


def get_accumulated_queries(max_level: HelixirLevel) -> list[str]:
    """
    Get all queries up to max_level inclusive.

    Args:
        max_level: Maximum level

    Returns:
        List of all queries

    Example:
        >>> get_accumulated_queries(HelixirLevel.LEVEL_1)
        ['addUser', 'getUser', 'addMemory', 'getMemory', ...]
    """
    queries = []

    for level in get_deployment_order(max_level):
        definition = LEVELS[level]
        queries.extend(definition.queries)

    return queries


def print_level_info(level: HelixirLevel) -> None:
    """Pretty print level information."""
    definition = LEVELS[level]

    if definition.schema_extends:
        pass
    if definition.notes:
        pass


def print_pyramid() -> None:
    """Print level pyramid."""

    for i in range(5, -1, -1):
        level = HelixirLevel(i)
        LEVELS[level]
        " " * (5 - i) * 2
        if i > 0:
            pass



__all__ = [
    "LEVELS",
    "LEVEL_0",
    "LEVEL_1",
    "LEVEL_2",
    "LEVEL_3",
    "LEVEL_4",
    "LEVEL_5",
    "HelixirLevel",
    "LevelDefinition",
    "get_accumulated_queries",
    "get_accumulated_schema",
    "get_all_levels",
    "get_deployment_order",
    "get_level_definition",
    "print_level_info",
    "print_pyramid",
    "validate_level_dependencies",
]
