"""
Helixir MCP Server - Memory & Knowledge Management for AI

Provides instant memory operations for Claude and other AI assistants:
- Add memories with automatic LLM extraction & graph markup
- Search memories with hybrid vector+graph+BM25
- Update/delete memories with proper cleanup
- Query project knowledge and documentation

Architecture:
    Claude/Cursor (MCP Client)
        ↓ MCP Protocol (STDIO)
    Helixir MCP Server
        ↓
    HelixirClient (async context)
        ↓
    HelixDB (graph-vector database)

Usage in Cursor ~/.cursor/mcp.json:
    {
      "mcpServers": {
        "helixir": {
          "command": "uv",
          "args": ["run", "python", "-m", "helixir.mcp.server"],
          "cwd": "/path/to/helixdb",
          "env": {
            "HELIX_HOST": "localhost",
            "HELIX_PORT": "6969",
            "HELIX_INSTANCE": "dev",
            "HELIX_LLM_PROVIDER": "cerebras",
            "HELIX_LLM_MODEL": "llama-3.3-70b",
            "HELIX_LLM_API_KEY": "your-api-key-here"
          }
        }
      }
    }

For Ollama (local):
    "env": {
      "HELIX_LLM_PROVIDER": "ollama",
      "HELIX_LLM_BASE_URL": "http://localhost:11434",
      "HELIX_LLM_MODEL": "llama3.1:8b"
    }
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import sys
from typing import Any

from fastmcp import Context, FastMCP

from helixir.core.config import HelixMemoryConfig
from helixir.core.helixir_client import HelixirClient

logger = logging.getLogger(__name__)


_client: HelixirClient | None = None


def normalize_param(value: str | list[str]) -> str:
    """
    Normalize MCP parameters that may arrive as lists.

    Some MCP clients send string params as single-element lists.
    """
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def get_client() -> HelixirClient:
    """Get initialized client or raise error."""
    if _client is None:
        msg = "HelixirClient not initialized. Server startup failed."
        raise RuntimeError(msg)
    return _client


@asynccontextmanager
async def lifespan(app: FastMCP):
    """
    Manage server lifecycle with proper async initialization and cleanup.

    This ensures HelixirClient is properly initialized in async context
    and cleaned up on server shutdown.
    """
    global _client

    logger.info("🚀 Initializing Helixir MCP Server...")

    try:
        config = HelixMemoryConfig()

        _client = HelixirClient(config)

        logger.info("✅ Helixir MCP Server ready")
        logger.info(f"   📍 HelixDB: {config.host}:{config.port}")
        logger.info(f"   🤖 LLM: {config.llm_provider}/{config.llm_model}")
        logger.info(f"   📊 Instance: {config.instance}")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}", exc_info=True)
        raise

    finally:
        if _client:
            logger.info("🔄 Shutting down Helixir MCP Server...")
            try:
                await _client.close()
                logger.info("✅ Server shutdown complete")
            except Exception as e:
                logger.error(f"⚠️  Error during shutdown: {e}", exc_info=True)

        _client = None


mcp = FastMCP(
    name="helixir",
    version="2.0.0",
    lifespan=lifespan,
)


@mcp.tool()
async def add_memory(
    message: str | list[str],
    user_id: str | list[str],
    agent_id: str | list[str] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Add memory with LLM-powered extraction.

    Extracts atomic facts, generates embeddings, creates graph relations.
    This is the CORE functionality - instant memory for AI.

    Args:
        message: Text to remember (will be extracted into atomic facts)
        user_id: User identifier (e.g., "claude", "developer")
        agent_id: Optional agent identifier

    Returns:
        {
            "memories_added": int,
            "entities": int,
            "relations": int,
            "memory_ids": list[str]
        }

    Example:
        await add_memory(
            message="We just implemented ConceptMapper for ontology linking",
            user_id="developer"
        )
    """
    message = normalize_param(message)
    user_id = normalize_param(user_id)
    if agent_id:
        agent_id = normalize_param(agent_id)

    if ctx:
        await ctx.info(f"🧠 Adding memory for user={user_id}")

    try:
        client = get_client()
        result = await client.add(message=message, user_id=user_id, agent_id=agent_id)

        if ctx:
            await ctx.info(
                f"✅ Added {result.get('memories_added', 0)} memories "
                f"({result.get('entities', 0)} entities, {result.get('relations', 0)} relations)"
            )

        return result

    except Exception as e:
        logger.error(f"Failed to add memory: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"❌ Failed to add memory: {e}")
        raise


@mcp.tool()
async def search_memory(
    query: str | list[str],
    user_id: str | list[str],
    limit: int | None = None,
    mode: str = "recent",
    temporal_days: float | None = None,
    graph_depth: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """
    🔍 Unified smart memory search with automatic strategy selection.

    **Automatically uses best search strategy based on mode:**
    - RECENT/CONTEXTUAL/DEEP: SmartTraversalV2 (vector-first + graph expansion)
    - FULL: Hybrid search (vector + BM25) for comprehensive results

    **Search Modes:**
    - **recent** (default): Fast recent memories (4 hours) + nearest graph (⚡ low cost)
    - **contextual**: Balanced search (30 days) + moderate graph (⚡⚡ medium cost)
    - **deep**: Deep search (90 days) + extensive graph (⚡⚡⚡ high cost)
    - **full**: Complete history + full graph (⚡⚡⚡⚡ very high cost)

    Args:
        query: Search query
        user_id: User identifier
        limit: Max results (None = use mode default)
        mode: Search mode (recent/contextual/deep/full)
        temporal_days: Override time window in days (None = use mode default)
        graph_depth: Override graph depth (None = use mode default)

    Returns:
        List of memories with graph context and relevance scores

    Examples:
        results = await search_memory(
            query="What did we implement today?",
            user_id="developer"
        )

        results = await search_memory(
            query="How does caching work?",
            user_id="developer",
            mode="contextual"
        )

        results = await search_memory(
            query="All memory-related features",
            user_id="developer",
            mode="deep",
            temporal_days=180
        )
    """
    from helixir.core.search_modes import SearchMode, estimate_token_cost

    query = normalize_param(query)
    user_id = normalize_param(user_id)

    search_mode = SearchMode.from_string(mode)
    mode_defaults = search_mode.get_defaults()

    if limit is None or limit <= 0:
        limit = mode_defaults["max_results"]
    if temporal_days is None:
        temporal_days = mode_defaults["temporal_days"]
    if graph_depth is None:
        graph_depth = mode_defaults["graph_depth"]

    use_smart = mode_defaults.get("use_smart_traversal", False)
    strategy = "smart_v2" if use_smart else "hybrid"

    cost_info = estimate_token_cost(search_mode, limit, graph_depth)

    warning = search_mode.get_cost_warning()
    if warning and ctx:
        await ctx.info(warning)

    if ctx:
        await ctx.info(
            f"🔍 Searching: '{query[:50]}...' "
            f"[mode={search_mode.value}, strategy={strategy}, depth={graph_depth}, "
            f"days={temporal_days or 'all'}, cost_tier={cost_info['cost_tier']}]"
        )

    try:
        client = get_client()
        results = await client.search(
            query=query,
            user_id=user_id,
            limit=limit,
            search_mode=search_mode.value,
            temporal_days=temporal_days,
            graph_depth=graph_depth,
        )

        if ctx:
            await ctx.info(
                f"✅ Found {len(results)} memories (strategy={strategy}, "
                f"~{cost_info['total_cost']} tokens, tier={cost_info['cost_tier']})"
            )

        return results

    except Exception as e:
        logger.exception("Search failed: %s", e)
        if ctx:
            await ctx.error(f"❌ Search failed: {e}")
        raise


@mcp.tool()
async def get_memory_graph(
    user_id: str | list[str],
    memory_id: str | list[str] | None = None,
    depth: int = 2,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Get memory graph visualization.

    Args:
        user_id: User identifier
        memory_id: Optional starting point
        depth: Traversal depth (default: 2)

    Returns:
        {"nodes": [...], "edges": [...]}
    """
    user_id = normalize_param(user_id)
    if memory_id:
        memory_id = normalize_param(memory_id)

    if ctx:
        await ctx.info(f"🕸️  Building graph (depth={depth})")

    try:
        client = get_client()
        graph = await client.get_graph(user_id=user_id, memory_id=memory_id, depth=depth)

        if ctx:
            await ctx.info(
                f"✅ Graph: {len(graph.get('nodes', []))} nodes, "
                f"{len(graph.get('edges', []))} edges"
            )

        return graph

    except Exception as e:
        logger.error(f"Failed to get graph: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"❌ Graph retrieval failed: {e}")
        raise


@mcp.tool()
async def update_memory(
    memory_id: str | list[str],
    new_content: str | list[str],
    user_id: str | list[str],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Update memory content (regenerates embedding & relations).

    Args:
        memory_id: Memory ID to update
        new_content: New content
        user_id: User identifier

    Returns:
        Update result
    """
    memory_id = normalize_param(memory_id)
    new_content = normalize_param(new_content)
    user_id = normalize_param(user_id)

    if ctx:
        await ctx.info(f"✏️  Updating memory: {memory_id[:8]}...")

    try:
        client = get_client()
        result = await client.update(memory_id=memory_id, new_content=new_content, user_id=user_id)

        if ctx:
            await ctx.info("✅ Memory updated")

        return result

    except Exception as e:
        logger.error(f"Failed to update memory: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"❌ Update failed: {e}")
        raise


@mcp.tool()
async def search_by_concept(
    query: str | list[str],
    user_id: str | list[str],
    concept_type: str | None = None,
    tags: str | list[str] | None = None,
    mode: str = "contextual",
    limit: int = 10,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """
    🎯 Search memories by ontology concepts and tags.

    Best for:
    - "What are my skills?" → concept_type="skill"
    - "What do I prefer?" → concept_type="preference"
    - "What are my goals?" → concept_type="goal"
    - Search by tags: tags="python,backend"

    **Concept Types:**
    - preference: Things user likes/dislikes
    - skill: Abilities and expertise
    - goal: Objectives and plans
    - fact: Known information
    - opinion: Views and judgments
    - experience: Past events
    - achievement: Accomplishments

    **Search Modes:**
    - recent: Last 4 hours, fast
    - contextual: Last 30 days, balanced (default)
    - deep: Last 90 days, thorough
    - full: All time, comprehensive

    Args:
        query: Search query (semantic matching)
        user_id: User identifier
        concept_type: Filter by concept (skill/preference/goal/fact/opinion)
        tags: Comma-separated tags to filter by
        mode: Search mode (recent/contextual/deep/full)
        limit: Max results

    Returns:
        List of memories with concept and tag scores

    Examples:
        await search_by_concept(
            query="What skills do I have?",
            user_id="developer",
            concept_type="skill"
        )

        await search_by_concept(
            query="programming preferences",
            user_id="developer",
            concept_type="preference",
            tags="python,rust"
        )
    """
    from helixir.toolkit.mind_toolbox.search import UnifiedSearchEngine

    query = normalize_param(query)
    user_id = normalize_param(user_id)
    if tags and isinstance(tags, list):
        tags = ",".join(tags)

    if ctx:
        concept_info = f", concept={concept_type}" if concept_type else ""
        tags_info = f", tags={tags}" if tags else ""
        await ctx.info(f"🎯 OntoSearch: '{query[:40]}...'{concept_info}{tags_info}")

    try:
        client = get_client()

        query_embedding = await client.embedder.generate(query)

        engine = UnifiedSearchEngine(
            client=client.db,
            ontology_manager=client.tooling.ontology_manager,
        )

        search_kwargs: dict[str, Any] = {
            "mode": mode,
        }
        if concept_type:
            search_kwargs["concept_filter"] = concept_type.capitalize()
        if tags:
            search_kwargs["tag_filter"] = [t.strip() for t in tags.split(",")]

        results = await engine.search(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            strategy="onto",
            **search_kwargs,
        )

        if ctx:
            await ctx.info(f"✅ Found {len(results)} memories via OntoSearch")

        return results

    except Exception as e:
        logger.error(f"OntoSearch failed: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"❌ OntoSearch failed: {e}")
        raise


@mcp.tool()
async def search_reasoning_chain(
    query: str | list[str],
    user_id: str | list[str],
    chain_mode: str = "both",
    max_depth: int = 5,
    limit: int = 3,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    🔗 Search with logical reasoning chains.

    Builds chains of memories connected by IMPLIES/BECAUSE/CONTRADICTS relations.
    Returns not just memories, but their logical connections - the "story" of knowledge.

    Best for:
    - "Why do I think X?" → chain_mode="causal" (follow BECAUSE)
    - "What follows from X?" → chain_mode="forward" (follow IMPLIES)
    - "Give me full context" → chain_mode="both"

    **Chain Modes:**
    - causal: Follow BECAUSE relations backward (find causes)
    - forward: Follow IMPLIES relations forward (find effects)
    - both: Follow all logical relations (default)
    - deep: Maximum depth traversal, all relation types

    Args:
        query: Search query
        user_id: User identifier
        chain_mode: How to traverse (causal/forward/both/deep)
        max_depth: Maximum chain depth (default: 5)
        limit: Number of seed memories to start chains from

    Returns:
        {
            "query": str,
            "chains": [
                {
                    "seed": {...},
                    "nodes": [...],
                    "chain_type": "causal|implication|mixed",
                    "reasoning_trail": "formatted string"
                }
            ],
            "total_memories": int,
            "deepest_chain": int
        }

    Examples:
        await search_reasoning_chain(
            query="Why do I use Rust?",
            user_id="developer",
            chain_mode="causal"
        )

        await search_reasoning_chain(
            query="Python programming",
            user_id="developer",
            chain_mode="forward"
        )
    """
    from helixir.toolkit.mind_toolbox.memory_chain import MemoryChainConfig, MemoryChainStrategy

    query = normalize_param(query)
    user_id = normalize_param(user_id)

    if ctx:
        await ctx.info(f"🔗 ChainSearch: '{query[:40]}...' [mode={chain_mode}, depth={max_depth}]")

    try:
        client = get_client()

        if chain_mode == "causal":
            config = MemoryChainConfig.causal_only()
        elif chain_mode == "forward":
            config = MemoryChainConfig.implications_only()
        elif chain_mode == "deep":
            config = MemoryChainConfig.deep_context()
        else:
            config = MemoryChainConfig.default()

        config.max_depth = max_depth

        strategy = MemoryChainStrategy(
            client=client.db,
            embedder=client.embedder,
            config=config,
        )

        result = await strategy.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        response = {
            "query": result.query,
            "total_chains": result.total_chains,
            "total_memories": result.total_memories,
            "deepest_chain": result.deepest_chain,
            "chains": [],
        }

        for chain in result.chains:
            chain_data = {
                "seed": {
                    "memory_id": chain.seed.memory_id,
                    "content": chain.seed.content,
                    "memory_type": chain.seed.memory_type,
                },
                "chain_type": chain.chain_type,
                "depth": chain.total_depth,
                "implies_count": chain.implies_count,
                "because_count": chain.because_count,
                "nodes": [
                    {
                        "memory_id": n.memory_id,
                        "content": n.content,
                        "relation": n.relation_type.value if n.relation_type else None,
                        "direction": n.relation_direction,
                        "depth": n.depth,
                    }
                    for n in chain.nodes
                ],
                "reasoning_trail": chain.get_reasoning_trail(),
            }
            response["chains"].append(chain_data)

        if ctx:
            await ctx.info(
                f"✅ Found {result.total_chains} chains, "
                f"{result.total_memories} memories, max depth {result.deepest_chain}"
            )

        return response

    except Exception as e:
        logger.error(f"ChainSearch failed: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"❌ ChainSearch failed: {e}")
        raise


@mcp.resource("config://helixir")
async def get_config() -> dict[str, Any]:
    """Get Helixir configuration."""
    client = get_client()
    config = client.config

    return {
        "version": "2.0.0",
        "helixdb": {
            "host": config.host,
            "port": config.port,
            "instance": config.instance,
        },
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "base_url": config.llm_base_url,
        },
        "capabilities": {
            "memory_management": True,
            "vector_search": True,
            "graph_traversal": True,
            "llm_extraction": True,
            "entity_linking": True,
            "ontology_mapping": True,
            "onto_search": True,
            "reasoning_chains": True,
        },
        "tools": [
            "add_memory",
            "search_memory",
            "search_by_concept",
            "search_reasoning_chain",
            "get_memory_graph",
            "update_memory",
        ],
    }


@mcp.resource("status://helixdb")
async def get_status() -> dict[str, Any]:
    """Get HelixDB connection status."""
    try:
        client = get_client()
        return {
            "status": "connected",
            "host": client.config.host,
            "port": client.config.port,
            "instance": client.config.instance,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.prompt()
def memory_summary(user_id: str, topic: str | None = None) -> str:
    """Generate prompt to summarize user's memories."""
    topic_filter = f" about {topic}" if topic else ""
    return f"""
Analyze memories for user_id={user_id}{topic_filter}.

Use search_memory tool to find relevant memories.
Provide a summary with:
1. Key patterns and themes
2. Important facts and preferences
3. Connections between memories
4. Timeline of events
"""


@mcp.prompt()
def tool_selection_guide() -> str:
    """
    Guide for AI to select the right memory tool.

    This prompt should be included in AI's system context
    to help it choose the appropriate tool for each query.
    """
    return """

You have access to powerful memory tools. Choose the RIGHT tool for each task:

**When to use:** Storing new information, facts, decisions, events
**Examples:**
- "Remember that we decided to use Rust for performance"
- "Save this: the API endpoint is /v1/memories"
- After learning something important from the user

**When to use:** General queries, finding relevant context, quick lookups
**Best for:**
- "What do I know about X?"
- "Find information about Y"
- "What did we discuss regarding Z?"
**Modes:**
- recent: Quick search in last 4 hours (default, fastest)
- contextual: Balanced search, last 30 days
- deep: Thorough search, last 90 days
- full: Complete history search

**When to use:** Searching by TYPE of memory or specific tags
**Best for:**
- "What are my skills?" → concept_type="skill"
- "What do I prefer/like?" → concept_type="preference"
- "What are my goals?" → concept_type="goal"
- "Find memories tagged with 'python'" → tags="python"
**Concept types:** skill, preference, goal, fact, opinion, experience, achievement

**When to use:** Understanding WHY or tracing logical connections
**Best for:**
- "Why do I think/know X?" → chain_mode="causal"
- "What follows from X?" → chain_mode="forward"
- "Give me the full reasoning behind X" → chain_mode="both"
- Understanding cause-effect relationships
**Returns:** Chains of connected memories with IMPLIES/BECAUSE relations

**When to use:** Visualizing connections, exploring memory structure
**Best for:**
- "Show me how memories are connected"
- "What's related to memory X?"
- Debugging or understanding memory topology

**When to use:** Correcting outdated information

**Status:** Not implemented yet - complex cascade deletion needed

---


1. **Storing info?** → add_memory
2. **General "what do I know"?** → search_memory
3. **Asking about skills/preferences/goals?** → search_by_concept
4. **Asking "why" or "what follows"?** → search_reasoning_chain
5. **Want to see connections?** → get_memory_graph


- Start with search_memory for general queries
- Use search_by_concept when user asks about specific types (skills, preferences)
- Use search_reasoning_chain for "why" questions or to explain reasoning
- Chain multiple tools: search first, then add_memory to save insights
- Always specify user_id for personalized results
"""


def run_server() -> None:
    """Run Helixir MCP server via STDIO (for Cursor/Claude Desktop)."""
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    run_server()
