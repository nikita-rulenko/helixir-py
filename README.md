<p align="center">
  <img src="helixir_logo.jpeg" alt="Helixir Logo" width="400">
</p>

<h1 align="center">Helixir</h1>

<p align="center">
  <strong>Associative & Causal AI Memory</strong>
</p>

**The fastest AI memory system.** Built on [HelixDB](https://helix-db.com) (1000x faster than Neo4j) with [Cerebras](https://cerebras.ai) inference (70x faster than GPUs). Gives AI assistants **long-term memory with reasoning** — not just storage, but understanding.

**Author:** Nikita Rulenko ([x.com/dengoslav](https://x.com/dengoslav))

## ✨ Features

- **Persistent Memory** - Remember conversations across sessions
- **Hybrid Search** - Vector + Graph + BM25 combined for best results
- **Reasoning Chains** - Build logical connections (IMPLIES, BECAUSE, CONTRADICTS)
- **Cross-User Contradiction Detection** - Auto-detect conflicting memories between users
- **Memory Supersession** - Update memories while preserving full history
- **Concept-Based Retrieval** - Search by skills, preferences, goals
- **MCP Integration** - Works with Cursor, Claude Desktop, and other MCP clients
- **One-Command Setup** - `./setup.sh` handles everything automatically

## 🚀 Quick Start

### Prerequisites

- **Rust 1.88+** - [Install](https://rustup.rs)
- **Docker** - [Install](https://docs.docker.com/get-docker/)
- **uv** - [Install](https://astral.sh/uv) (Python package manager)
- **Ollama** (optional) - [Install](https://ollama.com) for local embeddings

### Installation

```bash
# Clone the repository
git clone https://github.com/helixdb/helixir.git
cd helixir

# Run setup wizard
chmod +x setup.sh
./setup.sh
```

The setup wizard will:
1. ✅ Check dependencies (Rust, Docker, uv)
2. ✅ Install Python 3.14 (via uv)
3. ✅ Create virtual environment
4. ✅ Install Helix CLI
5. ✅ Install Ollama + embedding model (optional)
6. ✅ Configure LLM and embedding providers
7. ✅ Deploy schema to HelixDB
8. ✅ Seed base memories
9. ✅ Add to Cursor MCP config

### Manual Setup

If you prefer manual setup:

```bash
# 1. Install Helix CLI
curl -sSL https://install.helix-db.com | bash

# 2. Create virtual environment
uv venv --python 3.14
source .venv/bin/activate

# 3. Install dependencies
uv sync

# 4. Start HelixDB
helix push dev

# 5. Configure (edit config.yaml)
cp helixir/config.example.yaml config.yaml

# 6. Add to ~/.cursor/mcp.json (see MCP Configuration below)
```

### HelixDB Configuration (helix.toml)

The `helix.toml` file configures HelixDB instances. It's already included in the repo:

```toml
[project]
name = "helixir"
queries = "./schema/"    # Path to .hx files (schema + queries)

[local.dev]
port = 6969              # HelixDB port
build_mode = "debug"     # debug | release | dev
mcp = true               # Enable MCP support
bm25 = true              # Enable BM25 full-text search
```

#### Build Modes

| Mode | Use Case | Debug Symbols | Optimizations |
|------|----------|---------------|---------------|
| `debug` | Development | Yes | None |
| `release` | Production | No | Full |
| `dev` | Dashboard UI | Yes | None |

#### Multiple Instances

```toml
[local.dev]
port = 6969
build_mode = "debug"

[local.production]
port = 6970
build_mode = "release"
```

#### Useful Commands

```bash
helix push dev      # Build and deploy instance
helix stop dev      # Stop instance
helix start dev     # Start stopped instance
helix status        # Show all instances status
helix prune dev     # Clean unused containers/images
```

## 🔧 Configuration

### config.yaml

```yaml
# === LLM Provider (for memory extraction & reasoning) ===
# RECOMMENDED: Cerebras — free, 70x faster than GPUs
llm_provider: "cerebras"
llm_model: "llama-3.3-70b"
llm_api_key: null  # Get free key at https://cloud.cerebras.ai

# === Embedding Provider (for semantic search) ===
# RECOMMENDED: OpenRouter with text-embedding-3-large
embedding_provider: "openai"
embedding_model: "openai/text-embedding-3-large"
embedding_url: "https://openrouter.ai/api/v1"
embedding_api_key: null  # Get key at https://openrouter.ai

# NOTE: For local/offline use, replace with Ollama:
# embedding_provider: "ollama"
# embedding_model: "nomic-embed-text"
# embedding_url: "http://localhost:11434"

# === HelixDB Connection ===
host: "localhost"
port: 6969
instance: "dev"
```

### MCP Configuration

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "helixir": {
      "command": "uv",
      "args": ["run", "python", "-m", "helixir.mcp.server"],
      "cwd": "/path/to/helixir",
      "env": {
        "HELIX_HOST": "localhost",
        "HELIX_PORT": "6969",
        
        "HELIX_LLM_PROVIDER": "cerebras",
        "HELIX_LLM_MODEL": "llama-3.3-70b",
        "HELIX_LLM_API_KEY": "your-cerebras-key",
        
        "HELIX_EMBEDDING_PROVIDER": "openai",
        "HELIX_EMBEDDING_MODEL": "openai/text-embedding-3-large",
        "HELIX_EMBEDDING_URL": "https://openrouter.ai/api/v1",
        "HELIX_EMBEDDING_API_KEY": "your-openrouter-key"
      }
    }
  }
}
```

#### Local Setup (Ollama)

For offline/local use, replace embedding config with Ollama:

```json
{
  "env": {
    "HELIX_EMBEDDING_PROVIDER": "ollama",
    "HELIX_EMBEDDING_MODEL": "nomic-embed-text",
    "HELIX_EMBEDDING_URL": "http://localhost:11434"
  }
}
```

### Cursor Rules (Important!)

To make your AI assistant actually USE the memory, add these rules to **Cursor Settings → Rules**:

```
- Always use Helixir MCP to remember important things about the project
- Always use Helixir MCP first to recall context about the current project
- At the start of chat, store the user's prompt to always remember your role and goals
- After reaching context window limit (when Cursor summarizes), read your role and user goals from memory again
- For memory search, use appropriate mode:
  - "recent" for quick context (last 4 hours)
  - "contextual" for balanced search (30 days)
  - "deep" for thorough search (90 days)
  - "full" for complete history
- Use search_by_concept for skill/preference/goal queries
- Use search_reasoning_chain for "why" questions and logical connections
```

These rules ensure the AI:
1. **Reads memory** at the start of each session
2. **Writes memory** after important changes
3. **Uses memory** to recover context after summarization
4. **Learns** from past mistakes by searching for similar issues

### Environment Variables

All config values can be overridden via environment:

```bash
# HelixDB
export HELIX_HOST="localhost"
export HELIX_PORT="6969"

# LLM (Cerebras recommended)
export HELIX_LLM_PROVIDER="cerebras"
export HELIX_LLM_MODEL="llama-3.3-70b"
export HELIX_LLM_API_KEY="your-cerebras-key"

# Embeddings (OpenRouter recommended)
export HELIX_EMBEDDING_PROVIDER="openai"
export HELIX_EMBEDDING_MODEL="openai/text-embedding-3-large"
export HELIX_EMBEDDING_URL="https://openrouter.ai/api/v1"
export HELIX_EMBEDDING_API_KEY="your-openrouter-key"
```

## 📖 Usage

### MCP Tools (for AI Assistants)

| Tool | Description |
|------|-------------|
| `add_memory` | Store new information |
| `search_memory` | Find memories by semantic query |
| `search_by_concept` | Search by concept type (skill, preference, goal) |
| `search_reasoning_chain` | Find logical connections between memories |
| `update_memory` | Update existing memory |
| `get_memory_graph` | Visualize memory connections |

### Search Modes

| Mode | Time Window | Use Case |
|------|-------------|----------|
| `recent` | 4 hours | Quick context |
| `contextual` | 30 days | Balanced search |
| `deep` | 90 days | Thorough search |
| `full` | All time | Complete history |

### Python API

```python
from helixir import HelixirClient

client = HelixirClient.from_yaml("config.yaml")

# Add memory
result = await client.add(
    content="User prefers dark mode in all applications",
    user_id="user123",
    memory_type="preference",
    context_tags="ui,settings"
)

# Search memories
memories = await client.search(
    query="What are the user's UI preferences?",
    user_id="user123",
    mode="contextual"
)

# Search by concept
skills = await client.search_by_concept(
    query="programming skills",
    user_id="user123",
    concept_type="skill"
)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              MCP Server                     │
│  (add_memory, search_memory, etc.)          │
├─────────────────────────────────────────────┤
│              HelixirClient                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Memory   │ │ Search   │ │ Chain    │    │
│  │ Manager  │ │ Engine   │ │ Builder  │    │
│  └──────────┘ └──────────┘ └──────────┘    │
├─────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ LLM      │ │ Embedder │ │ Chunker  │    │
│  │ Provider │ │          │ │          │    │
│  └──────────┘ └──────────┘ └──────────┘    │
└─────────────────────────────────────────────┘
                    │
            ┌───────▼───────┐
            │    HelixDB    │
            │ (Graph+Vector)│
            └───────────────┘
```

## 🔗 Links

- [HelixDB Documentation](https://docs.helix-db.com)
- [Cerebras Cloud](https://cloud.cerebras.ai) - Free LLM inference (70x faster!)
- [Ollama](https://ollama.com) - Local LLM/embeddings

## 📄 License

**AGPL-3.0** — See [LICENSE](LICENSE)

If you modify and deploy Helixir as a service, you must open-source your codebase.

⚠️ **This is NOT MIT!** If you use Helixir as a SaaS, you must open-source your code.
