# Helixir

**Persistent Memory Framework for AI Assistants powered by HelixDB**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 🎯 What is Helixir?

Helixir is a Python framework for building **persistent memory systems** for AI assistants. Unlike simple context windows, Helixir provides:

- 🧠 **Ontological Graph**: Hierarchical concept taxonomy with semantic relationships
- 🔗 **Reasoning Chains**: IMPLIES, BECAUSE, CONTRADICTS edges for logical inference
- ⏰ **Temporal Awareness**: Search by time window (recent, contextual, deep, full)
- 🎯 **Concept-Based Search**: Find memories by skill, preference, goal, fact
- 🔍 **Hybrid Search**: Vector + Graph + BM25 in a single query
- ⚡ **Single Database**: HelixDB handles graph, vector, and full-text search

## 🚀 Quick Start

See the main [README](../README.md) for installation instructions.

### Basic Usage

```python
from helixir import HelixirClient

# Initialize from config
client = HelixirClient.from_yaml("config.yaml")

# Add a memory
result = await client.add(
    content="User prefers Python over JavaScript",
    memory_type="preference",
    user_id="alice",
    context_tags="programming,languages"
)

# Search memories
results = await client.search(
    query="What programming languages does Alice like?",
    user_id="alice",
    mode="contextual"  # recent, contextual, deep, full
)

# Search by concept type
skills = await client.search_by_concept(
    query="programming",
    user_id="alice",
    concept_type="skill"
)

# Get reasoning chain
chain = await client.search_reasoning_chain(
    query="Why does user prefer Python?",
    user_id="alice",
    chain_mode="causal"  # causal, forward, both
)
```

## 🏗️ Architecture

```
helixir/
├── core/               # Client, config, exceptions
│   ├── helixir_client.py   # Main entry point
│   └── config.py           # Configuration management
├── llm/                # LLM providers
│   ├── factory.py          # Provider factory (singleton)
│   ├── embeddings.py       # Embedding generation
│   └── providers/          # Cerebras, Ollama, OpenAI
├── toolkit/
│   └── mind_toolbox/
│       ├── memory/         # Memory CRUD
│       ├── search/         # Search strategies
│       ├── memory_chain/   # Reasoning chains
│       └── ontology/       # Concept management
├── mcp/                # MCP server for Cursor/Claude
│   └── server.py
└── setup/              # Installation wizard
    ├── wizard.py
    ├── deploy_schema.py
    └── seed_memories.py
```

## 🔧 Configuration Priority

```
ENV (mcp.json) > YAML (config.yaml) > Defaults
```

Environment variables override YAML, which overrides defaults.

## 🛠️ Development

```bash
# Create virtual environment
uv venv --python 3.14
source .venv/bin/activate

# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## 📄 License

**AGPL-3.0** (with Commercial License option)

See [LICENSE.txt](../LICENSE.txt) for details. This is NOT MIT - if you deploy as SaaS, you must open-source your code or get a commercial license.
