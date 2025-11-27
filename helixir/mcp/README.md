# Helixir MCP Server - Installation Guide for Cursor

## 🎯 What is This?

**Ultimate Helixir MCP Server** provides 3 powerful capabilities to Cursor AI:

1. **Memory Management** - Advanced AI memory (core feature)
2. **RAG System** - Search HelixDB documentation 
3. **DBA Agent** - Validate/generate HelixQL queries

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd /path/to/helixdb
source .venv/bin/activate
pip install fastmcp
```

### Step 2: Test the Server

```bash
python -m helixir.mcp.server
```

You should see:
```
✅ Ultimate Helixir MCP Server initialized
   📍 HelixDB: localhost:6969
   🤖 LLM: cerebras/llama-3.3-70b
   🛠️  Tools: 6
   📚 Resources: 2
   📝 Prompts: 1
```

### Step 3: Add to Cursor

Edit `~/.cursor/mcp.json`:

```json
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
        "HELIX_LLM_API_KEY": "your-cerebras-api-key"
      }
    }
  }
}
```

### Step 4: Restart Cursor

Close and reopen Cursor. The MCP server will auto-start!

---

## 🛠️ Available Tools

### 📝 Memory Management (Core)

```
add_memory(message, user_id, agent_id?)
  → Add memory with LLM extraction

search_memory(query, user_id, limit=10)
  → Hybrid search (vector + graph + BM25)

get_memory_graph(user_id, memory_id?, depth=2)
  → Get memory graph visualization

update_memory(memory_id, new_content, user_id)
  → Update memory & regenerate embeddings

delete_memory(memory_id)
  → Delete memory
```

### 📚 RAG System

```
search_docs(query, limit=5)
  → Search HelixDB documentation

update_docs(source_url?)
  → Update docs in HelixDB (keeps RAG fresh!)

ask_docs(question)
  → Ask about HelixDB using RAG + LLM
```

### 🤖 DBA Agent

```
validate_schema(schema)
  → Validate HelixQL schema

validate_query(query)
  → Validate HelixQL query

fix_schema(schema, errors?)
  → Auto-fix schema errors

generate_query(description, schema_context?)
  → Generate query from natural language

explain_query(query)
  → Explain what a query does
```

---

## 📖 Resources

```
memories://{user_id}
  → List user's memories

docs://helixdb/syntax
  → HelixQL syntax reference

docs://helixdb/types
  → Valid HelixQL types

config://helixir
  → Current configuration
```

---

## 📝 Prompts

```
memory_summary(user_id, topic?)
  → Template to summarize memories

create_helixql_query(description)
  → Template to create HelixQL query
```

---

## 💡 Usage Examples

### Example 1: Remember Something

```
User: Remember that I love Python programming

Cursor (via MCP):
  → add_memory(
      message="I love Python programming",
      user_id="alice"
    )
  
Result:
  ✅ Added 1 memory (1 entity: Python, 1 relation: LOVES)
```

### Example 2: Search Documentation

```
User: How do I create a node in HelixQL?

Cursor (via MCP):
  → search_docs(query="create node HelixQL syntax")
  
Result:
  📚 Found 3 relevant chunks from HelixDB docs:
  - N::NodeName { field: Type, ... }
  - AddN<NodeType>({ ... })
  - Examples: ...
```

### Example 3: Validate Query

```
User: Is this HelixQL valid?
      QUERY getUser(id: String) =>
        user <- N<User>::WHERE(_::{id}::EQ(id))::FIRST
        RETURN user

Cursor (via MCP):
  → validate_query(query="...")
  
Result:
  ⚠️  1 error: ::FIRST is deprecated
  Suggestion: Use ::LIMIT(1) instead
```

### Example 4: Generate Query

```
User: Generate a query to find all memories from last 7 days

Cursor (via MCP):
  → generate_query(
      description="find all memories from last 7 days"
    )
  
Result:
  QUERY getRecentMemories(days: I64) =>
    memories <- N<Memory>::WHERE(
      _::{created_at}::GT({{now - days*24*3600}})
    )
    RETURN memories
```

---

## 🔧 Troubleshooting

### Server Not Starting?

```bash
# Check logs
python -m helixir.mcp.server 2>&1 | tee mcp.log

# Check HelixDB connection
curl http://localhost:6969/health

# Check Ollama (if using local Ollama)
curl http://localhost:11434/api/tags
```

### Tools Not Working?

Check `~/.cursor/mcp.json` is valid JSON:
```bash
cat ~/.cursor/mcp.json | python -m json.tool
```

### Need More Logs?

Add to `mcp.json`:
```json
{
  "helixir": {
    ...
    "env": {
      ...
      "HELIX_LOG_LEVEL": "DEBUG"
    }
  }
}
```

---

## 🎯 What Makes This Special?

### 1. **Always Up-to-Date RAG**
```
update_docs() → scrapes latest HelixDB docs → stores in HelixDB
              → Cursor always has fresh documentation!
```

### 2. **Smart Auto-Fix**
```
Invalid schema → validate_schema() → fix_schema() → Valid schema
                 Uses gemma3 + RAG from actual docs!
```

### 3. **True Memory**
```
Not just context window!
→ Stores in HelixDB (persistent)
→ Vector + Graph + BM25 (hybrid)
→ Multi-hop reasoning
→ Temporal validity
```

---

## 🚀 Advanced: HTTP Mode

Want to access from multiple clients?

```bash
# Start HTTP server
python -c "
from helixir.mcp.server import run_server
run_server(transport='http', host='0.0.0.0', port=8000)
"
```

Then in Cursor `mcp.json`:
```json
{
  "helixir": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/client", "http://localhost:8000"]
  }
}
```

---

## 📚 Learn More

- [HelixDB Docs](https://docs.helix-db.com)
- [MCP Protocol](https://modelcontextprotocol.io)
- [FastMCP](https://github.com/jlowin/fastmcp)

---

**Made with 🔥 by Helixir Team**

