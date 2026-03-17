# lean-mcp
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green) 

RAG-first MCP tool routing SDK that reduces prompt token cost and model reasoning overhead by selecting the best tool before the LLM call.

## What this implements

- Retrieval over MCP tool metadata with pluggable embeddings
- Optional validation stage for ambiguous retrieval cases
- Top-1 context assembly for MCP tool invocation
- Installable plugin API for MCP servers:
  - `register_tools()`
  - `route_query()`
  - `assemble_for_llm()`
- Persistent local embedding store via SQLite

## Install

```bash
pip install -e .
```

Optional local ONNX embeddings:

```bash
pip install -e ".[fastembed]"
```

## Quickstart (client)

```python
from rag_mcp import McpTool, RagMcpClient

client = RagMcpClient(top_k=5, inject_top_n=1, db_path="rag_mcp.db")
client.add_tools(
    [
        McpTool(
            name="web_search",
            description="Search the web for fresh information.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server="search",
        )
    ]
)

context = client.build_context("search latest AI news")
print(context.tools)  # pass this to your LLM API tools argument
```

## Quickstart (plugin)

```python
from rag_mcp import McpTool, RagMcpPlugin

plugin = RagMcpPlugin(
    server_name="my_server",
    db_path="rag_mcp.db",
    confidence_threshold=0.15,
    margin_threshold=0.05,
)

plugin.register_tools([...])  # your MCP tools
decision = plugin.route_query("find weather in sf")

if decision.best_tool is None:
    # fallback strategy: ask clarification or use broader tool set
    ...
else:
    context = plugin.assemble_for_llm("find weather in sf", routing=decision)
```

## Design defaults

- Uses top-1 injection to keep context focused
- Stores vectors with `tool_hash` to support incremental re-indexing
- Embedding backend is pluggable via `EmbeddingProvider`
- `FastEmbedProvider` falls back to deterministic hash embeddings if dependency is missing
