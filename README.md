# lean-mcp
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

LeanMCP is a local MCP chat runtime that turns **tool selection into retrieval** using a SQLite-backed vector index.

No Bedrock dependency is required for the retrieval layer.

## End-to-End Flow (one user message)

### 0) System boot (once)

1. Start MCP server (stdio).
2. Connect through MCP client.
3. Call `tools/list`.
4. Convert MCP tools into internal tool objects.
5. Index tools in SQLite vector store.

Result: you have a searchable tool index.

### 1) User sends message

Example:

```text
show me files in this directory
```

### 2) Retrieve relevant tools

Run vector retrieval over tool metadata in SQLite and get top-K relevant tools.

Example output:

```text
[list_directory, read_file]
```

### 3) Build model/tool context

Send chat state + user input + only retrieved tools to the decision step.

### 4) Decide action

Either:

- answer directly, or
- emit tool call (name + args)

### 5) Execute tool via MCP

Forward the chosen tool call to MCP `tools/call`.

### 6) Feed tool result back

Append tool output back into the working context.

### 7) Generate final answer

Return user-facing response.

### 8) Multi-step loop (if needed)

Repeat retrieval -> decision -> tool execution until completion or max rounds.

## What is different from traditional MCP

- Traditional: LLM sees all tools every turn.
- LeanMCP: runtime injects only top-K retrieved tools every turn.

Everything else (tool execution loop via MCP) stays standard.

## Download and run

### 1) Clone

```bash
git clone https://github.com/diyagk01/lean-mcp
cd lean-mcp
```

### 2) Install

```bash
pip install -e .
```

Optional local ONNX embeddings:

```bash
pip install -e ".[fastembed]"
```

### 3) Start chat CLI

You need an MCP server command:

```bash
lean-mcp-chat --mcp-command python --mcp-args "path/to/your_mcp_server.py --stdio" --db-path rag_mcp.db
```

Run one query and exit:

```bash
lean-mcp-chat --mcp-command python --mcp-args "path/to/your_mcp_server.py --stdio" --query "show me files in this directory"
```

## CLI commands

- `q`, `exit`: quit
- `help`: show help
- `sync`: re-list tools from MCP and refresh SQLite index
- `tools`: list indexed tools
- `history`: show conversation history
- `clear`: clear conversation history

## Core modules

- `rag_mcp/mcp_stdio.py`: stdio MCP client (`initialize`, `tools/list`, `tools/call`)
- `rag_mcp/runtime.py`: boot + per-message retrieval/decision/tool loop
- `rag_mcp/chat_cli.py`: interactive chat interface
