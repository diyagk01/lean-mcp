# mcp-tool-router

This SDK **routes tools and assembles LLM tool-calling context** from MCP-style tool/resource catalogs.

It **does not**:

- run MCP servers
- execute tools
- call any LLM API
- parse tool calls from model output
- orchestrate an agent loop

It stops at: **(selected tool(s) + compact LLM tool payload + system context)**.

## What it does

- **Indexes tools** (name/description/schema/server/version/metadata) into a local SQLite store with stable IDs and content hashes (incremental/idempotent re-index).
- **Retrieves and routes** tools for a user query using embedding similarity and configurable abstention thresholds.
- **Assembles minimal LLM payload**:
  - `tools`: only the top-N tool specs (name/description/input schema) for your downstream LLM tool/function parameter
  - `system_context`: compact system prompt context including the user query and optionally relevant resource chunks
  - `token_estimate`: rough estimate of prompt size
- **(Optional)** Syncs conventional descriptor layouts:
  - Tools: `<root>/<server>/tools/*.json`
  - Resources: `<root>/<server>/resources/*.json`
- **(Optional)** Indexes resource content in chunks and retrieves top-k relevant chunks for context.

## Quick start

```python
from mcp_tool_router import Client, RouterConfig, SentenceTransformersEmbeddingProvider

client = Client(
    db_path="tool_router.sqlite3",
    # Optional (recommended): better semantic embeddings.
    # Install: pip install -e ".[embeddings]"
    embedder=SentenceTransformersEmbeddingProvider(model_name="all-MiniLM-L6-v2"),
)

# Optional: sync from descriptor layout on disk
client.sync_descriptors(root="./descriptors", delete_stale=True)

decision = client.route(
    "Summarize the latest issues in my repo and create a triage list",
    top_k=8,
)

payload = client.build_llm_context(
    user_query="Summarize the latest issues in my repo and create a triage list",
    decision=decision,
    tool_top_n=1,
    resource_top_k=4,
)

print(decision.best_tool)
print([t["name"] for t in payload["tools"]])
print(payload["system_context"])
```

## Recommended production usage

In production, you will typically:

1. **Choose an embedder**
   - For local/dev or offline, the default `HashingEmbeddingProvider` is fine.
   - For higher accuracy, use `SentenceTransformersEmbeddingProvider` with a CPU-friendly model
     (e.g. `"all-MiniLM-L6-v2"`).
2. **Sync descriptors at startup** (or whenever your catalog changes):

   ```python
   client.sync_descriptors(root="./descriptors", delete_stale=True)
   ```

3. **Route each user query** (before calling your LLM):

   ```python
   decision, trace = client.route_with_trace(
       "Summarize issues for the repo",
       top_k=8,
       server_filter="github",          # optional but recommended
       min_confidence=0.10,             # tune per catalog
       min_margin=0.02,
   )

   if decision.best_tool is None:
       # router abstained; either ask the user for clarification,
       # or fall back to a broader tool set.
       ...
   else:
       payload = client.build_llm_context(
           user_query="Summarize issues for the repo",
           decision=decision,
           tool_top_n=1,
           resource_top_k=4,
       )
       # Pass payload["tools"] and payload["system_context"] to your LLM.
   ```

4. **Log or inspect `trace`** in production when needed:
   - `trace["reason"]` tells you **why** the router abstained or selected.
   - `trace["candidates"]` shows the top tools and scores.

## Recommended benchmark workflow

1. **Create descriptors** under `./descriptors/<server>/tools/*.json` (and `resources/*.json` if needed).
2. **Create a label file** (JSON or JSONL) with:
   - `query`
   - `expected_tool_id` (e.g. `"github:search_issues:1"`) or `expected_tool_name` + `server`
   - optional `expected_abstain` for queries where "no tool" is the correct choice.
3. **Run the real-catalog benchmark**:

   ```bash
   python benchmarks/run_accuracy.py \
     --descriptor-root ./descriptors \
     --labels benchmarks/labels.filesystem.example.jsonl \
     --embedder st \
     --model-name all-MiniLM-L6-v2 \
     --server filesystem \
     --top-k 8 \
     --sweep-thresholds
   ```

4. **Pick thresholds** from the table (e.g. `min_confidence=0.10`, `min_margin=0.02` for a balanced router).
5. **Re-run with `--debug`** on the same labels to inspect any remaining failures.

## Public API

- `Client`: high-level entry point for sync, routing, and context assembly.
- `RouterConfig`: thresholds and defaults (top-k, abstain margin/confidence, chunk sizes).
- `EmbeddingProvider`: pluggable embedding interface (default is deterministic local hashing; no network calls).
- `SQLiteStore`: persists tools/resources/embeddings and supports similarity queries.
- `types`: structured tool/resource/routing types.

## Notes

- The default embedding is **deterministic and local** to keep the SDK lightweight and testable. You can swap in a stronger embedding provider by implementing `EmbeddingProvider`.
- Similarity search is done in Python over stored embeddings (sufficient for small/medium catalogs). The storage format supports future optimization.
