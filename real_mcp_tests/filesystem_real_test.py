from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp_tool_router import Client, SentenceTransformersEmbeddingProvider


@dataclass(frozen=True)
class RealTestQuery:
    query: str
    expected_tool_name: str


def _tool_to_descriptor(tool: Any) -> dict[str, Any]:
    """
    Convert MCP tool object from tools/list into our descriptor JSON format.
    We keep the tool name exactly as provided by the server.
    """
    # mcp tools typically have: name, description, inputSchema
    name = getattr(tool, "name", None) or tool.get("name")  # type: ignore[union-attr]
    description = getattr(tool, "description", None) or tool.get("description", "")  # type: ignore[union-attr]
    input_schema = getattr(tool, "inputSchema", None) or tool.get("inputSchema", {})  # type: ignore[union-attr]

    return {
        "name": str(name),
        "description": str(description or ""),
        "arguments": dict(input_schema) if isinstance(input_schema, dict) else {},
        "version": "1",
        "metadata": {"source": "real_mcp_server"},
    }


async def fetch_tools_from_filesystem_server(*, allowed_dir: str) -> list[Any]:
    # Official server is Node-based; we run it with stdio transport.
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            # result.tools is a list of Tool objects
            return list(result.tools)


def _print_query_block(*, label: str, value: str) -> None:
    print(f"{label}: {value}")


def _short_rerank_explanation(trace: dict[str, Any], *, selected_tool_id: str) -> str:
    """
    Show a compact explanation from rerank trace (if present).
    """
    rerank = trace.get("rerank") or {}
    per_tool = rerank.get("per_tool") or {}
    info = per_tool.get(selected_tool_id)
    if not isinstance(info, dict):
        return "no rerank details"
    contrib = info.get("contrib") or {}
    # pick the top 3 positive contributions besides base_embedding
    items = []
    for k, v in contrib.items():
        if k == "base_embedding":
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            items.append((k, fv))
    items.sort(key=lambda x: x[1], reverse=True)
    top = ", ".join(f"{k}={v:.3f}" for k, v in items[:3]) if items else "no positive features"
    return top


async def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    allowed_dir = str(repo_root)

    print("REAL FILESYSTEM MCP TEST")
    print(f"Repo root (allowed dir): {allowed_dir}")
    print("")

    tools = await fetch_tools_from_filesystem_server(allowed_dir=allowed_dir)
    tool_names = sorted([getattr(t, "name", None) or t.get("name") for t in tools])  # type: ignore[union-attr]
    print(f"Discovered {len(tools)} real tools from filesystem server.")
    print("Tool names:", ", ".join(str(x) for x in tool_names))
    print("")

    # Export descriptors to a temp descriptor root.
    with tempfile.TemporaryDirectory() as td:
        desc_root = Path(td) / "descriptors"
        server_dir = desc_root / "filesystem" / "tools"
        server_dir.mkdir(parents=True, exist_ok=True)
        for t in tools:
            d = _tool_to_descriptor(t)
            (server_dir / f"{d['name']}.json").write_text(
                json.dumps(d, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        # Build router client, sync from exported real descriptors.
        router = Client(
            db_path=str(Path(td) / "router.sqlite3"),
            embedder=SentenceTransformersEmbeddingProvider(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                normalize=True,
            ),
        )
        router.sync_descriptors(root=str(desc_root), delete_stale=True)

        min_confidence = 0.10
        min_margin = 0.02

        tests = [
            RealTestQuery(
                query="Read the file at ./README.md and show me its contents.",
                expected_tool_name="read_file",
            ),
            RealTestQuery(
                query="Replace 'http' with 'https' in ./pyproject.toml.",
                expected_tool_name="edit_file",
            ),
            RealTestQuery(
                query="List everything inside the ./src directory.",
                expected_tool_name="list_directory",
            ),
            RealTestQuery(
                query="Find all Python files under ./src.",
                expected_tool_name="search_files",
            ),
        ]

        print("Routing tests (reranking enabled).")
        print(f"min_confidence={min_confidence} min_margin={min_margin}")
        print("")

        passed = 0
        for t in tests:
            decision, trace = router.route_with_trace(
                t.query,
                server_filter="filesystem",
                top_k=8,
                min_confidence=min_confidence,
                min_margin=min_margin,
            )
            selected = trace.get("best_tool")
            selected_name = (
                selected.get("name") if isinstance(selected, dict) and selected else None
            )
            ok = (selected_name == t.expected_tool_name) and (decision.best_tool is not None)

            print("-" * 80)
            _print_query_block(label="query", value=t.query)
            _print_query_block(label="abstained", value=str(decision.abstained))
            _print_query_block(label="confidence", value=f"{decision.confidence:.4f}")
            _print_query_block(label="margin", value=f"{decision.margin:.4f}")
            _print_query_block(
                label="expected_tool",
                value=t.expected_tool_name,
            )
            _print_query_block(
                label="selected_tool",
                value="None" if not selected else f"{selected.get('name')} (score={selected.get('score'):.4f})",
            )

            orig = trace.get("original_candidates") or []
            new = trace.get("candidates") or []
            _print_query_block(
                label="original_ranking",
                value=", ".join(f"{c['name']}:{c['score']:.3f}" for c in orig[:8]) if orig else "(none)",
            )
            _print_query_block(
                label="reranked_ranking",
                value=", ".join(f"{c['name']}:{c['score']:.3f}" for c in new[:8]) if new else "(none)",
            )
            # Show explanation for the best reranked candidate even if we abstained.
            explain_tool_id = None
            if selected and isinstance(selected, dict) and selected.get("tool_id"):
                explain_tool_id = str(selected.get("tool_id"))
            elif new and isinstance(new, list) and isinstance(new[0], dict) and new[0].get("tool_id"):
                explain_tool_id = str(new[0].get("tool_id"))

            expl = (
                _short_rerank_explanation(trace, selected_tool_id=explain_tool_id)
                if explain_tool_id
                else "n/a"
            )
            _print_query_block(label="rerank_explanation", value=expl)
            _print_query_block(label="matches_expectation", value=str(ok))
            if ok:
                passed += 1

        print("-" * 80)
        print(f"PASS SUMMARY: {passed}/{len(tests)} matched expected tool names.")
        print("")
        print("REALNESS CHECK: REAL (tools were fetched from a real MCP server at runtime).")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

