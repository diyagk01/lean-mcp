from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .mcp_stdio import McpClientError, McpStdioClient
from .runtime import AgentResponse, SqliteRagAgent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive LeanMCP CLI with live MCP + SQLite RAG routing.",
    )
    parser.add_argument("--mcp-command", required=True, help="MCP server command.")
    parser.add_argument(
        "--mcp-args",
        default="",
        help="Space-separated MCP server args. Example: 'server.py --stdio'",
    )
    parser.add_argument(
        "--db-path",
        default="rag_mcp.db",
        help="SQLite path for tool vector index.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Top-K tool retrieval.")
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=4,
        help="Maximum tool call rounds per user message.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run one query non-interactively and exit.",
    )
    return parser


def _print_help() -> None:
    print("Commands:")
    print("  q, exit      Exit chat")
    print("  clear        Clear conversation history")
    print("  history      Show conversation history")
    print("  tools        Show currently indexed tools")
    print("  sync         Refresh tools from MCP server into SQLite index")
    print("  help         Show this help")


def _render_agent_response(response: AgentResponse) -> str:
    lines = []
    for r in response.rounds:
        lines.append(
            f"[round {r.round_index}] retrieved={r.retrieved_tools} chosen={r.chosen_tool or 'none'}"
        )
        if r.tool_args is not None:
            lines.append(f"  args={json.dumps(r.tool_args)}")
        if r.tool_output is not None:
            preview = r.tool_output[:500]
            lines.append(f"  tool_output={preview}")
        if r.final_answer is not None:
            lines.append("  final_answer_ready=true")
    lines.append("")
    lines.append(response.final_answer)
    return "\n".join(lines)


def _run_chat(agent: SqliteRagAgent) -> None:
    print("LeanMCP Chat CLI")
    print("Type 'help' for commands.")
    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"q", "exit"}:
            print("Goodbye.")
            return
        if lowered == "help":
            _print_help()
            continue
        if lowered == "clear":
            agent.clear_history()
            print("Conversation history cleared.")
            continue
        if lowered == "history":
            messages = agent.history()
            if not messages:
                print("No history.")
            else:
                for idx, msg in enumerate(messages, start=1):
                    print(f"{idx}. {msg['role']}: {msg['text']}")
            continue
        if lowered == "tools":
            tools = agent.list_indexed_tools()
            if not tools:
                print("No tools indexed.")
            else:
                for tool in tools:
                    print(tool)
            continue
        if lowered == "sync":
            report = agent.sync_tools()
            print(
                "Synced tools from MCP: "
                f"total={report.total_tools_from_mcp}, indexed={report.indexed_tools}"
            )
            continue

        response = agent.handle_message(user_input)
        print(_render_agent_response(response))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    mcp_args = [a for a in args.mcp_args.split(" ") if a]

    mcp_client = McpStdioClient(command=args.mcp_command, args=mcp_args)
    agent = SqliteRagAgent(
        mcp_client=mcp_client,
        db_path=args.db_path,
        top_k=args.top_k,
        max_tool_rounds=args.max_tool_rounds,
    )

    try:
        report = agent.boot()
    except McpClientError as exc:
        print(f"Failed to boot MCP runtime: {exc}")
        return 1

    print(
        "Boot complete: "
        f"tools_from_mcp={report.total_tools_from_mcp}, indexed={report.indexed_tools}"
    )

    try:
        if args.query:
            response = agent.handle_message(args.query)
            print(_render_agent_response(response))
            return 0
        _run_chat(agent)
        return 0
    finally:
        mcp_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
