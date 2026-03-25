from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .client import RagMcpClient
from .mcp_stdio import McpCallResult, McpStdioClient
from .types import McpTool


@dataclass(frozen=True)
class BootReport:
    total_tools_from_mcp: int
    indexed_tools: int


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolRound:
    round_index: int
    retrieved_tools: List[str]
    chosen_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_output: Optional[str]
    final_answer: Optional[str]


@dataclass(frozen=True)
class AgentResponse:
    final_answer: str
    rounds: List[ToolRound]


class SqliteRagAgent:
    """
    End-to-end runtime:
    0) Boot: list tools from live MCP and index in SQLite
    1+) Per message: retrieve top-k tools, decide, call tool (if needed), loop.
    """

    def __init__(
        self,
        *,
        mcp_client: McpStdioClient,
        db_path: str = "rag_mcp.db",
        top_k: int = 2,
        max_tool_rounds: int = 4,
    ) -> None:
        self.mcp_client = mcp_client
        self.rag_client = RagMcpClient(top_k=top_k, inject_top_n=top_k, db_path=db_path)
        self.max_tool_rounds = max(1, max_tool_rounds)
        self._tool_registry: Dict[str, McpTool] = {}
        self._history: List[Dict[str, str]] = []

    def boot(self) -> BootReport:
        self.mcp_client.start()
        tools = self.mcp_client.list_tools()
        self.rag_client.add_tools(tools)
        self._tool_registry = {tool.name: tool for tool in tools}
        return BootReport(total_tools_from_mcp=len(tools), indexed_tools=len(tools))

    def sync_tools(self) -> BootReport:
        tools = self.mcp_client.list_tools()
        self.rag_client.add_tools(tools)
        self._tool_registry = {tool.name: tool for tool in tools}
        return BootReport(total_tools_from_mcp=len(tools), indexed_tools=len(tools))

    def list_indexed_tools(self) -> List[str]:
        return sorted(self._tool_registry.keys())

    def clear_history(self) -> None:
        self._history.clear()

    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def handle_message(self, user_query: str) -> AgentResponse:
        rounds: List[ToolRound] = []
        working_query = user_query
        last_call_result: Optional[McpCallResult] = None

        for idx in range(1, self.max_tool_rounds + 1):
            context = self.rag_client.build_context(working_query)
            retrieved_tool_names = [t.tool.name for t in context.retrieved_tools]
            selected_tools = [r.tool for r in context.retrieved_tools[:2]]

            # Decide whether to call a tool or answer now.
            maybe_call = _decide_tool_call(
                query=user_query,
                candidates=selected_tools,
                last_tool_result=last_call_result,
            )
            if maybe_call is None:
                answer = _compose_final_answer(user_query, last_call_result, context.system_context)
                rounds.append(
                    ToolRound(
                        round_index=idx,
                        retrieved_tools=retrieved_tool_names,
                        chosen_tool=None,
                        tool_args=None,
                        tool_output=None,
                        final_answer=answer,
                    )
                )
                self._history.append({"role": "user", "text": user_query})
                self._history.append({"role": "assistant", "text": answer})
                return AgentResponse(final_answer=answer, rounds=rounds)

            call_result = self.mcp_client.call_tool(maybe_call.name, maybe_call.arguments)
            rounds.append(
                ToolRound(
                    round_index=idx,
                    retrieved_tools=retrieved_tool_names,
                    chosen_tool=maybe_call.name,
                    tool_args=maybe_call.arguments,
                    tool_output=call_result.text,
                    final_answer=None,
                )
            )
            last_call_result = call_result
            working_query = (
                f"{user_query}\n\nPrevious tool: {maybe_call.name}\nTool output:\n{call_result.text}"
            )

        fallback = _compose_final_answer(user_query, last_call_result, "")
        self._history.append({"role": "user", "text": user_query})
        self._history.append({"role": "assistant", "text": fallback})
        return AgentResponse(final_answer=fallback, rounds=rounds)


def _decide_tool_call(
    *,
    query: str,
    candidates: Sequence[McpTool],
    last_tool_result: Optional[McpCallResult],
) -> Optional[ToolCall]:
    # If we already have one tool result, finalize instead of endlessly chaining.
    if last_tool_result is not None:
        return None
    if not candidates:
        return None
    best = candidates[0]
    args = _build_arguments(best.input_schema, query)
    return ToolCall(name=best.name, arguments=args)


def _build_arguments(schema: Dict[str, Any], query: str) -> Dict[str, Any]:
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        props = {}
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    args: Dict[str, Any] = {}
    for key in required:
        if not isinstance(key, str):
            continue
        args[key] = _value_for_argument_name(key, query)

    # If no required args, still provide useful defaults for common params.
    if not args:
        for candidate_key in ("path", "query", "q", "pattern"):
            if candidate_key in props:
                args[candidate_key] = _value_for_argument_name(candidate_key, query)
                break
    return args


def _value_for_argument_name(name: str, query: str) -> Any:
    key = name.lower()
    path_match = re.search(r"""["']([^"']+[/\\][^"']*)["']""", query)
    single_token_path = re.search(r"(\./[^\s]+|/[^\s]+)", query)
    path_guess = (
        path_match.group(1)
        if path_match
        else (single_token_path.group(1) if single_token_path else ".")
    )
    if key in {"path", "dir", "directory", "folder", "file", "filepath", "file_path"}:
        return path_guess
    if key in {"query", "q", "text", "prompt", "search"}:
        return query
    if key in {"pattern", "glob"}:
        return "*"
    if key in {"limit", "top_k", "count"}:
        return 10
    return query


def _compose_final_answer(
    user_query: str,
    last_tool_result: Optional[McpCallResult],
    system_context: str,
) -> str:
    if last_tool_result is None:
        return (
            "I could not find a reliable tool to execute this request.\n\n"
            f"Request: {user_query}\n"
            f"Context used:\n{system_context}"
        )
    if last_tool_result.is_error:
        return (
            "Tool execution failed.\n\n"
            f"Request: {user_query}\n"
            f"Tool output:\n{last_tool_result.text}"
        )

    # If output is JSON-like list/dict, present pretty format.
    text = last_tool_result.text.strip()
    if not text:
        return f"I executed the tool for '{user_query}', but it returned an empty result."
    try:
        parsed = json.loads(text)
        pretty = json.dumps(parsed, indent=2)
        return f"Here is the tool result for your request:\n{pretty}"
    except Exception:
        return f"Here is the tool result for your request:\n{text}"
