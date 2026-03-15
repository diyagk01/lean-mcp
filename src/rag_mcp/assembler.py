from __future__ import annotations

from typing import List, Optional

from .types import AssembledContext, RetrievedTool


class ContextAssembler:
    def __init__(self, inject_top_n: int = 1) -> None:
        self.inject_top_n = max(1, inject_top_n)

    def assemble(
        self,
        query: str,
        retrieved_tools: List[RetrievedTool],
        resource_chunks: Optional[List[str]] = None,
    ) -> AssembledContext:
        selected = retrieved_tools[: self.inject_top_n]
        tools_payload = [
            {
                "name": r.tool.name,
                "description": r.tool.description,
                "input_schema": r.tool.input_schema,
            }
            for r in selected
        ]
        context_lines = [f"User query: {query}"]
        if resource_chunks:
            context_lines.append("Relevant resource context:")
            context_lines.extend(resource_chunks)
        system_context = "\n".join(context_lines)
        token_estimate = _rough_token_count(
            system_context + " " + " ".join(t["description"] for t in tools_payload)
        )
        return AssembledContext(
            tools=tools_payload,
            system_context=system_context,
            retrieved_tools=retrieved_tools,
            token_estimate=token_estimate,
        )


def _rough_token_count(text: str) -> int:
    # Cheap approximation that works well enough for budget checks.
    return max(1, len(text) // 4)
