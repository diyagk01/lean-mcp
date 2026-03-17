from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .embeddings import EmbeddingProvider
from .store import SQLiteStore
from .types import ContextPayload, RoutingDecision
from .util import rough_token_estimate


def _tool_spec_from_payload(payload_json: str) -> dict:
    raw = json.loads(payload_json)
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "input_schema": raw.get("input_schema", {}),
    }


@dataclass
class ContextBuilder:
    store: SQLiteStore
    embedder: EmbeddingProvider

    def build(
        self,
        *,
        user_query: str,
        decision: RoutingDecision,
        tool_top_n: int = 1,
        resource_top_k: int = 0,
        resource_server_filter: Optional[str] = None,
    ) -> ContextPayload:
        self.store.init()

        tools: list[dict] = []
        if decision.abstained:
            tools = []
        else:
            top_n = max(1, int(tool_top_n))
            for cand in decision.candidates[:top_n]:
                payload_json = self.store.get_tool_payload_json(cand.tool_id)
                if payload_json:
                    tools.append(_tool_spec_from_payload(payload_json))

        system_lines = [f"User query: {user_query}"]

        if resource_top_k and resource_top_k > 0:
            q_emb = self.embedder.embed_text(user_query)
            chunks = self.store.query_resource_chunks(
                q_emb,
                top_k=int(resource_top_k),
                server_filter=resource_server_filter,
            )
            if chunks:
                system_lines.append("")
                system_lines.append("Relevant resource context:")
                for ch in chunks:
                    system_lines.append(
                        f"- [{ch['server']}] {ch['uri']} (chunk {ch['chunk_index']}, score={ch['score']:.3f}):"
                    )
                    system_lines.append(ch["text"])

        system_context = "\n".join(system_lines).strip() + "\n"
        token_estimate = rough_token_estimate(system_context) + sum(
            rough_token_estimate(json.dumps(t, ensure_ascii=False)) for t in tools
        )

        return {
            "tools": tools,
            "system_context": system_context,
            "token_estimate": int(token_estimate),
        }

