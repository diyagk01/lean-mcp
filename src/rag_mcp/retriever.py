from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from .embedding import EmbeddingProvider
from .types import McpTool, RetrievedTool
from .vector_store import SQLiteVectorStore


class ToolRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: SQLiteVectorStore,
        namespace: str = "tools",
        top_k: int = 5,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.namespace = namespace
        self.top_k = top_k

    def _tool_id(self, tool: McpTool) -> str:
        version = tool.version or "0"
        server = tool.server or "default"
        return f"{server}:{tool.name}:{version}"

    def _tool_hash(self, tool: McpTool) -> str:
        canonical = {
            "name": tool.name,
            "description": tool.description,
            "server": tool.server,
            "version": tool.version,
            "metadata": tool.metadata,
            "input_schema": tool.input_schema,
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _tool_text(self, tool: McpTool) -> str:
        tags = tool.metadata.get("tags", [])
        tags_text = ",".join(tags) if isinstance(tags, list) else ""
        return f"{tool.server or 'default'} | {tool.name} | {tool.description} | tags:{tags_text}"

    def add_tools(self, tools: List[McpTool]) -> None:
        texts = [self._tool_text(t) for t in tools]
        vectors = self.embedding_provider.embed_texts(texts)

        for tool, vector, text in zip(tools, vectors, texts):
            tool_id = self._tool_id(tool)
            tool_hash = self._tool_hash(tool)
            existing = self.vector_store.get_payload(self.namespace, tool_id)

            if existing and existing.get("tool_hash") == tool_hash:
                continue

            payload: Dict[str, object] = {
                "tool": _serialize_tool(tool),
                "tool_text": text,
                "tool_hash": tool_hash,
                "embedding_model": self.embedding_provider.model_name,
            }
            self.vector_store.upsert(self.namespace, tool_id, vector, payload)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedTool]:
        q_vec = self.embedding_provider.embed_text(query)
        matches = self.vector_store.search(
            namespace=self.namespace,
            query_vector=q_vec,
            top_k=top_k or self.top_k,
        )
        return [
            RetrievedTool(tool=_deserialize_tool(payload["tool"]), score=score)
            for _, score, payload in matches
            if "tool" in payload
        ]


def _serialize_tool(tool: McpTool) -> Dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "server": tool.server,
        "metadata": tool.metadata,
        "version": tool.version,
    }


def _deserialize_tool(raw: Dict[str, object]) -> McpTool:
    return McpTool(
        name=str(raw["name"]),
        description=str(raw["description"]),
        input_schema=dict(raw["input_schema"]),  # type: ignore[arg-type]
        server=raw.get("server"),  # type: ignore[arg-type]
        metadata=dict(raw.get("metadata", {})),  # type: ignore[arg-type]
        version=raw.get("version"),  # type: ignore[arg-type]
    )
