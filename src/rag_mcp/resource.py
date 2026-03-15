from __future__ import annotations

from typing import List, Optional

from .embedding import EmbeddingProvider
from .types import McpResource
from .vector_store import SQLiteVectorStore


class ResourceRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: SQLiteVectorStore,
        namespace: str = "resources",
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.namespace = namespace

    def add_resources(self, resources: List[McpResource], chunk_size: int = 500) -> None:
        chunks: List[str] = []
        keys: List[str] = []
        payloads: List[dict] = []
        for resource in resources:
            parts = _chunk_text(resource.content, chunk_size=chunk_size)
            for idx, part in enumerate(parts):
                chunk_id = f"{resource.id}:{idx}"
                keys.append(chunk_id)
                chunks.append(part)
                payloads.append(
                    {
                        "resource_id": resource.id,
                        "server": resource.server,
                        "uri": resource.uri,
                        "chunk_index": idx,
                        "text": part,
                    }
                )
        vectors = self.embedding_provider.embed_texts(chunks) if chunks else []
        for key, vector, payload in zip(keys, vectors, payloads):
            self.vector_store.upsert(self.namespace, key, vector, payload)

    def retrieve(self, query: str, top_k: int = 4, server: Optional[str] = None) -> List[str]:
        q_vec = self.embedding_provider.embed_text(query)
        matches = self.vector_store.search(self.namespace, q_vec, top_k=top_k)
        if server is None:
            return [payload.get("text", "") for _, _, payload in matches]
        return [
            payload.get("text", "")
            for _, _, payload in matches
            if payload.get("server") == server
        ]


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks
